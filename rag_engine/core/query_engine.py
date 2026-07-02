import re
from typing import List, Optional, Generator
from .logger import get_logger
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryEngine:
    def __init__(self, planner_llm, brain_llm, embedder, index, storage, system_persona=None):
        self.log = get_logger("QueryEngine")
        self.planner_llm = planner_llm
        self.brain_llm = brain_llm
        self.embedder = embedder
        self.index = index
        self.storage = storage
        
        # Initialize FlashRank Reranker (Small & Fast)
        self.ranker = Ranker()
        
        self.system_persona = system_persona or (
            "You are a professional Document Intelligence AI."
        )

    def _extract_page_number(self, query: str) -> Optional[int]:
        match = re.search(r'page\s+(\d+)', query.lower())
        return int(match.group(1)) if match else None

    def _generate_multi_queries(self, question: str, chat_history: List[str]) -> List[str]:
        prompt = ChatPromptTemplate.from_template("""
            You are a Search Planner. Break the user's question into 2 specific search queries 
            that will find the exact data needed from the documents.
            
            History: {history}
            Question: {question}
            
            Provide only the 2 queries, one per line. No numbers or bullets.
        """)
        chain = prompt | self.planner_llm | StrOutputParser()
        # Only take last 2 messages for planning to save tokens
        short_history = chat_history[-2:] if chat_history else []
        response = chain.invoke({"history": short_history, "question": question})
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        return queries[:2]

    def _get_unified_context(self, question: str, user_id: str, chat_history: List[str]):
        """Combines Metadata, Vector Search, and Neural Reranking."""
        # 1. SQL Metadata Context (Brief)
        files = self.storage.get_user_files(user_id)
        unique_files = list({f['file_name']: f for f in files}.values())
        sql_context = ", ".join([f"{f['file_name']}" for f in unique_files])

        # 2. Multi-Query Vector Search
        page_filter = self._extract_page_number(question)
        search_queries = self._generate_multi_queries(question, chat_history)
        # Always include the original question in search
        search_queries.append(question)
        
        all_chunks = []
        search_filter = {"user_id": {"$eq": user_id}}
        if page_filter:
            search_filter["page_number"] = {"$eq": page_filter}

        for q in search_queries:
            vec = self.embedder.get_embeddings([q])[0]
            # Reduce top_k to 4 to avoid token overflow initially
            res = self.index.query(vector=vec, top_k=4, include_metadata=True, filter=search_filter)
            for m in res['matches']:
                all_chunks.append({
                    "id": m['id'],
                    "text": m['metadata']['text'],
                    "metadata": m['metadata']
                })

        # 3. Neural Reranking (The Solution to the 413 Error)
        # We de-duplicate chunks by text content
        unique_passages = []
        seen_text = set()
        for chunk in all_chunks:
            if chunk['text'] not in seen_text:
                unique_passages.append(chunk)
                seen_text.add(chunk['text'])

        if not unique_passages:
            return sql_context, "No relevant document content found."

        # Prepare for FlashRank
        # We take the top 3-4 most relevant passages to stay under token limits
        rerank_request = RerankRequest(query=question, passages=unique_passages)
        results = self.ranker.rerank(rerank_request)
        
        # We only take the top 3 results because the new parser outputs 
        # very large chunks (full pages/tables).
        top_results = results[:3]
        
        doc_context = ""
        for res in top_results:
            # Add a clear separator for the Brain LLM
            doc_context += f"\n[Source: {res['metadata'].get('file_name', 'Unknown')} | Page: {res['metadata'].get('page_number', 'N/A')}]\n"
            doc_context += f"{res['text']}\n"

        return sql_context, doc_context

    def stream_ask(self, question: str, user_id: str, chat_history: List[str] = None) -> Generator[str, None, None]:
        # A. Quick Greeting Check
        if len(question.split()) < 4 and any(w in question.lower() for w in ["hi", "hello", "hey"]):
            for chunk in self.planner_llm.stream(f"Greet the user naturally: {question}"):
                yield chunk.content if hasattr(chunk, 'content') else str(chunk)
            return

        # B. Smart Retrieval (Includes Reranking)
        sql_info, doc_context = self._get_unified_context(question, user_id, chat_history or [])

        # C. Token-Safe Prompt
        prompt = ChatPromptTemplate.from_template("""
            {persona}
            
            FILES AVAILABLE: {sql_info}
            
            RELEVANT CONTENT:
            {context}
            
            CONVERSATION HISTORY:
            {history}
            
            USER QUESTION: {question}
            
            INSTRUCTIONS:
            - Answer the question accurately using the RELEVANT CONTENT provided.
            - If the information is not in the content, use your general knowledge but state that it's not in the documents.
            - Keep tables and formatting clean.
        """)
        
        # Only pass the last 4 messages of history to the Brain to save token budget
        trimmed_history = chat_history[-4:] if chat_history else "No previous history."

        formatted_prompt = prompt.format(
            persona=self.system_persona,
            sql_info=sql_info,
            context=doc_context,
            history=trimmed_history,
            question=question
        )

        # D. Safe Streaming
        try:
            for chunk in self.brain_llm.stream(formatted_prompt):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                yield content
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower() or "413" in str(e):
                yield "\n\n⚠️ **System Note:** This specific part of the document is too large for the current AI processing tier. Please try asking a more specific question about a smaller section."
            else:
                raise e

    def ask(self, question: str, user_id: str, chat_history: List[str] = None) -> str:
        full_response = []
        for chunk in self.stream_ask(question, user_id, chat_history):
            full_response.append(chunk)
        return "".join(full_response)