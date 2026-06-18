import re
from typing import List, Optional, Generator
from .logger import get_logger
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryEngine:
    """
    Core Logic for retrieval and response generation.
    Supports both standard blocking responses and real-time streaming.
    """
    def __init__(self, planner_llm, brain_llm, embedder, index, storage, system_persona=None):
        self.log = get_logger("QueryEngine")
        self.planner_llm = planner_llm
        self.brain_llm = brain_llm
        self.embedder = embedder
        self.index = index
        self.storage = storage
        self.ranker = Ranker()
        
        self.system_persona = system_persona or (
            "You are a professional Document Intelligence AI. "
            "You have access to the user's uploaded files and conversation history."
        )

    def _extract_page_number(self, query: str) -> Optional[int]:
        match = re.search(r'page\s+(\d+)', query.lower())
        return int(match.group(1)) if match else None

    def _generate_multi_queries(self, question: str, chat_history: List[str]) -> List[str]:
        prompt = ChatPromptTemplate.from_template("""
            You are a Search Planner. Break the user's question into 3 concise search queries 
            that will find all relevant parts of a document (text, tables, or history).
            
            History: {history}
            Question: {question}
            
            Provide only the 3 queries, one per line. No numbers.
        """)
        chain = prompt | self.planner_llm | StrOutputParser()
        response = chain.invoke({"history": chat_history, "question": question})
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        return queries[:3]

    def _get_unified_context(self, question: str, user_id: str, chat_history: List[str]):
        """Combines SQL Metadata, Vector Search, and BM25 Reranking."""
        # 1. SQL Metadata Context
        files = self.storage.get_user_files(user_id)
        unique_files = list({f['file_name']: f for f in files}.values())
        sql_context = f"Files in DB: " + ", ".join([f"{f['file_name']} ({f['total_parts']} parts)" for f in unique_files])

        # 2. Multi-Query Vector Search
        page_filter = self._extract_page_number(question)
        search_queries = self._generate_multi_queries(question, chat_history)
        
        all_chunks = []
        search_filter = {"user_id": {"$eq": user_id}}
        if page_filter:
            search_filter["page_number"] = {"$eq": page_filter}

        for q in search_queries:
            vec = self.embedder.get_embeddings([q])[0]
            res = self.index.query(vector=vec, top_k=5, include_metadata=True, filter=search_filter)
            all_chunks.extend([m['metadata']['text'] for m in res['matches']])

        # 3. BM25 Keyword Scoring
        passages = list(set(all_chunks))
        if not passages:
            return sql_context, ""

        tokenized_corpus = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(question.lower().split())
        
        combined = sorted(zip(passages, bm25_scores), key=lambda x: x[1], reverse=True)
        top_passages = [res[0] for res in combined[:6]]
        
        return sql_context, "\n\n---\n\n".join(top_passages)

    def stream_ask(self, question: str, user_id: str, chat_history: List[str] = None) -> Generator[str, None, None]:
        """
        The REAL Engine logic. Streams chunks of text as they are generated.
        """
        # A. Quick Chat Optimization
        if len(question.split()) < 4 and any(w in question.lower() for w in ["hi", "hello", "how are you"]):
            for chunk in self.planner_llm.stream(f"Greet the user naturally: {question}"):
                yield chunk.content if hasattr(chunk, 'content') else str(chunk)
            return

        # B. Unified Retrieval
        sql_info, doc_context = self._get_unified_context(question, user_id, chat_history or [])

        # C. Final Prompt Construction
        prompt = ChatPromptTemplate.from_template("""
            {persona}
            
            KNOWLEDGE BASE OVERVIEW: {sql_info}
            DOCUMENT CONTENT: {context}
            HISTORY: {history}
            QUESTION: {question}
            
            ANSWER:
        """)
        
        formatted_prompt = prompt.format(
            persona=self.system_persona,
            sql_info=sql_info,
            context=doc_context,
            history=chat_history or "No previous history.",
            question=question
        )

        # D. Stream from the Brain LLM
        for chunk in self.brain_llm.stream(formatted_prompt):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            yield content

    def ask(self, question: str, user_id: str, chat_history: List[str] = None) -> str:
        """
        The Wrapper logic. Collects the stream and returns a single full string.
        """
        full_response = []
        # We simply 'watch' our own stream and join the results
        for chunk in self.stream_ask(question, user_id, chat_history):
            full_response.append(chunk)
        
        return "".join(full_response)