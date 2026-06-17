import re
from .logger import get_logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryEngine:
    """
    Core Logic for retrieval and response generation.
    Uses Dependency Injection for LLM providers to remain model-agnostic.
    """
    def __init__(self, planner_llm, brain_llm, embedder, index, storage,system_persona=None):
        """
        Args:
            planner_llm: Injected LLM object for search planning (e.g. Llama-3-8B).
            brain_llm: Injected LLM object for final synthesis (e.g. Gemini-1.5-Pro).
            embedder: The EmbeddingEngine instance.
            index: The Pinecone index instance.
            storage: The StorageManager instance for SQL metadata.
        """
        self.log = get_logger("QueryEngine")
        self.planner_llm = planner_llm
        self.brain_llm = brain_llm
        self.embedder = embedder
        self.index = index
        self.storage = storage
        
        self.system_persona = system_persona or (
            "You are a professional Document Intelligence AI. "
            "You have access to the user's uploaded files and previous conversation history."
        )

    def _generate_multi_queries(self, question, chat_history):
        """
        Universal Strategy: Generates 3 search variations to ensure high recall 
        across complex document types (Resumes, Medical, Finance).
        """
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

    def ask(self, question, user_id, chat_history=None):
        """
        The main entry point for querying the RAG system.
        
        Args:
            question (str): User input.
            user_id (str): ID for multi-tenant isolation.
            chat_history (list): Optional previous messages.
        """
        # Path A: Simple Greeting (Optimization to save search costs)
        if len(question.split()) < 4 and any(w in question.lower() for w in ["hi", "hello", "hey", "how are you"]):
            return self.planner_llm.invoke(f"Greet the user naturally: {question}").content

        # Path B: Unified Knowledge Retrieval
        # 1. Get SQL Context (Metadata Overview)
        files = self.storage.get_user_files(user_id)
        unique_files = list({f['file_name']: f for f in files}.values())
        sql_info = f"Files available in database: {', '.join([f['file_name'] for f in unique_files])}"

        # 2. Multi-Query Vector Search
        search_queries = self._generate_multi_queries(question, chat_history)
        all_context_chunks = []
        
        for q in search_queries:
            vec = self.embedder.get_embeddings([q])[0]
            res = self.index.query(vector=vec, top_k=3, include_metadata=True, filter={"user_id": user_id})
            all_context_chunks.extend([m['metadata']['text'] for m in res['matches']])

        # Remove duplicate chunks and join
        context_text = "\n\n---\n\n".join(list(set(all_context_chunks)))

        # 3. Final Synthesis using the 'Brain' LLM
        prompt = ChatPromptTemplate.from_template("""
            {persona}
            
            KNOWLEDGE BASE OVERVIEW (SQL):
            {sql_info}
            
            DOCUMENT CONTENT (VECTOR):
            {context}
            
            CONVERSATION HISTORY:
            {history}

            USER QUESTION: {question}
            
            INSTRUCTIONS:
            - Answer the question precisely using the Overview and Content.
            - If details are in a table or list, preserve that formatting.
            - If you cannot find the answer, state that clearly.
        """)
        
        return self.brain_llm.invoke(prompt.format(
            persona=self.system_persona,
            sql_info=sql_info,
            context=context_text,
            history=chat_history or "No previous history.",
            question=question
        )).content