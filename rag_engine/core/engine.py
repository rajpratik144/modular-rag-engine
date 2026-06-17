from ..parsers.cloud_parser import UniversalParser
from ..storage.manager import StorageManager
from .query_engine import QueryEngine
from .logger import get_logger

class RAGCoreEngine:
    """
    Entry point for the Modular RAG Library. 
    Accepts pre-initialized LLM objects via Dependency Injection.
    """
    def __init__(self, config, planner_llm, brain_llm,system_persona=None):
        self.log = get_logger("RAGCoreEngine")
        self.config = config

        # Initialize internal modules
        self.parser = UniversalParser(config)
        self.storage = StorageManager(config)
        
        # Inject the parent-provided LLMs
        self.query_engine = QueryEngine(
            planner_llm=planner_llm,
            brain_llm=brain_llm,
            embedder=self.storage.embedder,
            index=self.storage.index,
            storage=self.storage,
            system_persona=system_persona
        )

        self.log.info("RAG Engine successfully assembled with injected models.")

    def ingest(self, file_paths, user_id):
        """
        The full pipeline: Parse -> Store. Now with SHA-256 Duplicate Detection.
        """
        self.log.info(f"--- Starting Ingestion for User: {user_id} ---")
        final_doc_ids = []

        for path in file_paths:
            # 1. Calculate Fingerprint (SHA-256)
            # FIXED TYPO: added the 'c' in calculate
            file_hash = self.storage.calculate_hash(path)

            # 2. Check for Duplicate in Supabase
            existing_id = self.storage.check_duplicate(user_id, file_hash)
            
            if existing_id:
                self.log.info(f"File {path} is a duplicate. ID: {existing_id}")
                final_doc_ids.append(existing_id)
                continue 

            # 3. If new, proceed with Parse -> Embed -> Store
            parsed_data = self.parser.parse_files([path])
            
            if parsed_data:
                # Attach hash so StorageManager can save it
                parsed_data[0]['metadata']['file_hash'] = file_hash
                doc_id = self.storage.save_document(user_id, parsed_data)
                final_doc_ids.append(doc_id)
            else:
                self.log.error(f"Failed to parse file: {path}")

        # FIXED LOGIC: We check our ID list instead of the 'parsed_data' variable
        if not final_doc_ids:
            self.log.warning("No documents were processed (all duplicates or errors).")
            return []
        
        return final_doc_ids
    
    def delete_files(self, doc_id, user_id):
        return self.storage.delete_document(doc_id, user_id)

    def ask(self, question, user_id, chat_history=None):
        return self.query_engine.ask(question, user_id, chat_history)