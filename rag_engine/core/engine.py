from typing import List, Dict, Any

from ..parsers.cloud_parser import UniversalParser
from ..storage.manager import StorageManager
from .query_engine import QueryEngine
from .logger import get_logger


class RAGCoreEngine:
    def __init__(
        self,
        config: Dict[str, Any],
        planner_llm,
        brain_llm,
        system_persona: str = None,
    ):
        self.log = get_logger("RAGCoreEngine")
        self.config = config

        self.parser = UniversalParser(config)
        self.storage = StorageManager(config)

        self.query_engine = QueryEngine(
            planner_llm=planner_llm,
            brain_llm=brain_llm,
            embedder=self.storage.embedder,
            index=self.storage.index,
            storage=self.storage,
            system_persona=system_persona,
        )

        self.log.info("ENGINE is fueled and ready.")

    async def ingest(
        self,
        file_paths: List[str],
        user_id: str,
    ) -> List[str]:
        """
        Handles fingerprinting, parsing and storage.
        """

        self.log.info(f"--- Starting Ingestion for User: {user_id} ---")

        final_doc_ids = []

        for path in file_paths:

            file_hash = self.storage.calculate_hash(path)

            existing_id = self.storage.check_duplicate(
                user_id,
                file_hash,
            )

            if existing_id:
                self.log.info(
                    f"File {path} is a duplicate. ID: {existing_id}"
                )
                final_doc_ids.append(existing_id)
                continue

            # await the async parser
            parsed_data = await self.parser.parse_files([path])

            if parsed_data:

                parsed_data[0]["metadata"]["file_hash"] = file_hash

                doc_id = self.storage.save_document(
                    user_id,
                    parsed_data,
                )

                if doc_id:
                    final_doc_ids.append(doc_id)

        return final_doc_ids

    def get_user_files(self, user_id: str):
        return self.storage.get_user_files(user_id)

    def delete_files(self, doc_id: str, user_id: str) -> bool:
        return self.storage.delete_document(doc_id, user_id)

    def ask(self,question: str,user_id: str,chat_history: List[str] = None,) -> str:
        return self.query_engine.ask(question,user_id,chat_history,)

    def stream_ask(self,question: str,user_id: str,chat_history: List[str] = None,):
        return self.query_engine.stream_ask(question,user_id,chat_history,)