# ==========================================
# File: rag_engine/storage/manager.py
# ==========================================


from pinecone import Pinecone
from supabase import create_client
import hashlib
from ..core.logger import get_logger
from ..core.embeddings import EmbeddingEngine

class StorageManager:
    def __init__(self, config):
        self.log = get_logger("StorageManager")
        
        # Initialize Supabase (SQL Brain)
        self.supabase = create_client(config["SUPABASE_URL"], config["SUPABASE_KEY"])
        
        # Initialize Pinecone (Vector Brain)
        self.pc = Pinecone(api_key=config["PINECONE_API_KEY"])
        
        index_name = config.get("PINECONE_INDEX_NAME", "modular-rag")
        self.index = self.pc.Index(index_name)

        # Initialize the embedding engine
        self.embedder = EmbeddingEngine(config)
        self.log.info("StorageManager initialized with Duplicate Detection.")

    def calculate_hash(self, file_path):
        """Creates a unique SHA-256 fingerprint for a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in small blocks to save RAM on your MacBook Air
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def check_duplicate(self, user_id, file_hash):
        """Checks Supabase for an existing file with the same fingerprint."""
        res = self.supabase.table("document_metadata") \
            .select("id") \
            .eq("user_id", user_id) \
            .eq("file_hash", file_hash) \
            .execute()
        return res.data[0]['id'] if res.data else None

    def save_document(self, user_id, parsed_results):
        file_name = parsed_results[0]['metadata']['file_name']
        file_hash = parsed_results[0]['metadata'].get('file_hash')

        # STEP 1: Generate Embeddings FIRST
        texts = [part['content'] for part in parsed_results]
        all_vectors = self.embedder.get_embeddings(texts)

        if not all_vectors or len(all_vectors) != len(parsed_results):
            self.log.error("Embedding failed. No record created in Supabase.")
            return None

        # STEP 2: If embeddings are successful, NOW create Supabase record
        doc_record = {
            "user_id": user_id,
            "file_name": file_name,
            "total_parts": len(parsed_results),
            "file_type": parsed_results[0]['metadata']['file_type'],
            "file_hash": file_hash
        }
        db_res = self.supabase.table("document_metadata").insert(doc_record).execute()
        doc_id = db_res.data[0]['id'] 
        
        # STEP 3: Upload to Pinecone
        vectors_to_upsert = []
        for i, part in enumerate(parsed_results):
            chunk_id = f"{doc_id}#chunk_{i}"
            page_val = int(part['metadata'].get('page_number', i + 1))
            metadata = {
                "user_id": user_id,
                "doc_id": doc_id,
                "text": part['content'],
                "file_name": file_name,
                "page_number": page_val
            }
            vectors_to_upsert.append((chunk_id, all_vectors[i], metadata))

        self.index.upsert(vectors=vectors_to_upsert)
        self.log.info(f"Successfully saved {file_name} with ID: {doc_id}")
        return doc_id
    
    def get_user_files(self, user_id):
        """Returns a list of all documents belonging to a specific user."""
        res = self.supabase.table("document_metadata") \
            .select("id, file_name, total_parts, created_at") \
            .eq("user_id", user_id) \
            .execute()
        return res.data

    def delete_document(self, doc_id, user_id):
        """Wipes a document from both Pinecone and Supabase."""
        self.log.info(f"Deleting document {doc_id} for user {user_id}")

        # 1. Delete from Pinecone (Free tier Serverless supports this)
        self.index.delete(filter={"doc_id": {"$eq": doc_id}, "user_id": {"$eq": user_id}})
        
        # 2. Delete from Supabase
        self.supabase.table("document_metadata").delete().eq("id", doc_id).execute()
        
        return True