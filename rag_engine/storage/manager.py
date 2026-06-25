# ==========================================
# File: rag_engine/storage/manager.py
# ==========================================

import hashlib
import uuid
from sqlalchemy import create_engine, Column, String, Integer, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pinecone import Pinecone

from ..core.logger import get_logger
from ..core.embeddings import EmbeddingEngine

# Define the Universal Schema
Base = declarative_base()

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"
    # Using String for ID ensures compatibility between SQLite and Postgres UUIDs
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    file_name = Column(String, nullable=False)
    total_parts = Column(Integer)
    file_type = Column(String)
    file_hash = Column(String, index=True)
    created_at = Column(DateTime, server_default=func.now())

class StorageManager:
    def __init__(self, config):
        self.log = get_logger("StorageManager")
        
        # 1. Initialize Relational Database (SQLAlchemy)
        db_url = config.get("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL must be provided in config (e.g., sqlite:///./test.db or postgresql://...)")
        
        self.engine = create_engine(db_url)
        # Automatically create tables if they don't exist
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # 2. Initialize Pinecone (Vector Brain)
        self.pc = Pinecone(api_key=config["PINECONE_API_KEY"])
        index_name = config.get("PINECONE_INDEX_NAME", "modular-rag")
        self.index = self.pc.Index(index_name)

        # 3. Initialize the embedding engine
        self.embedder = EmbeddingEngine(config)
        self.log.info(f"StorageManager initialized. Database Dialect: {self.engine.name}")

    def calculate_hash(self, file_path):
        """Creates a unique SHA-256 fingerprint for a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def check_duplicate(self, user_id, file_hash):
        """Universal SQL check for duplicate fingerprints."""
        session = self.Session()
        try:
            doc = session.query(DocumentMetadata).filter_by(
                user_id=user_id, 
                file_hash=file_hash
            ).first()
            return doc.id if doc else None
        finally:
            session.close()

    def save_document(self, user_id, parsed_results):
        file_name = parsed_results[0]['metadata']['file_name']
        file_hash = parsed_results[0]['metadata'].get('file_hash')

        # STEP 1: Generate Embeddings
        texts = [part['content'] for part in parsed_results]
        all_vectors = self.embedder.get_embeddings(texts)

        if not all_vectors or len(all_vectors) != len(parsed_results):
            self.log.error("Embedding failed. No record created.")
            return None

        # STEP 2: Save to SQL Database (Independent of Provider)
        session = self.Session()
        try:
            new_doc = DocumentMetadata(
                user_id=user_id,
                file_name=file_name,
                total_parts=len(parsed_results),
                file_type=parsed_results[0]['metadata']['file_type'],
                file_hash=file_hash
            )
            session.add(new_doc)
            session.flush() # Gets the ID without committing yet
            doc_id = new_doc.id

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
            session.commit() # Finalize SQL entry after Pinecone success
            self.log.info(f"Successfully saved {file_name} with ID: {doc_id}")
            return doc_id
        except Exception as e:
            session.rollback()
            self.log.error(f"Failed to save document: {e}")
            raise e
        finally:
            session.close()
    
    def get_user_files(self, user_id):
        """Universal SQL retrieval for user documents."""
        session = self.Session()
        try:
            docs = session.query(DocumentMetadata).filter_by(user_id=user_id).all()
            # Convert to list of dicts to keep the API response consistent
            return [
                {
                    "id": d.id, 
                    "file_name": d.file_name, 
                    "total_parts": d.total_parts, 
                    "created_at": d.created_at.isoformat()
                } for d in docs
            ]
        finally:
            session.close()

    def delete_document(self, doc_id, user_id):
        """Wipes a document from both Pinecone and SQL database."""
        self.log.info(f"Deleting document {doc_id} for user {user_id}")
        session = self.Session()
        try:
            # 1. Delete from Pinecone
            self.index.delete(filter={"doc_id": {"$eq": doc_id}, "user_id": {"$eq": user_id}})
            
            # 2. Delete from SQL
            doc = session.query(DocumentMetadata).filter_by(id=doc_id, user_id=user_id).first()
            if doc:
                session.delete(doc)
                session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.log.error(f"Delete failed: {e}")
            return False
        finally:
            session.close()