# ==========================================
# File: rag_engine/core/embeddings.py
# ==========================================


import os
import time
from google import genai
from ..core.logger import get_logger

class EmbeddingEngine:
    def __init__(self, config):
        self.log = get_logger("EmbeddingEngine")
        
        api_key = config.get("GOOGLE_API_KEY")
        self.model_name = config.get("EMBEDDING_MODEL", "text-embedding-004")
        self.batch_size = config.get("EMBEDDING_BATCH_SIZE", 16)
        
        try:
            self.client = genai.Client(api_key=api_key)
            # Locked in your preferred model
            self.model_name = "gemini-embedding-2-preview"
            self.log.info(f"Google GenAI initialized with: {self.model_name}")
        except Exception as e:
            self.log.error(f"Initialization failed: {e}")
            raise

    def get_embeddings(self, text_list):
        """
        Generates 768-dimension vectors for each chunk using Gemini 2 Preview.
        """
        self.log.info(f"Generating embeddings for {len(text_list)} chunks individually...")
        
        all_embeddings = []
        
        for i, text in enumerate(text_list):
            retries = 3
            while retries > 0:
                try:
                    # We explicitly request 768 dimensions to match Pinecone
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=text,
                        config={
                            'output_dimensionality': 768
                        }
                    )
                    
                    vector = response.embeddings[0].values
                    all_embeddings.append(vector)
                    
                    # Progress tracker
                    if (i + 1) % 10 == 0:
                        self.log.info(f"   Progress: {i + 1}/{len(text_list)} chunks processed...")
                    
                    # Safety delay for free-tier rate limits
                    time.sleep(0.6) 
                    break 
                    
                except Exception as e:
                    error_msg = str(e).upper()
                    # If Google is busy (429) or down (503/500), we wait and retry
                    if "429" in error_msg or "503" in error_msg or "500" in error_msg:
                        self.log.warning(f"Cloud API busy ({error_msg}). Retrying in 10s...")
                        time.sleep(10)
                        retries -= 1
                    else:
                        self.log.error(f"Permanent failure at chunk {i}: {str(e)}")
                        return None
                        
        self.log.info(f"Successfully generated {len(all_embeddings)} vectors (768-dim).")
        return all_embeddings