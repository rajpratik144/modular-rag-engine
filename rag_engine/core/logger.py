# ==========================================
# File: rag_engine/core/logger.py
# ==========================================


import logging
import sys

def get_logger(name):
    # 1. Capture the logger
    logger = logging.getLogger(name)
    
    # 2. If it doesn't have handlers, set it up
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Save to file
        file_handler = logging.FileHandler("rag_system.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Optional: Add a very small StreamHandler that only shows IMPORTANT errors to terminal
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.ERROR) # ONLY Errors on terminal
        logger.addHandler(console_handler)

    # 3. THE SILENCER: This stops other libraries from printing to your terminal
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    
    return logger