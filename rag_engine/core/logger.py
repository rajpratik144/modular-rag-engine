import logging
import sys

def get_logger(name):
    logger = logging.getLogger(name)
    
    # If the logger is already configured, don't do it again
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 1. THE FILE HANDLER (Saves everything to the file)
        file_handler = logging.FileHandler("rag_system.log")
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 2. NO TERMINAL HANDLER
        # We intentionally do NOT add a StreamHandler(sys.stdout) here.
        # This keeps the terminal clean for your print() statements.

    # 3. SILENCE THIRD-PARTY LIBRARIES (The ones causing the AFC/Retry noise)
    # We set them to 'WARNING' so they only show up if something actually breaks.
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    
    return logger