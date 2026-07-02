# ==========================================
# File: rag_engine/parsers/cloud_parser.py
# ==========================================


import os
from llama_parse import LlamaParse, ResultType
from ..core.logger import get_logger

class UniversalParser:
    def __init__(self, config):
        self.log = get_logger("UniversalParser")
        
        api_key = config.get("LLAMA_CLOUD_API_KEY")
        vision_model = config.get("VISION_MODEL", "openai-gpt4o-mini")
        workers = config.get("PARSER_WORKERS", 1)

        self.parser = LlamaParse(
            api_key=api_key,
            result_type=ResultType.MD,
            tier="agentic",
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name=vision_model,
            num_workers=workers,
            verbose=True,
            language="en"
        )

    def parse_files(self, file_paths):
        self.log.info(f"Starting Visual Parsing for {len(file_paths)} files...")

        try:
            documents = self.parser.load_data(file_paths)
            final_data = []

            for i, doc in enumerate(documents):
                # 1. Try to get page number from API
                # 2. If missing, assume (index + 1) because files are read in order
                page_num = doc.metadata.get("page_number")
                if page_num is None:
                    page_num = i + 1 
                
                source_path = doc.metadata.get("file_path") or file_paths[0]
                file_name = os.path.basename(source_path)

                metadata = {
                    "source_path": source_path,
                    "file_name": file_name,
                    "page_number": int(page_num), # Force it to be a Number
                    "file_type": file_name.split('.')[-1].lower()
                }

                final_data.append({
                    "content": doc.text,
                    "metadata": metadata
                })

            self.log.info(f"Successfully processed {len(final_data)} pages.")
            return final_data
        
        except Exception as e:
            self.log.error(f"Visual Parsing failed: {str(e)}")
            return None