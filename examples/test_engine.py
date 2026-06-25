import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_groq import ChatGroq # Parent handles the import
from rag_engine.core.engine import RAGCoreEngine

# 1. Load credentials from .env
load_dotenv()

# 2. THE FUEL: Credentials & Settings
config = {
    "LLAMA_CLOUD_API_KEY": os.getenv("LLAMA_CLOUD_API_KEY"),
    "DATABASE_URL": os.getenv("DATABASE_URL"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "VISION_MODEL": os.getenv("VISION_MODEL")
}


# 3. OBJECT INJECTION: Initialize the LLM here
# This is where you solve the 'model' vs 'model_name' issue once and for all
llm_object = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# 4. START THE ENGINE
# We inject the llm_object as both the planner and the brain
engine = RAGCoreEngine(config, planner_llm=llm_object, brain_llm=llm_object)

# 5. RUN INGESTION
user_id = "user_pratik_002"
file_path = "data/hr_poilcy.md"

print(f"\n📂 Starting ingestion for: {file_path}")
doc_ids = engine.ingest([file_path], user_id)

if doc_ids:
    print(f"\n🚀 SUCCESS! Your document is now in the Cloud Vault.")
    print(f"User ID: {user_id}")
    print(f"Doc IDs Created: {doc_ids}")
else:
    print("\n❌ Ingestion failed or file is a duplicate.")