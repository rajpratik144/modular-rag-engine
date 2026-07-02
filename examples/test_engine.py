import os
import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_engine.core.engine import RAGCoreEngine

load_dotenv()

config = {
    "LLAMA_CLOUD_API_KEY": os.getenv("LLAMA_CLOUD_API_KEY"),
    "DATABASE_URL": os.getenv("DATABASE_URL"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "VISION_MODEL": os.getenv("VISION_MODEL"),
}

llm_object = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

engine = RAGCoreEngine(
    config=config,
    planner_llm=llm_object,
    brain_llm=llm_object,
)

user_id = "user_pratik_002"

file_path = "/Users/pratikraj/Desktop/modular_rag_system/data/0cb769a0-2027-4d86-91ac-750f7e40f51d.pdf (2).pdf"


async def main():

    print(f"\n📂 Starting ingestion for: {file_path}")

    doc_ids = await engine.ingest(
        [file_path],
        user_id,
    )

    if doc_ids:
        print("\n🚀 SUCCESS! Your document is now in the Cloud Vault.")
        print(f"User ID: {user_id}")
        print(f"Doc IDs Created: {doc_ids}")
    else:
        print("\n❌ Ingestion failed or file is a duplicate.")


if __name__ == "__main__":
    asyncio.run(main())