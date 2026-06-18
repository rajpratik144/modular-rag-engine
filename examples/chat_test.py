import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from rag_engine.core.engine import RAGCoreEngine

# 1. Load your credentials
load_dotenv()

config = {
    "LLAMA_CLOUD_API_KEY": os.getenv("LLAMA_CLOUD_API_KEY"),
    "SUPABASE_URL": os.getenv("SUPABASE_URL"),
    "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
}

# 2. Setup the Brain (Object Injection)
llm = ChatGroq(
    model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# 3. Define a Persona (Optional)
# hr_persona = "You are a helpful HR Assistant. Use the provided context to answer questions."

# 4. Initialize the Engine
# engine = RAGCoreEngine(config, planner_llm=llm, brain_llm=llm, system_persona=hr_persona)
engine = RAGCoreEngine(config, planner_llm=llm, brain_llm=llm)

# 5. Interactive Chat Loop with STREAMING
user_id = "pratik"
chat_memory = []

print("\n" + "="*50)
print("🚀 MODULAR RAG SYSTEM: ONLINE (STREAMING MODE)")
print("="*50)

while True:
    query = input("\n👤 User: ")
    if query.lower() in ['exit', 'quit']:
        break
    
    print("\n🤖 AI: ", end="", flush=True)
    
    # We collect the full answer in a list so we can add it to memory later
    full_answer_parts = []
    
    # --- THIS IS THE STREAMING MAGIC ---
    # We loop through the chunks as they are 'born' in the cloud
    for chunk in engine.stream_ask(query, user_id, chat_history=chat_memory[-4:]):
        print(chunk, end="", flush=True) # Print immediately to screen
        full_answer_parts.append(chunk)
    
    print() # New line after the stream finishes
    
    # Join parts to save the full response in history
    final_answer = "".join(full_answer_parts)
    chat_memory.append(f"User: {query}")
    chat_memory.append(f"AI: {final_answer}")
    
    print("-" * 50)