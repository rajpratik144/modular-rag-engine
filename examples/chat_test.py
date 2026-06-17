import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from rag_engine.core.engine import RAGCoreEngine

load_dotenv()

# 1. THE PARENT chooses the "Fuel" (Providers)
# This solves the 'model_name' vs 'model' parameter error
gemini_brain = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# 2. THE MANIFEST (Credentials & Constants)
config = {
    "LLAMA_CLOUD_API_KEY": os.getenv("LLAMA_CLOUD_API_KEY"),
    "SUPABASE_URL": os.getenv("SUPABASE_URL"),
    "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY")
}

persona = """
You are the Global Vision Tech Virtual HR Assistant. 
Your tone is professional, helpful, and formal. 
When answering questions:
1. Use ONLY the provided HR Policy context.
2. If a user asks about something not in the policy, politely inform them that you 
   cannot find that information and suggest they contact the HR Hotline.
3. If you find information in a table (like bonuses or grades), format your 
   output clearly as a table or list.
4. Always prioritize compliance and security rules mentioned in the document.
"""

# 3. ASSEMBLY
# We inject the gemini_brain for both planning and answering
engine = RAGCoreEngine(config, 
                       planner_llm=gemini_brain, 
                       brain_llm=gemini_brain, 
                       system_persona=persona
                       )

# 4. CHAT LOOP
user_id = "user_pratik_002"
chat_memory = []

print("\n🚀 MODULAR RAG SYSTEM")
while True:
    query = input("\n👤 User: ")
    if query.lower() in ['exit', 'quit']: break
    
    answer = engine.ask(query, user_id, chat_history=chat_memory[-4:])
    print(f"\n🤖 AI: {answer}")
    
    chat_memory.append(f"User: {query}")
    chat_memory.append(f"AI: {answer}")