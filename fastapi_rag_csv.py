# --- Standard Libraries ---
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DataFrameLoader

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# --- FastAPI app instance ---
app = FastAPI()

# --- Load CSV data path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.getenv("CSV_PATH", os.path.join(BASE_DIR, "data", "data.csv"))

# --- Validate CSV file existence ---
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

# --- Load CSV into DataFrame ---
df = pd.read_csv(csv_path)

# --- Combine all columns into one for embedding ---
df["combined"] = df.astype(str).agg(" | ".join, axis=1)

# --- Load documents using DataFrameLoader ---
loader = DataFrameLoader(df, page_content_column="combined")
docs = loader.load()

# --- Split large documents into smaller chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# --- Embed and create vector store ---
embeddings = OpenAIEmbeddings()
vs = Chroma.from_documents(splits, embedding=embeddings)
retriever = vs.as_retriever(search_kwargs={"k": 3})

# --- System prompt for LLM context ---
SYSTEM_PROMPT = """You are a data analyst with exceptional analytics skills. Use chat history and context to think step-by-step.
If you don't know anything, say so.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

# --- Prompt template setup ---
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{question}")
])

# --- LLM setup ---
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# --- Build the RAG chain ---
chain = (
    {
        "question": RunnablePassthrough(),
        "context": RunnablePassthrough(),
        "chat_history": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- In-memory session store for chat history ---
memory_store: dict[str, list[tuple[str, str]]] = {}

# --- Pydantic model for request payload ---
class Query(BaseModel):
    question: str
    session_id: str

# --- POST endpoint for querying the assistant ---
@app.post("/ask")
async def ask(query: Query):
    # Retrieve or initialize session history
    history = memory_store.setdefault(query.session_id, [])

    # Format chat history for the prompt
    chat_hist_str = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history) or "No prior history."

    # Fetch relevant documents from the vector store
    docs = retriever.invoke(query.question)
    context_str = "\n".join(d.page_content for d in docs)

    # Generate the answer using the RAG chain
    try:
        answer = chain.invoke({
            "question": query.question,
            "context": context_str,
            "chat_history": chat_hist_str
        })
    except Exception as e:
        return {"error": str(e)}

    # Append interaction to memory
    history.append((query.question, answer))

    return {
        "answer": answer,
        "sources": [d.page_content for d in docs],
        "session_id": query.session_id
    }

# --- Simple health check endpoint ---
@app.get("/")
def health():
    return {"status": "ok"}






"""

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What is the price of the camera?",
        "session_id": "user123"
      }'



uvicorn fastapi_rag_csv:app --reload
"""