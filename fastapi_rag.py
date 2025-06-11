import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

app = FastAPI()

# ------- Data + Vectorstore Setup -------
data = [
    {"product": "Laptop", "category": "Electronics", "price": "999.99", "stock": "50", "region": "NA"},
    {"product": "Tablet", "category": "Electronics", "price": "499.99", "stock": "80", "region": "EU"},
    {"product": "Chair",  "category": "Furniture",   "price": "199.99", "stock": "30", "region": "AS"},
]

docs = [Document(page_content=", ".join(f"{k}: {v}" for k, v in row.items()), metadata={"row": i})
        for i, row in enumerate(data)]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vs = Chroma.from_documents(splits, embedding=embeddings)

# ------- Prompt & Chain -------
SYSTEM_PROMPT = """You are a data analyst. Use chat history and context to think step-by-step.
If you don’t know, say so.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{question}")
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

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

# ------- In-Memory Session Store -------
memory_store: dict[str, list[tuple[str, str]]] = {}

# ------- API Models & Endpoints -------
class Query(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask(query: Query):
    # Build chat history string
    history = memory_store.setdefault(query.session_id, [])
    chat_hist_str = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history) or "No prior history."

    # Retrieve and format context
    docs = vs.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query.question)
    context_str = "\n".join(d.page_content for d in docs)

    # Invoke RAG chain with flat inputs
    answer = chain.invoke({
        "question": query.question,
        "context": context_str,
        "chat_history": chat_hist_str
    })

    # Update memory
    history.append((query.question, answer))

    return {
        "answer": answer,
        "sources": [d.page_content for d in docs],
        "session_id": query.session_id
    }

@app.get("/")
def health():
    return {"status": "ok"}



"""
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the price of the Camera?", "session_id": "session_124"}'
"""