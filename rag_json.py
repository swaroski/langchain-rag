import os
import json
from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.metrics import classification_report

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# ───────────────────────────────────────────────────────────────
# 📁 Load environment variables
# ───────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ───────────────────────────────────────────────────────────────
# 🚀 Initialize FastAPI app
# ───────────────────────────────────────────────────────────────
app = FastAPI()

# ───────────────────────────────────────────────────────────────
# 📦 Define Request Schema
# ───────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    text: str
    history: List[Dict[str, str]] = []

# ───────────────────────────────────────────────────────────────
# 📄 Load JSON dataset
# ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.getenv("JSON_PATH", os.path.join(BASE_DIR, "data", "dataset.json"))

if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON file not found at {json_path}")

with open(json_path) as f:
    eval_data = json.load(f)

# ───────────────────────────────────────────────────────────────
# 📚 Setup embeddings and vector store
# ───────────────────────────────────────────────────────────────
embedding_model = OpenAIEmbeddings()
docs = [Document(page_content=example["text"]) for example in eval_data]
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# ───────────────────────────────────────────────────────────────
# 💬 LLM and prompt setup
# ───────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

prompt = ChatPromptTemplate.from_template("""
You are a classification assistant. Given a user input, classify it into one of the following categories: {categories}.
Use the context provided to assist your classification decision.
Return only a JSON in this format: {{"category": "..."}}.

Context:
{context}

User Input:
{input}
""")

output_parser = JsonOutputParser(key="category")
memory = ConversationBufferMemory(return_messages=True, memory_key="history")

# ───────────────────────────────────────────────────────────────
# 🔗 LangChain Runnable Chain
# ───────────────────────────────────────────────────────────────
chain = (
    RunnableMap({
        "input": RunnablePassthrough(),
        "context": lambda x: retriever.get_relevant_documents(x["text"]),
        "categories": lambda _: ["Billing", "Technical", "General", "Sales"]
    })
    | prompt
    | llm
    | output_parser
)

# ───────────────────────────────────────────────────────────────
# 🌐 API Endpoints
# ───────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "LangChain RAG API is running."}


@app.post("/classify")
async def classify(req: QueryRequest):
    response = chain.invoke({"text": req.text})
    return {"category": response}


@app.get("/evaluate")
def evaluate():
    y_true = []
    y_pred = []

    for item in eval_data:
        gold = item["label"]
        pred = chain.invoke({"text": item["text"]})
        y_true.append(gold)
        y_pred.append(pred)

    report = classification_report(y_true, y_pred, output_dict=True)
    return report





"""
Run is thus:
uvicorn rag_json:app --reload


curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
        "text": "I need help updating my credit card information.",
        "history": []
      }'


"""