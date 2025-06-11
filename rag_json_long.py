import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics import classification_report

# Load env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ───────────────────────────────────────────────────────────────
# 📄 Load JSON dataset
# ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.getenv("JSON_PATH", os.path.join(BASE_DIR, "data", "dataset_long.json"))

if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON file not found at {json_path}")

with open(json_path) as f:
    eval_data = json.load(f)

# Initialize FastAPI app
app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    text: str
    history: List[Dict[str, str]] = []



# Chunking long text for vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_chunks = []

for item in eval_data:
    chunks = text_splitter.create_documents([item["text"]], metadatas=[{"label": item["label"]}])
    all_chunks.extend(chunks)

# Embeddings + vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_chunks, embedding_model)
retriever = vectorstore.as_retriever()

# LLM & Prompt
llm = ChatOpenAI(model="gpt-4", temperature=0.2)
prompt = ChatPromptTemplate.from_template("""
You are a classification assistant. Classify the following user input into one of the categories: {categories}.

Relevant context: {context}

Return ONLY a JSON like: {{"category": "..."}}.

User input: {input}
""")
output_parser = JsonOutputParser(key="category")

# Chain
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

# Routes
@app.get("/")
def read_root():
    return {"message": "RAG JSON Classification API running."}

@app.post("/classify")
async def classify(req: QueryRequest):
    result = chain.invoke({"text": req.text})
    return {"category": result}

@app.get("/evaluate")
def evaluate():
    y_true = []
    y_pred = []

    for item in eval_data:
        gold = item["label"]
        pred = chain.invoke({"text": item["text"]})
        y_true.append(gold)
        y_pred.append(pred["category"])  # <- FIX HERE

    report = classification_report(y_true, y_pred, output_dict=True)
    return report

