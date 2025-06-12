import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain_core.documents import Document
from tqdm import tqdm

# --- Setup Environment ---
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

# --- Setup Environment ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Load sales data ---
df = pd.read_csv("sales_data.csv")

# --- Format as text chunks (one row per doc) ---
rows = df.astype(str).apply(lambda r: ", ".join(r), axis=1).tolist()
doc_text = "\n".join(rows)

"""
# --- Chunking ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
docs = text_splitter.create_documents([doc_text])
for i, doc in enumerate(docs):
    doc.metadata["chunk_id"] = i
""" 

# Skip chunking since rows are small and self-contained
docs = [Document(page_content=f"""
Product: {row['product']}
Revenue: {row['revenue']}
Sales: {row['sales']}
Profits: {row['profits']}
Description: {row['description']}
Region: {row['region']}
Date: {row['date']}
""", metadata={"row_id": row.name}) for _, row in df.iterrows()]



# --- Embedding + FAISS vector store ---
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Prompt template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an experienced analyst working with a sales table containing:
- product
- revenue
- sales
- profits
- description
- region
- date

Use ONLY the provided context to answer the question. Follow these steps:
1. Filter the context to find rows matching the question (e.g., specific region, product, or date).
2. For numerical questions (e.g., total revenue), sum the relevant values and round to two decimal places.
3. For questions asking for a single item (e.g., product with highest sales), return only the most relevant answer.
4. If the data is insufficient, respond with "Insufficient data".
5. Ensure answers are concise and match the expected format (e.g., numerical or single product name).

Context:
{context}

Question: {question}

Answer:
    """),
    ("human", "{question}")
])

# --- GPT-4o model ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- RAG QA Chain ---
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# --- Evaluation dataset ---
examples = [
    {"query": "What is the total revenue in North America?", "answer": "110000"},
    {"query": "Which product had the highest sales?", "answer": "Widget B"},
    {"query": "What was the profit of Gadget C?", "answer": "9000"},
    {"query": "What is the revenue on 2024-09-17?", "answer": "45000"},
    {"query": "Which product was sold in Asia?", "answer": "Gadget C"},
    {"query": "What was the revenue of Widget Z?", "answer": "Insufficient data"},
]

# --- Generate predictions ---
print("🔍 Generating answers via RAG...\n")
predictions = []
for ex in tqdm(examples):
    result = rag_chain.invoke({"query": ex["query"]})
    predictions.append({"output": result["result"].strip()})

# --- Evaluate predictions ---
eval_chain = QAEvalChain.from_llm(llm)
graded = eval_chain.evaluate(
    examples=examples,
    predictions=predictions,
    prediction_key="output"
)

# --- Reporting ---
correct = 0
results = []
for ex, pred, grade in zip(examples, predictions, graded):
    is_correct = grade["results"].strip().upper().startswith("CORRECT")  # ✅ FIXED
    results.append({
        "Question": ex["query"],
        "Ground Truth": ex["answer"],
        "GPT Answer": pred["output"],
        "Evaluation": grade["results"],
        "Correct": is_correct
    })
    print(f"Q: {ex['query']}")
    print(f"GT: {ex['answer']}")
    print(f"GPT: {pred['output']}")
    print(f"Eval: {grade['results']}")
    print("---")
    if is_correct:
        correct += 1

# --- Accuracy Summary ---
incorrect = len(examples) - correct
print(f"\n✅ Correct: {correct} ❌ Incorrect: {incorrect}")
print(f"🎯 LLM Accuracy: {(correct / len(examples)) * 100:.1f}%")

# --- Export to CSV ---
pd.DataFrame(results).to_csv("evaluation_results.csv", index=False)
print("📄 Results saved to evaluation_results.csv")

