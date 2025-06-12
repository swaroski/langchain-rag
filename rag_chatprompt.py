import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from tqdm import tqdm
from langchain_community.document_loaders import DataFrameLoader 

# --- Setup Environment ---
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""


# --- Setup Environment ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Load sales data ---
df = pd.read_csv("sales_data.csv")

"""
# --- Chunking ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
docs = text_splitter.create_documents([doc_text])
for i, doc in enumerate(docs):
    doc.metadata["chunk_id"] = i
""" 



# Combine all columns into one for embedding
"""
df.astype(str).apply(...): Converts all values to strings and joins each rows 
key-value pairs with newlines 
(e.g., Product: Widget A\nRevenue: 10000).
DataFrameLoader: Creates Document objects with page_content from the combined 
column and metadata from other columns (or a custom index).
"""
# --- Format as structured documents ---
# Combine all columns into one for embedding
df["combined"] = df.astype(str).apply(lambda row: "\n".join(f"{k}: {v}" for k, v in row.items()), axis=1)
# Add row_id for metadata
df["row_id"] = df.index
# Load documents using DataFrameLoader
loader = DataFrameLoader(df[["combined", "row_id"]], page_content_column="combined")
docs = loader.load()

# --- Embedding + FAISS vector store ---
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# --- Prompt Factory ---
def get_prompt(mode="few_shot"):
    if mode == "few_shot":
        return ChatPromptTemplate.from_messages([
            ("system", """
You are a senior sales analyst. Use the context to answer the question by aggregating data as needed.
"""),
            ("human", """
Example:
Context:
Product: Widget A, Revenue: $10000.00, Sales: 50, Region: North.
Product: Widget A, Revenue: $15000.00, Sales: 75, Region: North.

Question: What is the total revenue for Widget A in North?
Answer: The total revenue for Widget A in North is $25,000.00.

---
Context:
{context}

Question: {question}
Answer:
""")
        ])
    elif mode == "chain_of_thought":
        return ChatPromptTemplate.from_messages([
            ("system", """
You are a senior sales analyst.

Instructions:
- Step 1: Identify matching entries from context.
- Step 2: Sum their values (Revenue, Sales).
- Step 3: Show reasoning first, then final answer in one of these formats:
  - Total Revenue: $xx,xxx.xx
  - Units Sold: xxx
  - Insufficient data
"""),
            ("human", """
Context:
{context}

Question: {question}

Answer:
""")
        ])
    elif mode == "structured":
        return ChatPromptTemplate.from_messages([
            ("system", """
You are a sales assistant. Based on the context, answer the question and return the result as structured JSON.
"""),
            ("human", """
Context:
{context}

Question: {question}

Output (only JSON):
```json
{{ "answer": "Total Revenue: $25,000.00" }}
```
""")
        ])
    elif mode == "strict_instruction":
        return ChatPromptTemplate.from_messages([
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
1. Identify and filter rows in the context that exactly match the question's criteria (e.g., region='North America', date='2024-09-17', or product='Gadget C').
2. For numerical questions (e.g., total revenue, profit):
   - Sum the relevant values (e.g., revenue, profits) from the filtered rows.
   - Round the result to two decimal places.
   - Format as 'Total Revenue: $xx,xxx.xx' or 'Total Profit: $xx,xxx.xx'.
3. For questions asking for a single item (e.g., product with highest sales):
   - Compare relevant values (e.g., sales) across filtered rows.
   - Return only the most relevant item (e.g., product name) without additional text.
4. If no rows match the criteria or data is missing, respond with 'Insufficient data'.
5. Ensure answers are concise and match the expected format:
   - Numerical: 'Total Revenue: $xx,xxx.xx' or 'Total Profit: $xx,xxx.xx'.
   - Single item: Product name only (e.g., 'Widget B').
   - Missing data: 'Insufficient data'.
"""),
            ("human", """
Context:
{context}

Question: {question}
Answer:
""")
        ])
    elif mode == "self_ask":
        return ChatPromptTemplate.from_messages([
            ("system", """
You are a senior sales analyst. Break the question into sub-steps if complex. Think step-by-step.

Example:
Question: What's the average revenue for Widget A in Asia and North America?

Step 1: Identify all entries for Widget A in Asia and North America.
Step 2: Sum total revenue and count entries.
Step 3: Compute average.

Return:
- Average Revenue: $xx,xxx.xx
OR
- Insufficient data
"""),
            ("human", """
Context:
{context}

Question: {question}
Answer:
""")
        ])
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

# --- Prompt template ---
prompt = get_prompt(mode="strict_instruction")  # Change mode here to test others

# --- GPT-4o model ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Custom retrieval and inference ---
def run_query(query):
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = prompt.format_prompt(context=context, question=query)
    response = llm.invoke(formatted_prompt.to_messages()).content.strip()
    return response

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
    response = run_query(ex["query"])
    predictions.append({"output": response})

# --- Evaluate predictions ---
correct = 0
results = []
for ex, pred in zip(examples, predictions):
    gt = ex["answer"]
    pred_output = pred["output"]
    is_correct = str(gt).strip() == str(pred_output).strip()  # Exact match
    results.append({
        "Question": ex["query"],
        "Ground Truth": gt,
        "GPT Answer": pred_output,
        "Correct": is_correct
    })
    print(f"Q: {ex['query']}")
    print(f"GT: {gt}")
    print(f"GPT: {pred_output}")
    print(f"Correct: {is_correct}")
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

# --- Debug Dataset ---
print("\nDataset Verification:")
print(f"North America Revenue Sum: {df[df['region'] == 'North America']['revenue'].sum()}")
print(f"Widget B Max Sales: {df[df['product'] == 'Widget B']['sales'].max()}")
print(f"Gadget C Profit: {df[df['product'] == 'Gadget C']['profits'].tolist()}")
print(f"2024-09-17 Revenue: {df[df['date'] == '2024-09-17']['revenue'].tolist()}")
print(f"Asia Products: {df[df['region'] == 'Asia']['product'].unique().tolist()}")