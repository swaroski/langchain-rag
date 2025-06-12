import os
import pandas as pd
import argparse
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DataFrameLoader
from tqdm import tqdm

# --- Setup Environment ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""


# --- Load sales data ---
df = pd.read_csv("sales_data.csv")

# --- Load and preprocess data ---
#df = pd.read_json("sales_data.json")

# Read a text file with space as the delimiter
#df = pd.read_csv('file.txt', sep=' ')

# Read a text file with tab as the delimiter
#df = pd.read_table('file.txt', sep='\t')

# --- Format as structured documents ---
df["combined"] = df.fillna("").astype(str).apply(lambda row: "\n".join(f"{k}: {v}" for k, v in row.items()), axis=1)
df["row_id"] = df.index
loader = DataFrameLoader(df[["combined", "row_id"]], page_content_column="combined")
docs = loader.load()

# --- Embedding + FAISS vector store ---
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Increased k for better retrieval


# --- Prompt Factory ---
def get_prompt(mode="prompt"):
    if mode == "zero_shot_prompt":
        return ChatPromptTemplate.from_messages([
            ("system", """You are a senior sales analyst.

You are provided with context in the form of rows from a sales table. Each row contains:
- product
- revenue
- sales
- profits
- description
- region
- date

Your task is to answer the question using **only** the provided context. Follow these rules:
- Use exact values from matching rows.
- If aggregation is needed (e.g., total revenue or total sales), perform the correct calculation.
- If the required data is not in the context, reply with: "Insufficient data".
- Keep your answer concise and factual.
"""),
            ("human", """
Context:
{context}

Question: {question}

Answer:
""")
        ])
    elif mode == "few_shot_prompt":
        return ChatPromptTemplate.from_messages([
            ("system", """
You are a senior sales analyst. Use the context to answer the question by aggregating data as needed.
"""),
            ("human", """
Example:
Context:
Product: Widget A
Revenue: 10000.00
Sales: 50
Region: North
Product: Widget A
Revenue: 15000.00
Sales: 75
Region: North

Question: What is the total revenue for Widget A in North?
Answer: The total revenue for Widget A in North is $25,000.00.

---
Context:
{context}

Question: {question}
Answer:
""")
        ])
    elif mode == "chain_of_thought_prompt":
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
    elif mode == "structured_prompt":
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
    elif mode == "strict_instruction_prompt":
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
1. Identify and filter rows in the context that exactly match the question's criteria (e.g., Region: North America, Date: 2024-09-17, or Product: Gadget C).
2. For numerical questions (e.g., total revenue, profit):
   - Sum the relevant values (e.g., Revenue, Profits) from filtered rows.
   - Round the result to two decimal places.
   - Format as 'Total Revenue: $xx,xxx.xx' or 'Total Profit: $xx,xxx.xx'.
3. For questions asking for a single item (e.g., product with highest sales):
   - Compare relevant values (e.g., Sales) across filtered rows.
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
    elif mode == "self_ask_prompt":
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

# --- Evaluation Prompt ---
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert evaluator comparing predicted answers to ground truth answers for a sales dataset. Follow these steps:
1. Compare the predicted answer to the ground truth answer.
2. For numerical questions (e.g., revenue, profit):
   - Extract numerical values, ignoring formatting (e.g., '$110,000.00' or 'Total Revenue: $110,000.00' → 110000).
   - Consider answers equivalent if they match within 0.01.
3. For single-item questions (e.g., product name):
   - Check if the predicted product name exactly matches the ground truth (case-sensitive).
4. For 'Insufficient data':
   - Check if both predicted and ground truth are exactly 'Insufficient data'.
5. Return a JSON object with:
   - is_correct: boolean (true if equivalent, false otherwise)
   - reasoning: string explaining why the answer is correct or incorrect

Examples:
Question: What is the total revenue in North America?
Ground Truth: 110000
Predicted: Total Revenue: $110,000.00
Output: {{"is_correct": true, "reasoning": "The predicted answer '$110,000.00' matches the ground truth 110000 numerically."}}

Question: Which product had the highest sales?
Ground Truth: Widget B
Predicted: Gadget I
Output: {{"is_correct": false, "reasoning": "The predicted product 'Gadget I' does not match the ground truth 'Widget B'."}}

Question: What was the revenue of Widget Z?
Ground Truth: Insufficient data
Predicted: Insufficient data
Output: {{"is_correct": true, "reasoning": "Both predicted and ground truth are 'Insufficient data'."}}
"""),
    ("human", """
Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Output (JSON):
""")
])


# --- Prompt template ---
prompt = get_prompt(mode="zero_shot_prompt")

# --- GPT-4o model ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Custom retrieval and inference ---
def run_query(query, verbose=False):
    if not query.strip():
        return "Please enter a valid question."
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = prompt.format_prompt(context=context, question=query)
    response = llm.invoke(formatted_prompt.to_messages()).content.strip()
    if verbose:
        print(f"\nRetrieved Context:\n{[doc.page_content for doc in retrieved_docs]}\n")
    return response

# --- Evaluation function ---
def run_evaluation():
    examples = [
        {"query": "What is the total revenue in North America?", "answer": "110000"},
        {"query": "Which product had the highest sales?", "answer": "Widget B"},
        {"query": "What was the profit of Gadget C?", "answer": "9000"},
        {"query": "What is the revenue on 2024-09-17?", "answer": "45000"},
        {"query": "Which product was sold in Asia?", "answer": "Gadget C"},
        {"query": "What was the revenue of Widget Z?", "answer": "Insufficient data"},
    ]
    
    print("🔍 Generating answers via RAG...\n")
    predictions = []
    for ex in tqdm(examples, desc="Generating predictions"):
        response = run_query(ex["query"])
        predictions.append({"output": response})

    print("🔍 Evaluating predictions...\n")
    correct = 0
    results = []
    for ex, pred in zip(examples, predictions):
        try:
            formatted_eval_prompt = eval_prompt.format_prompt(
                question=ex["query"],
                ground_truth=ex["answer"],
                predicted=pred["output"]
            )
            eval_response = llm.invoke(formatted_eval_prompt.to_messages()).content.strip()
            eval_result = json.loads(eval_response)
            is_correct = eval_result["is_correct"]
            reasoning = eval_result["reasoning"]
        except (json.JSONDecodeError, KeyError) as e:
            is_correct = False
            reasoning = f"Evaluation failed: {str(e)}. Response: {eval_response}"
        
        results.append({
            "Question": ex["query"],
            "Ground Truth": ex["answer"],
            "GPT Answer": pred["output"],
            "Correct": is_correct,
            "Reasoning": reasoning
        })
        print(f"Q: {ex['query']} | GT: {ex['answer']} | GPT: {pred['output']} | Correct: {is_correct} | Reasoning: {reasoning}")

    accuracy = (correct / len(examples)) * 100
    print(f"\n✅ Correct: {correct} ❌ Incorrect: {len(examples) - correct} 🎯 Accuracy: {accuracy:.1f}%")
    pd.DataFrame(results).to_csv("evaluation_results.csv", index=False)
    print("📄 Results saved to evaluation_results.csv")

    if accuracy < 95:
        print("\nDataset Verification (low accuracy detected):")
        print(f"North America Revenue Sum: {df[df['region'] == 'North America']['revenue'].sum()}")
        print(f"Widget B Max Sales: {df[df['product'] == 'Widget B']['sales'].max()}")
        print(f"Gadget C Profit: {df[df['product'] == 'Gadget C']['profits'].tolist()}")
        print(f"2024-09-17 Revenue: {df[df['date'] == '2024-09-17']['revenue'].tolist()}")
        print(f"Asia Products: {df[df['region'] == 'Asia']['product'].unique().tolist()}")

# --- Terminal Q&A System ---
def run_terminal_qa(verbose=False):
    print("Welcome to the Sales Data Q&A System!")
    print("Enter your question (or 'quit' to exit).")
    while True:
        query = input("\nEnter your question: ").strip()
        if query.lower() in ['quit', 'exit']:
            print("Exiting Q&A system.")
            break
        response = run_query(query, verbose=verbose)
        print(f"Answer: {response}")

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Data Q&A System")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation instead of interactive Q&A")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved context for debugging")
    args = parser.parse_args()

    if args.evaluate:
        run_evaluation()
    else:
        run_terminal_qa(verbose=args.verbose)