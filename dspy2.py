import os
import json
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset

# --- Setup Environment ---
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Step 1: Load and preprocess dataset ---
def load_sales_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["text"] = df.apply(
        lambda row: (
            f"Product: {row['product']}, Revenue: ${row['revenue']:.2f}, Sales: {row['sales']}, "
            f"Region: {row['region']}, Date: {row['date']}."
        ),
        axis=1
    )
    return df

# --- Step 2: Create vector store ---
def create_vector_store(texts: List[str]) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_texts(chunks, embeddings)

# --- Step 3: Q&A Chain ---
def setup_qa_chain(vector_store: FAISS) -> RunnableSequence:
    def debug_and_return_context(question, retriever):
        docs = retriever.invoke(question)
        print(f"\n🔍 Retrieved for '{question}':")
        for doc in docs:
            print("-", doc.page_content)
        return "\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a sales analyst. Use the provided context to answer the question.
If multiple entries are shown, aggregate the values to compute totals.
If the answer cannot be found, respond with \"Insufficient data\".

Context:
{context}

Question: {question}
Answer:
"""
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    chain = RunnableSequence(
        {
            "context": lambda x: debug_and_return_context(x["question"], retriever),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Step 4: Classification ---
def classify_performance(data: Dict) -> Dict:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        input_variables=["data"],
        template="""
Classify the product's performance as 'strong' if revenue > $10,000, otherwise 'weak'. 
Return the answer in JSON format.

Product Data: {data}
Answer:
```json
{{ "performance": "strong" }}
```
"""
    )
    chain = prompt | llm | StrOutputParser()
    data_str = f"Product: {data['product']}, Revenue: ${data['revenue']:.2f}"
    response = chain.invoke({"data": data_str})

    try:
        json_str = response.split("```json")[-1].split("```")[0].strip()
        return json.loads(json_str)
    except Exception as e:
        print("Classification parse error:", e)
        return {"performance": "unknown"}

# --- Step 5: Synthetic Q&A Generation with Aggregation ---
def generate_synthetic_qa(df: pd.DataFrame, num_questions: int = 10) -> Dataset:
    qa_pairs = []
    sampled_rows = df[["product", "region"]].drop_duplicates().sample(n=min(num_questions // 2, len(df)))

    for _, row in sampled_rows.iterrows():
        product = row["product"]
        region = row["region"]

        subset = df[(df["product"] == product) & (df["region"] == region)]
        if subset.empty:
            continue

        total_revenue = round(subset["revenue"].sum(), 2)
        total_sales = int(subset["sales"].sum())

        qa_pairs.append({
            "question": f"What was the total revenue generated from {product} in {region}?",
            "answer": f"The total revenue generated from {product} in {region} was ${total_revenue:,.2f}."
        })
        qa_pairs.append({
            "question": f"How many units of {product} were sold in {region}?",
            "answer": f"A total of {total_sales} units of {product} were sold in {region}."
        })

    qa_pairs = qa_pairs[:num_questions]

    return Dataset.from_dict({
        "question": [pair["question"] for pair in qa_pairs],
        "ground_truth": [pair["answer"] for pair in qa_pairs]
    })

# --- Step 6: Evaluation ---
def evaluate_answers(chain: RunnableSequence, testset: Dataset) -> List[Dict]:
    results = []
    for pair in testset:
        question = pair["question"]
        ground_truth = pair["ground_truth"]
        answer = chain.invoke({"question": question})
        exact_match = answer.strip().lower() == ground_truth.strip().lower()
        results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "exact_match": exact_match
        })
    accuracy = sum(r["exact_match"] for r in results) / len(results)
    return results, accuracy

# --- Main Execution ---
def main():
    df = load_sales_data("sales_data.csv")

    vector_store = create_vector_store(df["text"].tolist())
    qa_chain = setup_qa_chain(vector_store)

    test_question = "What is the revenue for Product A in North region?"
    print(f"Q&A Test: {test_question}\nAnswer: {qa_chain.invoke({'question': test_question})}\n")

    sample_data = {"product": "A", "revenue": 15000}
    classification = classify_performance(sample_data)
    print(f"Classification Test: {sample_data}\nResult: {classification}\n")

    testset = generate_synthetic_qa(df, num_questions=4)
    print("Synthetic Q&A Test:")
    for pair in testset:
        print(f"Question: {pair['question']}\nGround Truth: {pair['ground_truth']}\n")

    results, accuracy = evaluate_answers(qa_chain, testset)
    print(f"Evaluation Accuracy: {accuracy:.2f}")
    for result in results:
        print(f"Question: {result['question']}\nAnswer: {result['answer']}\nGround Truth: {result['ground_truth']}\nMatch: {result['exact_match']}\n")

if __name__ == "__main__":
    main()