import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DataFrameLoader 

# Securely load your OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load CSV Data path--------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
csv_path = os.getenv("CSV_PATH", os.path.join(BASE_DIR, "data", "data.csv")) 

# Validate CSV file existence 
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}") 

# Load CSV into dataframe 
df  = pd.read_csv(csv_path) 

# Combine all columns into one one for embedding 
df["combined"] = df.astype(str).agg(" | ".join, axis=1) 


# Load documents using Dataframeloader 
loader = DataFrameLoader(df, page_content_column="combined") 
docs = loader.load() 

# Split large text rows (not always needed for CSVs, but adds flexibility)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# Embed and store vectors
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt setup
system_prompt = """You are a data analyst answering questions based on a provided dataset. Use the data to reason step-by-step and provide a clear, concise answer. If the data is insufficient, say so.

Example:
Data: "product: Laptop, category: Electronics, price: 999.99, stock: 50, region: North America"
Question: "What is the price of the Laptop?"
Reasoning:
1. Identify the relevant data: The data includes a row for the product 'Laptop'.
2. Extract the price information: The price is listed as 999.99.
3. Summarize the answer clearly.
Answer: The price of the Laptop is 999.99.

Now, use the following data to answer the question at the end, following a similar step-by-step reasoning process:

{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{question}")
])

# LLM with streaming (optional)
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

# LCEL Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query function
def query_rag(query: str):
    response = chain.invoke(query)
    return {
        "answer": response,
        "source_documents": retriever.invoke(query)
    }

# Q&A loop
def run_qa_loop():
    print("📊 Data Q&A System (CSV)")
    print(f"📁 Dataset loaded from: {csv_path}")
    print("📄 Columns:", ", ".join(df.columns))
    while True:
        query = input("\n🔍 Ask a question (or type 'quit'): ")
        if query.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        if not query.strip():
            print("⚠️ Please enter a valid question.")
            continue

        result = query_rag(query)
        print("\n💡 Answer:")
        print(result["answer"])
        print("\n📑 Relevant Rows:")
        for i, doc in enumerate(result["source_documents"], 1):
            snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            print(f"\nRow {i} (Index {doc.metadata.get('row_index', 'unknown')}):")
            print(snippet)

# Entry point
if __name__ == "__main__":
    try:
        run_qa_loop()
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
