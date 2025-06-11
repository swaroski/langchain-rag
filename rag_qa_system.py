import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load CSV data
csv_path = "path/to/sample_data.csv"  # update this
df = pd.read_csv(csv_path)

# Convert rows to LangChain Documents
documents = []
for _, row in df.iterrows():
    content = ", ".join([f"{key}: {value}" for key, value in row.items()])
    metadata = {"row_index": row.name}
    documents.append(Document(page_content=content, metadata=metadata))

# Optional: split long rows
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(documents)

# Embeddings + Vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# In-memory chat
chat_history = InMemoryChatMessageHistory()

# Prompt
system_prompt = """You are a data analyst answering questions based on a provided dataset. Use the data and chat history to reason step-by-step and provide a clear, concise answer. If the data or history is insufficient, say so.

Chat History:
{chat_history}

Data Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{question}")
])

# LLM (GPT-4o with streaming)
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

# LCEL Chain
base_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "chat_history": lambda _: "\n".join([
            f"Human: {msg.content}\nAssistant: {msg.additional_kwargs.get('answer', '')}"
            for msg in chat_history.messages
        ]) or "No history yet."
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Add memory
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: chat_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# Query handler
def query_rag(query):
    response = chain_with_history.invoke(
        {"question": query},
        config={"configurable": {"session_id": "default"}}
    )
    chat_history.add_user_message(query)
    chat_history.add_ai_message(response, additional_kwargs={"answer": response})
    return {
        "answer": response,
        "source_documents": retriever.get_relevant_documents(query)
    }

# Interactive QA
def run_qa_loop():
    print("📊 Data Q&A System with Memory (type 'quit' or 'exit' to stop)")
    print(f"✅ Loaded CSV: {csv_path}")
    print("📁 Columns:", ", ".join(df.columns))
    
    while True:
        query = input("\n🔍 Your question: ")
        if query.lower() in ["quit", "exit"]:
            print("👋 Exiting. Goodbye!")
            break
        if not query.strip():
            print("❗ Please enter a valid question.")
            continue
        
        result = query_rag(query)
        print("\n💡 Answer:")
        print(result["answer"])
        
        print("\n📄 Source Rows:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\nRow {i} (Index: {doc.metadata.get('row_index', 'N/A')}):")
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

        print("\n🗣️ Chat History:")
        for msg in chat_history.messages:
            print(f"Human: {msg.content}")
            print(f"Assistant: {msg.additional_kwargs.get('answer', msg.content)}")

if __name__ == "__main__":
    try:
        run_qa_loop()
    except KeyboardInterrupt:
        print("\n👋 Exiting. Goodbye!")
