import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables securely
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load PDF document
pdf_path = "path/to/starfish_document.pdf"  # Replace with your file path
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# Embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt
system_prompt = """You are a helpful assistant that answers questions based on provided context about starfish. Use the context to reason step-by-step and provide a clear, concise answer. If you don't know the answer, say so.

Example:
Context: "Starfish are part of the phylum Echinodermata, which includes sea urchins and sea cucumbers. They have a water vascular system that aids movement and feeding."
Question: "What is the water vascular system in starfish?"
Reasoning:
1. Identify the relevant information: The context mentions the water vascular system as a feature of starfish.
2. Explain its function: It aids in movement and feeding.
3. Summarize clearly.
Answer: The water vascular system in starfish is a hydraulic system that controls tube feet movement, enabling locomotion and feeding.

Now, use the following context to answer the question at the end, following a similar step-by-step reasoning process:

{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{question}")
])

# LLM with streaming
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

# LCEL chain
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
        "source_documents": retriever.get_relevant_documents(query)
    }

# CLI loop
def run_qa_loop():
    print("🐚 Starfish Q&A System (type 'quit' or 'exit' to stop)")
    print(f"Loaded: {pdf_path}")
    while True:
        query = input("\n🔍 Ask a question: ")
        if query.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        if not query.strip():
            print("⚠️ Please enter a valid question.")
            continue

        result = query_rag(query)
        print("\n💡 Answer:")
        print(result["answer"])

        print("\n📄 Source Snippets:")
        for i, doc in enumerate(result["source_documents"], 1):
            text = doc.page_content.strip()
            snippet = text[:300] + "..." if len(text) > 300 else text
            page = doc.metadata.get("page", "unknown")
            print(f"\n📘 Doc {i} (Page {page}):\n{snippet}")

if __name__ == "__main__":
    try:
        run_qa_loop()
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
