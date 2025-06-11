import os
from dotenv import load_dotenv
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
env_path = os.path.abspath(os.path.join(os.getcwd(), ".env"))
load_dotenv(dotenv_path=env_path)
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY not found in .env")

# Optional: set as global environment var for some clients
os.environ["OPENAI_API_KEY"] = openai_api_key

# Model (consider using 'gpt-4o' instead of 'gpt-4o-mini' for broader support)
openai_model = "gpt-4o"

# Load and process PDF
pdf_path = "starfish.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ PDF not found at: {pdf_path}")
    sys.exit(1)

loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# Vectorstore
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

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{question}")
])

# LLM with optional streaming
llm = ChatOpenAI(model=openai_model, temperature=0, streaming=True)

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
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

# CLI Test
if __name__ == "__main__":
    prompt = "What is the main topic of the starfish document?"
    result = query_rag(prompt)

    print("\n✅ Answer:\n", result["answer"])
    print("\n📄 Source Snippets:")
    for i, doc in enumerate(result["source_documents"], 1):
        content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        page = doc.metadata.get("page", "unknown")
        print(f"\nDocument {i} (Page {page}):\n{content}")
