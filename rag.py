import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set your OpenAI API key
#os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

import os
from dotenv import load_dotenv

# Construct the correct relative path
env_path = os.path.abspath(os.path.join(os.getcwd(), ".env"))
load_dotenv(dotenv_path=env_path)

# Read OpenAI key from .env
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

os.environ["OPENAI_API_KEY"] = openai_api_key  # Optional: make it globally available

openai_model = "gpt-4o-mini"

# Load PDF documents (replace with your PDF paths)
#documents = ["path/to/document1.pdf", "path/to/document2.pdf"]
documents = ["starfish.pdf"]
docs = []
for doc_path in documents:
    loader = PyPDFLoader(doc_path)
    docs.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define the prompt template
system_prompt = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}"""
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("{question}")
])

# Set up the LCEL chain
llm = ChatOpenAI(model=openai_model, temperature=0)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Function to query the RAG system
def query_rag(query):
    response = chain.invoke(query)
    return {
        "answer": response,
        "source_documents": retriever.get_relevant_documents(query)
    }

# Example usage
if __name__ == "__main__":
    prompt = "What is the main topic of the documents?"
    response = query_rag(prompt)
    print("Answer:", response["answer"])
    print("\nSource Documents:")
    for doc in response["source_documents"]:
        print(doc.page_content[:100] + "...")