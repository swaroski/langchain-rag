import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DataFrameLoader

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Load and prepare CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.getenv("CSV_PATH", os.path.join(BASE_DIR, "data", "data.csv"))

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

df = pd.read_csv(csv_path)
df["combined"] = df.astype(str).agg(" | ".join, axis=1)

loader = DataFrameLoader(df, page_content_column="combined")
docs = loader.load()

# Chunking and embeddings
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vs = Chroma.from_documents(splits, embedding=embeddings)
retriever = vs.as_retriever(search_kwargs={"k": 3})

# Prompt setup
SYSTEM_PROMPT = """You are a data analyst with exceptional analytics skills. Use chat history and context to think step-by-step.
If you don't know anything, say so.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{question}")
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

chain = (
    {
        "question": RunnablePassthrough(),
        "context": RunnablePassthrough(),
        "chat_history": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.set_page_config(page_title="CSV RAG Assistant", layout="wide")
st.title("📊 CSV RAG Chat Assistant")
st.markdown("Ask questions based on your CSV file.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask your question:")
if st.button("Submit") and user_input:
    # Prepare history and context
    chat_hist_str = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in st.session_state.chat_history) or "No prior history."
    docs = retriever.get_relevant_documents(user_input)
    context_str = "\n".join(d.page_content for d in docs)

    # Run chain
    with st.spinner("Thinking..."):
        try:
            answer = chain.invoke({
                "question": user_input,
                "context": context_str,
                "chat_history": chat_hist_str
            })
        except Exception as e:
            st.error(f"Error: {e}")
            answer = ""

    # Store memory
    st.session_state.chat_history.append((user_input, answer))

# Display history
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**User:** {q}")
        st.markdown(f"**Assistant:** {a}")
        st.markdown("---")

# Optional: Show data preview
with st.expander("🔍 View CSV Data"):
    st.dataframe(df)




"""
streamlit run rag_streamlit.py
"""