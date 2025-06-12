import pandas as pd
import dspy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from dspy.teleprompt import BootstrapFewShot
import asyncio
from uuid import uuid4
import os
from typing import Dict

# Step 1: CSV Loader
def load_sales_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Assume CSV has columns: question, answer, context
    return df

# Step 2: Chunking
def chunk_text(texts: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks

# Step 3: Embedding and FAISS VectorStore
def create_vector_store(chunks: list) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Step 4: Retriever
def setup_retriever(vector_store: FAISS):
    return vector_store.as_retriever(search_kwargs={"k": 3})

# Step 5: PromptTemplate + LangChain LLM with Memory
def setup_llm_chain() -> tuple[RunnableSequence, ConversationBufferMemory]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""
        You are a sales assistant. Use the following context and conversation history to answer the question concisely.

        Context: {context}

        Conversation History: {history}

        Question: {question}

        Answer:
        """
    )
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question",
        return_messages=True
    )
    # Use RunnableSequence instead of LLMChain
    chain = RunnableSequence(
        {
            "context": lambda x: "\n".join([doc.page_content for doc in x["docs"]]),
            "question": lambda x: x["question"],
            "history": lambda x: memory.load_memory_variables({})["history"]
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain, memory

# Step 6: Response Generation
async def generate_response(chain: RunnableSequence, retriever, question: str, memory: ConversationBufferMemory) -> tuple[str, str]:
    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Generate response
    response = await chain.ainvoke({"question": question, "docs": docs})
    
    # Save to memory
    memory.save_context({"question": question}, {"output": response})
    
    return response, context

# Step 7: DSPy Evaluation
class EvaluateQA(dspy.Signature):
    """Evaluate if the generated answer matches the expected answer."""
    question: str = dspy.InputField()
    generated_answer: str = dspy.InputField()
    expected_answer: str = dspy.InputField()
    is_correct: bool = dspy.OutputField()
    feedback: str = dspy.OutputField(desc="Explanation of evaluation")

class QAEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.Predict(EvaluateQA)
    
    def forward(self, question, generated_answer, expected_answer):
        return self.evaluate(
            question=question,
            generated_answer=generated_answer,
            expected_answer=expected_answer
        )

# Optimize Evaluator
def optimize_evaluator(evaluator: QAEvaluator, training_examples: list) -> QAEvaluator:
    optimizer = BootstrapFewShot(metric=lambda x, y: y.is_correct)
    # Ensure input fields are explicitly set
    training_examples = [
        ex.with_inputs("question", "generated_answer", "expected_answer")
        for ex in training_examples
    ]
    optimized_evaluator = optimizer.compile(evaluator, trainset=training_examples)
    return optimized_evaluator

# Async Evaluation for Multiple Q&A Pairs
async def evaluate_qa_pairs(pairs: list, evaluator: QAEvaluator, chain: RunnableSequence, retriever, memory: ConversationBufferMemory) -> list:
    results = []
    for pair in pairs:
        question = pair["question"]
        expected_answer = pair["answer"]
        generated_answer, _ = await generate_response(chain, retriever, question, memory)
        eval_result = evaluator(
            question=question,
            generated_answer=generated_answer,
            expected_answer=expected_answer
        )
        results.append({
            "question": question,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "is_correct": eval_result.is_correct,
            "feedback": eval_result.feedback
        })
    return results

# Main Execution
async def main():
    # Configure DSPy
    llm = dspy.LM(model="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.settings.configure(lm=llm)

    # Load and process data
    csv_path = "sales_qa_data.csv"
    df = load_sales_data(csv_path)
    contexts = df["context"].tolist()
    chunks = chunk_text(contexts)
    vector_store = create_vector_store(chunks)
    retriever = setup_retriever(vector_store)
    
    # Setup LangChain
    chain, memory = setup_llm_chain()

    # Setup DSPy Evaluator
    evaluator = QAEvaluator()
    
    # Example training data for optimization
    training_examples = [
        dspy.Example(
            question="What is the price of Product X?",
            generated_answer="Product X costs $49.99.",
            expected_answer="Product X costs $49.99.",
            is_correct=True,
            feedback="Exact match."
        ),
        dspy.Example(
            question="What is our return policy?",
            generated_answer="30-day returns with receipt.",
            expected_answer="30-day return policy with original receipt.",
            is_correct=True,
            feedback="Minor phrasing difference but correct."
        )
    ]
    optimized_evaluator = optimize_evaluator(evaluator, training_examples)

    # Evaluate Q&A pairs
    qa_pairs = df[["question", "answer"]].to_dict("records")
    results = await evaluate_qa_pairs(qa_pairs, optimized_evaluator, chain, retriever, memory)

    # Print results
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Generated: {result['generated_answer']}")
        print(f"Expected: {result['expected_answer']}")
        print(f"Correct: {result['is_correct']}")
        print(f"Feedback: {result['feedback']}\n")

# Run the pipeline
if __name__ == "__main__":
    asyncio.run(main())

