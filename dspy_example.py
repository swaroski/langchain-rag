import dspy
from dspy import Signature, InputField, OutputField, Predict
from dotenv import load_dotenv
import os 


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4")

llm = dspy.LM(
    model="openai/gpt-4o", 
    api_key=OPENAI_API_KEY,
    max_tokens=4096
)
dspy.settings.configure(lm=llm)

class CodeExplanation(dspy.Signature):
    """Explain what a piece of code does."""
    code: str = dspy.InputField(desc="The code to explain")
    language: str = dspy.InputField(desc="Programming language")
    explanation: str = dspy.OutputField(desc="Clear explanation of the code. Not more than 150 characters.")
    key_concepts: str = dspy.OutputField(desc="Main concepts used")

class CodeExplainer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(CodeExplanation)
    def forward(self, code, language="Python"):
        return self.predict(code=code, language=language)

# Try it out
explainer = CodeExplainer()
sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
result = explainer(sample_code)
print(f"Explanation: {result.explanation}")
print(f"Key concepts: {result.key_concepts}")