


## 🚀 Getting Started
### Clone the repository and install dependencies:

### UV Python Install for Version Management
#### Manage Python versions with UV python install:
```bash
uv python install 3.11
uv python pin 3.11
```

```bash
git clone https://github.com/swaroski/langchain-rag.git
cd langchain-rag

uv init
uv venv
uv pip install -r requirements.txt

uv run rag.py
```

## 📦 Add Packages
### To add a new package to your project:

```bash
uv add <package-name>
```

## 🛠️ Create a New Project with UV Venv
```bash
uv venv my-project
source my-project/bin/activate      # Linux/macOS
my-project\Scripts\activate         # Windows
```

## 📌 Pin Dependencies with UV
### To lock exact versions for consistency across environments:

```bash

uv pip compile requirements.in --output-file requirements.txt
```

### Then sync your environment:

```bash
uv pip sync requirements.txt
```


## Install FastAPI:
```bash
uv pip install fastapi uvicorn
```

## Write a Simple App:
### Create main.py:
```bash
from fastapi import FastAPI app = FastAPI() @app.get("/") async def root(): return {"message": "Hello, UV Python!"}
```

### Run the App:
```bash
uvicorn main:app --reload
```

### Pin Dependencies:
```bash
uv pip compile requirements.in --output-file requirements.txt uv pip sync requirements.txt
```

## Docker setup

```bash 
FROM python:3.11-slim AS builder
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
COPY requirements.in .
RUN uv pip compile requirements.in --output-file requirements.txt
RUN uv pip sync requirements.txt --target /install

FROM python:3.11-slim
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY . /app
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

