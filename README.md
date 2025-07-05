# 🧠 Hulo Knowledge Base

> An intelligent knowledge base system for the Hulo programming language, powered by RAG (Retrieval-Augmented Generation), vector databases, and large language models.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

- **🔍 Semantic Search**: Advanced vector-based document search with multi-round retrieval
- **🤖 AI-Powered Q&A**: Get intelligent answers based on Hulo documentation
- **📚 Multi-language Support**: Handle both English and Chinese documentation
- **🏗️ Modular Architecture**: Pluggable vector stores (Memory/ChromaDB) and LLM providers
- **⚡ High Performance**: Async processing with FastAPI
- **🔒 Privacy-First**: Support for local LLM deployment with Ollama
- **📊 Rich Analytics**: Document statistics and search insights
- **🛠️ Developer Friendly**: RESTful API with automatic documentation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Vector Store  │    │   LLM Service   │
│   Processor     │───▶│   (Memory/      │───▶│   (OpenAI/      │
│                 │    │    ChromaDB)    │    │    Ollama)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Knowledge     │    │   FastAPI       │    │   Multi-round   │
│   Service       │◀───│   HTTP API      │◀───│   Retrieval     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Ollama](https://ollama.ai) (optional, for local LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/hulo-knowledge-base.git
   cd hulomind
   ```

2. **Initialize the documentation submodule**
   ```bash
   git submodule update --init --recursive
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize the knowledge base**
   ```bash
   uv run python scripts/init_knowledge_base.py
   ```

6. **Start the server**
   ```bash
   uv run python -m src.main
   ```

7. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 📖 Usage

### API Endpoints

#### 🔍 Search Documents
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "function definition in Hulo"}'
```

#### 🤖 Ask Questions
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the difference between let and var in Hulo?"}'
```

#### 📊 Get Statistics
```bash
curl "http://localhost:8000/api/stats"
```

#### 📚 List Documents
```bash
curl "http://localhost:8000/api/documents?category=grammar&language=en&limit=10"
```

### Python Client Example

```sh
uv run python -m tests.test_api
```

```python
import requests

# Search for documents
response = requests.post("http://localhost:8000/api/search", json={
    "query": "variable declaration"
})
results = response.json()
print(f"Found {results['total_results']} documents")

# Ask a question
response = requests.post("http://localhost:8000/api/ask", json={
    "query": "How do I create a function in Hulo?"
})
answer = response.json()
print(f"Answer: {answer['answer']}")
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |
| `DEBUG` | Enable debug mode | `false` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `DASHSCOPE_API_KEY` | Qwen API key | - |
| `DEFAULT_VECTOR_STORE` | Vector store type (`memory`/`chroma`) | `memory` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB persistence directory | `./data/chroma` |
| `EMBEDDING_MODEL` | Sentence transformer model | `sentence-transformers/all-MiniLM-L6-v2` |
| `DOCS_PATH` | Documentation path | `./docs/src` |

### LLM Providers

The system supports multiple LLM providers with automatic fallback:

1. **OpenAI** (requires `OPENAI_API_KEY`)
2. **Qwen API** (requires `DASHSCOPE_API_KEY`)
3. **Local Ollama** (requires Ollama running with `qwen2.5:7b`)
4. **Mock Service** (fallback for testing)

### Vector Stores

- **Memory Store**: Fast, in-memory storage (default)
- **ChromaDB**: Persistent, production-ready storage

## 🧪 Development

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black src/
uv run isort src/
```

### Type Checking
```bash
uv run mypy src/
```

### Linting
```bash
uv run ruff check src/
```

## 📁 Project Structure

```
hulo-knowledge-base/
├── src/
│   ├── config/           # Configuration management
│   ├── models/           # Data models
│   ├── processors/       # Document processing
│   ├── vectorstore/      # Vector storage backends
│   ├── services/         # Business logic services
│   ├── mcp/             # MCP protocol support
│   └── main.py          # FastAPI application
├── scripts/             # Utility scripts
├── tests/               # Test suite
├── docs/                # Documentation (submodule)
├── data/                # Data storage
└── pyproject.toml       # Project configuration
```

## 🔧 Advanced Features

### Multi-round Retrieval

The system uses a sophisticated two-stage retrieval process:

1. **Broad Search**: Low threshold (0.3) to capture more candidates
2. **Refined Search**: High threshold (0.7) to filter high-quality results
3. **Merge & Deduplicate**: Combine results and remove duplicates

### Smart Document Chunking

Documents are intelligently split based on Markdown headers rather than fixed line counts, preserving semantic integrity.

### Vector Store Abstraction

```python
from src.vectorstore import VectorStoreFactory

# Use memory store
memory_store = VectorStoreFactory.create("memory")

# Use ChromaDB store
chroma_store = VectorStoreFactory.create("chroma")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
