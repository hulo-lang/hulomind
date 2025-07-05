"""Main FastAPI application for Hulo Knowledge Base."""

import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.settings import settings
from .processors.document_loader import DocumentLoader
from .vectorstore import VectorStoreFactory
from .services.knowledge_service import KnowledgeService
from .services.llm_service import LLMService, LocalLLMService, OpenAIService, QwenService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
knowledge_service: Optional[KnowledgeService] = None
llm_service: Optional[LLMService] = None
vector_store = None


def create_llm_service() -> LLMService:
    """Create LLM service based on available options."""
    import os
    import requests
    
    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("Using OpenAI service")
        return LLMService(OpenAIService(openai_key, "gpt-4o-mini"))
    
    # Try Qwen API
    qwen_key = os.getenv("DASHSCOPE_API_KEY")
    if qwen_key:
        logger.info("Using Qwen API service")
        return LLMService(QwenService(qwen_key, "qwen-plus"))
    
    # Try local Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            logger.info("Using local Ollama service")
            return LLMService(LocalLLMService("qwen2.5:7b"))
    except:
        pass
    
    # Fallback to mock service
    logger.warning("No LLM service available, using mock responses")
    from .services.llm_service import MockLLMProvider
    return LLMService(MockLLMProvider())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI application."""
    global knowledge_service, llm_service, vector_store
    
    # Startup
    logger.info("Starting Hulo Knowledge Base...")
    
    # Initialize LLM service
    llm_service = create_llm_service()
    
    # Load documents and create chunks
    logger.info("Loading documents and creating chunks...")
    loader = DocumentLoader()
    documents = loader.load_documents(max_workers=2)
    
    if documents:
        chunks = loader.create_all_chunks(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStoreFactory.create()
        vector_store.add_chunks(chunks)
        
        # Initialize knowledge service
        knowledge_service = KnowledgeService(vector_store)
        
        logger.info("Hulo Knowledge Base started successfully!")
    else:
        logger.warning("No documents found")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Hulo Knowledge Base...")
    # Add any cleanup code here if needed


# Create FastAPI app
app = FastAPI(
    title="Hulo Knowledge Base",
    description="RAG + MCP + LLMs based knowledge base for Hulo programming language",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Hulo Knowledge Base",
        "version": "0.1.0",
        "description": "RAG + MCP + LLMs based knowledge base for Hulo programming language",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "knowledge_service": "running" if knowledge_service else "unavailable",
            "llm_service": "available" if llm_service else "unavailable",
            "vector_store": "running" if vector_store else "unavailable"
        }
    }


@app.post("/api/search")
async def search_documents(request: Dict[str, Any]):
    """Search documents using semantic search."""
    try:
        if not knowledge_service:
            raise HTTPException(status_code=503, detail="Knowledge service not initialized")
        
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Multi-round search
        results, context = knowledge_service.search_with_context(query, max_results=5)
        
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": result.chunk.id,
                    "title": result.chunk.metadata.get("title", ""),
                    "content": result.chunk.content,
                    "similarity": result.similarity,
                    "round": result.round,
                    "metadata": result.chunk.metadata
                }
                for result in results
            ],
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
async def ask_question(request: Dict[str, Any]):
    """Ask a question and get AI-powered answer."""
    try:
        if not knowledge_service or not llm_service:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Multi-round search
        results, context = knowledge_service.search_with_context(query, max_results=3)
        
        # Generate answer using LLM
        answer = llm_service.answer_question(context, query)
        
        return {
            "query": query,
            "answer": answer,
            "search_results": [
                {
                    "chunk_id": result.chunk.id,
                    "title": result.chunk.metadata.get("title", ""),
                    "similarity": result.similarity,
                    "round": result.round,
                    "metadata": result.chunk.metadata
                }
                for result in results
            ],
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    """Get knowledge base statistics."""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        stats = vector_store.get_stats()
        
        # Add additional stats
        stats.update({
            "total_documents": len(set(chunk.metadata.get("document_id") for chunk in vector_store.chunks)),
            "languages": {},
            "categories": {}
        })
        
        # Count languages and categories
        for chunk in vector_store.chunks:
            lang = chunk.metadata.get("language", "unknown")
            cat = chunk.metadata.get("category", "unknown")
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reload")
async def reload_documents():
    """Reload documents from source directory."""
    try:
        global knowledge_service, vector_store
        
        # Reload documents
        loader = DocumentLoader()
        documents = loader.load_documents(max_workers=2)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found")
        
        chunks = loader.create_all_chunks(documents)
        
        # Reinitialize vector store
        vector_store = VectorStoreFactory.create()
        vector_store.add_chunks(chunks)
        
        # Reinitialize knowledge service
        knowledge_service = KnowledgeService(vector_store)
        
        return {
            "success": True,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error reloading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def get_documents(
    category: str = None,
    language: str = None,
    limit: int = 50
):
    """Get documents with optional filtering."""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        # Get unique documents from chunks
        documents = {}
        for chunk in vector_store.chunks:
            doc_id = chunk.metadata.get("document_id")
            if doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "title": chunk.metadata.get("title", ""),
                    "category": chunk.metadata.get("category", ""),
                    "language": chunk.metadata.get("language", ""),
                    "tags": chunk.metadata.get("tags", []),
                    "chunk_count": 0
                }
            documents[doc_id]["chunk_count"] += 1
        
        # Filter by category or language
        filtered_docs = list(documents.values())
        if category:
            filtered_docs = [doc for doc in filtered_docs if doc["category"] == category]
        if language:
            filtered_docs = [doc for doc in filtered_docs if doc["language"] == language]
        
        # Apply limit
        filtered_docs = filtered_docs[:limit]
        
        return {
            "documents": filtered_docs,
            "total": len(filtered_docs)
        }
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get a specific document by ID."""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        # Find chunks for this document
        doc_chunks = [chunk for chunk in vector_store.chunks if chunk.metadata.get("document_id") == document_id]
        
        if not doc_chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reconstruct document
        document = {
            "id": document_id,
            "title": doc_chunks[0].metadata.get("title", ""),
            "category": doc_chunks[0].metadata.get("category", ""),
            "language": doc_chunks[0].metadata.get("language", ""),
            "tags": doc_chunks[0].metadata.get("tags", []),
            "content": "\n\n".join(chunk.content for chunk in doc_chunks),
            "chunk_count": len(doc_chunks)
        }
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize")
async def summarize_document(request: Dict[str, Any]):
    """Summarize a document using AI."""
    try:
        content = request.get("content")
        language = request.get("language", "en")
        
        if not content:
            raise HTTPException(status_code=400, detail="Content is required")
        
        llm_response = llm_service.summarize_document(content, language)
        
        return {
            "summary": llm_response.get("summary"),
            "llm_available": llm_service.is_available(),
            "error": llm_response.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-keywords")
async def extract_keywords(request: Dict[str, Any]):
    """Extract keywords from document content."""
    try:
        content = request.get("content")
        language = request.get("language", "en")
        
        if not content:
            raise HTTPException(status_code=400, detail="Content is required")
        
        llm_response = llm_service.extract_keywords(content, language)
        
        return {
            "keywords": llm_response.get("keywords", []),
            "llm_available": llm_service.is_available(),
            "error": llm_response.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    ) 