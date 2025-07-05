#!/usr/bin/env python3
"""Test script for complete RAG Q&A functionality."""

import time
import os
from typing import Dict, Any
from src.processors.document_loader import DocumentLoader
from src.vectorstore.memory_store import MemoryVectorStore
from src.services.knowledge_service import KnowledgeService
from src.services.llm_service import LLMService, LocalLLMService, OpenAIService, QwenService
from src.utils.logger import section, step, info, success, warning, error

def create_llm_service() -> LLMService:
    """Create LLM service based on available options."""
    
    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        info("Using OpenAI service")
        return LLMService(OpenAIService(openai_key, "gpt-4o-mini"))
    
    # Try Qwen API
    qwen_key = os.getenv("DASHSCOPE_API_KEY")
    if qwen_key:
        info("Using Qwen API service")
        return LLMService(QwenService(qwen_key, "qwen-plus"))
    
    # Try local Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            info("Using local Ollama service")
            return LLMService(LocalLLMService("qwen2.5:7b"))
    except:
        pass
    
    # Fallback to mock service
    warning("No LLM service available, using mock responses")
    return MockLLMService()


class MockLLMProvider:
    """Mock LLM provider for testing without real LLM."""
    
    def generate(self, prompt: str, context: str, query: str) -> str:
        """Generate mock response."""
        return f"""Based on the provided context, here's what I found about "{query}":

{context[:200]}...

This is a mock response. To get real answers, please configure one of the following:
1. Set OPENAI_API_KEY environment variable for OpenAI
2. Set DASHSCOPE_API_KEY environment variable for Qwen API  
3. Install and run Ollama locally with qwen2.5:7b model"""
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Mock",
            "model": "mock-model",
            "type": "mock"
        }


class MockLLMService(LLMService):
    """Mock LLM service for testing without real LLM."""
    
    def __init__(self):
        super().__init__(MockLLMProvider())


def main():
    """Test complete RAG Q&A functionality."""
    
    section("Hulo Knowledge Base - Complete RAG Q&A Test")
    
    # Load documents and create chunks
    step("Loading documents and creating chunks")
    loader = DocumentLoader()
    documents = loader.load_documents(max_workers=2)
    
    if not documents:
        error("No documents found")
        return
    
    chunks = loader.create_all_chunks(documents)
    success(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Initialize vector store and knowledge service
    step("Initializing vector store and knowledge service")
    vector_store = MemoryVectorStore()
    
    # Add chunks to vector store
    step("Adding chunks to vector store")
    start_time = time.time()
    vector_store.add_chunks(chunks)
    end_time = time.time()
    
    success(f"Vectorization completed in {end_time - start_time:.2f}s")
    
    # Initialize services
    knowledge_service = KnowledgeService(vector_store)
    llm_service = create_llm_service()
    
    # Show LLM provider info
    provider_info = llm_service.get_provider_info()
    info(f"LLM Provider: {provider_info['provider']} ({provider_info['model']})")
    
    # Test queries
    test_queries = [
        "What is the difference between let and var in Hulo?",
        "How do I define a function in Hulo?",
        "What are the basic data types in Hulo?",
        "How do I create a package in Hulo?",
        "What is the syntax for loops in Hulo?",
    ]
    
    section("RAG Q&A Results")
    
    for query in test_queries:
        step(f"Processing query: {query}")
        
        # Multi-round search
        results, context = knowledge_service.search_with_context(query, max_results=3)
        
        if results:
            success(f"Found {len(results)} relevant documents")
            
            # Generate answer
            answer = llm_service.answer_question(context, query)
            
            # Display results
            info("Retrieved documents:")
            for i, result in enumerate(results, 1):
                round_type = "refined" if result.round == 2 else "broad"
                info(f"  {i}. {result.chunk.metadata['title']} (similarity: {result.similarity:.3f}, {round_type})")
            
            info("Generated answer:")
            print(f"  {answer}")
            print()
        else:
            warning("No relevant documents found")
        print()

if __name__ == "__main__":
    main() 