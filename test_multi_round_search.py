#!/usr/bin/env python3
"""Test script for multi-round search functionality."""

import time
from src.processors.document_loader import DocumentLoader
from src.vectorstore.memory_store import MemoryVectorStore
from src.services.knowledge_service import KnowledgeService
from src.utils.logger import section, step, info, success, warning, error

def main():
    """Test multi-round search functionality."""
    
    section("Hulo Knowledge Base - Multi-Round Search Test")
    
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
    
    # Initialize knowledge service
    knowledge_service = KnowledgeService(vector_store)
    
    # Test queries that were problematic before
    test_queries = [
        "What is the difference between let and var?",
        "let var const scope",
        "How to define a function?",
        "What are the basic data types?",
        "How to create a package?",
        "What are traits in Hulo?",
        "How to handle errors?",
        "What is the syntax for loops?",
    ]
    
    section("Multi-Round Search Results")
    
    for query in test_queries:
        step(f"Testing query: {query}")
        
        # Multi-round search
        results, context = knowledge_service.search_with_context(query, max_results=3)
        
        if results:
            success(f"Found {len(results)} results")
            
            # Show results summary
            for i, result in enumerate(results, 1):
                round_type = "refined" if result.round == 2 else "broad"
                info(f"  {i}. {result.chunk.metadata['title']} (similarity: {result.similarity:.3f}, {round_type})")
                info(f"     Path: {result.chunk.metadata['file_path']}")
            
            # Show context preview
            info("Context preview:")
            context_preview = context[:300] + "..." if len(context) > 300 else context
            print(f"  {context_preview}")
            print()
        else:
            warning("No results found")
        print()

if __name__ == "__main__":
    main() 