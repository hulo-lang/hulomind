#!/usr/bin/env python3
"""Test script for vector search and prompt testing."""

import time
from src.processors.document_loader import DocumentLoader
from src.vectorstore.memory_store import MemoryVectorStore
from src.utils.logger import section, step, info, success, warning, error

def main():
    """Test vector search and prompt functionality."""
    
    section("Hulo Knowledge Base - Vector Search Test")
    
    # Load documents and create chunks
    step("Loading documents and creating chunks")
    loader = DocumentLoader()
    documents = loader.load_documents(max_workers=2)
    
    if not documents:
        error("No documents found")
        return
    
    chunks = loader.create_all_chunks(documents)
    success(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Initialize vector store
    step("Initializing vector store")
    vector_store = MemoryVectorStore()
    
    # Add chunks to vector store
    step("Adding chunks to vector store")
    start_time = time.time()
    vector_store.add_chunks(chunks)
    end_time = time.time()
    
    success(f"Vectorization completed in {end_time - start_time:.2f}s")
    
    # Get vector store stats
    stats = vector_store.get_stats()
    info(f"Vector store stats: {stats}")
    
    # Test searches with different thresholds
    test_queries = [
        "let",
        "var", 
        "const",
        "let var difference",
        "scope let var",
        "What is the difference between let and var?",
        "How to define a function in Hulo?",
        "What are the basic data types?",
        "How to use async/await?",
        "How to create a package?",
        "What are traits in Hulo?",
        "How to handle errors?",
        "What is the syntax for loops?",
    ]
    
    section("Vector Search Results")
    
    for query in test_queries:
        step(f"Searching for: {query}")
        
        # Try with lower threshold first
        results = vector_store.search(query, top_k=5, threshold=0.3)
        
        if results:
            success(f"Found {len(results)} relevant chunks")
            for i, (chunk, similarity) in enumerate(results, 1):
                info(f"  {i}. {chunk.metadata['title']} (similarity: {similarity:.3f})")
                info(f"     Path: {chunk.metadata['file_path']}")
                info(f"     Content: {chunk.content[:150]}...")
                print()
        else:
            warning("No relevant chunks found")
        print()

if __name__ == "__main__":
    main() 