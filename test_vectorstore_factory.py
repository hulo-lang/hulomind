#!/usr/bin/env python3
"""Test script for vector store factory."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vectorstore import VectorStoreFactory
from src.processors.document_loader import DocumentLoader
import time

def test_vectorstore_factory():
    """Test vector store factory with different types."""
    
    print("=== Testing Vector Store Factory ===\n")
    
    # Test available types
    available_types = VectorStoreFactory.get_available_types()
    print(f"Available vector store types: {available_types}")
    
    # Load some test documents
    print("\nLoading test documents...")
    loader = DocumentLoader()
    documents = loader.load_documents(max_workers=1)
    
    if not documents:
        print("No documents found!")
        return
    
    # Create chunks
    chunks = loader.create_all_chunks(documents[:5])  # Just use first 5 docs
    print(f"Created {len(chunks)} chunks for testing")
    
    # Test each vector store type
    for store_type in available_types:
        print(f"\n--- Testing {store_type.upper()} Vector Store ---")
        
        try:
            # Create vector store
            start_time = time.time()
            vector_store = VectorStoreFactory.create(store_type)
            creation_time = time.time() - start_time
            print(f"✓ Created {store_type} vector store in {creation_time:.3f}s")
            
            # Add chunks
            start_time = time.time()
            success = vector_store.add_chunks(chunks)
            add_time = time.time() - start_time
            print(f"✓ Added {len(chunks)} chunks in {add_time:.3f}s (success: {success})")
            
            # Get stats
            stats = vector_store.get_stats()
            print(f"✓ Stats: {stats}")
            
            # Test search
            start_time = time.time()
            results = vector_store.search("function definition", top_k=3)
            search_time = time.time() - start_time
            print(f"✓ Search returned {len(results)} results in {search_time:.3f}s")
            
            # Show top result
            if results:
                top_result = results[0]
                print(f"  Top result: {top_result[0].metadata.get('title', 'Unknown')} (similarity: {top_result[1]:.3f})")
            
            # Test clear
            success = vector_store.clear()
            print(f"✓ Clear operation: {success}")
            
        except Exception as e:
            print(f"✗ Error with {store_type}: {e}")
    
    print("\n=== Factory Test Completed ===")

def test_environment_switching():
    """Test switching between vector stores via environment variable."""
    
    print("\n=== Testing Environment Variable Switching ===\n")
    
    # Test with different environment variables
    test_cases = [
        ("memory", "DEFAULT_VECTOR_STORE=memory"),
        ("chroma", "DEFAULT_VECTOR_STORE=chroma"),
    ]
    
    for expected_type, env_var in test_cases:
        print(f"Testing {env_var}...")
        
        # Set environment variable
        os.environ["DEFAULT_VECTOR_STORE"] = expected_type
        
        try:
            # Create vector store (should use environment variable)
            vector_store = VectorStoreFactory.create()
            stats = vector_store.get_stats()
            actual_type = stats.get("type", "unknown")
            
            if actual_type == expected_type:
                print(f"✓ Correctly created {actual_type} vector store")
            else:
                print(f"✗ Expected {expected_type}, got {actual_type}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Clean up
        if "DEFAULT_VECTOR_STORE" in os.environ:
            del os.environ["DEFAULT_VECTOR_STORE"]

if __name__ == "__main__":
    test_vectorstore_factory()
    test_environment_switching() 