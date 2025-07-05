#!/usr/bin/env python3
"""Test script for Hulo Knowledge Base API."""

import requests
import json

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health")
        print("=== Health Check ===")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Health check failed: {e}")

def test_ask_question(question):
    """Test ask endpoint."""
    try:
        response = requests.post(
            "http://localhost:8000/api/ask",
            json={"query": question}
        )
        print(f"=== Question: {question} ===")
        result = response.json()
        
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Total results: {result.get('total_results', 0)}")
        
        # Show top search results
        search_results = result.get('search_results', [])
        if search_results:
            print("\nTop search results:")
            for i, result in enumerate(search_results[:3], 1):
                print(f"{i}. {result.get('title', 'Unknown')} (similarity: {result.get('similarity', 0):.3f})")
                print(f"   Round: {result.get('round', 'Unknown')}")
        
        print()
        return result
    except Exception as e:
        print(f"Ask question failed: {e}")
        return None

def test_search(query):
    """Test search endpoint."""
    try:
        response = requests.post(
            "http://localhost:8000/api/search",
            json={"query": query}
        )
        print(f"=== Search: {query} ===")
        result = response.json()
        
        print(f"Total results: {result.get('total_results', 0)}")
        
        # Show top search results
        search_results = result.get('results', [])
        if search_results:
            print("\nTop search results:")
            for i, result in enumerate(search_results[:3], 1):
                print(f"{i}. {result.get('title', 'Unknown')} (similarity: {result.get('similarity', 0):.3f})")
                print(f"   Content: {result.get('content', '')[:100]}...")
                print(f"   Round: {result.get('round', 'Unknown')}")
        
        print()
        return result
    except Exception as e:
        print(f"Search failed: {e}")
        return None

def main():
    """Main test function."""
    print("Testing Hulo Knowledge Base API...\n")
    
    # Test health
    test_health()
    
    # Test questions
    questions = [
        "What is the difference between let and var in Hulo?",
        "How do I create a package in Hulo?",
        "What are the basic data types in Hulo?",
        "How do I define a function in Hulo?"
    ]
    
    for question in questions:
        test_ask_question(question)
    
    # Test search
    test_search("let var scope")

if __name__ == "__main__":
    main() 