#!/usr/bin/env python3
"""Test script for document processing with rich logging."""

import time
from src.processors.document_loader import DocumentLoader
from src.utils.logger import section, success, info

def main():
    """Test document processing with detailed logging."""
    
    section("Hulo Knowledge Base - Document Processing")
    
    # Initialize loader
    loader = DocumentLoader()
    
    # Load documents
    start_time = time.time()
    documents = loader.load_documents(max_workers=2)
    
    if not documents:
        info("No documents found, exiting")
        return
    
    # Create chunks
    chunks = loader.create_all_chunks(documents)
    
    # Get statistics
    stats = loader.get_document_statistics(documents)
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    section("Processing Summary")
    success(f"Total processing time: {total_time:.2f}s")
    info(f"Documents processed: {len(documents)}")
    info(f"Chunks created: {len(chunks)}")
    info(f"Average chunks per document: {len(chunks) / len(documents):.1f}")

if __name__ == "__main__":
    main() 