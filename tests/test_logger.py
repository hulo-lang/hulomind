#!/usr/bin/env python3
"""Test script for goreleaser-style logger."""

import time
from src.utils.logger import (
    info, success, warning, error, debug, step, 
    section, stats, progress, status
)

def main():
    """Demo the logger functionality."""
    
    section("Hulo Knowledge Base - Document Processing")
    
    step("Initializing document loader")
    info("Found 150 markdown files in docs directory")
    
    with status("Processing documents..."):
        time.sleep(2)
    
    step("Loading documents")
    with progress("Loading markdown files", total=150) as progress_bar:
        task = progress_bar.add_task("Loading markdown files", total=150)
        for i in range(150):
            time.sleep(0.01)
            progress_bar.update(task, advance=1)
    
    success("Successfully loaded 150 documents")
    
    step("Creating document chunks")
    with progress("Creating chunks", total=150) as progress_bar:
        task = progress_bar.add_task("Creating chunks", total=150)
        for i in range(150):
            time.sleep(0.005)
            progress_bar.update(task, advance=1)
    
    success("Created 2,847 document chunks")
    
    step("Building vector index")
    with status("Building ChromaDB index..."):
        time.sleep(1.5)
    
    success("Vector index built successfully")
    
    # Show statistics
    stats({
        "Documents processed": 150,
        "Chunks created": 2847,
        "Average chunk size": 456,
        "Processing time": 3.2,
        "Languages": "en, zh",
        "Categories": "grammar, blueprints, libs, guide"
    })
    
    section("Knowledge Base Ready")
    info("Server starting on http://localhost:8000")
    info("MCP server on ws://localhost:8001")

if __name__ == "__main__":
    main() 