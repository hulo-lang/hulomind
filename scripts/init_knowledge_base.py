#!/usr/bin/env python3
"""Initialize the Hulo Knowledge Base."""

import sys
import os
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Change to src directory for relative imports
os.chdir(src_path)

from services.knowledge_service import KnowledgeService
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Initialize the knowledge base."""
    logger.info("Initializing Hulo Knowledge Base...")
    
    # Check if docs path exists
    docs_path = Path(settings.docs_path)
    if not docs_path.exists():
        logger.error(f"Docs path does not exist: {docs_path}")
        logger.info("Please make sure the docs submodule is properly initialized:")
        logger.info("git submodule update --init --recursive")
        return 1
    
    # Initialize knowledge service
    knowledge_service = KnowledgeService()
    
    # Load and index documents
    logger.info("Loading and indexing documents...")
    result = knowledge_service.load_and_index_documents(force_reload=True)
    
    if result["success"]:
        logger.info("✅ Knowledge base initialized successfully!")
        logger.info(f"📚 Loaded {result['documents_loaded']} documents")
        logger.info(f"🔍 Created {result['chunks_created']} chunks")
        logger.info(f"⏱️  Processing time: {result['processing_time']:.2f}s")
        
        # Print statistics
        if "document_stats" in result:
            stats = result["document_stats"]
            logger.info("\n📊 Document Statistics:")
            logger.info(f"   Languages: {stats.get('languages', {})}")
            logger.info(f"   Categories: {stats.get('categories', {})}")
            logger.info(f"   Total content length: {stats.get('total_content_length', 0):,} characters")
            logger.info(f"   Average content length: {stats.get('average_content_length', 0):.0f} characters")
        
        if "vector_stats" in result:
            vector_stats = result["vector_stats"]
            logger.info("\n🔢 Vector Store Statistics:")
            logger.info(f"   Total chunks: {vector_stats.get('total_chunks', 0)}")
            logger.info(f"   Embedding model: {vector_stats.get('embedding_model', 'unknown')}")
        
        logger.info("\n🚀 You can now start the server with:")
        logger.info("   uv run uvicorn src.main:app --reload")
        
        return 0
    else:
        logger.error("❌ Failed to initialize knowledge base")
        logger.error(f"Error: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 