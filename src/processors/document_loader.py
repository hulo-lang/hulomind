"""Document loader for processing Hulo documentation."""

import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .markdown_processor import MarkdownProcessor
from ..models.document import Document, DocumentChunk
from ..config.settings import settings
from ..utils.logger import logger, step, info, success, warning, error, stats, progress, section, debug

logger = logging.getLogger(__name__)
_output_lock = threading.Lock()


class DocumentLoader:
    """Load and process documents from the docs directory."""
    
    def __init__(self):
        self.processor = MarkdownProcessor()
        self.docs_path = Path(settings.docs_path)
    
    def find_markdown_files(self, directory: Path) -> List[Path]:
        """Find all markdown files in the directory."""
        markdown_files = []
        
        if not directory.exists():
            warning(f"Directory does not exist: {directory}")
            return markdown_files
        
        step("Scanning for markdown files")
        for file_path in directory.rglob("*.md"):
            # Skip README files in root directories
            if file_path.name.lower() == "readme.md" and file_path.parent.name in ["src", "zh"]:
                debug(f"Skipping README: {file_path}")
                continue
            markdown_files.append(file_path)

        info(f"Found {len(markdown_files)} markdown files")
        return markdown_files
    
    def process_single_file(self, file_path: Path) -> Optional[Document]:
        """Process a single markdown file."""
        try:
            document = self.processor.process_file(file_path)
            chunks = self.processor.create_chunks(document)
            
            # 显示相对路径来区分 en/zh 文档
            relative_path = file_path.relative_to(self.docs_path)
            with _output_lock:
                success(f"{relative_path} - {len(chunks)} chunks")
            return document
        except Exception as e:
            relative_path = file_path.relative_to(self.docs_path)
            with _output_lock:
                error(f"✗ {relative_path} - {str(e)}")
            return None
    
    def load_documents(self, max_workers: int = 4) -> List[Document]:
        """Load all documents from the docs directory."""
        import time
        start_time = time.time()
        
        section("Document Processing")
        info(f"Loading documents from: {self.docs_path}")
        
        # Find all markdown files
        markdown_files = self.find_markdown_files(self.docs_path)
        
        if not markdown_files:
            warning("No markdown files found")
            return []
        
        step("Processing markdown files")
        documents = []
        processed_count = 0
        error_count = 0
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path 
                for file_path in markdown_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    document = future.result()
                    if document:
                        documents.append(document)
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    relative_path = file_path.relative_to(self.docs_path)
                    error(f"✗ {relative_path} - {str(e)}")
                    error_count += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        success(f"Document processing completed in {processing_time:.2f}s")
        info(f"Processed: {processed_count} files")
        if error_count > 0:
            warning(f"Errors: {error_count} files")
        
        return documents
    
    def load_documents_by_category(self, category: str) -> List[Document]:
        """Load documents from a specific category."""
        category_path = self.docs_path / category
        if not category_path.exists():
            logger.warning(f"Category path does not exist: {category_path}")
            return []
        
        markdown_files = self.find_markdown_files(category_path)
        documents = []
        
        for file_path in markdown_files:
            document = self.process_single_file(file_path)
            if document:
                documents.append(document)
        
        return documents
    
    def load_documents_by_language(self, language: str) -> List[Document]:
        """Load documents in a specific language."""
        if language == "zh":
            # Load from zh subdirectory
            zh_path = self.docs_path / "zh"
            if not zh_path.exists():
                logger.warning(f"Chinese docs path does not exist: {zh_path}")
                return []
            
            markdown_files = self.find_markdown_files(zh_path)
        else:
            # Load from root src directory (excluding zh)
            markdown_files = []
            for file_path in self.docs_path.rglob("*.md"):
                if "zh" not in file_path.parts and file_path.name.lower() != "readme.md":
                    markdown_files.append(file_path)
        
        documents = []
        for file_path in markdown_files:
            document = self.process_single_file(file_path)
            if document and document.language == language:
                documents.append(document)
        
        return documents
    
    def create_all_chunks(self, documents: List[Document]) -> List[DocumentChunk]:
        """Create chunks for all documents."""
        import time
        start_time = time.time()
        
        section("Chunk Creation")
        step("Creating document chunks")
        
        all_chunks = []
        total_chunks = 0
        
        for document in documents:
            try:
                chunks = self.processor.create_chunks(document)
                all_chunks.extend(chunks)
                total_chunks += len(chunks)
                debug(f"Created {len(chunks)} chunks for {document.title}")
            except Exception as e:
                error(f"Error creating chunks for {document.title}: {e}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        success(f"Chunk creation completed in {processing_time:.2f}s")
        info(f"Total chunks: {total_chunks}")
        
        return all_chunks
    
    def get_document_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the loaded documents."""
        step("Calculating document statistics")
        
        stats = {
            "total_documents": len(documents),
            "languages": {},
            "categories": {},
            "total_content_length": 0,
            "average_content_length": 0,
        }
        
        for doc in documents:
            # Language stats
            lang = doc.language
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            
            # Category stats
            cat = doc.category
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
            
            # Content length
            stats["total_content_length"] += len(doc.content)
        
        if documents:
            stats["average_content_length"] = stats["total_content_length"] / len(documents)
        
        # Display statistics
        section("Document Statistics")
        stats_display = {
            "Total Documents": stats["total_documents"],
            "Total Content": f"{stats['total_content_length']:,} chars",
            "Average Content": f"{stats['average_content_length']:.0f} chars",
            "Languages": ", ".join(f"{k} ({v})" for k, v in stats["languages"].items()),
            "Categories": ", ".join(f"{k} ({v})" for k, v in stats["categories"].items()),
        }
        
        for key, value in stats_display.items():
            info(f"{key}: {value}")
        
        return stats 