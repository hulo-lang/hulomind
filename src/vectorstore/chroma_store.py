"""ChromaDB vector store for document embeddings."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np

from ..models.document import DocumentChunk
from ..models.query import SearchResult
from ..config.settings import settings
from .base import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(self):
        self.persist_directory = Path(settings.chroma_persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Collection names
        self.collection_name = "hulo_documents"
        self.collection = None
        
        # Initialize collection
        self._init_collection()
    
    def _init_collection(self):
        """Initialize the ChromaDB collection."""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Hulo documentation chunks"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text."""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store."""
        if not chunks:
            return True
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Generate embedding
                embedding = self._get_embedding(chunk.content)
                
                ids.append(chunk.id)
                embeddings.append(embedding)
                documents.append(chunk.content)
                metadatas.append({
                    "document_id": chunk.document_id,
                    "language": chunk.language,
                    "category": chunk.category,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.metadata.get("title", ""),
                    "tags": ",".join(chunk.metadata.get("tags", [])),
                    "file_path": chunk.metadata.get("file_path", "")
                })
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        threshold: float = 0.7,
        language: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = {}
            if language:
                where_clause["language"] = language
            if category:
                where_clause["category"] = category
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    metadata = results["metadatas"][0][i]
                    
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    if similarity_score >= threshold:
                        # Create DocumentChunk from metadata
                        chunk = DocumentChunk(
                            id=chunk_id,
                            document_id=metadata["document_id"],
                            content=results["documents"][0][i],
                            chunk_index=metadata["chunk_index"],
                            language=metadata["language"],
                            category=metadata["category"],
                            metadata={
                                "title": metadata["title"],
                                "tags": metadata["tags"].split(",") if metadata["tags"] else [],
                                "file_path": metadata["file_path"]
                            }
                        )
                        search_results.append((chunk, similarity_score))
            
            logger.info(f"Found {len(search_results)} results for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def clear(self) -> bool:
        """Clear all data from the vector store."""
        try:
            self.collection.delete(where={})
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
    
    @property
    def chunks(self) -> List[DocumentChunk]:
        """Get all chunks in the vector store."""
        try:
            # Get all documents from collection
            results = self.collection.get()
            chunks = []
            
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    chunk = DocumentChunk(
                        id=chunk_id,
                        document_id=results["metadatas"][i]["document_id"],
                        content=results["documents"][i],
                        chunk_index=results["metadatas"][i]["chunk_index"],
                        language=results["metadatas"][i]["language"],
                        category=results["metadatas"][i]["category"],
                        metadata={
                            "title": results["metadatas"][i]["title"],
                            "tags": results["metadatas"][i]["tags"].split(",") if results["metadatas"][i]["tags"] else [],
                            "file_path": results["metadatas"][i]["file_path"]
                        }
                    )
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.query(
                query_embeddings=[[0.0] * 384],  # Dummy embedding
                n_results=min(100, count)
            )
            
            # Analyze languages and categories
            languages = {}
            categories = {}
            
            if sample_results["metadatas"] and sample_results["metadatas"][0]:
                for metadata in sample_results["metadatas"][0]:
                    lang = metadata.get("language", "unknown")
                    cat = metadata.get("category", "unknown")
                    
                    languages[lang] = languages.get(lang, 0) + 1
                    categories[cat] = categories.get(cat, 0) + 1
            
            return {
                "total_chunks": count,
                "languages": languages,
                "categories": categories,
                "embedding_model": settings.embedding_model,
                "type": "chroma"
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete and recreate)."""
        try:
            self.delete_collection()
            self._init_collection()
            logger.info("Collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False 