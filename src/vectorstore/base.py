"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..models.document import DocumentChunk
from ..models.query import SearchResult


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        threshold: float = 0.7,
        language: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            language: Filter by language (optional)
            category: Filter by category (optional)
            
        Returns:
            List of tuples containing (chunk, similarity_score)
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dictionary containing statistics
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all data from the vector store.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def chunks(self) -> List[DocumentChunk]:
        """Get all chunks in the vector store.
        
        Returns:
            List of all document chunks
        """
        pass 