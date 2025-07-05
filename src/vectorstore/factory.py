"""Factory for creating vector stores."""

from typing import Optional
from .base import VectorStore
from .memory_store import MemoryVectorStore
from .chroma_store import ChromaVectorStore
from ..config.settings import settings


class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    @staticmethod
    def create(store_type: Optional[str] = None) -> VectorStore:
        """Create a vector store instance.
        
        Args:
            store_type: Type of vector store ("memory" or "chroma"). 
                       If None, uses settings.default_vector_store
        
        Returns:
            VectorStore instance
        """
        if store_type is None:
            store_type = getattr(settings, 'default_vector_store', 'memory')
        
        store_type = store_type.lower()
        
        if store_type == 'memory':
            return MemoryVectorStore()
        elif store_type == 'chroma':
            return ChromaVectorStore()
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
    
    @staticmethod
    def get_available_types() -> list[str]:
        """Get list of available vector store types."""
        return ['memory', 'chroma'] 