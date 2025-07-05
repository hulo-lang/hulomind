"""In-memory vector store for temporary storage."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging

from ..models.document import DocumentChunk
from ..config.settings import settings
from .base import VectorStore

logger = logging.getLogger(__name__)


class MemoryVectorStore(VectorStore):
    """In-memory vector store using sentence transformers."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self._chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []
        self.chunk_to_embedding: Dict[str, int] = {}  # chunk_id -> embedding_index
        
    def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks to the vector store."""
        if not chunks:
            return True
        
        def get_embedding_text(chunk: DocumentChunk) -> str:
            """Combine title, tags, category and content for embedding."""
            title = chunk.metadata.get('title', '')
            tags = chunk.metadata.get('tags', [])
            if isinstance(tags, list):
                tags_text = ' '.join(tags)
            else:
                tags_text = str(tags) if tags else ''
            category = chunk.category or ''
            language = chunk.language or ''
            
            # Combine all metadata with content
            combined_text = f"{title} {tags_text} {category} {language} {chunk.content}"
            return combined_text.strip()
            
        # Extract combined text content for embedding
        texts = [get_embedding_text(chunk) for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Store chunks and embeddings
        start_idx = len(self._chunks)
        self._chunks.extend(chunks)
        self.embeddings.extend(embeddings.tolist())
        
        # Update mapping
        for i, chunk in enumerate(chunks):
            self.chunk_to_embedding[chunk.id] = start_idx + i
        
        return True
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        threshold: float = 0.7,
        language: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks."""
        if not self._chunks:
            return []
        
        # Filter chunks by language and category if specified
        filtered_chunks = self._chunks
        if language or category:
            filtered_chunks = []
            for chunk in self._chunks:
                if language and chunk.language != language:
                    continue
                if category and chunk.category != category:
                    continue
                filtered_chunks.append(chunk)
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]
        
        # Calculate similarities for filtered chunks
        similarities = []
        chunk_indices = []
        for i, chunk in enumerate(self._chunks):
            if chunk in filtered_chunks:
                similarity = self._cosine_similarity(query_embedding, self.embeddings[i])
                similarities.append(similarity)
                chunk_indices.append(i)
        
        # Get top-k results
        if similarities:
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity >= threshold:
                    chunk = self._chunks[chunk_indices[idx]]
                    results.append((chunk, similarity))
            
            return results
        
        return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_chunks": len(self._chunks),
            "total_embeddings": len(self.embeddings),
            "embedding_dimension": len(self.embeddings[0]) if self.embeddings else 0,
            "model_name": settings.embedding_model,
            "type": "memory"
        }
    
    def clear(self) -> bool:
        """Clear all data from the vector store."""
        try:
            self._chunks.clear()
            self.embeddings.clear()
            self.chunk_to_embedding.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
    
    @property
    def chunks(self) -> List[DocumentChunk]:
        """Get all chunks in the vector store."""
        return self._chunks 