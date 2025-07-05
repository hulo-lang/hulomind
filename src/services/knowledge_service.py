"""Knowledge service for document retrieval and Q&A."""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from ..vectorstore.memory_store import MemoryVectorStore
from ..models.document import DocumentChunk
from ..utils.logger import info, success, warning

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with metadata."""
    chunk: DocumentChunk
    similarity: float
    round: int  # 1: broad, 2: refined


class KnowledgeService:
    """Knowledge service for multi-round document retrieval."""
    
    def __init__(self, vector_store: MemoryVectorStore):
        self.vector_store = vector_store
    
    def multi_round_search(
        self, 
        query: str, 
        broad_top_k: int = 20,
        refined_top_k: int = 5,
        broad_threshold: float = 0.3,
        refined_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Multi-round search strategy:
        1. Broad search: Get more candidates with lower threshold
        2. Refined search: Get top results with higher threshold
        """
        info(f"Multi-round search for: {query}")
        
        # Round 1: Broad search
        info("Round 1: Broad search")
        broad_results = self.vector_store.search(
            query, 
            top_k=broad_top_k, 
            threshold=broad_threshold
        )
        
        if not broad_results:
            warning("No results found in broad search")
            return []
        
        success(f"Broad search found {len(broad_results)} candidates")
        
        # Round 2: Refined search
        info("Round 2: Refined search")
        refined_results = self.vector_store.search(
            query, 
            top_k=refined_top_k, 
            threshold=refined_threshold
        )
        
        success(f"Refined search found {len(refined_results)} high-quality results")
        
        # Combine and deduplicate results
        all_results = []
        
        # Add refined results first (higher priority)
        for chunk, similarity in refined_results:
            all_results.append(SearchResult(
                chunk=chunk,
                similarity=similarity,
                round=2
            ))
        
        # Add broad results that aren't already included
        existing_ids = {result.chunk.id for result in all_results}
        for chunk, similarity in broad_results:
            if chunk.id not in existing_ids:
                all_results.append(SearchResult(
                    chunk=chunk,
                    similarity=similarity,
                    round=1
                ))
        
        # Sort by similarity (highest first)
        all_results.sort(key=lambda x: x.similarity, reverse=True)
        
        success(f"Combined search returned {len(all_results)} unique results")
        return all_results
    
    def search_with_context(
        self, 
        query: str, 
        max_results: int = 5
    ) -> Tuple[List[SearchResult], str]:
        """
        Search and build context for LLM.
        Returns search results and formatted context.
        """
        # Multi-round search
        results = self.multi_round_search(query)
        
        if not results:
            return [], ""
        
        # Take top results
        top_results = results[:max_results]
        
        # Build context
        context = self._build_context(top_results)
        
        return top_results, context
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """Build formatted context from search results."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            round_info = "refined" if result.round == 2 else "broad"
            
            context_part = f"""## {i}. {chunk.metadata.get('title', 'Untitled')} (similarity: {result.similarity:.3f}, {round_info})
Source: {chunk.metadata.get('file_path', 'Unknown')}
Language: {chunk.language}
Category: {chunk.category}

{chunk.content}

---"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        stats = self.vector_store.get_stats()
        return {
            "vector_store": stats,
            "search_strategy": "multi-round",
            "description": "Broad search (threshold=0.3) + Refined search (threshold=0.5)"
        } 