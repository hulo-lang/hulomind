"""MCP Server for Hulo Knowledge Base."""

import asyncio
import logging
from typing import Dict, Any, Optional

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ReadResourceRequest,
    ReadResourceResult,
    ListResourcesRequest,
    ListResourcesResult,
)

from ..services.knowledge_service import KnowledgeService
from ..services.llm_service import LLMService
from ..vectorstore.memory_store import MemoryVectorStore
from ..processors.document_loader import DocumentLoader

logger = logging.getLogger(__name__)


class HuloKnowledgeMCPServer:
    """MCP Server for Hulo Knowledge Base."""
    
    def __init__(self):
        self.server = Server("hulo-knowledge")
        self.knowledge_service: Optional[KnowledgeService] = None
        self.llm_service: Optional[LLMService] = None
        self.vector_store: Optional[MemoryVectorStore] = None
        
        # Register tools
        self.server.list_tools()(self.list_tools)
        self.server.call_tool()(self.call_tool)
        self.server.list_resources()(self.list_resources)
        self.server.read_resource()(self.read_resource)
        
        # Initialize services
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize knowledge base services."""
        try:
            logger.info("Initializing Hulo Knowledge Base services...")
            
            # Load documents and create chunks
            loader = DocumentLoader()
            documents = loader.load_documents(max_workers=2)
            
            if documents:
                chunks = loader.create_all_chunks(documents)
                logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
                
                # Initialize vector store
                self.vector_store = MemoryVectorStore()
                self.vector_store.add_chunks(chunks)
                
                # Initialize knowledge service
                self.knowledge_service = KnowledgeService(self.vector_store)
                
                # Initialize LLM service (try local Ollama first)
                self.llm_service = self._create_llm_service()
                
                logger.info("Hulo Knowledge Base services initialized successfully!")
            else:
                logger.warning("No documents found")
                
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
    
    def _create_llm_service(self) -> LLMService:
        """Create LLM service based on available options."""
        import os
        import requests
        
        # Try local Ollama first
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info("Using local Ollama service")
                from ..services.llm_service import LocalLLMService
                return LLMService(LocalLLMService("qwen2.5:7b"))
        except:
            pass
        
        # Try OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            logger.info("Using OpenAI service")
            from ..services.llm_service import OpenAIService
            return LLMService(OpenAIService(openai_key, "gpt-4o-mini"))
        
        # Try Qwen API
        qwen_key = os.getenv("DASHSCOPE_API_KEY")
        if qwen_key:
            logger.info("Using Qwen API service")
            from ..services.llm_service import QwenService
            return LLMService(QwenService(qwen_key, "qwen-plus"))
        
        # Fallback to mock service
        logger.warning("No LLM service available, using mock responses")
        from ..services.llm_service import MockLLMProvider
        return LLMService(MockLLMProvider())
    
    async def list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """List available tools."""
        tools = [
            Tool(
                name="search_documents",
                description="Search Hulo documentation using semantic search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="ask_question",
                description="Ask a question about Hulo programming language and get AI-powered answer",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to ask about Hulo"
                        }
                    },
                    "required": ["question"]
                }
            ),
            Tool(
                name="get_document_info",
                description="Get information about a specific document",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Document ID to retrieve"
                        }
                    },
                    "required": ["document_id"]
                }
            ),
            Tool(
                name="list_documents",
                description="List available documents with optional filtering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Filter by category (e.g., 'grammar', 'guide', 'libs')"
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by language ('en' or 'zh')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return",
                            "default": 20
                        }
                    }
                }
            ),
            Tool(
                name="get_statistics",
                description="Get knowledge base statistics",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]
        
        return ListToolsResult(tools=tools)
    
    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Call a specific tool."""
        try:
            if request.name == "search_documents":
                return await self._search_documents(request.arguments)
            elif request.name == "ask_question":
                return await self._ask_question(request.arguments)
            elif request.name == "get_document_info":
                return await self._get_document_info(request.arguments)
            elif request.name == "list_documents":
                return await self._list_documents(request.arguments)
            elif request.name == "get_statistics":
                return await self._get_statistics(request.arguments)
            else:
                raise ValueError(f"Unknown tool: {request.name}")
                
        except Exception as e:
            logger.error(f"Error calling tool {request.name}: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")]
            )
    
    async def _search_documents(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Search documents using semantic search."""
        if not self.knowledge_service:
            return CallToolResult(
                content=[TextContent(type="text", text="Knowledge service not initialized")]
            )
        
        query = arguments.get("query")
        max_results = arguments.get("max_results", 5)
        
        if not query:
            return CallToolResult(
                content=[TextContent(type="text", text="Query is required")]
            )
        
        # Multi-round search
        results, context = self.knowledge_service.search_with_context(query, max_results=max_results)
        
        # Format results
        result_text = f"Search results for: {query}\n\n"
        for i, result in enumerate(results, 1):
            title = result.chunk.metadata.get("title", "Unknown")
            similarity = result.similarity
            round_num = result.round
            result_text += f"{i}. {title} (similarity: {similarity:.3f}, round: {round_num})\n"
            result_text += f"   Content: {result.chunk.content[:200]}...\n\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)]
        )
    
    async def _ask_question(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Ask a question and get AI-powered answer."""
        if not self.knowledge_service or not self.llm_service:
            return CallToolResult(
                content=[TextContent(type="text", text="Services not initialized")]
            )
        
        question = arguments.get("question")
        if not question:
            return CallToolResult(
                content=[TextContent(type="text", text="Question is required")]
            )
        
        # Multi-round search
        results, context = self.knowledge_service.search_with_context(question, max_results=3)
        
        # Generate answer using LLM
        answer = self.llm_service.answer_question(context, question)
        
        # Format response
        response_text = f"Question: {question}\n\n"
        response_text += f"Answer: {answer}\n\n"
        response_text += f"Sources ({len(results)} documents):\n"
        
        for i, result in enumerate(results, 1):
            title = result.chunk.metadata.get("title", "Unknown")
            similarity = result.similarity
            response_text += f"{i}. {title} (similarity: {similarity:.3f})\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=response_text)]
        )
    
    async def _get_document_info(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Get information about a specific document."""
        if not self.vector_store:
            return CallToolResult(
                content=[TextContent(type="text", text="Vector store not initialized")]
            )
        
        document_id = arguments.get("document_id")
        if not document_id:
            return CallToolResult(
                content=[TextContent(type="text", text="Document ID is required")]
            )
        
        # Find chunks for this document
        doc_chunks = [chunk for chunk in self.vector_store.chunks if chunk.metadata.get("document_id") == document_id]
        
        if not doc_chunks:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Document not found: {document_id}")]
            )
        
        # Get document info
        first_chunk = doc_chunks[0]
        info_text = f"Document: {document_id}\n"
        info_text += f"Title: {first_chunk.metadata.get('title', 'Unknown')}\n"
        info_text += f"Category: {first_chunk.metadata.get('category', 'Unknown')}\n"
        info_text += f"Language: {first_chunk.metadata.get('language', 'Unknown')}\n"
        info_text += f"Tags: {', '.join(first_chunk.metadata.get('tags', []))}\n"
        info_text += f"Chunks: {len(doc_chunks)}\n"
        info_text += f"Total Content Length: {sum(len(chunk.content) for chunk in doc_chunks)} characters\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=info_text)]
        )
    
    async def _list_documents(self, arguments: Dict[str, Any]) -> CallToolResult:
        """List available documents."""
        if not self.vector_store:
            return CallToolResult(
                content=[TextContent(type="text", text="Vector store not initialized")]
            )
        
        category = arguments.get("category")
        language = arguments.get("language")
        limit = arguments.get("limit", 20)
        
        # Get unique documents from chunks
        documents = {}
        for chunk in self.vector_store.chunks:
            doc_id = chunk.metadata.get("document_id")
            if doc_id not in documents:
                documents[doc_id] = {
                    "id": doc_id,
                    "title": chunk.metadata.get("title", ""),
                    "category": chunk.metadata.get("category", ""),
                    "language": chunk.metadata.get("language", ""),
                    "chunk_count": 0
                }
            documents[doc_id]["chunk_count"] += 1
        
        # Filter documents
        filtered_docs = list(documents.values())
        if category:
            filtered_docs = [doc for doc in filtered_docs if doc["category"] == category]
        if language:
            filtered_docs = [doc for doc in filtered_docs if doc["language"] == language]
        
        # Apply limit
        filtered_docs = filtered_docs[:limit]
        
        # Format response
        result_text = f"Documents ({len(filtered_docs)} found):\n\n"
        for i, doc in enumerate(filtered_docs, 1):
            result_text += f"{i}. {doc['title']}\n"
            result_text += f"   ID: {doc['id']}\n"
            result_text += f"   Category: {doc['category']}\n"
            result_text += f"   Language: {doc['language']}\n"
            result_text += f"   Chunks: {doc['chunk_count']}\n\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)]
        )
    
    async def _get_statistics(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Get knowledge base statistics."""
        if not self.vector_store:
            return CallToolResult(
                content=[TextContent(type="text", text="Vector store not initialized")]
            )
        
        # Calculate statistics
        total_chunks = len(self.vector_store.chunks)
        unique_docs = len(set(chunk.metadata.get("document_id") for chunk in self.vector_store.chunks))
        
        languages = {}
        categories = {}
        for chunk in self.vector_store.chunks:
            lang = chunk.metadata.get("language", "unknown")
            cat = chunk.metadata.get("category", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1
        
        # Format response
        stats_text = "Knowledge Base Statistics:\n\n"
        stats_text += f"Total Chunks: {total_chunks}\n"
        stats_text += f"Total Documents: {unique_docs}\n"
        stats_text += f"Average Chunks per Document: {total_chunks / unique_docs:.1f}\n\n"
        
        stats_text += "Languages:\n"
        for lang, count in sorted(languages.items()):
            stats_text += f"  {lang}: {count} chunks\n"
        
        stats_text += "\nCategories:\n"
        for cat, count in sorted(categories.items()):
            stats_text += f"  {cat}: {count} chunks\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=stats_text)]
        )
    
    async def list_resources(self, request: ListResourcesRequest) -> ListResourcesResult:
        """List available resources."""
        # For now, return empty list as we don't have file-based resources
        return ListResourcesResult(resources=[])
    
    async def read_resource(self, request: ReadResourceRequest) -> ReadResourceResult:
        """Read a specific resource."""
        # For now, return error as we don't have file-based resources
        raise ValueError("No resources available")


async def main():
    """Main entry point for MCP server."""
    # Create server instance
    server = HuloKnowledgeMCPServer()
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="hulo-knowledge",
                server_version="0.1.0",
                capabilities=server.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main()) 