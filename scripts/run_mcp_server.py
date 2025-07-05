#!/usr/bin/env python3
"""Run the MCP server for Hulo Knowledge Base."""

import sys
import os
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Change to src directory for relative imports
os.chdir(src_path)

from mcp.mcp_server import MCPServer
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the MCP server."""
    logger.info("Starting Hulo Knowledge Base MCP Server...")
    
    # Create and run MCP server
    mcp_server = MCPServer()
    
    logger.info(f"🌐 MCP Server will be available at:")
    logger.info(f"   HTTP: http://{settings.mcp_host}:{settings.mcp_port}")
    logger.info(f"   WebSocket: ws://{settings.mcp_host}:{settings.mcp_port}/ws")
    logger.info(f"   API Docs: http://{settings.mcp_host}:{settings.mcp_port}/docs")
    
    try:
        mcp_server.run()
    except KeyboardInterrupt:
        logger.info("MCP Server stopped by user")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 