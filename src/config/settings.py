"""Application settings and configuration."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=False)
    
    # LLM Settings
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-3.5-turbo")
    openai_base_url: Optional[str] = Field(default=None)
    
    # Vector Store Settings
    default_vector_store: str = Field(default="memory")
    chroma_persist_directory: str = Field(default="./data/chroma")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    
    # Document Processing Settings
    docs_path: str = Field(default="../docs/src")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    # MCP Settings
    mcp_host: str = Field(default="localhost")
    mcp_port: int = Field(default=8001)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_prefix="",
        env_file_encoding="utf-8",
        extra="ignore",
        # 映射环境变量到字段
        env_map={
            "api_host": "API_HOST",
            "api_port": "API_PORT",
            "debug": "DEBUG",
            "openai_api_key": "OPENAI_API_KEY",
            "openai_model": "OPENAI_MODEL",
            "openai_base_url": "OPENAI_BASE_URL",
            "default_vector_store": "DEFAULT_VECTOR_STORE",
            "chroma_persist_directory": "CHROMA_PERSIST_DIRECTORY",
            "embedding_model": "EMBEDDING_MODEL",
            "docs_path": "DOCS_PATH",
            "chunk_size": "CHUNK_SIZE",
            "chunk_overlap": "CHUNK_OVERLAP",
            "mcp_host": "MCP_HOST",
            "mcp_port": "MCP_PORT",
        }
    )

# Global settings instance
settings = Settings() 