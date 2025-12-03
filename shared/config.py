"""Configuration settings for the RAGme application."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Application configuration."""

    # Vector database settings
    vector_db_host: str = field(
        default_factory=lambda: os.getenv("VECTOR_DB_HOST", "localhost")
    )
    vector_db_port: int = field(
        default_factory=lambda: int(os.getenv("VECTOR_DB_PORT", "6333"))
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "documents")
    )

    # Embedding settings
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "384"))
    )

    # API settings
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # Ingestion settings
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )


# Global configuration instance
config = Config()
