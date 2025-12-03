"""Embeddings utilities for the RAGme application."""

from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .config import config


class EmbeddingModel:
    """Wrapper for embedding model operations."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to config.embedding_model.
        """
        self.model_name = model_name or config.embedding_model
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def encode_single(self, text: str) -> List[float]:
        """Encode a single text into an embedding.

        Args:
            text: Text string to encode.

        Returns:
            Embedding vector.
        """
        return self.encode([text])[0]


# Global embedding model instance
embedding_model = EmbeddingModel()
