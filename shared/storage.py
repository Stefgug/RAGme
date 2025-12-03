"""Storage utilities for the RAGme application."""

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import config


class VectorStorage:
    """Vector database storage operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """Initialize the vector storage.

        Args:
            host: Vector database host. Defaults to config.vector_db_host.
            port: Vector database port. Defaults to config.vector_db_port.
            collection_name: Name of the collection. Defaults to config.collection_name.
        """
        self.host = host or config.vector_db_host
        self.port = port or config.vector_db_port
        self.collection_name = collection_name or config.collection_name
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Lazy load the Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    def create_collection(
        self, dimension: Optional[int] = None, recreate: bool = False
    ) -> None:
        """Create a collection in the vector database.

        Args:
            dimension: Vector dimension. Defaults to config.embedding_dimension.
            recreate: If True, delete and recreate the collection.
        """
        dimension = dimension or config.embedding_dimension

        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )
        except Exception:
            # Collection already exists
            pass

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """Upsert vectors into the collection.

        Args:
            ids: List of unique identifiers.
            embeddings: List of embedding vectors.
            payloads: List of metadata payloads.
        """
        points = [
            models.PointStruct(id=i, vector=embedding, payload=payload)
            for i, (embedding, payload) in enumerate(zip(embeddings, payloads))
        ]

        # Use provided IDs for the point IDs
        for idx, point_id in enumerate(ids):
            points[idx].id = hash(point_id) % (2**63)

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_embedding: Query embedding vector.
            limit: Maximum number of results to return.

        Returns:
            List of search results with payloads and scores.
        """
        results = self.client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=limit
        )

        return [
            {"payload": hit.payload, "score": hit.score}
            for hit in results
        ]


# Global storage instance
vector_storage = VectorStorage()
