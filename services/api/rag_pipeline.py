"""RAG pipeline for query processing and response generation."""

import sys
from pathlib import Path
from typing import List, Optional

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.embeddings import embedding_model
from shared.storage import vector_storage


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self, top_k: int = 5):
        """Initialize the RAG pipeline.

        Args:
            top_k: Number of documents to retrieve.
        """
        self.top_k = top_k

    def retrieve(self, query: str, limit: Optional[int] = None) -> List[dict]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query.
            limit: Maximum number of results. Defaults to self.top_k.

        Returns:
            List of retrieved documents with scores.
        """
        limit = limit or self.top_k

        # Generate query embedding
        query_embedding = embedding_model.encode_single(query)

        # Search for similar documents
        results = vector_storage.search(query_embedding, limit=limit)

        return results

    def generate_context(self, documents: List[dict]) -> str:
        """Generate context string from retrieved documents.

        Args:
            documents: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            payload = doc.get("payload", {})
            text = payload.get("text", "")
            score = doc.get("score", 0)
            context_parts.append(f"[{i}] (Score: {score:.3f})\n{text}")

        return "\n\n".join(context_parts)

    def query(self, query: str, limit: Optional[int] = None) -> dict:
        """Process a query through the RAG pipeline.

        Args:
            query: User query.
            limit: Maximum number of documents to retrieve.

        Returns:
            Dictionary with retrieved documents and context.
        """
        # Retrieve relevant documents
        documents = self.retrieve(query, limit=limit)

        # Generate context
        context = self.generate_context(documents)

        return {
            "query": query,
            "documents": documents,
            "context": context,
            "num_results": len(documents),
        }


# Global pipeline instance
rag_pipeline = RAGPipeline()
