"""API service for RAGme application."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_pipeline import rag_pipeline
from shared.storage import vector_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Initialize storage
    vector_storage.create_collection()
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="RAGme API",
    description="API for querying the RAG knowledge base",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for queries."""

    query: str
    limit: Optional[int] = 5


class DocumentResult(BaseModel):
    """Model for a single document result."""

    text: str
    score: float
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    metadata: Optional[dict] = None


class QueryResponse(BaseModel):
    """Response model for queries."""

    query: str
    results: List[DocumentResult]
    context: str
    num_results: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "api"}


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base.

    Args:
        request: Query request with search text and optional limit.

    Returns:
        Query response with matched documents and context.
    """
    try:
        # Process query through RAG pipeline
        result = rag_pipeline.query(request.query, limit=request.limit)

        # Format results
        formatted_results = []
        for doc in result["documents"]:
            payload = doc.get("payload", {})
            formatted_results.append(
                DocumentResult(
                    text=payload.get("text", ""),
                    score=doc.get("score", 0),
                    document_id=payload.get("document_id"),
                    chunk_index=payload.get("chunk_index"),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ["text", "document_id", "chunk_index"]
                    },
                )
            )

        return QueryResponse(
            query=result["query"],
            results=formatted_results,
            context=result["context"],
            num_results=result["num_results"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search(q: str, limit: int = 5):
    """Simple search endpoint.

    Args:
        q: Search query string.
        limit: Maximum number of results.

    Returns:
        Search results.
    """
    request = QueryRequest(query=q, limit=limit)
    return await query_knowledge_base(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
