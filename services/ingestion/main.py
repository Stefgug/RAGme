"""Ingestion service for processing and storing documents."""

import hashlib
import sys
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import config
from shared.embeddings import embedding_model
from shared.storage import vector_storage

app = FastAPI(
    title="RAGme Ingestion Service",
    description="Service for ingesting and processing documents",
    version="1.0.0",
)


class Document(BaseModel):
    """Document model for ingestion."""

    content: str
    metadata: Optional[dict] = None


class IngestionResponse(BaseModel):
    """Response model for ingestion operations."""

    status: str
    document_id: str
    chunks_created: int


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to split.
        chunk_size: Maximum size of each chunk. Defaults to config.chunk_size.
        chunk_overlap: Overlap between chunks. Defaults to config.chunk_overlap.

    Returns:
        List of text chunks.
    """
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

        if start >= len(text):
            break

    return chunks


def generate_document_id(content: str) -> str:
    """Generate a unique document ID based on content hash.

    Args:
        content: Document content.

    Returns:
        Unique document ID.
    """
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"doc_{content_hash}_{uuid.uuid4().hex[:8]}"


@app.on_event("startup")
async def startup_event():
    """Initialize storage on startup."""
    vector_storage.create_collection()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ingestion"}


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(document: Document):
    """Ingest a document into the vector store.

    Args:
        document: Document to ingest.

    Returns:
        Ingestion response with status and metadata.
    """
    try:
        # Generate document ID
        doc_id = generate_document_id(document.content)

        # Chunk the document
        chunks = chunk_text(document.content)

        # Generate embeddings for chunks
        embeddings = embedding_model.encode(chunks)

        # Prepare payloads
        payloads = [
            {
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                **(document.metadata or {}),
            }
            for i, chunk in enumerate(chunks)
        ]

        # Generate unique IDs for each chunk
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        # Store in vector database
        vector_storage.upsert(chunk_ids, embeddings, payloads)

        return IngestionResponse(
            status="success", document_id=doc_id, chunks_created=len(chunks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestionResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a file into the vector store.

    Args:
        file: Uploaded file to ingest.

    Returns:
        Ingestion response with status and metadata.
    """
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode("utf-8")

        # Create document with file metadata
        document = Document(
            content=text_content,
            metadata={"filename": file.filename, "content_type": file.content_type},
        )

        return await ingest_document(document)

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="File must be a valid text file"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
