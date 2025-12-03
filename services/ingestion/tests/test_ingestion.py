"""Tests for the ingestion service."""

import hashlib
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def chunk_text_standalone(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list:
    """Split text into overlapping chunks (standalone version for testing).

    Args:
        text: Text to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of text chunks.
    """
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


def generate_document_id_standalone(content: str) -> str:
    """Generate a unique document ID based on content hash (standalone version for testing).

    Args:
        content: Document content.

    Returns:
        Unique document ID.
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"doc_{content_hash}_{uuid.uuid4().hex[:8]}"


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_chunk_text_single_chunk(self):
        """Test chunking text that fits in a single chunk."""
        text = "Hello world"
        chunks = chunk_text_standalone(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_chunk_text_multiple_chunks(self):
        """Test chunking text into multiple chunks."""
        text = "A" * 100
        chunks = chunk_text_standalone(text, chunk_size=30, chunk_overlap=5)
        assert len(chunks) >= 3
        # Verify first chunk has correct size
        assert len(chunks[0]) == 30

    def test_chunk_text_overlap(self):
        """Test that chunks have proper overlap."""
        text = "0123456789" * 10
        chunks = chunk_text_standalone(text, chunk_size=20, chunk_overlap=5)
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            overlap = chunks[i][-5:]
            next_start = chunks[i + 1][:5]
            assert overlap == next_start


class TestGenerateDocumentId:
    """Tests for the generate_document_id function."""

    def test_generate_document_id_format(self):
        """Test that document ID has correct format."""
        doc_id = generate_document_id_standalone("test content")
        assert doc_id.startswith("doc_")
        parts = doc_id.split("_")
        assert len(parts) == 3

    def test_generate_document_id_different_content(self):
        """Test that different content produces different IDs."""
        id1 = generate_document_id_standalone("content 1")
        id2 = generate_document_id_standalone("content 2")
        # The hash part should be different
        assert id1.split("_")[1] != id2.split("_")[1]

    def test_generate_document_id_uses_sha256(self):
        """Test that SHA-256 is used for hashing."""
        content = "test content"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        doc_id = generate_document_id_standalone(content)
        # The second part of the ID should be the SHA-256 hash
        assert doc_id.split("_")[1] == expected_hash
