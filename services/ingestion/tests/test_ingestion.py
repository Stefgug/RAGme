"""Tests for the ingestion service."""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.ingestion.main import chunk_text, generate_document_id


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_chunk_text_single_chunk(self):
        """Test chunking text that fits in a single chunk."""
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_chunk_text_multiple_chunks(self):
        """Test chunking text into multiple chunks."""
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
        assert len(chunks) >= 3
        # Verify first chunk has correct size
        assert len(chunks[0]) == 30

    def test_chunk_text_overlap(self):
        """Test that chunks have proper overlap."""
        text = "0123456789" * 10
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            overlap = chunks[i][-5:]
            next_start = chunks[i + 1][:5]
            assert overlap == next_start


class TestGenerateDocumentId:
    """Tests for the generate_document_id function."""

    def test_generate_document_id_format(self):
        """Test that document ID has correct format."""
        doc_id = generate_document_id("test content")
        assert doc_id.startswith("doc_")
        parts = doc_id.split("_")
        assert len(parts) == 3

    def test_generate_document_id_different_content(self):
        """Test that different content produces different IDs."""
        id1 = generate_document_id("content 1")
        id2 = generate_document_id("content 2")
        # The hash part should be different
        assert id1.split("_")[1] != id2.split("_")[1]
