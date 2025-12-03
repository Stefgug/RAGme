"""Frontend application for RAGme using Streamlit."""

import os

import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
INGESTION_URL = os.getenv("INGESTION_URL", "http://localhost:8001")

st.set_page_config(
    page_title="RAGme - Knowledge Base",
    page_icon="📚",
    layout="wide",
)


def query_api(query: str, limit: int = 5) -> dict:
    """Query the API service.

    Args:
        query: Search query.
        limit: Maximum number of results.

    Returns:
        API response.
    """
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query, "limit": limit},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def ingest_document(content: str, metadata: dict = None) -> dict:
    """Ingest a document via the ingestion service.

    Args:
        content: Document content.
        metadata: Optional metadata.

    Returns:
        Ingestion response.
    """
    try:
        response = requests.post(
            f"{INGESTION_URL}/ingest",
            json={"content": content, "metadata": metadata or {}},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def main():
    """Main application entry point."""
    st.title("📚 RAGme - Personal Knowledge Base")
    st.markdown("A personal library for technical knowledge and learning resources.")

    # Sidebar for navigation
    page = st.sidebar.selectbox("Navigate", ["Search", "Add Document", "About"])

    if page == "Search":
        search_page()
    elif page == "Add Document":
        add_document_page()
    else:
        about_page()


def search_page():
    """Render the search page."""
    st.header("🔍 Search Knowledge Base")

    # Search input
    query = st.text_input("Enter your search query", placeholder="e.g., machine learning basics")
    limit = st.slider("Number of results", min_value=1, max_value=20, value=5)

    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                results = query_api(query, limit)

            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                st.success(f"Found {results.get('num_results', 0)} results")

                # Display results
                for i, result in enumerate(results.get("results", []), 1):
                    with st.expander(f"Result {i} (Score: {result.get('score', 0):.3f})"):
                        st.markdown(f"**Text:**\n{result.get('text', 'N/A')}")
                        if result.get("document_id"):
                            st.caption(f"Document ID: {result['document_id']}")
                        if result.get("metadata"):
                            st.json(result["metadata"])
        else:
            st.warning("Please enter a search query")


def add_document_page():
    """Render the add document page."""
    st.header("📝 Add Document")

    # Document input
    content = st.text_area(
        "Document Content",
        placeholder="Paste your document content here...",
        height=300,
    )

    # Metadata
    st.subheader("Metadata (optional)")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Title")
    with col2:
        source = st.text_input("Source")

    tags = st.text_input("Tags (comma-separated)", placeholder="python, tutorial, basics")

    if st.button("Add Document", type="primary"):
        if content:
            metadata = {}
            if title:
                metadata["title"] = title
            if source:
                metadata["source"] = source
            if tags:
                metadata["tags"] = [t.strip() for t in tags.split(",")]

            with st.spinner("Ingesting document..."):
                result = ingest_document(content, metadata)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(
                    f"Document ingested successfully!\n"
                    f"Document ID: {result.get('document_id', 'N/A')}\n"
                    f"Chunks created: {result.get('chunks_created', 0)}"
                )
        else:
            st.warning("Please enter document content")


def about_page():
    """Render the about page."""
    st.header("ℹ️ About RAGme")

    st.markdown(
        """
    **RAGme** is a personal knowledge base application that uses 
    Retrieval-Augmented Generation (RAG) to help you store, organize, 
    and search your technical knowledge and learning resources.

    ### Features
    - 📄 **Document Ingestion**: Add documents to your personal knowledge base
    - 🔍 **Semantic Search**: Find relevant information using natural language queries
    - 🧠 **Smart Chunking**: Documents are automatically split for optimal retrieval

    ### Architecture
    - **Frontend**: Streamlit-based web interface
    - **API Service**: FastAPI-based query service with RAG pipeline
    - **Ingestion Service**: Document processing and embedding generation
    - **Vector Database**: Qdrant for efficient similarity search

    ### Technology Stack
    - Python 3.11
    - FastAPI & Streamlit
    - Sentence Transformers for embeddings
    - Qdrant vector database
    - Docker Compose for orchestration
    """
    )


if __name__ == "__main__":
    main()
