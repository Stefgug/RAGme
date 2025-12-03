# RAGme Architecture

## Overview

RAGme is a personal knowledge base application that uses Retrieval-Augmented Generation (RAG) 
to help users store, organize, and search their technical knowledge and learning resources.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Frontend   │    │   API       │    │   Ingestion         │  │
│  │  (Streamlit)│───▶│  (FastAPI)  │    │   (FastAPI)         │  │
│  │   :8501     │    │   :8000     │    │   :8001             │  │
│  └─────────────┘    └──────┬──────┘    └──────────┬──────────┘  │
│                            │                      │             │
│                            ▼                      ▼             │
│                    ┌──────────────────────────────────────┐     │
│                    │         Shared Components            │     │
│                    │  (embeddings, storage, config)       │     │
│                    └──────────────────┬───────────────────┘     │
│                                       │                         │
│                                       ▼                         │
│                            ┌─────────────────────┐              │
│                            │   Vector Database   │              │
│                            │   (Qdrant)          │              │
│                            │   :6333             │              │
│                            └─────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Frontend Service (`services/frontend/`)

The frontend is a Streamlit-based web application that provides:
- **Search Interface**: Natural language query interface for searching the knowledge base
- **Document Ingestion**: Form for adding new documents with metadata
- **About Page**: Information about the application

**Port**: 8501

### API Service (`services/api/`)

The API service is a FastAPI application that handles:
- **Query Processing**: Processes search queries through the RAG pipeline
- **Semantic Search**: Converts queries to embeddings and searches the vector database
- **Response Formatting**: Formats and returns search results

**Port**: 8000

### Ingestion Service (`services/ingestion/`)

The ingestion service is a FastAPI application responsible for:
- **Document Processing**: Receives documents for ingestion
- **Text Chunking**: Splits documents into overlapping chunks
- **Embedding Generation**: Creates vector embeddings for each chunk
- **Storage**: Stores embeddings in the vector database

**Port**: 8001

### Shared Components (`shared/`)

Common utilities shared across services:

- **`config.py`**: Centralized configuration management using environment variables
- **`embeddings.py`**: Wrapper for the sentence-transformers embedding model
- **`storage.py`**: Interface for the Qdrant vector database

### Vector Database (Qdrant)

Qdrant is used as the vector database for storing and searching embeddings:
- **Collection**: Documents are stored in a collection named "documents"
- **Distance Metric**: Cosine similarity
- **Vector Dimension**: 384 (default for MiniLM model)

**Port**: 6333

## Data Flow

### Ingestion Flow

1. User submits a document through the frontend or API
2. Ingestion service receives the document
3. Document is split into overlapping chunks
4. Each chunk is converted to an embedding using sentence-transformers
5. Embeddings are stored in Qdrant with metadata

### Query Flow

1. User enters a search query in the frontend
2. API service receives the query
3. Query is converted to an embedding
4. Vector database is searched for similar embeddings
5. Top matching chunks are retrieved with scores
6. Results are formatted and returned to the user

## Configuration

The application is configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_DB_HOST` | localhost | Qdrant host |
| `VECTOR_DB_PORT` | 6333 | Qdrant port |
| `COLLECTION_NAME` | documents | Qdrant collection name |
| `EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| `EMBEDDING_DIMENSION` | 384 | Embedding vector dimension |
| `CHUNK_SIZE` | 512 | Maximum chunk size |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `API_HOST` | 0.0.0.0 | API service host |
| `API_PORT` | 8000 | API service port |

## Deployment

### Using Docker Compose

```bash
docker-compose up -d
```

This will start all services including:
- Frontend (http://localhost:8501)
- API Service (http://localhost:8000)
- Ingestion Service (http://localhost:8001)
- Qdrant (http://localhost:6333)

### Development

For local development, run each service individually:

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Ingestion Service
cd services/ingestion
pip install -r requirements.txt
python main.py

# Start API Service
cd services/api
pip install -r requirements.txt
python main.py

# Start Frontend
cd services/frontend
pip install -r requirements.txt
streamlit run app.py
```
