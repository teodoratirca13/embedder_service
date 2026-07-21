# Embedder Service

A FastAPI microservice for document embedding, chunking, and vector storage. Part of a RAG (Retrieval-Augmented Generation) platform — ingests PDF documents and makes them searchable via semantic query embeddings.

## Features

- **PDF Ingestion** — Fetch PDFs from MinIO, extract text with PyMuPDF, clean and normalize content
- **Text Chunking** — Split documents into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`
- **Embedding Generation** — Encode text chunks into 1024-dimension vectors using BGE-M3 via `sentence-transformers`
- **Vector Storage** — Store embeddings in Qdrant with metadata (course, week, professor, document title)
- **Re-indexing** — Automatically deletes old chunks before re-indexing the same document
- **Query Embedding** — Embed student queries for semantic search against stored vectors
- **JSON Structured Logging** — All logs emitted as structured JSON for centralized log aggregation
- **Health Checks** — Exposes model and Qdrant connectivity status

## Tech Stack

| Component        | Technology                                |
|------------------|-------------------------------------------|
| Framework        | FastAPI                                   |
| Runtime          | Python 3.12                               |
| Embedding Model  | BAAI/bge-m3 (sentence-transformers)       |
| Vector Database  | Qdrant                                    |
| Object Storage   | MinIO                                     |
| PDF Parsing      | PyMuPDF (fitz)                            |
| Text Splitting   | LangChain `RecursiveCharacterTextSplitter`|
| Config           | pydantic-settings + `.env`                |
| Package Manager  | uv                                        |
| Containerization | Docker / Docker Compose                   |

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — Python package manager
- Docker and Docker Compose (for containerized run)
- Access to a Qdrant instance (or run via Docker Compose)
- Access to a MinIO instance (or run via Docker Compose)

## Installation

### Local Development (Windows)

```powershell
# Clone the repository
git clone <repo-url>
cd embedder-service

# Install dependencies (creates .venv automatically)
uv sync

# Activate virtual environment
.venv\Scripts\activate
```

## Environment Variables

Configuration is managed via `.env` file or environment variables. All variables have sensible defaults for local development.

| Variable             | Default                 | Description                    |
|----------------------|-------------------------|--------------------------------|
| `QDRANT_URL`         | `http://localhost:6333` | Qdrant vector DB URL           |
| `MINIO_ENDPOINT`     | `localhost:9000`        | MinIO server endpoint          |
| `MINIO_ACCESS_KEY`   | `minioadmin`            | MinIO access key               |
| `MINIO_SECRET_KEY`   | `minioadmin`            | MinIO secret key               |
| `MINIO_BUCKET`       | `documents`             | MinIO bucket name              |
| `BGE_MODEL_PATH`     | `BAAI/bge-m3`           | Sentence-transformers model    |
| `LOG_LEVEL`          | `INFO`                  | Logging level                  |

## Running the Project

### With Docker Compose

```powershell
# Start all services (Qdrant, MinIO, Embedder Service)
docker-compose up --build

# The service will be available at http://localhost:8001
# Qdrant at http://localhost:6333
# MinIO console at http://localhost:9001
```

## API Overview

| Method | Path                    | Called By                       | Description                                          |
|--------|-------------------------|---------------------------------|------------------------------------------------------|
| GET    | `/`                     | Anyone                          | Root — returns service info and docs links           |
| GET    | `/api/health`           | Docker, monitoring              | Health check (model + Qdrant connectivity)           |
| POST   | `/api/documents/ingest` | Spring Boot                     | Ingest PDF document (fetch → extract → chunk → embed → store) |
| POST   | `/api/query/embed`      | LLM Response Service (Person C) | Embed a query text into a vector                     |

> ⚠️ **Architecture note:** `/api/query/embed` is called by the **LLM Response Service (Person C)**,
> not by Spring Boot directly. Spring Boot calls Person C's `/api/chat` endpoint,
> which internally calls this endpoint as part of its pipeline.

Interactive API documentation is available at [http://localhost:8001/docs](http://localhost:8001/docs).

### POST /api/documents/ingest

**Called by:** Spring Boot when professor clicks "Indexează document"

**Request body:**
```json
{
  "document_id": 123,
  "course_id": 5,
  "week_id": 15,
  "path_minio": "algorithms/week3/lecture.pdf",
  "document_title": "Lecture 3",
  "professor_id": 7
}
```

**Response (success):**
```json
{
  "document_id": 123,
  "status": "INDEXED",
  "chunks_count": 42,
  "error": null,
  "processing_time_ms": 3500
}
```

**Response (failure — always HTTP 200, never 500):**
```json
{
  "document_id": 123,
  "status": "FAILED",
  "chunks_count": null,
  "error": "MinIO fetch error: ...",
  "processing_time_ms": 150
}
```

The ingestion pipeline:
1. Fetch PDF from MinIO
2. Extract text with PyMuPDF (skips image-only pages)
3. Split text into overlapping chunks (size: 800, overlap: 150)
4. Filter out chunks shorter than 50 characters
5. Encode each chunk into a 1024-dim embedding vector (batch size: 8)
6. Delete any previously stored chunks for this document
7. Upsert all chunks with metadata into Qdrant `course_chunks` collection

### POST /api/query/embed

**Called by:** LLM Response Service (Person C)

**Request body:**
```json
{
  "text": "Explică programarea dinamică"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789],
  "dimension": 1024,
  "model": "BAAI/bge-m3"
}
```

### GET /api/health

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "qdrant_connected": true
}
```

Possible statuses: `"ok"` (both healthy), `"degraded"` (one healthy), `"error"` (none healthy).

## Running Tests

```powershell
uv run pytest tests/ -v
```