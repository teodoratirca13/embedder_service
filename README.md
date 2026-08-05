# Embedder Service

A FastAPI microservice for document embedding, chunking, and vector storage. Part of a RAG (Retrieval-Augmented Generation) platform — ingests PDF documents and makes them searchable via semantic query embeddings.

## Features

- **PDF Ingestion** — Fetch PDFs from MinIO, extract text with PyMuPDF, clean and normalize content
- **Header/Footer Deduplication** — Detects headers/footers that repeat across pages (using bounding-box position + fuzzy text matching via `rapidfuzz`) and keeps each one only once, instead of duplicating it into every chunk
- **Text Chunking** — Split documents into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`
- **Embedding Generation** — Encode text chunks into 1024-dimension vectors using BGE-M3 via `sentence-transformers`
- **Vector Storage** — Store embeddings in Qdrant with metadata (course, week, professor, document title)
- **Re-indexing** — Automatically deletes old chunks before re-indexing the same document
- **Chunk Deletion** — Explicit endpoint to delete all stored chunks for a document
- **Query Embedding** — Embed student queries for semantic search against stored vectors
- **Basic Auth** — Protects the ingest, query-embed, and delete endpoints with HTTP Basic Auth
- **Structured Request Logging** — Every request gets a request ID, JSON logs, and automatic masking of sensitive fields (passwords, tokens)
- **Health Checks** — Exposes model and Qdrant connectivity status (unauthenticated, for Docker healthchecks)

## Tech Stack

| Component        | Technology                                |
|------------------|-------------------------------------------|
| Framework        | FastAPI                                   |
| Runtime          | Python 3.12                               |
| Embedding Model  | BAAI/bge-m3 (sentence-transformers)       |
| Vector Database  | Qdrant                                    |
| Object Storage   | MinIO                                     |
| PDF Parsing      | PyMuPDF (fitz) + rapidfuzz (header/footer dedup) |
| Text Splitting   | LangChain `RecursiveCharacterTextSplitter`|
| Config           | pydantic-settings + `.env`                |
| Auth             | HTTP Basic Auth (`fastapi.security`)      |
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

Configuration is managed via `.env` file or environment variables. All variables have sensible defaults for local development, **except the auth credentials, which have no default and must be set**.

| Variable                | Default                 | Description                                              |
|--------------------------|-------------------------|-----------------------------------------------------------|
| `QDRANT_URL`             | `http://localhost:6333` | Qdrant vector DB URL                                       |
| `MINIO_ENDPOINT`         | `localhost:9000`        | MinIO server endpoint                                       |
| `MINIO_ACCESS_KEY`       | `minioadmin`            | MinIO access key                                            |
| `MINIO_SECRET_KEY`       | `minioadmin`            | MinIO secret key                                             |
| `MINIO_BUCKET`           | `documents`             | MinIO bucket name                                            |
| `BGE_MODEL_PATH`         | `BAAI/bge-m3`           | Sentence-transformers model                                  |
| `LOG_LEVEL`              | `INFO`                  | Logging level                                                 |
| `RAG_SERVICE_USERNAME`   | *(none — required)*     | Basic Auth username for protected endpoints                   |
| `RAG_SERVICE_PASSWORD`   | *(none — required)*     | Basic Auth password for protected endpoints                    |

> ⚠️ If `RAG_SERVICE_USERNAME` / `RAG_SERVICE_PASSWORD` are not set, any request to a protected endpoint returns **HTTP 500** ("Autentificarea nu este configurată pe server"), not a silent bypass.

## Running the Project

### With Docker Compose

```powershell
# Start all services (Qdrant, MinIO, Embedder Service)
docker-compose up --build

# The service will be available at http://localhost:8001
# Qdrant at http://localhost:6333
# MinIO console at http://localhost:9001
```

## Authentication

The following endpoints require **HTTP Basic Auth**:

- `POST /api/documents/ingest`
- `DELETE /api/documents/{document_id}`
- `POST /api/query/embed`

`GET /api/health` and `GET /` are **not** authenticated, so Docker/monitoring healthchecks (see `docker-compose.yml`) keep working without credentials.

Credentials are compared with `secrets.compare_digest` (constant-time) against `RAG_SERVICE_USERNAME` / `RAG_SERVICE_PASSWORD`, read from the environment via `fastapi_app/auth.py`.

Example authenticated request:

```bash
curl -X POST http://localhost:8001/api/query/embed \
  -u your_username:your_password \
  -H "Content-Type: application/json" \
  -d '{"text": "Explică programarea dinamică"}'
```

## API Overview

| Method | Path                                | Auth required | Called By                       | Description                                                    |
|--------|--------------------------------------|:--------------:|----------------------------------|------------------------------------------------------------------|
| GET    | `/`                                   | No             | Anyone                          | Root — returns service info and docs links                       |
| GET    | `/api/health`                         | No             | Docker, monitoring              | Health check (model + Qdrant connectivity)                        |
| POST   | `/api/documents/ingest`               | Yes            | Spring Boot                     | Ingest PDF document (fetch → extract → chunk → embed → store)     |
| DELETE | `/api/documents/{document_id}`        | Yes            | Spring Boot                     | Delete all stored chunks for a document from Qdrant                |
| POST   | `/api/query/embed`                    | Yes            | LLM Response Service (Person C) | Embed a query text into a vector                                   |

> ⚠️ **Architecture note:** `/api/query/embed` is called by the **LLM Response Service (Person C)**,
> not by Spring Boot directly. Spring Boot calls Person C's `/api/chat` endpoint,
> which internally calls this endpoint as part of its pipeline.

Interactive API documentation is available at [http://localhost:8001/docs](http://localhost:8001/docs) (note: the Basic Auth prompt will appear when trying protected endpoints from the Swagger UI).

### POST /api/documents/ingest

**Called by:** Spring Boot when professor clicks "Indexează document"
**Auth:** HTTP Basic Auth required

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
3. Detect repeating headers/footers (bounding-box position + `rapidfuzz` similarity) and keep each one only once across the whole document, instead of once per page
4. Split text into overlapping chunks (size: 800, overlap: 150)
5. Filter out chunks shorter than 50 characters
6. Encode each chunk into a 1024-dim embedding vector (batch size: 8)
7. Delete any previously stored chunks for this document
8. Upsert all chunks with metadata into Qdrant `course_chunks` collection

### DELETE /api/documents/{document_id}

**Called by:** Spring Boot, when a document is deleted (or to explicitly clear chunks outside the normal re-index flow)
**Auth:** HTTP Basic Auth required
**Idempotent:** succeeds even if the document was never indexed (no matching chunks in Qdrant)

**Response (success):**
```json
{
  "document_id": 123,
  "status": "SUCCESS",
  "error": null
}
```

**Response (failure):**
```json
{
  "document_id": 123,
  "status": "FAILED",
  "error": "Qdrant connection failed"
}
```

### POST /api/query/embed

**Called by:** LLM Response Service (Person C)
**Auth:** HTTP Basic Auth required

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

**Auth:** Not required

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "qdrant_connected": true
}
```

Possible statuses: `"ok"` (both healthy), `"degraded"` (one healthy), `"error"` (none healthy).

## Logging

All requests pass through a logging middleware (`fastapi_app/middleware.py`) that:

- Assigns a request ID (`X-Request-ID` header — reused if the caller already sends one, generated otherwise) and includes it in every log line for that request
- Logs method, path, status code, duration, and the calling user (`X-User` header, or `"anonymous"`)
- Logs the request payload and response body as JSON, with sensitive fields (`password`, `parola`, `token`, `secret`, `access_token`, `authorization`) automatically masked as `***`
- Skips logging the **response body** for `/api/query/embed` and `/api/documents/ingest`, since embeddings and extracted PDF text are large and not useful in logs
- Truncates any logged payload/response to 2000 characters

## Running Tests

```powershell
uv run pytest tests/ -v
```