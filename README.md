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
- **Async Image Captioning** — Images found in the PDF are extracted, captioned with Gemini Vision, embedded, and indexed in the background (after the synchronous text response), then Spring Boot is notified via a callback (see [Async Image Processing](#async-image-processing))
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
| `GEMINI_API_KEY`         | *(none — required for image captioning)* | API key for Gemini Vision (image captioning)     |
| `GEMINI_VISION_MODEL`    | `gemini-3.1-flash-lite` | Gemini model used to caption extracted images                 |
| `GEMINI_REQUEST_DELAY_SECONDS` | `13.0`            | Delay between Gemini requests (rate-limit friendly)            |
| `MIN_IMAGE_WIDTH`        | `100`                   | Minimum width (px) for an extracted image to be processed      |
| `MIN_IMAGE_HEIGHT`       | `100`                   | Minimum height (px) for an extracted image to be processed     |
| `MAX_IMAGES_PER_DOCUMENT`| `20`                    | Cap on images processed per document                          |
| `SPRING_BOOT_CALLBACK_URL`      | `""`             | URL Spring Boot exposes to receive the image-processing status callback (`PATCH`) |
| `SPRING_BOOT_CALLBACK_USERNAME` | `""`             | Basic Auth username used when calling Spring Boot back          |
| `SPRING_BOOT_CALLBACK_PASSWORD` | `""`             | Basic Auth password used when calling Spring Boot back          |

> ⚠️ If `RAG_SERVICE_USERNAME` / `RAG_SERVICE_PASSWORD` are not set, any request to a protected endpoint returns **HTTP 500** ("Autentificarea nu este configurată pe server"), not a silent bypass.
>
> ⚠️ `SPRING_BOOT_CALLBACK_URL` currently has **no matching endpoint on the Spring Boot side yet** — the async image callback (`PATCH .../image-status`) will fail and retry, then log an error and give up (best-effort, non-fatal; the document just stays `INDEXED_TEXT_ONLY`). See [Async Image Processing](#async-image-processing).

## Running the Project

### With Docker Compose

```powershell
# Start all services (Qdrant, MinIO, Embedder Service)
docker-compose up --build

# The service will be available at http://localhost:8001
# Qdrant at http://localhost:6333
# MinIO console at http://localhost:9001
```

> The embedder container/service is named **`embedder`** (not `embedder-service`) so that the Spring Boot backend can reach it at the hostname it expects by default (`http://embedder:8001` / `RAG_EMBEDDER_URL`).

### Rulare integrată cu akadion

`docker-compose.yml` atașează containerul `embedder` și la rețeaua Docker externă **`akadion_shared`**, ca să fie vizibil pentru backend-ul Spring Boot din repo-ul `akadion` (care rulează separat, cu propriul `compose.yaml`).

1. Rețeaua `akadion_shared` trebuie să existe înainte de `docker-compose up`. Launcherul din akadion (`Start Akadion.cmd`) o creează automat când pornești stack-ul akadion. Dacă pornești embedder-ul separat/primul, creaz-o manual:
   ```powershell
   docker network create akadion_shared
   ```
2. În `.env`, setează valorile ca să corespundă cu ce așteaptă `akadion/backend` (`application.properties` / `compose.yaml`):

   | Variabilă | Valoare pentru integrare cu akadion |
   |---|---|
   | `RAG_SERVICE_USERNAME` | aceeași valoare ca `app.rag.auth.username` din backend (implicit `akadion-spring-backend`) |
   | `RAG_SERVICE_PASSWORD` | aceeași valoare ca `app.rag.auth.password` din backend (implicit `parola_spring_rag`) |
   | `MINIO_ENDPOINT` | `akadion-minio:9000` (MinIO-ul real din stack-ul akadion, nu cel local) |
   | `MINIO_BUCKET` | `course-documents` (bucket-ul real, creat de `minio-setup` din akadion) |
   | `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | aceleași ca `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` din `akadion/compose.yaml` (implicit `minioadmin` / `MinioParola123!`) |

   Dacă folosești `MINIO_ENDPOINT=akadion-minio:9000`, MinIO-ul local propriu (serviciul `minio` din acest `docker-compose.yml`) devine redundant — poți porni doar `embedder` și `qdrant`:
   ```powershell
   docker-compose up --build embedder qdrant
   ```
3. `SPRING_BOOT_CALLBACK_URL` rămâne fără efect real până când backend-ul akadion implementează endpoint-ul de primire (vezi avertismentul din secțiunea de variabile de mediu de mai sus).

## Authentication

The following endpoints require **HTTP Basic Auth**:

- `POST /api/documents/ingest`
- `DELETE /api/documents/ingest/{document_id}`
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
| POST   | `/api/documents/ingest`               | Yes            | Spring Boot                     | Ingest PDF document (fetch → extract → chunk → embed → store); schedules image captioning in the background |
| DELETE | `/api/documents/ingest/{document_id}` | Yes            | Spring Boot                     | Delete all stored chunks for a document from Qdrant                |
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

**Response (success — text indexed, images still processing in background):**
```json
{
  "document_id": 123,
  "status": "INDEXED_TEXT_ONLY",
  "chunks_count": 42,
  "images_queued": null,
  "error": null,
  "processing_time_ms": 3500
}
```

> `status` is `"INDEXED_TEXT_ONLY"`, not `"INDEXED"` — text indexing is synchronous and complete by the time this response is sent, but images from the PDF are captioned and indexed afterwards, in the background. `images_queued` is currently always `null` in the response (the exact count is only known once background extraction runs); Spring Boot only needs to treat any non-`"FAILED"` status as success.

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
9. Schedule image extraction + captioning as a background task (see [Async Image Processing](#async-image-processing)) and return the response above immediately

### DELETE /api/documents/ingest/{document_id}

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

## Async Image Processing

Text ingestion (`POST /api/documents/ingest`) stays synchronous, but images embedded in the PDF are handled afterwards, as a `BackgroundTasks` job, so a document full of images can't blow the ingest timeout:

1. `ingest_document` returns `status: "INDEXED_TEXT_ONLY"` to Spring Boot immediately after text is stored in Qdrant, and schedules `process_images_background` (`fastapi_app/services/image_pipeline.py`) with a fresh `ingest_version` UUID.
2. In the background: extract images from the PDF (`image_extractor.py`, filtered by `MIN_IMAGE_WIDTH`/`MIN_IMAGE_HEIGHT`, capped at `MAX_IMAGES_PER_DOCUMENT`), caption each with Gemini Vision (`vision_captioner.py`, paced by `GEMINI_REQUEST_DELAY_SECONDS`), embed the captions, and upsert them into Qdrant with `source_type="image"` and `page_number` metadata.
3. If the professor re-indexes the same document before the background job finishes, `register_ingest_version`/`ingest_version` makes the stale job detect it's no longer current and discard its results quietly, instead of racing the newer ingest.
4. On completion (success or failure), `springboot_callback.py` sends `PATCH {SPRING_BOOT_CALLBACK_URL}` with `{"document_id", "status": "INDEXED" | "FAILED_IMAGES", "images_indexed", "images_failed"}`, retrying up to 3 times.

> ⚠️ **Not yet wired up end-to-end:** as of now, `akadion/backend` does not implement the endpoint that's supposed to receive this callback (no `PATCH .../image-status` route exists there yet — see `Embedder_Async_Images_Contract_SpringBoot.md` in the akadion repo root). Until the Spring Boot side adds it, the callback will 404, retry, and log an error — harmless (the document simply stays at `INDEXED_TEXT_ONLY` from Spring Boot's point of view), but images won't be reflected as "done" anywhere outside this service's own logs/Qdrant.

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