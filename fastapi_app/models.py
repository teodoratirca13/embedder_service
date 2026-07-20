from pydantic import BaseModel
from typing import Optional


#asemanatoare cu DTO-urile

# ============================================================================
# INGEST ENDPOINT (POST /api/documents/ingest)
# ============================================================================

class IngestRequest(BaseModel):
    """Request body for document ingestion."""
    document_id: int
    course_id: int
    week_id: int
    path_minio: str
    document_title: str
    professor_id: int

    class Config:
        json_schema_extra = {  #pentru swagger
            "example": {
                "document_id": 123,
                "course_id": 5,
                "week_id": 15,
                "path_minio": "algorithms/week3/lecture.pdf",
                "document_title": "Lecture 3",
                "professor_id": 7
            }
        }


class IngestResponse(BaseModel):
    """Response body from document ingestion."""
    document_id: int
    status: str  # "INDEXED" or "FAILED"
    chunks_count: Optional[int] = None
    error: Optional[str] = None
    processing_time_ms: int

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": 123,
                "status": "INDEXED",
                "chunks_count": 42,
                "error": None,
                "processing_time_ms": 3500
            }
        }


# ============================================================================
# QUERY EMBEDDING ENDPOINT (POST /api/query/embed)
# ============================================================================

class EmbedRequest(BaseModel):
    """Request body for query embedding."""
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Explică programarea dinamică"
            }
        }


class EmbedResponse(BaseModel):
    """Response body from query embedding."""
    embedding: list[float]
    dimension: int
    model: str

    class Config:
        json_schema_extra = {
            "example": {
                "embedding": [0.123, -0.456, 0.789],  # simplified, actually 1024 values
                "dimension": 1024,
                "model": "BAAI/bge-m3"
            }
        }


# ============================================================================
# HEALTH CHECK ENDPOINT (GET /api/health)
# ============================================================================

class HealthResponse(BaseModel):
    """Response body from health check."""
    status: str  # "ok", "degraded", or "error"
    model_loaded: bool
    qdrant_connected: bool

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model_loaded": True,
                "qdrant_connected": True
            }
        }