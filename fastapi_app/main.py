import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi_app.routes import health_router, documents_router, query_router
from fastapi_app.utils.logger import setup_logger
from fastapi_app.utils.config import get_settings

logger = setup_logger(__name__)


# ============================================================================
# LIFESPAN CONTEXT MANAGER (startup/shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage app startup and shutdown.
    Everything before 'yield' runs on startup.
    Everything after 'yield' runs on shutdown.
    """
    # ===== STARTUP =====
    logger.info("Embedder Service starting up...")

    settings = get_settings()
    logger.info(
        "Configuration loaded",
        extra={"extra_data": {
            "qdrant_url": settings.qdrant_url,
            "minio_endpoint": settings.minio_endpoint,
            "bge_model_path": settings.bge_model_path,
            "log_level": settings.log_level
        }}
    )

    # Pre-load the embedding model
    try:
        from fastapi_app.services import get_embedding_model
        model = get_embedding_model()
        logger.info(
            "Embedding model loaded",
            extra={"extra_data": {"dimension": model.get_embedding_dimension()}}
        )
    except Exception as e:
        logger.error("Failed to load embedding model", extra={"extra_data": {"error": str(e)}})
        raise

    # Test Qdrant connection
    try:
        from fastapi_app.services import get_qdrant_client
        qdrant = get_qdrant_client()
        is_healthy = qdrant.is_healthy()
        logger.info("Qdrant connected", extra={"extra_data": {"healthy": is_healthy}})
    except Exception as e:
        logger.error("Failed to connect to Qdrant", extra={"extra_data": {"error": str(e)}})
        raise

    logger.info("Embedder Service startup complete ✓")

    # ===== YIELD (app runs here) =====
    yield

    # ===== SHUTDOWN =====
    logger.info("Embedder Service shutting down...")
    # Add cleanup code here if needed
    logger.info("Embedder Service shutdown complete")


# ============================================================================
# CREATE FASTAPI APP WITH LIFESPAN
# ============================================================================

app = FastAPI(
    title="Embedder Service",
    description="Document embedding, chunking, and vector storage for RAG platform",
    version="0.1.0",
    lifespan=lifespan  # ← Pass the lifespan context manager
)

# Configure CORS (allow Spring Boot to call us)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REGISTER ROUTERS
# ============================================================================

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
def root():
    """Root endpoint — redirects to API docs."""
    return {
        "message": "Embedder Service",
        "docs": "http://localhost:8001/docs",
        "openapi": "http://localhost:8001/openapi.json"
    }


# ============================================================================
# If running directly (not through uvicorn)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
