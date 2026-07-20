import logging
from fastapi import APIRouter, HTTPException

from fastapi_app.models import HealthResponse
from fastapi_app.services import get_embedding_model, get_qdrant_client
from fastapi_app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:  # ← Changed from async def
    """
    Health check endpoint.

    Returns:
        HealthResponse with status ("ok", "degraded", or "error")
    """
    logger.debug("Health check requested")

    model_loaded = False
    qdrant_connected = False

    # Check if embedding model is loaded
    try:
        model = get_embedding_model()
        model_loaded = model.get_embedding_dimension() == 1024
        logger.debug("Embedding model status checked", extra={"extra_data": {"model_loaded": model_loaded}})
    except Exception as e:
        logger.error("Failed to check embedding model", extra={"extra_data": {"error": str(e)}})
        model_loaded = False

    # Check if Qdrant is connected
    try:
        qdrant = get_qdrant_client()
        qdrant_connected = qdrant.is_healthy()
        logger.debug("Qdrant status checked", extra={"extra_data": {"qdrant_connected": qdrant_connected}})
    except Exception as e:
        logger.error("Failed to check Qdrant", extra={"extra_data": {"error": str(e)}})
        qdrant_connected = False

    # Determine overall status
    if model_loaded and qdrant_connected:
        status = "ok"
    elif model_loaded or qdrant_connected:
        status = "degraded"
    else:
        status = "error"

    logger.info(
        "Health check completed",
        extra={"extra_data": {
            "status": status,
            "model_loaded": model_loaded,
            "qdrant_connected": qdrant_connected
        }}
    )

    response = HealthResponse(
        status=status,
        model_loaded=model_loaded,
        qdrant_connected=qdrant_connected
    )

    return response