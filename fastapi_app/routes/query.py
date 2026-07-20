import logging
from fastapi import APIRouter

from fastapi_app.models import EmbedRequest, EmbedResponse
from fastapi_app.services import get_embedding_model
from fastapi_app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query/embed", response_model=EmbedResponse)
def embed_query(request: EmbedRequest) -> EmbedResponse:
    #embeds the question from the student
    text = request.text

    logger.info(
        "Query embedding requested",
        extra={"extra_data": {"text_length": len(text)}}
    )

    try:
        embedder = get_embedding_model()
        embedding = embedder.embed_text(text)

        logger.info(
            "Query embedding complete",
            extra={"extra_data": {
                "text_length": len(text),
                "embedding_dimension": len(embedding),
                "embedding_time_ms": int(embedder.last_embedding_time_ms)
            }}
        )

        return EmbedResponse(
            embedding=embedding,
            dimension=len(embedding),
            model="BAAI/bge-m3"
        )

    except Exception as e:
        error_msg = f"Embedding failed: {str(e)}"
        logger.error(error_msg, extra={"extra_data": {"text_length": len(text)}})
        raise