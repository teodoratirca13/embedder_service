import time
from sentence_transformers import SentenceTransformer

import numpy as np

from fastapi_app.utils.logger import setup_logger
from fastapi_app.utils.config import get_settings

logger = setup_logger(__name__)


class EmbeddingModel:

    def __init__(self, model_path: str = "BAAI/bge-m3"):

       # the first
        logger.info(
            "Loading embedding model",
            extra={"extra_data": {"model_path": model_path}}  #verifica .cache mai intai
        )

        try:
            start = time.time()
            self.model = SentenceTransformer(model_path)
            elapsed_ms = (time.time() - start) * 1000

            self.last_embedding_time_ms = 0
            dimensions = self.model.get_embedding_dimension()

            logger.info(
                "Embedding model loaded successfully",
                extra={"extra_data": {
                    "model_path": model_path,
                    "output_dimension": dimensions,
                    "load_time_ms": int(elapsed_ms)
                }}
            )

        except Exception as e:
            error_msg = f"Failed to load embedding model: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"model_path": model_path}})
            raise RuntimeError(error_msg) from e

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string (e.g., a query).

        Args:
            text: Text to embed

        Returns:
            List of 1024 floats (the embedding vector)
        """
        logger.debug(
            "Embedding single text",
            extra={"extra_data": {"text_length": len(text)}}
        )

        start = time.time()

        try:
            # Encode to numpy array
            embedding = self.model.encode(text, convert_to_numpy=True)

            # Track time
            self.last_embedding_time_ms = (time.time() - start) * 1000

            logger.debug(
                "Single text embedded",
                extra={"extra_data": {
                    "text_length": len(text),
                    "embedding_time_ms": int(self.last_embedding_time_ms)
                }}
            )

            # Convert numpy array to Python list
            return embedding.tolist()

        except Exception as e:
            error_msg = f"Embedding failed: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"text_length": len(text)}})
            raise RuntimeError(error_msg) from e

    def embed_batch(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """
        Embed multiple texts efficiently using batch processing.

        Args:
            texts: List of text strings to embed
            batch_size: Process N texts at a time (default 8, safe for CPU)

        Returns:
            List of embeddings (each is a list of 1024 floats)
        """
        logger.info(
            "Embedding batch of texts",
            extra={"extra_data": {
                "batch_count": len(texts),
                "batch_size": batch_size
            }}
        )

        start = time.time()

        try:
            # Encode all texts in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            elapsed_ms = (time.time() - start) * 1000
            self.last_embedding_time_ms = elapsed_ms

            logger.info(
                "Batch embedding complete",
                extra={"extra_data": {
                    "batch_count": len(texts),
                    "total_time_ms": int(elapsed_ms),
                    "avg_time_per_text_ms": int(elapsed_ms / len(texts)) if texts else 0
                }}
            )

            # Convert each numpy array to Python list
            return [e.tolist() for e in embeddings]

        except Exception as e:
            error_msg = f"Batch embedding failed: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"batch_count": len(texts)}})
            raise RuntimeError(error_msg) from e

    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings (should be 1024 for BGE-M3)."""
        return self.model.get_embedding_dimension()


# Create a singleton instance (loaded once on startup)
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create embedding model singleton."""
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        _embedding_model = EmbeddingModel(model_path=settings.bge_model_path)
    return _embedding_model