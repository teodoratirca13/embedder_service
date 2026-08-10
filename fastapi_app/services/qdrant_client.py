import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from typing import Optional

from fastapi_app.utils import setup_logger
from fastapi_app.utils import get_settings

logger = setup_logger(__name__)

COLLECTION_NAME = "course_chunks"
VECTOR_DIMENSION = 1024


class QdrantVectorStore:

    def __init__(self, qdrant_url: str):

         #qdrant_url: URL to Qdrant server (e.g., "http://localhost:6333")

        logger.info(
            "Initializing Qdrant client",
            extra={"extra_data": {"qdrant_url": qdrant_url}}
        )

        try:
            self.client = QdrantClient(url=qdrant_url)

            # Test connection
            self.client.get_collections()
            logger.info(
                "Qdrant client connected successfully",
                extra={"extra_data": {"qdrant_url": qdrant_url}}
            )

        except Exception as e:
            error_msg = f"Failed to connect to Qdrant: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"qdrant_url": qdrant_url}})
            raise RuntimeError(error_msg) from e

        self._ensure_collection_exists() #creaza colectia daca nu exista deja

    def _ensure_collection_exists(self):
        """Create the course_chunks collection if it doesn't exist."""
        try:
            if self.client.collection_exists(COLLECTION_NAME):
                logger.info(
                    "Collection already exists",
                    extra={"extra_data": {"collection": COLLECTION_NAME}}
                )
                return

            logger.info(
                "Creating collection",
                extra={"extra_data": {"collection": COLLECTION_NAME, "dimension": VECTOR_DIMENSION}}
            )

            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=Distance.COSINE  #cum compara vectorii
                )
            )

            logger.info(
                "Collection created successfully",
                extra={"extra_data": {"collection": COLLECTION_NAME}}
            )

        except Exception as e:
            error_msg = f"Failed to create collection: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _generate_point_id(self, document_id: int, source_type: str, chunk_index: int) -> str:  # pentru a nu exista puncte duplicate

        namespace_str = f"{document_id}_{source_type}_{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_OID, namespace_str))

    def delete_document_chunks(self, document_id: int):
        #sterge documente inainte de reindexare

        logger.info(
            "Deleting existing chunks for document",
            extra={"extra_data": {"document_id": document_id}}
        )

        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )

            logger.info(
                "Deleted existing chunks for document",
                extra={"extra_data": {"document_id": document_id}}
            )

        except Exception as e:
            error_msg = f"Failed to delete existing chunks: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"document_id": document_id}})
            raise RuntimeError(error_msg) from e

    def upsert_chunks(
            self,
            document_id: int,
            course_id: int,
            week_id: int,
            document_title: str,
            professor_id: int,
            chunks: list[str],
            embeddings: list[list[float]],
            source_type: str = "text",
            page_numbers: Optional[list[int]] = None
    ):
        """
        Store document chunks and their embeddings in Qdrant.

        Args:
            ...(neschimbat)...
            source_type: "text" (implicit) sau "image" — distinge sursa chunk-ului
            page_numbers: lista de numere de pagina, aliniata index-cu-index cu
                chunks (doar pentru source_type="image"); None pentru text
        """
        logger.info(
            "Upserting chunks to Qdrant",
            extra={"extra_data": {
                "document_id": document_id,
                "course_id": course_id,
                "chunk_count": len(chunks),
                "source_type": source_type
            }}
        )

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")

        if page_numbers is not None and len(page_numbers) != len(chunks):
            raise ValueError(f"page_numbers count ({len(page_numbers)}) != chunks count ({len(chunks)})")

        try:
            points = []

            for chunk_index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = self._generate_point_id(document_id, source_type, chunk_index)

                payload = {
                    "chunk_text": chunk_text,
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "document_id": document_id,
                    "course_id": course_id,
                    "week_id": week_id,
                    "document_title": document_title,
                    "professor_id": professor_id,
                    "source_type": source_type
                }

                if page_numbers is not None:
                    payload["page_number"] = page_numbers[chunk_index]

                point = PointStruct(id=point_id, vector=embedding, payload=payload)
                points.append(point)

            # restul metodei ramane neschimbat (batch upsert, logging)
            batch_size = 64
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(collection_name=COLLECTION_NAME, points=batch)
                logger.debug(
                    "Upserted batch",
                    extra={"extra_data": {
                        "document_id": document_id,
                        "batch_index": i // batch_size,
                        "batch_size": len(batch)
                    }}
                )

            logger.info(
                "Chunks upserted successfully",
                extra={"extra_data": {
                    "document_id": document_id,
                    "total_points": len(points),
                    "source_type": source_type
                }}
            )

        except Exception as e:
            error_msg = f"Failed to upsert chunks: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"document_id": document_id}})
            raise RuntimeError(error_msg) from e

    def is_healthy(self) -> bool:

        try:
            self.client.get_collection(COLLECTION_NAME)
            return True
        except Exception:
            return False


_vector_store = None


def get_qdrant_client() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = QdrantVectorStore(qdrant_url=settings.qdrant_url)
    return _vector_store