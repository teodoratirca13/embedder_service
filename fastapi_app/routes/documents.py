import time
from fastapi import APIRouter, HTTPException

from fastapi_app.models import IngestRequest, IngestResponse
from fastapi_app.services import (
    get_pdf_extractor,
    get_text_chunker,
    get_embedding_model,
    get_qdrant_client
)
from fastapi_app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/documents/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Ingest a document: fetch PDF → extract → chunk → embed → store in Qdrant.

    Called by Spring Boot when professor clicks "Indexează document".
    Spring Boot waits synchronously for this response.

    Args:
        request: IngestRequest with document metadata and MinIO path

    Returns:
        IngestResponse with status ("INDEXED" or "FAILED") and chunk count
    """
    start_time = time.time()
    document_id = request.document_id

    logger.info(
        "Ingest started",
        extra={"extra_data": {
            "document_id": document_id,
            "course_id": request.course_id,
            "week_id": request.week_id,
            "path_minio": request.path_minio,
            "document_title": request.document_title,
            "professor_id": request.professor_id
        }}
    )

    try:
        # ===== STEP 1: Extract text from PDF =====
        logger.info("Step 1: Extracting text from PDF", extra={"extra_data": {"document_id": document_id}})

        pdf_extractor = get_pdf_extractor()
        extracted_text = pdf_extractor.extract_text(request.path_minio)

        logger.info(
            "Step 1 complete: Text extracted",
            extra={"extra_data": {
                "document_id": document_id,
                "text_length": len(extracted_text)
            }}
        )

        # ===== STEP 2: Chunk the text =====
        logger.info("Step 2: Chunking text", extra={"extra_data": {"document_id": document_id}})

        chunker = get_text_chunker()
        chunks = chunker.chunk_text(extracted_text)

        if not chunks:
            error_msg = "No chunks produced from text"
            logger.error(error_msg, extra={"extra_data": {"document_id": document_id}})
            return IngestResponse(
                document_id=document_id,
                status="FAILED",
                error=error_msg,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        logger.info(
            "Step 2 complete: Text chunked",
            extra={"extra_data": {
                "document_id": document_id,
                "chunk_count": len(chunks)
            }}
        )

        # ===== STEP 3: Embed all chunks =====
        logger.info("Step 3: Embedding chunks", extra={"extra_data": {"document_id": document_id}})

        embedder = get_embedding_model()
        embeddings = embedder.embed_batch(chunks, batch_size=8)

        if len(embeddings) != len(chunks):
            error_msg = f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}"
            logger.error(error_msg, extra={"extra_data": {"document_id": document_id}})
            return IngestResponse(
                document_id=document_id,
                status="FAILED",
                error=error_msg,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        logger.info(
            "Step 3 complete: Chunks embedded",
            extra={"extra_data": {
                "document_id": document_id,
                "embedding_count": len(embeddings)
            }}
        )

        # ===== STEP 4: Delete old chunks (if re-indexing) =====
        logger.info("Step 4: Deleting old chunks (if re-indexing)", extra={"extra_data": {"document_id": document_id}})

        qdrant_client = get_qdrant_client()
        qdrant_client.delete_document_chunks(document_id)

        logger.info("Step 4 complete: Old chunks deleted", extra={"extra_data": {"document_id": document_id}})

        # ===== STEP 5: Store in Qdrant =====
        logger.info("Step 5: Storing chunks in Qdrant", extra={"extra_data": {"document_id": document_id}})

        qdrant_client.upsert_chunks(
            document_id=document_id,
            course_id=request.course_id,
            week_id=request.week_id,
            document_title=request.document_title,
            professor_id=request.professor_id,
            chunks=chunks,
            embeddings=embeddings
        )

        logger.info("Step 5 complete: Chunks stored in Qdrant", extra={"extra_data": {"document_id": document_id}})

        # ===== SUCCESS =====
        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Ingest complete: SUCCESS",
            extra={"extra_data": {
                "document_id": document_id,
                "status": "INDEXED",
                "chunk_count": len(chunks),
                "processing_time_ms": processing_time_ms
            }}
        )

        return IngestResponse(
            document_id=document_id,
            status="INDEXED",
            chunks_count=len(chunks),
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        # ===== FAILURE =====
        processing_time_ms = int((time.time() - start_time) * 1000)
        error_msg = str(e)

        logger.error(
            "Ingest complete: FAILED",
            extra={"extra_data": {
                "document_id": document_id,
                "status": "FAILED",
                "error": error_msg,
                "processing_time_ms": processing_time_ms
            }}
        )

        return IngestResponse(
            document_id=document_id,
            status="FAILED",
            error=error_msg,
            processing_time_ms=processing_time_ms
        )