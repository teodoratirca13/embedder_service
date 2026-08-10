import time
import uuid
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks

from fastapi_app.auth import verify_credentials
from fastapi_app.models import IngestRequest, IngestResponse, DeleteResponse
from fastapi_app.services import (
    get_pdf_extractor,
    get_text_chunker,
    get_embedding_model,
    get_qdrant_client,
    process_images_background,
    register_ingest_version
)
from fastapi_app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(
    prefix="/api",
    tags=["documents"],
    dependencies=[Depends(verify_credentials)]
)


@router.post("/documents/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """
    Ingest a document: fetch PDF → extract → chunk → embed → store in Qdrant.

    Called by Spring Boot when professor clicks "Indexează document".
    Spring Boot waits synchronously for this response.

    Partea de text (Step 1-5) e sincrona — Spring Boot primeste raspunsul
    imediat dupa ce textul e salvat in Qdrant. Imaginile din PDF sunt
    procesate separat, in fundal (BackgroundTasks), ca sa nu riste sa
    depaseasca timeout-ul de ingest. La final, embedder-ul notifica Spring
    Boot printr-un apel separat (springboot_callback.py).

    Args:
        request: IngestRequest with document metadata and MinIO path
        background_tasks: injectat automat de FastAPI

    Returns:
        IngestResponse with status ("INDEXED_TEXT_ONLY" or "FAILED") and chunk count
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

        # ===== STEP 6: Programeaza procesarea imaginilor, in fundal =====
        # Genereaza un "bilet" unic pentru aceasta rulare — daca profesorul
        # reindexeaza documentul inainte ca job-ul de imagini sa se termine,
        # image_pipeline.py va detecta ca acest bilet nu mai e cel curent si
        # va renunta liniștit la rezultat (vezi sectiunea 6 din plan).
        ingest_version = str(uuid.uuid4())
        register_ingest_version(document_id, ingest_version)
        background_tasks.add_task(process_images_background, request, ingest_version)

        logger.info(
            "Step 6 complete: Image processing scheduled in background",
            extra={"extra_data": {"document_id": document_id, "ingest_version": ingest_version}}
        )

        # ===== SUCCESS (text) =====
        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Ingest complete: TEXT INDEXED, images queued",
            extra={"extra_data": {
                "document_id": document_id,
                "status": "INDEXED_TEXT_ONLY",
                "chunk_count": len(chunks),
                "processing_time_ms": processing_time_ms
            }}
        )

        return IngestResponse(
            document_id=document_id,
            status="INDEXED_TEXT_ONLY",
            chunks_count=len(chunks),
            images_queued=None,  # numarul exact se afla doar dupa extragere, in job-ul de fundal
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

@router.delete("/documents/ingest/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: int) -> DeleteResponse:
    """
    Delete all Qdrant chunks for a document.

    Called by Spring Boot (RagIngestService.stergeDinIngest) when a document
    is deleted, or before re-indexing outside the normal ingest flow.
    Idempotent — succeeds even if the document was never indexed.

    Path-ul e /documents/ingest/{id} (nu /documents/{id}) ca sa corespunda
    exact cu ce apeleaza backend-ul Spring Boot — vezi RagIngestService.java.
    """
    logger.info(
        "Delete request received",
        extra={"extra_data": {"document_id": document_id}}
    )

    try:
        qdrant_client = get_qdrant_client()
        qdrant_client.delete_document_chunks(document_id)

        logger.info(
            "Document chunks deleted",
            extra={"extra_data": {"document_id": document_id}}
        )

        return DeleteResponse(
            document_id=document_id,
            status="SUCCESS"
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(
            "Delete failed",
            extra={"extra_data": {"document_id": document_id, "error": error_msg}}
        )

        return DeleteResponse(
            document_id=document_id,
            status="FAILED",
            error=error_msg
        )