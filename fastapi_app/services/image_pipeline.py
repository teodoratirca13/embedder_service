import time

from fastapi_app.models import IngestRequest
from fastapi_app.services.image_extractor import get_image_extractor
from fastapi_app.services.vision_captioner import get_vision_captioner
from fastapi_app.services.embedder import get_embedding_model
from fastapi_app.services.qdrant_client import get_qdrant_client
from fastapi_app.services.springboot_callback import get_springboot_callback_client
from fastapi_app.utils import setup_logger, get_settings

logger = setup_logger(__name__)


#profesul poate da "reindexeaza"intre timp daca a modificat cursul
#dar noi trebuie sa stim ca aia e varianta cea mai recenta si sa stim sa o patram pe aia
_active_ingest_versions: dict[int, str] = {}


def register_ingest_version(document_id: int, ingest_version: str) -> None:
    """Apelat din documents.py, chiar inainte de a programa job-ul de fundal."""
    _active_ingest_versions[document_id] = ingest_version


def _is_ingest_version_current(document_id: int, ingest_version: str) -> bool:
    """True daca 'biletul' cu care a pornit acest job e inca cel mai recent."""
    return _active_ingest_versions.get(document_id) == ingest_version


def _notify_final_status(document_id: int, status: str, images_indexed: int, images_failed: int) -> None:
    """
    Anunta Spring Boot despre statusul final al job-ului de imagini.

    Daca adresa de callback nu e configurata (punct de blocaj cunoscut),
    nu oprim procesul — documentul ramane oricum corect salvat in Qdrant,
    doar Spring Boot nu afla de status.
    """
    try:
        callback_client = get_springboot_callback_client()
        callback_client.notify_image_status(
            document_id=document_id,
            status=status,
            images_indexed=images_indexed,
            images_failed=images_failed
        )
    except RuntimeError as e:
        logger.warning(
            "SpringBoot callback not configured — skipping notification",
            extra={"extra_data": {"document_id": document_id, "status": status, "error": str(e)}}
        )


def process_images_background(request: IngestRequest, ingest_version: str) -> None:
    """
    Job de fundal: extrage imaginile dintr-un document, le captioneaza cu
    Gemini Vision, le transforma in embeddings si le salveaza in Qdrant.
    La final, anunta Spring Boot despre rezultat.

    Nu arunca exceptii in afara — orice eroare neasteptata e prinsa, logata,
    si urmata de o notificare "FAILED_IMAGES" catre Spring Boot.
    """
    document_id = request.document_id

    logger.info(
        "Starting background image processing",
        extra={"extra_data": {"document_id": document_id, "ingest_version": ingest_version}}
    )

    images_indexed = 0
    images_failed = 0

    try:
        # 1. Extragere imagini din PDF
        extractor = get_image_extractor()
        extracted_images = extractor.extract_images(request.path_minio)

        if not extracted_images:
            logger.info(
                "No images found in document — nothing to process",
                extra={"extra_data": {"document_id": document_id}}
            )
            _notify_final_status(document_id, "INDEXED", images_indexed=0, images_failed=0)
            return

        # 2. Captioning, imagine cu imagine — un esec individual nu opreste bucla
        captioner = get_vision_captioner()
        request_delay = get_settings().gemini_request_delay_seconds
        captions: list[str] = []
        page_numbers: list[int] = []
        total_images = len(extracted_images)

        for idx, img in enumerate(extracted_images):
            caption = captioner.caption_image(img.image_bytes, mime_type=img.mime_type)
            if caption is None:
                images_failed += 1
            else:
                captions.append(caption)
                page_numbers.append(img.page_number)

            is_last_image = (idx == total_images - 1)
            if request_delay > 0 and not is_last_image:
                time.sleep(request_delay)

        if not captions:
            logger.warning(
                "All images failed captioning",
                extra={"extra_data": {"document_id": document_id, "total_images": len(extracted_images)}}
            )
            _notify_final_status(document_id, "FAILED_IMAGES", images_indexed=0, images_failed=images_failed)
            return

        # 3. Embedding peste caption-uri (fiecare caption e deja un singur chunk)
        embedder = get_embedding_model()
        embeddings = embedder.embed_batch(captions)

        # 4. Verificare ingest_version — chiar inainte de a scrie in Qdrant
        if not _is_ingest_version_current(document_id, ingest_version):
            logger.warning(
                "Ingest version expired — newer reindex detected, discarding image results",
                extra={"extra_data": {"document_id": document_id}}
            )
            return  # fara callback — job-ul nou, mai recent, se ocupa de status

        # 5. Salvare in Qdrant
        qdrant = get_qdrant_client()
        qdrant.upsert_chunks(
            document_id=document_id,
            course_id=request.course_id,
            week_id=request.week_id,
            document_title=request.document_title,
            professor_id=request.professor_id,
            chunks=captions,
            embeddings=embeddings,
            source_type="image",
            page_numbers=page_numbers
        )

        images_indexed = len(captions)

        logger.info(
            "Background image processing complete",
            extra={"extra_data": {
                "document_id": document_id,
                "images_indexed": images_indexed,
                "images_failed": images_failed
            }}
        )

        _notify_final_status(document_id, "INDEXED", images_indexed, images_failed)

    except Exception as e:
        logger.error(
            "Background image processing failed",
            extra={"extra_data": {"document_id": document_id, "error": str(e)}}
        )
        _notify_final_status(document_id, "FAILED_IMAGES", images_indexed, images_failed)