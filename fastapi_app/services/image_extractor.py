import io
from dataclasses import dataclass

import fitz  # PyMuPDF
from minio import Minio
from minio.error import S3Error

from fastapi_app.utils import get_settings, setup_logger

logger = setup_logger(__name__)


@dataclass
class ExtractedImage:
    image_bytes: bytes
    mime_type: str
    page_number: int
    width: int
    height: int


class ImageExtractor:
    """Extrage imagini din PDF-uri stocate in MinIO, filtrate dupa dimensiune si numar maxim."""

    def __init__(self):
        settings = get_settings()  #se citeste configuratia
        self.minio_client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False
        )
        self.bucket_name = settings.minio_bucket
        self.min_width = settings.min_image_width
        self.min_height = settings.min_image_height
        self.max_images = settings.max_images_per_document

    def _extract_images_from_page(self, doc, page, page_number: int) -> list[ExtractedImage]:
        images = []

        for img_info in page.get_images(full=True):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                logger.warning(
                    "Failed to extract image — skipping",
                    extra={"extra_data": {"page": page_number, "xref": xref, "error": str(e)}}
                )
                continue

            image_bytes = base_image["image"]
            ext = base_image["ext"]
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            images.append(ExtractedImage(
                image_bytes=image_bytes,
                mime_type=f"image/{ext}",
                page_number=page_number,
                width=width,
                height=height
            ))

        return images

    def _filter_images(self, images: list[ExtractedImage]) -> list[ExtractedImage]:
        filtered = [
            img for img in images
            if img.width >= self.min_width and img.height >= self.min_height
        ]

        skipped_small = len(images) - len(filtered)
        if skipped_small > 0:
            logger.info(
                "Filtered out small images",
                extra={"extra_data": {"skipped_count": skipped_small}}
            )

        if len(filtered) > self.max_images:
            logger.warning(
                "Too many images — truncating to max_images_per_document",
                extra={"extra_data": {"total_found": len(filtered), "kept": self.max_images}}
            )
            filtered = filtered[:self.max_images]

        return filtered

    def extract_images(self, path_minio: str) -> list[ExtractedImage]:
        logger.info(
            "Fetching PDF from MinIO for image extraction",
            extra={"extra_data": {"path_minio": path_minio}}
        )

        try:
            response = self.minio_client.get_object(bucket_name=self.bucket_name, object_name=path_minio) #se aduce un stream cu pdf ul
            pdf_bytes = response.read() #se descarca in memorie (se citeste streamul)
        except S3Error as e:
            error_msg = f"MinIO fetch error: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"path_minio": path_minio}})
            raise RuntimeError(error_msg) from e

        all_images: list[ExtractedImage] = []

        try:
            pdf_stream = io.BytesIO(pdf_bytes) #citeste bytes -> un fisier virtual (nu se mai salveaza pe disk)
            pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")

            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                page_images = self._extract_images_from_page(pdf_doc, page, page_number=page_num + 1)
                all_images.extend(page_images)

            pdf_doc.close()

        except Exception as e:
            error_msg = f"Image extraction failed: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"path_minio": path_minio}})
            raise RuntimeError(error_msg) from e

        filtered_images = self._filter_images(all_images)

        logger.info(
            "Image extraction complete",
            extra={"extra_data": {
                "path_minio": path_minio,
                "total_found": len(all_images),
                "kept_after_filter": len(filtered_images)
            }}
        )

        return filtered_images


_image_extractor = None


def get_image_extractor() -> ImageExtractor:
    global _image_extractor
    if _image_extractor is None:
        _image_extractor = ImageExtractor()
    return _image_extractor