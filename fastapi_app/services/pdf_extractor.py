import io
from minio import Minio
from minio.error import S3Error
import fitz  # PyMuPDF

from fastapi_app.utils import setup_logger
from fastapi_app.utils import get_settings

logger = setup_logger(__name__)


class PDFExtractor:
    """Extract text from PDF files stored in MinIO."""

    def __init__(self):
        settings = get_settings()  #citeste configuratia

        self.minio_client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False  # pentru productia este True ca sa fie HTTPS
        )
        self.bucket_name = settings.minio_bucket

    def extract_text(self, path_minio: str) -> str:
        """
        Fetch PDF from MinIO and extract text.

        Args:
            path_minio: Path to file in MinIO (e.g., "algorithms/week3/lecture.pdf")

        Returns:
            Extracted text string

        Raises:
            S3Error: If MinIO fetch fails
            RuntimeError: If PDF extraction fails or no text found
        """


        # Step 1: Fetch PDF from MinIO
        logger.info(
            "Fetching PDF from MinIO",
            extra={"extra_data": {"path_minio": path_minio}}
        )

        try:
            response = self.minio_client.get_object(
                bucket_name=self.bucket_name,
                object_name=path_minio
            )
            pdf_bytes = response.read()  # descarca pdf ul
            logger.info(
                "PDF fetched successfully",
                extra={"extra_data": {"path_minio": path_minio, "size_bytes": len(pdf_bytes)}}
            )
        except S3Error as e:
            error_msg = f"MinIO fetch error: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"path_minio": path_minio}})
            raise RuntimeError(error_msg) from e

        # Step 2: Extract text with PyMuPDF
        logger.info("Extracting text from PDF", extra={"extra_data": {"path_minio": path_minio}})

        try:
            pdf_stream = io.BytesIO(pdf_bytes) # transforma bytes in fisier
            pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")

            extracted_text = ""
            image_only_pages = []

            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                page_text = page.get_text()

                # Check if page is mostly empty (image-only)
                if len(page_text.strip()) < 50:
                    image_only_pages.append(page_num + 1)  # 1-indexed for humans
                    logger.warning(
                        "Skipping image-only page",
                        extra={"extra_data": {
                            "path_minio": path_minio,
                            "page": page_num + 1
                        }}
                    )
                else:
                    extracted_text += page_text + "\n"

            pdf_doc.close()

            # Log which pages were skipped
            if image_only_pages:
                logger.info(
                    "Skipped image-only pages",
                    extra={"extra_data": {
                        "path_minio": path_minio,
                        "image_only_pages": image_only_pages
                    }}
                )

            # Check if we got any text at all
            if len(extracted_text.strip()) < 100:
                error_msg = "No text found — may be image-only PDF"
                logger.error(error_msg, extra={"extra_data": {"path_minio": path_minio}})
                raise RuntimeError(error_msg)

            logger.info(
                "Text extracted successfully",
                extra={"extra_data": {
                    "path_minio": path_minio,
                    "pages": pdf_doc.page_count,
                    "chars": len(extracted_text)
                }}
            )

        except RuntimeError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            error_msg = f"PDF parsing error: {str(e)}"
            logger.error(error_msg, extra={"extra_data": {"path_minio": path_minio}})
            raise RuntimeError(error_msg) from e

        # Step 3: Clean text
        logger.info("Cleaning extracted text", extra={"extra_data": {"path_minio": path_minio}})
        cleaned_text = self._clean_text(extracted_text)

        logger.info(
            "Text cleaning complete",
            extra={"extra_data": {
                "path_minio": path_minio,
                "original_chars": len(extracted_text),
                "cleaned_chars": len(cleaned_text)
            }}
        )

        return cleaned_text

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing artifacts.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace (3+ newlines → 2 newlines)
        import re
        text = re.sub(r'\n\n\n+', '\n\n', text)

        # Remove lines that are just page numbers (single number on a line)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # Strip leading/trailing whitespace per paragraph
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        text = '\n\n'.join(paragraphs)

        # Normalize spaces (but preserve paragraph breaks)
        lines = text.split('\n')
        lines = [' '.join(line.split()) for line in lines]
        text = '\n'.join(lines)

        return text


# Creaza un singleton -> se refoloseste pe percurs
_extractor = None


def get_pdf_extractor() -> PDFExtractor:
    """Get or create PDF extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = PDFExtractor()
    return _extractor