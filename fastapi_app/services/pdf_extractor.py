import io
from minio import Minio
from minio.error import S3Error
import fitz  # PyMuPDF
from rapidfuzz import fuzz

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

    def _collect_header_footer_candidates(self, doc, header_ratio: float = 0.10, footer_ratio: float = 0.10) -> list[
        str]:
        """Aduna textele blocurilor din zona de header/footer, de pe toate paginile."""
        candidates = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_height = page.rect.height
            header_limit = page_height * header_ratio
            footer_limit = page_height * (1 - footer_ratio)

            page_dict = page.get_text("dict")

            for block in page_dict["blocks"]:
                if "lines" not in block:
                    continue

                y0 = block["bbox"][1]
                if y0 < header_limit or y0 > footer_limit:
                    block_text = "".join(
                        span["text"] for line in block["lines"] for span in line["spans"]
                    ).strip()
                    if block_text:
                        candidates.append(block_text)

        return candidates

    def _get_confirmed_header_footer_texts(
            self, doc,
            header_ratio: float = 0.10,
            footer_ratio: float = 0.10,
            similarity_threshold: int = 90,
            frequency_threshold: float = 0.5
    ) -> list[str]:
        """Grupeaza candidatii similari si intoarce textele care se repeta suficient de des."""
        candidates = self._collect_header_footer_candidates(doc, header_ratio, footer_ratio)

        clusters = []
        for text in candidates:
            matched = False
            for cluster in clusters:
                if fuzz.ratio(text, cluster["representative"]) >= similarity_threshold:
                    cluster["count"] += 1
                    matched = True
                    break
            if not matched:
                clusters.append({"representative": text, "count": 1})

        min_occurrences = doc.page_count * frequency_threshold
        return [c["representative"] for c in clusters if c["count"] >= min_occurrences]

    def _extract_page_text_deduped(
            self, page, confirmed_texts: list[str], already_included: set,
            header_ratio: float = 0.10, footer_ratio: float = 0.10, similarity_threshold: int = 90
    ) -> str:
        """Extrage textul unei pagini, pastrand fiecare header/footer confirmat o singura data."""
        page_height = page.rect.height
        header_limit = page_height * header_ratio
        footer_limit = page_height * (1 - footer_ratio)

        page_dict = page.get_text("dict")
        page_lines = []

        for block in page_dict["blocks"]:
            if "lines" not in block:
                continue

            block_text = "".join(
                span["text"] for line in block["lines"] for span in line["spans"]
            ).strip()
            if not block_text:
                continue

            y0 = block["bbox"][1]
            is_in_zone = y0 < header_limit or y0 > footer_limit

            if is_in_zone:
                matched = next(
                    (ct for ct in confirmed_texts if fuzz.ratio(block_text, ct) >= similarity_threshold),
                    None
                )
                if matched is not None:
                    if matched in already_included:
                        continue
                    already_included.add(matched)
                    page_lines.append(block_text)
                    continue

            page_lines.append(block_text)

        return "\n".join(page_lines)

    def _extract_page_text(self, page, header_ratio=0.10, footer_ratio=0.10):

        page_height = page.rect.height
        header_limit = page_height * header_ratio
        footer_limit = page_height * (1 - footer_ratio)

        page_dict = page.get_text("dict")
        kept_lines = []

        for block in page_dict["blocks"]:
            if "lines" not in block:
                continue

            y0 = block["bbox"][1]

            if y0 < header_limit or y0 > footer_limit:
                continue

            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text = line_text + span["text"]
                kept_lines.append(line_text)
        return "\n".join(kept_lines)


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

            confirmed_texts = self._get_confirmed_header_footer_texts(pdf_doc)
            already_included = set()

            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                raw_page_text = page.get_text()  # neschimbat, pentru verificarea "pagina doar-imagine"

                if len(raw_page_text.strip()) < 50:
                    image_only_pages.append(page_num + 1)
                    logger.warning(
                        "Skipping image-only page",
                        extra={"extra_data": {"path_minio": path_minio, "page": page_num + 1}}
                    )
                else:
                    page_text = self._extract_page_text_deduped(page, confirmed_texts, already_included)
                    extracted_text += page_text + "\n"

            # Save page_count BEFORE closing the document —
            # pdf_doc.page_count is not accessible after close()
            total_pages = pdf_doc.page_count

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
                    "pages": total_pages,
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