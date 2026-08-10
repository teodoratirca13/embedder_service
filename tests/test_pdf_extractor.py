"""
Unit tests for fastapi_app.services.pdf_extractor.PDFExtractor.

MinIO is mocked (no live object storage needed); the PDFs themselves are
real, built in-memory with PyMuPDF, so the text/header-footer extraction
logic runs against actual PDF structure.
"""
import io
from unittest.mock import MagicMock

import pytest
from minio.error import S3Error

from fastapi_app.services.pdf_extractor import PDFExtractor


@pytest.fixture
def extractor(monkeypatch):
    """A PDFExtractor with a mocked MinIO client (no real network calls)."""
    monkeypatch.setenv("MINIO_ENDPOINT", "fake-minio:9000")
    ext = PDFExtractor()
    ext.minio_client = MagicMock()
    return ext


def _mock_minio_response(pdf_bytes: bytes) -> MagicMock:
    response = MagicMock()
    response.read.return_value = pdf_bytes
    return response


class TestCleanText:
    """_clean_text is a pure string transform — test it directly."""

    def setup_method(self):
        # _clean_text doesn't touch MinIO/settings, so a bare instance is fine
        self.extractor = PDFExtractor.__new__(PDFExtractor)

    def test_collapses_excess_blank_lines(self):
        text = "Paragraph one.\n\n\n\n\nParagraph two."
        cleaned = self.extractor._clean_text(text)
        assert "\n\n\n" not in cleaned
        assert "Paragraph one." in cleaned
        assert "Paragraph two." in cleaned

    def test_removes_standalone_page_number_lines(self):
        text = "Some heading\n\n42\n\nMore content here."
        cleaned = self.extractor._clean_text(text)
        assert "42" not in cleaned.split("\n")

    def test_normalizes_internal_whitespace(self):
        text = "Word1    Word2\t\tWord3"
        cleaned = self.extractor._clean_text(text)
        assert cleaned == "Word1 Word2 Word3"

    def test_strips_paragraph_whitespace(self):
        text = "   Leading and trailing spaces around a paragraph.   \n\n  Second paragraph.  "
        cleaned = self.extractor._clean_text(text)
        paragraphs = cleaned.split("\n\n")
        assert all(p == p.strip() for p in paragraphs)

    def test_empty_text_stays_empty(self):
        assert self.extractor._clean_text("") == ""


class TestExtractText:

    def test_fetches_from_correct_bucket_and_path(self, extractor, make_pdf_bytes):
        pdf_bytes = make_pdf_bytes(n_pages=1)
        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)

        extractor.extract_text("courses/week1/lecture.pdf")

        extractor.minio_client.get_object.assert_called_once_with(
            bucket_name=extractor.bucket_name,
            object_name="courses/week1/lecture.pdf",
        )

    def test_extracts_body_text(self, extractor, make_pdf_bytes):
        body_text = "Dynamic programming explained clearly for students, with enough length to pass the extractor's minimum text threshold."
        # Wide page so PyMuPDF doesn't clip this single-line string at the page edge.
        pdf_bytes = make_pdf_bytes(n_pages=1, body=lambda i: body_text, width=2000)
        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)

        text = extractor.extract_text("doc.pdf")

        assert body_text in text

    def test_deduplicates_repeated_header_across_pages(self, extractor, make_pdf_bytes):
        header = "Course ABC - Repeated Header Block"
        pdf_bytes = make_pdf_bytes(
            n_pages=3,
            header=header,
            body=lambda i: f"Unique body content specific to page number {i} of the lecture notes here.",
        )
        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)

        text = extractor.extract_text("doc.pdf")

        # The confirmed repeated header should appear only once in the output,
        # even though it was present on every page of the source PDF.
        assert text.count(header) == 1
        # But the unique body text from every page should still be present.
        for i in (1, 2, 3):
            assert f"page number {i}" in text

    def test_deduplicates_repeated_footer_across_pages(self, extractor, make_pdf_bytes):
        footer = "Confidential - do not distribute"
        pdf_bytes = make_pdf_bytes(
            n_pages=3,
            footer=footer,
            body=lambda i: f"Unique body content specific to page number {i} of the lecture notes here.",
        )
        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)

        text = extractor.extract_text("doc.pdf")

        assert text.count(footer) == 1

    def test_does_not_deduplicate_non_repeated_header_like_text(self, extractor, make_pdf_bytes):
        # Different "header" text on each page shouldn't be recognized as a
        # repeated header/footer — each occurrence is unique, so none should
        # cross the frequency_threshold and all should be kept.
        pdf_bytes = make_pdf_bytes(
            n_pages=3,
            header=None,
            body=lambda i: f"Unique body content specific to page number {i} of the lecture notes here.",
        )
        # Manually vary a header-zone line per page instead of using the shared header
        import fitz
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        doc.close()

        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)
        text = extractor.extract_text("doc.pdf")
        for i in (1, 2, 3):
            assert f"page number {i}" in text

    def test_minio_fetch_error_raises_runtime_error(self, extractor):
        extractor.minio_client.get_object.side_effect = S3Error(
            code="NoSuchKey", message="not found", resource="doc.pdf",
            request_id="req1", host_id="host1", response=None
        )

        with pytest.raises(RuntimeError, match="MinIO fetch error"):
            extractor.extract_text("missing.pdf")

    def test_image_only_pdf_raises_runtime_error(self, extractor, make_pdf_bytes):
        # A PDF with no text at all (no body inserted) should be treated as
        # image-only and rejected.
        pdf_bytes = make_pdf_bytes(n_pages=1, body=None)
        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)

        with pytest.raises(RuntimeError, match="No text found"):
            extractor.extract_text("image_only.pdf")

    def test_corrupt_pdf_bytes_raise_runtime_error(self, extractor):
        extractor.minio_client.get_object.return_value = _mock_minio_response(b"not a real pdf")

        with pytest.raises(RuntimeError):
            extractor.extract_text("corrupt.pdf")


class TestSingleton:
    def test_get_pdf_extractor_returns_singleton(self, monkeypatch):
        import fastapi_app.services.pdf_extractor as mod
        monkeypatch.setattr(mod, "_extractor", None)

        first = mod.get_pdf_extractor()
        second = mod.get_pdf_extractor()

        assert first is second
