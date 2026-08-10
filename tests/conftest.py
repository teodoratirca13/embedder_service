"""Shared pytest fixtures for the embedder_service test suite."""
import fitz
import pytest


@pytest.fixture
def make_pdf_bytes():
    """
    Factory fixture: builds a real in-memory PDF (via PyMuPDF) so extraction
    logic can be exercised against actual PDF structure instead of mocks.

    Usage:
        pdf_bytes = make_pdf_bytes(
            n_pages=3,
            header="Course ABC — Week 3",
            footer="Confidential",
            body=lambda i: f"Unique body text for page {i}. " * 5,
        )
    """

    def _make(
        n_pages: int = 1,
        header: str | None = None,
        footer: str | None = None,
        body=lambda i: (
            f"Body paragraph for page {i}, long enough to not be filtered out by the "
            f"extractor's minimum length checks for a real page of content."
        ),
        width: float = 600,
        height: float = 800,
    ) -> bytes:
        doc = fitz.open()
        for i in range(1, n_pages + 1):
            page = doc.new_page(width=width, height=height)
            if header:
                page.insert_text((50, 30), header)
            body_text = body(i) if callable(body) else body
            if body_text:
                page.insert_text((50, height / 2), body_text)
            if footer:
                page.insert_text((50, height - 30), footer)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    return _make


@pytest.fixture
def make_pdf_with_image():
    """Factory fixture: builds a real in-memory PDF containing one embedded raster image."""

    def _make(width_px: int = 200, height_px: int = 200, page_width: float = 600, page_height: float = 800) -> bytes:
        pix = fitz.Pixmap(fitz.csRGB, (0, 0, width_px, height_px), False)
        pix.set_rect(pix.irect, (255, 0, 0))
        png_bytes = pix.tobytes("png")

        doc = fitz.open()
        page = doc.new_page(width=page_width, height=page_height)
        rect = fitz.Rect(50, 50, 50 + min(width_px, 400), 50 + min(height_px, 400))
        page.insert_image(rect, stream=png_bytes)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    return _make
