"""
Unit tests for fastapi_app.services.image_extractor.ImageExtractor.

MinIO is mocked; PDFs (with real embedded raster images) are built
in-memory with PyMuPDF so the extraction/filtering logic runs for real.
"""
from unittest.mock import MagicMock

import pytest
from minio.error import S3Error

from fastapi_app.services.image_extractor import ExtractedImage, ImageExtractor


@pytest.fixture
def extractor(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "fake-minio:9000")
    monkeypatch.setenv("MIN_IMAGE_WIDTH", "100")
    monkeypatch.setenv("MIN_IMAGE_HEIGHT", "100")
    monkeypatch.setenv("MAX_IMAGES_PER_DOCUMENT", "20")
    ext = ImageExtractor()
    ext.minio_client = MagicMock()  #fara sa vorbim cu un server rea;
    return ext


def _mock_minio_response(pdf_bytes: bytes) -> MagicMock:
    response = MagicMock()
    response.read.return_value = pdf_bytes
    return response


class TestExtractImages:

    def test_extracts_embedded_image(self, extractor, make_pdf_with_image):
        pdf_bytes = make_pdf_with_image(width_px=200, height_px=200)
        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)

        images = extractor.extract_images("doc.pdf")

        assert len(images) == 1
        assert isinstance(images[0], ExtractedImage)
        assert images[0].page_number == 1
        assert images[0].width == 200
        assert images[0].height == 200
        assert images[0].mime_type.startswith("image/")
        assert images[0].image_bytes

    def test_no_images_returns_empty_list(self, extractor, make_pdf_bytes):
        pdf_bytes = make_pdf_bytes(n_pages=1, body="Just text, no images here.")
        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)

        images = extractor.extract_images("doc.pdf")

        assert images == []

    def test_filters_images_smaller_than_min_dimensions(self, extractor, make_pdf_with_image):
        extractor.min_width = 500
        extractor.min_height = 500
        pdf_bytes = make_pdf_with_image(width_px=100, height_px=100)
        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)

        images = extractor.extract_images("doc.pdf")

        assert images == []

    def test_keeps_images_meeting_min_dimensions(self, extractor, make_pdf_with_image):
        extractor.min_width = 50
        extractor.min_height = 50
        pdf_bytes = make_pdf_with_image(width_px=100, height_px=100)
        extractor.minio_client.get_object.return_value = _mock_minio_response(pdf_bytes)

        images = extractor.extract_images("doc.pdf")

        assert len(images) == 1

    def test_truncates_to_max_images_per_document(self, extractor):
        # Directly exercise the private filter to avoid building a PDF with
        # many embedded images.
        extractor.max_images = 2
        extractor.min_width = 0
        extractor.min_height = 0
        images = [
            ExtractedImage(image_bytes=b"x", mime_type="image/png", page_number=i, width=200, height=200)
            for i in range(5)
        ]

        filtered = extractor._filter_images(images)

        assert len(filtered) == 2

    def test_minio_fetch_error_raises_runtime_error(self, extractor):
        extractor.minio_client.get_object.side_effect = S3Error(
            code="NoSuchKey", message="not found", resource="doc.pdf",
            request_id="req1", host_id="host1", response=None
        )

        with pytest.raises(RuntimeError, match="MinIO fetch error"):
            extractor.extract_images("missing.pdf")

    def test_corrupt_pdf_raises_runtime_error(self, extractor):
        extractor.minio_client.get_object.return_value = _mock_minio_response(b"not a real pdf")

        with pytest.raises(RuntimeError, match="Image extraction failed"):
            extractor.extract_images("corrupt.pdf")


class TestSingleton:
    def test_get_image_extractor_returns_singleton(self, monkeypatch):
        import fastapi_app.services.image_extractor as mod
        monkeypatch.setattr(mod, "_image_extractor", None)

        first = mod.get_image_extractor()
        second = mod.get_image_extractor()

        assert first is second
