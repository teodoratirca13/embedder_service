"""
Unit tests for fastapi_app.services.image_pipeline.

All collaborators (image extractor, vision captioner, embedder, Qdrant
client, Spring Boot callback client, settings) are mocked at the module
level so the orchestration logic can be tested without any live service.
"""
from unittest.mock import MagicMock

import pytest

import fastapi_app.services.image_pipeline as pipeline
from fastapi_app.models import IngestRequest
from fastapi_app.services.image_extractor import ExtractedImage


@pytest.fixture(autouse=True)
def reset_ingest_versions():
    pipeline._active_ingest_versions.clear()
    yield
    pipeline._active_ingest_versions.clear()


@pytest.fixture
def request_obj():
    return IngestRequest(
        document_id=1,
        course_id=2,
        week_id=3,
        path_minio="course/week3/lecture.pdf",
        document_title="Lecture 3",
        professor_id=9,
    )


@pytest.fixture
def mocks(monkeypatch):
    extractor = MagicMock()
    captioner = MagicMock()
    embedder = MagicMock()
    qdrant = MagicMock()
    callback = MagicMock()
    settings = MagicMock(gemini_request_delay_seconds=0)

    monkeypatch.setattr(pipeline, "get_image_extractor", lambda: extractor)
    monkeypatch.setattr(pipeline, "get_vision_captioner", lambda: captioner)
    monkeypatch.setattr(pipeline, "get_embedding_model", lambda: embedder)
    monkeypatch.setattr(pipeline, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(pipeline, "get_springboot_callback_client", lambda: callback)
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)

    return {
        "extractor": extractor,
        "captioner": captioner,
        "embedder": embedder,
        "qdrant": qdrant,
        "callback": callback,
        "settings": settings,
    }


class TestRegisterAndCheckIngestVersion:

    def test_register_then_check_current(self):
        pipeline.register_ingest_version(document_id=1, ingest_version="v1")
        assert pipeline._is_ingest_version_current(document_id=1, ingest_version="v1") is True

    def test_check_stale_version_returns_false(self):
        pipeline.register_ingest_version(document_id=1, ingest_version="v1")
        pipeline.register_ingest_version(document_id=1, ingest_version="v2")
        assert pipeline._is_ingest_version_current(document_id=1, ingest_version="v1") is False

    def test_check_unknown_document_returns_false(self):
        assert pipeline._is_ingest_version_current(document_id=999, ingest_version="v1") is False


class TestProcessImagesBackground:

    def test_no_images_notifies_indexed_with_zero_counts(self, request_obj, mocks):
        mocks["extractor"].extract_images.return_value = []
        pipeline.register_ingest_version(1, "v1")

        pipeline.process_images_background(request_obj, "v1")

        mocks["captioner"].caption_image.assert_not_called()
        mocks["qdrant"].upsert_chunks.assert_not_called()
        mocks["callback"].notify_image_status.assert_called_once_with(
            document_id=1, status="INDEXED", images_indexed=0, images_failed=0
        )

    def test_all_captions_fail_notifies_failed_images(self, request_obj, mocks):
        mocks["extractor"].extract_images.return_value = [
            ExtractedImage(image_bytes=b"x", mime_type="image/png", page_number=1, width=200, height=200)
        ]
        mocks["captioner"].caption_image.return_value = None
        pipeline.register_ingest_version(1, "v1")

        pipeline.process_images_background(request_obj, "v1")

        mocks["qdrant"].upsert_chunks.assert_not_called()
        mocks["callback"].notify_image_status.assert_called_once_with(
            document_id=1, status="FAILED_IMAGES", images_indexed=0, images_failed=1
        )

    def test_successful_run_upserts_and_notifies_indexed(self, request_obj, mocks):
        images = [
            ExtractedImage(image_bytes=b"a", mime_type="image/png", page_number=1, width=200, height=200),
            ExtractedImage(image_bytes=b"b", mime_type="image/png", page_number=2, width=200, height=200),
        ]
        mocks["extractor"].extract_images.return_value = images
        mocks["captioner"].caption_image.side_effect = ["caption A", "caption B"]
        mocks["embedder"].embed_batch.return_value = [[0.1], [0.2]]
        pipeline.register_ingest_version(1, "v1")

        pipeline.process_images_background(request_obj, "v1")

        mocks["embedder"].embed_batch.assert_called_once_with(["caption A", "caption B"])
        mocks["qdrant"].upsert_chunks.assert_called_once()
        _, kwargs = mocks["qdrant"].upsert_chunks.call_args
        assert kwargs["chunks"] == ["caption A", "caption B"]
        assert kwargs["embeddings"] == [[0.1], [0.2]]
        assert kwargs["source_type"] == "image"
        assert kwargs["page_numbers"] == [1, 2]
        assert kwargs["document_id"] == 1

        mocks["callback"].notify_image_status.assert_called_once_with(
            document_id=1, status="INDEXED", images_indexed=2, images_failed=0
        )

    def test_partial_caption_failure_counts_correctly(self, request_obj, mocks):
        images = [
            ExtractedImage(image_bytes=b"a", mime_type="image/png", page_number=1, width=200, height=200),
            ExtractedImage(image_bytes=b"b", mime_type="image/png", page_number=2, width=200, height=200),
        ]
        mocks["extractor"].extract_images.return_value = images
        mocks["captioner"].caption_image.side_effect = ["caption A", None]
        mocks["embedder"].embed_batch.return_value = [[0.1]]
        pipeline.register_ingest_version(1, "v1")

        pipeline.process_images_background(request_obj, "v1")

        _, kwargs = mocks["qdrant"].upsert_chunks.call_args
        assert kwargs["chunks"] == ["caption A"]
        assert kwargs["page_numbers"] == [1]
        mocks["callback"].notify_image_status.assert_called_once_with(
            document_id=1, status="INDEXED", images_indexed=1, images_failed=1
        )

    def test_stale_ingest_version_discards_results_without_notifying(self, request_obj, mocks):
        images = [ExtractedImage(image_bytes=b"a", mime_type="image/png", page_number=1, width=200, height=200)]
        mocks["extractor"].extract_images.return_value = images
        mocks["captioner"].caption_image.return_value = "a caption"
        mocks["embedder"].embed_batch.return_value = [[0.1]]

        # Simulate a newer reindex having registered a different version.
        pipeline.register_ingest_version(1, "v2")

        pipeline.process_images_background(request_obj, "v1")

        mocks["qdrant"].upsert_chunks.assert_not_called()
        mocks["callback"].notify_image_status.assert_not_called()

    def test_unexpected_exception_is_caught_and_reported(self, request_obj, mocks):
        mocks["extractor"].extract_images.side_effect = RuntimeError("MinIO down")
        pipeline.register_ingest_version(1, "v1")

        # Should not raise.
        pipeline.process_images_background(request_obj, "v1")

        mocks["callback"].notify_image_status.assert_called_once_with(
            document_id=1, status="FAILED_IMAGES", images_indexed=0, images_failed=0
        )

    def test_callback_not_configured_does_not_raise(self, request_obj, mocks):
        mocks["extractor"].extract_images.return_value = []
        mocks["callback"].notify_image_status.side_effect = RuntimeError("not configured")
        pipeline.register_ingest_version(1, "v1")

        # Should not propagate the RuntimeError from the callback client.
        pipeline.process_images_background(request_obj, "v1")
