"""Unit tests for the Pydantic request/response models in fastapi_app.models."""
import pytest
from pydantic import ValidationError

from fastapi_app.models import (
    DeleteResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)


class TestIngestRequest:

    def test_valid_payload_parses(self):
        req = IngestRequest(
            document_id=123, course_id=5, week_id=15,
            path_minio="algorithms/week3/lecture.pdf",
            document_title="Lecture 3", professor_id=7,
        )
        assert req.document_id == 123
        assert req.path_minio == "algorithms/week3/lecture.pdf"

    @pytest.mark.parametrize("missing_field", [
        "document_id", "course_id", "week_id", "path_minio", "document_title", "professor_id",
    ])
    def test_missing_required_field_raises(self, missing_field):
        payload = {
            "document_id": 1, "course_id": 1, "week_id": 1,
            "path_minio": "x.pdf", "document_title": "T", "professor_id": 1,
        }
        del payload[missing_field]
        with pytest.raises(ValidationError):
            IngestRequest(**payload)

    def test_non_integer_document_id_raises(self):
        with pytest.raises(ValidationError):
            IngestRequest(
                document_id="not-an-int", course_id=1, week_id=1,
                path_minio="x.pdf", document_title="T", professor_id=1,
            )


class TestIngestResponse:

    def test_minimal_response_only_requires_document_id_status_processing_time(self):
        resp = IngestResponse(document_id=1, status="FAILED", processing_time_ms=10)
        assert resp.chunks_count is None
        assert resp.images_queued is None
        assert resp.error is None

    def test_full_response_round_trips(self):
        resp = IngestResponse(
            document_id=1, status="INDEXED_TEXT_ONLY", chunks_count=42,
            images_queued=None, error=None, processing_time_ms=3500,
        )
        assert resp.chunks_count == 42


class TestEmbedRequestResponse:

    def test_embed_request_requires_text(self):
        with pytest.raises(ValidationError):
            EmbedRequest()

    def test_embed_response_shape(self):
        resp = EmbedResponse(embedding=[0.1, 0.2, 0.3], dimension=3, model="BAAI/bge-m3")
        assert len(resp.embedding) == 3
        assert resp.dimension == 3


class TestHealthResponse:

    def test_valid_statuses(self):
        for status in ("ok", "degraded", "error"):
            resp = HealthResponse(status=status, model_loaded=True, qdrant_connected=True)
            assert resp.status == status


class TestDeleteResponse:

    def test_success_response_has_no_error(self):
        resp = DeleteResponse(document_id=1, status="SUCCESS")
        assert resp.error is None

    def test_failed_response_includes_error(self):
        resp = DeleteResponse(document_id=1, status="FAILED", error="not found")
        assert resp.error == "not found"
