"""
Unit tests for POST /api/documents/ingest and DELETE /api/documents/ingest/{id}.

Auth is bypassed via dependency_overrides. All services used by the route
(pdf extractor, chunker, embedder, Qdrant client, background image
pipeline) are mocked at the route-module level, since documents.py imports
them by name (`from fastapi_app.services import ...`).
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import fastapi_app.routes.documents as documents_route
from fastapi_app.auth import verify_credentials
from fastapi_app.main import app

VALID_PAYLOAD = {
    "document_id": 123,
    "course_id": 5,
    "week_id": 15,
    "path_minio": "algorithms/week3/lecture.pdf",
    "document_title": "Lecture 3",
    "professor_id": 7,
}


@pytest.fixture
def client():
    app.dependency_overrides[verify_credentials] = lambda: "test_user"
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def happy_path_mocks(monkeypatch):
    """Wires up documents_route's service getters for a successful ingest."""
    pdf_extractor = MagicMock()
    pdf_extractor.extract_text.return_value = "Extracted document text " * 20

    chunker = MagicMock()
    chunker.chunk_text.return_value = ["chunk one", "chunk two", "chunk three"]

    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1], [0.2], [0.3]]

    qdrant = MagicMock()

    background_calls = []

    def fake_add_task(request, ingest_version):
        background_calls.append((request, ingest_version))

    monkeypatch.setattr(documents_route, "get_pdf_extractor", lambda: pdf_extractor)
    monkeypatch.setattr(documents_route, "get_text_chunker", lambda: chunker)
    monkeypatch.setattr(documents_route, "get_embedding_model", lambda: embedder)
    monkeypatch.setattr(documents_route, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(documents_route, "process_images_background", fake_add_task)

    return {
        "pdf_extractor": pdf_extractor,
        "chunker": chunker,
        "embedder": embedder,
        "qdrant": qdrant,
        "background_calls": background_calls,
    }


class TestIngestDocumentSuccess:

    def test_returns_indexed_text_only_with_chunk_count(self, client, happy_path_mocks):
        resp = client.post("/api/documents/ingest", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert body["document_id"] == 123
        assert body["status"] == "INDEXED_TEXT_ONLY"
        assert body["chunks_count"] == 3
        assert body["error"] is None
        assert body["processing_time_ms"] >= 0

    def test_deletes_old_chunks_before_upserting(self, client, happy_path_mocks):
        client.post("/api/documents/ingest", json=VALID_PAYLOAD)

        happy_path_mocks["qdrant"].delete_document_chunks.assert_called_once_with(123)
        happy_path_mocks["qdrant"].upsert_chunks.assert_called_once()

    def test_upserts_with_request_metadata(self, client, happy_path_mocks):
        client.post("/api/documents/ingest", json=VALID_PAYLOAD)

        _, kwargs = happy_path_mocks["qdrant"].upsert_chunks.call_args
        assert kwargs["document_id"] == 123
        assert kwargs["course_id"] == 5
        assert kwargs["week_id"] == 15
        assert kwargs["document_title"] == "Lecture 3"
        assert kwargs["professor_id"] == 7
        assert kwargs["chunks"] == ["chunk one", "chunk two", "chunk three"]

    def test_schedules_background_image_processing(self, client, happy_path_mocks):
        resp = client.post("/api/documents/ingest", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        # process_images_background is invoked as a BackgroundTask after the
        # response is produced; TestClient runs background tasks synchronously.
        assert len(happy_path_mocks["background_calls"]) == 1
        called_request, ingest_version = happy_path_mocks["background_calls"][0]
        assert called_request.document_id == 123
        assert isinstance(ingest_version, str) and ingest_version


class TestIngestDocumentFailures:

    def test_no_chunks_produced_returns_failed(self, client, happy_path_mocks):
        happy_path_mocks["chunker"].chunk_text.return_value = []

        resp = client.post("/api/documents/ingest", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "FAILED"
        assert "No chunks produced" in body["error"]

    def test_embedding_count_mismatch_returns_failed(self, client, happy_path_mocks):
        happy_path_mocks["embedder"].embed_batch.return_value = [[0.1]]  # only 1, but 3 chunks

        resp = client.post("/api/documents/ingest", json=VALID_PAYLOAD)

        body = resp.json()
        assert body["status"] == "FAILED"
        assert "mismatch" in body["error"].lower()

    def test_pdf_extraction_error_returns_failed(self, client, happy_path_mocks):
        happy_path_mocks["pdf_extractor"].extract_text.side_effect = RuntimeError("MinIO fetch error: not found")

        resp = client.post("/api/documents/ingest", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "FAILED"
        assert "MinIO fetch error" in body["error"]

    def test_qdrant_upsert_error_returns_failed(self, client, happy_path_mocks):
        happy_path_mocks["qdrant"].upsert_chunks.side_effect = RuntimeError("Failed to upsert chunks: down")

        resp = client.post("/api/documents/ingest", json=VALID_PAYLOAD)

        body = resp.json()
        assert body["status"] == "FAILED"

    def test_failure_does_not_schedule_background_images(self, client, happy_path_mocks):
        happy_path_mocks["chunker"].chunk_text.return_value = []

        client.post("/api/documents/ingest", json=VALID_PAYLOAD)

        assert happy_path_mocks["background_calls"] == []

    def test_missing_required_field_returns_422(self, client, happy_path_mocks):
        payload = dict(VALID_PAYLOAD)
        del payload["document_id"]

        resp = client.post("/api/documents/ingest", json=payload)

        assert resp.status_code == 422


class TestIngestDocumentAuth:

    def test_requires_auth_when_not_overridden(self):
        real_client = TestClient(app)
        resp = real_client.post("/api/documents/ingest", json=VALID_PAYLOAD)
        assert resp.status_code == 401


class TestDeleteDocument:

    def test_delete_success(self, client, monkeypatch):
        qdrant = MagicMock()
        monkeypatch.setattr(documents_route, "get_qdrant_client", lambda: qdrant)

        resp = client.delete("/api/documents/ingest/123")

        assert resp.status_code == 200
        body = resp.json()
        assert body == {"document_id": 123, "status": "SUCCESS", "error": None}
        qdrant.delete_document_chunks.assert_called_once_with(123)

    def test_delete_failure_returns_failed_status(self, client, monkeypatch):
        qdrant = MagicMock()
        qdrant.delete_document_chunks.side_effect = RuntimeError("Qdrant unreachable")
        monkeypatch.setattr(documents_route, "get_qdrant_client", lambda: qdrant)

        resp = client.delete("/api/documents/ingest/123")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "FAILED"
        assert "Qdrant unreachable" in body["error"]

    def test_delete_is_idempotent_for_nonexistent_document(self, client, monkeypatch):
        # Qdrant's delete-by-filter succeeds silently even with no matches —
        # the route reports SUCCESS regardless.
        qdrant = MagicMock()
        monkeypatch.setattr(documents_route, "get_qdrant_client", lambda: qdrant)

        resp = client.delete("/api/documents/ingest/999999")

        assert resp.status_code == 200
        assert resp.json()["status"] == "SUCCESS"

    def test_delete_non_integer_id_returns_422(self, client):
        resp = client.delete("/api/documents/ingest/not-an-int")
        assert resp.status_code == 422

    def test_delete_requires_auth_when_not_overridden(self):
        real_client = TestClient(app)
        resp = real_client.delete("/api/documents/ingest/123")
        assert resp.status_code == 401
