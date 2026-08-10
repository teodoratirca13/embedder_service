"""
Unit tests for fastapi_app.middleware.

_safe() (payload/response redaction) is tested directly as a pure function.
The request_context middleware itself is exercised end-to-end through the
FastAPI app via TestClient, on the unauthenticated /api/health endpoint.
"""
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import fastapi_app.routes.health as health_route
from fastapi_app.main import app
from fastapi_app.middleware import HEADER, _safe


@pytest.fixture(autouse=True)
def mock_health_dependencies(monkeypatch):
    # The middleware tests exercise a real endpoint end-to-end, but the
    # underlying embedding model / Qdrant client stay mocked so no heavy
    # model load or live infra is required just to check header handling.
    model = MagicMock()
    model.get_embedding_dimension.return_value = 1024
    qdrant = MagicMock()
    qdrant.is_healthy.return_value = True
    monkeypatch.setattr(health_route, "get_embedding_model", lambda: model)
    monkeypatch.setattr(health_route, "get_qdrant_client", lambda: qdrant)


class TestSafeRedaction:

    def test_empty_bytes_returns_none(self):
        assert _safe(b"") is None

    def test_redacts_password_field(self):
        raw = json.dumps({"username": "bob", "password": "hunter2"}).encode()
        result = json.loads(_safe(raw))
        assert result["password"] == "***"
        assert result["username"] == "bob"

    def test_redacts_case_insensitively(self):
        raw = json.dumps({"Token": "abc123"}).encode()
        result = json.loads(_safe(raw))
        assert result["Token"] == "***"

    def test_redacts_all_sensitive_keys(self):
        raw = json.dumps({
            "parola": "x", "secret": "x", "access_token": "x", "authorization": "x",
        }).encode()
        result = json.loads(_safe(raw))
        assert all(v == "***" for v in result.values())

    def test_non_json_bytes_returned_as_text(self):
        result = _safe(b"not json at all")
        assert result == "not json at all"

    def test_non_dict_json_passed_through(self):
        raw = json.dumps([1, 2, 3]).encode()
        result = _safe(raw)
        assert json.loads(result) == [1, 2, 3]

    def test_truncates_long_payloads(self):
        raw = json.dumps({"data": "x" * 5000}).encode()
        result = _safe(raw)
        assert len(result) <= 2001  # MAX_LEN + ellipsis char
        assert result.endswith("…")


class TestRequestContextMiddleware:

    def test_generates_request_id_when_absent(self):
        client = TestClient(app)
        resp = client.get("/api/health")
        assert HEADER in resp.headers
        assert len(resp.headers[HEADER]) > 0

    def test_echoes_provided_request_id(self):
        client = TestClient(app)
        resp = client.get("/api/health", headers={HEADER: "my-custom-request-id"})
        assert resp.headers[HEADER] == "my-custom-request-id"
