"""
Unit tests for POST /api/query/embed.

Auth is bypassed via FastAPI's dependency_overrides (unit tests for the auth
logic itself live in tests/test_auth.py). The embedding model getter is
mocked at the route-module level.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import fastapi_app.routes.query as query_route
from fastapi_app.auth import verify_credentials
from fastapi_app.main import app


@pytest.fixture
def client():
    app.dependency_overrides[verify_credentials] = lambda: "test_user"
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestEmbedQuery:

    def test_returns_embedding_dimension_and_model(self, client, monkeypatch):
        model = MagicMock()
        model.embed_text.return_value = [0.1, 0.2, 0.3]
        model.last_embedding_time_ms = 12.3
        monkeypatch.setattr(query_route, "get_embedding_model", lambda: model)

        resp = client.post("/api/query/embed", json={"text": "Explică programarea dinamică"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["embedding"] == [0.1, 0.2, 0.3]
        assert body["dimension"] == 3
        assert body["model"] == "BAAI/bge-m3"

    def test_passes_request_text_to_embedder(self, client, monkeypatch):
        model = MagicMock()
        model.embed_text.return_value = [0.1]
        model.last_embedding_time_ms = 1.0
        monkeypatch.setattr(query_route, "get_embedding_model", lambda: model)

        client.post("/api/query/embed", json={"text": "What is dynamic programming?"})

        model.embed_text.assert_called_once_with("What is dynamic programming?")

    def test_missing_text_field_returns_422(self, client):
        resp = client.post("/api/query/embed", json={})
        assert resp.status_code == 422

    def test_embedder_failure_returns_500(self, monkeypatch):
        # The route re-raises unhandled exceptions after logging them, relying
        # on FastAPI's default handler to turn them into a 500 response — that
        # only happens with server exceptions disabled on the test client
        # (TestClient re-raises them by default, for easier debugging).
        model = MagicMock()
        model.embed_text.side_effect = RuntimeError("model crashed")
        monkeypatch.setattr(query_route, "get_embedding_model", lambda: model)
        app.dependency_overrides[verify_credentials] = lambda: "test_user"
        try:
            no_raise_client = TestClient(app, raise_server_exceptions=False)
            resp = no_raise_client.post("/api/query/embed", json={"text": "hello"})
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 500


class TestEmbedQueryAuth:

    def test_requires_auth_when_not_overridden(self, monkeypatch):
        # Use a fresh client without the auth override from the `client` fixture.
        real_client = TestClient(app)
        resp = real_client.post("/api/query/embed", json={"text": "hello"})
        assert resp.status_code == 401
