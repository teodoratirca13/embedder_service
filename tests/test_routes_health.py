"""
Unit tests for GET /api/health.

No auth is required on this endpoint. The embedding model and Qdrant client
getters are mocked at the route-module level (they're imported with
`from fastapi_app.services import ...`, which binds names into
fastapi_app.routes.health's own namespace).
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from fastapi_app.main import app
import fastapi_app.routes.health as health_route


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthCheck:

    def test_ok_when_model_and_qdrant_healthy(self, client, monkeypatch):
        model = MagicMock()
        model.get_embedding_dimension.return_value = 1024
        qdrant = MagicMock()
        qdrant.is_healthy.return_value = True

        monkeypatch.setattr(health_route, "get_embedding_model", lambda: model)
        monkeypatch.setattr(health_route, "get_qdrant_client", lambda: qdrant)

        resp = client.get("/api/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body == {"status": "ok", "model_loaded": True, "qdrant_connected": True}

    def test_degraded_when_only_model_loaded(self, client, monkeypatch):
        model = MagicMock()
        model.get_embedding_dimension.return_value = 1024
        qdrant = MagicMock()
        qdrant.is_healthy.return_value = False

        monkeypatch.setattr(health_route, "get_embedding_model", lambda: model)
        monkeypatch.setattr(health_route, "get_qdrant_client", lambda: qdrant)

        resp = client.get("/api/health")

        assert resp.json()["status"] == "degraded"

    def test_degraded_when_only_qdrant_healthy(self, client, monkeypatch):
        def raise_error():
            raise RuntimeError("model not loaded")

        qdrant = MagicMock()
        qdrant.is_healthy.return_value = True

        monkeypatch.setattr(health_route, "get_embedding_model", raise_error)
        monkeypatch.setattr(health_route, "get_qdrant_client", lambda: qdrant)

        resp = client.get("/api/health")

        assert resp.json()["status"] == "degraded"

    def test_error_when_both_unavailable(self, client, monkeypatch):
        def raise_model_error():
            raise RuntimeError("model not loaded")

        def raise_qdrant_error():
            raise RuntimeError("qdrant unreachable")

        monkeypatch.setattr(health_route, "get_embedding_model", raise_model_error)
        monkeypatch.setattr(health_route, "get_qdrant_client", raise_qdrant_error)

        resp = client.get("/api/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "error", "model_loaded": False, "qdrant_connected": False}

    def test_model_loaded_false_when_dimension_not_1024(self, client, monkeypatch):
        model = MagicMock()
        model.get_embedding_dimension.return_value = 512
        qdrant = MagicMock()
        qdrant.is_healthy.return_value = True

        monkeypatch.setattr(health_route, "get_embedding_model", lambda: model)
        monkeypatch.setattr(health_route, "get_qdrant_client", lambda: qdrant)

        resp = client.get("/api/health")

        assert resp.json()["model_loaded"] is False

    def test_no_auth_required(self, client, monkeypatch):
        model = MagicMock()
        model.get_embedding_dimension.return_value = 1024
        qdrant = MagicMock()
        qdrant.is_healthy.return_value = True
        monkeypatch.setattr(health_route, "get_embedding_model", lambda: model)
        monkeypatch.setattr(health_route, "get_qdrant_client", lambda: qdrant)

        # No Authorization header sent at all.
        resp = client.get("/api/health")
        assert resp.status_code == 200
