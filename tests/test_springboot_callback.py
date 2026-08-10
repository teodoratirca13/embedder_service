"""
Unit tests for fastapi_app.services.springboot_callback.SpringBootCallbackClient.

The httpx.Client is mocked — no real HTTP calls happen, and retry delays are
patched out so the retry tests run instantly.
"""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from fastapi_app.services.springboot_callback import MAX_RETRIES, SpringBootCallbackClient


class TestInit:

    def test_missing_callback_url_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="SPRING_BOOT_CALLBACK_URL"):
            SpringBootCallbackClient(callback_url="", username="u", password="p")

    def test_valid_url_constructs_client(self):
        client = SpringBootCallbackClient(
            callback_url="http://springboot:8080/image-status", username="u", password="p"
        )
        assert client.callback_url == "http://springboot:8080/image-status"
        assert client.auth == ("u", "p")


@pytest.fixture
def callback_client():
    client = SpringBootCallbackClient(
        callback_url="http://springboot:8080/image-status", username="u", password="p"
    )
    client.client = MagicMock()
    return client


class TestNotifyImageStatus:

    def test_success_on_first_attempt_returns_true(self, callback_client):
        response = MagicMock()
        response.raise_for_status.return_value = None
        callback_client.client.patch.return_value = response

        result = callback_client.notify_image_status(
            document_id=1, status="INDEXED", images_indexed=3, images_failed=0
        )

        assert result is True
        callback_client.client.patch.assert_called_once()
        _, kwargs = callback_client.client.patch.call_args
        assert kwargs["json"] == {
            "document_id": 1,
            "status": "INDEXED",
            "images_indexed": 3,
            "images_failed": 0,
        }
        assert kwargs["auth"] == ("u", "p")

    def test_retries_then_succeeds(self, callback_client, monkeypatch):
        monkeypatch.setattr("fastapi_app.services.springboot_callback.time.sleep", lambda s: None)

        failing_response = MagicMock()
        failing_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        ok_response = MagicMock()
        ok_response.raise_for_status.return_value = None

        callback_client.client.patch.side_effect = [failing_response, ok_response]

        result = callback_client.notify_image_status(
            document_id=1, status="INDEXED", images_indexed=1, images_failed=0
        )

        assert result is True
        assert callback_client.client.patch.call_count == 2

    def test_exhausts_retries_and_returns_false(self, callback_client, monkeypatch):
        monkeypatch.setattr("fastapi_app.services.springboot_callback.time.sleep", lambda s: None)
        callback_client.client.patch.side_effect = Exception("connection refused")

        result = callback_client.notify_image_status(
            document_id=1, status="FAILED_IMAGES", images_indexed=0, images_failed=5
        )

        assert result is False
        assert callback_client.client.patch.call_count == MAX_RETRIES

    def test_does_not_raise_on_failure(self, callback_client, monkeypatch):
        monkeypatch.setattr("fastapi_app.services.springboot_callback.time.sleep", lambda s: None)
        callback_client.client.patch.side_effect = Exception("boom")

        # Should not raise — errors are swallowed and reported via return value.
        callback_client.notify_image_status(document_id=1, status="FAILED_IMAGES", images_indexed=0, images_failed=1)


class TestSingleton:
    def test_get_springboot_callback_client_returns_singleton(self, monkeypatch):
        import fastapi_app.services.springboot_callback as mod
        monkeypatch.setattr(mod, "_callback_client", None)
        monkeypatch.setenv("SPRING_BOOT_CALLBACK_URL", "http://springboot:8080/image-status")

        first = mod.get_springboot_callback_client()
        second = mod.get_springboot_callback_client()

        assert first is second
