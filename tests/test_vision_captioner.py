"""
Unit tests for fastapi_app.services.vision_captioner.VisionCaptioner.

The Gemini SDK client is mocked — no real API calls (and no billing) happen.
"""
from unittest.mock import MagicMock, patch

import pytest

from fastapi_app.services.vision_captioner import VisionCaptioner


class TestInit:

    def test_missing_api_key_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            VisionCaptioner(api_key="", model="gemini-3.1-flash-lite")

    def test_client_init_failure_raises_runtime_error(self):
        with patch("fastapi_app.services.vision_captioner.genai.Client", side_effect=Exception("bad key")):
            with pytest.raises(RuntimeError, match="Failed to initialize Gemini client"):
                VisionCaptioner(api_key="fake-key", model="gemini-3.1-flash-lite")

    def test_valid_api_key_constructs_client(self):
        with patch("fastapi_app.services.vision_captioner.genai.Client") as mock_client_cls:
            captioner = VisionCaptioner(api_key="fake-key", model="gemini-3.1-flash-lite")
            mock_client_cls.assert_called_once_with(api_key="fake-key")
            assert captioner.model == "gemini-3.1-flash-lite"


@pytest.fixture
def captioner():
    with patch("fastapi_app.services.vision_captioner.genai.Client"):
        return VisionCaptioner(api_key="fake-key", model="gemini-3.1-flash-lite")


class TestCaptionImage:

    def test_returns_caption_text_on_success(self, captioner):
        response = MagicMock()
        response.text = "  O diagrama cu doi vectori.  "
        captioner.client.models.generate_content.return_value = response

        caption = captioner.caption_image(b"fake-bytes", mime_type="image/png")

        assert caption == "O diagrama cu doi vectori."

    def test_passes_image_bytes_and_mime_type(self, captioner):
        response = MagicMock()
        response.text = "caption"
        captioner.client.models.generate_content.return_value = response

        captioner.caption_image(b"raw-bytes", mime_type="image/jpeg")

        _, kwargs = captioner.client.models.generate_content.call_args
        assert kwargs["model"] == "gemini-3.1-flash-lite"

    def test_empty_response_text_returns_none(self, captioner):
        response = MagicMock()
        response.text = "   "
        captioner.client.models.generate_content.return_value = response

        assert captioner.caption_image(b"fake-bytes") is None

    def test_none_response_text_returns_none(self, captioner):
        response = MagicMock()
        response.text = None
        captioner.client.models.generate_content.return_value = response

        assert captioner.caption_image(b"fake-bytes") is None

    def test_api_exception_returns_none_instead_of_raising(self, captioner):
        captioner.client.models.generate_content.side_effect = Exception("Gemini is down")

        assert captioner.caption_image(b"fake-bytes") is None


class TestSingleton:
    def test_get_vision_captioner_returns_singleton(self, monkeypatch):
        import fastapi_app.services.vision_captioner as mod
        monkeypatch.setattr(mod, "_vision_captioner", None)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        with patch("fastapi_app.services.vision_captioner.genai.Client"):
            first = mod.get_vision_captioner()
            second = mod.get_vision_captioner()

        assert first is second
