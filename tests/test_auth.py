"""
Unit tests for fastapi_app.auth.verify_credentials.

fastapi_app.auth reads RAG_SERVICE_USERNAME/PASSWORD from the environment
once, at import time, into module-level constants — so tests monkeypatch
those constants directly rather than the environment.
"""
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

import fastapi_app.auth as auth_module
from fastapi_app.auth import verify_credentials


@pytest.fixture(autouse=True)
def configured_credentials(monkeypatch):
    monkeypatch.setattr(auth_module, "RAG_SERVICE_USERNAME", "correct_user")  #monkeypatch.setattr() schimbă temporar variabilele din auth.py.
    monkeypatch.setattr(auth_module, "RAG_SERVICE_PASSWORD", "correct_pass")


class TestVerifyCredentials:

    def test_correct_credentials_returns_username(self):
        creds = HTTPBasicCredentials(username="correct_user", password="correct_pass")
        assert verify_credentials(creds) == "correct_user"

    def test_wrong_username_raises_401(self):
        creds = HTTPBasicCredentials(username="wrong_user", password="correct_pass")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds)
        assert exc_info.value.status_code == 401

    def test_wrong_password_raises_401(self):
        creds = HTTPBasicCredentials(username="correct_user", password="wrong_pass")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds)
        assert exc_info.value.status_code == 401

    def test_both_wrong_raises_401(self):
        creds = HTTPBasicCredentials(username="nope", password="nope")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds)
        assert exc_info.value.status_code == 401

    def test_401_includes_www_authenticate_header(self):
        creds = HTTPBasicCredentials(username="nope", password="nope")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds)
        assert exc_info.value.headers.get("WWW-Authenticate") == "Basic"


class TestServerMisconfiguration:

    def test_missing_username_config_raises_500(self, monkeypatch):
        monkeypatch.setattr(auth_module, "RAG_SERVICE_USERNAME", None)
        creds = HTTPBasicCredentials(username="any", password="any")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds)
        assert exc_info.value.status_code == 500

    def test_missing_password_config_raises_500(self, monkeypatch):
        monkeypatch.setattr(auth_module, "RAG_SERVICE_PASSWORD", None)
        creds = HTTPBasicCredentials(username="any", password="any")
        with pytest.raises(HTTPException) as exc_info:
            verify_credentials(creds)
        assert exc_info.value.status_code == 500
