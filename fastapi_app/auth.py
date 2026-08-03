import os
import secrets
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

load_dotenv()

security = HTTPBasic()

RAG_SERVICE_USERNAME = os.getenv("RAG_SERVICE_USERNAME")
RAG_SERVICE_PASSWORD = os.getenv("RAG_SERVICE_PASSWORD")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:  #dependency injection
    if not RAG_SERVICE_USERNAME or not RAG_SERVICE_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Autentificarea nu este configurată pe server (lipsesc variabilele de mediu).",
        )

    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"), RAG_SERVICE_USERNAME.encode("utf-8")
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"), RAG_SERVICE_PASSWORD.encode("utf-8")
    )

    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credențiale invalide",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username