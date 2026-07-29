import os
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

RAG_USERNAME = os.getenv("RAG_SERVICE_USERNAME", "akadion-spring-backend")
RAG_PASSWORD = os.getenv("RAG_SERVICE_PASSWORD", "parola_spring_rag")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    user_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"), RAG_USERNAME.encode("utf-8")
    )
    pass_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"), RAG_PASSWORD.encode("utf-8")
    )
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credentiale invalide",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
