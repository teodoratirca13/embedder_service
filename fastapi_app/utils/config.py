from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    # Qdrant
    qdrant_url: str = "http://localhost:6333" #sunt valori implicite

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"

    # Model
    bge_model_path: str = "BAAI/bge-m3"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache() #pentru a nu executa functia de fiecare data
def get_settings() -> Settings:
    return Settings()