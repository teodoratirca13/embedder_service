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

    # Gemini Vision (image captioning)
    gemini_api_key: str = ""
    gemini_vision_model: str = "gemini-3.1-flash-lite"
    gemini_request_delay_seconds: float = 13.0

    min_image_width: int = 100
    min_image_height: int = 100
    max_images_per_document: int = 20

    # Spring Boot callback (async image job status)
    spring_boot_callback_url: str = ""
    spring_boot_callback_username: str = ""
    spring_boot_callback_password: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  #pentru RAG_SERVICE_USERNAME si RAG_SERVICE_PASSWORD

@lru_cache() #pentru a nu executa functia de fiecare data
def get_settings() -> Settings:
    return Settings()