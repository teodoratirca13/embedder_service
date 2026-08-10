import pytest
from fastapi_app.services.chunker import TextChunker
from fastapi_app.services.embedder import EmbeddingModel
from fastapi_app.utils.config import get_settings


# ============================================================================
# CHUNKER TESTS
# ============================================================================

class TestChunker:

    def setup_method(self):
        self.chunker = TextChunker(
            chunk_size=800,
            chunk_overlap=150,
            min_chunk_length=50
        )

    def test_chunker_splits_long_text(self):
        #Long text should be split into multiple chunks
        text = "This is a test sentence. " * 100  # ~2500 chars
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_chunker_respects_chunk_size(self):
        #No chunk should exceed chunk_size significantly
        text = "This is a test sentence. " * 100
        chunks = self.chunker.chunk_text(text)
        for chunk in chunks:
            assert len(chunk) <= 900, f"Chunk too large: {len(chunk)} chars"

    def test_chunker_filters_short_chunks(self):
        #Chunks shorter than min_chunk_length should be filtered out
        text = "Hi"  # Very short, below min_chunk_length=50
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) == 0, "Short text should produce no chunks"

    def test_chunker_short_text_single_chunk(self):
        #Text slightly above min_chunk_length should produce one chunk
        text = "This is a test sentence that is long enough to be a chunk." * 2
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) >= 1, "Should produce at least one chunk"

    def test_chunker_preserves_content(self):
        #Chunker should not lose content
        text = "Dynamic programming " * 50
        chunks = self.chunker.chunk_text(text)
        combined = " ".join(chunks)
        assert "Dynamic programming" in combined, "Content should be preserved"

    def test_chunker_empty_text(self):
        #Empty text should produce no chunks
        chunks = self.chunker.chunk_text("")
        assert len(chunks) == 0, "Empty text should produce no chunks"

    def test_get_text_chunker_returns_singleton(self):
        from fastapi_app.services.chunker import get_text_chunker
        first = get_text_chunker()
        second = get_text_chunker()
        assert first is second


# ============================================================================
# EMBEDDER TESTS
# ============================================================================

class TestEmbedder:

    def setup_method(self):
        #Load the embedding model once per test
        settings = get_settings()
        self.model = EmbeddingModel(model_path=settings.bge_model_path)

    def test_embed_text_returns_correct_dimension(self):
        #Embedding should always be 1024-dimensional
        vector = self.model.embed_text("Test text")
        assert len(vector) == 1024, f"Expected 1024 dims, got {len(vector)}"

    def test_embed_text_returns_list_of_floats(self):
        #Embedding should be a list of floats
        vector = self.model.embed_text("Test text")
        assert isinstance(vector, list), "Should return a list"
        assert all(isinstance(v, float) for v in vector), "All values should be floats"

    def test_embed_text_romanian(self):
        #Model should handle Romanian text
        vector = self.model.embed_text("Explică programarea dinamică")
        assert len(vector) == 1024, "Romanian text should embed correctly"

    def test_embed_text_english(self):
        #Model should handle English text
        vector = self.model.embed_text("What is dynamic programming?")
        assert len(vector) == 1024, "English text should embed correctly"

    def test_embed_batch_returns_correct_count(self):
        #Batch embedding should return one vector per input text
        texts = ["First text", "Second text", "Third text"]
        vectors = self.model.embed_batch(texts)
        assert len(vectors) == len(texts), "Should return one vector per text"

    def test_embed_batch_correct_dimensions(self):
        #Each vector in batch should be 1024-dimensional
        texts = ["First text", "Second text"]
        vectors = self.model.embed_batch(texts)
        for i, vector in enumerate(vectors):
            assert len(vector) == 1024, f"Vector {i} has wrong dimension: {len(vector)}"

    def test_similar_texts_similar_vectors(self):
        #Similar texts should produce similar vectors
        vector1 = self.model.embed_text("Dynamic programming optimization")
        vector2 = self.model.embed_text("Dynamic programming technique")
        vector3 = self.model.embed_text("Banana fruit smoothie recipe")

        # Calculate dot product as similarity
        similarity_12 = sum(a * b for a, b in zip(vector1, vector2))
        similarity_13 = sum(a * b for a, b in zip(vector1, vector3))

        assert similarity_12 > similarity_13, \
            "Similar texts should be more similar than unrelated texts"

    def test_get_embedding_dimension(self):
        #Model should report correct dimension
        assert self.model.get_embedding_dimension() == 1024

    def test_get_embedding_model_returns_singleton(self):
        from fastapi_app.services.embedder import get_embedding_model
        first = get_embedding_model()
        second = get_embedding_model()
        assert first is second


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestConfig:

    def test_settings_loads(self):
        #Settings should load without errors
        settings = get_settings()
        assert settings is not None

    def test_settings_has_required_fields(self):
        #All required settings should be present
        settings = get_settings()
        assert settings.qdrant_url is not None
        assert settings.minio_endpoint is not None
        assert settings.bge_model_path is not None
        assert settings.minio_bucket is not None

    def test_settings_has_image_pipeline_fields(self):
        #Fields added for the async image captioning pipeline should be present
        settings = get_settings()
        assert settings.gemini_api_key is not None
        assert settings.gemini_vision_model is not None
        assert settings.gemini_request_delay_seconds is not None
        assert settings.min_image_width is not None
        assert settings.min_image_height is not None
        assert settings.max_images_per_document is not None

    def test_settings_has_springboot_callback_fields(self):
        #Fields added for the Spring Boot image-status callback should be present
        settings = get_settings()
        assert settings.spring_boot_callback_url is not None
        assert settings.spring_boot_callback_username is not None
        assert settings.spring_boot_callback_password is not None

    def test_settings_default_values(self, monkeypatch, tmp_path):
        #Default values (no .env, no env vars) should match the hardcoded defaults in Settings.
        # get_settings() is lru_cached and reads the real .env, so defaults are
        # verified against a bare Settings() instantiated with no .env file
        # present (env_file resolution is relative to the current working
        # directory) and no relevant environment variables set.
        for var in (
            "QDRANT_URL", "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY",
            "MINIO_BUCKET", "BGE_MODEL_PATH", "LOG_LEVEL", "GEMINI_API_KEY",
            "GEMINI_VISION_MODEL", "GEMINI_REQUEST_DELAY_SECONDS", "MIN_IMAGE_WIDTH",
            "MIN_IMAGE_HEIGHT", "MAX_IMAGES_PER_DOCUMENT", "SPRING_BOOT_CALLBACK_URL",
            "SPRING_BOOT_CALLBACK_USERNAME", "SPRING_BOOT_CALLBACK_PASSWORD",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.chdir(tmp_path)  # no .env file here

        from fastapi_app.utils.config import Settings
        settings = Settings()

        assert settings.qdrant_url == "http://localhost:6333"
        assert settings.minio_endpoint == "localhost:9000"
        assert settings.minio_bucket == "documents"
        assert settings.bge_model_path == "BAAI/bge-m3"
        assert settings.log_level == "INFO"
        assert settings.gemini_vision_model == "gemini-3.1-flash-lite"
        assert settings.gemini_request_delay_seconds == 13.0
        assert settings.min_image_width == 100
        assert settings.min_image_height == 100
        assert settings.max_images_per_document == 20
        assert settings.spring_boot_callback_url == ""

    def test_get_settings_returns_cached_singleton(self):
        #get_settings is @lru_cache'd — repeated calls return the same instance
        assert get_settings() is get_settings()

    def test_env_vars_override_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MINIO_BUCKET", "custom-bucket")
        monkeypatch.setenv("MAX_IMAGES_PER_DOCUMENT", "5")

        from fastapi_app.utils.config import Settings
        settings = Settings()

        assert settings.minio_bucket == "custom-bucket"
        assert settings.max_images_per_document == 5

    def test_unknown_env_vars_are_ignored(self, monkeypatch, tmp_path):
        # extra = "ignore" — RAG_SERVICE_USERNAME/PASSWORD and other unrelated
        # env vars must not raise a validation error.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("RAG_SERVICE_USERNAME", "someone")
        monkeypatch.setenv("SOME_UNRELATED_VAR", "value")

        from fastapi_app.utils.config import Settings
        Settings()  # should not raise