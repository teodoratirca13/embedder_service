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

    def test_settings_default_values(self):
        #Default values should be correct
        settings = get_settings()
        assert settings.log_level == "INFO"
        assert settings.minio_bucket == "documents"