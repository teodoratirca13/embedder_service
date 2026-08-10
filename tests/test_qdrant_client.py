"""
Unit tests for fastapi_app.services.qdrant_client.QdrantVectorStore.

The underlying qdrant_client.QdrantClient is fully mocked — no live Qdrant
instance is required, and the test verifies the point/payload construction
logic instead of network behavior.
"""
from unittest.mock import MagicMock, call

import pytest

import fastapi_app.services.qdrant_client as qdrant_module
from fastapi_app.services.qdrant_client import COLLECTION_NAME, QdrantVectorStore


@pytest.fixture
def fake_client_cls(monkeypatch):
    """Patches the QdrantClient class used inside qdrant_client.py with a MagicMock factory."""
    instances = []

    def factory(*args, **kwargs):
        mock_instance = MagicMock()
        mock_instance.collection_exists.return_value = True  # skip creation path by default
        instances.append(mock_instance)
        return mock_instance

    fake_cls = MagicMock(side_effect=factory)
    monkeypatch.setattr(qdrant_module, "QdrantClient", fake_cls)
    return fake_cls, instances


class TestInit:

    def test_connects_and_skips_creation_when_collection_exists(self, fake_client_cls):
        fake_cls, instances = fake_client_cls

        store = QdrantVectorStore(qdrant_url="http://fake-qdrant:6333")

        fake_cls.assert_called_once_with(url="http://fake-qdrant:6333")
        store.client.get_collections.assert_called_once()
        store.client.collection_exists.assert_called_once_with(COLLECTION_NAME)
        store.client.create_collection.assert_not_called()

    def test_creates_collection_when_missing(self, fake_client_cls):
        fake_cls, instances = fake_client_cls
        instances_created = []

        def factory(*args, **kwargs):
            mock_instance = MagicMock()
            mock_instance.collection_exists.return_value = False
            instances_created.append(mock_instance)
            return mock_instance

        fake_cls.side_effect = factory

        store = QdrantVectorStore(qdrant_url="http://fake-qdrant:6333")

        store.client.create_collection.assert_called_once()
        _, kwargs = store.client.create_collection.call_args
        assert kwargs["collection_name"] == COLLECTION_NAME

    def test_connection_failure_raises_runtime_error(self, monkeypatch):
        def factory(*args, **kwargs):
            raise ConnectionError("boom")

        monkeypatch.setattr(qdrant_module, "QdrantClient", MagicMock(side_effect=factory))

        with pytest.raises(RuntimeError, match="Failed to connect to Qdrant"):
            QdrantVectorStore(qdrant_url="http://fake-qdrant:6333")


@pytest.fixture
def store(fake_client_cls):
    return QdrantVectorStore(qdrant_url="http://fake-qdrant:6333")


class TestGeneratePointId:

    def test_deterministic_for_same_inputs(self, store):
        id1 = store._generate_point_id(document_id=1, source_type="text", chunk_index=0)
        id2 = store._generate_point_id(document_id=1, source_type="text", chunk_index=0)
        assert id1 == id2

    def test_differs_for_different_chunk_index(self, store):
        id1 = store._generate_point_id(document_id=1, source_type="text", chunk_index=0)
        id2 = store._generate_point_id(document_id=1, source_type="text", chunk_index=1)
        assert id1 != id2

    def test_differs_for_different_source_type(self, store):
        id1 = store._generate_point_id(document_id=1, source_type="text", chunk_index=0)
        id2 = store._generate_point_id(document_id=1, source_type="image", chunk_index=0)
        assert id1 != id2


class TestDeleteDocumentChunks:

    def test_calls_delete_with_document_id_filter(self, store):
        store.delete_document_chunks(document_id=42)

        store.client.delete.assert_called_once()
        _, kwargs = store.client.delete.call_args
        assert kwargs["collection_name"] == COLLECTION_NAME

    def test_delete_failure_raises_runtime_error(self, store):
        store.client.delete.side_effect = Exception("network down")

        with pytest.raises(RuntimeError, match="Failed to delete existing chunks"):
            store.delete_document_chunks(document_id=42)


class TestUpsertChunks:

    def test_mismatched_chunks_and_embeddings_raises(self, store):
        with pytest.raises(ValueError, match="Chunk count"):
            store.upsert_chunks(
                document_id=1, course_id=1, week_id=1,
                document_title="T", professor_id=1,
                chunks=["a", "b"], embeddings=[[0.1]]
            )

    def test_mismatched_page_numbers_raises(self, store):
        with pytest.raises(ValueError, match="page_numbers count"):
            store.upsert_chunks(
                document_id=1, course_id=1, week_id=1,
                document_title="T", professor_id=1,
                chunks=["a", "b"], embeddings=[[0.1], [0.2]],
                source_type="image", page_numbers=[1]
            )

    def test_upserts_points_with_expected_payload(self, store):
        store.upsert_chunks(
            document_id=7, course_id=3, week_id=2,
            document_title="Lecture 1", professor_id=9,
            chunks=["chunk one", "chunk two"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )

        store.client.upsert.assert_called_once()
        _, kwargs = store.client.upsert.call_args
        assert kwargs["collection_name"] == COLLECTION_NAME
        points = kwargs["points"]
        assert len(points) == 2

        payload0 = points[0].payload
        assert payload0["chunk_text"] == "chunk one"
        assert payload0["chunk_index"] == 0
        assert payload0["total_chunks"] == 2
        assert payload0["document_id"] == 7
        assert payload0["course_id"] == 3
        assert payload0["week_id"] == 2
        assert payload0["document_title"] == "Lecture 1"
        assert payload0["professor_id"] == 9
        assert payload0["source_type"] == "text"
        assert "page_number" not in payload0

    def test_upsert_includes_page_number_for_images(self, store):
        store.upsert_chunks(
            document_id=7, course_id=3, week_id=2,
            document_title="Lecture 1", professor_id=9,
            chunks=["caption one"],
            embeddings=[[0.1, 0.2]],
            source_type="image",
            page_numbers=[5],
        )

        _, kwargs = store.client.upsert.call_args
        points = kwargs["points"]
        assert points[0].payload["source_type"] == "image"
        assert points[0].payload["page_number"] == 5

    def test_upsert_batches_large_point_lists(self, store):
        n = 150  # > batch_size of 64, should trigger 3 upsert calls
        chunks = [f"chunk {i}" for i in range(n)]
        embeddings = [[0.0] for _ in range(n)]

        store.upsert_chunks(
            document_id=1, course_id=1, week_id=1,
            document_title="T", professor_id=1,
            chunks=chunks, embeddings=embeddings,
        )

        assert store.client.upsert.call_count == 3

    def test_upsert_failure_raises_runtime_error(self, store):
        store.client.upsert.side_effect = Exception("qdrant unreachable")

        with pytest.raises(RuntimeError, match="Failed to upsert chunks"):
            store.upsert_chunks(
                document_id=1, course_id=1, week_id=1,
                document_title="T", professor_id=1,
                chunks=["a"], embeddings=[[0.1]],
            )


class TestIsHealthy:

    def test_returns_true_when_collection_reachable(self, store):
        store.client.get_collection.return_value = MagicMock()
        assert store.is_healthy() is True

    def test_returns_false_when_collection_unreachable(self, store):
        store.client.get_collection.side_effect = Exception("down")
        assert store.is_healthy() is False


class TestSingleton:
    def test_get_qdrant_client_returns_singleton(self, monkeypatch, fake_client_cls):
        monkeypatch.setattr(qdrant_module, "_vector_store", None)

        first = qdrant_module.get_qdrant_client()
        second = qdrant_module.get_qdrant_client()

        assert first is second
