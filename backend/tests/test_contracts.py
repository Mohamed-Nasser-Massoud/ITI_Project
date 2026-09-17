from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.retrieval import RetrievedChunk


class FakeRetrievalService:
    collection_count = 2687
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="book-p12-c0",
                document="hands_on_ml.pdf",
                page=12,
                text="Overfitting occurs when a model fits training data too closely.",
                distance=0.1,
            )
        ]

    def status(self) -> dict[str, object]:
        return {
            "status": "ok",
            "collection": "hands_on_ml_book",
            "count": self.collection_count,
            "embedding_model": self.embedding_model,
        }


class FakeGenerationService:
    model = "llama3.2:3b"

    def generate(self, question: str, passages: list[RetrievedChunk]) -> str:
        return "Overfitting is fitting the training data too closely [hands_on_ml.pdf | page 12 | book-p12-c0]."

    def status(self) -> dict[str, object]:
        return {"status": "ok", "model": self.model}


def build_test_client() -> TestClient:
    settings = Settings(
        vector_store_path=Path("/tmp/test-vector-store"),
        manifest_path=Path("/tmp/test-manifest.json"),
    )
    return TestClient(
        create_app(
            settings=settings,
            retrieval_service=FakeRetrievalService(),
            generation_service=FakeGenerationService(),
        )
    )


def test_health_reports_each_runtime_component() -> None:
    with build_test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["retrieval"]["count"] == 2687
    assert payload["generation"]["model"] == "llama3.2:3b"


def test_metadata_exposes_manifest_and_pipeline_contract() -> None:
    with build_test_client() as client:
        response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["collection_name"] == "hands_on_ml_book"
    assert payload["chunk_size"] == 900
    assert payload["chunk_overlap"] == 150
    assert payload["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert payload["provenance"]["source_notebook"] == "notebooks/rag_pipeline.ipynb"


def test_query_returns_retrieval_evidence() -> None:
    with build_test_client() as client:
        response = client.post("/query", json={"question": "What is overfitting?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["passages"][0]["page"] == 12
    assert payload["passages"][0]["distance"] == 0.1
