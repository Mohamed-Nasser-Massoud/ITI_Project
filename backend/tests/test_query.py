from fastapi.testclient import TestClient

from app.main import create_app
from app.services.retrieval import RetrievedChunk


class FakeRetrievalService:
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


class FakeGenerationService:
    def generate(self, question: str, passages: list[RetrievedChunk]) -> str:
        return "Overfitting is fitting the training data too closely [hands_on_ml.pdf | page 12 | book-p12-c0]."


def test_query_returns_grounded_answer_and_sources() -> None:
    app = create_app(retrieval_service=FakeRetrievalService(), generation_service=FakeGenerationService())
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "What is overfitting?"})

    assert response.status_code == 200
    assert "Overfitting" in response.json()["answer"]
    assert response.json()["sources"] == ["hands_on_ml.pdf | page 12 | book-p12-c0"]


def test_query_rejects_blank_question() -> None:
    app = create_app(retrieval_service=FakeRetrievalService(), generation_service=FakeGenerationService())
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422
