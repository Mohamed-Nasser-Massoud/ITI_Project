from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import Settings


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document: str
    page: int
    text: str
    distance: float


class RetrievalService:
    """Load the persisted Chroma collection once and retrieve cited passages."""

    def __init__(self, settings: Settings) -> None:
        if not settings.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {settings.vector_store_path}. "
                "Copy the notebook output into backend/data/vector_store/."
            )
        self._embedder = SentenceTransformer(
            settings.embedding_model,
            cache_folder=str(settings.embedding_cache_path),
        )
        client = chromadb.PersistentClient(path=str(settings.vector_store_path))
        self._collection = client.get_collection(settings.collection_name)
        self._top_k = settings.top_k
        self._embedding_model = settings.embedding_model
        self._collection_name = settings.collection_name
        self._vector_store_path = str(settings.vector_store_path)

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        count = self._collection.count()
        if count == 0:
            return []

        response = self._collection.query(
            query_embeddings=self._embedder.encode(
                [question], normalize_embeddings=True
            ).tolist(),
            n_results=min(self._top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=ids[index],
                document=str(metadatas[index]["source"]),
                page=int(metadatas[index]["page"]),
                text=documents[index],
                distance=float(distances[index]),
            )
            for index in range(len(ids))
        ]

    def status(self) -> dict[str, Any]:
        try:
            count = self._collection.count()
            return {
                "status": "ok" if count > 0 else "degraded",
                "collection": self._collection_name,
                "count": count,
                "embedding_model": self._embedding_model,
                "vector_store_path": self._vector_store_path,
            }
        except Exception as exc:
            return {"status": "degraded", "detail": str(exc)}
