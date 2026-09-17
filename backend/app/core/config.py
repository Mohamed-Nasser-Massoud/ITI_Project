from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"


class Settings(BaseSettings):
    """Runtime configuration for the API and its persisted RAG artifacts."""

    app_name: str = "Hands-On Machine Learning Book Assistant API"
    environment: str = "development"
    project_root: Path = PROJECT_ROOT
    vector_store_path: Path = Path("data/vector_store")
    manifest_path: Path = Path("data/rag_output/rag_config.json")
    provenance_path: Path = Path("data/rag_output/artifact_provenance.json")
    embedding_cache_path: Path = Path(".model_cache")
    collection_name: str = "hands_on_ml_book"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    top_k: int = 4
    cors_origins: str = "http://localhost:8501"

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context: Any) -> None:
        self.project_root = self.project_root.expanduser().resolve()
        for field_name in ("vector_store_path", "manifest_path", "provenance_path", "embedding_cache_path"):
            value = getattr(self, field_name).expanduser()
            if not value.is_absolute():
                value = self.project_root / "backend" / value
            setattr(self, field_name, value.resolve())

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def pipeline_metadata(self) -> dict[str, Any]:
        """Return the frozen notebook contract plus runtime locations."""
        metadata: dict[str, Any] = {
            "embedding_model": self.embedding_model,
            "collection_name": self.collection_name,
            "chunk_size": 900,
            "chunk_overlap": 150,
            "top_k": self.top_k,
            "generation_model": self.ollama_model,
            "vector_store_path": str(self.vector_store_path),
            "manifest_path": str(self.manifest_path),
            "provenance_path": str(self.provenance_path),
            "provenance": {
                "source_notebook": "notebooks/rag_pipeline.ipynb",
                "source_output": "backend/data/rag_output/rag_config.json",
                "vector_store": "backend/data/vector_store",
                "records": 1157,
                "chunks": 2687,
            },
            "limitations": [
                "Answers are grounded only in retrieved passages from the indexed book.",
                "The backend requires the local Ollama server and configured model.",
                "The source PDF is not redistributed by this project.",
            ],
        }
        if self.manifest_path.exists():
            try:
                notebook_metadata = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                for key in (
                    "embedding_model",
                    "collection_name",
                    "chunk_size",
                    "chunk_overlap",
                    "top_k",
                    "vector_store_path",
                ):
                    if key in notebook_metadata:
                        metadata[key] = notebook_metadata[key]
            except (OSError, json.JSONDecodeError):
                metadata["manifest_warning"] = "The notebook manifest could not be read."
        if self.provenance_path.exists():
            try:
                provenance = json.loads(self.provenance_path.read_text(encoding="utf-8"))
                if isinstance(provenance, dict):
                    metadata["provenance"].update(provenance)
            except (OSError, json.JSONDecodeError):
                metadata["provenance_warning"] = "The artifact provenance record could not be read."
        return metadata


@lru_cache
def get_settings() -> Settings:
    return Settings()
