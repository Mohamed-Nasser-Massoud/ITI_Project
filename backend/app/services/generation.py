from typing import Any
import re

from ollama import Client

from app.core.config import Settings
from app.services.retrieval import RetrievedChunk


class GenerationService:
    """Call the local Ollama model using only retrieved book passages."""

    def __init__(self, settings: Settings) -> None:
        self._client = Client(host=settings.ollama_base_url)
        self._model = settings.ollama_model

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def build_prompt(question: str, passages: list[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"[{item.document} | page {item.page} | {item.chunk_id}]\n{item.text}"
            for item in passages
        )
        return f"""You are a helpful assistant for the Hands-On Machine Learning book.
Answer only from the supplied book context. Do not use outside knowledge.
If the context is insufficient, say exactly: "I do not have enough information in the provided book passages."
Use concise prose. Cite every factual claim with an exact bracketed source label copied from the context.
The only valid citation format is [source | page N | chunk_id]. Never use page-only citations.
Never invent a page number, source, quote, or concept that is not in the context.

Context:
{context}

Question: {question}
Answer:"""

    @staticmethod
    def ensure_citations(answer: str, passages: list[RetrievedChunk]) -> str:
        """Normalize common page-only citations and guarantee an evidence footer."""
        labels = [
            f"[{item.document} | page {item.page} | {item.chunk_id}]"
            for item in passages
        ]
        labels_by_page = {str(item.page): label for item, label in zip(passages, labels)}

        def replace_page_marker(match: re.Match[str]) -> str:
            return labels_by_page.get(match.group(1), match.group(0))

        normalized = re.sub(r"\[(\d{1,4})(?:\s*\|\s*[^\]]+)?\]", replace_page_marker, answer).strip()
        if labels and not any(label in normalized for label in labels):
            normalized = f"{normalized}\n\nCitations: {' '.join(labels)}"
        return normalized

    def generate(self, question: str, passages: list[RetrievedChunk]) -> str:
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": self.build_prompt(question, passages)}],
            options={"temperature": 0},
        )
        message = getattr(response, "message", None)
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return self.ensure_citations(str(content), passages)

    def status(self) -> dict[str, Any]:
        try:
            response = self._client.list()
            models = getattr(response, "models", response if isinstance(response, list) else [])
            names: list[str] = []
            for item in models:
                name = getattr(item, "model", None)
                if name is None and isinstance(item, dict):
                    name = item.get("name") or item.get("model")
                if name:
                    names.append(str(name))
            available = any(name == self._model or name.split(":")[0] == self._model for name in names)
            return {
                "status": "ok" if available else "degraded",
                "model": self._model,
                "available": available,
                "installed_models": names,
                "detail": "" if available else "Configured model is not installed.",
            }
        except Exception as exc:
            return {
                "status": "degraded",
                "model": self._model,
                "available": False,
                "detail": f"Ollama is unreachable: {exc}",
            }
