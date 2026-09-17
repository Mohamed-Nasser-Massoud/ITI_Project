from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.query import PassageEvidence, QueryRequest, QueryResponse


router = APIRouter(tags=["RAG"])


def _service_status(service: Any, unavailable_detail: str) -> dict[str, Any]:
    if service is None:
        return {"status": "degraded", "detail": unavailable_detail}
    method = getattr(service, "status", None)
    if method is None:
        return {"status": "ok"}
    try:
        return method()
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    retrieval = _service_status(request.app.state.retrieval_service, "Retrieval service is not loaded.")
    generation = _service_status(request.app.state.generation_service, "Generation service is not loaded.")
    ready = bool(request.app.state.ready)
    overall = "ok" if ready and retrieval.get("status") == "ok" and generation.get("status") == "ok" else "degraded"
    payload: dict[str, Any] = {
        "status": overall,
        "retrieval": retrieval,
        "generation": generation,
    }
    if request.app.state.startup_error:
        payload["startup_error"] = request.app.state.startup_error
    return payload


@router.get("/metadata")
def metadata(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    return settings.pipeline_metadata()


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    if not request.app.state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=request.app.state.startup_error or "RAG services are not ready.",
        )

    try:
        passages = request.app.state.retrieval_service.retrieve(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The retrieval service could not search the vector store: {exc}",
        ) from exc

    evidence = [
        PassageEvidence(
            chunk_id=item.chunk_id,
            source=item.document,
            page=item.page,
            distance=item.distance,
            text=item.text,
        )
        for item in passages
    ]
    sources = list(dict.fromkeys(
        f"{item.document} | page {item.page} | {item.chunk_id}" for item in passages
    ))
    if not passages:
        return QueryResponse(
            answer="I do not have enough information in the provided book passages.",
            sources=[],
            passages=[],
        )

    try:
        answer = request.app.state.generation_service.generate(payload.question, passages)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The local Ollama model could not generate an answer: {exc}",
        ) from exc

    return QueryResponse(answer=answer, sources=sources, passages=evidence)
