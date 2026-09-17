from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import Settings, get_settings
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService


def create_app(
    settings: Settings | None = None,
    retrieval_service: Any | None = None,
    generation_service: Any | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.retrieval_service = (
                retrieval_service if retrieval_service is not None else RetrievalService(settings)
            )
            app.state.generation_service = (
                generation_service if generation_service is not None else GenerationService(settings)
            )
            app.state.ready = True
            app.state.startup_error = ""
        except Exception as exc:
            app.state.retrieval_service = None
            app.state.generation_service = None
            app.state.ready = False
            app.state.startup_error = str(exc)
        yield

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="A cited RAG API over the indexed Hands-On Machine Learning book.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(query_router)
    return app


app = create_app()
