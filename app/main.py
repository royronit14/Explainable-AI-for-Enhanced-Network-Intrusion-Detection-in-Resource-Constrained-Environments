"""FastAPI entry point for the XAI NIDS backend."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import get_artifacts, router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

DEFAULT_ORIGINS = ["http://localhost:3000"]


def _parse_origins() -> List[str]:
    raw = os.environ.get("CORS_ORIGINS")
    if not raw:
        return DEFAULT_ORIGINS
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Explainable AI Network Intrusion Detection",
        description="HTTP backend around the XAI NIDS inference engine.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.on_event("startup")
    def _warmup() -> None:
        # Pre-load artifacts so the first request is fast and /health reflects
        # reality immediately.
        get_artifacts()

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
