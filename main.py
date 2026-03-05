

import logging
import os
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.api.routes import router

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("intelligence_service.starting")

    import asyncio
    from app.services.intelligence_service import (
        _load_sentiment_pipeline, _load_zero_shot_pipeline, _load_ner_pipeline
    )
    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(None, _load_sentiment_pipeline),
        loop.run_in_executor(None, _load_zero_shot_pipeline),
        loop.run_in_executor(None, _load_ner_pipeline),
    )
    log.info("intelligence_service.models_loaded")
    yield
    log.info("intelligence_service.stopped")

app = FastAPI(
    title="Telco Intelligence Service",
    description="Real-time NLP: sentiment analysis, intent detection, entity extraction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/metrics", make_asgi_app())
app.include_router(router, prefix="/api/v1")

@app.get("/healthz")
async def health():
    return {"status": "ok", "service": "intelligence-service"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8002")), workers=1)
