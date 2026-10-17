from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api.routes.analysis import router
from app.core.config import Settings
from app.core.logging import configure_logging
from app.services.jobs import JobStore
from app.services.rules import RuleEngine


class UploadLimitMiddleware:
    """Bound the entire multipart request before Starlette spools it to disk."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/v1/analyze":
            return await self.app(scope, receive, send)
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                raise HTTPException(413, "Upload exceeds configured size limit")
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings: Settings | None = None, jobs: JobStore | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        if not settings.api_key:
            raise RuntimeError("API_KEY must be configured")
        app.state.rules = RuleEngine.load(settings.rules_path)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(title="Forte CCTV Analysis", lifespan=lifespan)
    app.state.settings = settings
    app.state.jobs = jobs or JobStore(settings)
    app.add_middleware(UploadLimitMiddleware, limit=settings.max_upload_mb * 1024 * 1024 + 65536)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict:
        try:
            await run_in_threadpool(app.state.jobs.connection.ping)
        except Exception as exc:
            raise HTTPException(503, "Redis unavailable") from exc
        return {"status": "ok", "regulations_configured": app.state.rules.configured}

    return app


app = create_app()
