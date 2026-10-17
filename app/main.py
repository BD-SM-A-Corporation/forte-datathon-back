from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite tables on startup
    await init_db()
    yield


app = FastAPI(
    title="Forte Datathon Video Analytics Backend",
    description="Video compliance analysis API for cashier desks adhering to NBRK regulations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration to allow direct uploads from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Forte Datathon Compliance Analysis Service is running.",
        "docs_url": "/docs",
        "health_url": "/health",
    }
