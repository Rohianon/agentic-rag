"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, documents, query
from .dependencies import PipelineState


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline components on startup."""
    print("Initializing RAG pipeline...")
    state = PipelineState()
    state.initialize()
    app.state.pipeline = state
    print("Pipeline ready!")
    yield
    print("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agentic RAG API",
        description="Enterprise document analysis with RAG",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
    app.include_router(query.router, prefix="/api/query", tags=["query"])

    # Serve frontend static files
    frontend_dir = Path(__file__).parent.parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()
