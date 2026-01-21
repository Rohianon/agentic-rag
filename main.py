"""Run the FastAPI server for the RAG pipeline."""

import uvicorn


def main():
    """Start the web server."""
    print("Starting Agentic RAG server...")
    print("Open http://localhost:8000 in your browser")
    print("API docs at http://localhost:8000/docs")
    uvicorn.run(
        "agentic_rag.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
