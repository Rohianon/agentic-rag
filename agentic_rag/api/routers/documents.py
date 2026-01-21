"""Document upload and management endpoints."""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from ..models import UploadResponse, DocumentInfo, DocumentListResponse, DocumentListItem
from ..dependencies import get_pipeline, PipelineState

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    extract_tables: bool = True,
    pipeline: PipelineState = Depends(get_pipeline),
):
    """
    Upload and index a PDF document.

    Pipeline: Parse -> Extract Tables -> Chunk -> Index
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Phase 1: Parse
        document = pipeline.parser.parse(tmp_path)
        # Update filename to original
        document.filename = file.filename

        # Phase 2: Extract tables (optional)
        all_tables = []
        if extract_tables:
            all_tables = pipeline.table_extractor.extract_from_pages(
                document.pages,
                only_table_pages=True
            )

        # Phase 3: Chunk
        all_chunks = []
        for page in document.pages:
            chunks = pipeline.chunker.chunk_text(
                text=page.text,
                source_file=file.filename,
                page_num=page.page_num,
            )
            all_chunks.extend(chunks)

        for i, table in enumerate(all_tables):
            chunk = pipeline.chunker.chunk_table(
                table_json=table.table_json,
                table_summary=table.table_summary,
                source_file=file.filename,
                page_num=table.page_num,
                chunk_id=f"{file.filename}_table_{i}"
            )
            all_chunks.append(chunk)

        # Phase 4: Index
        added = pipeline.index.add_chunks(all_chunks)

        # Track document
        pipeline.indexed_documents[file.filename] = document

        return UploadResponse(
            success=True,
            filename=file.filename,
            message=f"Successfully indexed {added} chunks",
            document_info=DocumentInfo(
                filename=file.filename,
                total_pages=document.total_pages,
                pages_with_tables=len(document.get_pages_with_tables()),
                chunks_indexed=added,
                indexed_at=datetime.now()
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        tmp_path.unlink(missing_ok=True)


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(pipeline: PipelineState = Depends(get_pipeline)):
    """List all indexed documents."""
    return DocumentListResponse(
        documents=[
            DocumentListItem(
                filename=filename,
                total_pages=doc.total_pages,
                pages_with_tables=len(doc.get_pages_with_tables()),
            )
            for filename, doc in pipeline.indexed_documents.items()
        ]
    )


@router.post("/clear")
async def clear_index(pipeline: PipelineState = Depends(get_pipeline)):
    """Clear all documents from the index."""
    pipeline.index.clear()
    pipeline.indexed_documents.clear()
    return {"success": True, "message": "Index cleared"}
