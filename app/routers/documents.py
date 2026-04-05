from fastapi import APIRouter, File, UploadFile, HTTPException
import io
from pypdf import PdfReader
from typing import List
import time
from app.models import IngestResponse, DeleteResponse, DocumentsResponse
from app.services.pdf import chunk_text
from app.services.embeddings import embed_chunks
from app.services.storage import store_in_chroma, get_all_chunks, get_all_documents
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, collection
from app.utils import format_duration


router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse)
def list_documents():
    start = time.time()

    documents = get_all_documents()

    total_chunks = sum(doc.chunk_count for doc in documents)

    return DocumentsResponse(
        documents=documents,
        total_documents=len(documents),
        duration=format_duration(time.time() - start),
        message=(
            f"{len(documents)} document(s) ingested, "
            f"{total_chunks} total chunks stored"
        )
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    start = time.time()

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    existing = get_all_chunks(file.filename)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"'{file.filename}' has already been ingested "
                f"({len(existing)} chunks stored). "
                "Delete it first if you want to re-ingest."
            )
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        pdf_file = io.BytesIO(contents)
        reader = PdfReader(pdf_file)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not parse PDF. The file may be corrupted."
        )

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    if not full_text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No text could be extracted from this PDF. "
                "It may be a scanned image. Only text-based PDFs are supported."
            )
        )

    try:
        chunks = chunk_text(full_text)
        embeddings = await embed_chunks(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service unavailable: {str(e)}"
        )

    store_in_chroma(chunks, embeddings, file.filename)

    return IngestResponse(
        filename=file.filename,
        size_bytes=len(contents),
        page_count=len(reader.pages),
        chunk_count=len(chunks),
        embedding_dimension=len(embeddings[0]) if embeddings else 0,
        first_chunk_preview=chunks[0][:300] if chunks else "",
        duration=format_duration(time.time() - start),
        message=f"Successfully stored {len(chunks)} chunks from {file.filename}"
    )


@router.delete("/document/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str):
    start = time.time()

    existing = get_all_chunks(filename)

    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"No content found for '{filename}'. Nothing to delete."
        )
    
    collection.delete(where={"filename": filename})

    return DeleteResponse(
        filename=filename,
        chunks_deleted=len(existing),
        duration=format_duration(time.time() - start),
        message=f"Deleted {len(existing)} chunks for '{filename}"
    )