from fastapi import APIRouter, HTTPException
import time
from app.models import QueryRequest, QueryResponse, SummarizeRequest, SummarizeResponse
from app.services.embeddings import embed_chunks
from app.services.storage import get_all_chunks
from app.services.llm import generate_answer, summarize_chunks
from app.config import collection
from app.utils import format_duration


router = APIRouter()


async def retrieve_relevant_chunks(question: str, filename: str, n_results: int = 3) -> List[str]:
    question_embedding = await embed_chunks([question])

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
        where={"filename": filename}
    )

    chunks = results["documents"][0]
    return chunks


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    start = time.time()

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        chunks = await retrieve_relevant_chunks(request.question, request.filename)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval service unavailable: {str(e)}"
        )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No content found for '{request.filename}'. Please ingest the document first."
        )
    
    try:
        answer = await generate_answer(request.question, chunks)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Answer question unavailable: {str(e)}"
        )

    return QueryResponse(
        question=request.question,
        answer=answer,
        chunks_used=len(chunks),
        duration=format_duration(time.time() - start),
        message=f"Answer generated from {len(chunks)} chunks"
    )


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest):
    start = time.time()

    chunks = get_all_chunks(request.filename)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No content found for '{request.filename}'. Please ingest the document first."
        )
    
    try:
        summary = await summarize_chunks(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Summarization service unavailable: {str(e)}"
        )

    return SummarizeResponse(
        filename=request.filename,
        chunk_count=len(chunks),
        summary=summary,
        duration=format_duration(time.time() - start),
        message=f"Summarized {len(chunks)} chunks from {request.filename}"
    )
