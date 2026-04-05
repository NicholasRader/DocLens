from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import io
from pypdf import PdfReader
from typing import List
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import chromadb
import asyncio
import time

load_dotenv()

app = FastAPI()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="documents")
CONCURRENT_LLM_CALLS = 10
semaphore = asyncio.Semaphore(CONCURRENT_LLM_CALLS)


class IngestResponse(BaseModel):
    filename: str
    size_bytes: int
    page_count: int
    chunk_count: int
    embedding_dimension: int
    first_chunk_preview: str
    duration: str
    message: str


class QueryRequest(BaseModel):
    filename: str
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int
    duration: str
    message: str


class SummarizeRequest(BaseModel):
    filename: str


class SummarizeResponse(BaseModel):
    filename: str
    chunk_count: int
    summary: str
    duration: str
    message: str


class DeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    duration: str
    message: str


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    duration: str
    message: str


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


async def embed_chunks(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    embeddings = [item.embedding for item in response.data]
    return embeddings


def store_in_chroma(chunks: List[str], embeddings: List[List[float]], filename: str) -> None:
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )


async def retrieve_relevant_chunks(question: str, filename: str, n_results: int = 3) -> List[str]:
    question_embedding = await embed_chunks([question])

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results,
        where={"filename": filename}
    )

    chunks = results["documents"][0]
    return chunks


async def generate_answer(question: str, chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions about documents. "
        "Answer the question using only the context provided below. "
        "If the answer cannot be found in the context, say so clearly. "
        "Do not make up information."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content


def get_all_chunks(filename: str) -> List[str]:
    results = collection.get(
        where={"filename": filename}
    )

    return results["documents"]


async def summarize_chunks(chunks: List[str], batch_size: int = 10) -> str:
    if not chunks:
        return ""
    
    batches = [
        chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)
    ]
    
    async def summarize_batch(batch: List[str]) -> str:
        combined_chunks = "\n\n---\n\n".join(batch)
        async with semaphore:
            response = await client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": ( 
                        "You are a summarization assistant. "
                        "Summarize the following text concisely in 2-3 sentences. "
                        "Preserve the most important facts and figures."
                    )},
                    {"role": "user", "content": combined_chunks}
                ]
            )
        return response.choices[0].message.content
        
    batch_summaries = await asyncio.gather(*[summarize_batch(batch) for batch in batches])

    if len(batch_summaries) == 1:
        return batch_summaries[0]
    
    combined = "\n\n".join(batch_summaries)
    final_response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": (
                "You are a summarization assistant. "
                "You will receive several partial summaries of sections of a document. "
                "Combine them into one coherent summary of the whole document in 4-6 sentences. "
                "Do not repeat information. Write in flowing prose, not bullet points."
            )},
            {"role": "user", "content": combined}
        ]
    )

    return final_response.choices[0].message.content


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.2f}s"


def get_all_documents() -> List[DocumentInfo]:
    results = collection.get()

    if not results["metadatas"]:
        return []
    
    counts: dict[str, int] = {}
    for metadata in results["metadatas"]:
        filename = metadata["filename"]
        counts[filename] = counts.get(filename, 0) + 1

    return [
        DocumentInfo(filename=filename, chunk_count=count) for filename, count in sorted(counts.items())
    ]


@app.post("/ingest", response_model=IngestResponse)
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


@app.post("/query", response_model=QueryResponse)
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


@app.post("/summarize", response_model=SummarizeResponse)
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


@app.delete("/document/{filename}", response_model=DeleteResponse)
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


@app.get("/documents", response_model=DocumentsResponse)
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