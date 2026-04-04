from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import io
from pypdf import PdfReader
from typing import List
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import chromadb

load_dotenv()

app = FastAPI()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="documents")


class IngestResponse(BaseModel):
    filename: str
    size_bytes: int
    page_count: int
    chunk_count: int
    embedding_dimension: int
    first_chunk_preview: str
    message: str


class QueryRequest(BaseModel):
    filename: str
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int
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


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    contents = await file.read()

    pdf_file = io.BytesIO(contents)
    reader = PdfReader(pdf_file)

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    chunks = chunk_text(full_text)
    embeddings = await embed_chunks(chunks)
    store_in_chroma(chunks, embeddings, file.filename)

    return IngestResponse(
        filename=file.filename,
        size_bytes=len(contents),
        page_count=len(reader.pages),
        chunk_count=len(chunks),
        embedding_dimension=len(embeddings[0]) if embeddings else 0,
        first_chunk_preview=chunks[0][:300] if chunks else "",
        message=f"Successfully stored {len(chunks)} chunks from {file.filename}"
    )


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    chunks = await retrieve_relevant_chunks(request.question, request.filename)

    if not chunks:
        return QueryResponse(
            question=request.question,
            answer="No relevant content found for this document. Please ingest the document first.",
            chunks_used=0,
            message="No chunks retrieved"
        )
    
    answer = await generate_answer(request.question, chunks)

    return QueryResponse(
        question=request.question,
        answer=answer,
        chunks_used=len(chunks),
        message=f"Answer generated from {len(chunks)} chunks"
    )