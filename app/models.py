from pydantic import BaseModel
from typing import List


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