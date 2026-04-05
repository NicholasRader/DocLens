from typing import List
from app.config import collection
from app.models import DocumentInfo


def store_in_chroma(chunks: List[str], embeddings: List[List[float]], filename: str) -> None:
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )


def get_all_chunks(filename: str) -> List[str]:
    results = collection.get(
        where={"filename": filename}
    )

    return results["documents"]


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