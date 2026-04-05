from typing import List
from app.config import client


async def embed_chunks(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    return [item.embedding for item in response.data]