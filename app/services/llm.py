from typing import List
import asyncio
from app.config import client, semaphore, BATCH_SIZE


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


async def summarize_chunks(chunks: List[str], batch_size: int = BATCH_SIZE) -> str:
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