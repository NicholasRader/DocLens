import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import chromadb
import asyncio

load_dotenv()

CONCURRENT_LLM_CALLS = 10
BATCH_SIZE = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="documents")
semaphore = asyncio.Semaphore(CONCURRENT_LLM_CALLS)