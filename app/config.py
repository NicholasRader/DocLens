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

CHROMA_PATH = "/tmp/chroma_data" if "AWS_LAMBDA_FUNCTION_NAME" in os.environ else "./chroma_data"

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
semaphore = asyncio.Semaphore(CONCURRENT_LLM_CALLS)

_chroma_client = None
_collection = None


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(name="documents")
    return _collection