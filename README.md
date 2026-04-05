# DocLens

A document intelligence API that lets you ingest PDFs, ask questions about them, and generate summaries. Built with a RAG (Retrieval-Augmented Generation) pipeline - documents are chunked, embedded, and stored as vectors so queries retrieve only relevant context rather than sending entire documents to the LLM on every call.

## Stack

- **FastAPI** — async REST API with Pydantic request/response validation
- **AWS Lambda + API Gateway** — serverless deployment, scales to zero when idle
- **ChromaDB** — vector database for semantic similarity search
- **OpenAI API** — `text-embedding-3-small` for embeddings, `gpt-5-nano` for generation
- **pypdf** — PDF text extraction
- **Docker + GitHub Actions** — containerized builds, automated Lambda deployment

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/documents` | List all ingested documents |
| POST | `/ingest` | Upload and process a PDF |
| POST | `/query` | Ask a question about a document |
| POST | `/summarize` | Summarize a document |
| DELETE | `/document/{filename}` | Remove a document |

## How it works

**Ingestion:** PDF is parsed with pypdf, split into overlapping 1000-character chunks, embedded via OpenAI, and stored in ChromaDB with filename metadata.

**Query:** The question is embedded with the same model, ChromaDB finds the 3 most similar chunks via cosine similarity, and the LLM answers using only that retrieved context.

**Summarization:** Chunks are batched into groups of 10 and summarized concurrently via `asyncio.gather` with a semaphore capping concurrent LLM calls. Batch summaries are reduced into a single final summary.

## Running locally
```bash
git clone https://github.com/NicholasRader/DocLens
cd doclens
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

Add a `.env` file:
```
OPENAI_API_KEY=your-key-here
```
```bash
uvicorn app.main:app --reload
```

API docs at `http://localhost:8000/docs`.

## Notes

Scanned/image-based PDFs are not supported - text extraction requires a text layer. For large documents (100+ pages), summarization takes 30-60 seconds depending on chunk count.