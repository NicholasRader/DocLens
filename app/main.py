from fastapi import FastAPI
from app.routers import documents, query

app = FastAPI(
    title="DocLens",
    description="Document ingestion, summarization, and Q&A API",
    version="1.0.0"
)

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}