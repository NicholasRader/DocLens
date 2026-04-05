from fastapi import FastAPI
from app.routers import documents, query
from mangum import Mangum
import os

ROOT_PATH = "/prod" if "AWS_LAMBDA_FUNCTION_NAME" in os.environ else ""

app = FastAPI(
    title="DocLens",
    description="Document ingestion, summarization, and Q&A API",
    version="1.0.0",
    root_path="/prod"
)

app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}


handler = Mangum(app, lifespan="off", api_gateway_base_path="/prod")