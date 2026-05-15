from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from core.orchestrator import orchestrate


app = FastAPI(title="SHL Assessment Recommender")


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    return orchestrate(request.messages)
