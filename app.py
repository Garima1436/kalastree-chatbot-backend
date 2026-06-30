from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from rag import ask_bot
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str   # "user" or "ai"
    text: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[HistoryMessage]] = []


@app.get("/")
def home():
    return {"message": "KalaStree GI Assistant Running"}


@app.post("/chat")
def chat(data: ChatRequest):
    result = ask_bot(
        data.question,
        history=[m.model_dump() for m in (data.history or [])],
    )
    return result
