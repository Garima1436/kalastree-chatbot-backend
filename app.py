from fastapi import FastAPI
from pydantic import BaseModel

from rag import ask_bot
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # we'll restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {
        "message": "KalaStree GI Assistant Running"
    }

@app.post("/chat")
def chat(data: Question):

    result = ask_bot(
        data.question
    )

    return result