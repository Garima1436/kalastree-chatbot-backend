from dotenv import load_dotenv
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load Chroma DB
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 5}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Maps raw filenames to human-readable source labels
SOURCE_LABELS = {
    "KalaStree_FINAL_Complete.xlsx": "KalaStree Research Report",
    "Women_GI_Multimodal_Research_Report__with_images_v2.xlsx": "GI Multimodal Research Report",
    "women_fintech_gi_survey_india.csv": "Women FinTech & GI Survey",
    "garima_literature_methodology.html": "Literature & Methodology",
    "_Garima's Literaturchrome___newtab_e Review Tracker (1).xlsx": "Literature Review Tracker",
}

SYSTEM_PROMPT = """You are KalaStree GI Assistant, a specialist in Indian Geographical Indication (GI) products, women artisans, and FinTech adoption research.

Rules:
- Answer ONLY using the provided context.
- If the answer is not in the context, say exactly: "I could not find that information in the knowledge base."
- Be concise and accurate. Use bullet points or short paragraphs where appropriate.
- Do not fabricate statistics, names, or policy details not found in the context.
- You may refer to previous messages in the conversation to give coherent follow-up answers."""


def ask_bot(question: str, history: list[dict] | None = None) -> dict:
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(content=f"{SYSTEM_PROMPT}\n\nContext:\n{context}")
    ]

    # Inject prior conversation turns (capped at last 8 messages = 4 turns)
    if history:
        for msg in history[-8:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["text"]))
            elif msg.get("role") == "ai":
                messages.append(AIMessage(content=msg["text"]))

    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)

    sources = []
    for doc in docs:
        raw = doc.metadata.get("source", "unknown")
        filename = Path(raw).name
        label = SOURCE_LABELS.get(filename, filename)
        if label not in sources:
            sources.append(label)

    return {
        "answer": response.content,
        "sources": sources,
    }
