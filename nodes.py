
# nodes.py

from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


class GraphState(TypedDict):

    question: str
    rewritten_query: str

    context: str
    answer: str
    sources: list

    retrieval_grade: str
    grounding_grade: str


# =====================================================
# QUERY REWRITE
# =====================================================

def rewrite_query(state):

    question = state["question"]

    prompt = f"""
You are an expert query understanding engine.
You are KalaStree AI.
Make it relevant to the context of Geographical Indication (GI) products present in the database only.
Don't answer from general world knowledge.

IMPORTANT DOMAIN KNOWLEDGE:

GI = Geographical Indication products.

Never interpret GI as Gastrointestinal.

The database contains:

- Geographical Indication (GI) products
- Women FinTech Adoption
- Women Empowerment
- Survey Data
- Research Reports

Rewrite the user query for retrieval.

Fix:
- spelling mistakes
- grammar
- abbreviations
- state names
- product names

Examples:

GI products
→ Geographical Indication products

orrisa
→ Odisha

gujrat
→ Gujarat

Question:
{question}

Return ONLY the rewritten query.
"""

    response = llm.invoke(prompt)

    return {
        "rewritten_query":
        response.content.strip()
    }


# =====================================================
# RETRIEVAL GRADER
# =====================================================

def grade_retrieval(state):

    question = state["question"]
    context = state["context"]

    prompt = f"""
Question:
{question}

Context:
{context}

Can this context answer the question?

Reply ONLY:

GOOD
PARTIAL
BAD
"""

    response = llm.invoke(prompt)

    return {
        "retrieval_grade":
        response.content.strip()
    }


# =====================================================
# ANSWER GENERATOR
# =====================================================

def generate_answer(state):

    question = state["question"]
    context = state.get("context", "")
    docs = state.get("docs", [])

    if not docs:
        return {"answer": "I could not find that information in the knowledge base.", "sources": []}

    MAX_CHARS = 3000
    answer_parts = []
    sources = []
    total = 0
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        chunk = text
        remaining = MAX_CHARS - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        answer_parts.append(chunk)
        total += len(chunk)
        src = getattr(doc, "metadata", {}).get("source") if hasattr(doc, "metadata") else None
        if src and src not in sources:
            sources.append(src)

    answer = "\n\n".join(answer_parts).strip()

    if not answer:
        return {"answer": "I could not find that information in the knowledge base.", "sources": []}

    return {"answer": answer, "sources": sources}


# =====================================================
# GROUNDING CHECK
# =====================================================

def check_grounding(state):

    question = state["question"]

    context = state["context"]

    answer = state["answer"]

    prompt = f"""
Question:
{question}

Context:
{context}

Answer:
{answer}

Is answer fully supported?

Reply ONLY:

SUPPORTED

or

UNSUPPORTED
"""

    response = llm.invoke(prompt)

    return {
        "grounding_grade":
        response.content.strip()
    }