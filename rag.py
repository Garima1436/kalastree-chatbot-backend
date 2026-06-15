from graph import graph
from retrieval import retrieve_context
from nodes import generate_answer
import threading
import queue
import re


def _run_graph(question, out_q):
    try:
        result = graph.invoke({
            "question": question,
            "rewritten_query": "",
            "context": "",
            "answer": "",
            "retrieval_grade": "",
            "grounding_grade": ""
        })
        out_q.put(("ok", result))
    except Exception as e:
        out_q.put(("error", str(e)))


def _keyword_overlap(question, answer):
    q_tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
    if not q_tokens:
        return 0.0
    a_text = (answer or "").lower()
    hits = sum(1 for t in q_tokens if t in a_text)
    return hits / len(q_tokens)


def ask_bot(question, rag_timeout: float = 20.0):
    """Run the fast extractive pipeline and the RAG pipeline in parallel.

    Return the answer that scores higher by a simple keyword-overlap metric,
    preferring the faster extractive result when RAG is unavailable or slower.
    """

    try:
        context, docs = retrieve_context(question, k=8)
        extractive = generate_answer({"question": question, "context": context, "docs": docs})
    except Exception:
        extractive = {"answer": "I could not find that information in the knowledge base.", "sources": []}

    out_q = queue.Queue()
    t = threading.Thread(target=_run_graph, args=(question, out_q), daemon=True)
    t.start()

    rag_result = None
    try:
        status, payload = out_q.get(timeout=rag_timeout)
        if status == "ok":
            rag_result = payload
    except queue.Empty:
        rag_result = None

    extractive_answer = extractive.get("answer", "")
    extractive_score = _keyword_overlap(question, extractive_answer)

    rag_answer = ""
    rag_score = 0.0
    if rag_result:
        rag_answer = rag_result.get("answer", "")
        rag_score = _keyword_overlap(question, rag_answer)

    if rag_result and rag_score > extractive_score + 0.05:
        chosen = (rag_answer, rag_result.get("sources", []), "rag")
    else:
        chosen = (extractive_answer, extractive.get("sources", []), "extractive")

    answer_text, sources, pipeline = chosen

    return {
        "answer": answer_text or "I could not find that information in the knowledge base.",
        "sources": sources or [],
        "pipeline": pipeline
    }