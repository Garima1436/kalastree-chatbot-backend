import json
from retrieval import retrieve_context
from nodes import generate_answer

queries = [
  "What is Banglar Rasogolla?",
  "When was Banglar Rasogolla given GI registration?",
  "Which states have the most GI products mentioned?",
  "Summarize any findings about women fintech adoption in the data.",
  "List examples of GI handicrafts mentioned for Rajasthan.",
  "Give a short list of GI food items mentioned for West Bengal.",
]

for q in queries:
    try:
        c, docs = retrieve_context(q, k=5)
        res = generate_answer({"question": q, "context": c, "docs": docs})
    except Exception as e:
        res = {"answer": f"ERROR: {e}", "sources": []}
    print('---')
    print('Query:', q)
    print(json.dumps(res, ensure_ascii=False, indent=2))
