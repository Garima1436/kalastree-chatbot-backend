import json
from rag import ask_bot

queries = [
  "What is Banglar Rasogolla?",
  "When was Banglar Rasogolla given GI registration?",
  "Which states have the most GI products mentioned?",
  "Summarize any findings about women fintech adoption in the data.",
  "List examples of GI handicrafts mentioned for Rajasthan."
]

for q in queries:
    try:
        res = ask_bot(q)
    except Exception as e:
        res = {"answer": f"ERROR: {e}", "sources": []}
    print('---')
    print('Query:', q)
    print(json.dumps(res, ensure_ascii=False, indent=2))
