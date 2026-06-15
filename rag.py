from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# =====================================================
# EMBEDDINGS
# =====================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# =====================================================
# VECTOR DATABASE
# =====================================================

vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 10}
)

# =====================================================
# GEMINI
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# =====================================================
# QUERY REWRITER
# =====================================================

def rewrite_query(question):

    prompt = f"""
You are an expert query understanding engine.

Rewrite the user's question so that it becomes
clear, grammatically correct and optimized
for retrieval from a research database.

Rules:
- Fix spelling mistakes
- Expand abbreviations
- Infer user intent
- Preserve meaning
- Return ONLY the rewritten query

Examples:

Input:
category of products that belong from orrisa

Output:
What GI product categories are associated with Odisha?

Input:
gujrat products

Output:
What GI products are associated with Gujarat?

Input:
highest fintech state

Output:
Which state has the highest fintech index?

Question:
{question}
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()

    except:
        return question


# =====================================================
# ANSWER GENERATION
# =====================================================

def ask_bot(question):

    try:

        # ------------------------------
        # Rewrite query
        # ------------------------------

        rewritten_query = rewrite_query(question)

        # ------------------------------
        # Retrieve context
        # ------------------------------

        docs = retriever.invoke(rewritten_query)

        if not docs:

            return {
                "answer":
                "I could not find that information in the knowledge base.",
                "sources": []
            }

        # ------------------------------
        # Build context
        # ------------------------------

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # ------------------------------
        # Main Prompt
        # ------------------------------

        prompt = f"""
You are KalaStree AI.

You answer questions about:

- GI Products
- Women Entrepreneurship
- Women FinTech Adoption
- Empowerment Index
- Research Reports
- Survey Data
- Policy Recommendations

IMPORTANT RULES:

1. Use ONLY the provided context.

2. The user may:
   - make spelling mistakes
   - use informal language
   - ask summary questions
   - ask comparison questions
   - ask aggregation questions

3. If information is not present in the context,
   respond EXACTLY:

   I could not find that information in the knowledge base.

4. Never hallucinate.

5. For numeric values:
   report exact values.

6. For lists:
   use bullet points.

7. If multiple records are relevant:
   summarize them clearly.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        # ------------------------------
        # Sources
        # ------------------------------

        sources = []

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "unknown"
            )

            if source not in sources:
                sources.append(source)

        return {
            "answer": response.content,
            "sources": sources,
            "rewritten_query": rewritten_query
        }

    except Exception as e:

        print("RAG ERROR:", str(e))

        return {
            "answer":
            "The AI service is temporarily unavailable. Please try again later.",
            "sources": []
        }