from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

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

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def ask_bot(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are KalaStree GI Assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
say:

'I could not find that information in the knowledge base.'

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

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
        "sources": sources
    }