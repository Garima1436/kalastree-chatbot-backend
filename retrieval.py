
# retrieval.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


def retrieve_context(query, k=10):

    docs = vectordb.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=50
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return context, docs