# ingest.py

import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import shutil

# -----------------------------
# LangChain Document Import
# -----------------------------
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

# -----------------------------
# Embeddings
# -----------------------------
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# Text Splitter
# -----------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Vector DB
# -----------------------------
from langchain_community.vectorstores import Chroma

documents = []

# ==================================================
# DATA DIRECTORY
# ==================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ==================================================
# CSV FILE
# ==================================================

for data_file in sorted(DATA_DIR.iterdir()):
    if data_file.is_dir() or data_file.name == "chroma_db":
        continue

    if data_file.suffix.lower() == ".csv":
        df = pd.read_csv(data_file)

        for _, row in df.iterrows():
            text = "\n".join(
                [f"{col}: {row[col]}" for col in df.columns]
            )

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": data_file.name,
                        "type": "csv",
                    }
                )
            )

    elif data_file.suffix.lower() in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(data_file)

        for sheet in xls.sheet_names:
            sheet_df = pd.read_excel(
                data_file,
                sheet_name=sheet
            )

            for _, row in sheet_df.iterrows():
                text = "\n".join(
                    [f"{col}: {row[col]}" for col in sheet_df.columns]
                )

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": data_file.name,
                            "type": "excel",
                            "sheet": sheet
                        }
                    )
                )

    elif data_file.suffix.lower() in {".html", ".htm"}:
        with open(data_file, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        text = soup.get_text(separator="\n")

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": data_file.name,
                    "type": "html"
                }
            )
        )

print(f"Loaded {len(documents)} documents")

# ==================================================
# EMBEDDINGS
# ==================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# ==================================================
# CHUNKING
# ==================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(
    documents
)

print(f"Chunks created: {len(chunks)}")

# ==================================================
# VECTOR DATABASE
# ==================================================

vector_db_dir = Path(__file__).resolve().parent / "chroma_db"

if vector_db_dir.exists():
    shutil.rmtree(vector_db_dir)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(vector_db_dir)
)

print("Vector DB created successfully!")

print(f"Total chunks stored: {len(chunks)}")