# ingest.py

import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

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

csv_file = DATA_DIR / "women_fintech_gi_survey_india.csv"

df = pd.read_csv(csv_file)

for _, row in df.iterrows():

    text = "\n".join(
        [f"{col}: {row[col]}" for col in df.columns]
    )

    documents.append(
        Document(
            page_content=text,
            metadata={
                "source": "survey_csv"
            }
        )
    )

# ==================================================
# EXCEL FILE
# ==================================================

excel_file = DATA_DIR / "Women_GI_Multimodal_Research_Report (1).xlsx"

xls = pd.ExcelFile(excel_file)

for sheet in xls.sheet_names:

    sheet_df = pd.read_excel(
        excel_file,
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
                    "source": "excel",
                    "sheet": sheet
                }
            )
        )

# ==================================================
# HTML FILE
# ==================================================

html_file = DATA_DIR / "garima_literature_methodology.html"

with open(html_file, "r", encoding="utf-8") as f:
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
            "source": "literature"
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

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Vector DB created successfully!")

print(f"Total chunks stored: {len(chunks)}")