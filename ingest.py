"""
Ingests PDFs into a local FAISS vectorstore for Student-AI.

Folder convention:
    pdfs/<subject-slug>/<any-name>.pdf

Each PDF's text is chunked and tagged with metadata:
    subject  -> derived from the folder name (e.g. "computer-networks")
    source   -> original filename
    unit     -> optional, parsed if the filename starts with "unit<N>_"
                e.g. "unit1_osi-model.pdf" -> unit = 1

Run this once whenever you add/replace PDFs:
    python ingest.py
"""

import os
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_ROOT = Path("pdfs")
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

UNIT_PATTERN = re.compile(r"^unit(\d+)[_\-]", re.IGNORECASE)


def infer_unit(filename: str):
    match = UNIT_PATTERN.match(filename)
    return int(match.group(1)) if match else None


def load_all_documents():
    if not PDF_ROOT.exists():
        raise SystemExit(f"'{PDF_ROOT}/' not found. Create it and add subject subfolders with PDFs.")

    subject_folders = [p for p in PDF_ROOT.iterdir() if p.is_dir()]
    if not subject_folders:
        raise SystemExit(f"No subject folders found under '{PDF_ROOT}/'. Expected e.g. pdfs/computer-networks/*.pdf")

    all_docs = []
    for subject_folder in subject_folders:
        subject_slug = subject_folder.name
        pdf_files = list(subject_folder.glob("*.pdf"))
        if not pdf_files:
            print(f"  (skipping '{subject_slug}' - no PDFs found)")
            continue

        for pdf_path in pdf_files:
            print(f"  loading {pdf_path}")
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            unit = infer_unit(pdf_path.name)
            for page in pages:
                page.metadata["subject"] = subject_slug
                page.metadata["source"] = pdf_path.name
                if unit is not None:
                    page.metadata["unit"] = unit
            all_docs.extend(pages)

    return all_docs


def main():
    print("Step 1/3: loading PDFs...")
    docs = load_all_documents()
    print(f"  loaded {len(docs)} pages total")

    print("Step 2/3: splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"  created {len(chunks)} chunks")

    print("Step 3/3: building embeddings + FAISS index (first run downloads the model, ~90MB)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORSTORE_DIR)

    print(f"\nDone. Vectorstore saved to '{VECTORSTORE_DIR}/'.")
    print("Commit this folder to your repo (or rebuild it in Streamlit Cloud on first boot).")


if __name__ == "__main__":
    main()
