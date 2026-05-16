"""
PDF Ingestion Module for SARS
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Larger chunks preserve architectural specification context.
# Vitruvius and structural reports need full paragraphs to retain meaning.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 150

CATEGORY_PRIORITY = {
    "primary":     1,
    "local":       2,
    "comparative": 3,
    "theory":      4,
}


def load_pdf(file_path: Path) -> Dict[str, Any]:
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num, "text": text.strip()})

    doc.close()

    category = file_path.parent.name
    priority = CATEGORY_PRIORITY.get(category, 99)

    return {
        "content": "\n\n".join(p["text"] for p in pages),
        "metadata": {
            "filename": file_path.name,
            "page_count": len(pages),
            "category": category,
            "priority": priority,
        },
    }


def _detect_section(line: str) -> bool:
    """True if a line looks like a section heading."""
    s = line.strip()
    if not s or len(s) > 120:
        return False
    return (
        s.isupper() or
        s.startswith(("Chapter", "Section", "§", "CHAPTER", "SECTION")) or
        (len(s) > 1 and len(s) < 60 and s[0].isdigit() and s[1] in ". )")
    )


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        meta = doc["metadata"]
        text = doc["content"]

        # Detect section headings to inject as context prefix into each chunk.
        # This way a chunk about "column proportions" also carries its section title.
        lines = text.split("\n")
        current_section: Optional[str] = None
        section_annotated_text = []
        for line in lines:
            if _detect_section(line):
                current_section = line.strip()
            section_annotated_text.append(line)
        annotated = "\n".join(section_annotated_text)

        raw_chunks = splitter.split_text(annotated)

        for chunk_idx, chunk in enumerate(raw_chunks):
            if len(chunk) < MIN_CHUNK_LENGTH:
                continue
            chunks.append({
                "content": chunk,
                "metadata": {
                    "filename": meta["filename"],
                    "source": meta["filename"],
                    "chunk_index": chunk_idx,
                    "category": meta["category"],
                    "priority": meta["priority"],
                },
            })

    return chunks


def ingest_pdfs() -> Dict[str, Any]:
    if not PAPERS_DIR.exists():
        print(f"Creating papers directory: {PAPERS_DIR}")
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        return {"n_docs": 0, "n_chunks": 0, "avg_chunk_length": 0, "chunks": []}

    pdf_files = sorted(PAPERS_DIR.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found under {PAPERS_DIR}")
        return {"n_docs": 0, "n_chunks": 0, "avg_chunk_length": 0, "chunks": []}

    print(f"Found {len(pdf_files)} PDF file(s)")

    documents = []
    for pdf_file in pdf_files:
        print(f"  Loading [{pdf_file.parent.name}]: {pdf_file.name}")
        try:
            documents.append(load_pdf(pdf_file))
        except Exception as e:
            print(f"  ERROR loading {pdf_file.name}: {e}")

    chunks = chunk_documents(documents)
    avg_len = sum(len(c["content"]) for c in chunks) / len(chunks) if chunks else 0

    print(f"\n{'='*50}")
    print("INGESTION SUMMARY")
    print(f"{'='*50}")
    print(f"Documents processed : {len(documents)}")
    print(f"Chunks created      : {len(chunks)}")
    print(f"Avg chunk length    : {avg_len:.1f} chars  (target: {CHUNK_SIZE})")
    print(f"{'='*50}\n")

    return {
        "n_docs": len(documents),
        "n_chunks": len(chunks),
        "avg_chunk_length": avg_len,
        "chunks": chunks,
    }


if __name__ == "__main__":
    result = ingest_pdfs()
    sys.exit(0 if result["n_chunks"] > 0 else 1)
