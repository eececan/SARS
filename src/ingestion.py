"""
PDF Ingestion Module for SARS

This module handles PDF document loading, text extraction, and chunking
for the Templum Divi Augusti research project.

Why PyMuPDF (fitz)?
- Fast and memory-efficient text extraction
- Preserves page structure and metadata
- Handles scanned PDFs better than alternatives
- No external dependencies beyond the library itself

Why RecursiveCharacterTextSplitter?
- Preserves semantic coherence better than fixed-size chunks
- Respects natural language boundaries (paragraphs, sentences)
- Overlap ensures context continuity across chunk boundaries
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Anchor paths to project root (src/ is one level below root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_CHUNK_LENGTH = 100

# Relevance priority order for each category folder
CATEGORY_PRIORITY = {
    "primary": 1,
    "local": 2,
    "comparative": 3,
    "theory": 4,
}


def load_pdf(file_path: Path) -> Dict[str, Any]:
    """
    Load a single PDF and extract text with metadata.

    Why return a dict instead of raw text?
    - Preserves source provenance for citation
    - Enables page-level reference in results
    - Supports section detection downstream

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary containing:
        - content: Extracted text joined across all pages
        - metadata: filename, page_count, category, priority
    """
    doc = fitz.open(file_path)
    full_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            full_text.append({
                "page": page_num,
                "text": text.strip()
            })

    doc.close()

    # Determine category from the immediate parent folder name
    category = file_path.parent.name
    priority = CATEGORY_PRIORITY.get(category, 99)

    return {
        "content": "\n\n".join([p["text"] for p in full_text]),
        "metadata": {
            "filename": file_path.name,
            "page_count": len(full_text),
            "category": category,
            "priority": priority,
        }
    }


def detect_section(page_text: str) -> Optional[str]:
    """
    Attempt to detect section headings from page text.

    Why this approach?
    - Roman academic texts often have clear section markers
    - Headers like "Chapter", "Section", or numbered headings
    - Falls back gracefully if no clear structure detected

    Args:
        page_text: Text content from a page

    Returns:
        Detected section name or None
    """
    lines = page_text.split('\n')
    for line in lines[:5]:  # Check first few lines
        line = line.strip()
        if (line.isupper() and len(line) < 100) or \
           line.startswith(('Chapter', 'Section', '§')):
            return line
    return None


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Why RecursiveCharacterTextSplitter?
    - Tries multiple levels of splitting (paragraphs, sentences, characters)
    - Produces more semantically coherent chunks than fixed-size splitting
    - Overlap of 64 chars preserves context at chunk boundaries

    Args:
        documents: List of document dictionaries with 'content' and 'metadata'

    Returns:
        List of chunk dictionaries with content and metadata including category/priority
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        text = doc["content"]
        metadata = doc["metadata"]

        text_chunks = text_splitter.split_text(text)

        for chunk_idx, chunk in enumerate(text_chunks):
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        "filename": metadata["filename"],
                        "source": metadata["filename"],
                        "chunk_index": chunk_idx,
                        "category": metadata["category"],
                        "priority": metadata["priority"],
                    }
                })

    return chunks


def ingest_pdfs() -> Dict[str, Any]:
    """
    Main ingestion function: load all PDFs recursively, chunk them, return summary.

    Why rglob instead of glob?
    - Papers are organised into subdirectories by category
      (primary/, local/, comparative/, theory/)
    - rglob("*.pdf") finds all PDFs regardless of nesting depth
    - Category is captured from each file's parent folder name

    Returns:
        Dictionary with:
        - n_docs: Number of PDFs processed
        - n_chunks: Number of chunks created
        - avg_chunk_length: Average chunk character length
        - chunks: The actual chunk list (for use by retrieval.build())
    """
    papers_path = PAPERS_DIR

    if not papers_path.exists():
        print(f"Creating papers directory: {papers_path}")
        papers_path.mkdir(parents=True, exist_ok=True)
        return {"n_docs": 0, "n_chunks": 0, "avg_chunk_length": 0, "chunks": []}

    # Recurse into subdirectories
    pdf_files = sorted(papers_path.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found under {papers_path}")
        print("Add PDF files to data/papers/<category>/ and run ingestion again")
        return {"n_docs": 0, "n_chunks": 0, "avg_chunk_length": 0, "chunks": []}

    print(f"Found {len(pdf_files)} PDF file(s)")

    documents = []
    for pdf_file in pdf_files:
        print(f"  Loading [{pdf_file.parent.name}]: {pdf_file.name}")
        try:
            doc = load_pdf(pdf_file)
            documents.append(doc)
        except Exception as e:
            print(f"  ERROR loading {pdf_file.name}: {e}")

    chunks = chunk_documents(documents)

    avg_chunk_length = (
        sum(len(c["content"]) for c in chunks) / len(chunks) if chunks else 0
    )

    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Documents processed : {len(documents)}")
    print(f"Chunks created      : {len(chunks)}")
    print(f"Avg chunk length    : {avg_chunk_length:.1f} characters")
    print("=" * 50 + "\n")

    return {
        "n_docs": len(documents),
        "n_chunks": len(chunks),
        "avg_chunk_length": avg_chunk_length,
        "chunks": chunks,
    }


if __name__ == "__main__":
    result = ingest_pdfs()
    sys.exit(0 if result["n_chunks"] > 0 else 1)
