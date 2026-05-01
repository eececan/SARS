"""
Retrieval Module for SARS

This module handles vector storage, embedding, and MMR-based retrieval
for the Templum Divi Augusti research project.

Why sentence-transformers/all-MiniLM-L6-v2?
- Lightweight (~80MB) model optimized for semantic similarity
- Good balance of speed and quality for CPU inference
- Well-suited for academic/technical text retrieval
- No API keys required - runs entirely local

Why ChromaDB?
- Lightweight embedded vector store
- Easy persistence to filesystem
- Good integration with LangChain
- No external service dependencies

Why MMR (Maximum Marginal Relevance)?
- Balances relevance with diversity in results
- Prevents redundant results from same source
- lambda_mult=0.7 weights relevance slightly higher than diversity
- fetch_k=20 pools more candidates for better selection
"""

import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion import ingest_pdfs, CHROMA_DIR


# Configuration constants
COLLECTION_NAME = "augustus_temple"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEARCH_K = 5
FETCH_K = 20
LAMBDA_MULT = 0.7


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return the embedding model.

    Why separate this function?
    - Lazy loading delays model download until needed
    - Enables reuse across multiple operations
    - Single point for model configuration

    Returns:
        Configured HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build(persist: bool = True) -> Chroma:
    """
    Build or rebuild the vector store from all PDFs under data/papers/.

    Why call ingest_pdfs() once and reuse its chunks?
    - Avoids double-reading every PDF (the old code read files twice)
    - ingest_pdfs() now returns the chunk list alongside statistics

    Why wipe chroma_db before rebuilding?
    - Prevents stale embeddings from a previous run mixing with new ones
    - ChromaDB does not deduplicate on its own

    Args:
        persist: Whether to persist the vector store to disk (pass False for tests)

    Returns:
        Chroma vector store instance, ready to query
    """
    print("Building vector store...")

    result = ingest_pdfs()

    if result["n_chunks"] == 0:
        print("No chunks to index. Add PDFs to data/papers/<category>/ first.")
        embeddings = get_embedding_model()
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR) if persist else None,
        )

    # Convert chunks to LangChain Document objects
    langchain_docs = [
        Document(
            page_content=chunk["content"],
            metadata=chunk["metadata"],
        )
        for chunk in result["chunks"]
    ]

    embeddings = get_embedding_model()

    # Wipe existing store to avoid duplicate embeddings on rebuild
    chroma_path = CHROMA_DIR
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    persist_dir = str(chroma_path) if persist else None

    vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )

    print(f"Vector store built with {len(langchain_docs)} chunks")
    return vectorstore


def query(
    query_text: str,
    vectorstore: Optional[Chroma] = None,
    k: int = SEARCH_K,
) -> List[Dict[str, Any]]:
    """
    Search the vector store using MMR (Maximum Marginal Relevance).

    Why MMR over plain similarity search?
    - Returns more diverse results
    - Avoids multiple chunks from the same source dominating the list
    - Better for exploratory research where variety matters

    Why return a custom dict format?
    - Standardises output for the evaluation module
    - Includes all fields needed for citation (source, page, category, priority)
    - Flat structure is easier to work with than nested LangChain objects

    Args:
        query_text: The search query string
        vectorstore: Optional Chroma instance; loads from disk if None
        k: Number of results to return

    Returns:
        List of result dicts with keys:
        - content: The retrieved text chunk
        - source: Source filename
        - page: Chunk index (proxy for page position)
        - category: Relevance category (primary / local / comparative / theory)
        - priority: Integer priority (1 = most relevant category)
        - citation: Formatted citation string
        - relevance_score: Approximate positional score (1.0 → 0.5)
    """
    if vectorstore is None:
        embeddings = get_embedding_model()
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    results = vectorstore.max_marginal_relevance_search(
        query_text,
        k=k,
        fetch_k=FETCH_K,
        lambda_mult=LAMBDA_MULT,
    )

    formatted_results = []
    for i, doc in enumerate(results):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("chunk_index", "N/A")
        category = meta.get("category", "unknown")
        priority = meta.get("priority", 99)

        formatted_results.append({
            "content": doc.page_content,
            "source": source,
            "page": page,
            "category": category,
            "priority": priority,
            "citation": f"{source} [{category}], chunk {page}",
            # Positional approximation: rank 1 = 1.0, rank k = ~0.5
            "relevance_score": round(1.0 - (i / (2 * k)), 3),
        })

    return formatted_results


if __name__ == "__main__":
    print("Building vector store...")
    vs = build()

    print("\nTest query: 'Corinthian column proportions Vitruvius'")
    results = query("Corinthian column proportions Vitruvius", vectorstore=vs)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} [{r['category']}] ---")
        print(f"Source: {r['source']}")
        print(f"Content: {r['content'][:200]}...")
