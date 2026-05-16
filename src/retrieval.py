"""
Retrieval Module for SARS
"""

import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion import ingest_pdfs, CHROMA_DIR

COLLECTION_NAME  = "augustus_temple"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
SEARCH_K         = 12   # was 5 — wider net catches more relevant passages
FETCH_K          = 50   # was 20 — larger MMR candidate pool enables real diversity
LAMBDA_MULT      = 0.65 # slightly more diversity weight than before (was 0.7)


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build(persist: bool = True) -> Chroma:
    print("Building vector store...")

    result = ingest_pdfs()

    embeddings = get_embedding_model()

    if result["n_chunks"] == 0:
        print("No chunks to index. Add PDFs to data/papers/<category>/ first.")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR) if persist else None,
        )

    langchain_docs = [
        Document(
            page_content=chunk["content"],
            metadata=chunk["metadata"],
        )
        for chunk in result["chunks"]
    ]

    chroma_path = CHROMA_DIR
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_path) if persist else None,
    )

    print(f"Vector store built: {len(langchain_docs)} chunks indexed")
    return vectorstore


def _load_vectorstore() -> Chroma:
    embeddings = get_embedding_model()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def query(
    query_text: str,
    vectorstore: Optional[Chroma] = None,
    k: int = SEARCH_K,
) -> List[Dict[str, Any]]:
    """
    MMR search returning up to k results with real similarity scores.

    Scores come from a separate similarity_search_with_relevance_scores call
    so they reflect actual semantic distance, not just rank position.
    """
    if vectorstore is None:
        vectorstore = _load_vectorstore()

    # MMR for the returned set (diversity-aware)
    mmr_docs = vectorstore.max_marginal_relevance_search(
        query_text,
        k=k,
        fetch_k=FETCH_K,
        lambda_mult=LAMBDA_MULT,
    )

    # Real similarity scores for the same query (to attach to results)
    scored = vectorstore.similarity_search_with_relevance_scores(
        query_text, k=k * 2
    )
    score_map: Dict[str, float] = {}
    for doc, score in scored:
        key = doc.page_content[:80]
        score_map[key] = round(float(score), 4)

    results = []
    for doc in mmr_docs:
        meta = doc.metadata
        source   = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk_index", "N/A")
        category = meta.get("category", "unknown")
        priority = meta.get("priority", 99)
        sim_score = score_map.get(doc.page_content[:80], None)

        results.append({
            "content":         doc.page_content,
            "source":          source,
            "page":            chunk_idx,
            "category":        category,
            "priority":        priority,
            "citation":        f"{source} [{category}], chunk {chunk_idx}",
            "relevance_score": sim_score,
        })

    # Sort by priority category first, then by score descending
    results.sort(key=lambda r: (
        r["priority"],
        -(r["relevance_score"] or 0),
    ))

    return results


if __name__ == "__main__":
    print("Building vector store...")
    vs = build()

    print("\nTest query: 'Corinthian column proportions Vitruvius'")
    results = query("Corinthian column proportions Vitruvius", vectorstore=vs)

    for i, r in enumerate(results, 1):
        score_str = f"{r['relevance_score']:.4f}" if r["relevance_score"] is not None else "n/a"
        print(f"\n--- Result {i} [{r['category']}] score={score_str} ---")
        print(f"Source : {r['source']}")
        print(f"Content: {r['content'][:200]}...")
