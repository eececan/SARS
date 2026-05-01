"""
Evaluation Module for SARS

This module provides retrieval quality testing for the Templum Divi Augusti
research project.

Why these specific test queries?
- Cover different aspects of Augustan temple architecture
- Range from specific (column proportions) to conceptual (pronaos-cella relationship)
- Include Vitruvius references since he's the primary source
- Test both architectural terminology and historical context

Why manual validation?
- Automated metrics (precision/recall) don't capture semantic relevance
- Architectural research requires domain expert judgment
- Preview length of 100 chars is enough to assess relevance without overflow
"""

import sys
from pathlib import Path

# Import retrieval functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.retrieval import build, query, CHROMA_DIR, SEARCH_K


# Ten test queries relevant to Augustan temple architecture
TEST_QUERIES = [
    "Corinthian column proportions Vitruvius",
    "Augustan temple peristyle intercolumniation",
    "Roman pronaos cella relationship",
    "Templum Divi Augusti Ankara architecture",
    "Vitruvian orders column height diameter",
    "Roman temple podium stylobate architecture",
    "Augustan era temple decoration Corinthian capitals",
    "Roman temple entablature frieze cornice",
    "Temple of Augustus peripteral colonnade",
    "Vitruvius temple design principles Augustan"
]


def evaluate_retrieval(k: int = SEARCH_K) -> None:
    """
    Run evaluation on all test queries.
    
    Why run build() first?
    - Ensures vector store is up to date
    - Handles case where no documents exist yet
    - Provides clear error message if evaluation can't proceed
    
    Args:
        k: Number of results to retrieve per query
    """
    print("="*60)
    print("RETRIEVAL EVALUATION")
    print("="*60)
    print(f"Testing {len(TEST_QUERIES)} queries with k={k} results each\n")
    
    # Build vector store
    print("Building vector store...")
    vectorstore = build()
    print()
    
    # Run each query
    for query_idx, query_text in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"Query {query_idx}/{len(TEST_QUERIES)}: {query_text}")
        print("="*60)
        
        # Retrieve results
        results = query(query_text, vectorstore=vectorstore, k=k)
        
        if not results:
            print("No results found")
            continue
        
        # Print each result
        for i, result in enumerate(results, 1):
            source = result["source"]
            page = result["page"]
            category = result.get("category", "unknown")
            content_preview = result["content"][:100].replace("\n", " ")

            print(f"\n  Result {i}:")
            print(f"    Source   : {source} [{category}], chunk {page}")
            print(f"    Preview  : {content_preview}...")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nManual validation checklist:")
    print("  - Do results relate to the query topic?")
    print("  - Are sources diverse (not all from same document)?")
    print("  - Is content coherent (not fragmented)?")
    print("  - Are citations accurate?")
    print("="*60)


if __name__ == "__main__":
    evaluate_retrieval()