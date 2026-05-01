# SARS - Scientific Architectural Restoration System

**Step 1: PDF Ingestion and RAG Pipeline**

A research project for the Templum Divi Augusti in Ankara, combining RAG, VLMs, and diffusion models for architectural restoration research.

---

## Project Structure

```
SARS/
├── data/
│   ├── papers/          # Academic PDFs go here
│   └── chroma_db/       # Vector store persisted here
├── src/
│   ├── ingestion.py     # PDF loading, text extraction, chunking
│   ├── retrieval.py      # Vector store, embedding, MMR search
│   └── evaluate.py       # Retrieval quality testing
├── notebooks/
│   └── test_retrieval.ipynb
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
cd SARS
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download the embedding model (~80MB) and PyTorch (CPU version).

---

## How to Add PDFs

1. Place PDF files in `data/papers/`
2. Supported: `.pdf` files
3. Recommended: Academic papers on Roman architecture, Vitruvius, Augustan temples

Example:
```bash
cp /path/to/vitruvius.pdf data/papers/
cp /path/to/augustan_temple_paper.pdf data/papers/
```

---

## How to Run Ingestion

### Option A: Python script

```bash
python src/ingestion.py
```

### Option B: From Python code

```python
from src.ingestion import ingest_pdfs

result = ingest_pdfs()
# Returns: {"n_docs": N, "n_chunks": N, "avg_chunk_length": N}
```

**Output**: Prints summary with document count, chunk count, and average chunk length.

---

## How to Run Evaluation

### Option A: Run all test queries

```bash
python src/evaluate.py
```

This runs 10 predefined queries relevant to Augustan temple architecture:
1. Corinthian column proportions Vitruvius
2. Augustan temple peristyle intercolumniation
3. Roman pronaos cella relationship
4. Templum Divi Augusti Ankara architecture
5. Vitruvian orders column height diameter
6. Roman temple podium stylobate architecture
7. Augustan era temple decoration Corinthian capitals
8. Roman temple entablature frieze cornice
9. Temple of Augustus peripteral colonnade
10. Vitruvius temple design principles Augustan

### Option B: Custom query

```python
from src.retrieval import build, query

# Build vector store
vs = build()

# Query
results = query("Your research question here", vectorstore=vs, k=5)
```

### Option C: Jupyter Notebook

```bash
code notebooks/test_retrieval.ipynb
```

Run cells to interactively test retrieval.

---

## Technical Details

| Component | Implementation |
|-----------|-----------------|
| PDF Extraction | PyMuPDF (fitz) |
| Text Chunking | RecursiveCharacterTextSplitter (512 chars, 64 overlap) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (persisted to `data/chroma_db/`) |
| Search | MMR (k=5, fetch_k=20, lambda_mult=0.7) |
| CPU Only | No GPU required |

---

## Next Steps (Future)

- **Step 2**: VLM integration for image-based queries
- **Step 3**: Diffusion models for restoration visualization
- **Step 4**: Web interface for interactive research

---

## Troubleshooting

### No PDFs found
```
No PDF files found in data/papers/
Add PDF files to data/papers/ and run ingestion again
```
→ Add PDFs to `data/papers/` and re-run ingestion.

### Model download slow
First run downloads ~80MB embedding model. Subsequent runs are faster.

### Empty results
- Verify PDFs contain searchable text (not scanned images)
- Check that chunk size filtering isn't removing content (min 100 chars)

---

## License

Research project. Add appropriate license for your institution.