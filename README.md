# SARS — Scholarly Architectural Reconstruction System
### Templum Divi Augusti, Ankara · 25 BCE

A citation-grounded pipeline that reconstructs the first-century appearance of the Temple of Augustus in Ankara (Monumentum Ancyranum) using VLM structural analysis, RAG-retrieved academic constraints, and SDXL + ControlNet image generation.

---

## What it does

The temple survives partially incorporated into the Hacı Bayram Veli Mosque (1427 CE) and is severely fragmented. This system:

1. **Ingests** 18 academic PDFs across four categories (primary, comparative, theory, local) into a vector store
2. **Analyzes** 62 photographs and architectural models using Gemma-3-4b-it VLM, separating Roman from Ottoman fabric per image
3. **Aggregates** all 62 analyses into a single reconstruction brief — confirmed architectural elements, stratigraphic confidence, and unique academic citations
4. **Generates** four canonical views (front elevation, three-quarter, side, interior) using SDXL + Canny ControlNet, grounded in hardcoded archaeological facts and RAG-retrieved dimensional constraints

---

## Architecture

```
data/
├── papers/
│   ├── primary/       # 8 site-specific publications
│   ├── comparative/   # parallel temple studies
│   ├── theory/        # Vitruvius Books III & IV
│   └── local/         # Ankara urban context
├── chroma_db/         # vector store (sentence-transformers/all-MiniLM-L6-v2)
├── visual_sources/
│   ├── full_shots/        # 27 photographs (Wikimedia Commons, DAI)
│   ├── architectural_models/  # 6 reconstruction models
│   ├── details/           # capital, inscription, entablature close-ups
│   ├── inscriptions/      # Res Gestae panels
│   ├── parallels/
│   │   ├── maison_carree/ # 10 images (Nîmes, best-preserved Augustan temple)
│   │   └── pula/          # 5 images (Pula Augustus Temple)
│   └── plans/             # 9 measured drawings (DAI archive)
├── conditioning/
│   ├── canny/         # edge maps per image
│   └── registry/      # conditioning_registry.json
└── analysis/
    └── vlm_outputs/   # 62 *_analysis.json files
```

```
src/
├── ingestion.py        # PDF → chunks → ChromaDB
├── retrieval.py        # MMR search over vector store
├── conditioning_prep.py # Canny edge + MiDaS depth extraction
├── vlm_analysis.py     # Gemma-3-4b-it per-image analysis (local CPU)
├── generation.py       # SDXL + ControlNet reconstruction
└── evaluate.py         # Analysis quality evaluation
```

---

## Hardcoded archaeological facts

Entered from direct reading of DAI measured drawings and published scholarship. Injected into every prompt:

```
8 columns across exterior facade (octastyle)
4 columns in antis (between facade and cella entrance)
12 columns total across front elevation
15 columns per side flank
```

Sources: Coulton 1976; Hänlein-Schäfer 1985; Mitchell & Waelkens 1998.

---

## Pipeline steps

### Step 1 — Build vector store (run once locally)
```bash
cd /home/ece/VSCodeProjects/SARS
source sars-env/bin/activate
python src/retrieval.py
```
Ingests all PDFs and persists `data/chroma_db/`.

### Step 2 — Extract conditioning maps (run once locally)
```bash
python src/conditioning_prep.py
```
Generates canny edge maps for all images into `data/conditioning/canny/`.

### Step 3 — VLM analysis (run locally on CPU)
```bash
export HF_TOKEN=your_token
python src/vlm_analysis.py
```
Analyzes all 62 images (~14 min/image on CPU). Outputs `data/analysis/vlm_outputs/*_analysis.json`.
Resume-safe — skips already-completed images.

### Step 4 — SDXL generation (run on Colab GPU)
Upload to Google Drive and run `notebooks/colab_sdxl.ipynb`.

---

## Running on Google Colab

### What to upload to Drive (folder named `SARS`)

```
src/
  generation.py
  retrieval.py
  ingestion.py
data/
  chroma_db/                          ← entire folder
  conditioning/
    registry/
      conditioning_registry.json
    canny/                            ← entire folder (subfolders)
  analysis/
    vlm_outputs/                      ← all 62 *_analysis.json files
```

Do **not** upload: `data/visual_sources/`, `sars-env/`, `notebooks/`.

### Notebook cell order

| Cell | What it does |
|------|-------------|
| 1 | Hardware check — raises if no GPU |
| 2 | Install dependencies |
| 3 | Mount Drive, auto-detect project folder |
| 4 | Remap local paths → Colab paths in registry |
| 5 | Load vector store |
| 6 | Check VLM analysis outputs exist |
| 7 | Load SDXL + Canny ControlNet (~7GB, cached after first run) |
| **8b** | Preview aggregated prompt and confirmed elements list |
| **8c** | Generate 4 canonical views (~20 min T4 / ~3 min A100) |
| **8d** | Display 2×2 grid + full citation list |

Skip cells 8, 9, 10, 11 — those are the per-image batch path.

---

## Prompt strategy

Every generated image is grounded in three layers:

1. **Hardcoded column facts** — octastyle, 12 front total, 15 per side (from plans)
2. **Aggregated VLM findings** — top 20 confirmed Roman elements seen in ≥2 sources across all 62 analyses
3. **RAG constraints** — dimensional sentences from academic PDFs (Vitruvius proportions, structural survey data)

Hard 70-word prompt limit enforced to stay within CLIP's token window.

Contamination filtering:
- Ottoman/mosque terms stripped from all VLM-generated prompt components
- Canny maps from mosque-contaminated photos are cleaned before ControlNet use (top 25% removed for minaret shots, left 20% for mosque-adjacent shots)
- Source images scored before selection: parallels +4, models +2, mosque-dominant −4

---

## Key design decisions

**Why aggregate all 62 analyses instead of generating one image per photograph?**
Any single photograph shows partial and contaminated fabric. Aggregating across all sources and counting element confirmations by frequency gives a statistically grounded reconstruction brief rather than one image's perspective.

**Why use Maison Carrée and Pula as preferred conditioning sources?**
These are intact first-century Corinthian temples of the same type and period. Their canny maps provide uncontaminated structural geometry for ControlNet, unlike photographs of the ruined Ankara temple mixed with the mosque.

**Why not run VLM on architectural plans?**
VLM misreads column counts depending on plan angle — a side-view plan returns different numbers than a front-view plan. Column facts are hardcoded from direct reading of the published DAI measured drawings.

---

## Models

| Model | Purpose | Runtime |
|-------|---------|---------|
| `google/gemma-3-4b-it` | VLM structural analysis | Local CPU |
| `sentence-transformers/all-MiniLM-L6-v2` | Text embeddings for RAG | Local CPU |
| `stabilityai/stable-diffusion-xl-base-1.0` | Image generation | Colab GPU |
| `diffusers/controlnet-canny-sdxl-1.0` | Structural conditioning | Colab GPU |
| `madebyollin/sdxl-vae-fp16-fix` | FP16-stable VAE | Colab GPU |

---

## Requirements

```bash
pip install -r requirements.txt
```

Key packages: `transformers`, `diffusers`, `accelerate`, `langchain`, `langchain-chroma`, `langchain-huggingface`, `chromadb`, `sentence-transformers`, `pymupdf`, `opencv-python`, `psutil`, `xformers`
