# SARS — Scientific Architectural Reconstruction System
### Templum Divi Augusti · Ankara · 25 BCE

A citation-grounded pipeline that reconstructs the first-century appearance of the Temple of Augustus in Ankara (Monumentum Ancyranum) using VLM structural analysis, RAG-retrieved academic constraints, and SDXL + ControlNet image generation.

<p align="center">
  <img src="web/images/idai1_heroimage.png" width="320" alt="Source photograph" />
  <img src="web/images/idai1_reconstructed.jpeg" width="320" alt="First-century BCE reconstruction" />
</p>
<p align="center"><em>Left: source photograph dominated by the Hacı Bayram minaret. Right: SARS reconstruction of the Templum Divi Augusti, 25 BCE.</em></p>

---

## Overview

The temple survives partially incorporated into the Hacı Bayram Veli Mosque (1427 CE) and is severely fragmented. SARS reconstructs its original appearance in four stages:

1. **Ingest** academic literature into a vector store (RAG)
2. **Analyze** photographs with a vision-language model
3. **Aggregate** findings into a reconstruction brief
4. **Generate** canonical views with SDXL + ControlNet

---

## Pipeline

| Step | Module | Runtime | Output |
|------|--------|---------|--------|
| 1 | `ingestion.py` + `retrieval.py` | Local CPU | ChromaDB vector store |
| 2 | `conditioning_prep.py` | Local CPU | Canny + depth maps |
| 3 | `vlm_analysis.py` | Local CPU (~14 min/img) | Per-image JSON analyses |
| 4 | `generation.py` | Colab GPU | SDXL reconstructions |

### Quick start

```bash
# Setup
cd /home/ece/VSCodeProjects/SARS
source sars-env/bin/activate

# Steps 1-3 (local)
python src/retrieval.py              # build vector store
python src/conditioning_prep.py      # extract canny/depth
python -m src.vlm_analysis --token hf_xxx   # analyze images

# Step 4 (Colab) — open notebooks/colab_sdxl.ipynb
```

---

## Worked example — image `idai1`

A single photograph from the DAI archive passed through the full pipeline.

<p align="center">
  <img src="web/images/idai1_minaretcleaned.png" width="220" alt="Cleaned canny" />
  <img src="web/images/idai1_depth.png" width="220" alt="Depth map" />
  <img src="web/images/idai1_reconstructed.jpeg" width="220" alt="Reconstruction" />
</p>
<p align="center"><em>
  1. Canny edges with minaret region zeroed &nbsp; · &nbsp;
  2. MiDaS monocular depth &nbsp; · &nbsp;
  3. SDXL + ControlNet reconstruction
</em></p>

---

## Project structure

```
data/
├── papers/             primary · comparative · theory · local
├── visual_sources/     full_shots · models · details · inscriptions · parallels · plans
├── chroma_db/          vector store (all-MiniLM-L6-v2 embeddings)
├── conditioning/       canny/ + depth/ + registry.json
└── analysis/           vlm_outputs/ — one JSON per image

src/
├── ingestion.py        PDF → chunks → ChromaDB
├── retrieval.py        MMR search over vector store
├── conditioning_prep.py Canny edge + MiDaS depth extraction
├── vlm_analysis.py     Gemma-3-4b-it per-image analysis
├── generation.py       SDXL + ControlNet reconstruction
└── evaluate.py         Analysis quality evaluation
```

---

## Hardcoded archaeological facts

These constraints are injected into every generation prompt. Numbers come from direct reading of DAI measured drawings.

```
8 columns across exterior facade (octastyle)
4 columns in antis (between facade and cella entrance)
12 columns total across front elevation
15 columns per side flank
```

Sources: Coulton 1976 · Hänlein-Schäfer 1985 · Mitchell & Waelkens 1998

---

## Running on Colab

### Upload to Drive (folder named `SARS_COLAB`)

```
src/    generation.py · retrieval.py · ingestion.py
data/   chroma_db/  ·  conditioning/  ·  analysis/vlm_outputs/
```

**Do not upload:** `data/visual_sources/`, `sars-env/`, `notebooks/`

### Notebook cells (`colab_sdxl.ipynb`)

| Cell | Purpose |
|------|---------|
| 1–7 | Setup: GPU check, install deps, mount Drive, load resources, load SDXL |
| 8b | Preview aggregated prompt + confirmed elements |
| **8c** | Generate 4 canonical views (~20 min T4 / ~3 min A100) |
| 8d | Display 2×2 grid + citations |
| 10 | Per-image batch generation (optional) |

---

## Design decisions

**Why aggregate all analyses instead of generating one image per photo?**
Any single photograph shows partial and contaminated fabric. Aggregating across sources and counting element confirmations by frequency yields a statistically grounded reconstruction brief.

**Why use Maison Carrée and Pula as preferred conditioning sources?**
Both are intact first-century Corinthian temples of the same type. Their canny maps provide uncontaminated structural geometry for ControlNet, unlike the ruined Ankara temple entangled with the mosque.

**Why not run VLM on architectural plans?**
VLM misreads column counts depending on plan angle. Column facts are hardcoded from direct reading of published DAI measured drawings.

---

## Prompt strategy

Every generated image is grounded in three layers:

1. **Hardcoded column facts** — from DAI plans
2. **Aggregated VLM findings** — top confirmed Roman elements seen in ≥2 sources
3. **RAG constraints** — dimensional sentences from academic PDFs

**Contamination handling:**
- Ottoman/mosque terms stripped from VLM-generated prompt components
- Canny maps from contaminated photos cleaned at prep time (minaret region zeroed, mosque-side trimmed)
- Source scoring: parallels +4 · models +2 · mosque-dominant −4

---

## Models

| Model | Purpose | Runtime |
|-------|---------|---------|
| `google/gemma-3-4b-it` | VLM structural analysis | Local CPU |
| `sentence-transformers/all-MiniLM-L6-v2` | RAG embeddings | Local CPU |
| `stabilityai/stable-diffusion-xl-base-1.0` | Image generation | Colab GPU |
| `diffusers/controlnet-canny-sdxl-1.0` | Structural conditioning | Colab GPU |
| `madebyollin/sdxl-vae-fp16-fix` | FP16-stable VAE | Colab GPU |

---

## Requirements

```bash
pip install -r requirements.txt
```

Key packages: `transformers` · `diffusers` · `accelerate` · `langchain-chroma` · `langchain-huggingface` · `chromadb` · `sentence-transformers` · `pymupdf` · `opencv-python` · `xformers`
