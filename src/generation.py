"""
Step 4: SDXL + ControlNet Reconstruction
Generates first-century BCE visualizations of Templum Divi Augusti
grounded in VLM analysis and RAG-retrieved academic citations.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_hardware() -> dict:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name}")
        print(f"VRAM: {vram:.1f}GB")
        print("Estimated generation time: ~3-5 min/image")
        return {
            "device": "cuda",
            "torch_dtype": torch.float16,
            "variant": "fp16",
            "use_cpu_offload": False,
            "gpu_name": gpu_name,
            "vram_gb": round(vram, 1),
        }
    else:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
        print(f"CPU mode: {ram:.1f}GB RAM")
        print("Estimated generation time: ~60-90 min/image")
        print("Consider using Google Colab for GPU.")
        return {
            "device": "cpu",
            "torch_dtype": torch.float32,
            "variant": None,
            "use_cpu_offload": True,
            "gpu_name": None,
            "ram_gb": round(ram, 1),
        }


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_sdxl_pipeline(
    hw: dict,
    use_canny: bool = True,
    use_depth: bool = False,
):
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
    )

    if not use_canny and not use_depth:
        raise ValueError("At least one of use_canny or use_depth must be True.")

    controlnets = []

    if use_canny:
        print("Loading Canny ControlNet...")
        controlnets.append(
            ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=hw["torch_dtype"],
            )
        )

    if use_depth:
        print("Loading Depth ControlNet...")
        controlnets.append(
            ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=hw["torch_dtype"],
            )
        )

    controlnet_arg = controlnets[0] if len(controlnets) == 1 else controlnets

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=hw["torch_dtype"],
    )

    print("Loading SDXL base...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet_arg,
        vae=vae,
        torch_dtype=hw["torch_dtype"],
        variant=hw["variant"],
        use_safetensors=True,
    )

    if hw["device"] == "cuda":
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    else:
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

    return pipe


# ---------------------------------------------------------------------------
# Confirmed archaeological facts (hardcoded — from published scholarship)
# ---------------------------------------------------------------------------

# Coulton 1976, Hänlein-Schäfer 1985, Mitchell & Waelkens 1998:
# 8 columns across the exterior facade (octastyle),
# 4 columns between the exterior columns and the cella wall (in antis),
# giving 12 columns total across the front.
# 15 columns along each side (including corner columns).
CONFIRMED_COLUMN_FACTS = (
    "STRICT ARCHITECTURAL CONSTRAINTS: "
    "octastyle facade with EXACTLY 8 exterior columns across front, "
    "4 columns in antis (between exterior and cella), "
    "12 columns TOTAL across front elevation, "
    "15 columns per side flank (including corners), "
    "peripteral layout with continuous colonnade all sides"
)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

OTTOMAN_TERMS = [
    "ottoman", "mosque", "pointed arch", "islamic", "minaret",
    "hacı bayram", "haci bayram", "geometric ornament", "byzantine", "medieval",
]

MODERN_TERMS = [
    "modern", "scaffolding", "steel", "concrete", "metal support",
    "overcast", "damaged", "weathered", "fragmentary", "restoration",
    "ruins", "destroyed", "partially", "barrier",
]

STYLE_PREFIX = (
    "first century BCE Roman temple, "
    "Templum Divi Augusti Ankara, "
    "Corinthian order, white Pentelic marble, "
    "pristine complete condition, "
    "photorealistic, golden Mediterranean light, "
)

STYLE_SUFFIX = (
    ", peripteral octastyle, "
    "no people, clear blue sky"
)


def sanitize_prompt_component(text: str) -> str:
    if not text:
        return text
    all_terms = OTTOMAN_TERMS + MODERN_TERMS
    phrases = [p.strip() for p in text.split(",")]
    clean = [p for p in phrases if not any(t in p.lower() for t in all_terms)]
    return ", ".join(clean)


def load_plan_context(analysis_dir: str) -> str:
    constraints = []

    for f in Path(analysis_dir).glob("*_analysis.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        if data.get("pass") != "plans_extraction" or "error" in data:
            continue

        parts = []
        if data.get("columns_front"):
            parts.append(f"{data['columns_front']} columns across facade")
        if data.get("columns_side"):
            parts.append(f"{data['columns_side']} columns per side")
        if data.get("temple_length_m") and data.get("temple_width_m"):
            parts.append(f"temple {data['temple_length_m']}m x {data['temple_width_m']}m")
        if data.get("cella_length_m"):
            parts.append(f"cella {data['cella_length_m']}m x {data['cella_width_m']}m")
        if data.get("column_diameter_m"):
            parts.append(f"column diameter {data['column_diameter_m']}m")
        if data.get("intercolumniation_m"):
            parts.append(f"intercolumniation {data['intercolumniation_m']}m")
        if data.get("sdxl_prompt_component"):
            parts.append(data["sdxl_prompt_component"])

        constraints.extend(parts)

    if not constraints:
        return ""

    seen = set()
    unique = []
    for c in constraints:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return "documented floor plan dimensions: " + ", ".join(unique)


def build_positive_prompt(
    analysis: dict,
    vector_store=None,
    plan_context: str = "",
) -> str:
    sdxl_component = sanitize_prompt_component(
        analysis.get("sdxl_prompt_component", "")
    )

    rag_supplement = ""
    if vector_store:
        try:
            from src.retrieval import query as rag_query
            rag_q = analysis.get(
                "rag_search_query",
                "Corinthian column proportions",
            )
            sub_queries = [q.strip() for q in rag_q.split(",")[:2] if q.strip()]
            all_docs = []
            seen = set()
            for q in sub_queries:
                docs = rag_query(q, vectorstore=vector_store, k=2)
                for d in docs:
                    if d["citation"] not in seen:
                        seen.add(d["citation"])
                        all_docs.append(d)

            constraints = []
            for doc in all_docs[:3]:
                for sent in doc["content"].split("."):
                    if any(c.isdigit() for c in sent) and 10 < len(sent) < 80:
                        constraints.append(sent.strip()[:60])
                        break

            if constraints:
                rag_supplement = constraints[0]
        except Exception:
            pass

    parts = [
        STYLE_PREFIX,
        f"{CONFIRMED_COLUMN_FACTS}, " + sdxl_component[:80] if sdxl_component else CONFIRMED_COLUMN_FACTS,
        plan_context[:40] if plan_context else "",
        rag_supplement[:60] if rag_supplement else "",
        STYLE_SUFFIX,
    ]

    positive = ", ".join(p for p in parts if p)

    words = positive.split()
    if len(words) > 70:
        positive = " ".join(words[:70])

    print(f"  Prompt words: {len(positive.split())}")
    return positive


def build_negative_prompt(analysis: dict) -> str:
    # Note: do NOT enumerate "wrong" column counts (e.g. "6 columns, 7 columns")
    # — SDXL's text encoder has no logical NOT operator, so listing those tokens
    # actually injects those concepts as features. Let the canny conditioning
    # hold geometric structure; the prompt only handles style/material.
    base = (
        "ruins, damage, missing sections, broken stone, weathering, moss, "
        "vegetation, cracks, graffiti, scaffolding, metal barriers, "
        "concrete repairs, steel framework, tourist infrastructure, "
        "modern buildings, power lines, cars, people, contemporary elements, "
        "anachronistic details, fantasy architecture, CGI artifacts, "
        "oversaturation, lens flare, HDR, 3D render look, plastic appearance, "
        "low quality, blurry, deformed, "
        "disproportionate, asymmetrical layout, truncated structure"
    )
    mosque_addition = (
        ", Ottoman architecture, pointed arches, Islamic geometric ornament, "
        "mosque, minaret, Ottoman masonry, Arabic calligraphy, Islamic tiles, "
        "muqarnas, Byzantine decoration, medieval elements, "
        "post-classical architecture"
    )
    if analysis.get("mosque_interference") in ("partial", "dominant"):
        return base + mosque_addition
    return base


# ---------------------------------------------------------------------------
# Conditioning image loading
# ---------------------------------------------------------------------------

def load_conditioning_image(
    source_filename: str,
    conditioning_registry_path: str,
    mode: str = "canny",
) -> Image.Image:
    with open(conditioning_registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    entry = next(
        (r for r in registry if r["source_filename"].strip() == source_filename.strip()),
        None,
    )
    if entry is None:
        raise ValueError(f"No registry entry for {source_filename}")

    path = entry.get(f"{mode}_path")
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"No {mode} conditioning image for {source_filename}")

    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_analysis(a: dict) -> int:
    score = 0
    if a.get("credibility_tier") == "high":
        score += 2
    if a.get("source_folder") == "full_shots":
        score += 2
    if a.get("mosque_interference") in ("none", "minimal"):
        score += 1
    if a.get("roman_fabric_quality") in ("excellent", "good"):
        score += 1
    sc = a.get("stratigraphic_confidence", {})
    if sc.get("confirmed_roman", 0) >= 0.8:
        score += 1
    return score


# ---------------------------------------------------------------------------
# Canny map cleaning
# ---------------------------------------------------------------------------

_CLEAN_TERMS = ["minaret", "ottoman", "mosque", "scaffold", "support", "haci"]


def _needs_cleaning(filename: str) -> bool:
    return any(t in filename.lower() for t in _CLEAN_TERMS)


def clean_canny_map(
    canny_path: str,
    analysis: dict,
    output_dir: str = "data/conditioning/canny_clean",
) -> str:
    import cv2
    import numpy as np

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    img = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    cleaned = img.copy()
    filename = analysis.get("source_filename", "").lower()

    if any(t in filename for t in ["minaret", "ottoman"]):
        cutoff = int(h * 0.25)
        cleaned[:cutoff, :] = 0
        print("  [CLEAN] Removed top 25% (minaret region)")

    if any(t in filename for t in ["scaffold", "support", "metal"]):
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
        print("  [CLEAN] Applied morphological opening (scaffolding removal)")

    if any(t in filename for t in ["mosque", "haci", "north"]):
        cutoff = int(w * 0.20)
        cleaned[:, :cutoff] = 0
        print("  [CLEAN] Removed left 20% (mosque region)")

    stem = Path(canny_path).stem
    out_path = Path(output_dir) / f"{stem}_clean.png"
    cv2.imwrite(str(out_path), cleaned)
    return str(out_path)


def _load_canny(
    source_filename: str,
    conditioning_registry_path: str,
    analysis: dict,
) -> Image.Image:
    if _needs_cleaning(source_filename):
        import tempfile
        raw = load_conditioning_image(source_filename, conditioning_registry_path, mode="canny")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            raw.save(tmp.name)
            clean_path = clean_canny_map(tmp.name, analysis)
        print("  [CLEAN] Using cleaned canny map")
        return Image.open(clean_path).convert("RGB")
    return load_conditioning_image(source_filename, conditioning_registry_path, mode="canny")


# ---------------------------------------------------------------------------
# Single image generation
# ---------------------------------------------------------------------------

def generate_reconstruction(
    analysis: dict,
    conditioning_registry_path: str,
    pipe,
    hw: dict,
    vector_store=None,
    output_dir: str = "data/outputs",
    num_inference_steps: int = 40,
    guidance_scale: float = 7.0,
    controlnet_conditioning_scale: float = 1.0,
    use_canny: bool = True,
    use_depth: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_filename = analysis["source_filename"]
    stem = Path(source_filename.strip()).stem

    positive = build_positive_prompt(analysis, vector_store)
    negative = build_negative_prompt(analysis)

    print(f"  Positive: {len(positive)} chars")
    print(f"  Negative: {len(negative)} chars")

    conditioning_images = []
    conditioning_scales = []

    if use_canny:
        canny_img = _load_canny(source_filename, conditioning_registry_path, analysis)
        conditioning_images.append(canny_img)
        conditioning_scales.append(controlnet_conditioning_scale)

    if use_depth:
        depth_img = load_conditioning_image(
            source_filename, conditioning_registry_path, mode="depth"
        )
        conditioning_images.append(depth_img)
        conditioning_scales.append(controlnet_conditioning_scale * 0.6)

    image_arg = conditioning_images[0] if len(conditioning_images) == 1 else conditioning_images
    scale_arg = conditioning_scales[0] if len(conditioning_scales) == 1 else conditioning_scales

    result_image = pipe(
        prompt=positive,
        negative_prompt=negative,
        image=image_arg,
        controlnet_conditioning_scale=scale_arg,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=1024,
        height=1024,
    ).images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_path = recon_dir / f"{stem}_reconstruction_{timestamp}.png"
    result_image.save(reconstruction_path)

    provenance = {
        "source_image": source_filename,
        "source_folder": analysis.get("source_folder", ""),
        "credibility_tier": analysis.get("credibility_tier", ""),
        "mosque_interference": analysis.get("mosque_interference", ""),
        "roman_fabric_quality": analysis.get("roman_fabric_quality", ""),
        "stratigraphic_confidence": analysis.get("stratigraphic_confidence", {}),
        "confidence_zones": analysis.get("confidence_zones", {}),
        "citations_used": analysis.get("retrieved_citations", []),
        "positive_prompt": positive,
        "negative_prompt": negative,
        "generation_params": {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "use_canny": use_canny,
            "use_depth": use_depth,
            "hardware": hw.get("device"),
            "gpu_name": hw.get("gpu_name"),
        },
        "timestamp": timestamp,
        "reconstruction_path": str(reconstruction_path),
    }

    prov_dir = output_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = prov_dir / f"{stem}_provenance_{timestamp}.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {reconstruction_path.name}")

    return {
        "reconstruction_path": str(reconstruction_path),
        "provenance_path": str(provenance_path),
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def run_batch_generation(
    analysis_dir: str,
    conditioning_registry_path: str,
    vector_store,
    output_dir: str,
    pipe,
    hw: dict,
    num_inference_steps: int = 40,
    guidance_scale: float = 7.0,
    resume: bool = True,
    max_images: int = None,
) -> list:
    output_dir = Path(output_dir)
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    analyses = []
    for f in sorted(Path(analysis_dir).glob("*_analysis.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if "error" not in data and data.get("sdxl_prompt_component"):
            analyses.append(data)

    analyses.sort(key=_score_analysis, reverse=True)

    if resume and recon_dir.exists():
        done = {
            f.stem.split("_reconstruction")[0]
            for f in recon_dir.glob("*_reconstruction_*.png")
        }
        before = len(analyses)
        analyses = [
            a for a in analyses
            if Path(a["source_filename"].strip()).stem not in done
        ]
        print(f"Resuming: skipped {before - len(analyses)} already generated")

    if max_images:
        analyses = analyses[:max_images]

    total = len(analyses)
    results = []
    failed = []

    print(f"Generating {total} reconstructions")
    print(f"Steps: {num_inference_steps} | Guidance: {guidance_scale}")
    print()

    for i, analysis in enumerate(analyses):
        filename = analysis["source_filename"]
        folder = analysis.get("source_folder", "")
        is_plan = folder == "plans"
        # Higher scale = stronger adherence to source canny geometry.
        # Plans need near-perfect adherence; other shots need strong adherence
        # so SDXL reconstructs ON the input rather than hallucinating layout.
        conditioning_scale = 1.1 if is_plan else 1.0

        print(f"[{i+1}/{total}] {filename}")
        print(f"  folder: {folder} | scale: {conditioning_scale}"
              f"{'  [PLAN]' if is_plan else ''}")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU mem: {mem:.1f}GB")

        start = time.time()

        try:
            result = generate_reconstruction(
                analysis=analysis,
                conditioning_registry_path=conditioning_registry_path,
                pipe=pipe,
                hw=hw,
                vector_store=vector_store,
                output_dir=str(output_dir),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=conditioning_scale,
                use_canny=True,
                use_depth=False,
            )
            elapsed = time.time() - start
            print(f"  ✓ {elapsed:.0f}s")
            results.append(result)

        except FileNotFoundError as e:
            print(f"  ⚠ Skipped (no conditioning): {e}")
            failed.append(filename)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(filename)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"Complete: {len(results)} generated, {len(failed)} skipped/failed")
    if failed:
        print("Skipped/failed:")
        for name in failed:
            print(f"  {name}")

    return results


# ---------------------------------------------------------------------------
# Aggregated prompt from all analyses
# ---------------------------------------------------------------------------

def aggregate_all_analyses(analysis_dir: str) -> dict:
    from collections import defaultdict

    elements = defaultdict(list)
    all_rag_queries = []
    all_citations = []
    all_sdxl_components = []
    mosque_flags = []
    confidence_scores = defaultdict(list)

    for f in Path(analysis_dir).glob("*_analysis.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        if "error" in data or data.get("source_folder") == "plans":
            continue

        # Exclude contaminated full_shots from aggregation (mosque interference present)
        # Only use: parallels, models, or full_shots with mosque=none
        folder = data.get("source_folder", "")
        mosque = data.get("mosque_interference", "")
        if folder == "full_shots" and mosque in ("partial", "dominant"):
            continue  # Skip contaminated full_shots
        if folder in ("details", "inscriptions"):
            continue  # Skip detail crops (too specific)

        for elem in data.get("architectural_elements", []):
            if elem.get("period") == "Roman":
                key = elem.get("element", "")
                if key:
                    elements[key].append(elem.get("confidence", ""))

        q = data.get("rag_search_query", "")
        if q:
            all_rag_queries.append(q)

        for c in data.get("retrieved_citations", []):
            if c not in all_citations:
                all_citations.append(c)

        comp = data.get("sdxl_prompt_component", "")
        if comp:
            all_sdxl_components.append(comp)

        mi = data.get("mosque_interference", "")
        if mi:
            mosque_flags.append(mi)

        for k, v in data.get("stratigraphic_confidence", {}).items():
            if isinstance(v, (int, float)):
                confidence_scores[k].append(v)

    confirmed = []
    inferred = []
    for elem, confidences in sorted(
        elements.items(), key=lambda x: len(x[1]), reverse=True
    ):
        count = len(confidences)
        high_conf = confidences.count("high")
        if count >= 2 or high_conf >= 1:
            confirmed.append(f"{elem} (confirmed in {count} sources)")
        else:
            inferred.append(elem)

    avg_confidence = {
        k: round(sum(v) / len(v), 2)
        for k, v in confidence_scores.items()
        if v
    }

    dominant_mosque = (
        max(set(mosque_flags), key=mosque_flags.count)
        if mosque_flags else "unknown"
    )

    return {
        "confirmed_elements": confirmed,
        "inferred_elements": inferred,
        "all_citations": all_citations,
        "all_sdxl_components": all_sdxl_components,
        "all_rag_queries": all_rag_queries,
        "avg_stratigraphic_confidence": avg_confidence,
        "dominant_mosque_interference": dominant_mosque,
        "source_count": len(elements),
    }


def build_aggregated_prompt(
    aggregated: dict,
    vector_store=None,
    analysis_dir: str = "data/analysis/vlm_outputs",
) -> tuple:
    confirmed_str = ", ".join(
        e.split(" (confirmed")[0]
        for e in aggregated["confirmed_elements"][:20]
    )

    rag_supplement = ""
    if vector_store:
        try:
            from src.retrieval import query as rag_query
            combined_query = " ".join(
                q.split(",")[0].strip()
                for q in aggregated["all_rag_queries"][:5]
            )
            docs = rag_query(combined_query, vectorstore=vector_store, k=7)
            constraints = []
            for doc in docs:
                for sent in doc["content"].split("."):
                    if any(c.isdigit() for c in sent) and len(sent) > 20:
                        constraints.append(sent.strip())
            if constraints:
                rag_supplement = (
                    " proportional constraints: " + ". ".join(constraints[:5])
                )
        except Exception:
            pass

    positive = (
        "photorealistic architectural render, "
        "first century BCE Roman temple, "
        "Templum Divi Augusti Ankara 25 BCE, "
        "pristine marble construction, "
        "Mediterranean golden hour sunlight, "
        f"{CONFIRMED_COLUMN_FACTS}, "
        f"{confirmed_str}, "
        "peripteral Corinthian order, "
        "complete entablature and pediment, "
        "white Pentelic marble podium with steps, "
        f"{rag_supplement}, "
        "archaeologically accurate reconstruction, "
        "sharp architectural detail, "
        "no people, clear blue Anatolian sky"
    )

    base_negative = (
        "ruins, damage, missing sections, broken stone, weathering, moss, "
        "vegetation, cracks, scaffolding, metal barriers, concrete repairs, "
        "modern buildings, power lines, cars, people, fantasy architecture, "
        "incorrect proportions, CGI artifacts, oversaturation, lens flare, "
        "HDR, 3D render look, low quality, blurry"
    )

    if aggregated.get("dominant_mosque_interference") in ("partial", "dominant"):
        base_negative += (
            ", Ottoman architecture, pointed arches, Islamic ornament, "
            "mosque, minaret, Byzantine elements, medieval architecture"
        )

    return positive, base_negative


# ---------------------------------------------------------------------------
# Canonical views using aggregated prompt
# ---------------------------------------------------------------------------

CANONICAL_VIEWS = [
    {
        "name": "front_elevation",
        "title": "Front Elevation",
        "prefer_folder": "parallels/maison_carree",
        "prefer_keywords": ["front"],
        "fallback_folder": "architectural_models",
        "conditioning_scale": 0.90,
    },
    {
        "name": "three_quarter",
        "title": "Three-Quarter View",
        "prefer_folder": "parallels/maison_carree",
        "prefer_keywords": ["column"],
        "fallback_folder": "architectural_models",
        "conditioning_scale": 0.85,
    },
    {
        "name": "side_elevation",
        "title": "Side Elevation",
        "prefer_folder": "parallels/maison_carree",
        "prefer_keywords": ["side", "column"],
        "fallback_folder": "parallels/pula",
        "conditioning_scale": 0.90,
    },
    {
        "name": "interior_cella",
        "title": "Interior — Cella",
        "prefer_folder": "full_shots",
        "prefer_keywords": ["interior", "cella", "inside"],
        "fallback_folder": "full_shots",
        "conditioning_scale": 0.80,
    },
]


def _score_for_generation(analysis: dict, entry: dict) -> int:
    score = _score_analysis(analysis)
    filename = analysis.get("source_filename", "").lower()
    folder = analysis.get("source_folder", "")

    # Strongly prefer clean parallel temples and models
    if folder in ("parallels/maison_carree", "parallels/pula", "parallels"):
        score += 4
    if folder == "architectural_models":
        score += 2
    if "reconstruction" in filename:
        score += 2

    # Penalize contaminated sources
    penalize = [
        "minaret", "mosque", "ottoman", "scaffolding", "scaffold",
        "support", "card", "display", "label", "rubble", "debris",
        "destroyed", "haci", "hacibayram",
    ]
    if any(t in filename for t in penalize):
        score -= 4

    # Penalize full_shots with mosque content
    if folder == "full_shots" and analysis.get("mosque_interference") in ("partial", "dominant"):
        score -= 3

    if entry.get("has_ottoman_elements"):
        score -= 3

    return score


def generate_canonical_views(
    analysis_dir: str,
    conditioning_registry_path: str,
    pipe,
    hw: dict,
    vector_store=None,
    output_dir: str = "data/outputs",
    num_inference_steps: int = 40,
    guidance_scale: float = 11.0,
    plan_context: str = "",
) -> list:
    if not plan_context:
        plan_context = load_plan_context(analysis_dir)
    print(f"  Plan context: {len(plan_context)} chars")

    print("Aggregating all analyses...")
    aggregated = aggregate_all_analyses(analysis_dir)
    print(f"  Sources aggregated: {aggregated['source_count']}")
    print(f"  Confirmed elements: {len(aggregated['confirmed_elements'])}")
    print(f"  Total citations: {len(aggregated['all_citations'])}")

    positive, negative = build_aggregated_prompt(aggregated, vector_store)
    print(f"\nAggregated positive prompt: {len(positive)} chars")
    print(f"  {positive[:200]}...")

    with open(conditioning_registry_path, encoding="utf-8") as f:
        registry = json.load(f)
    registry_map = {e["source_filename"].strip(): e for e in registry}

    all_analyses = []
    for f in Path(analysis_dir).glob("*_analysis.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        if "error" not in data and data.get("source_folder") != "plans":
            all_analyses.append(data)

    output_dir = Path(output_dir)
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    prov_dir = output_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for view in CANONICAL_VIEWS:
        print(f"\n── {view['title']} ──────────")

        candidate = None

        # Collect preferred folder + keyword matches, score and sort
        preferred = []
        for a in all_analyses:
            if a.get("source_folder") != view["prefer_folder"]:
                continue
            fname = a.get("source_filename", "").lower()
            if not any(kw in fname for kw in view["prefer_keywords"]):
                continue
            entry = registry_map.get(a["source_filename"].strip(), {})
            if entry.get("canny_path") and Path(entry["canny_path"]).exists():
                preferred.append((a, entry))

        if preferred:
            preferred.sort(key=lambda x: _score_for_generation(x[0], x[1]), reverse=True)
            candidate, _ = preferred[0]
            print(f"  Score: {_score_for_generation(candidate, registry_map.get(candidate['source_filename'].strip(), {}))}")

        # Fallback: best-scored image in fallback_folder with a canny map
        if not candidate:
            fallback = []
            for a in all_analyses:
                if a.get("source_folder") != view["fallback_folder"]:
                    continue
                entry = registry_map.get(a["source_filename"].strip(), {})
                if entry.get("canny_path") and Path(entry["canny_path"]).exists():
                    fallback.append((a, entry))

            if fallback:
                fallback.sort(key=lambda x: _score_for_generation(x[0], x[1]), reverse=True)
                candidate, _ = fallback[0]
                print(f"  Score (fallback): {_score_for_generation(candidate, registry_map.get(candidate['source_filename'].strip(), {}))}")

        if not candidate:
            print("  ⚠ No suitable source found, skipping")
            continue

        print(f"  Source: {candidate['source_filename']}")

        try:
            canny_img = _load_canny(
                candidate["source_filename"],
                conditioning_registry_path,
                candidate,
            )
        except Exception as e:
            print(f"  ✗ Canny load failed: {e}")
            continue

        start = time.time()

        result_image = pipe(
            prompt=positive,
            negative_prompt=negative,
            image=canny_img,
            controlnet_conditioning_scale=view["conditioning_scale"],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=1024,
            height=1024,
        ).images[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = recon_dir / f"{view['name']}_reconstruction_{timestamp}.png"
        result_image.save(out_path)

        elapsed = time.time() - start
        print(f"  ✓ {elapsed:.0f}s → {out_path.name}")

        provenance = {
            "view": view["name"],
            "title": view["title"],
            "source_image": candidate["source_filename"],
            "aggregated_from": f"{len(all_analyses)} analyses",
            "confirmed_elements": aggregated["confirmed_elements"],
            "all_citations": aggregated["all_citations"],
            "positive_prompt": positive,
            "negative_prompt": negative,
            "generation_params": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "controlnet_scale": view["conditioning_scale"],
                "hardware": hw.get("device"),
            },
            "timestamp": timestamp,
        }

        provenance_path = prov_dir / f"{view['name']}_provenance_{timestamp}.json"
        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)

        results.append({
            "view": view["name"],
            "title": view["title"],
            "reconstruction_path": str(out_path),
            "provenance_path": str(provenance_path),
            "provenance": provenance,
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"Canonical views generated: {len(results)}/4")

    return results
