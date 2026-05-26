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

def load_sdxl_inpaint_pipeline(hw: dict, base_pipe=None):
    """Load SDXL inpainting pipeline for stage-2 completion.

    Shares weights with base_pipe when provided to avoid double VRAM cost.
    """
    from diffusers import StableDiffusionXLInpaintPipeline

    if base_pipe is not None:
        pipe = StableDiffusionXLInpaintPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            text_encoder_2=base_pipe.text_encoder_2,
            tokenizer=base_pipe.tokenizer,
            tokenizer_2=base_pipe.tokenizer_2,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
        )
    else:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=hw["torch_dtype"],
        )
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
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
# octastyle facade (8 exterior columns), 15 along each flank, peripteral.
# The actual column count is anchored by canny — CLIP doesn't reason
# about "EXACTLY 8" — so this string is kept short to leave token
# budget for STYLE_PREFIX + per-image VLM description (CLIP cap = 77).
CONFIRMED_COLUMN_FACTS = (
    "octastyle peripteral Corinthian temple, complete colonnade all sides"
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
# Canny map loading
# ---------------------------------------------------------------------------
# Canny cleaning (minaret/mosque/scaffold masking) happens once in Step 2
# (conditioning_prep.py). It is intentionally NOT repeated here.


def _canny_coverage(canny_path: str) -> float:
    """Fraction of non-zero pixels in a canny map (0.0-1.0), or None if
    the map is missing/unreadable."""
    if not canny_path or not Path(canny_path).exists():
        return None
    import cv2
    img = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float((img > 0).sum()) / img.size


def _load_canny(
    source_filename: str,
    conditioning_registry_path: str,
    analysis: dict,
) -> Image.Image:
    # Canny maps are already cleaned in Step 2 (conditioning_prep.py:
    # _apply_canny_cleaning — viewpoint-aware minaret/mosque/scaffold
    # masking baked into the saved *_canny.png). Re-cleaning here would
    # zero regions twice and apply a second morphological pass, which
    # collapsed the structural signal. Use the saved map as-is.
    return load_conditioning_image(source_filename, conditioning_registry_path, mode="canny")


# ---------------------------------------------------------------------------
# Stage-2: blank/low-detail region detection for inpainting
# ---------------------------------------------------------------------------

def _detect_completion_mask(
    image: Image.Image,
    canny_image: Image.Image = None,
    variance_threshold: float = 80.0,
    block: int = 32,
    dilate_px: int = 24,
) -> Image.Image:
    """Find regions of `image` that look unfinished — flat/uniform patches
    far from any source canny edges. These are the 'collage gray' areas
    where ControlNet had no guidance and SDXL left a blank.

    Returns a binary PIL mask (white = inpaint, black = keep).
    """
    import numpy as np
    import cv2

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, block):
        for x in range(0, w, block):
            patch = gray[y:y + block, x:x + block]
            if patch.size == 0:
                continue
            if patch.var() < variance_threshold:
                mask[y:y + block, x:x + block] = 255

    if canny_image is not None:
        c = np.array(canny_image.convert("L"))
        if c.shape != (h, w):
            c = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        trust = cv2.dilate((c > 0).astype(np.uint8) * 255, kernel)
        mask[trust > 0] = 0

    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)

    return Image.fromarray(mask)


def _inpaint_completion(
    base_image: Image.Image,
    mask: Image.Image,
    positive: str,
    negative: str,
    inpaint_pipe,
    num_inference_steps: int = 30,
    guidance_scale: float = 8.5,
    strength: float = 0.95,
) -> Image.Image:
    """Run SDXL inpainting on the flagged regions to complete missing
    structure. High strength (0.95) so the gray placeholder gets fully
    replaced with prompt-driven content."""
    import numpy as np

    if np.array(mask).max() == 0:
        # Nothing to inpaint — stage 1 was already complete
        return base_image

    result = inpaint_pipe(
        prompt=positive,
        negative_prompt=negative,
        image=base_image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        width=1024,
        height=1024,
    ).images[0]
    return result


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
    inpaint_pipe=None,
    control_guidance_end: float = 0.55,
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

    # Stage 1: ControlNet only guides the first ~55% of denoising. Canny
    # anchors layout early; later steps let SDXL complete missing geometry
    # instead of leaving the gray/blank regions you'd get at full scale.
    result_image = pipe(
        prompt=positive,
        negative_prompt=negative,
        image=image_arg,
        controlnet_conditioning_scale=scale_arg,
        control_guidance_end=control_guidance_end,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=1024,
        height=1024,
    ).images[0]

    # Stage 2: find any flat/uniform patches that still look unfinished
    # (regions where the source had no canny edges) and inpaint them.
    inpaint_applied = False
    if inpaint_pipe is not None and use_canny:
        first_canny = conditioning_images[0] if conditioning_images else None
        completion_mask = _detect_completion_mask(result_image, canny_image=first_canny)
        import numpy as np
        mask_coverage = float((np.array(completion_mask) > 0).sum()) / (1024 * 1024)
        print(f"  Stage 2 mask coverage: {mask_coverage:.1%}")
        if mask_coverage > 0.02:
            result_image = _inpaint_completion(
                result_image, completion_mask, positive, negative, inpaint_pipe,
            )
            inpaint_applied = True

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
            "control_guidance_end": control_guidance_end,
            "inpaint_stage_applied": inpaint_applied,
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
    inpaint_pipe=None,
) -> list:
    output_dir = Path(output_dir)
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    # Skip folders that:
    # - produce close-up canny maps (no full structure visible): details, inscriptions
    # - are intact reference temples that should NOT be regenerated: parallels/*
    #   (those are comparanda — reconstructing them just produces softer copies)
    SKIP_FOLDERS = {"details", "inscriptions"}
    SKIP_FOLDER_PREFIXES = ("parallels",)
    # Skip filenames that indicate close-up shots (column-only, capital-only)
    # or fragmentary technical drawings that confuse the structural prior.
    SKIP_KEYWORDS = ("columncapital", "lion", "relief", "_columns.", "fragmentary")

    analyses = []
    for f in sorted(Path(analysis_dir).glob("*_analysis.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if "error" in data or not data.get("sdxl_prompt_component"):
            continue
        folder = data.get("source_folder", "")
        fname = data.get("source_filename", "").lower()
        if folder in SKIP_FOLDERS:
            continue
        if any(folder.startswith(pfx) for pfx in SKIP_FOLDER_PREFIXES):
            continue
        if any(kw in fname for kw in SKIP_KEYWORDS):
            continue
        analyses.append(data)

    analyses.sort(key=_score_analysis, reverse=True)
    print(f"Filtered to {len(analyses)} full-structure sources (excluded details/inscriptions/close-ups)")

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
        # ControlNet scale: lowered from 0.75 → 0.45 because the canny maps
        # capture broken/partial structure. At high scale, SDXL faithfully
        # reproduces fragmentation (floating columns, half-walls). At 0.45
        # plus control_guidance_end=0.55, canny anchors layout in the first
        # half of denoising and SDXL completes the missing geometry in the
        # second half. Plans stay at full scale — they ARE complete.
        conditioning_scale = 1.0 if is_plan else 0.45

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
                inpaint_pipe=inpaint_pipe,
                control_guidance_end=1.0 if is_plan else 0.55,
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

# Sources to exclude from canonical-view selection: technical drawings
# with multiple scaled views, partial cutaways, or fragmentary states.
# These canny maps look like "multiple buildings on one page" to ControlNet
# and cause the temple-inside-temple compositional failure.
CANONICAL_BAD_KEYWORDS = ("fragmentary", "cutaway", "section")


CANONICAL_VIEWS = [
    # Three exterior elevations use PROCEDURAL canny+depth synthesised
    # from the documented temple dimensions (8 cols front, 15 cols side,
    # 2m intercolumniation, etc.). The geometry is ours, not borrowed
    # from a hexastyle parallel — so the output has exactly 8 columns.
    {
        "name": "front_elevation",
        "title": "Front Elevation",
        "mode": "procedural",
        "canny_scale": 0.85,         # high — we trust our own geometry
        "depth_scale": 0.45,
        "control_guidance_end": 0.8,
    },
    {
        "name": "side_elevation",
        "title": "Side Elevation",
        "mode": "procedural",
        "canny_scale": 0.85,
        "depth_scale": 0.45,
        "control_guidance_end": 0.8,
    },
    {
        "name": "three_quarter",
        "title": "Three-Quarter View",
        "mode": "procedural",
        "canny_scale": 0.80,
        "depth_scale": 0.55,          # depth carries the 3D structure here
        "control_guidance_end": 0.8,
    },
    # Interior cella: no procedural option (no plan data for interior).
    # Stays on the Ankara ruin's canny + MiDaS depth.
    {
        "name": "interior_cella",
        "title": "Interior — Cella",
        "mode": "source",
        "prefer_folder": "full_shots",
        "prefer_keywords": ["interior", "cella", "inside"],
        "fallback_folder": "full_shots",
        "canny_scale": 0.50,
        "depth_scale": 0.40,
        "control_guidance_end": 0.62,
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

    # Reject near-blank canny maps. Normal architectural photos sit around
    # 9-14% edge coverage (lots of flat sky/marble, 1px edges); maps below
    # ~6% are genuinely empty — heavy contamination masking or a near-
    # featureless source — and let SDXL hallucinate the composition.
    coverage = _canny_coverage(entry.get("canny_path"))
    if coverage is not None and coverage < 0.06:
        score -= 100  # effectively disqualify

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
    inpaint_pipe=None,
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

    # Procedural canny+depth use documented dimensions (column counts,
    # intercolumniation, temple footprint) extracted by analyze_plans.py.
    # Falls back to scholarship defaults if no plans_extraction yet.
    from src.procedural_canny import (
        load_plan_dims_aggregated,
        render_view as render_procedural_view,
    )
    plan_dims = load_plan_dims_aggregated(analysis_dir)
    print(f"\nPlan dimensions in use:")
    for k in ("columns_front", "columns_side", "intercolumniation_m",
              "column_diameter_m", "temple_length_m", "temple_width_m"):
        print(f"  {k}: {plan_dims.get(k)}")

    proc_canny_dir = output_dir / "procedural"
    proc_canny_dir.mkdir(parents=True, exist_ok=True)

    for view in CANONICAL_VIEWS:
        print(f"\n── {view['title']} ──────────")

        mode = view.get("mode", "source")
        source_label = None
        canny_img = None
        depth_img = None

        if mode == "procedural":
            # Synthesise conditioning from documented dimensions
            canny_img, depth_img = render_procedural_view(view["name"], plan_dims)
            # Persist for the web app to display
            canny_img.save(proc_canny_dir / f"{view['name']}_canny.png")
            depth_img.save(proc_canny_dir / f"{view['name']}_depth.png")
            source_label = f"procedural ({view['name']} from plan dimensions)"
            print(f"  Source: {source_label}")

        else:
            # Pick the best-scored real photo for this view
            candidate = None
            preferred = []
            for a in all_analyses:
                if a.get("source_folder") != view["prefer_folder"]:
                    continue
                fname = a.get("source_filename", "").lower()
                if any(bad in fname for bad in CANONICAL_BAD_KEYWORDS):
                    continue
                if not any(kw in fname for kw in view["prefer_keywords"]):
                    continue
                entry = registry_map.get(a["source_filename"].strip(), {})
                if entry.get("canny_path") and Path(entry["canny_path"]).exists():
                    preferred.append((a, entry))

            if preferred:
                preferred.sort(key=lambda x: _score_for_generation(x[0], x[1]), reverse=True)
                candidate, _ = preferred[0]

            if not candidate:
                fallback = []
                for a in all_analyses:
                    if a.get("source_folder") != view["fallback_folder"]:
                        continue
                    fname = a.get("source_filename", "").lower()
                    if any(bad in fname for bad in CANONICAL_BAD_KEYWORDS):
                        continue
                    entry = registry_map.get(a["source_filename"].strip(), {})
                    if entry.get("canny_path") and Path(entry["canny_path"]).exists():
                        fallback.append((a, entry))
                if fallback:
                    fallback.sort(key=lambda x: _score_for_generation(x[0], x[1]), reverse=True)
                    candidate, _ = fallback[0]

            if not candidate:
                print("  ⚠ No suitable source found, skipping")
                continue

            source_label = candidate["source_filename"]
            print(f"  Source: {source_label}")
            try:
                canny_img = _load_canny(
                    candidate["source_filename"],
                    conditioning_registry_path,
                    candidate,
                )
            except Exception as e:
                print(f"  ✗ Canny load failed: {e}")
                continue

            # Optional MiDaS depth for source mode (interior cella)
            try:
                depth_img = load_conditioning_image(
                    candidate["source_filename"],
                    conditioning_registry_path,
                    mode="depth",
                )
            except (ValueError, FileNotFoundError):
                depth_img = None

        start = time.time()
        cg_end = view.get("control_guidance_end", 0.7)

        # Assemble multi-ControlNet inputs. If depth is available we pass
        # [canny, depth] with separate scales — the pipeline was loaded
        # with both ControlNets (use_canny=True, use_depth=True).
        if depth_img is not None:
            image_arg = [canny_img, depth_img]
            scale_arg = [view["canny_scale"], view["depth_scale"]]
        else:
            image_arg = canny_img
            scale_arg = view["canny_scale"]

        try:
            result_image = pipe(
                prompt=positive,
                negative_prompt=negative,
                image=image_arg,
                controlnet_conditioning_scale=scale_arg,
                control_guidance_end=cg_end,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=1024,
                height=1024,
            ).images[0]
        except Exception as e:
            # Some diffusers builds reject a list for control_guidance_end
            # when given multi-control. Retry without that arg.
            print(f"  ! retry without control_guidance_end ({e})")
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

        # Stage 2: inpaint any leftover flat/blank patches
        inpaint_applied = False
        if inpaint_pipe is not None:
            completion_mask = _detect_completion_mask(result_image, canny_image=canny_img)
            import numpy as np
            mask_coverage = float((np.array(completion_mask) > 0).sum()) / (1024 * 1024)
            print(f"  Stage 2 mask coverage: {mask_coverage:.1%}")
            if mask_coverage > 0.02:
                result_image = _inpaint_completion(
                    result_image, completion_mask, positive, negative, inpaint_pipe,
                )
                inpaint_applied = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = recon_dir / f"{view['name']}_reconstruction_{timestamp}.png"
        result_image.save(out_path)

        elapsed = time.time() - start
        print(f"  ✓ {elapsed:.0f}s → {out_path.name}")

        provenance = {
            "view": view["name"],
            "title": view["title"],
            "mode": mode,
            "source_image": source_label,
            "plan_dimensions_used": plan_dims if mode == "procedural" else None,
            "aggregated_from": f"{len(all_analyses)} analyses",
            "confirmed_elements": aggregated["confirmed_elements"],
            "all_citations": aggregated["all_citations"],
            "positive_prompt": positive,
            "negative_prompt": negative,
            "generation_params": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "canny_scale": view["canny_scale"],
                "depth_scale": view.get("depth_scale"),
                "depth_used": depth_img is not None,
                "control_guidance_end": cg_end,
                "inpaint_stage_applied": inpaint_applied,
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
