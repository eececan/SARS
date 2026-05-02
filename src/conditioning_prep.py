"""
Step 2: Image Conditioning Preparation
MiDaS monocular depth estimation + Canny edge extraction
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

SUFFIX_KEYWORDS = [
    "capital", "frame", "reconstructed_model", "inscription", "portal",
    "column", "frieze", "pediment", "entablature", "cornice", "architrave",
    "base", "stylobate", "cella", "pronaos", "opisthodomos", "model",
    "drawing", "plan", "elevation", "section", "detail", "exterior",
    "interior", "north", "south", "east", "west", "facade", "wall",
    "miniature", "lion", "figure", "frontal", "angle", "ruins", "ground",
    "partially", "destroyed", "rectangular", "structure",
]

FOLDER_PROCESSING = {
    "full_shots":              {"canny": True, "depth": True,  "vlm": True},
    "architectural_models":    {"canny": True, "depth": True,  "vlm": True},
    "details":                 {"canny": True, "depth": False, "vlm": True},
    "inscriptions":            {"canny": True, "depth": False, "vlm": True},
    "parallels/maison_carree": {"canny": True, "depth": True,  "vlm": True},
    "parallels/pula":          {"canny": True, "depth": True,  "vlm": True},
    "plans":                   {"canny": True, "depth": False, "vlm": False},
}

CREDIBILITY = {
    "idai":    "high",
    "dai":     "high",
    "commons": "medium",
    "myphoto": "personal",
}

OTTOMAN_FILENAME_TERMS = {
    "ottoman", "minaret", "mosque", "hacibayram", "haci_bayram", "haci",
}


def resize_for_processing(image_path: str, max_size: int = 896) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image


def _detect_viewpoint(stem: str) -> str:
    s = stem.lower()
    if "interior" in s or "cella" in s:
        return "interior"
    if any(d in s for d in ("north_east", "north_west", "south_east", "south_west",
                             "northeast", "northwest", "southeast", "southwest")):
        return "oblique"
    if "front" in s:
        return "frontal"
    if any(d in s for d in ("north", "south", "east", "west")):
        return "cardinal"
    if any(d in s for d in ("detail", "capital", "inscription")):
        return "detail"
    if any(d in s for d in ("plan", "drawing")):
        return "plan"
    return "unspecified"


def parse_filename_metadata(filename: str, folder: str) -> dict:
    stem = Path(filename).stem.lower()

    source_type = "unknown"
    for prefix in ("idai", "dai", "commons", "myphoto"):
        if stem.startswith(prefix):
            source_type = prefix
            break

    found_keywords = [kw for kw in SUFFIX_KEYWORDS if kw in stem]

    tokens = re.split(r"[-_ ]", stem)
    for tok in tokens:
        tok = tok.strip()
        if tok and tok not in found_keywords and not tok.isdigit() and tok != source_type:
            found_keywords.append(tok)

    has_ottoman = any(term in stem for term in OTTOMAN_FILENAME_TERMS)
    viewpoint = _detect_viewpoint(stem)

    return {
        "source_type": source_type,
        "keywords": found_keywords,
        "folder": folder,
        "credibility_tier": CREDIBILITY.get(source_type, "unknown"),
        "viewpoint": viewpoint,
        "has_ottoman_elements": has_ottoman,
    }


def extract_canny(image_path: str, output_dir: str, subfolder: str) -> str:
    out_dir = Path(output_dir) / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    out_path = out_dir / f"{stem}_canny.png"

    pil_img = resize_for_processing(image_path)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    resized = cv2.resize(edges, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_path), resized)

    return str(out_path)


def _get_subfolder_key(rel_folder: str) -> str:
    rel = rel_folder.replace("\\", "/")
    for key in FOLDER_PROCESSING:
        if rel == key or rel.endswith("/" + key) or rel.startswith(key):
            return key
    return rel


def _load_midas_once():
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.eval()
    return processor, model


def _run_depth_with_model(image_path: str, output_dir: str, subfolder: str,
                           processor, model) -> str:
    import torch

    out_dir = Path(output_dir) / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    out_path = out_dir / f"{stem}_depth.png"

    pil_img = resize_for_processing(image_path)
    inputs = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth

    depth_np = depth.squeeze().numpy()
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max > d_min:
        normalized = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth_np, dtype=np.uint8)

    resized = cv2.resize(normalized, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_path), resized)
    return str(out_path)


def process_image_collection(visual_sources_dir: str, conditioning_dir: str) -> list:
    visual_sources_dir = Path(visual_sources_dir)
    conditioning_dir = Path(conditioning_dir)
    canny_dir = conditioning_dir / "canny"
    depth_dir = conditioning_dir / "depth"
    registry_dir = conditioning_dir / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(
        p for p in visual_sources_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    total = len(all_images)
    print(f"Found {total} images to process.\n")

    needs_depth = any(
        FOLDER_PROCESSING.get(
            _get_subfolder_key(p.parent.relative_to(visual_sources_dir).as_posix()),
            {}
        ).get("depth", False)
        for p in all_images
    )

    midas_processor, midas_model = None, None
    if needs_depth:
        print("Loading MiDaS (Intel/dpt-large) model — this may take a moment...")
        midas_processor, midas_model = _load_midas_once()
        print("Model loaded.\n")

    registry = []

    for idx, img_path in enumerate(all_images, start=1):
        filename = img_path.name
        rel_folder = img_path.parent.relative_to(visual_sources_dir).as_posix()
        key = _get_subfolder_key(rel_folder)
        rules = FOLDER_PROCESSING.get(key, {"canny": True, "depth": False, "vlm": False})

        print(f"Processing {idx}/{total}: {rel_folder}/{filename}")

        metadata = parse_filename_metadata(filename, rel_folder)

        suitable_for = []
        if rules.get("vlm"):
            suitable_for.append("vlm_analysis")
        if rules.get("canny"):
            suitable_for.append("canny_conditioning")
        if rules.get("depth"):
            suitable_for.append("depth_conditioning")

        canny_path = None
        depth_path = None

        if rules.get("canny"):
            try:
                canny_path = extract_canny(str(img_path), str(canny_dir), rel_folder)
            except Exception as e:
                print(f"  [WARN] Canny failed for {filename}: {e}")

        if rules.get("depth") and midas_processor is not None:
            try:
                depth_path = _run_depth_with_model(
                    str(img_path), str(depth_dir), rel_folder,
                    midas_processor, midas_model,
                )
            except Exception as e:
                print(f"  [WARN] Depth failed for {filename}: {e}")

        entry = {
            "source_filename": filename,
            "source_folder": rel_folder,
            "source_path": str(img_path),
            "source_type": metadata["source_type"],
            "credibility_tier": metadata["credibility_tier"],
            "keywords": metadata["keywords"],
            "viewpoint": metadata["viewpoint"],
            "has_ottoman_elements": metadata["has_ottoman_elements"],
            "canny_path": canny_path,
            "depth_path": depth_path,
            "suitable_for": suitable_for,
        }
        registry.append(entry)

    registry_path = registry_dir / "conditioning_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Registry saved to {registry_path}")
    return registry


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    process_image_collection(
        visual_sources_dir=str(base / "data" / "visual_sources"),
        conditioning_dir=str(base / "data" / "conditioning"),
    )
