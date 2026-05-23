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

# Semantic keywords to extract from filename stems
CONTENT_KEYWORDS = [
    "fullshot", "full_shot", "capital", "columncapital", "column", "frieze",
    "entablature", "cornice", "architrave", "cella", "pronaos", "opisthodomos",
    "podium", "stylobate", "inscription", "inscriptions", "plan", "drawing",
    "model", "elevation", "section", "interior", "exterior", "facade", "wall",
    "ground", "ruins", "fragmentary", "front", "side", "back", "relief",
    "lion", "figure", "portal", "entrance",
]

DIRECTION_KEYWORDS = {
    "north", "south", "east", "west",
    "north_east", "north_west", "south_east", "south_west",
    "northeast", "northwest", "southeast", "southwest",
    "top", "bottom", "left", "right", "middle", "center",
    "topleft", "topright", "middleleft", "middleright",
    "top_left", "top_right", "middle_left", "middle_right",
}

SCAFFOLD_POSITIONS = {
    "top":         {"zero_top": 0.30},
    "topleft":     {"zero_top": 0.30, "zero_left": 0.20},
    "topright":    {"zero_top": 0.30, "zero_right": 0.20},
    "top_left":    {"zero_top": 0.30, "zero_left": 0.20},
    "top_right":   {"zero_top": 0.30, "zero_right": 0.20},
    "middle":      {"zero_center_h": 0.30},
    "middleleft":  {"zero_left": 0.30},
    "middleright": {"zero_right": 0.30},
    "middle_left": {"zero_left": 0.30},
    "middle_right":{"zero_right": 0.30},
    "left":        {"zero_left": 0.25},
    "right":       {"zero_right": 0.25},
    "bottom":      {"zero_bottom": 0.20},
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
    if "front" in s or "frontal" in s:
        return "frontal"
    if any(d in s for d in ("north", "south", "east", "west")):
        return "cardinal"
    if any(d in s for d in ("detail", "capital", "columncapital", "inscription", "relief")):
        return "detail"
    if any(d in s for d in ("plan", "drawing")):
        return "plan"
    return "unspecified"


def _parse_scaffold_position(stem: str) -> list:
    """
    Extract 1-2 position tokens after 'scaffolding' in the filename stem.
    Returns list of matched SCAFFOLD_POSITIONS keys found.
    """
    tokens = re.split(r"[-_]", stem.lower())
    try:
        idx = tokens.index("scaffolding")
    except ValueError:
        return []

    positions = []
    for offset in (1, 2):
        if idx + offset < len(tokens):
            candidate = tokens[idx + offset]
            # Also try two-token compound e.g. "top" + "left" → "topleft"
            if candidate in SCAFFOLD_POSITIONS:
                positions.append(candidate)
            elif offset == 1 and idx + 2 < len(tokens):
                compound = candidate + tokens[idx + 2]
                if compound in SCAFFOLD_POSITIONS:
                    positions.append(compound)
                    break
    return positions


def parse_filename_metadata(filename: str, folder: str) -> dict:
    stem = Path(filename).stem.lower()

    source_type = "unknown"
    for prefix in ("idai", "dai", "commons", "myphoto"):
        if stem.startswith(prefix):
            source_type = prefix
            break

    # Semantic content keywords only — no noise tokens
    keywords = [kw for kw in CONTENT_KEYWORDS if kw in stem]

    # Contamination flags
    has_minaret   = "minaret" in stem
    has_scaffold  = "scaffolding" in stem
    has_mosque    = any(t in stem for t in ("mosque", "hacibayram", "haci_bayram"))
    has_church    = "church" in stem
    has_ottoman   = has_minaret or has_mosque or "ottoman" in stem
    is_ground     = "ground" in stem
    is_model      = any(t in stem for t in ("model", "drawing"))
    is_fragmentary = "fragmentary" in stem

    scaffold_positions = _parse_scaffold_position(stem) if has_scaffold else []
    viewpoint = _detect_viewpoint(stem)

    return {
        "source_filename": filename,
        "source_type": source_type,
        "keywords": keywords,
        "folder": folder,
        "credibility_tier": CREDIBILITY.get(source_type, "unknown"),
        "viewpoint": viewpoint,
        "has_ottoman_elements": has_ottoman,
        "has_minaret": has_minaret,
        "has_scaffold": has_scaffold,
        "scaffold_positions": scaffold_positions,
        "has_mosque": has_mosque,
        "has_church": has_church,
        "is_ground_shot": is_ground,
        "is_model_or_drawing": is_model,
        "is_fragmentary": is_fragmentary,
    }


# Hacı Bayram Mosque sits north/northwest of the temple cella.
# The mosque silhouette (and its minaret) appears in different parts of the
# canny frame depending on which side the photo was taken from.
# Values are (zero_left_fraction, zero_right_fraction, zero_top_fraction).
MOSQUE_REGION_BY_VIEWPOINT = {
    "south_east": {"left": 0.30, "top": 0.35},   # mosque is upper-left/back
    "southeast":  {"left": 0.30, "top": 0.35},
    "south_west": {"right": 0.30, "top": 0.35},  # mosque is upper-right/back
    "southwest":  {"right": 0.30, "top": 0.35},
    "south":      {"top": 0.40},                  # mosque is behind, upper band
    "east":       {"left": 0.35, "top": 0.30},
    "west":       {"right": 0.35, "top": 0.30},
    "north_east": {"top": 0.50, "left": 0.25},   # mosque dominates upper frame
    "northeast":  {"top": 0.50, "left": 0.25},
    "north_west": {"top": 0.50, "right": 0.25},
    "northwest":  {"top": 0.50, "right": 0.25},
    "north":      {"top": 0.55},                  # mosque fills front — heaviest masking
}


def _viewpoint_token(stem: str) -> str:
    """Return the dominant cardinal/oblique direction token in stem, else ''."""
    s = stem.lower()
    for tok in ("south_east", "south_west", "north_east", "north_west",
                "southeast", "southwest", "northeast", "northwest"):
        if tok in s:
            return tok
    for tok in ("north", "south", "east", "west"):
        if tok in s:
            return tok
    return ""


def _suppress_thin_lines(edges: np.ndarray, min_thickness: int = 2) -> np.ndarray:
    """
    Suppress isolated short edge fragments (typical of steel scaffolding
    tubes) while preserving connected stone-edge structure.

    A morphological opening with a square kernel deletes ALL edges thinner
    than the kernel — and canny edges are only 1-2px wide, so opening also
    erases stone edges. Instead, remove small connected components: real
    structural edges form large connected contours, scaffolding tubes show
    up as many small isolated specks after region masking.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (edges > 0).astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(edges)
    # Keep components above an area threshold; drop tiny isolated specks.
    # Kept low so genuine short stone-edge segments survive — only true
    # scaffolding specks (a handful of pixels) are removed.
    min_area = 15
    for label in range(1, num):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _apply_canny_cleaning(edges: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Zero out contaminated regions of a 1024x1024 canny map based on
    filename-derived flags. Returns cleaned array.

    Strategy:
    - Mosque/minaret masking is viewpoint-aware (mosque is geographically
      north of the temple, so its image position depends on shot direction)
    - Scaffold positions still respected as before
    - When scaffolding is present, run a morphological opening to suppress
      thin steel-tube lines while preserving thicker stone edges
    """
    h, w = edges.shape
    cleaned = edges.copy()

    stem = Path(metadata.get("source_filename", "")).stem if metadata.get("source_filename") else ""
    viewpoint_tok = _viewpoint_token(stem)

    # Mosque + minaret: viewpoint-aware masking
    if metadata.get("has_mosque") or metadata.get("has_minaret"):
        region = MOSQUE_REGION_BY_VIEWPOINT.get(viewpoint_tok)
        if region is None:
            # Unknown viewpoint or interior shot — fall back to top band only
            cutoff = int(h * 0.30)
            cleaned[:cutoff, :] = 0
            print(f"    [CLEAN] Mosque/minaret: zeroed top 30% (viewpoint unknown)")
        else:
            applied = []
            if "top" in region:
                c = int(h * region["top"])
                cleaned[:c, :] = 0
                applied.append(f"top {int(region['top']*100)}%")
            if "left" in region:
                c = int(w * region["left"])
                cleaned[:, :c] = 0
                applied.append(f"left {int(region['left']*100)}%")
            if "right" in region:
                c = int(w * region["right"])
                cleaned[:, w-c:] = 0
                applied.append(f"right {int(region['right']*100)}%")
            tag = "minaret+mosque" if (metadata.get("has_mosque") and metadata.get("has_minaret")) \
                  else ("minaret" if metadata.get("has_minaret") else "mosque")
            print(f"    [CLEAN] {tag} ({viewpoint_tok}): zeroed {', '.join(applied)}")

    # Scaffolding: zero per detected position (poles/sheets), then suppress
    # any remaining thin steel lines morphologically
    if metadata.get("has_scaffold"):
        positions = metadata.get("scaffold_positions", [])
        if not positions:
            cleaned[:int(h * 0.25), :] = 0
            print("    [CLEAN] Zeroed top 25% (scaffolding, position unknown)")
        else:
            for pos in positions:
                ops = SCAFFOLD_POSITIONS.get(pos, {})
                if "zero_top" in ops:
                    c = int(h * ops["zero_top"])
                    cleaned[:c, :] = 0
                    print(f"    [CLEAN] Zeroed top {int(ops['zero_top']*100)}% (scaffold {pos})")
                if "zero_bottom" in ops:
                    c = int(h * ops["zero_bottom"])
                    cleaned[h-c:, :] = 0
                    print(f"    [CLEAN] Zeroed bottom {int(ops['zero_bottom']*100)}% (scaffold {pos})")
                if "zero_left" in ops:
                    c = int(w * ops["zero_left"])
                    cleaned[:, :c] = 0
                    print(f"    [CLEAN] Zeroed left {int(ops['zero_left']*100)}% (scaffold {pos})")
                if "zero_right" in ops:
                    c = int(w * ops["zero_right"])
                    cleaned[:, w-c:] = 0
                    print(f"    [CLEAN] Zeroed right {int(ops['zero_right']*100)}% (scaffold {pos})")
                if "zero_center_h" in ops:
                    frac = ops["zero_center_h"]
                    c0 = int(w * (0.5 - frac / 2))
                    c1 = int(w * (0.5 + frac / 2))
                    cleaned[:, c0:c1] = 0
                    print(f"    [CLEAN] Zeroed center {int(frac*100)}% horizontally (scaffold {pos})")

        # Suppress thin steel-tube edges that survived the regional masking
        before_count = int((cleaned > 0).sum())
        cleaned = _suppress_thin_lines(cleaned, min_thickness=2)
        after_count = int((cleaned > 0).sum())
        removed_pct = 100 * (before_count - after_count) / max(before_count, 1)
        print(f"    [CLEAN] Morphological steel-line suppression: removed {removed_pct:.0f}% of remaining edges")

    return cleaned


def extract_canny(image_path: str, output_dir: str, subfolder: str,
                  metadata: dict = None) -> str:
    out_dir = Path(output_dir) / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    out_path = out_dir / f"{stem}_canny.png"

    pil_img = resize_for_processing(image_path)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Run Canny at the final 1024x1024 resolution. The previous code ran
    # Canny at 896px then upscaled the binary map with INTER_LINEAR, which
    # blurred 1px edges into 2-3px gray smears — inflating apparent edge
    # coverage with artifacts. Resize the grayscale image first instead.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)
    # Low thresholds + minimal blur: these are low-contrast weathered-stone
    # ruins and parallel temples; higher settings left genuine edge
    # coverage at only 3-5%, too sparse for ControlNet to anchor structure.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=15, threshold2=45)

    if metadata:
        edges = _apply_canny_cleaning(edges, metadata)

    cv2.imwrite(str(out_path), edges)
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
        print("Loading MiDaS (Intel/dpt-large) model...")
        midas_processor, midas_model = _load_midas_once()
        print("Model loaded.\n")

    registry = []

    for idx, img_path in enumerate(all_images, start=1):
        filename = img_path.name
        rel_folder = img_path.parent.relative_to(visual_sources_dir).as_posix()
        key = _get_subfolder_key(rel_folder)
        rules = FOLDER_PROCESSING.get(key, {"canny": True, "depth": False, "vlm": False})

        print(f"[{idx}/{total}] {rel_folder}/{filename}")

        metadata = parse_filename_metadata(filename, rel_folder)

        needs_cleaning = (
            metadata["has_minaret"] or
            metadata["has_scaffold"] or
            metadata["has_mosque"]
        )
        if needs_cleaning:
            flags = []
            if metadata["has_minaret"]:  flags.append("minaret")
            if metadata["has_scaffold"]: flags.append(f"scaffold({','.join(metadata['scaffold_positions']) or 'unknown'})")
            if metadata["has_mosque"]:   flags.append("mosque")
            print(f"  → Canny cleaning: {', '.join(flags)}")

        suitable_for = []
        if rules.get("vlm"):    suitable_for.append("vlm_analysis")
        if rules.get("canny"):  suitable_for.append("canny_conditioning")
        if rules.get("depth"):  suitable_for.append("depth_conditioning")

        canny_path = None
        depth_path = None

        if rules.get("canny"):
            try:
                canny_path = extract_canny(
                    str(img_path), str(canny_dir), rel_folder,
                    metadata=metadata,
                )
            except Exception as e:
                print(f"  [WARN] Canny failed: {e}")

        if rules.get("depth") and midas_processor is not None:
            try:
                depth_path = _run_depth_with_model(
                    str(img_path), str(depth_dir), rel_folder,
                    midas_processor, midas_model,
                )
            except Exception as e:
                print(f"  [WARN] Depth failed: {e}")

        entry = {
            "source_filename": filename,
            "source_folder": rel_folder,
            "source_path": str(img_path),
            "source_type": metadata["source_type"],
            "credibility_tier": metadata["credibility_tier"],
            "keywords": metadata["keywords"],
            "viewpoint": metadata["viewpoint"],
            "has_ottoman_elements": metadata["has_ottoman_elements"],
            "has_minaret": metadata["has_minaret"],
            "has_scaffold": metadata["has_scaffold"],
            "scaffold_positions": metadata["scaffold_positions"],
            "has_mosque": metadata["has_mosque"],
            "is_ground_shot": metadata["is_ground_shot"],
            "is_model_or_drawing": metadata["is_model_or_drawing"],
            "is_fragmentary": metadata["is_fragmentary"],
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
