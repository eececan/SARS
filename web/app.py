"""
Flask web application for the SARS project.
Templum Divi Augusti · Ankara · 25 BCE
Citation-Grounded Reconstruction Explorer
"""

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent  # project root
sys.path.insert(0, str(BASE_DIR))

ANALYSIS_DIR = BASE_DIR / "data" / "analysis" / "vlm_outputs"
RECONSTRUCTIONS_DIR = BASE_DIR / "data" / "outputs" / "reconstructions"
PROVENANCE_DIR = BASE_DIR / "data" / "outputs" / "provenance"
VISUAL_SOURCES_DIR = BASE_DIR / "data" / "visual_sources"
CANNY_DIR = BASE_DIR / "data" / "conditioning" / "canny"
DEPTH_DIR = BASE_DIR / "data" / "conditioning" / "depth"
REGISTRY_PATH = BASE_DIR / "data" / "conditioning" / "registry" / "conditioning_registry.json"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    """Return {source_filename: entry} mapping."""
    if not REGISTRY_PATH.exists():
        return {}
    entries = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {e["source_filename"].strip(): e for e in entries}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reconstructions")
def api_reconstructions():
    """List reconstruction PNGs with their provenance if available."""
    RECONSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(RECONSTRUCTIONS_DIR.glob("*.png"))
    results = []

    for img_path in images:
        stem = img_path.stem  # e.g. commons10_frontal_south_west_Angle_reconstruction_20260101_120000
        # Try to match a provenance JSON by stem prefix
        provenance = None
        # Pattern: <source_stem>_provenance_<timestamp>.json or <view>_provenance_<timestamp>.json
        source_key = stem.replace("_reconstruction_", "_provenance_")
        prov_path = PROVENANCE_DIR / f"{source_key}.json"
        if prov_path.exists():
            try:
                provenance = json.loads(prov_path.read_text(encoding="utf-8"))
            except Exception:
                provenance = None
        else:
            # Try fuzzy match: find any provenance file that shares the source stem prefix
            # stem up to "_reconstruction_" part
            if "_reconstruction_" in stem:
                source_prefix = stem.split("_reconstruction_")[0]
                for pf in PROVENANCE_DIR.glob(f"{source_prefix}_provenance_*.json"):
                    try:
                        provenance = json.loads(pf.read_text(encoding="utf-8"))
                        break
                    except Exception:
                        pass

        results.append({
            "filename": img_path.name,
            "url": f"/img/reconstruction/{img_path.name}",
            "provenance": provenance,
        })

    return jsonify(results)


@app.route("/api/analyses")
def api_analyses():
    """Return all analysis JSONs sorted by credibility_tier (high first)."""
    registry = _load_registry()

    tier_order = {"high": 0, "medium": 1, "personal": 2}
    analyses = []

    for f in sorted(ANALYSIS_DIR.glob("*_analysis.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        if "error" in data:
            continue

        # Enrich with canny/source relative paths from registry
        source_filename = data.get("source_filename", "").strip()
        reg_entry = registry.get(source_filename, {})

        canny_rel = ""
        source_rel = ""

        canny_abs = reg_entry.get("canny_path", "")
        if canny_abs:
            canny_p = Path(canny_abs)
            if canny_p.exists():
                try:
                    canny_rel = str(canny_p.relative_to(CANNY_DIR))
                except ValueError:
                    canny_rel = canny_p.name

        source_abs = reg_entry.get("source_path", "")
        if source_abs:
            source_p = Path(source_abs)
            if source_p.exists():
                try:
                    source_rel = str(source_p.relative_to(VISUAL_SOURCES_DIR))
                except ValueError:
                    source_rel = source_p.name

        data["canny_relative_path"] = canny_rel
        data["source_relative_path"] = source_rel

        analyses.append(data)

    # Sort: high → medium → personal → unknown
    analyses.sort(key=lambda a: tier_order.get(a.get("credibility_tier", ""), 99))

    return jsonify(analyses)


@app.route("/api/brief")
def api_brief():
    """Return the aggregated analysis brief."""
    try:
        from src.generation import aggregate_all_analyses
        result = aggregate_all_analyses(str(ANALYSIS_DIR))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Static image routes
# ---------------------------------------------------------------------------

@app.route("/img/reconstruction/<filename>")
def img_reconstruction(filename: str):
    path = RECONSTRUCTIONS_DIR / filename
    if not path.exists():
        return "Not found", 404
    return send_file(path)


@app.route("/img/source/<path:filepath>")
def img_source(filepath: str):
    path = VISUAL_SOURCES_DIR / filepath
    if not path.exists():
        return "Not found", 404
    return send_file(path)


@app.route("/img/canny/<path:filepath>")
def img_canny(filepath: str):
    path = CANNY_DIR / filepath
    if not path.exists():
        return "Not found", 404
    return send_file(path)


@app.route("/img/depth/<path:filepath>")
def img_depth(filepath: str):
    path = DEPTH_DIR / filepath
    if not path.exists():
        return "Not found", 404
    return send_file(path)


@app.route("/api/conditioning")
def api_conditioning():
    """Return all conditioning image triplets: source / canny / depth."""
    registry = _load_registry()
    if not registry:
        return jsonify([])

    result = []
    for filename, entry in registry.items():
        source_abs = entry.get("source_path", "")
        canny_abs = entry.get("canny_path", "")
        depth_abs = entry.get("depth_path", "")

        def rel_url(abs_path: str, base_dir: Path, route_prefix: str) -> str:
            if not abs_path:
                return ""
            p = Path(abs_path)
            if not p.exists():
                return ""
            try:
                rel = str(p.relative_to(base_dir))
            except ValueError:
                rel = p.name
            return f"{route_prefix}/{rel}"

        # Derive folder from source path if not stored in registry
        folder = entry.get("folder", "")
        if not folder and source_abs:
            source_p = Path(source_abs)
            try:
                rel_parts = source_p.relative_to(VISUAL_SOURCES_DIR).parts
                folder = rel_parts[0] if rel_parts else ""
            except ValueError:
                folder = source_p.parent.name

        result.append({
            "filename": filename,
            "folder": folder,
            "credibility_tier": entry.get("credibility_tier", ""),
            "keywords": entry.get("keywords", []),
            "source_url": rel_url(source_abs, VISUAL_SOURCES_DIR, "/img/source"),
            "canny_url": rel_url(canny_abs, CANNY_DIR, "/img/canny"),
            "depth_url": rel_url(depth_abs, DEPTH_DIR, "/img/depth"),
        })

    result.sort(key=lambda x: (x.get("folder", ""), x.get("filename", "")))
    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5050)
