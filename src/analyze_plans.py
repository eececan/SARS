"""
Step 3b: Plans-Only VLM Analysis
Focused single-pass extraction of measurable architectural data
from floor plans and measured drawings in data/visual_sources/plans/.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vlm_analysis import (
    detect_vlm_hardware,
    extract_json_from_response,
    load_gemma3_vlm,
    resize_for_vlm,
)

PLANS_PROMPT = """\
You are analyzing an architectural floor plan or measured drawing \
of the Temple of Augustus in Ankara (Templum Divi Augusti, 25 BCE).

Extract ONLY measurable architectural data. Be precise and concise.

Respond ONLY with valid JSON, no other text:
{
  "columns_front": null,
  "columns_side": null,
  "columns_total": null,
  "temple_length_m": null,
  "temple_width_m": null,
  "cella_length_m": null,
  "cella_width_m": null,
  "pronaos_depth_m": null,
  "podium_height_m": null,
  "intercolumniation_m": null,
  "column_diameter_m": null,
  "orientation": null,
  "scale_shown": false,
  "drawing_type": "floor_plan",
  "source_notes": null,
  "sdxl_prompt_component": null
}\
"""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _credibility_tier(filename: str) -> str:
    stem = filename.lower()
    if stem.startswith("idai") or stem.startswith("dai"):
        return "high"
    if stem.startswith("commons"):
        return "medium"
    if stem.startswith("myphoto"):
        return "personal"
    return "medium"


def analyze_plan_image(
    image_path: Path,
    processor,
    model,
) -> dict:
    image = resize_for_vlm(str(image_path))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PLANS_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_len = inputs["input_ids"].shape[-1]
    generate_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    target_device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    generate_inputs = {
        k: v.to(target_device) if hasattr(v, "to") else v
        for k, v in generate_inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **generate_inputs,
            max_new_tokens=400,
            do_sample=False,
        )

    generated = output_ids[0][input_len:]
    raw_response = processor.decode(generated, skip_special_tokens=True)
    return extract_json_from_response(raw_response)


def run_plans_analysis(
    plans_dir: str = "data/visual_sources/plans",
    output_dir: str = "data/analysis/vlm_outputs",
    resume: bool = True,
) -> list:
    plans_dir = Path(plans_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in plans_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No image files found in {plans_dir}")
        return []

    if resume:
        image_files = [
            p for p in image_files
            if not (output_dir / f"{p.stem}_analysis.json").exists()
        ]
        print(f"Resuming: {len(image_files)} remaining")

    total = len(image_files)
    if total == 0:
        print("All plans already analyzed.")
        return []

    hw = detect_vlm_hardware()
    print(f"Loading model ({hw['device']})...")
    processor, model = load_gemma3_vlm()
    print(f"Model ready. Analyzing {total} plan images.\n")

    results = []

    for i, image_path in enumerate(image_files):
        filename = image_path.name
        print(f"[{i+1}/{total}] {filename}", end="", flush=True)

        start = time.time()
        result = analyze_plan_image(image_path, processor, model)
        elapsed = time.time() - start

        result.update({
            "source_filename": filename,
            "source_folder": "plans",
            "credibility_tier": _credibility_tier(filename),
            "pass": "plans_extraction",
            "analysis_timestamp": datetime.now().isoformat(),
        })

        cols_f = result.get("columns_front")
        cols_s = result.get("columns_side")
        length = result.get("temple_length_m")
        width = result.get("temple_width_m")
        dims = f"{length}x{width}m" if length and width else "dims unknown"
        print(f" → columns_front: {cols_f}, columns_side: {cols_s}, {dims} ({elapsed:.0f}s)")

        out_path = output_dir / f"{image_path.stem}_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        results.append(result)

    print(f"\nPlans extracted: {len(results)}")

    confirmed = [
        r for r in results
        if r.get("columns_front") is not None and "error" not in r
    ]
    if confirmed:
        print("\nConfirmed column counts:")
        for r in confirmed:
            length = r.get("temple_length_m")
            width = r.get("temple_width_m")
            dims = f"{length}x{width}m" if length and width else "dims unknown"
            print(
                f"  {r['source_filename']}"
                f" | front: {r.get('columns_front')}"
                f" | side: {r.get('columns_side')}"
                f" | {dims}"
            )

    return results


if __name__ == "__main__":
    run_plans_analysis()
