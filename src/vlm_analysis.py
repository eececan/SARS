"""
Step 3: VLM Structural Analysis
Gemma-3-4b-it vision-language model for per-image architectural analysis
with two-pass RAG enrichment for high-credibility sources.
"""

import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.retrieval import query as rag_query

MODEL_ID = "google/gemma-3-4b-it"

OTTOMAN_TERMS = [
    "ottoman", "mosque", "pointed arch", "islamic", "minaret",
    "hacı bayram", "haci bayram", "geometric ornament", "byzantine", "medieval",
]

TWO_PASS_FOLDERS = {"full_shots", "architectural_models"}
PLANS_FOLDERS = {"plans"}


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_vlm_hardware() -> dict:
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "device": "cuda",
            "torch_dtype": torch.float16,
            "gpu_name": gpu,
            "vram_gb": round(vram, 1),
        }
    else:
        return {
            "device": "cpu",
            "torch_dtype": torch.bfloat16,
            "gpu_name": None,
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gemma3_vlm():
    hw = detect_vlm_hardware()
    if hw["device"] == "cuda":
        print(f"VLM on GPU: {hw['gpu_name']} | ~1-2 min/image")
    else:
        print("VLM on CPU | ~14 min/image")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=hw["torch_dtype"],
        device_map="auto",
        offload_buffers=True,
    )
    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# Prompt sections
# ---------------------------------------------------------------------------

SECTION_A = """\
You are analyzing a photograph of the Temple of Augustus in Ankara \
(Templum Divi Augusti, 25 BCE), also known as Monumentum Ancyranum.

CRITICAL STRATIGRAPHIC CONTEXT:
The Hacı Bayram Veli mosque (built 1427-1428 CE) was constructed directly \
against the temple's north wall and partially incorporates Roman fabric. \
Photographs may show both structures simultaneously.

You must identify and separate ALL elements by period:
- ROMAN (25 BCE): ashlar masonry, Corinthian order, Latin/Greek inscription \
panels, classical moldings, pilasters, entablature fragments, podium
- OTTOMAN (15th c+): pointed arches, Islamic geometric ornament, mosque \
windows, Ottoman masonry style
- BYZANTINE (4th-6th c): if any visible
- MODERN: concrete repairs, metal barriers, tourist signage, modern mortar, \
scaffolding

Only Roman elements should appear in restoration_plan and rag_search_query fields.\
"""

SECTION_C = """\
Respond ONLY with a valid JSON object. No preamble, no explanation outside the JSON.

{
  "architectural_elements": [
    {
      "element": "element name",
      "period": "Roman/Ottoman/Byzantine/Modern/Unknown",
      "condition": "intact/partial/fragmentary/inferred",
      "confidence": "high/medium/low",
      "description": "detailed architectural description",
      "proportions": "any measurable ratios visible",
      "location_in_image": "where in the frame"
    }
  ],
  "mosque_interference": "none/minimal/partial/dominant",
  "roman_fabric_quality": "excellent/good/fair/poor",
  "stratigraphic_notes": "observations about layering",
  "restoration_plan": [
    "ordered list of what needs reconstruction, Roman elements only"
  ],
  "rag_search_query": "Write a precise technical architectural query using specific terminology: column proportions, entablature measurements, order details, temple typology. DO NOT include site names (Ankara, Augustus) or generic terms. Focus on measurable architectural constraints. Example: 'Corinthian column height diameter ratio eustyle intercolumniation hexastyle peristyle'",
  "sdxl_prompt_component": "dense architectural description for image generation prompt, first century condition, Roman elements only",
  "confidence_zones": {
    "confirmed_roman": "describe what is certain",
    "roman_inferred": "describe what is typologically inferred",
    "uncertain": "describe what cannot be determined"
  },
  "stratigraphic_confidence": {
    "confirmed_roman": 0.0,
    "roman_beneath_mosque": 0.0,
    "roman_adjacent": 0.0,
    "fully_reconstructed": 0.0
  }
}\
"""

PLANS_PROMPT = """\
This is an architectural plan or measured drawing of the Temple of Augustus in Ankara.
Extract ONLY measurable data:
- Column grid: number and spacing
- Cella dimensions (length x width)
- Pronaos depth
- Overall temple footprint
- Any scale or measurements shown
- Podium extent
Respond in JSON only. Be brief.\
"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json_from_response(response: str, debug: bool = False) -> dict:
    import re

    if debug:
        print(f"\n  [DEBUG] Raw response length: {len(response)} chars")
        print(f"  [DEBUG] Raw response START:\n{response[:500]}")
        print(f"  [DEBUG] Raw response END:\n{response[-300:]}")

    text = response.strip()

    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    start = text.find('{')
    end = text.rfind('}')

    if start == -1 or end == -1:
        if debug:
            print(f"  [DEBUG] No JSON braces found in response.")
        return {"error": "no_json_found", "raw": response}

    text = text[start:end+1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if debug:
            print(f"  [DEBUG] json.loads failed: {e}")
            print(f"  [DEBUG] Extracted JSON candidate ({len(text)} chars):\n{text[:600]}")

    try:
        open_b = text.count('{') - text.count('}')
        open_br = text.count('[') - text.count(']')
        if debug:
            print(f"  [DEBUG] Bracket mismatch: {{}} unmatched={open_b}, [] unmatched={open_br}")
        fixed = text + (']' * open_br) + ('}' * open_b)
        return json.loads(fixed)
    except Exception as e2:
        if debug:
            print(f"  [DEBUG] Bracket-fix attempt also failed: {e2}")

    return {"error": "json_parse_failed", "raw": response}


# ---------------------------------------------------------------------------
# SDXL sanitization
# ---------------------------------------------------------------------------

def sanitize_sdxl_component(text: str) -> str:
    if not text:
        return text
    phrases = [p.strip() for p in text.split(",")]
    clean = [p for p in phrases if not any(t in p.lower() for t in OTTOMAN_TERMS)]
    return ", ".join(clean)


# ---------------------------------------------------------------------------
# Image resize
# ---------------------------------------------------------------------------

def resize_for_vlm(image_path: str, max_size: int = 896) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_analysis_prompt(
    keywords: list,
    folder: str,
    rag_context: str = "",
    prompt_override: str = "",
) -> str:
    if prompt_override:
        return prompt_override
    if folder in PLANS_FOLDERS:
        return PLANS_PROMPT

    parts = [SECTION_A]
    kw = [k.lower() for k in keywords]

    if "inscription" in kw or "inscriptions" in folder:
        parts.append(
            "Pay special attention to the Latin and Greek inscription panels. "
            "Note which lines are legible, their position on the wall, and their "
            "relationship to the pilaster framing."
        )

    if "capital" in kw:
        parts.append(
            "Examine the Corinthian capital carefully. Note acanthus leaf arrangement, "
            "volute condition, abacus proportions. Distinguish from any Ottoman "
            "decorative elements nearby."
        )

    if any(k in kw for k in ("model", "reconstructed_model", "reconstruction")):
        parts.append(
            "This is an architectural reconstruction model or drawing, not a photograph "
            "of current ruins. Treat all visible elements as intentional scholarly "
            "reconstruction choices."
        )

    if folder in ("parallels/maison_carree", "parallels/pula"):
        parts.append(
            "This is a PARALLEL TEMPLE, not the Ankara site. Extract proportional data "
            "and typological features that could apply to Augustan Corinthian temple "
            "reconstruction generally."
        )

    if rag_context:
        parts.append(
            "RETRIEVED ACADEMIC CONTEXT:\n"
            "The following constraints come from peer-reviewed sources in your "
            "knowledge base. Use them to refine your analysis:\n\n"
            + rag_context
        )

    parts.append(SECTION_C)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def analyze_image(
    image_path: str,
    keywords: list,
    folder: str,
    processor,
    model,
    rag_context: str = "",
    prompt_override: str = "",
    debug: bool = False,
) -> dict:
    image = resize_for_vlm(image_path)
    prompt_text = build_analysis_prompt(
        keywords, folder, rag_context, prompt_override
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Original working approach: let apply_chat_template handle image+text together
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    # token_type_ids is not accepted by generate()
    generate_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    # With device_map="auto", model.device can be "meta" or "cpu" — force cuda:0
    target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    generate_inputs = {k: v.to(target_device) if hasattr(v, "to") else v
                       for k, v in generate_inputs.items()}

    if debug:
        print(f"  [DEBUG] Input tokens: {input_len}")
        print(f"  [DEBUG] model.device: {model.device}")
        print(f"  [DEBUG] target_device: {target_device}")
        print(f"  [DEBUG] input_ids device: {generate_inputs['input_ids'].device}")
        print(f"  [DEBUG] generate inputs keys: {list(generate_inputs.keys())}")

    with torch.no_grad():
        output_ids = model.generate(
            **generate_inputs,
            max_new_tokens=2048,
            do_sample=False,
        )

    if debug:
        print(f"  [DEBUG] output_ids shape: {output_ids.shape}, input_len: {input_len}")

    generated = output_ids[0][input_len:]

    if debug:
        print(f"  [DEBUG] Output tokens generated: {len(generated)}")
        sample_ids = generated[:20].tolist()
        print(f"  [DEBUG] First 20 token IDs: {sample_ids}")
        if len(generated) >= 2047:
            print("  [DEBUG] *** OUTPUT MAY BE TRUNCATED — hit max_new_tokens limit ***")

    # Try processor.decode first (same as what worked locally), then tokenizer.decode
    raw_response = processor.decode(generated, skip_special_tokens=True)
    if debug:
        print(f"  [DEBUG] processor.decode result len={len(raw_response)} | preview: {repr(raw_response[:300])}")

    if not raw_response.strip():
        raw_response = processor.tokenizer.decode(generated, skip_special_tokens=True)
        if debug:
            print(f"  [DEBUG] tokenizer.decode fallback len={len(raw_response)} | preview: {repr(raw_response[:300])}")
    return extract_json_from_response(raw_response, debug=debug)


# ---------------------------------------------------------------------------
# Two-pass analysis
# ---------------------------------------------------------------------------

def run_two_pass_analysis(
    image_path: str,
    keywords: list,
    folder: str,
    vector_store,
    processor,
    model,
    debug: bool = False,
) -> dict:
    print("  [Pass 1] Running inference (no RAG)...", flush=True)
    result = analyze_image(image_path, keywords, folder, processor, model, debug=debug)

    if "error" in result:
        result["pass"] = "two_pass_failed"
        print(f"  [WARN] Pass 1 JSON parse failed (error={result['error']}) — skipping image, saving error.")
        return result

    rag_query_text = result.get("rag_search_query", "")
    citations = []

    if rag_query_text:
        print(f"  [RAG] Querying: {rag_query_text[:80]}", flush=True)
        retrieved_docs = rag_query(rag_query_text, vectorstore=vector_store, k=5)
        citations = [doc["citation"] for doc in retrieved_docs]
        print(f"  [RAG] Got {len(citations)} citations.", flush=True)

        rag_context = "\n\n".join(
            f"[{doc['citation']}]\n{doc['content'][:200]}"
            for doc in retrieved_docs
        )

        print("  [Pass 2] Running inference (with RAG context)...", flush=True)
        result = analyze_image(
            image_path, keywords, folder, processor, model,
            rag_context=rag_context,
            debug=debug,
        )

        if "error" in result:
            result["pass"] = "two_pass_failed"
            print(f"  [WARN] Pass 2 JSON parse failed (error={result['error']}) — saving error.")
            result["retrieved_citations"] = citations
            return result

    result["retrieved_citations"] = citations
    result["pass"] = "two_pass"
    return result


# ---------------------------------------------------------------------------
# GPU-optimized batch runner (Colab)
# ---------------------------------------------------------------------------

def run_colab_batch(
    conditioning_registry_path: str,
    vector_store,
    output_dir: str,
    resume: bool = True,
) -> list:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(conditioning_registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    eligible = [
        e for e in registry
        if "vlm_analysis" in e.get("suitable_for", [])
    ]

    if resume:
        completed = set()
        for f in output_dir.glob("*_analysis.json"):
            try:
                text = f.read_text(encoding="utf-8")
                if '"error"' not in text:
                    completed.add(f.stem.replace("_analysis", ""))
            except Exception:
                pass
        eligible = [
            e for e in eligible
            if Path(e["source_filename"]).stem not in completed
        ]
        print(f"Resuming: {len(eligible)} remaining")

    _self = sys.modules[__name__]
    processor = getattr(_self, "_colab_processor", None)
    model = getattr(_self, "_colab_model", None)
    if processor is None or model is None:
        raise RuntimeError(
            "Call load_gemma3_vlm() and assign results to "
            "vlm_analysis._colab_processor and vlm_analysis._colab_model "
            "before run_colab_batch(), or use the colab_pipeline notebook."
        )

    total = len(eligible)
    results = []

    for i, entry in enumerate(eligible):
        filename = entry["source_filename"]
        folder = entry["source_folder"]
        keywords = entry.get("keywords", [])

        print(f"\n[{i+1}/{total}] {filename}")
        print(f"  folder: {folder} | tier: {entry['credibility_tier']}")

        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {mem_used:.1f}/{mem_total:.1f}GB used")

        start = time.time()
        source_path = entry.get("source_path") or \
            f"data/visual_sources/{folder}/{filename}"

        try:
            if folder in PLANS_FOLDERS:
                result = analyze_image(
                    source_path, keywords, folder, processor, model,
                    prompt_override=PLANS_PROMPT,
                    debug=True,
                )
                result["pass"] = "plans_extraction"
            elif folder in TWO_PASS_FOLDERS:
                result = run_two_pass_analysis(
                    source_path, keywords, folder,
                    vector_store, processor, model,
                    debug=True,
                )
            else:
                result = analyze_image(
                    source_path, keywords, folder, processor, model,
                    debug=True,
                )
                result["pass"] = "single_pass"

            result["mosque_interference"] = result.get("mosque_interference", "unknown")
            if "sdxl_prompt_component" in result:
                result["sdxl_prompt_component"] = sanitize_sdxl_component(
                    result["sdxl_prompt_component"]
                )
            result.update({
                "source_filename": filename,
                "source_folder": folder,
                "credibility_tier": entry["credibility_tier"],
                "analysis_timestamp": datetime.now().isoformat(),
            })

            elapsed = time.time() - start
            print(f"  ✓ Done in {elapsed:.0f}s | "
                  f"pass: {result.get('pass')} | "
                  f"mosque: {result.get('mosque_interference')}")

        except Exception as e:
            result = {
                "error": str(e),
                "source_filename": filename,
                "source_folder": folder,
                "pass": "failed",
                "analysis_timestamp": datetime.now().isoformat(),
            }
            print(f"  ✗ Failed: {e}")

        stem = Path(filename).stem
        out_path = output_dir / f"{stem}_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    succeeded = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*50}")
    print(f"Batch complete: {succeeded}/{total} succeeded")
    return results


# ---------------------------------------------------------------------------
# CPU batch worker (local)
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def _save_result(result: dict, output_path: Path) -> None:
    with _write_lock:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def _analyze_worker(args: tuple) -> dict:
    entry, vector_store, processor, model, output_dir = args

    filename = entry["source_filename"]
    folder = entry["source_folder"]
    keywords = entry.get("keywords", [])
    out_file = Path(output_dir) / f"{Path(filename).stem}_analysis.json"

    if out_file.exists():
        print(f"  [SKIP] {filename} already done.", flush=True)
        with open(out_file, encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()
    print(f"\n>>> Starting: {filename} | {folder}", flush=True)

    stop_heartbeat = threading.Event()

    def _heartbeat():
        while not stop_heartbeat.wait(60):
            elapsed = int(time.time() - t0)
            print(f"  [alive] {filename} — {elapsed}s elapsed", flush=True)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    try:
        if folder in PLANS_FOLDERS:
            result = analyze_image(
                entry["source_path"], keywords, folder, processor, model,
                prompt_override=PLANS_PROMPT,
            )
            result["pass"] = "plans_extraction"
        elif folder in TWO_PASS_FOLDERS:
            result = run_two_pass_analysis(
                entry["source_path"], keywords, folder,
                vector_store, processor, model,
            )
        else:
            result = analyze_image(
                entry["source_path"], keywords, folder, processor, model,
            )
            result["pass"] = "single_pass"

        result["mosque_interference"] = result.get("mosque_interference", "unknown")
        if "sdxl_prompt_component" in result:
            result["sdxl_prompt_component"] = sanitize_sdxl_component(
                result["sdxl_prompt_component"]
            )
        result["source_filename"] = filename
        result["source_folder"] = folder
        result["credibility_tier"] = entry["credibility_tier"]
        result["analysis_timestamp"] = datetime.now().isoformat()

    except Exception as e:
        result = {
            "source_filename": filename,
            "source_folder": folder,
            "credibility_tier": entry["credibility_tier"],
            "error": str(e),
            "pass": "error",
            "analysis_timestamp": datetime.now().isoformat(),
        }

    stop_heartbeat.set()

    elapsed = time.time() - t0
    mosque = result.get("mosque_interference", "?")
    quality = result.get("roman_fabric_quality", "?")
    pass_type = result.get("pass", "?")
    print(
        f"  └─ Done in {elapsed:.1f}s | pass: {pass_type} | "
        f"mosque: {mosque} | quality: {quality}",
        flush=True,
    )

    _save_result(result, out_file)
    return result


# ---------------------------------------------------------------------------
# CPU batch processing (local)
# ---------------------------------------------------------------------------

def process_all_images(
    registry_path: str,
    vector_store,
    processor,
    model,
    output_dir: str,
    limit: int = None,
    max_workers: int = 2,
) -> list:
    registry_path = Path(registry_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    candidates = [e for e in registry if "vlm_analysis" in e.get("suitable_for", [])]
    if limit:
        candidates = candidates[:limit]

    two_pass = [e for e in candidates if e["source_folder"] in TWO_PASS_FOLDERS]
    single = [e for e in candidates if e["source_folder"] not in TWO_PASS_FOLDERS
              and e["source_folder"] not in PLANS_FOLDERS]
    plans = [e for e in candidates if e["source_folder"] in PLANS_FOLDERS]

    total = len(candidates)
    avg_min = 14
    print(f"Total images to analyze : {total}")
    print(f"  Two-pass (RAG)        : {len(two_pass)}")
    print(f"  Single-pass           : {len(single)}")
    print(f"  Plans extraction      : {len(plans)}")
    print(f"Estimated time          : ~{total * avg_min} min at ~{avg_min} min/image")
    print(f"With {max_workers} workers : ~{(total * avg_min) // max_workers} min estimated\n")

    tasks = [(e, vector_store, processor, model, output_dir) for e in candidates]
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (entry, result) in enumerate(
            zip(candidates, executor.map(_analyze_worker, tasks)), start=1
        ):
            filename = entry["source_filename"]
            folder = entry["source_folder"]
            pass_type = result.get("pass", "?")
            print(f"[ {idx}/{total} ] {filename} | {folder} | {pass_type}")
            all_results.append(result)

    succeeded = sum(1 for r in all_results if "error" not in r)
    failed = total - succeeded

    mosque_counts = Counter(r.get("mosque_interference", "unknown") for r in all_results)
    fabric_counts = Counter(r.get("roman_fabric_quality", "unknown") for r in all_results)
    tier_counts = Counter(r.get("credibility_tier", "unknown") for r in all_results)

    master = {
        "summary": {
            "total_analyzed": total,
            "succeeded": succeeded,
            "failed": failed,
            "by_mosque_interference": dict(mosque_counts),
            "by_roman_fabric_quality": dict(fabric_counts),
            "by_credibility_tier": dict(tier_counts),
        },
        "results": all_results,
    }

    master_path = registry_path.parent.parent / "analysis" / "analysis_registry.json"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    print(f"\nComplete. {succeeded} succeeded, {failed} failed.")
    print(f"Results saved to {output_dir}")
    print(f"Master registry saved to {master_path}")
    return all_results


# ---------------------------------------------------------------------------
# Negative prompt builder
# ---------------------------------------------------------------------------

def build_negative_prompt(analysis_results: list) -> str:
    base = (
        "ruins, damaged stone, missing sections, weathering, moss, vegetation, "
        "cracks, modern elements, concrete, steel, glass, tourist barriers, "
        "signage, scaffolding, fantasy architecture, incorrect proportions, "
        "CGI artifacts, oversaturation, lens flare, HDR photography, 3D render aesthetic"
    )
    mosque = (
        "Ottoman architecture, pointed arches, Islamic geometric ornament, "
        "mosque elements, minaret, Ottoman masonry, Ottoman windows, muqarnas, "
        "Arabic calligraphy, Islamic tiles"
    )
    period = (
        "Byzantine decoration, medieval architecture, Romanesque elements, "
        "Gothic arches, Renaissance details, Baroque ornament, "
        "any post-first-century-BCE architectural style"
    )

    parts = [base, mosque, period]

    heavy_mosque = any(
        r.get("mosque_interference") in ("partial", "dominant")
        for r in analysis_results
    )
    if heavy_mosque:
        parts.append(
            "mosque wall adjacent to temple, Ottoman-Roman hybrid appearance, "
            "mixed period architectural elements"
        )

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Entry point (local CPU run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from huggingface_hub import login
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    login()  # will prompt for token or use HF_TOKEN env var

    BASE = Path(__file__).parent.parent

    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        collection_name="augustus_temple",
        embedding_function=embeddings,
        persist_directory=str(BASE / "data" / "chroma_db"),
    )

    print(f"Loading {MODEL_ID}...")
    processor, model = load_gemma3_vlm()
    print("Model loaded.\n")

    results = process_all_images(
        registry_path=str(BASE / "data" / "conditioning" / "registry" / "conditioning_registry.json"),
        vector_store=vector_store,
        processor=processor,
        model=model,
        output_dir=str(BASE / "data" / "analysis" / "vlm_outputs"),
        limit=None,
        max_workers=1,
    )

    print("\n--- Negative Prompt ---")
    print(build_negative_prompt(results))
