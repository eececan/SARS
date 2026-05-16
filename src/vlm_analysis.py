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

TWO_PASS_FOLDERS  = {"full_shots", "architectural_models"}
PLANS_FOLDERS     = {"plans"}
DETAIL_FOLDERS    = {"details", "inscriptions"}
PARALLEL_FOLDERS  = {"parallels/maison_carree", "parallels/pula"}

PARALLEL_NAMES = {
    "parallels/maison_carree": "Maison Carrée, Nîmes, France",
    "parallels/pula":          "Temple of Augustus, Pula, Croatia",
}


# ---------------------------------------------------------------------------
# Hardware / model
# ---------------------------------------------------------------------------

def detect_vlm_hardware() -> dict:
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {"device": "cuda", "torch_dtype": torch.float16,
                "gpu_name": gpu, "vram_gb": round(vram, 1)}
    return {"device": "cpu", "torch_dtype": torch.bfloat16, "gpu_name": None}


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
# Prompt sections — one per folder type
# ---------------------------------------------------------------------------

# Full shots: full mosque/stratigraphic context, all schema fields
_PROMPT_FULLSHOTS = """\
You are analyzing a photograph of the Temple of Augustus in Ankara \
(Templum Divi Augusti, 25 BCE), also known as Monumentum Ancyranum.

STRATIGRAPHIC CONTEXT:
The Hacı Bayram Veli mosque (built 1427–1428 CE) was constructed directly \
against the temple's north wall and partially incorporates Roman fabric. \
The photograph may show Roman masonry, Ottoman masonry, modern scaffolding, \
or a mix. Identify and separate ALL elements by period:
- ROMAN (25 BCE): ashlar masonry, Corinthian order, Latin/Greek inscription \
panels, classical moldings, pilasters, entablature fragments, podium
- OTTOMAN (15th c+): pointed arches, Islamic geometric ornament, mosque \
windows, Ottoman masonry, minaret
- MODERN: concrete repairs, metal barriers, steel scaffolding, tourist signage

Only Roman elements should appear in restoration_plan and sdxl_prompt_component.\
"""

_SCHEMA_FULLSHOTS = """\
Respond ONLY with a raw JSON object — no markdown, no code fences, no explanation. All string values must use escaped quotes (\\") if they contain quotes. Keep free-text fields under 100 words. Schema:
{
  "architectural_elements": [
    {
      "element": "element name",
      "period": "Roman/Ottoman/Modern/Unknown",
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
  "restoration_plan": ["ordered list, Roman elements only"],
  "rag_search_query": "precise technical query — NO site names, NO generic terms. Focus on measurable constraints: column proportions, order details, entablature measurements, temple typology.",
  "sdxl_prompt_component": "dense architectural description, first-century condition, Roman elements only",
  "confidence_zones": {
    "confirmed_roman": "what is certain",
    "roman_inferred": "what is typologically inferred",
    "uncertain": "what cannot be determined"
  },
  "stratigraphic_confidence": {
    "confirmed_roman": 0.0,
    "roman_beneath_mosque": 0.0,
    "roman_adjacent": 0.0,
    "fully_reconstructed": 0.0
  }
}\
"""

# Architectural models / drawings: no mosque context; fully_reconstructed is key
_PROMPT_MODELS = """\
You are analyzing an architectural reconstruction model or measured drawing \
of the Temple of Augustus in Ankara (Templum Divi Augusti, 25 BCE).

This is a SCHOLARLY RECONSTRUCTION — a physical museum model, scale model, \
or architectural drawing produced by researchers. Every visible element \
represents an intentional scholarly interpretation of the Roman temple as it \
appeared in the first century BCE. There is no mosque, no Ottoman fabric, \
and no modern contamination in this image.

Focus entirely on:
- The Roman Corinthian order: column count, capital type, entablature
- Proportional relationships visible in the model or drawing
- Completeness of the reconstruction (which parts are shown as complete vs. missing)
- Quality and detail level of the reconstruction\
"""

_SCHEMA_MODELS = """\
Respond ONLY with a raw JSON object — no markdown, no code fences, no explanation. All string values must use escaped quotes (\\") if they contain quotes. Keep free-text fields under 100 words. Schema:
{
  "architectural_elements": [
    {
      "element": "element name",
      "period": "Roman",
      "condition": "intact/partial/fragmentary/inferred",
      "confidence": "high/medium/low",
      "description": "detailed architectural description",
      "proportions": "any measurable ratios or column counts visible",
      "location_in_image": "where in the frame"
    }
  ],
  "mosque_interference": "none",
  "roman_fabric_quality": "excellent/good/fair/poor",
  "stratigraphic_notes": "notes about completeness of the reconstruction",
  "restoration_plan": ["what the model suggests for reconstruction, in order"],
  "rag_search_query": "precise technical query about Corinthian proportions or Augustan temple typology — no site names",
  "sdxl_prompt_component": "dense architectural description based on what this model/drawing shows",
  "confidence_zones": {
    "confirmed_roman": "elements clearly shown in the model",
    "roman_inferred": "elements the model implies but does not show",
    "uncertain": "parts the model leaves ambiguous"
  },
  "stratigraphic_confidence": {
    "confirmed_roman": 0.0,
    "roman_beneath_mosque": 0.0,
    "roman_adjacent": 0.0,
    "fully_reconstructed": 0.0
  }
}\
"""

# Details / close-ups: no mosque context; only confirmed_roman matters
_PROMPT_DETAILS = """\
You are analyzing a close-up detail photograph of an architectural element \
from the Temple of Augustus in Ankara (Templum Divi Augusti, 25 BCE).

This is a tight crop of a single Roman architectural element — a capital, \
column shaft, entablature fragment, relief, lion figure, or similar detail. \
The mosque is NOT visible in this frame. Do NOT report mosque interference.

Focus entirely on:
- The specific Roman element visible
- Its condition, completeness, and carving quality
- Measurable proportions or dimensions visible
- Any inscriptions, relief decoration, or stylistic markers\
"""

_SCHEMA_DETAILS = """\
Respond ONLY with a raw JSON object — no markdown, no code fences, no explanation. All string values must use escaped quotes (\\") if they contain quotes. Keep free-text fields under 100 words. Schema:
{
  "architectural_elements": [
    {
      "element": "element name",
      "period": "Roman",
      "condition": "intact/partial/fragmentary",
      "confidence": "high/medium/low",
      "description": "detailed architectural and stylistic description",
      "proportions": "any measurable ratios or dimensions visible",
      "location_in_image": "where in the frame"
    }
  ],
  "mosque_interference": "none",
  "roman_fabric_quality": "excellent/good/fair/poor",
  "stratigraphic_notes": "condition and preservation notes for this element",
  "rag_search_query": "precise technical query about this specific element type, its proportions or style",
  "sdxl_prompt_component": "dense description of this element for use in image generation",
  "stratigraphic_confidence": {
    "confirmed_roman": 0.0
  }
}\
"""

# Inscriptions: no mosque context; focus on text content and wall condition
_PROMPT_INSCRIPTIONS = """\
You are analyzing a photograph of the inscription panels (Res Gestae Divi Augusti) \
on the cella wall of the Temple of Augustus in Ankara (25 BCE).

These are Latin and Greek inscription panels carved directly into the Roman \
cella wall. The mosque is NOT the subject of this photograph. \
Do NOT report mosque interference.

Focus entirely on:
- The inscription panels: legibility, position, framing by pilasters
- The Roman cella wall fabric: ashlar masonry type, condition
- Any visible pilasters, moldings, or architectural framing of the inscriptions
- Which lines or sections are legible vs. damaged\
"""

_SCHEMA_INSCRIPTIONS = """\
Respond ONLY with a raw JSON object — no markdown, no code fences, no explanation. All string values must use escaped quotes (\\") if they contain quotes. Keep free-text fields under 100 words. Schema:
{
  "architectural_elements": [
    {
      "element": "element name",
      "period": "Roman",
      "condition": "intact/partial/fragmentary",
      "confidence": "high/medium/low",
      "description": "description including legibility and position",
      "proportions": "panel dimensions or pilaster spacing if visible",
      "location_in_image": "where in the frame"
    }
  ],
  "mosque_interference": "none",
  "roman_fabric_quality": "excellent/good/fair/poor",
  "stratigraphic_notes": "preservation and legibility notes",
  "rag_search_query": "precise query about Roman cella wall construction, inscription panel framing, or pilaster proportions",
  "sdxl_prompt_component": "dense description of inscription wall for image generation",
  "stratigraphic_confidence": {
    "confirmed_roman": 0.0
  }
}\
"""

# Parallels: different location, no Ankara context
_PROMPT_PARALLELS = """\
You are analyzing a photograph of {parallel_name}, a well-preserved Roman \
Corinthian temple used as a typological parallel for Augustan temple reconstruction.

This temple is NOT in Ankara. There is no mosque, no Ottoman masonry, \
and no excavation scaffolding. Extract architectural data that would apply \
to any Augustan-period Corinthian peripteral temple:
- Column count, spacing (intercolumniation), and order details
- Entablature proportions: architrave / frieze / cornice ratios
- Capital type and acanthus leaf arrangement
- Pediment angle and proportions
- Overall temple proportions (length:width ratio, column height:diameter)\
"""

_SCHEMA_PARALLELS = """\
Respond ONLY with a raw JSON object — no markdown, no code fences, no explanation. All string values must use escaped quotes (\\") if they contain quotes. Keep free-text fields under 100 words. Schema:
{
  "architectural_elements": [
    {
      "element": "element name",
      "period": "Roman",
      "condition": "intact/partial/fragmentary",
      "confidence": "high/medium/low",
      "description": "proportional and typological description",
      "proportions": "measurable ratios — be specific",
      "location_in_image": "where in the frame"
    }
  ],
  "mosque_interference": "none",
  "roman_fabric_quality": "excellent/good/fair/poor",
  "stratigraphic_notes": "overall preservation and completeness of this parallel temple",
  "rag_search_query": "precise technical query about Corinthian proportions, Augustan temple typology, or entablature ratios",
  "sdxl_prompt_component": "dense Corinthian temple description for image generation, first-century condition",
  "stratigraphic_confidence": {
    "confirmed_roman": 0.0
  }
}\
"""

PLANS_PROMPT = """\
This is an architectural plan or measured drawing of the Temple of Augustus in Ankara.
Extract ONLY measurable data visible in the drawing:
- Column grid: number across front, number along side
- Cella dimensions (length x width)
- Pronaos depth
- Overall temple footprint
- Any scale or measurements shown
- Podium extent
Respond ONLY with a raw JSON object — no markdown, no code fences. Be brief.\
"""


# ---------------------------------------------------------------------------
# Post-processing: enforce per-folder overrides
# ---------------------------------------------------------------------------

def _postprocess(result: dict, folder: str, metadata: dict = None) -> dict:
    """
    Apply folder-level overrides to correct systematic VLM hallucinations.
    """
    # Parallels and models/drawings can never have mosque interference
    if folder in PARALLEL_FOLDERS or (metadata and metadata.get("is_model_or_drawing")):
        result["mosque_interference"] = "none"
        sc = result.get("stratigraphic_confidence", {})
        sc["roman_beneath_mosque"] = 0.0
        sc["roman_adjacent"]       = 0.0
        result["stratigraphic_confidence"] = sc

    # Detail and inscription crops cannot show mosque interference
    if folder in DETAIL_FOLDERS:
        result["mosque_interference"] = "none"
        sc = result.get("stratigraphic_confidence", {})
        sc.pop("roman_beneath_mosque", None)
        sc.pop("roman_adjacent", None)
        sc.pop("fully_reconstructed", None)
        result["stratigraphic_confidence"] = sc

    # Museum models and scholarly drawings are fully_reconstructed by definition
    if metadata and metadata.get("is_model_or_drawing"):
        sc = result.get("stratigraphic_confidence", {})
        # If model didn't give a high value, set it — these ARE reconstructions
        if sc.get("fully_reconstructed", 0.0) < 0.7:
            sc["fully_reconstructed"] = 1.0
            sc["confirmed_roman"]     = max(sc.get("confirmed_roman", 0.0), 0.8)
        result["stratigraphic_confidence"] = sc

    # Ground shots: mostly ruins, minimal mosque exposure
    if metadata and metadata.get("is_ground_shot"):
        current = result.get("mosque_interference", "partial")
        if current == "dominant":
            result["mosque_interference"] = "partial"

    return result


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
    end   = text.rfind('}')

    if start == -1 or end == -1:
        return {"error": "no_json_found", "raw": response}

    text = text[start:end+1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        open_b  = text.count('{') - text.count('}')
        open_br = text.count('[') - text.count(']')
        fixed   = text + (']' * open_br) + ('}' * open_b)
        return json.loads(fixed)
    except Exception:
        pass

    # Last resort: truncate at the last valid closing brace by trying progressively shorter slices
    for end_pos in range(len(text) - 1, 0, -1):
        if text[end_pos] == '}':
            try:
                return json.loads(text[:end_pos + 1])
            except Exception:
                continue

    return {"error": "json_parse_failed", "raw": response}


# ---------------------------------------------------------------------------
# Image resize
# ---------------------------------------------------------------------------

def resize_for_vlm(image_path: str, max_size: int = 896) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# Prompt builder — routes by folder
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

    # Select the right system context + schema for this folder type
    if folder in PARALLEL_FOLDERS:
        parallel_name = PARALLEL_NAMES.get(folder, "a Roman parallel temple")
        system = _PROMPT_PARALLELS.format(parallel_name=parallel_name)
        schema = _SCHEMA_PARALLELS

    elif folder in DETAIL_FOLDERS and "inscription" in folder:
        system = _PROMPT_INSCRIPTIONS
        schema = _SCHEMA_INSCRIPTIONS

    elif folder in DETAIL_FOLDERS:
        system = _PROMPT_DETAILS
        schema = _SCHEMA_DETAILS

    elif folder in TWO_PASS_FOLDERS:
        kw = [k.lower() for k in keywords]
        is_model = any(k in kw for k in ("model", "drawing"))
        if is_model:
            system = _PROMPT_MODELS
            schema = _SCHEMA_MODELS
        else:
            system = _PROMPT_FULLSHOTS
            schema = _SCHEMA_FULLSHOTS
    else:
        # Unknown folder — use full context as safe default
        system = _PROMPT_FULLSHOTS
        schema = _SCHEMA_FULLSHOTS

    parts = [system]

    # Keyword-specific enrichments
    kw = [k.lower() for k in keywords]

    if "capital" in kw or "columncapital" in kw:
        parts.append(
            "Examine the Corinthian capital carefully. Note acanthus leaf arrangement, "
            "volute condition, abacus proportions. Count leaf rows if possible."
        )

    if rag_context:
        parts.append(
            "RETRIEVED ACADEMIC CONTEXT (use to refine analysis):\n\n" + rag_context
        )

    parts.append(schema)
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
    prompt_text = build_analysis_prompt(keywords, folder, rag_context, prompt_override)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    generate_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    generate_inputs = {
        k: v.to(target_device) if hasattr(v, "to") else v
        for k, v in generate_inputs.items()
    }

    if debug:
        print(f"  [DEBUG] Input tokens: {input_len}")
        print(f"  [DEBUG] target_device: {target_device}")

    with torch.no_grad():
        output_ids = model.generate(
            **generate_inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=1.0,
        )

    generated = output_ids[0][input_len:]

    if debug and len(generated) >= 2047:
        print("  [DEBUG] *** OUTPUT MAY BE TRUNCATED — hit max_new_tokens limit ***")

    raw_response = processor.decode(generated, skip_special_tokens=True)
    if not raw_response.strip():
        raw_response = processor.tokenizer.decode(generated, skip_special_tokens=True)

    return extract_json_from_response(raw_response, debug=debug)


# ---------------------------------------------------------------------------
# Two-pass analysis (full_shots + architectural_models)
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
        return result

    rag_query_text = result.get("rag_search_query", "")
    citations = []

    if rag_query_text:
        print(f"  [RAG] Querying: {rag_query_text[:80]}", flush=True)
        retrieved_docs = rag_query(rag_query_text, vectorstore=vector_store)
        citations = [doc["citation"] for doc in retrieved_docs]
        print(f"  [RAG] Got {len(citations)} citations.", flush=True)

        rag_context = "\n\n".join(
            f"[{doc['citation']}]\n{doc['content'][:300]}"
            for doc in retrieved_docs
        )

        print("  [Pass 2] Running inference (with RAG context)...", flush=True)
        result = analyze_image(
            image_path, keywords, folder, processor, model,
            rag_context=rag_context, debug=debug,
        )

        if "error" in result:
            result["pass"] = "two_pass_failed"
            result["retrieved_citations"] = citations
            return result

    result["retrieved_citations"] = citations
    result["pass"] = "two_pass"
    return result


# ---------------------------------------------------------------------------
# Shared result finalisation
# ---------------------------------------------------------------------------

def _finalise(result: dict, entry: dict, metadata: dict = None) -> dict:
    """Apply sanitization, post-processing overrides, and standard metadata fields."""
    folder = entry["source_folder"]

    if "sdxl_prompt_component" in result:
        result["sdxl_prompt_component"] = sanitize_sdxl_component(
            result["sdxl_prompt_component"]
        )

    result = _postprocess(result, folder, metadata=metadata)

    result.update({
        "source_filename":  entry["source_filename"],
        "source_folder":    folder,
        "credibility_tier": entry["credibility_tier"],
        "analysis_timestamp": datetime.now().isoformat(),
    })
    return result


# ---------------------------------------------------------------------------
# GPU batch runner (Colab)
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

    eligible = [e for e in registry if "vlm_analysis" in e.get("suitable_for", [])]

    if resume:
        completed = set()
        for f in output_dir.glob("*_analysis.json"):
            try:
                if '"error"' not in f.read_text(encoding="utf-8"):
                    completed.add(f.stem.replace("_analysis", ""))
            except Exception:
                pass
        eligible = [e for e in eligible if Path(e["source_filename"]).stem not in completed]
        print(f"Resuming: {len(eligible)} remaining")

    _self = sys.modules[__name__]
    processor = getattr(_self, "_colab_processor", None)
    model     = getattr(_self, "_colab_model", None)
    if processor is None or model is None:
        raise RuntimeError(
            "Assign load_gemma3_vlm() results to "
            "vlm_analysis._colab_processor / _colab_model before calling run_colab_batch()."
        )

    total   = len(eligible)
    results = []

    for i, entry in enumerate(eligible):
        filename = entry["source_filename"]
        folder   = entry["source_folder"]
        keywords = entry.get("keywords", [])

        print(f"\n[{i+1}/{total}] {filename}")
        print(f"  folder: {folder} | tier: {entry['credibility_tier']}")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            tot = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {mem:.1f}/{tot:.1f}GB")

        start       = time.time()
        source_path = entry.get("source_path") or f"data/visual_sources/{folder}/{filename}"

        try:
            if folder in PLANS_FOLDERS:
                result = analyze_image(source_path, keywords, folder, processor, model,
                                       prompt_override=PLANS_PROMPT, debug=True)
                result["pass"] = "plans_extraction"
            elif folder in TWO_PASS_FOLDERS:
                result = run_two_pass_analysis(source_path, keywords, folder,
                                               vector_store, processor, model, debug=True)
            else:
                result = analyze_image(source_path, keywords, folder, processor, model, debug=True)
                result["pass"] = "single_pass"

            result = _finalise(result, entry, metadata=entry)

        except Exception as e:
            result = {
                "error": str(e), "pass": "failed",
                "source_filename": filename, "source_folder": folder,
                "analysis_timestamp": datetime.now().isoformat(),
            }
            print(f"  ✗ Failed: {e}")

        elapsed = time.time() - start
        print(f"  ✓ {elapsed:.0f}s | pass: {result.get('pass')} | "
              f"mosque: {result.get('mosque_interference')}")

        stem     = Path(filename).stem
        out_path = output_dir / f"{stem}_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    succeeded = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*50}\nBatch complete: {succeeded}/{total} succeeded")
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
    folder   = entry["source_folder"]
    keywords = entry.get("keywords", [])
    out_file = Path(output_dir) / f"{Path(filename).stem}_analysis.json"

    if out_file.exists():
        print(f"  [SKIP] {filename}", flush=True)
        with open(out_file, encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()
    print(f"\n>>> {filename} | {folder}", flush=True)

    stop_hb = threading.Event()

    def _heartbeat():
        while not stop_hb.wait(60):
            print(f"  [alive] {filename} — {int(time.time()-t0)}s", flush=True)

    threading.Thread(target=_heartbeat, daemon=True).start()

    try:
        if folder in PLANS_FOLDERS:
            result = analyze_image(entry["source_path"], keywords, folder,
                                   processor, model, prompt_override=PLANS_PROMPT)
            result["pass"] = "plans_extraction"
        elif folder in TWO_PASS_FOLDERS:
            result = run_two_pass_analysis(entry["source_path"], keywords, folder,
                                           vector_store, processor, model)
        else:
            result = analyze_image(entry["source_path"], keywords, folder, processor, model)
            result["pass"] = "single_pass"

        result = _finalise(result, entry, metadata=entry)

    except Exception as e:
        result = {
            "source_filename": filename, "source_folder": folder,
            "credibility_tier": entry["credibility_tier"],
            "error": str(e), "pass": "error",
            "analysis_timestamp": datetime.now().isoformat(),
        }

    stop_hb.set()
    elapsed = time.time() - t0
    print(f"  └─ {elapsed:.1f}s | pass: {result.get('pass')} | "
          f"mosque: {result.get('mosque_interference')} | "
          f"quality: {result.get('roman_fabric_quality')}", flush=True)

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
    max_workers: int = 1,
) -> list:
    registry_path = Path(registry_path)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    candidates = [e for e in registry if "vlm_analysis" in e.get("suitable_for", [])]
    if limit:
        candidates = candidates[:limit]

    two_pass = [e for e in candidates if e["source_folder"] in TWO_PASS_FOLDERS]
    single   = [e for e in candidates if e["source_folder"] not in TWO_PASS_FOLDERS
                and e["source_folder"] not in PLANS_FOLDERS]
    plans    = [e for e in candidates if e["source_folder"] in PLANS_FOLDERS]

    total = len(candidates)
    print(f"Total images to analyze : {total}")
    print(f"  Two-pass (RAG)        : {len(two_pass)}")
    print(f"  Single-pass           : {len(single)}")
    print(f"  Plans extraction      : {len(plans)}")
    print(f"Estimated time          : ~{total * 14} min at ~14 min/image")
    print(f"With {max_workers} workers : ~{(total * 14) // max_workers} min\n")

    tasks      = [(e, vector_store, processor, model, output_dir) for e in candidates]
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (entry, result) in enumerate(
            zip(candidates, executor.map(_analyze_worker, tasks)), start=1
        ):
            print(f"[ {idx}/{total} ] {entry['source_filename']} | "
                  f"{entry['source_folder']} | {result.get('pass','?')}")
            all_results.append(result)

    succeeded = sum(1 for r in all_results if "error" not in r)
    failed    = total - succeeded

    mosque_counts = Counter(r.get("mosque_interference", "unknown") for r in all_results)
    fabric_counts = Counter(r.get("roman_fabric_quality",  "unknown") for r in all_results)
    tier_counts   = Counter(r.get("credibility_tier",       "unknown") for r in all_results)

    master = {
        "summary": {
            "total_analyzed":        total,
            "succeeded":             succeeded,
            "failed":                failed,
            "by_mosque_interference":dict(mosque_counts),
            "by_roman_fabric_quality":dict(fabric_counts),
            "by_credibility_tier":   dict(tier_counts),
        },
        "results": all_results,
    }

    master_path = registry_path.parent.parent / "analysis" / "analysis_registry.json"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    print(f"\nComplete. {succeeded} succeeded, {failed} failed.")
    return all_results


if __name__ == "__main__":
    import argparse
    from src.retrieval import _load_vectorstore

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="HuggingFace access token for Gemma (gated model)")
    parser.add_argument("--limit", type=int, default=None, help="Process only N images (for testing)")
    args = parser.parse_args()

    if args.token:
        from huggingface_hub import login
        login(token=args.token)

    print("Loading vector store...")
    vector_store = _load_vectorstore()

    print("Loading Gemma-3 VLM...")
    processor, model = load_gemma3_vlm()

    base = Path(__file__).parent.parent
    process_all_images(
        registry_path=str(base / "data" / "conditioning" / "registry" / "conditioning_registry.json"),
        vector_store=vector_store,
        processor=processor,
        model=model,
        output_dir=str(base / "data" / "analysis" / "vlm_outputs"),
        limit=args.limit,
    )
