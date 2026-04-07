"""
SGO Web Interface — FastAPI backend wrapping the SGO pipeline.

Provides a browser UI for:
  1. Describing an entity to evaluate
  2. Generating an evaluator cohort (LLM-generated or uploaded)
  3. Running evaluation against the cohort
  4. Running counterfactual probes to get the semantic gradient

Usage:
    uv run python web/app.py
    # Opens at http://localhost:8000
"""

import json
import os
import re
import asyncio
import time
import uuid
import concurrent.futures
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

# Import core functions from existing scripts
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate import evaluate_one, analyze as analyze_eval, SYSTEM_PROMPT, BIAS_CALIBRATION_ADDENDUM
from counterfactual import probe_one, analyze_gradient, build_changes_block, compute_goal_weights
from ctr_calibrate import sigmoid, fit_platt_scaling, predict_ctr, ctr_derivative
from generate_cohort import generate_segment
from bias_audit import (
    reframe_entity, add_authority_signals, reorder_entity,
    run_paired_evaluation, analyze_probe, generate_report, HUMAN_BASELINES,
)
# Lazy imports — persona_loader pulls in HuggingFace datasets (~5s load)
_persona_loader = None
_stratified_sampler = None

def _lazy_persona_loader():
    global _persona_loader
    if _persona_loader is None:
        import persona_loader as _pl
        _persona_loader = _pl
    return _persona_loader

def _lazy_stratified_sampler():
    global _stratified_sampler
    if _stratified_sampler is None:
        import stratified_sampler as _ss
        _stratified_sampler = _ss
    return _stratified_sampler

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sgo")

app = FastAPI(title="SGO — Semantic Gradient Optimization")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# In-memory store for active sessions
sessions: dict = {}
SESSION_MAX_AGE_HOURS = 24

# Nemotron dataset — loaded once if available
_nemotron_ds = None
_nemotron_checked = False

NEMOTRON_SEARCH_PATHS = [
    Path("/data/nemotron"),  # HF Spaces persistent storage
    PROJECT_ROOT / "data" / "nemotron",
    Path.home() / "data" / "nvidia" / "Nemotron-Personas-USA",
    Path.home() / "data" / "nemotron",
    Path(os.getenv("NEMOTRON_DATA_DIR", "/nonexistent")),
]


def find_nemotron_path():
    """Find Nemotron dataset on disk. Returns path or None."""
    for path in NEMOTRON_SEARCH_PATHS:
        if (path / "dataset_info.json").exists():
            return path
    return None


def get_nemotron(data_dir=None):
    """Load Nemotron dataset. Returns None if not found."""
    global _nemotron_ds, _nemotron_checked
    if data_dir:
        # Explicit path — reset cache
        _nemotron_checked = False
        _nemotron_ds = None
        NEMOTRON_SEARCH_PATHS.insert(0, Path(data_dir))

    if _nemotron_checked:
        return _nemotron_ds

    _nemotron_checked = True
    path = find_nemotron_path()
    if path:
        try:
            _nemotron_ds = _lazy_persona_loader().load_personas(data_dir=path)
            print(f"Nemotron loaded: {len(_nemotron_ds)} personas from {path}")
            return _nemotron_ds
        except Exception as e:
            print(f"Failed to load Nemotron from {path}: {e}")
    return None


# LLM client — uses per-request headers or server env vars. Never stored.

def get_client(api_key=None, base_url=None):
    key = api_key or os.getenv("LLM_API_KEY")
    base = base_url or os.getenv("LLM_BASE_URL")
    if not key:
        raise HTTPException(400, "No API key configured. Enter your key above.")
    return OpenAI(api_key=key, base_url=base, timeout=45, max_retries=2)


def get_model(model=None):
    return model or os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")


def get_fast_model():
    """Smaller model for cheap setup calls (infer-spec, segments, filters, changes)."""
    return os.getenv("LLM_FAST_MODEL", "Qwen/Qwen2.5-7B-Instruct")


IS_SPACES = bool(os.getenv("SPACE_ID"))


def _llm_from_params(api_key: str = "", base_url: str = "", model: str = ""):
    """Extract LLM config from params. Falls back to env vars. Never stored."""
    return (
        get_client(api_key=api_key or None, base_url=base_url or None),
        get_model(model=model or None),
    )


# Rate limiting — per-IP pipeline run counter (not per LLM call)
_rate_limits: dict = {}  # ip -> {"count": N, "reset": timestamp}
RATE_LIMIT_MAX_RUNS = 2  # pipeline runs (evaluate, counterfactual, bias audit) per window
RATE_LIMIT_WINDOW = 3600  # 1 hour


def _check_rate_limit(ip: str):
    """Raise 429 if IP has exceeded pipeline run limit."""
    now = time.time()
    entry = _rate_limits.get(ip)
    if not entry or now > entry["reset"]:
        _rate_limits[ip] = {"count": 1, "reset": now + RATE_LIMIT_WINDOW}
        return
    if entry["count"] >= RATE_LIMIT_MAX_RUNS:
        remaining = int(entry["reset"] - now)
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT_MAX_RUNS} runs per hour. Try again in {remaining // 60}m.")
    entry["count"] += 1


@app.middleware("http")
async def inject_llm_config(request: Request, call_next):
    """Read LLM creds from custom headers (not Authorization — HF proxy intercepts that)."""
    request.state.api_key = request.headers.get("x-llm-key", "")
    request.state.base_url = request.headers.get("x-llm-base", "")
    request.state.model = request.headers.get("x-llm-model", "")
    return await call_next(request)


def llm_from_request(request: Request):
    """Get LLM client+model from the current request. Never logs credentials."""
    return _llm_from_params(
        request.state.api_key, request.state.base_url, request.state.model
    )

NEMOTRON_DATASETS = {
    "USA": "nvidia/Nemotron-Personas-USA",
    "Japan": "nvidia/Nemotron-Personas-Japan",
    "India": "nvidia/Nemotron-Personas-India",
    "Singapore": "nvidia/Nemotron-Personas-Singapore",
    "Brazil": "nvidia/Nemotron-Personas-Brazil",
    "France": "nvidia/Nemotron-Personas-France",
}


# ── Models ────────────────────────────────────────────────────────────────

class EntityInput(BaseModel):
    entity_text: str


class CohortConfig(BaseModel):
    description: str
    audience_context: str = ""
    segments: list[dict]  # [{"label": "...", "count": N}, ...]
    parallel: int = 3


class EvalConfig(BaseModel):
    session_id: str
    parallel: int = 5


class CounterfactualConfig(BaseModel):
    session_id: str
    changes: list[dict]  # [{"id": "...", "label": "...", "description": "..."}, ...]
    min_score: int = 4
    max_score: int = 7
    parallel: int = 5


class CalibrationAnchor(BaseModel):
    mean_score: float
    metric_value: float

class CalibrationInput(BaseModel):
    metric_name: str = "conversion rate"
    metric_unit: str = "%"
    anchors: list[CalibrationAnchor]  # At least 1; first is "current entity"

class SuggestSegmentsInput(BaseModel):
    entity_text: str
    audience_context: str


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/config")
async def get_config():
    """Return current LLM config and Nemotron status."""
    nem_path = find_nemotron_path()
    has_key = bool(os.getenv("LLM_API_KEY"))  # server-level key only; per-session keys checked client-side
    return {
        "model": get_model(),
        "has_api_key": has_key,
        "base_url": os.getenv("LLM_BASE_URL", ""),
        "nemotron_available": nem_path is not None,
        "is_spaces": IS_SPACES,
        "persona_datasets": list(NEMOTRON_DATASETS.keys()),
    }



class SuggestChangesInput(BaseModel):
    entity_text: str
    goal: str
    concerns: list[str]


class NemotronPathInput(BaseModel):
    path: str = "/data/nemotron" if IS_SPACES else "data/nemotron"
    dataset: str = "USA"


@app.post("/api/nemotron/setup")
async def setup_nemotron(input: NemotronPathInput):
    """Point to existing data, or download a Nemotron dataset to the given path."""
    p = Path(input.path).expanduser().resolve()
    # Prevent path traversal — must be within project or /tmp
    if not (p.is_relative_to(PROJECT_ROOT) or p.is_relative_to(Path("/tmp")) or p.is_relative_to(Path("/data"))):
        raise HTTPException(403, "Path must be within the project directory")
    hf_name = NEMOTRON_DATASETS.get(input.dataset, NEMOTRON_DATASETS["USA"])

    if (p / "dataset_info.json").exists():
        ds = get_nemotron(data_dir=str(p))
        if ds is None:
            raise HTTPException(500, "Failed to load dataset")
        return {"status": "loaded", "path": str(p), "count": len(ds), "dataset": input.dataset}

    # Download from HuggingFace
    try:
        from datasets import load_dataset
        print(f"Downloading {hf_name} ...")
        ds = load_dataset(hf_name, split="train")
        p.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(p))
        get_nemotron(data_dir=str(p))
        return {"status": "downloaded", "path": str(p), "count": len(ds), "dataset": input.dataset}
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")


@app.post("/api/session")
async def create_session(entity: EntityInput):
    """Create a new evaluation session with an entity."""
    sid = uuid.uuid4().hex[:12]
    log.info(f"New session {sid} ({len(entity.entity_text)} chars)")
    sessions[sid] = {
        "id": sid,
        "entity_text": entity.entity_text,
        "goal": "",
        "audience": "",
        "cohort": None,
        "eval_results": None,
        "gradient": None,
        "gradient_ranked": None,
        "bias_audit": None,
        "calibration": None,
        "created": datetime.now().isoformat(),
    }
    return {"session_id": sid}


class SessionMetaUpdate(BaseModel):
    goal: str = ""
    audience: str = ""


@app.patch("/api/session/{sid}")
async def update_session_meta(sid: str, meta: SessionMetaUpdate):
    """Update session metadata (goal, audience)."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    if meta.goal:
        sessions[sid]["goal"] = meta.goal
    if meta.audience:
        sessions[sid]["audience"] = meta.audience
    return {"ok": True}


@app.post("/api/calibrate/{sid}")
async def set_calibration(sid: str, cal: CalibrationInput):
    """Set metric calibration for a session. Requires eval results."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[sid]
    if not session["eval_results"]:
        raise HTTPException(400, "Run evaluation first")

    anchors = [{"mean_score": a.mean_score, "metric_value": a.metric_value}
               for a in cal.anchors if a.metric_value > 0]
    if not anchors:
        raise HTTPException(400, "Need at least one anchor with metric_value > 0")
    if any(a["mean_score"] <= 0 for a in anchors):
        raise HTTPException(400, "Mean score must be positive")

    if len(anchors) == 1:
        # Single anchor: linear scaling. metric = k * mean_score
        k = anchors[0]["metric_value"] / anchors[0]["mean_score"]
        session["calibration"] = {
            "metric_name": cal.metric_name,
            "metric_unit": cal.metric_unit,
            "method": "linear",
            "k": k,
            "anchors": anchors,
        }
    else:
        # 2+ anchors: Platt scaling
        platt_anchors = [{"mean_score": a["mean_score"], "real_ctr": a["metric_value"]}
                         for a in anchors]
        a, b = fit_platt_scaling(platt_anchors)
        session["calibration"] = {
            "metric_name": cal.metric_name,
            "metric_unit": cal.metric_unit,
            "method": "platt",
            "a": a, "b": b,
            "anchors": anchors,
        }

    # Re-calibrate existing gradient if available
    result = _apply_calibration(session)
    return {"ok": True, "calibration": session["calibration"], "calibrated_gradient": result}


@app.delete("/api/calibrate/{sid}")
async def clear_calibration(sid: str):
    """Remove metric calibration from a session."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    sessions[sid]["calibration"] = None
    return {"ok": True}


def _apply_calibration(session):
    """Apply calibration to existing gradient data. Returns calibrated ranked list or None."""
    cal = session.get("calibration")
    ranked = session.get("gradient_ranked")
    if not cal or not ranked:
        return None

    valid = [r for r in (session.get("eval_results") or []) if r and isinstance(r.get("score"), (int, float))]
    if not valid:
        return None
    mean_score = sum(r["score"] for r in valid) / len(valid)

    if cal["method"] == "linear":
        k = cal["k"]
        current_metric = k * mean_score
        result = []
        for r in ranked:
            metric_delta = r["avg_delta"] * k
            result.append({
                "id": r["id"],
                "label": r["label"],
                "avg_delta": r["avg_delta"],
                "metric_delta": round(metric_delta, 4),
                "predicted_metric": round(current_metric + metric_delta, 4),
            })
        return {"current_metric": round(current_metric, 4), "items": result}
    elif cal["method"] == "platt":
        a, b = cal["a"], cal["b"]
        current_metric = predict_ctr(a, b, mean_score)
        deriv = ctr_derivative(a, b, mean_score)
        result = []
        for r in ranked:
            metric_delta = r["avg_delta"] * deriv
            result.append({
                "id": r["id"],
                "label": r["label"],
                "avg_delta": r["avg_delta"],
                "metric_delta": round(metric_delta, 4),
                "predicted_metric": round(current_metric + metric_delta, 4),
            })
        return {"current_metric": round(current_metric, 4), "items": result}
    return None


@app.get("/api/session/{sid}")
async def get_session(sid: str):
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    s = sessions[sid]
    return {
        "id": s["id"],
        "has_cohort": s["cohort"] is not None,
        "cohort_size": len(s["cohort"]) if s["cohort"] else 0,
        "has_eval": s["eval_results"] is not None,
        "has_gradient": s["gradient"] is not None,
    }


class InferSpecInput(BaseModel):
    entity_text: str


@app.post("/api/infer-spec")
async def infer_spec(input: InferSpecInput, request: Request):
    """Infer goal and audience from entity text."""
    log.info(f"Infer spec ({request.client.host})")
    client, _ = llm_from_request(request)
    model = get_fast_model()

    prompt = f"""Read this entity and infer two things:
1. What is the most likely GOAL the author has? (what outcome they want)
2. Who is the intended AUDIENCE? (who evaluates or decides)

Entity:
{input.entity_text[:2000]}

Return JSON:
{{
    "goal": "<1 sentence — the outcome they're optimizing for>",
    "audience": "<1 sentence — who should evaluate this, with demographics if obvious>"
}}

Examples:
- Product landing page → goal: "Convert visitors to paying customers", audience: "Software developers evaluating dev tools"
- Resume → goal: "Get interview callbacks from target companies", audience: "Engineering hiring managers at mid-stage startups"
- Professional bio → goal: "Build credibility and attract inbound opportunities", audience: "Industry peers and potential collaborators"
- Pitch deck → goal: "Secure Series A funding", audience: "VCs and angels focused on B2B SaaS"

Be specific to THIS entity, not generic."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=256,
            temperature=0.5,
        )
        content = resp.choices[0].message.content
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return json.loads(content)
    except Exception as e:
        raise HTTPException(500, f"Failed to infer spec: {e}")


@app.post("/api/suggest-changes")
async def suggest_changes(input: SuggestChangesInput, request: Request):
    """Generate candidate changes from evaluation concerns and goal."""
    log.info(f"Suggest changes ({len(input.concerns)} concerns)")
    client, _ = llm_from_request(request)
    model = get_fast_model()

    concerns_text = "\n".join(f"- {c}" for c in input.concerns[:15])
    prompt = f"""Based on these evaluation results, suggest 3-5 specific, actionable changes.

Entity (first 1000 chars):
{input.entity_text[:1000]}

Goal: {input.goal or 'Improve overall reception'}

Top concerns from the persuadable middle (people who scored 4-7):
{concerns_text}

For each change, suggest something that directly addresses one or more concerns.
Only suggest changes the entity owner could realistically make.
Do NOT suggest changes that would fundamentally alter the entity's identity.

Return JSON:
{{
    "changes": [
        {{"id": "change_1", "label": "<short label>", "description": "<what specifically changes, 1-2 sentences>"}}
    ]
}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0.7,
        )
        content = resp.choices[0].message.content
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return json.loads(content)
    except Exception as e:
        raise HTTPException(500, f"Failed to suggest changes: {e}")


@app.post("/api/suggest-segments")
async def suggest_segments(input: SuggestSegmentsInput, request: Request):
    """Use LLM to suggest audience segments based on entity and context."""
    log.info("Suggest segments")
    client, _ = llm_from_request(request)
    model = get_fast_model()

    prompt = f"""Given this entity and audience context, suggest 4-5 evaluator segments.
Each segment should represent a distinct perspective that would evaluate this entity differently.

Entity:
{input.entity_text[:2000]}

Audience context: {input.audience_context}

Return JSON:
{{
    "segments": [
        {{"label": "<concise segment description, 5-10 words>", "count": <6-10>}}
    ]
}}

Make segments specific to THIS domain. For a product, use buyer personas.
For a resume, use different hiring managers. For a pitch, use different investor types. Etc.
Be concrete and relevant — no generic segments."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0.7,
        )
        content = resp.choices[0].message.content
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        data = json.loads(content)
        return data
    except Exception as e:
        raise HTTPException(500, f"Failed to suggest segments: {e}")


def extract_filters(client, model, audience_context, entity_text=""):
    """Use LLM to extract structured Nemotron filters from audience context."""
    if not audience_context.strip():
        return {}

    prompt = f"""Extract structured demographic filters from this audience description.
Only include filters that are explicitly stated or clearly implied.

Audience: {audience_context}
Entity context: {entity_text[:500]}

Return JSON with ONLY the fields that apply (omit fields that aren't specified):
{{
    "sex": "Male" or "Female",
    "age_min": <number>,
    "age_max": <number>,
    "state": "<2-letter state code, e.g. IL for Illinois>",
    "city": "<city name substring>",
    "education_level": ["bachelors", "graduate", ...],
    "occupation": "<occupation substring>"
}}

If the audience is "engineers in Texas aged 25-40", return:
{{"state": "TX", "age_min": 25, "age_max": 40, "occupation": "engineer"}}

If nothing specific is stated, return {{}}."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=256,
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        filters = json.loads(content)
        # Clean empty values
        return {k: v for k, v in filters.items() if v is not None and v != "" and v != []}
    except Exception:
        return {}


@app.post("/api/cohort/generate")
async def generate_cohort_endpoint(config: CohortConfig, request: Request):
    """Generate a cohort — from Nemotron if available, else LLM-generated."""
    total = sum(s.get("count", 8) for s in config.segments)
    log.info(f"Generate cohort: {total} personas, {len(config.segments)} segments")

    ds = get_nemotron()
    if ds is not None:
        # Use census-grounded Nemotron personas
        import random
        pl = _lazy_persona_loader()
        ss = _lazy_stratified_sampler()

        # Extract structured filters from audience context
        client, model = llm_from_request(request)
        filters = extract_filters(client, get_fast_model(), config.audience_context, config.description)
        print(f"Nemotron filters from audience context: {filters}")

        filtered = pl.filter_personas(ds, filters, limit=max(total * 20, 2000))
        profiles = [pl.to_profile(row, i) for i, row in enumerate(filtered)]

        # Use only age + education to keep strata count < total
        dim_fns = [
            lambda p: ss.age_bracket(p.get("age", 30)),
            lambda p: p.get("education_level", "") or "unknown",
        ]
        diversity_fn = lambda p: p.get("occupation", "unknown") or "unknown"

        all_personas = ss.stratified_sample(profiles, dim_fns, total=total,
                                            diversity_fn=diversity_fn)
        # Hard cap — stratified_sample can exceed total when strata > total
        if len(all_personas) > total:
            random.seed(42)
            all_personas = random.sample(all_personas, total)
        source = "nemotron"
    else:
        # Fallback: LLM-generated
        client, model = llm_from_request(request)
        all_personas = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.parallel) as pool:
            futs = {
                pool.submit(generate_segment, client, model,
                            seg["label"], seg["count"], config.description): seg
                for seg in config.segments
            }
            for fut in concurrent.futures.as_completed(futs):
                personas = fut.result()
                all_personas.extend(personas)
        source = "llm-generated"

    for i, p in enumerate(all_personas):
        p["user_id"] = i

    return {
        "cohort_size": len(all_personas),
        "cohort": all_personas, "source": source,
        "filters": filters if ds is not None else None,
    }


@app.post("/api/cohort/upload/{sid}")
async def upload_cohort(sid: str, cohort: list[dict]):
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    sessions[sid]["cohort"] = cohort
    return {"cohort_size": len(cohort)}


# ── SSE streaming endpoints ──────────────────────────────────────────────

@app.get("/api/evaluate/stream/{sid}")
async def evaluate_stream(sid: str, request: Request, parallel: int = 5,
                          bias_calibration: bool = False):
    """Run evaluation with Server-Sent Events for real-time progress."""
    _check_rate_limit(request.client.host)
    log.info(f"Evaluate stream {sid}")
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[sid]
    if not session["cohort"]:
        raise HTTPException(400, "No cohort — generate or upload one first")
    parallel = min(parallel, 10)

    # Capture LLM config from request headers before entering async generator
    _api_key = request.state.api_key
    _base_url = request.state.base_url
    _model = request.state.model

    async def event_generator():
        client, mdl = _llm_from_params(_api_key, _base_url, _model)
        cohort = session["cohort"]
        entity_text = session["entity_text"]
        total = len(cohort)
        sys_prompt = SYSTEM_PROMPT + BIAS_CALIBRATION_ADDENDUM if bias_calibration else None

        yield {"event": "start", "data": json.dumps({
            "total": total, "model": mdl,
            "bias_calibration": bias_calibration,
        })}

        results = [None] * total
        done = 0
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futs = {
                pool.submit(evaluate_one, client, mdl, ev, entity_text,
                            system_prompt=sys_prompt): i
                for i, ev in enumerate(cohort)
            }
            for fut in concurrent.futures.as_completed(futs):
                idx = futs[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"error": str(e), "_evaluator": {"name": "?"}}
                results[idx] = result
                done += 1

                ev = result.get("_evaluator", {})
                progress = {
                    "done": done,
                    "total": total,
                    "name": ev.get("name", "?"),
                    "score": result.get("score"),
                    "action": result.get("action"),
                    "error": result.get("error"),
                }
                yield {"event": "progress", "data": json.dumps(progress)}

        elapsed = time.time() - t0
        session["eval_results"] = results

        analysis = analyze_eval(results)
        valid = [r for r in results if "score" in r]
        scores = [r["score"] for r in valid]
        avg = sum(scores) / len(scores) if scores else 0
        actions = [r["action"] for r in valid]

        summary = {
            "elapsed": round(elapsed, 1),
            "total": len(valid),
            "avg_score": round(avg, 1),
            "positive": actions.count("positive"),
            "neutral": actions.count("neutral"),
            "negative": actions.count("negative"),
            "analysis": analysis,
            "results": results,
        }
        yield {"event": "complete", "data": json.dumps(summary)}

    return EventSourceResponse(event_generator(), ping=15)


class CounterfactualRequest(BaseModel):
    changes: list[dict]
    goal: str = ""
    min_score: int = 4
    max_score: int = 7
    parallel: int = 5


# Store pending counterfactual configs for SSE pickup (with timestamps)
_cf_pending: dict = {}  # ticket -> {"req": CounterfactualRequest, "ts": time.time()}


@app.post("/api/counterfactual/prepare/{sid}")
async def prepare_counterfactual(sid: str, req: CounterfactualRequest):
    """Stage counterfactual config, return a ticket for the SSE stream."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    ticket = uuid.uuid4().hex[:8]
    # Clean expired tickets (>10 min)
    now = time.time()
    expired = [k for k, v in _cf_pending.items() if now - v.get("ts", 0) > 600]
    for k in expired:
        del _cf_pending[k]
    _cf_pending[ticket] = {"req": req, "ts": now, "sid": sid}
    return {"ticket": ticket}


@app.get("/api/counterfactual/stream/{sid}")
async def counterfactual_stream(sid: str, ticket: str, request: Request):
    """Run counterfactual probes with SSE progress."""
    _check_rate_limit(request.client.host)
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[sid]
    if not session["eval_results"]:
        raise HTTPException(400, "Run evaluation first")
    entry = _cf_pending.pop(ticket, None)
    if not entry:
        raise HTTPException(400, "Invalid or expired ticket")
    if entry.get("sid") != sid:
        raise HTTPException(403, "Ticket does not belong to this session")
    req = entry["req"]

    all_changes = req.changes
    goal = req.goal
    min_score = req.min_score
    max_score = req.max_score
    parallel = req.parallel

    _api_key = request.state.api_key
    _base_url = request.state.base_url
    _model = request.state.model
    parallel = min(req.parallel, 10)

    async def event_generator():
        client, mdl = _llm_from_params(_api_key, _base_url, _model)
        cohort = session["cohort"]
        eval_results = session["eval_results"]
        cohort_map = {f"{p.get('name','')}_{p.get('user_id','')}": p for p in cohort}

        movable = [r for r in eval_results
                   if "score" in r and min_score <= r["score"] <= max_score]

        total = len(movable)
        has_goal = bool(goal.strip())
        yield {"event": "start", "data": json.dumps({
            "total": total, "changes": len(all_changes), "model": mdl,
            "goal": goal if has_goal else None,
        })}

        if total == 0:
            yield {"event": "complete", "data": json.dumps({
                "error": "No evaluators in movable middle",
                "gradient": "",
                "results": [],
            })}
            return

        # Compute goal-relevance weights (VJP) if goal is set
        goal_weights = None
        if has_goal:
            yield {"event": "goal_weights", "data": json.dumps({
                "status": "computing", "message": "Scoring evaluator relevance to goal..."
            })}
            goal_weights = compute_goal_weights(
                client, mdl, eval_results, cohort_map, goal, parallel=parallel,
            )
            relevant = sum(1 for v in goal_weights.values() if v["weight"] >= 0.5)
            yield {"event": "goal_weights", "data": json.dumps({
                "status": "done",
                "relevant": relevant,
                "total": len(goal_weights),
                "message": f"{relevant}/{len(goal_weights)} evaluators relevant to goal",
            })}

        results = [None] * total
        done = 0
        t0 = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futs = {
                pool.submit(probe_one, client, mdl, r, cohort_map, all_changes): i
                for i, r in enumerate(movable)
            }
            for fut in concurrent.futures.as_completed(futs):
                idx = futs[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"error": str(e), "_evaluator": {"name": "?"}}
                results[idx] = result
                done += 1

                ev = result.get("_evaluator", {})
                cfs = result.get("counterfactuals", [])
                top = max(cfs, key=lambda c: c.get("delta", 0)) if cfs else {}
                progress = {
                    "done": done,
                    "total": total,
                    "name": ev.get("name", "?"),
                    "original_score": result.get("original_score"),
                    "best_delta": top.get("delta", 0),
                    "best_change": top.get("change_id", "?"),
                    "error": result.get("error"),
                }
                yield {"event": "progress", "data": json.dumps(progress)}

        elapsed = time.time() - t0
        gradient_text, ranked_data = analyze_gradient(results, all_changes,
                                                      goal_weights=goal_weights)
        session["gradient"] = gradient_text
        session["gradient_ranked"] = ranked_data

        # Apply metric calibration if set
        calibrated = _apply_calibration(session)

        yield {"event": "complete", "data": json.dumps({
            "elapsed": round(elapsed, 1),
            "gradient": gradient_text,
            "ranked": ranked_data,
            "results": results,
            "goal": goal if has_goal else None,
            "calibrated": calibrated,
            "calibration": session.get("calibration"),
        })}

    return EventSourceResponse(event_generator(), ping=15)


@app.get("/api/bias-audit/stream/{sid}")
async def bias_audit_stream(
    sid: str, request: Request, probes: str = "framing,authority,order",
    sample: int = 10, parallel: int = 5
):
    """Run bias audit probes with SSE progress."""
    _check_rate_limit(request.client.host)
    parallel = min(parallel, 10)
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[sid]
    if not session["cohort"]:
        raise HTTPException(400, "No cohort — generate or upload one first")

    probe_list = [p.strip() for p in probes.split(",") if p.strip()]

    async def event_generator():
        import random
        _api_key = request.state.api_key
        _base_url = request.state.base_url
        _model = request.state.model
        client, mdl = _llm_from_params(_api_key, _base_url, _model)
        cohort = session["cohort"]
        entity_text = session["entity_text"]

        random.seed(42)
        evaluators = random.sample(cohort, min(sample, len(cohort)))

        yield {"event": "start", "data": json.dumps({
            "probes": probe_list,
            "sample_size": len(evaluators),
            "model": mdl,
        })}

        all_analyses = []

        for probe_name in probe_list:
            yield {"event": "probe_start", "data": json.dumps({"probe": probe_name})}

            t0 = time.time()

            if probe_name == "framing":
                gain_entity = reframe_entity(client, model, entity_text, "gain")
                loss_entity = reframe_entity(client, model, entity_text, "loss")
                results = run_paired_evaluation(
                    client, model, evaluators, gain_entity, loss_entity,
                    "gain", "loss", parallel,
                )
                label_a, label_b = "gain", "loss"
            elif probe_name == "authority":
                entity_with_auth = add_authority_signals(entity_text)
                results = run_paired_evaluation(
                    client, model, evaluators, entity_text, entity_with_auth,
                    "baseline", "authority", parallel,
                )
                label_a, label_b = "baseline", "authority"
            elif probe_name == "order":
                reordered = reorder_entity(entity_text)
                results = run_paired_evaluation(
                    client, model, evaluators, entity_text, reordered,
                    "original", "reordered", parallel,
                )
                label_a, label_b = "original", "reordered"
            else:
                continue

            elapsed = time.time() - t0
            analysis = analyze_probe(results, probe_name, label_a, label_b)
            analysis["elapsed_s"] = round(elapsed, 1)
            all_analyses.append(analysis)

            yield {"event": "probe_complete", "data": json.dumps({
                "probe": probe_name,
                "analysis": analysis,
            })}

        report = generate_report(all_analyses, mdl)
        session["bias_audit"] = {"analyses": all_analyses, "report": report}

        yield {"event": "complete", "data": json.dumps({
            "analyses": all_analyses,
            "report": report,
        })}

    return EventSourceResponse(event_generator(), ping=15)


@app.get("/api/results/{sid}")
async def get_results(sid: str):
    """Get full results for a session."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    s = sessions[sid]
    return {
        "eval_results": s["eval_results"],
        "gradient": s["gradient"],
        "cohort": s["cohort"],
    }


@app.get("/api/report/{sid}")
async def download_report(sid: str):
    """Generate and download a comprehensive markdown report for this session."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    s = sessions[sid]
    if not s["eval_results"]:
        raise HTTPException(400, "No evaluation results yet")

    lines = []
    lines.append("# SGO Evaluation Report")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # Entity
    lines.append("---\n")
    lines.append("## Entity Evaluated\n")
    lines.append(s["entity_text"])
    lines.append("")

    if s.get("goal"):
        lines.append(f"**Goal:** {s['goal']}\n")
    if s.get("audience"):
        lines.append(f"**Audience:** {s['audience']}\n")

    # Cohort summary
    cohort = s.get("cohort") or []
    if cohort:
        lines.append("---\n")
        lines.append(f"## Panel ({len(cohort)} evaluators)\n")
        lines.append("| # | Name | Age | Occupation | Location |")
        lines.append("|---|------|-----|------------|----------|")
        for i, p in enumerate(cohort, 1):
            name = p.get("name", "?")
            age = p.get("age", "")
            occ = p.get("occupation", "")
            loc = p.get("city", p.get("location", ""))
            if p.get("state"):
                loc = f"{loc}, {p['state']}" if loc else p["state"]
            lines.append(f"| {i} | {name} | {age} | {occ} | {loc} |")
        lines.append("")

    # Evaluation results
    results = s["eval_results"]
    valid = [r for r in results if r and "score" in r]
    scores = [r["score"] for r in valid]
    avg = sum(scores) / len(scores) if scores else 0

    lines.append("---\n")
    lines.append("## Evaluation Results\n")
    lines.append(f"**Average Score: {avg:.1f}/10** ({len(valid)} evaluators)\n")

    pos = sum(1 for r in valid if r.get("action") == "positive")
    neu = sum(1 for r in valid if r.get("action") == "neutral")
    neg = sum(1 for r in valid if r.get("action") == "negative")
    lines.append(f"- Would say yes: {pos}")
    lines.append(f"- Unsure: {neu}")
    lines.append(f"- Would say no: {neg}\n")

    # Full analysis from evaluate.py
    analysis = analyze_eval(results)
    lines.append(analysis)
    lines.append("")

    # Individual evaluator details
    lines.append("### All Evaluator Responses\n")
    lines.append("| Name | Age | Occupation | Score | Action | Summary |")
    lines.append("|------|-----|------------|-------|--------|---------|")
    sorted_results = sorted(valid, key=lambda r: r["score"], reverse=True)
    for r in sorted_results:
        ev = r.get("_evaluator", {})
        name = ev.get("name", "?")
        age = ev.get("age", "")
        occ = ev.get("occupation", "")
        score = r["score"]
        action = r.get("action", "")
        summary = r.get("summary", "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {name} | {age} | {occ} | {score}/10 | {action} | {summary} |")
    lines.append("")

    # Counterfactual gradient
    if s.get("gradient"):
        lines.append("---\n")
        lines.append("## Priority Actions (Counterfactual Gradient)\n")
        lines.append(s["gradient"])
        lines.append("")

    # Metric calibration
    if s.get("calibration"):
        cal = s["calibration"]
        lines.append("---\n")
        lines.append(f"## Metric Calibration ({cal['metric_name']})\n")
        lines.append(f"- **Method:** {cal['method']}")
        lines.append(f"- **Unit:** {cal['metric_unit']}")
        for anc in cal.get("anchors", []):
            lines.append(f"- Anchor: score {anc['mean_score']:.1f} = {anc['metric_value']}{cal['metric_unit']}")

        calibrated = _apply_calibration(s)
        if calibrated:
            lines.append(f"\n**Current predicted {cal['metric_name']}:** "
                         f"{calibrated['current_metric']}{cal['metric_unit']}\n")
            lines.append(f"| Change | Score Delta | {cal['metric_name']} Delta | Predicted |")
            lines.append("|--------|-----------|-------------|-----------|")
            for item in calibrated["items"]:
                lines.append(
                    f"| {item['label']} | {item['avg_delta']:+.1f} | "
                    f"{item['metric_delta']:+.4f}{cal['metric_unit']} | "
                    f"{item['predicted_metric']}{cal['metric_unit']} |"
                )
            lines.append("")

    # Bias audit
    if s.get("bias_audit"):
        audit = s["bias_audit"]
        lines.append("---\n")
        lines.append("## Panel Realism Check (Bias Audit)\n")
        if audit.get("report"):
            lines.append(audit["report"])
            lines.append("")
        if audit.get("analyses"):
            lines.append("| Probe | Shifted % | Avg Score Change | Human Baseline | Assessment |")
            lines.append("|-------|-----------|------------------|----------------|------------|")
            baselines = {"framing": 30, "authority": 20, "order": 0}
            for a in audit["analyses"]:
                if a.get("error"):
                    continue
                expected = baselines.get(a["probe"])
                gap = a["shifted_pct"] - (expected or 0)
                if expected is not None:
                    if gap > 10:
                        assessment = "Over-biased"
                    elif gap < -10:
                        assessment = "Under-biased"
                    else:
                        assessment = "Well-calibrated"
                else:
                    assessment = "—"
                lines.append(
                    f"| {a['probe']} | {a['shifted_pct']:.1f}% | "
                    f"{a['avg_abs_delta']:.2f} | "
                    f"{str(expected) + '%' if expected is not None else '—'} | "
                    f"{assessment} |"
                )
            lines.append("")

    lines.append("---\n")
    lines.append("*Report generated by [SGO — Semantic Gradient Optimization](https://github.com/anthropics/sgo)*")

    report_md = "\n".join(lines)
    filename = f"sgo-report-{sid}.md"
    return Response(
        content=report_md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860" if IS_SPACES else "8000"))
    host = "0.0.0.0" if IS_SPACES else "127.0.0.1"

    print(f"\n  SGO Web Interface")
    print(f"  http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, access_log=False)
