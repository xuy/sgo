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
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

# Import core functions from existing scripts
import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate import evaluate_one, analyze as analyze_eval, SYSTEM_PROMPT, BIAS_CALIBRATION_ADDENDUM
from counterfactual import probe_one, analyze_gradient, build_changes_block
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

app = FastAPI(title="SGO — Semantic Gradient Optimization")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# In-memory store for active sessions
sessions: dict = {}

# Nemotron dataset — loaded once if available
_nemotron_ds = None
_nemotron_checked = False

NEMOTRON_SEARCH_PATHS = [
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


def get_client():
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )


def get_model():
    return os.getenv("LLM_MODEL_NAME", "openai/gpt-4o-mini")


# ── Models ────────────────────────────────────────────────────────────────

class EntityInput(BaseModel):
    entity_text: str


class CohortConfig(BaseModel):
    description: str
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
    return {
        "model": get_model(),
        "has_api_key": bool(os.getenv("LLM_API_KEY")),
        "base_url": os.getenv("LLM_BASE_URL", ""),
        "nemotron_path": str(nem_path) if nem_path else None,
        "nemotron_available": nem_path is not None,
    }


class NemotronPathInput(BaseModel):
    path: str


@app.post("/api/nemotron/setup")
async def setup_nemotron(input: NemotronPathInput):
    """Point to existing Nemotron data, or download it to the given path."""
    p = Path(input.path).expanduser().resolve()

    if (p / "dataset_info.json").exists():
        # Already there — just load it
        ds = get_nemotron(data_dir=str(p))
        if ds is None:
            raise HTTPException(500, "Failed to load dataset")
        return {"status": "loaded", "path": str(p), "count": len(ds)}

    # Not there — download to this path
    from setup_data import setup
    try:
        ds = setup(data_dir=p)
        get_nemotron(data_dir=str(p))
        return {"status": "downloaded", "path": str(p), "count": len(ds)}
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")


@app.post("/api/session")
async def create_session(entity: EntityInput):
    """Create a new evaluation session with an entity."""
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = {
        "id": sid,
        "entity_text": entity.entity_text,
        "cohort": None,
        "eval_results": None,
        "gradient": None,
        "created": datetime.now().isoformat(),
    }
    return {"session_id": sid}


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


@app.post("/api/suggest-segments")
async def suggest_segments(input: SuggestSegmentsInput):
    """Use LLM to suggest audience segments based on entity and context."""
    client = get_client()
    model = get_model()

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


@app.post("/api/cohort/generate")
async def generate_cohort_endpoint(config: CohortConfig):
    """Generate a cohort — from Nemotron if available, else LLM-generated."""
    sid = uuid.uuid4().hex[:12]
    total = sum(s.get("count", 8) for s in config.segments)

    ds = get_nemotron()
    if ds is not None:
        # Use census-grounded Nemotron personas
        import random
        pl = _lazy_persona_loader()
        ss = _lazy_stratified_sampler()
        filtered = pl.filter_personas(ds, {}, limit=max(total * 20, 2000))
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
        client = get_client()
        model = get_model()
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

    sessions[sid] = {
        "id": sid,
        "entity_text": config.description,
        "cohort": all_personas,
        "eval_results": None,
        "gradient": None,
        "created": datetime.now().isoformat(),
    }

    return {
        "session_id": sid, "cohort_size": len(all_personas),
        "cohort": all_personas, "source": source,
    }


@app.post("/api/cohort/upload/{sid}")
async def upload_cohort(sid: str, cohort: list[dict]):
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    sessions[sid]["cohort"] = cohort
    return {"cohort_size": len(cohort)}


# ── SSE streaming endpoints ──────────────────────────────────────────────

@app.get("/api/evaluate/stream/{sid}")
async def evaluate_stream(sid: str, parallel: int = 5, bias_calibration: bool = False):
    """Run evaluation with Server-Sent Events for real-time progress."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[sid]
    if not session["cohort"]:
        raise HTTPException(400, "No cohort — generate or upload one first")

    async def event_generator():
        client = get_client()
        model = get_model()
        cohort = session["cohort"]
        entity_text = session["entity_text"]
        total = len(cohort)
        sys_prompt = SYSTEM_PROMPT + BIAS_CALIBRATION_ADDENDUM if bias_calibration else None

        yield {"event": "start", "data": json.dumps({
            "total": total, "model": model,
            "bias_calibration": bias_calibration,
        })}

        results = [None] * total
        done = 0
        t0 = time.time()
        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futs = {
                pool.submit(evaluate_one, client, model, ev, entity_text,
                            system_prompt=sys_prompt): i
                for i, ev in enumerate(cohort)
            }
            for fut in concurrent.futures.as_completed(futs):
                idx = futs[fut]
                result = fut.result()
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

    return EventSourceResponse(event_generator())


@app.get("/api/counterfactual/stream/{sid}")
async def counterfactual_stream(
    sid: str, changes_json: str, goal: str = "",
    min_score: int = 4, max_score: int = 7, parallel: int = 5
):
    """Run counterfactual probes with SSE progress. Goal enables VJP weighting."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[sid]
    if not session["eval_results"]:
        raise HTTPException(400, "Run evaluation first")

    all_changes = json.loads(changes_json)

    async def event_generator():
        client = get_client()
        model = get_model()
        cohort = session["cohort"]
        eval_results = session["eval_results"]
        cohort_map = {p["name"]: p for p in cohort}

        movable = [r for r in eval_results
                   if "score" in r and min_score <= r["score"] <= max_score]

        total = len(movable)
        has_goal = bool(goal.strip())
        yield {"event": "start", "data": json.dumps({
            "total": total, "changes": len(all_changes), "model": model,
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
                client, model, eval_results, cohort_map, goal, parallel=parallel,
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
                pool.submit(probe_one, client, model, r, cohort_map, all_changes): i
                for i, r in enumerate(movable)
            }
            for fut in concurrent.futures.as_completed(futs):
                idx = futs[fut]
                result = fut.result()
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
        gradient_text = analyze_gradient(results, all_changes,
                                         goal_weights=goal_weights)
        session["gradient"] = gradient_text

        yield {"event": "complete", "data": json.dumps({
            "elapsed": round(elapsed, 1),
            "gradient": gradient_text,
            "results": results,
            "goal": goal if has_goal else None,
        })}

    return EventSourceResponse(event_generator())


@app.get("/api/bias-audit/stream/{sid}")
async def bias_audit_stream(
    sid: str, probes: str = "framing,authority,order",
    sample: int = 10, parallel: int = 5
):
    """Run bias audit probes with SSE progress."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[sid]
    if not session["cohort"]:
        raise HTTPException(400, "No cohort — generate or upload one first")

    probe_list = [p.strip() for p in probes.split(",") if p.strip()]

    async def event_generator():
        import random
        client = get_client()
        model = get_model()
        cohort = session["cohort"]
        entity_text = session["entity_text"]

        random.seed(42)
        evaluators = random.sample(cohort, min(sample, len(cohort)))

        yield {"event": "start", "data": json.dumps({
            "probes": probe_list,
            "sample_size": len(evaluators),
            "model": model,
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

        report = generate_report(all_analyses, model)
        session["bias_audit"] = {"analyses": all_analyses, "report": report}

        yield {"event": "complete", "data": json.dumps({
            "analyses": all_analyses,
            "report": report,
        })}

    return EventSourceResponse(event_generator())


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


if __name__ == "__main__":
    import uvicorn
    print(f"\n  SGO Web Interface")
    print(f"  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
