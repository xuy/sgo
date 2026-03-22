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

app = FastAPI(title="SGO — Semantic Gradient Optimization")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# In-memory store for active sessions
sessions: dict = {}


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


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/config")
async def get_config():
    """Return current LLM config (model name, whether API key is set)."""
    return {
        "model": get_model(),
        "has_api_key": bool(os.getenv("LLM_API_KEY")),
        "base_url": os.getenv("LLM_BASE_URL", ""),
    }


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


@app.post("/api/cohort/generate")
async def generate_cohort_endpoint(config: CohortConfig):
    """Generate an LLM cohort and attach to a new session."""
    sid = uuid.uuid4().hex[:12]

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

    return {"session_id": sid, "cohort_size": len(all_personas), "cohort": all_personas}


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
    sid: str, changes_json: str, min_score: int = 4,
    max_score: int = 7, parallel: int = 5
):
    """Run counterfactual probes with SSE progress."""
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
        yield {"event": "start", "data": json.dumps({
            "total": total, "changes": len(all_changes), "model": model
        })}

        if total == 0:
            yield {"event": "complete", "data": json.dumps({
                "error": "No evaluators in movable middle",
                "gradient": "",
                "results": [],
            })}
            return

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
        gradient_text = analyze_gradient(results, all_changes)
        session["gradient"] = gradient_text

        yield {"event": "complete", "data": json.dumps({
            "elapsed": round(elapsed, 1),
            "gradient": gradient_text,
            "results": results,
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
