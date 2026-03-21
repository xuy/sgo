"""
LLM-generated cohort — for domains where Nemotron doesn't fit.

When you need personas that don't exist in the population dataset (e.g., B2B
buyer personas, VC investors, hiring managers), this script generates them
via LLM with explicit stratification constraints.

WARNING: See README.md § The Seeding Problem. LLM-generated personas are
subject to mode collapse and invisible bias. Use census-grounded datasets
(Nemotron) when possible. This script is the fallback.

Usage:
    uv run python scripts/generate_cohort.py \
      --description "B2B SaaS buyers evaluating a data pipeline tool" \
      --segments '[
        {"label": "Solo dev, bootstrap", "count": 8},
        {"label": "Startup eng manager, Series A", "count": 8},
        {"label": "Enterprise CTO, 500+ employees", "count": 8},
        {"label": "Data analyst, non-technical", "count": 8},
        {"label": "DevOps engineer, mid-size company", "count": 8}
      ]' \
      --output data/cohort.json
"""

import json
import os
import re
import argparse
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

SYSTEM_PROMPT = """You generate realistic, diverse personas for evaluation simulations.
Each persona must be a distinct, internally consistent individual — not a stereotype.
Include: name, age, location, education, occupation, personality traits, values,
priorities, budget constraints, technical background, and decision-making style.
Vary across gender, ethnicity, geography, and temperament.

You MUST respond with valid JSON only."""

GENERATE_PROMPT = """Generate {count} distinct personas matching this segment:

Segment: {segment_label}
Context: {description}

Each persona should be 200-400 words and feel like a real person, not a marketing archetype.

Return JSON:
{{
    "personas": [
        {{
            "name": "<realistic full name>",
            "age": <integer>,
            "sex": "<Male | Female>",
            "city": "<city>",
            "state": "<state abbreviation>",
            "country": "USA",
            "education_level": "<high_school | bachelors | graduate | etc>",
            "occupation": "<specific job title>",
            "marital_status": "<never_married | married | divorced | widowed | separated>",
            "interests": ["<hobby or skill, 3-5 items>"],
            "persona": "<200-400 word detailed persona narrative>",
            "segment": "{segment_label}"
        }}
    ]
}}"""


def generate_segment(client, model, segment_label, count, description):
    prompt = GENERATE_PROMPT.format(
        count=count, segment_label=segment_label, description=description
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=16384,
            temperature=0.8,
        )
        content = resp.choices[0].message.content
        if not content:
            return []
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        data = json.loads(content)
        return data.get("personas", [])
    except Exception as e:
        print(f"  ERROR generating '{segment_label}': {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", required=True, help="Context for persona generation")
    parser.add_argument("--segments", required=True, type=json.loads,
                        help='JSON array: [{"label": "...", "count": N}, ...]')
    parser.add_argument("--output", default="data/cohort.json")
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
    model = os.getenv("LLM_MODEL_NAME")

    print(f"Generating personas | Model: {model}")
    print(f"Context: {args.description}")
    print(f"Segments: {len(args.segments)}\n")

    print("⚠️  WARNING: LLM-generated personas are subject to mode collapse.")
    print("   Use census-grounded datasets (Nemotron) when possible.\n")

    all_personas = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = {
            pool.submit(generate_segment, client, model,
                        seg["label"], seg["count"], args.description): seg
            for seg in args.segments
        }
        for fut in concurrent.futures.as_completed(futs):
            seg = futs[fut]
            personas = fut.result()
            print(f"  {seg['label']}: {len(personas)} personas generated")
            all_personas.extend(personas)

    # Assign user_ids
    for i, p in enumerate(all_personas):
        p["user_id"] = i

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_personas, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_personas)} personas to {args.output}")


if __name__ == "__main__":
    main()
