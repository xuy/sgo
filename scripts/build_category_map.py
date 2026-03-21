"""
Build a target-aware category mapping for stratified sampling.

Instead of hardcoded keyword buckets, this sends all unique category values
from the dataset to an LLM along with the entity description. The LLM returns
a grouping that's meaningful for the specific evaluation target.

The mapping is cached so it's only generated once per entity + field combination.

Usage:
    # Build occupation mapping for a specific entity
    uv run python scripts/build_category_map.py \
      --entity entities/my_product.md \
      --field occupation \
      --output data/occupation_map.json

    # As a library
    from build_category_map import load_or_build_map
    occ_map = load_or_build_map("occupation", entity_text, unique_values)
"""

import json
import hashlib
import os
import re
import argparse
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

CACHE_DIR = PROJECT_ROOT / "data" / "category_maps"

SYSTEM_PROMPT = """You are a research methodology assistant. Your job is to create
meaningful groupings of categorical values for stratified sampling.

You will receive:
1. A list of unique category values from a dataset
2. A description of the entity being evaluated

Create 6-10 groups that ensure the evaluation cohort captures meaningfully
different perspectives on the entity. Groups should reflect how people in these
categories would DIFFER in their evaluation of the entity — not just demographic
similarity.

You MUST respond with valid JSON only."""

MAP_PROMPT = """## Entity Being Evaluated

{entity}

---

## Unique Values to Group

Field: {field}
Values ({count} unique):

{values}

---

## Task

Group these {count} values into 6-10 buckets that capture meaningfully different
perspectives on the entity above. Every value must appear in exactly one bucket.

Think about: Who would evaluate this entity differently? What professional/life
context changes how someone perceives this?

Return JSON:
{{
    "buckets": [
        {{
            "name": "<short bucket label>",
            "rationale": "<why this group evaluates the entity differently>",
            "values": ["<value1>", "<value2>", ...]
        }}
    ]
}}"""


def extract_unique_values(field, data_dir=None):
    """Extract unique values for a field from the Nemotron dataset.

    Uses HuggingFace datasets (load_from_disk), consistent with setup_data.py
    and persona_loader.py.
    """
    from collections import Counter
    from datasets import load_from_disk

    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "nemotron"

    if not (data_dir / "dataset_info.json").exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. Run: uv run python scripts/setup_data.py"
        )

    ds = load_from_disk(str(data_dir))
    return Counter(ds[field])


def build_map(field, entity_text, unique_values, client=None, model=None):
    """Call LLM to build a target-aware category mapping."""
    if client is None:
        client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
    if model is None:
        model = os.getenv("LLM_MODEL_NAME")

    values_text = "\n".join(f"  - {v}" for v in sorted(unique_values))

    prompt = MAP_PROMPT.format(
        entity=entity_text,
        field=field,
        count=len(unique_values),
        values=values_text,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=16384,
        temperature=0.3,
    )

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Empty response from LLM")
    content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
    data = json.loads(content)

    # Flatten to value -> bucket_name mapping
    mapping = {}
    for bucket in data["buckets"]:
        for val in bucket["values"]:
            mapping[val] = bucket["name"]

    # Check coverage
    mapped = set(mapping.keys())
    expected = set(unique_values)
    missing = expected - mapped
    if missing:
        print(f"  Warning: {len(missing)} values not mapped by LLM, assigning to 'other':")
        for v in sorted(missing)[:10]:
            print(f"    - {v}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        for v in missing:
            mapping[v] = "other"

    extra = mapped - expected
    if extra:
        print(f"  Note: LLM included {len(extra)} values not in dataset (ignored)")

    return {
        "field": field,
        "buckets": data["buckets"],
        "mapping": mapping,
    }


def cache_key(field, entity_text):
    """Generate a stable cache key from field + entity content."""
    h = hashlib.sha256(entity_text.encode()).hexdigest()[:12]
    return f"{field}_{h}"


def load_or_build_map(field, entity_text, unique_values,
                       client=None, model=None, cache_dir=None):
    """Load cached mapping or build a new one."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = cache_key(field, entity_text)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        print(f"  Loading cached {field} mapping: {cache_path.name}")
        with open(cache_path) as f:
            data = json.load(f)
        return data["mapping"]

    print(f"  Building {field} mapping ({len(unique_values)} unique values)...")
    data = build_map(field, entity_text, unique_values, client, model)

    with open(cache_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Cached to {cache_path.name}")

    # Print bucket summary
    for b in data["buckets"]:
        print(f"    {b['name']}: {len(b['values'])} values — {b['rationale']}")

    return data["mapping"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="Path to entity document")
    parser.add_argument("--field", default="occupation",
                        help="Dataset field to map (default: occupation)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to Nemotron arrow shards")
    parser.add_argument("--output", default=None,
                        help="Output path (default: data/category_maps/<key>.json)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if cached")
    args = parser.parse_args()

    entity_text = Path(args.entity).read_text()
    counts = extract_unique_values(args.field, args.data_dir and Path(args.data_dir))
    unique_values = list(counts.keys())

    print(f"Field: {args.field} | {len(unique_values)} unique values")
    print(f"Entity: {args.entity}")

    if args.force:
        key = cache_key(args.field, entity_text)
        cache_path = CACHE_DIR / f"{key}.json"
        if cache_path.exists():
            cache_path.unlink()
            print("  Cleared cache")

    mapping = load_or_build_map(args.field, entity_text, unique_values)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"\nSaved mapping to {args.output}")

    # Summary
    from collections import Counter
    bucket_counts = Counter(mapping.values())
    print(f"\nBucket distribution:")
    for bucket, cnt in bucket_counts.most_common():
        print(f"  {bucket}: {cnt} categories")


if __name__ == "__main__":
    main()
