"""Cross-family OpenRouter SCREENING (exploration, NOT a scored run).

Discipline (memory: openrouter-cross-family-supply.md): FREE models = screening only —
cheaply find which OpenRouter families *might* clear the admission bar. The scored run that
writes panel.json uses PINNED ids + the local seats via characterize.run. This script does
NOT write panel.json and its results are PROVISIONAL (free backends aren't pinned; the roster
churns; no content-hash provenance).

Fidelity where it's cheap: reuses the real OpenRouterModel adapter, the real admission_pairs
items, and the real parse_choice — so the per-family accuracy is comparable to the scored run.
Pragmatism where free demands it: 429-backoff + <=20/min request spacing (the runner does
neither, so a 429-saturated free endpoint would be dropped and misread as a weak judge).

Stratified subset: up to N_PER_CAT items per construct domain (category) for breadth at low
cost. Run: PowerShell injects OPENROUTER_API_KEY from User scope, then `uv run python screen.py`.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx

from ai_crucible.calibration.loader import load_items
from ai_crucible.characterize.run import parse_choice
from ai_crucible.models.openrouter_adapter import OpenRouterModel

# Free, clearly-disjoint-family candidates (none share lineage with the local seats
# gemma=Google / granite=IBM), plus a gemma:free POSITIVE CONTROL — the local gemma4:31b
# already seats, so its OpenRouter sibling clearing validates that the screen measures
# judge quality rather than noise.
CANDIDATES: list[tuple[str, str]] = [
    ("openrouter:meta-llama/llama-3.3-70b-instruct:free", "meta"),
    ("openrouter:qwen/qwen3-next-80b-a3b-instruct:free", "qwen"),
    ("openrouter:cohere/north-mini-code:free", "cohere"),
    ("openrouter:nvidia/nemotron-3-super-120b-a12b:free", "nvidia"),
    ("openrouter:google/gemma-4-31b-it:free", "gemma-control"),
]

N_PER_CAT = 3
SPACING_S = 3.2          # <= ~18/min, under the 20/min ceiling
MAX_429_RETRIES = 4
ITEMS_PATH = Path("src/ai_crucible/calibration/admission_pairs.json")
OUT_PATH = Path("swarm/openrouter-quorum/screening-report.json")


def stratified_subset() -> list:
    items = load_items(ITEMS_PATH)
    by_cat: dict[str, list] = {}
    for it in items:
        by_cat.setdefault(it.category.value, []).append(it)
    chosen = []
    for cat, group in sorted(by_cat.items()):
        group_sorted = sorted(group, key=lambda i: (i.difficulty or 0.0))
        chosen.extend(group_sorted[:N_PER_CAT])
    return chosen


async def call_with_429_backoff(model: OpenRouterModel, prompt: str):
    delay = 5.0
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            return await model.judge_item(prompt, run_index=0)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < MAX_429_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise


async def screen_model(spec: str, family: str, items: list) -> dict:
    model = OpenRouterModel(model_id=spec, family=family)
    n_correct = n_unparsed = n_dropped = 0
    per_item = []
    for it in items:
        gold = str(it.gold).upper()
        try:
            rec = await call_with_429_backoff(model, it.prompt)
        except Exception as exc:  # noqa: BLE001
            n_dropped += 1
            per_item.append({"id": it.id, "result": "DROPPED", "err": repr(exc)[:120]})
            await asyncio.sleep(SPACING_S)
            continue
        parsed = parse_choice(str(rec.predicted), gold)
        if parsed is None:
            n_unparsed += 1
            per_item.append({"id": it.id, "result": "UNPARSED", "raw": str(rec.predicted)[:60]})
        else:
            correct = parsed == gold
            n_correct += int(correct)
            per_item.append({"id": it.id, "gold": gold, "parsed": parsed, "correct": correct})
        await asyncio.sleep(SPACING_S)
    n_scored = len(items) - n_dropped
    acc = (n_correct / n_scored) if n_scored else None
    return {
        "spec": spec,
        "family": family,
        "n_items": len(items),
        "n_correct": n_correct,
        "n_unparsed": n_unparsed,
        "n_dropped": n_dropped,
        "accuracy": acc,
        "per_item": per_item,
    }


async def main() -> None:
    items = stratified_subset()
    print(f"screening {len(CANDIDATES)} free models over {len(items)} stratified items "
          f"(<= {1/SPACING_S*60:.0f}/min)\n", flush=True)
    results = []
    for spec, fam in CANDIDATES:
        t0 = time.monotonic()
        res = await screen_model(spec, fam, items)
        dt = time.monotonic() - t0
        acc = res["accuracy"]
        acc_s = f"{acc:.3f}" if acc is not None else "n/a"
        print(f"  {fam:14s} acc={acc_s}  correct={res['n_correct']}/{res['n_items']-res['n_dropped']} "
              f"unparsed={res['n_unparsed']} dropped={res['n_dropped']}  ({dt:.0f}s)", flush=True)
        results.append(res)
    report = {
        "kind": "SCREENING (provisional, free models — NOT a scored run; no panel.json written)",
        "n_items_per_model": len(items),
        "spacing_s": SPACING_S,
        "candidates": results,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nscreening report -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
