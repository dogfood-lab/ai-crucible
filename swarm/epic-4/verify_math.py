#!/usr/bin/env python3
"""verify_math.py — cross-family EXTERNAL_VERIFIER for the Epic-4 catalog MATH.

Sibling to verify_findings.py, but inverted: that one refutes claimed *defects*; this
one refutes claimed *correctness*. Each entry in math-claims.json is a load-bearing
statistical/logical implementation (Newcombe diff-CI, the saturation e-process,
graduation precedence, the differential classifier, BH-FDR) + its stated property + the
actual code. A panel of DIFFERENT (non-Anthropic) model families pressure-tests each
REFUTE-BY-DEFAULT: find any statistical or logical error, off-by-one, or invalid
assumption. Same-family (Claude) review over-rates (mechanistic self-preference —
Panickssery 2024); a cross-family jury is the mitigation (Verga 2024 PoLL). The seat
ALSO over-flags, so the coordinator synthesizes each verdict against the real code + a
2nd family before acting (cross-family-cloud-verification memo).

Usage:  $env:PYTHONUTF8='1'; python swarm/epic-4/verify_math.py
Env:    AIC_VERIFY_TIMEOUT (per-call s, 300) · AIC_VERIFY_WORKERS (3)
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").replace("http://", "").replace("https://", "").rstrip("/")
URL = f"http://{HOST}/api/chat"
TIMEOUT = int(os.environ.get("AIC_VERIFY_TIMEOUT", "300"))
WORKERS = int(os.environ.get("AIC_VERIFY_WORKERS", "3"))

# Code-strongest cross-family panel (live-verified serving 2026-06-21; all non-Anthropic).
PANEL = ["deepseek-v4-pro:cloud", "qwen3-coder:480b-cloud", "glm-5:cloud"]

SYSTEM = (
    "You are an adversarial, cross-family statistician + code auditor. You did NOT write this "
    "code and share NO model family with its author. The subject is ai-crucible, a measurement "
    "instrument for LLMs; these are its load-bearing statistics. Judge ONE claim REFUTE-BY-DEFAULT: "
    "assume the stated_property is WRONG until the code convinces you. Check, concretely: (1) is the "
    "mathematical formula correct for its stated property (derive/verify it yourself — Newcombe diff "
    "intervals, e-values/test-supermartingales, Wilson, Benjamini-Hochberg, two-proportion tests)? "
    "(2) any off-by-one, mis-paired term, wrong inequality direction, or boundary bug? (3) is the "
    "stated statistical guarantee actually delivered (e.g. is an e-value's expectation <=1 under the "
    "WHOLE composite null, not just the boundary; does BH reject the right set; does the decision "
    "logic have an input that reaches the wrong branch)? Be specific and technical; cite the exact "
    "line/term if you find an error. If it is correct, say so plainly.\n\n"
    "Reply with ONLY a JSON object: "
    '{"verdict":"correct|incorrect|uncertain","confidence":"high|medium|low",'
    '"error_if_any":"<the specific flaw, with the exact term/line, or empty>",'
    '"reasoning":"<3 sentences max, technical and concrete>"}'
)


def _norm(m):
    m = (m or "").strip().replace(":latest", "")
    for suffix in ("-cloud", ":cloud"):
        if m.endswith(suffix):
            return m[: -len(suffix)]
    return m


def call_model(model, claim):
    user = (
        "Adjudicate this ONE math/implementation claim. Verify the formula yourself before deciding.\n\n"
        + json.dumps(claim, ensure_ascii=False, indent=1)
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "stream": False, "format": "json", "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        resp = json.loads(r.read().decode("utf-8"))
    served = resp.get("model", "")
    if served and _norm(served) != _norm(model):
        raise RuntimeError(f"model mismatch: requested {model}, served {served}")
    content = ((resp.get("message") or {}).get("content", "") or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{"): content.rfind("}") + 1]
    v = json.loads(content)
    v["model"] = served or model
    return v


def review_claim(claim):
    verdicts = []
    for m in PANEL:
        for attempt in (1, 2):
            try:
                verdicts.append(call_model(m, claim))
                break
            except Exception as e:
                if attempt == 2:
                    verdicts.append({"model": m, "verdict": "error", "reasoning": str(e)})
                else:
                    time.sleep(3)
    correct = sum(1 for v in verdicts if v.get("verdict") == "correct")
    incorrect = sum(1 for v in verdicts if v.get("verdict") == "incorrect")
    consensus = "correct" if correct > incorrect else ("incorrect" if incorrect > correct else "split")
    return claim["id"], {"id": claim["id"], "consensus": consensus,
                         "correct": correct, "incorrect": incorrect, "verdicts": verdicts}


def main():
    with open(os.path.join(HERE, "math-claims.json"), encoding="utf-8") as f:
        claims = json.load(f)
    print(f"cross-family math verify: {len(claims)} claims, panel={PANEL}, {WORKERS} workers\n")
    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(review_claim, c): c for c in claims}
        for fut in as_completed(futs):
            cid, rec = fut.result()
            results[cid] = rec
            c = rec
            print(f"  {cid:26s} -> {c['consensus']:9s} (correct={c['correct']} incorrect={c['incorrect']})")
            for v in c["verdicts"]:
                if v.get("verdict") == "incorrect":
                    print(f"      ! {v.get('model')}: {v.get('error_if_any','')[:200]}")
    out = os.path.join(HERE, "math-verdicts.json")
    ordered = [results[c["id"]] for c in claims if c["id"] in results]
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"panel": PANEL, "verdicts": ordered}, f, ensure_ascii=False, indent=1)
    tally = {"correct": 0, "incorrect": 0, "split": 0}
    for v in results.values():
        tally[v["consensus"]] = tally.get(v["consensus"], 0) + 1
    print(f"\nverdicts -> swarm/epic-4/math-verdicts.json")
    print("  consensus: " + " · ".join(f"{k}={v}" for k, v in tally.items()))


if __name__ == "__main__":
    main()
