#!/usr/bin/env python3
"""verify_findings.py — cross-family EXTERNAL_VERIFIER for ai-crucible dogfood-swarm findings.

Adapted from E:/AI/readouts/xrpl-knowledge/scripts/verify_cloud.py. The Claude swarm
auditors GENERATE candidate findings + severities; this seat — a panel of DIFFERENT model
families (large Ollama Cloud models served via the local daemon) — pressure-tests each one
REFUTE-BY-DEFAULT against an external rubric declared separately from the claim. Same-family
judges over-rate (self-preference is mechanistic — Panickssery 2024 arXiv:2404.13076; mitigated
by a cross-family jury, Verga 2024 PoLL arXiv:2404.18796).

Reads a FLAT findings list (swarm/wave-a-audit/findings-flat.json), writes verdicts incrementally
(crash-safe) to swarm/wave-a-audit/verdicts.json. A re-run SKIPS findings already adjudicated by
their full reviewer set (re-run retries only failures). Model-fallback guard: a tier-timeout silently
falls back to the local hermes3:8b — the served model field catches that (compared modulo -cloud).

Reviewer assignment scales with claimed severity (convergence where it matters):
  CRITICAL -> 3 families   HIGH -> 2 families   MEDIUM/LOW -> 1 family (rotated)

Usage:  $env:PYTHONUTF8='1'; python swarm/verify_findings.py [swarm/wave-a-audit/findings-flat.json]
Env:    AIC_VERIFY_TIMEOUT (per-call s, 240) · AIC_VERIFY_WORKERS (4) · AIC_VERIFY_FORCE (1=redo all)
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").replace("http://", "").replace("https://", "").rstrip("/")
URL = f"http://{HOST}/api/chat"
TIMEOUT = int(os.environ.get("AIC_VERIFY_TIMEOUT", "240"))
WORKERS = int(os.environ.get("AIC_VERIFY_WORKERS", "4"))
FORCE = os.environ.get("AIC_VERIFY_FORCE", "") not in ("", "0", "false")

# Cross-family panel (live-verified serving 2026-06-21; all non-Anthropic). Code-strongest first.
PANEL = [
    "deepseek-v4-pro:cloud",     # DeepSeek — coding specialist
    "qwen3-coder:480b-cloud",    # Qwen/Alibaba — code-specialized 480B
    "glm-5:cloud",               # Z.ai — 744B generalist
    "kimi-k2.6:cloud",           # Moonshot — frontier coding
    "gpt-oss:120b-cloud",        # OpenAI — fast
]
N_BY_SEV = {"CRITICAL": 3, "HIGH": 3, "MEDIUM": 1, "LOW": 1}

SYSTEM = (
    "You are an adversarial, cross-family code & measurement-instrument auditor. You did NOT write "
    "this finding and you share NO model family with its author. The subject is ai-crucible, a Python "
    "research instrument (a diagnostic adversarial game for LLMs, thin policy layer on Inspect AI) whose "
    "purpose is a SEALED MEASUREMENT BOUNDARY: a hidden oracle grades out-of-band, a cross-family panel "
    "validates novelty, and the model under measurement is the primary adversary.\n\n"
    "Judge the ONE finding below REFUTE-BY-DEFAULT against this external rubric:\n"
    "  1. REALITY — does the described defect actually exist in the cited code (read the evidence excerpt; "
    "use your own knowledge of Python, statistics, security, concurrency)? Or is it a misread / a documented "
    "intentional design / a non-issue?\n"
    "  2. TRIGGERABILITY — is there a realistic path that triggers it, or is it purely theoretical?\n"
    "  3. SEVERITY CALIBRATION — is the claimed severity justified by the WORST REALISTIC consequence for a "
    "MEASUREMENT INSTRUMENT? CRITICAL/HIGH are reserved for: oracle/answer-key leak or reach; sealed-boundary "
    "(chrome) bleed into scored context; a statistics/grading bug that FLIPS a pass/fail verdict or mis-reports "
    "a measurement; a same-family judge not excluded (breaks external-verifier); a budget/andon bypass; a "
    "security-boundary breach; silent data loss; unrecoverable state; a vacuous gate/seal relied upon. "
    "'could be more defensive / observability / docs drifted / style' is MEDIUM-or-below. Deflate inflated "
    "severities; inflate genuinely under-rated ones.\n\n"
    "Reply with ONLY a JSON object: "
    '{"verdict":"real|false_positive|uncertain","severity_assessment":"agree|deflate|inflate",'
    '"corrected_severity":"CRITICAL|HIGH|MEDIUM|LOW","triggerable":"yes|no|unclear",'
    '"reasoning":"<two sentences max, concrete>"}'
)

_io_lock = threading.Lock()


def _norm(m):
    # Cloud tags come in TWO forms: "...:480b-cloud" (hyphen) and "...:cloud" (colon,
    # e.g. glm-5:cloud / deepseek-v4-pro:cloud). The daemon serves both WITHOUT the cloud
    # suffix. Strip either trailing form so a request matches its served name — while a
    # REAL fallback to the local tier model (hermes3:8b) is still caught.
    m = (m or "").strip().replace(":latest", "")
    for suffix in ("-cloud", ":cloud"):
        if m.endswith(suffix):
            m = m[: -len(suffix)]
            break
    return m


def strip_finding(f):
    """Reasoning-stripped view: the claim + evidence, NOT the auditor's reasoning chain."""
    return {
        "id": f.get("id"),
        "title": f.get("title"),
        "category": f.get("category"),
        "claimed_severity": f.get("severity"),
        "file": f.get("file"),
        "line": f.get("line"),
        "worst_realistic_consequence": f.get("worst_realistic_consequence"),
        "description": f.get("description"),
        "code_evidence": f.get("evidence"),
    }


def call_model(model, finding):
    user = (
        "Adjudicate this ONE finding. Read the code_evidence carefully before deciding.\n\n"
        + json.dumps(strip_finding(finding), ensure_ascii=False, indent=1)
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
        raise RuntimeError(f"model mismatch: requested {model}, daemon served {served}")
    content = ((resp.get("message") or {}).get("content", "") or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content[content.find("{"): content.rfind("}") + 1]
    v = json.loads(content)
    v["model"] = served or model
    return v


def reviewers_for(finding, idx):
    n = N_BY_SEV.get((finding.get("severity") or "MEDIUM").upper(), 1)
    if n == 1:
        return [PANEL[idx % len(PANEL)]]
    return PANEL[:n]  # top-n code-strongest for HIGH/CRITICAL


def process(finding, idx):
    fid = finding.get("id") or f"finding-{idx}"
    models = reviewers_for(finding, idx)
    verdicts, err = [], None
    for m in models:
        for attempt in (1, 2):
            try:
                verdicts.append(call_model(m, finding))
                break
            except Exception as e:
                err = e
                if attempt == 1:
                    time.sleep(3)
        else:
            verdicts.append({"model": m, "verdict": "error", "reasoning": str(err)})
    # consensus
    reals = sum(1 for v in verdicts if v.get("verdict") == "real")
    fps = sum(1 for v in verdicts if v.get("verdict") == "false_positive")
    sev_votes = [v.get("corrected_severity") for v in verdicts if v.get("corrected_severity")]
    consensus = {
        "n_reviewers": len(models),
        "real": reals, "false_positive": fps,
        "uncertain": sum(1 for v in verdicts if v.get("verdict") == "uncertain"),
        "verdict": "real" if reals > fps else ("false_positive" if fps > reals else "split"),
        "severity_votes": sev_votes,
    }
    return fid, {"id": fid, "claimed_severity": finding.get("severity"),
                 "cloud_verdicts": verdicts, "consensus": consensus}


def main(path):
    with open(path, "r", encoding="utf-8") as f:
        findings = json.load(f)
    if isinstance(findings, dict):
        findings = findings.get("findings", [])
    out_path = os.path.join(HERE, "wave-a-audit", "verdicts.json")
    prior = {}
    if os.path.exists(out_path) and not FORCE:
        with open(out_path, "r", encoding="utf-8") as f:
            prior = {v["id"]: v for v in json.load(f).get("verdicts", [])}

    todo = [(i, f) for i, f in enumerate(findings)
            if FORCE or f.get("id") not in prior or "error" in json.dumps(prior.get(f.get("id"), {}))]
    results = dict(prior)
    print(f"cross-family verify: {len(todo)} findings ({len(findings) - len(todo)} cached), "
          f"panel={PANEL}, {WORKERS} workers")

    def save():
        with _io_lock:
            ordered = [results[f.get("id")] for f in findings if f.get("id") in results]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"panel": PANEL, "n_findings": len(findings), "verdicts": ordered}, f,
                          ensure_ascii=False, indent=1)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process, f, i): f for i, f in todo}
        for fut in as_completed(futs):
            fid, rec = fut.result()
            results[fid] = rec
            save()
            c = rec["consensus"]
            print(f"  {fid:22s} claimed={rec['claimed_severity']:8s} -> "
                  f"{c['verdict']:14s} (real={c['real']} fp={c['false_positive']} n={c['n_reviewers']}) "
                  f"sev_votes={c['severity_votes']}")

    # summary
    tally = {"real": 0, "false_positive": 0, "split": 0}
    for v in results.values():
        tally[v["consensus"]["verdict"]] = tally.get(v["consensus"]["verdict"], 0) + 1
    print(f"\nverdicts -> {os.path.relpath(out_path, REPO)}")
    print("  consensus: " + " · ".join(f"{k}={v}" for k, v in tally.items()))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "wave-a-audit", "findings-flat.json")
    main(p)
