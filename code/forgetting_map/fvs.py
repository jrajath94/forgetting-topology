"""Forgetting Vulnerability Score (FVS).

FVS(c) aggregates per-stage, per-prior-task causal-trace deltas into a single
scalar in [0,1] per component. High FVS => updating c drives forgetting.

The formula (see paper Section 3.3):

    FVS(c) = (1/Z) * sum_k sum_{j<k} w_jk * max(0, Delta_c^{(j,k)})

where w_jk = (K - k) weights earlier-stage tasks proportionally to the number
of *subsequent* fine-tunes they survive, and Z normalises FVS(c) into [0,1]
across components.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json


def load_trace(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_fvs(trace_files: dict[tuple[int, int], str], num_stages: int) -> dict:
    """Aggregate per-stage causal-trace deltas into per-component FVS in [0, 1].

    trace_files: {(j, k): path}  where j is the prior-task index, k the
    current-stage index, j<k. Each file is written by
    ``forgetting_map.causal_trace.causal_trace`` and contains rows keyed by
    ``kind`` + ``layer``.

    Returns: ``{comp_key: fvs_float}`` with ``comp_key = f"{kind}.L{layer}"``.
    Components whose delta is ``NaN`` (e.g., skipped by TG-LoRA and thus
    unable to drive forgetting by update) are silently dropped.
    """
    agg: dict[str, float] = defaultdict(float)
    for (j, k), path in trace_files.items():
        weight = max(1, num_stages - k)
        trace = load_trace(path)
        for row in trace["rows"]:
            delta = row.get("delta")
            if delta is None or (isinstance(delta, float) and delta != delta):  # NaN check
                continue
            key = f"{row['kind']}.L{row['layer']}"
            agg[key] += weight * max(0.0, delta)

    if not agg:
        return {}
    mx = max(agg.values()) or 1.0
    return {k: v / mx for k, v in agg.items()}


def save_fvs(fvs: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fvs, f, indent=2)


def fvs_rankings(fvs: dict) -> list[tuple[str, float]]:
    return sorted(fvs.items(), key=lambda kv: kv[1])


def transferability(fvs_a: dict, fvs_b: dict) -> dict:
    """Spearman rho between FVS_a and FVS_b over shared keys."""
    from scipy.stats import spearmanr
    shared = sorted(set(fvs_a) & set(fvs_b))
    if len(shared) < 10:
        return {"rho": None, "p": None, "n_shared": len(shared)}
    xa = [fvs_a[k] for k in shared]
    xb = [fvs_b[k] for k in shared]
    rho, p = spearmanr(xa, xb)
    return {"rho": float(rho), "p": float(p), "n_shared": len(shared)}
