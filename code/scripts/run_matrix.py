"""Orchestrate the full (method x seed) experiment matrix for a single model.

One model takes a config YAML and sweeps every ``(method, seed)`` cell, running
the full task sequence (NER -> QA -> Sum -> Code) per cell. Per-cell results
are written as JSON under ``experiments/results/`` and can be resumed from.

Typical invocation on an A40 pod::

    python scripts/run_matrix.py \
      --config experiments/configs/qwen3_8b_main.yaml \
      --out-dir experiments/results/qwen3_8b \
      --resume

Resume behaviour: if a cell's result JSON already exists, the cell is skipped.
Safe to run multiple times; safe to re-invoke after a pod eviction.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# Make local packages importable when run as ``python scripts/run_matrix.py``
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# run_sequence import is deferred until we actually run a cell, so --dry-run
# works without torch/HF being installed.


def _cell_result_path(out_dir: Path, method_name: str, seed: int) -> Path:
    return out_dir / f"{method_name}__s{seed}.json"


def _cell_ckpt_dir(out_dir: Path, method_name: str, seed: int) -> Path:
    return out_dir / "ckpts" / f"{method_name}__s{seed}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config from experiments/configs/")
    p.add_argument("--out-dir", required=True, help="Per-cell results + checkpoints land here")
    p.add_argument("--subsample", type=int, default=None, help="Cap train/eval per task (for debug)")
    p.add_argument("--resume", action="store_true", help="Skip cells whose result JSON already exists")
    p.add_argument("--methods", default=None, help="Comma-separated subset of methods to run")
    p.add_argument("--seeds", default=None, help="Comma-separated subset of seeds to run")
    p.add_argument("--dry-run", action="store_true", help="Print the cell plan, do not train")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_filter = set(args.methods.split(",")) if args.methods else None
    seed_filter = set(map(int, args.seeds.split(","))) if args.seeds else None

    cells = []
    for method_spec in cfg["methods"]:
        name = method_spec["name"]
        if method_filter and name not in method_filter:
            continue
        for seed in cfg["seeds"]:
            if seed_filter and seed not in seed_filter:
                continue
            cells.append((name, seed, method_spec))
    print(f"[matrix] {len(cells)} cells queued for model={cfg['model']}")

    if args.dry_run:
        for name, seed, spec in cells:
            print(f"  [dry] {name:12s} seed={seed} extra={{ {spec} }}")
        return

    t_start = time.time()
    for i, (name, seed, spec) in enumerate(cells):
        result_path = _cell_result_path(out_dir, name, seed)
        if args.resume and result_path.exists():
            print(f"[matrix] [{i+1}/{len(cells)}] SKIP (exists): {result_path.name}")
            continue

        ckpt_dir = _cell_ckpt_dir(out_dir, name, seed)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[matrix] [{i+1}/{len(cells)}] RUN method={name} seed={seed}")

        t_cell = time.time()
        extra = {}
        if name == "ewc":
            extra["ewc_lambda"] = spec.get("ewc_lambda", 500.0)
        elif name == "lora_replay":
            extra["replay_frac"] = spec.get("replay_frac", 0.1)
        elif name == "tg_lora":
            fvs_path = spec.get("fvs_path")
            assert fvs_path, f"tg_lora cell requires fvs_path in config; got {spec}"
            extra["fvs_path"] = fvs_path

        try:
            from forgetting_map.sequential_finetune import run_sequence
            _model, records = run_sequence(
                base_model=cfg["model"],
                method=name,
                tasks=cfg["tasks"],
                seed=seed,
                out_dir=str(ckpt_dir),
                subsample=args.subsample,
                **extra,
            )
        except Exception as e:
            print(f"[matrix]   FAILED: {type(e).__name__}: {e}")
            (out_dir / f"{name}__s{seed}.ERROR.log").write_text(f"{type(e).__name__}: {e}")
            continue

        payload = {
            "model": cfg["model"],
            "method": name,
            "seed": seed,
            "tasks": cfg["tasks"],
            "wall_seconds": time.time() - t_cell,
            "records": [r.__dict__ for r in records],
        }
        result_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[matrix]   DONE in {payload['wall_seconds']:.0f}s -> {result_path.name}")

    print(f"[matrix] total wall = {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
