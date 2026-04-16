"""End-to-end pilot for the forgetting topology experiment.

Usage (local, small model, to smoke-test the harness)::

    python run_pilot.py \
      --model Qwen/Qwen3-1.7B \
      --tasks ner,qa \
      --method lora \
      --subsample 100 \
      --seed 42 \
      --out experiments/results/pilot.json

Usage (A40, full)::

    python run_pilot.py \
      --model Qwen/Qwen3-8B \
      --tasks ner,qa,sum,code \
      --method tg_lora --fvs-path experiments/results/fvs_qwen3_8b.json \
      --seed 42 \
      --out experiments/results/qwen3_8b_tg_lora_s42.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Make local packages importable when run as ``python run_pilot.py``
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forgetting_map.sequential_finetune import run_sequence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="ner,qa,sum,code")
    p.add_argument("--method", default="lora",
                   choices=["full_ft", "lora", "tg_lora", "ewc", "sdft", "lora_replay"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=int, default=None,
                   help="Subsample train/eval to N for pilot runs")
    p.add_argument("--fvs-path", default=None,
                   help="Required when method=tg_lora; path to FVS JSON")
    p.add_argument("--out", default="experiments/results/pilot.json")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    _, records = run_sequence(
        base_model=args.model,
        method=args.method,
        tasks=tasks,
        seed=args.seed,
        out_dir=str(out_dir / "ckpts"),
        subsample=args.subsample,
        fvs_path=args.fvs_path,
    )

    payload = {
        "model": args.model,
        "method": args.method,
        "tasks": tasks,
        "seed": args.seed,
        "subsample": args.subsample,
        "records": [r.__dict__ for r in records],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
