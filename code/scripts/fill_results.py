"""Fill \\pend{key} placeholders in the paper's .tex files from a results JSON.

Usage
-----
From ``papers/emnlp-2026-001``::

    python code/scripts/fill_results.py \\
        --results experiments/results/paper_fill.json \\
        --latex-dir latex \\
        --output-dir latex_filled

Every ``\\pend{key}`` in every .tex file under ``latex/`` becomes the
corresponding value from ``paper_fill.json``. Missing keys stay as
``\\pend{...}`` and the script reports them so the paper cannot silently
ship with unfilled numbers --- the red marker is still visible in the PDF.

Input JSON schema (example)
---------------------------
    {
      "forget_reduction_pct":        "31.4\\\\%",
      "spearman_qwen_llama":         "$\\\\rho = 0.72$",
      "spearman_qwen_llama_p":       "$p < 0.01$",
      "early_layer_cutoff":          "8",
      "anchor_layer_range":          "6--10",
      "mlp_high_fvs_range":          "layers 18--28",
      "tg_lora_vs_lora_current":     "0.4 F1",
      "tg_lora_vs_full_current":     "0.7 F1",
      "tglora_vs_lora_p":            "0.004",
      "tglora_vs_dora_forget":       "2.1 pp better",
      "tglora_vs_vera_forget":       "1.4 pp better",
      "lofit_tglora_jaccard":        "0.38",
      "random_target_forget_delta":  "3.9 pp",
      "tau_sweep_summary":           "tau=0.5 is best; tau=0.25 gives ...",
      "rank_sweep_summary":          "r=16 matches r=32 at 1/2 params",
      "spearman_order":              "$\\\\rho = 0.69$",
      "causal_vs_grad_spearman":     "$\\\\rho = 0.58$",
      "failure_mode_tasks":          "code generation on HumanEval",
      "total_a40_hrs":               "288",
      "carbon_kgco2e":               "104"
    }

Values may contain TeX; the script does a literal textual substitution.
Prefer explicit units (``31.4\\%``, ``0.72``) rather than raw floats.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

PEND_RE = re.compile(r"\\pend\{([^}]+)\}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True,
                   help="JSON mapping pend-key -> TeX value")
    p.add_argument("--latex-dir", required=True,
                   help="Directory containing main.tex and sections/")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the filled .tex tree")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if ANY \\pend{key} remains unfilled")
    return p.parse_args()


def fill_text(text: str, values: dict[str, str]) -> tuple[str, list[str], list[str]]:
    """Return (new_text, filled_keys, still_missing_keys).

    Missing keys are left as ``\\pend{key}`` so the red marker is still
    rendered by the LaTeX macro defined in main.tex.
    """
    filled: list[str] = []
    missing: list[str] = []

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        value = values.get(key)
        # Empty strings (and missing keys) count as "not yet filled" so the
        # red \pend{key} marker keeps rendering. Only non-empty values get
        # substituted. Keys whose name begins with an underscore are
        # treated as JSON-layout comments and ignored.
        if key.startswith("_"):
            return match.group(0)
        if value is None or value == "":
            missing.append(key)
            return match.group(0)
        filled.append(key)
        return value

    new_text = PEND_RE.sub(_sub, text)
    return new_text, filled, missing


def walk_latex(src_dir: Path, dst_dir: Path, values: dict[str, str]) -> tuple[int, dict[str, int]]:
    """Process every .tex and .bib file under src_dir into dst_dir.

    Returns (num_pend_substitutions, missing_key_histogram).
    """
    total_filled = 0
    missing_hist: dict[str, int] = {}

    for src_file in src_dir.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if src_file.suffix in {".tex", ".bib"}:
            text = src_file.read_text()
            new_text, filled, missing = fill_text(text, values)
            dst_file.write_text(new_text)
            total_filled += len(filled)
            for k in missing:
                missing_hist[k] = missing_hist.get(k, 0) + 1
        else:
            # copy bytewise (figures, style files, etc.)
            dst_file.write_bytes(src_file.read_bytes())

    return total_filled, missing_hist


def main() -> int:
    args = parse_args()
    values = json.loads(Path(args.results).read_text())
    src_dir = Path(args.latex_dir)
    dst_dir = Path(args.output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    filled, missing = walk_latex(src_dir, dst_dir, values)
    total_missing_sites = sum(missing.values())

    print(f"[fill] filled {filled} \\pend{{...}} sites -> {dst_dir}")
    if missing:
        print(f"[fill] {total_missing_sites} sites remain unfilled across {len(missing)} distinct keys:")
        for k, n in sorted(missing.items(), key=lambda kv: -kv[1]):
            print(f"  - {k}  (used {n}x)")

    if args.strict and missing:
        print("[fill] strict mode: aborting with non-zero exit")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
