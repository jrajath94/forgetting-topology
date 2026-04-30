"""Render the per-component Forgetting Vulnerability Score heatmap.

Input
-----
One or more FVS JSON files written by ``forgetting_map.fvs.save_fvs``.
Each file maps ``{kind}.L{layer}`` -> float in [0, 1].

Output
------
A vector PDF (``--out-pdf``) suitable for ``\\includegraphics`` in main.tex.
When two FVS files are passed, the PDF is a side-by-side comparison
(Qwen3-8B left, Llama-3.1-8B right).

Usage
-----
::

    python code/scripts/make_fig_topology.py \\
        --fvs-files experiments/results/fvs_qwen3_8b.json \\
                    experiments/results/fvs_llama31_8b.json \\
        --labels    Qwen3-8B Llama-3.1-8B \\
        --out-pdf   papers/neurips-2026-forgetting/latex/figures/fig_topology.pdf

Design notes
------------
* We use ``viridis`` reversed so high vulnerability is warm (yellow/red) and
  resistant components are cold (purple/blue) -- matches the paper's
  "warm/cool" caption convention.
* Projection-kind order is fixed to ``[q, k, v, o, up, gate, down]`` so the
  attention block (q,k,v,o) and MLP block (up,gate,down) read left-to-right.
* If a model is missing a kind (e.g. no ``gate`` for Llama-3.1 vs gated MLP
  in Qwen3), the cell is rendered as light gray rather than dropped, so the
  reader can see architecture-driven gaps separately from low FVS.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KIND_ORDER = ["q", "k", "v", "o", "up", "gate", "down"]
KEY_RE = re.compile(r"^([a-z_]+)\.L(\d+)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fvs-files", nargs="+", required=True,
                   help="One or two FVS JSON paths.")
    p.add_argument("--labels", nargs="+",
                   help="Optional human labels per FVS file (e.g. Qwen3-8B).")
    p.add_argument("--out-pdf", required=True,
                   help="Output PDF path. Parents will be created.")
    p.add_argument("--cmap", default="viridis",
                   help="Matplotlib colormap name (default: viridis).")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    return p.parse_args()


def load_fvs(path: str) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)


def fvs_to_matrix(fvs: dict[str, float]) -> tuple[np.ndarray, list[int], list[str]]:
    """Convert {kind.Llayer: score} into a (num_layers, len(KIND_ORDER)) array.

    Returns the matrix, the list of layer indices used, and the kind list.
    Missing cells are filled with NaN so they render as the masked colour.
    """
    parsed: list[tuple[str, int, float]] = []
    for k, v in fvs.items():
        m = KEY_RE.match(k)
        if not m:
            continue
        parsed.append((m.group(1), int(m.group(2)), float(v)))

    if not parsed:
        raise ValueError("FVS dict is empty or malformed; expected '{kind}.L{layer}' keys")

    layers = sorted({layer for _, layer, _ in parsed})
    layer_idx = {layer: i for i, layer in enumerate(layers)}
    mat = np.full((len(layers), len(KIND_ORDER)), np.nan, dtype=float)
    for kind, layer, val in parsed:
        if kind not in KIND_ORDER:
            continue
        mat[layer_idx[layer], KIND_ORDER.index(kind)] = val
    return mat, layers, KIND_ORDER


def render_panel(ax, mat: np.ndarray, layers: list[int], kinds: list[str],
                  title: str, cmap: str, vmin: float, vmax: float):
    """Render one heatmap panel with NaN-aware masking."""
    masked = np.ma.array(mat, mask=np.isnan(mat))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", interpolation="nearest")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, fontsize=9)
    n = len(layers)
    yticks = list(range(0, n, max(1, n // 8)))
    if (n - 1) not in yticks:
        yticks.append(n - 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(layers[i]) for i in yticks], fontsize=9)
    ax.set_xlabel("projection kind", fontsize=10)
    ax.set_ylabel("layer index", fontsize=10)
    ax.set_title(title, fontsize=11)

    # Light-gray background for masked (architectural-NA) cells.
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#E0E0E0")
    im.set_cmap(cmap_obj)
    return im


def main() -> int:
    args = parse_args()
    fvs_paths = [Path(p) for p in args.fvs_files]
    labels = args.labels or [p.stem for p in fvs_paths]
    if len(labels) != len(fvs_paths):
        raise SystemExit(
            f"--labels has {len(labels)} entries but --fvs-files has {len(fvs_paths)}"
        )

    n_panels = len(fvs_paths)
    fig_w = 3.6 * n_panels + 0.8  # leave room for shared colourbar
    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_w, 5.0), sharey=False, squeeze=False
    )

    last_im = None
    for ax, path, label in zip(axes[0], fvs_paths, labels):
        fvs = load_fvs(str(path))
        mat, layers, kinds = fvs_to_matrix(fvs)
        last_im = render_panel(ax, mat, layers, kinds, label,
                               cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)

    cbar = fig.colorbar(last_im, ax=axes[0].tolist(), shrink=0.85,
                        pad=0.02, fraction=0.04)
    cbar.set_label("FVS (1 = vulnerable, 0 = resistant)", fontsize=9)

    out = Path(args.out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[fig] wrote {out}  ({n_panels} panel(s), {fig.get_size_inches()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
