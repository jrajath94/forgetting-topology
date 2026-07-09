# Forgetting Topology

> Mapping catastrophic forgetting across transformer components, then using
> that map to make parameter-efficient fine-tuning forgetting-aware.

This repository contains the source code, experimental orchestrator, and
LaTeX manuscripts for the paper:

**The Forgetting Topology: Mapping Catastrophic Forgetting Across
Transformer Components to Enable Targeted Parameter-Efficient Fine-Tuning.**

This manuscript is prepared as an independent research contribution.
Both manuscript variants live side-by-side in this repo and share
section sources.

---

## Status

This is a research-in-progress repository. The code is feature-complete
for the planned experimental matrix, the LaTeX manuscripts compile
cleanly, and the only thing not yet in the repo is the populated results
table. **Per the project's standing integrity rule, no result number is
filled in until the corresponding experiment has actually run.** A custom
LaTeX macro, `\pend{key}`, renders unfilled cells as a visible red
`[PENDING: key]` marker in the PDF so a paper cannot ship with silent
gaps.

| Component               | State                                        |
| ----------------------- | -------------------------------------------- |
| Method (FVS, TG-LoRA)   | Implemented and pilot-validated              |
| Forgetting measurement  | Causal-tracing intervention, PEFT-aware      |
| Baselines               | Full FT, LoRA, EWC, SDFT, LoRA+Replay, DoRA, VeRA, LOFIT, random-target — all implemented |
| Evaluation              | NER (CoNLL-2003), QA (SQuAD), Sum (XSum), Code (HumanEval pass@1 sandbox) |
| Manuscripts             | Independent research variants — both compile, both have unfilled `\pend{}` markers awaiting real results |
| Main experimental matrix | Not yet executed; estimated 36 cells × ~8 GPU-hours = ~288 A40-hours |
| Reproducibility         | All hyperparameters, asset licenses, and seeds documented (`papers/*/latex/sections/A_reproducibility.tex`) |

---

## What the paper is about, in three paragraphs

When a language model is sequentially fine-tuned on multiple tasks
(say, NER, then QA, then summarisation, then code), it tends to forget
earlier tasks --- the well-known *catastrophic forgetting* problem. The
standard PEFT response is to apply LoRA-style adapters everywhere and
hope the low-rank constraint preserves prior behaviour. It does not,
uniformly. Some components in the network forget very little under
sequential adaptation; others forget aggressively.

This paper asks: **which components forget, exactly?** We define a
per-component **Forgetting Vulnerability Score (FVS)** by combining
ROME-style causal-tracing interventions across every (prior task,
current stage) pair. Each component receives a scalar in `[0, 1]`
indicating how much updating it during a downstream stage would cost
on prior-task performance. We compute this map for two open-weight 8B
models (Qwen3-8B, Llama-3.1-8B), check that it transfers between the
two via Spearman's ρ, and then act on it.

The actuation step is **Topology-Guided LoRA (TG-LoRA)**: instead of
applying LoRA to all attention and MLP projections at every layer,
we restrict the adapter target set to the bottom-50% of FVS components
--- the ones whose update is causally least implicated in forgetting
prior tasks --- and report current-task performance, prior-task
forgetting, and backward-transfer against eight baselines.

---

## Repository layout

```
forgetting-topology/
├── code/                              # All Python research code
│   ├── forgetting_map/                # Causal tracing + FVS + sequential FT
│   │   ├── causal_trace.py            # PEFT-aware ROME-style intervention
│   │   ├── fvs.py                     # Aggregation into [0,1] component scores
│   │   ├── ewc.py                     # Online Fisher + EWC regulariser
│   │   ├── sdft.py                    # Self-distillation FT (Shi et al. 2024)
│   │   └── sequential_finetune.py     # TRL SFTTrainer wrapper for the 4-task seq.
│   ├── tg_lora/                       # Topology-guided LoRA target selection
│   │   └── config.py                  # Builds peft.LoraConfig from FVS dict
│   ├── evaluation/                    # NER F1, QA F1/EM, ROUGE-L, pass@1
│   │   ├── runner.py                  # Routes per-task evaluation
│   │   ├── metrics.py
│   │   └── code_runner.py             # HumanEval subprocess sandbox
│   ├── utils/                         # Data loaders, seed control
│   │   ├── data.py                    # CoNLL/SQuAD/XSum/HumanEval -> prompt+completion
│   │   └── seeds.py
│   ├── scripts/
│   │   ├── run_matrix.py              # End-to-end orchestrator (resumable)
│   │   ├── smoke_pilot.py             # Single-cell smoke test (2 hr, 1 GPU)
│   │   ├── make_fig_topology.py       # FVS heatmap renderer
│   │   └── fill_results.py            # Substitutes \pend{key} -> measured value
│   └── requirements.txt               # Pinned dependency versions
├── experiments/
│   ├── configs/
│   │   └── qwen3_8b_main.yaml         # Matrix definition
│   └── results/
│       ├── paper_fill_template.json   # Source of truth for all \pend{} keys
│       └── smoke_pilot_qwen3_1_7b.json # Real Apr-16 pilot validation result
├── papers/
│   ├── neurips-2026/latex/            # Single-column NeurIPS variant + Paper Checklist
│   └── emnlp-2026/latex/              # Two-column ACL/ARR variant
├── docs/                              # (reserved for tutorials / extended notes)
├── LICENSE                            # Apache-2.0
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Quick start

### Environment

```bash
git clone https://github.com/<owner>/forgetting-topology.git
cd forgetting-topology

python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

The pinned stack is `transformers>=5.5.0`, `peft>=0.19.0`, `trl>=1.1.0`,
`datasets>=4.0.0`, and `accelerate>=1.1.1`. Earlier `transformers==4.46`
does **not** support the `qwen3` model type.

### Smoke test (single A40, ~2 hours, ~$1 on a commodity cloud)

This validates that the end-to-end pipeline works on your hardware
before you commit to the full matrix.

```bash
export HF_TOKEN=<your token>
python code/scripts/smoke_pilot.py \
    --model Qwen/Qwen3-1.7B \
    --task ner \
    --max-steps 50 \
    --output-dir experiments/results
```

A passing smoke run produces a JSON in `experiments/results/`
matching the schema of `smoke_pilot_qwen3_1_7b.json`.

### The full matrix

```bash
python code/scripts/run_matrix.py \
    --config experiments/configs/qwen3_8b_main.yaml \
    --out-dir experiments/results/qwen3_8b \
    --resume
```

`--resume` is essential on commodity clouds where pods can be evicted
mid-run; the orchestrator skips any cell that already has a complete
result JSON on disk.

### Filling the manuscript

Once the matrix completes, aggregate the per-cell JSONs into
`experiments/results/paper_fill.json`, then:

```bash
python code/scripts/fill_results.py \
    --results experiments/results/paper_fill.json \
    --latex-dir papers/neurips-2026/latex \
    --output-dir papers/neurips-2026/latex_filled
cd papers/neurips-2026/latex_filled && latexmk -pdf main.tex
```

The script reports any `\pend{}` keys still unfilled and exits
non-zero in `--strict` mode, so the manuscript cannot ship with silent
gaps.

---

## Reproducibility

Every hyperparameter the paper depends on is recorded in
`papers/<venue>/latex/sections/A_reproducibility.tex` (Tables of
hyperparameters and asset licenses). The matrix uses three seeds
(`{42, 1337, 2024}`), reports mean ± standard deviation, and uses a
paired t-test for the headline TG-LoRA-vs-LoRA forgetting comparison.
Cross-family FVS transferability is reported as Spearman's ρ.

The code is deterministic up to GPU non-determinism in cuDNN; we set
`torch.use_deterministic_algorithms(True)` where compatible and report
the seed in every output JSON.

---

## Citation

```bibtex
@misc{anon2026forgettingtopology,
  title     = {The Forgetting Topology: Mapping Catastrophic Forgetting
               Across Transformer Components to Enable Targeted
               Parameter-Efficient Fine-Tuning},
  author    = {Anonymous},
  year      = {2026},
  note      = {Independent research manuscript}
}
```

---

## License

Code in this repository is released under the
[Apache License 2.0](./LICENSE). The LaTeX style files
(`acl.sty`, `acl_natbib.bst`, `neurips_2026.sty`) are redistributed
under their respective upstream licenses --- they are vendored from
the official ACL Rolling Review and NeurIPS 2026 distributions and
are not modified. Datasets are not redistributed; loading is delegated
to the HuggingFace Hub at the pinned revisions in
`code/requirements.txt`.

---

## Acknowledgements

We thank the maintainers of the
HuggingFace `transformers` / `peft` / `trl` / `datasets` libraries,
the authors of EWC, ROME, SDFT, LOFIT, DoRA, and VeRA for foundational
work that this paper builds on, and the operators of the open-weight
Qwen3 and Llama-3.1 model releases for making this study possible.
