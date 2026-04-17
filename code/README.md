# EMNLP 2026-001 Code

Reference implementation for *The Forgetting Topology: Mapping Catastrophic Forgetting Across Transformer Components*.

## Model choices (Apr 2026)

| Role | Model | Why |
|---|---|---|
| Primary | **Qwen3-8B** (Apr 2025) | Dense, Apache 2.0, outperforms Qwen2.5-14B. Dual thinking/non-thinking mode; we force non-thinking for deterministic eval. Note: no `-Instruct` suffix — base IS the instruct model. |
| Second | **Llama-3.1-8B-Instruct** | Dense 8B, directly comparable architecture; widely benchmarked. |
| Pilot | **Qwen3-1.7B** | Validated end-to-end on A40 2026-04-16 — load → LoRA rank-16 → SFTTrainer 50 samples → generate → span-F1 in 133s wall. See `experiments/results/smoke_pilot_qwen3_1_7b.json`. |
| Excluded | Llama 4 Scout, Qwen3-MoE | MoE — incompatible with dense-attention causal tracing. Extension = future work. |

## Layout

```
code/
├── forgetting_map/
│   ├── sequential_finetune.py   # sequential fine-tuning loop
│   ├── causal_trace.py          # ROME-style per-component intervention
│   └── fvs.py                   # Forgetting Vulnerability Score aggregation
├── tg_lora/
│   └── config.py                # TG-LoRA target-module selection from FVS
├── evaluation/
│   ├── metrics.py               # F1 (NER), EM/F1 (QA), ROUGE-L (Sum), pass@1 (Code)
│   └── runner.py                # unified scoring over all prior tasks
├── utils/
│   ├── data.py                  # dataset loaders for CoNLL, SQuAD, XSum, CodeAlpaca
│   └── seeds.py                 # reproducibility helpers
└── run_pilot.py                 # single-command end-to-end pilot
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke-test on small model (local, no A40 needed)

```bash
# Pilot runs NER->QA on Qwen3-1.7B, 100 examples per task, to catch bugs
python run_pilot.py --model Qwen/Qwen3-1.7B --tasks ner,qa --subsample 100
```

## Full run (A40 required)

```bash
python run_pilot.py \
  --model Qwen/Qwen3-8B \
  --tasks ner,qa,sum,code \
  --method tg_lora \
  --fvs-path experiments/results/fvs_qwen3_8b.json \
  --seed 42 \
  --out experiments/results/qwen3_8b_tg_lora_s42.json
```

## Expected wall-clock on A40 48GB
- 8B LoRA, 1 epoch, bf16, 10k examples ≈ 55 min
- Per-component causal trace on 8B (sparse 30%) ≈ 40 min
- Full matrix (1 model × 4 stages × 6 methods × 3 seeds) ≈ 78 A40-hrs

## Reproducibility
- All seeds logged
- All hyperparameters in `experiments/configs/*.yaml`
- Results written as JSON with SHA of commit that produced them
