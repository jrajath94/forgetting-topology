"""Sequential fine-tuning loop over (NER -> QA -> Sum -> Code).

Each stage trains on its own task with the method-specific regulariser / data
transform, then evaluates on the current task *and* all prior tasks so we
can compute forgetting afterwards. Per-stage checkpoints and metric rows
are persisted after every stage so a pod termination never costs more than
one stage of work.

Methods:

- ``full_ft``       -- standard fine-tune, no regulariser
- ``lora``          -- LoRA rank-16 on all attn + MLP projections
- ``tg_lora``       -- LoRA restricted to the 50% most forgetting-resistant
                       projections selected by the pre-computed FVS
- ``lora_replay``   -- LoRA + random replay of ``replay_frac`` prior-task
                       examples mixed into the current stage's dataset
- ``ewc``           -- LoRA + Elastic Weight Consolidation quadratic penalty
                       (cumulative online Fisher)
- ``sdft``          -- LoRA + Self-Distillation from the stage-(k-1) model,
                       pure variant: gold completion replaced with teacher's
                       greedy generation
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
import random
import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from utils.data import load_all
from utils.seeds import set_seed
from evaluation.runner import score_on_task
from forgetting_map.ewc import EWCState, EWCTrainer, compute_fisher_diagonal, snapshot_params
from forgetting_map.sdft import build_sdft_dataset


@dataclass
class StageRecord:
    stage_idx: int
    current_task: str
    ckpt_path: str
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Method wiring
# ---------------------------------------------------------------------------
def _apply_method(model, method: str, fvs_path: str | None = None):
    """Attach PEFT adapter if the method requires one. Returns (model, meta)."""
    if method == "full_ft":
        return model, None
    if method in {"lora", "lora_replay", "ewc", "sdft"}:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, cfg), cfg
    if method == "tg_lora":
        from peft import LoraConfig, get_peft_model
        from tg_lora.config import tg_lora_target_modules
        assert fvs_path, "tg_lora requires a pre-computed FVS JSON"
        targets = tg_lora_target_modules(model, fvs_path, tau=0.5)
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=targets, task_type="CAUSAL_LM")
        return get_peft_model(model, cfg), cfg
    raise ValueError(f"Unknown method: {method!r}")


def _build_stage_dataset(
    method: str,
    stage_idx: int,
    tname: str,
    td,
    model,
    tok,
    all_data,
    tasks,
    replay_frac: float = 0.1,
    seed: int = 42,
) -> Dataset:
    """Apply method-specific transforms to the stage's train dataset.

    - lora_replay: mix in a random slice of examples from every prior task
    - sdft: replace completion with teacher (current model) generation
    - otherwise: return the gold dataset unchanged
    """
    train = td.train

    if method == "lora_replay" and stage_idx > 0:
        rng = random.Random(seed + stage_idx)
        replay_pieces = [train]
        for prior in tasks[:stage_idx]:
            prior_train = all_data[prior].train
            n_keep = max(1, int(replay_frac * len(prior_train)))
            idxs = rng.sample(range(len(prior_train)), min(n_keep, len(prior_train)))
            replay_pieces.append(prior_train.select(idxs))
        train = concatenate_datasets(replay_pieces).shuffle(seed=seed + stage_idx)

    if method == "sdft" and stage_idx > 0:
        # At the start of stage k (pre-training), ``model`` holds theta_{k-1}.
        # Use it as the teacher. For stage 0 we have no teacher so we fall
        # back to gold -- this matches the SDFT paper.
        train = build_sdft_dataset(
            task_name=tname,
            teacher_model=model,
            tokenizer=tok,
            train_ds=train,
            cache_dir=os.environ.get("SDFT_CACHE", "/workspace/sdft_cache"),
            teacher_ckpt_tag=f"stage{stage_idx - 1}",
            max_new_tokens=128,
            batch_size=4,
        )

    return train


def _make_trainer(
    method: str,
    model,
    tok,
    train_ds: Dataset,
    cfg: SFTConfig,
    ewc_state: EWCState | None,
    ewc_lambda: float,
):
    kwargs = dict(model=model, args=cfg, train_dataset=train_ds, processing_class=tok)
    if method == "ewc":
        return EWCTrainer(**kwargs, ewc_state=ewc_state, ewc_lambda=ewc_lambda)
    return SFTTrainer(**kwargs)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_sequence(
    base_model: str,
    method: str,
    tasks: list[str],
    seed: int,
    out_dir: str,
    subsample: int | None = None,
    fvs_path: str | None = None,
    ewc_lambda: float = 500.0,
    replay_frac: float = 0.1,
) -> tuple[object, list[StageRecord]]:
    set_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto", token=hf_token
    )
    model, _meta = _apply_method(model, method, fvs_path=fvs_path)

    all_data = load_all(subsample=subsample)
    records: list[StageRecord] = []
    ewc_state = EWCState() if method == "ewc" else None

    for k, tname in enumerate(tasks):
        td = all_data[tname]
        train_ds = _build_stage_dataset(
            method, k, tname, td, model, tok, all_data, tasks,
            replay_frac=replay_frac, seed=seed,
        )

        cfg = SFTConfig(
            output_dir=str(out / f"stage_{k}_{tname}"),
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5 if method == "full_ft" else 2e-4,
            num_train_epochs=3,
            warmup_ratio=0.03,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=1,
            max_length=1024,
            report_to=[],
            seed=seed,
        )

        trainer = _make_trainer(method, model, tok, train_ds, cfg, ewc_state, ewc_lambda)
        trainer.train()

        # After-stage bookkeeping for EWC
        if method == "ewc":
            fisher = compute_fisher_diagonal(
                model, trainer.get_train_dataloader(), max_samples=1000
            )
            theta_star = snapshot_params(model)
            ewc_state.consolidate(fisher, theta_star, gamma=1.0)

        ckpt_path = str(out / f"stage_{k}_{tname}/final")
        trainer.save_model(ckpt_path)

        # Score on current + all prior tasks
        rec = StageRecord(stage_idx=k, current_task=tname, ckpt_path=ckpt_path, metrics={})
        for prior in tasks[: k + 1]:
            rec.metrics[prior] = score_on_task(model, tok, all_data[prior])
        records.append(rec)

        with (out / "stage_records.json").open("w") as f:
            json.dump([asdict(r) for r in records], f, indent=2)

    return model, records
