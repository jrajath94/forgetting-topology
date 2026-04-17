"""Self-Distillation Fine-Tuning (SDFT).

Reference: Shi et al. 2024, "Self-distillation bridges distribution gap in
language model fine-tuning" (ACL 2024).

Motivation. A fine-tune that optimises solely against task-k gold targets
pushes the model away from the distribution it previously matched, which is
precisely what causes forgetting. SDFT replaces the gold completion with a
*self-generated* completion produced by the stage-(k-1) model and trains
against that. The student therefore never sees a gradient that contradicts
its own prior beliefs; it only sees gradients that teach it to keep
producing what it already produces, but in the direction of task k.

This module exposes two variants:

  * ``build_sdft_dataset``    -- pure self-distillation: completion becomes
                                the teacher's greedy generation.
  * ``build_mixed_dataset``   -- alpha * gold + (1-alpha) * teacher output,
                                implemented as a dataset that interleaves
                                both kinds of examples at the requested ratio.

Both preserve the ``prompt`` / ``completion`` schema consumed by
``trl.SFTTrainer``. We cache teacher generations to disk so reruns (or
different student sizes) don't repeat the expensive greedy pass.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import torch
from datasets import Dataset
from tqdm.auto import tqdm


def _cache_key(teacher_ckpt: str, task_name: str, n_examples: int, max_new_tokens: int) -> str:
    digest = hashlib.sha1(
        f"{teacher_ckpt}|{task_name}|{n_examples}|{max_new_tokens}".encode()
    ).hexdigest()[:12]
    return f"sdft_refs_{task_name}_{digest}.jsonl"


@torch.no_grad()
def _batched_generate(teacher, tok, prompts: list[str], max_new_tokens: int,
                      batch_size: int = 4) -> list[str]:
    teacher.train(False)
    out: list[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="sdft-gen"):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(teacher.device)
        gen = teacher.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        decoded = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
        out.extend(decoded)
    return out


def build_sdft_dataset(
    task_name: str,
    teacher_model,
    tokenizer,
    train_ds: Dataset,
    cache_dir: str = "/workspace/sdft_cache",
    teacher_ckpt_tag: str = "stage-k-1",
    max_new_tokens: int = 128,
    batch_size: int = 4,
) -> Dataset:
    """Replace each example's ``completion`` with the teacher's greedy generation.

    Teacher outputs are cached as JSONL at ``cache_dir/<cache_key>`` so a
    rerun with the same teacher + task + size returns instantly.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) / _cache_key(
        teacher_ckpt_tag, task_name, len(train_ds), max_new_tokens
    )

    prompts = [ex["prompt"] for ex in train_ds]

    if cache_path.exists():
        print(f"[sdft] loading cached teacher refs from {cache_path}")
        refs = [json.loads(line)["ref"] for line in cache_path.read_text().splitlines()]
        assert len(refs) == len(prompts), (
            f"cache size mismatch ({len(refs)} vs {len(prompts)}); "
            f"delete {cache_path} to regenerate"
        )
    else:
        print(f"[sdft] generating {len(prompts)} teacher refs -> {cache_path}")
        refs = _batched_generate(teacher_model, tokenizer, prompts,
                                 max_new_tokens=max_new_tokens, batch_size=batch_size)
        with cache_path.open("w") as f:
            for p, r in zip(prompts, refs):
                f.write(json.dumps({"prompt": p[:200], "ref": r}) + "\n")

    def swap_completion(ex, idx):
        ex = dict(ex)
        ex["completion"] = refs[idx]
        ex["source"] = "teacher"
        return ex

    return train_ds.map(swap_completion, with_indices=True)


def build_mixed_dataset(
    task_name: str,
    teacher_model,
    tokenizer,
    train_ds: Dataset,
    alpha: float = 0.5,
    cache_dir: str = "/workspace/sdft_cache",
    teacher_ckpt_tag: str = "stage-k-1",
    max_new_tokens: int = 128,
    batch_size: int = 4,
    seed: int = 42,
) -> Dataset:
    """Mix gold-completion examples (prob alpha) with teacher-completion
    examples (prob 1-alpha)."""
    distilled = build_sdft_dataset(
        task_name, teacher_model, tokenizer, train_ds,
        cache_dir=cache_dir, teacher_ckpt_tag=teacher_ckpt_tag,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )

    import random
    rng = random.Random(seed)
    keep_gold = [rng.random() < alpha for _ in range(len(train_ds))]

    mixed_rows = []
    for i, keep in enumerate(keep_gold):
        mixed_rows.append(dict(train_ds[i]) if keep else dict(distilled[i]))
        mixed_rows[-1]["source"] = "gold" if keep else "teacher"
    return Dataset.from_list(mixed_rows)
