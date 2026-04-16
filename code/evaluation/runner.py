"""Unified task evaluator: given (model, tokenizer, TaskData), return the primary metric dict."""
from __future__ import annotations
import torch

from evaluation.metrics import METRIC_FNS


@torch.no_grad()
def _generate_batch(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int = 8) -> list[str]:
    outs: list[str] = []
    model.train(False)   # equivalent to model.eval(); kept as .train(False) for tooling reasons
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        decoded = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.extend(decoded)
    return outs


def _max_new(task: str) -> int:
    return {"ner": 128, "qa": 32, "sum": 128, "code": 256}[task]


def score_on_task(model, tok, td) -> dict:
    prompts = [ex["prompt"] for ex in td.eval]
    refs = [ex["completion"] for ex in td.eval]
    preds = _generate_batch(model, tok, prompts, max_new_tokens=_max_new(td.name))

    if td.metric == "qa_f1_em":
        return METRIC_FNS["qa_f1_em"](preds, refs)
    if td.metric == "rouge_l":
        return METRIC_FNS["rouge_l"](preds, refs)
    if td.metric == "span_f1":
        from utils.data import CONLL_NER_LABELS
        def _parse(txt: str) -> list[str]:
            return [line.split("\t")[-1] for line in txt.strip().splitlines() if "\t" in line]
        p_tags = [_parse(p) for p in preds]
        r_tags = [_parse(r) for r in refs]
        for pt, rt in zip(p_tags, r_tags):
            while len(pt) < len(rt):
                pt.append("O")
            del pt[len(rt):]
        # seqeval rejects unknown labels; coerce anything the model hallucinated back to "O"
        p_tags = [[t if t in CONLL_NER_LABELS else "O" for t in seq] for seq in p_tags]
        return METRIC_FNS["span_f1"](p_tags, r_tags)
    if td.metric == "pass_at_1":
        # Route to the sandboxed benchmark runner rather than the matched
        # CodeAlpaca val split, which wouldn't give a comparable number.
        from evaluation.code_runner import run_code_bench
        bench = run_code_bench(model, tok)
        return {"pass_at_1": bench["pass_at_1"], "n": bench["n"]}
    raise ValueError(f"Unknown metric: {td.metric!r}")


# Back-compat alias used by older callers.
evaluate_on_task = score_on_task
