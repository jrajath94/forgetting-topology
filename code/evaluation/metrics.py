"""Task metrics. All return a dict with a primary score under the same key.

We keep these as pure functions over (predictions, references) lists so the
harness can call them identically across tasks.
"""
from __future__ import annotations
import re
import string
from collections import Counter
from typing import Sequence


def _normalise(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------- QA (SQuAD-style EM / F1) --------
def _token_f1(pred: str, gold: str) -> float:
    pt, gt = _normalise(pred).split(), _normalise(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    p = n_same / len(pt)
    r = n_same / len(gt)
    return 2 * p * r / (p + r)


def qa_f1_em(preds: Sequence[str], refs: Sequence[str]) -> dict:
    em = sum(int(_normalise(p) == _normalise(r)) for p, r in zip(preds, refs)) / len(preds)
    f1 = sum(_token_f1(p, r) for p, r in zip(preds, refs)) / len(preds)
    return {"qa_f1_em": f1, "em": em, "f1": f1}


# -------- NER span-F1 (seqeval wrapper) --------
def span_f1(preds: Sequence[list[str]], refs: Sequence[list[str]]) -> dict:
    from seqeval.metrics import f1_score, precision_score, recall_score
    return {
        "span_f1": f1_score(refs, preds),
        "precision": precision_score(refs, preds),
        "recall": recall_score(refs, preds),
    }


# -------- ROUGE-L --------
def rouge_l(preds: Sequence[str], refs: Sequence[str]) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return {"rouge_l": sum(scores) / len(scores)}


# -------- pass@1 for HumanEval --------
def pass_at_1(completions: Sequence[str], test_cases: Sequence[str], entry_points: Sequence[str]) -> dict:
    """Executes each completion against its test_case and returns pass@1.

    SAFETY: executes arbitrary code. Only call inside a sandbox / ephemeral
    pod. Uses the ``human-eval`` package's check_correctness utility when
    available.
    """
    try:
        from human_eval.execution import check_correctness  # type: ignore
    except ImportError:
        # Fallback: direct exec in subprocess with timeout. Kept minimal here;
        # users should install human-eval for production runs.
        raise RuntimeError("Install human-eval for pass@1 execution.")

    results = []
    for completion, test, entry in zip(completions, test_cases, entry_points):
        problem = {"prompt": "", "test": test, "entry_point": entry}
        r = check_correctness(problem, completion, timeout=10.0)
        results.append(int(r.get("passed", False)))
    return {"pass_at_1": sum(results) / len(results)}


METRIC_FNS = {
    "span_f1": span_f1,
    "qa_f1_em": qa_f1_em,
    "rouge_l": rouge_l,
    "pass_at_1": pass_at_1,
}
