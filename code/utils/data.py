"""Dataset loaders for the NER -> QA -> Sum -> Code sequence.

All loaders return a TaskData with Dataset columns:
  - ``prompt``: input prompt (no target)
  - ``completion``: gold continuation (loss computed on these tokens only by TRL SFTTrainer)
  - ``task``: task tag for bookkeeping

Schema change (2026-04-16, post-pilot): switched from ``input_text/target_text``
to TRL's ``prompt/completion`` schema so ``SFTTrainer`` auto-masks the prompt
from the supervised loss. Also switched CoNLL NER labels from ints to their
IOB2 string form so ``seqeval`` accepts them without pre-processing.
"""
from __future__ import annotations
from dataclasses import dataclass
from datasets import load_dataset, Dataset


# Official CoNLL-2003 IOB2 label mapping (index -> string). Datasets library
# exposes these via ``ds["train"].features["ner_tags"].feature.names`` — we
# duplicate here so downstream code doesn't need to hold a feature handle.
CONLL_NER_LABELS = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC",
]


@dataclass
class TaskData:
    name: str
    train: Dataset
    eval: Dataset
    metric: str  # "span_f1" | "qa_f1_em" | "rouge_l" | "pass_at_1"


def _conll_to_tagged(example):
    tokens = example["tokens"]
    tags = [CONLL_NER_LABELS[t] for t in example["ner_tags"]]
    prompt = (
        "Tag named entities (one per line as token\\tTAG):\n"
        + " ".join(tokens) + "\n"
    )
    completion = "\n".join(f"{t}\t{tag}" for t, tag in zip(tokens, tags))
    return {"prompt": prompt, "completion": completion, "task": "ner"}


def load_conll(subsample: int | None = None) -> TaskData:
    # We use the parquet mirror ``tomaarsen/conll2003`` rather than the
    # canonical ``conll2003`` repo, because the canonical one ships a Python
    # loader script that datasets>=4.0 refuses to execute. The mirror has
    # identical splits (14041 train / 3250 val / 3453 test), identical
    # ``ner_tags`` ClassLabel mapping (0-8 = O/B-PER/.../I-MISC), and
    # identical token-level content. It only adds two extra columns
    # (document_id, sentence_id) which we discard in the map.
    ds = load_dataset("tomaarsen/conll2003")
    train = ds["train"].map(_conll_to_tagged, remove_columns=ds["train"].column_names)
    val = ds["validation"].map(_conll_to_tagged, remove_columns=ds["validation"].column_names)
    if subsample:
        train = train.select(range(min(subsample, len(train))))
        val = val.select(range(min(subsample, len(val))))
    return TaskData("ner", train, val, "span_f1")


def _squad_to_qa(example):
    ans = example["answers"]["text"][0] if example["answers"]["text"] else ""
    prompt = f"Question: {example['question']}\nContext: {example['context']}\nAnswer:"
    return {"prompt": prompt, "completion": " " + ans, "task": "qa"}


def load_squad(subsample: int | None = None) -> TaskData:
    ds = load_dataset("squad")
    train = ds["train"].map(_squad_to_qa, remove_columns=ds["train"].column_names)
    val = ds["validation"].map(_squad_to_qa, remove_columns=ds["validation"].column_names)
    if subsample:
        train = train.select(range(min(subsample, len(train))))
        val = val.select(range(min(subsample, len(val))))
    return TaskData("qa", train, val, "qa_f1_em")


def _xsum_to_summary(example):
    return {
        "prompt": "Summarise this article in one sentence:\n" + example["document"] + "\nSummary:",
        "completion": " " + example["summary"],
        "task": "sum",
    }


def load_xsum(subsample: int | None = None) -> TaskData:
    ds = load_dataset("EdinburghNLP/xsum")
    train = ds["train"].map(_xsum_to_summary, remove_columns=ds["train"].column_names)
    val = ds["validation"].map(_xsum_to_summary, remove_columns=ds["validation"].column_names)
    if subsample:
        train = train.select(range(min(subsample, len(train))))
        val = val.select(range(min(subsample, len(val))))
    return TaskData("sum", train, val, "rouge_l")


def _codealpaca_to_code(example):
    prompt = example["instruction"]
    if example.get("input"):
        prompt += "\n" + example["input"]
    return {"prompt": prompt, "completion": example["output"], "task": "code"}


def load_codealpaca(subsample: int | None = None) -> TaskData:
    # Train on CodeAlpaca; HumanEval is evaluated separately in the runner.
    ds = load_dataset("sahil2801/CodeAlpaca-20k")
    split = ds["train"].train_test_split(test_size=0.05, seed=42)
    train = split["train"].map(_codealpaca_to_code, remove_columns=split["train"].column_names)
    val = split["test"].map(_codealpaca_to_code, remove_columns=split["test"].column_names)
    if subsample:
        train = train.select(range(min(subsample, len(train))))
        val = val.select(range(min(subsample, len(val))))
    return TaskData("code", train, val, "pass_at_1")


SEQUENCE = ["ner", "qa", "sum", "code"]

LOADERS = {
    "ner": load_conll,
    "qa": load_squad,
    "sum": load_xsum,
    "code": load_codealpaca,
}


def load_all(subsample: int | None = None) -> dict[str, TaskData]:
    return {name: LOADERS[name](subsample=subsample) for name in SEQUENCE}
