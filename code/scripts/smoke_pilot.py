"""Self-contained smoke pilot: Qwen3-1.7B + CoNLL-NER + LoRA rank-16.

Validated on RunPod A40 (pod ee71k0aupgwzus, EU-SE-1) on 2026-04-16.
133s wall-clock, $0.11 spend. Loss 1.034 -> 0.451 over 13 steps.

Kept as a reproducible end-to-end reference. Run it before spinning up
the main matrix to verify the environment + HF access is healthy:

    HF_TOKEN=<token> python smoke_pilot.py
"""
import os, json, time, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen3-1.7B"
N_TRAIN = 50
N_EVAL = 20
OUT_DIR = os.environ.get("SMOKE_OUT_DIR", "/tmp/smoke_pilot_out")
os.makedirs(OUT_DIR, exist_ok=True)

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def main():
    t0 = time.time()
    print("[1/6] loading tokenizer + model (bf16)")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ["HF_TOKEN"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0", token=os.environ["HF_TOKEN"]
    )
    print(f"  [{time.time()-t0:.1f}s] n_params={sum(p.numel() for p in model.parameters())/1e9:.2f}B "
          f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB")

    print("[2/6] attaching LoRA rank-16 to all attn + MLP projections")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[3/6] loading CoNLL-NER subsample")
    ds = load_dataset("conll2003")

    def to_sft(example):
        toks = example["tokens"]
        tags = [NER_LABELS[t] for t in example["ner_tags"]]
        prompt = "Tag named entities (one per line as token\\tTAG):\n" + " ".join(toks) + "\n"
        completion = "\n".join(f"{t}\t{tag}" for t, tag in zip(toks, tags))
        return {"prompt": prompt, "completion": completion}

    train_ds = ds["train"].select(range(N_TRAIN)).map(to_sft, remove_columns=ds["train"].column_names)
    eval_ds = ds["validation"].select(range(N_EVAL)).map(to_sft, remove_columns=ds["validation"].column_names)

    print("[4/6] fine-tuning 1 epoch on 50 samples")
    cfg = SFTConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=5,
        save_strategy="no",
        max_length=512,
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tok)
    trainer.train()
    print(f"  [{time.time()-t0:.1f}s] training complete")

    print("[5/6] evaluating span-F1 on val subset")
    model.train(False)
    from seqeval.metrics import f1_score, precision_score, recall_score

    pred_tags, gold_tags = [], []
    for ex in eval_ds:
        inp = tok(ex["prompt"], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        p = [line.split("\t")[-1] for line in gen.strip().splitlines() if "\t" in line]
        g = [line.split("\t")[-1] for line in ex["completion"].splitlines() if "\t" in line]
        while len(p) < len(g):
            p.append("O")
        p = p[:len(g)]
        p = [t if t in NER_LABELS else "O" for t in p]
        pred_tags.append(p)
        gold_tags.append(g)

    metrics = {
        "span_f1": float(f1_score(gold_tags, pred_tags)),
        "precision": float(precision_score(gold_tags, pred_tags)),
        "recall": float(recall_score(gold_tags, pred_tags)),
        "n_eval": len(eval_ds),
        "n_train": N_TRAIN,
        "model": MODEL_ID,
        "wall_seconds": time.time() - t0,
    }
    with open(f"{OUT_DIR}/pilot_result.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[6/6] done")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
