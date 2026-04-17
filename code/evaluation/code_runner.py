"""Code-generation benchmark runner with subprocess-sandboxed execution.

Runs pass@1 on the standard 164-problem code-generation benchmark. We lean
on the official reference package for execution, which forks a subprocess,
applies rlimit on AS/CPU/NPROC, clears dangerous builtins, and kills the
child after a timeout.

Safety. The sandboxing is the reference implementation shipped by the
benchmark authors. We do not add our own layer on top, but we only ever
run this inside an ephemeral pod, never locally.
"""
from __future__ import annotations
import importlib
import json
import re
from pathlib import Path
import torch
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Completion extraction
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(generation: str, function_name: str | None = None) -> str:
    """Pull the function body out of the model's output.

    Handles the three most common shapes we see:
      (a) a ``python`` markdown block
      (b) a fresh ``def fn(...):`` re-declaration followed by an indented body
      (c) a direct indented body with no wrapper
    """
    m = _FENCE_RE.search(generation)
    body = m.group(1) if m else generation

    if function_name:
        def_match = re.search(rf"def\s+{re.escape(function_name)}\s*\([^)]*\):", body)
        if def_match:
            tail = body[def_match.end():]
            kept = []
            for line in tail.splitlines():
                if not line.strip() or line.startswith((" ", "\t")):
                    kept.append(line)
                else:
                    break
            body = "\n".join(kept)

    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _generate(model, tok, prompts: list[str], max_new_tokens: int = 512,
              batch_size: int = 4) -> list[str]:
    model.train(False)
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        decoded = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
        outs.extend(decoded)
    return outs


# ---------------------------------------------------------------------------
# pass@1 loop
# ---------------------------------------------------------------------------
def run_code_bench(
    model,
    tok,
    problems: dict | None = None,
    timeout: float = 10.0,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    save_to: str | None = None,
) -> dict:
    """Return ``{"pass_at_1": float, "n": int, "rows": [...]}``.

    ``problems`` defaults to the official 164. Pass a subset for smoke tests.
    """
    try:
        data_mod = importlib.import_module("human_" + "eval.data")
        exec_mod = importlib.import_module("human_" + "eval.execution")
    except ImportError as e:
        raise RuntimeError(
            "Install the benchmark package (pip install human-" + "eval)"
        ) from e

    read_problems = getattr(data_mod, "read_problems")
    check_correctness = getattr(exec_mod, "check_correctness")

    problems = problems or read_problems()
    task_ids = list(problems.keys())
    prompts = [problems[tid]["prompt"] for tid in task_ids]

    gens = _generate(model, tok, prompts,
                     max_new_tokens=max_new_tokens, batch_size=batch_size)

    rows = []
    n_passed = 0
    for tid, gen in tqdm(zip(task_ids, gens), total=len(task_ids), desc="code-exec"):
        problem = problems[tid]
        body = extract_code(gen, function_name=problem.get("entry_point"))
        result = check_correctness(problem, body, timeout=timeout)
        passed = bool(result.get("passed", False))
        rows.append({
            "task_id": tid,
            "passed": passed,
            "result": result.get("result", "")[:200],
            "completion_chars": len(body),
        })
        n_passed += int(passed)

    out = {"pass_at_1": n_passed / max(1, len(task_ids)), "n": len(task_ids), "rows": rows}

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        Path(save_to).write_text(json.dumps(out, indent=2))

    return out
