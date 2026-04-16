"""TG-LoRA: Topology-Guided LoRA target-module selection.

Standard LoRA targets a fixed set of module name patterns across *all* layers
and heads. TG-LoRA keeps the same module *kinds* but restricts to the set of
forgetting-resistant (low-FVS) components.

This module returns either:
  (a) a list of module name patterns (if HuggingFace PEFT is used without
      modification), where we compile names like ``model.layers.3.self_attn.q_proj``
      corresponding to low-FVS components; or
  (b) a LoraConfig whose ``target_modules`` is such a list.
"""
from __future__ import annotations
import json
import re
from pathlib import Path


_KIND_TO_MODNAME = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_o": "self_attn.o_proj",
    "mlp_gate": "mlp.gate_proj",
    "mlp_up": "mlp.up_proj",
    "mlp_down": "mlp.down_proj",
}


def _parse_key(key: str) -> dict:
    # key = "attn_q.L12.H3" or "mlp_down.L12"
    m = re.match(r"([a-z_]+)\.L(\d+)(?:\.H(\d+))?", key)
    assert m, f"Malformed FVS key: {key}"
    kind, layer, head = m.group(1), int(m.group(2)), m.group(3)
    return {"kind": kind, "layer": layer, "head": int(head) if head is not None else None}


def tg_lora_target_modules(
    model,
    fvs_path: str,
    tau: float = 0.5,
) -> list[str]:
    """Return the list of target module name patterns for TG-LoRA.

    Parameters
    ----------
    model : the transformer model (used to infer depth & layer name style).
    fvs_path : path to JSON produced by ``forgetting_map.fvs.compute_fvs``.
    tau : fraction of *most resistant* components to keep.  tau=0.5 => bottom
          50% of FVS values are LoRA targets.
    """
    with open(fvs_path) as f:
        fvs: dict[str, float] = json.load(f)
    threshold = sorted(fvs.values())[int(tau * len(fvs)) - 1]
    kept = {k for k, v in fvs.items() if v <= threshold}

    # We currently target at the *projection* granularity (entire layer.proj),
    # because PEFT's target_modules expects name patterns, not head-level
    # slicing. Head-level LoRA would need a custom wrapper. This matches our
    # paper's first-pass experiments; a head-level variant is future work.
    target_names: set[str] = set()
    for key in kept:
        p = _parse_key(key)
        modname = _KIND_TO_MODNAME[p["kind"]]
        target_names.add(f"model.layers.{p['layer']}.{modname}")

    return sorted(target_names)


def fraction_kept(target_names: list[str], model) -> float:
    """Report how many of model.named_modules' projections TG-LoRA kept."""
    n_all = 0
    for n, m in model.named_modules():
        if any(m_end in n for m_end in ("q_proj","k_proj","v_proj","o_proj",
                                          "gate_proj","up_proj","down_proj")):
            n_all += 1
    return len(target_names) / max(1, n_all)
