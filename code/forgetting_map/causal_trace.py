"""ROME-style causal tracing adapted for *forgetting* rather than knowledge localisation.

For each transformer component c (an attention or MLP projection at layer l),
we construct a counterfactual: the model as it exists *now* but with c's
post-training changes undone. We then measure

    Delta_c = Acc_prior_task(theta_restored_c) - Acc_prior_task(theta_post)

A large positive Delta_c means "reverting c recovers prior performance",
i.e., c is a forgetting site.

Model-type handling
-------------------
We support two training modes and pick the correct intervention per model:

* **Full fine-tuning** (``mode="full_ft"``): the learned change lives in the
  base projection weight. To undo it we overwrite ``proj.weight.data`` with
  the pre-training snapshot.

* **LoRA / TG-LoRA** (``mode="lora"``): the learned change lives in the
  ``lora_A @ lora_B`` delta added at forward time. To undo it we zero the
  LoRA adapter for this one projection for the duration of the eval call,
  then restore it. This is attribution option (b) from the v1 design note:
  it asks "how much does this component's *entire* training trajectory hurt
  prior tasks?", which is the right question for the FVS contribution.

We operate at **projection granularity** (one q_proj / k_proj / etc. per layer),
not head-level, so the FVS bins exactly match what ``tg_lora.config`` can
target. A head-level extension is future work.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import torch


# ---------------------------------------------------------------------------
# Component descriptor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentId:
    layer: int
    kind: str        # "attn_q" | "attn_k" | "attn_v" | "attn_o" | "mlp_gate" | "mlp_up" | "mlp_down"

    def to_dict(self) -> dict:
        return {"layer": self.layer, "kind": self.kind}

    def key(self) -> str:
        return f"{self.kind}.L{self.layer}"


# ---------------------------------------------------------------------------
# Model-type introspection
# ---------------------------------------------------------------------------
def _base_decoder(model):
    """Return the object whose ``.layers`` is the list of transformer blocks.

    * Raw HF causal LM: ``model.model``
    * PEFT-wrapped:     ``model.base_model.model.model``
    """
    cur = model
    # Unwrap PEFT (``PeftModelForCausalLM``) if present
    if hasattr(cur, "base_model") and hasattr(cur.base_model, "model"):
        cur = cur.base_model.model
    # Inner HF causal-LM wrapper
    if hasattr(cur, "model") and hasattr(cur.model, "layers"):
        return cur.model
    if hasattr(cur, "layers"):
        return cur
    raise RuntimeError(f"Cannot find transformer layers on {type(model).__name__}")


def _get_projection(model, layer: int, kind: str):
    """Return the nn.Module for a given (layer, kind). Works for raw + PEFT models."""
    decoder = _base_decoder(model)
    block = decoder.layers[layer]
    attn = block.self_attn
    mlp = block.mlp
    mapping = {
        "attn_q": attn.q_proj, "attn_k": attn.k_proj,
        "attn_v": attn.v_proj, "attn_o": attn.o_proj,
        "mlp_gate": mlp.gate_proj, "mlp_up": mlp.up_proj, "mlp_down": mlp.down_proj,
    }
    return mapping[kind]


def _is_lora_wrapped(proj) -> bool:
    """A PEFT lora.Linear exposes ``lora_A`` / ``lora_B`` as ModuleDicts."""
    return hasattr(proj, "lora_A") and hasattr(proj, "lora_B")


def n_layers_of(model) -> int:
    return len(_base_decoder(model).layers)


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------
@torch.no_grad()
def patch_component_full_ft(model, pre_weights: dict, comp: ComponentId):
    """Swap the base weight of ``comp`` to its pre-stage snapshot.

    Returns the original post-stage tensor so the caller can restore.
    """
    proj = _get_projection(model, comp.layer, comp.kind)
    pre = pre_weights[(comp.layer, comp.kind)]
    orig = proj.weight.data.clone()
    proj.weight.data.copy_(pre)
    return orig


@torch.no_grad()
def restore_component_full_ft(model, comp: ComponentId, orig: torch.Tensor) -> None:
    proj = _get_projection(model, comp.layer, comp.kind)
    proj.weight.data.copy_(orig)


@torch.no_grad()
def patch_component_lora_zero(model, comp: ComponentId):
    """Zero the LoRA adapter(s) at ``comp`` so its delta contributes nothing
    at forward time. Returns a snapshot for restoration.
    """
    proj = _get_projection(model, comp.layer, comp.kind)
    if not _is_lora_wrapped(proj):
        raise RuntimeError(
            f"{comp.key()} is not LoRA-wrapped; use patch_component_full_ft instead"
        )
    snap: dict[str, torch.Tensor] = {}
    for adapter_name, module in proj.lora_A.items():
        snap[("A", adapter_name)] = module.weight.data.clone()
        module.weight.data.zero_()
    for adapter_name, module in proj.lora_B.items():
        snap[("B", adapter_name)] = module.weight.data.clone()
        module.weight.data.zero_()
    return snap


@torch.no_grad()
def restore_component_lora(model, comp: ComponentId, snap: dict) -> None:
    proj = _get_projection(model, comp.layer, comp.kind)
    for adapter_name, module in proj.lora_A.items():
        module.weight.data.copy_(snap[("A", adapter_name)])
    for adapter_name, module in proj.lora_B.items():
        module.weight.data.copy_(snap[("B", adapter_name)])


# ---------------------------------------------------------------------------
# Topology enumeration
# ---------------------------------------------------------------------------
PROJ_KINDS = ("attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down")


def enumerate_components(n_layers: int,
                         sample_frac: float = 1.0,
                         kinds: tuple = PROJ_KINDS,
                         seed: int = 42) -> list[ComponentId]:
    import random
    rng = random.Random(seed)
    comps = [ComponentId(layer=l, kind=k) for l in range(n_layers) for k in kinds]
    if sample_frac < 1.0:
        n_keep = max(1, int(sample_frac * len(comps)))
        comps = rng.sample(comps, n_keep)
    return comps


# ---------------------------------------------------------------------------
# Pre-weight snapshots (for full_ft mode)
# ---------------------------------------------------------------------------
def snapshot_weights(model, kinds=PROJ_KINDS) -> dict:
    """Snapshot base projection weights keyed by (layer, kind). Used in full_ft mode.
    For LoRA mode we don't need this -- we just zero the adapter at trace time.
    """
    n_layers = n_layers_of(model)
    out = {}
    for l in range(n_layers):
        for k in kinds:
            proj = _get_projection(model, l, k)
            out[(l, k)] = proj.weight.data.detach().clone()
    return out


# ---------------------------------------------------------------------------
# Main tracing loop
# ---------------------------------------------------------------------------
def causal_trace(
    model_post,
    mode: str,  # "full_ft" | "lora"
    prior_task_evaluator,
    components: Iterable[ComponentId],
    out_path: str,
    pre_weights: dict | None = None,
) -> dict:
    """For each component c, apply the intervention, measure prior-task acc, restore.

    Parameters
    ----------
    model_post : the post-stage model. Mutated temporarily during the loop,
        restored after each iteration.
    mode : "full_ft" restores base weight from ``pre_weights``; "lora" zeroes
        the LoRA adapter for the component.
    prior_task_evaluator : callable(model) -> float. Returns the metric value
        for the prior task on an eval subset.
    components : iterable of ComponentId to trace.
    out_path : JSON output file, written incrementally every 50 components.
    pre_weights : dict[(layer,kind) -> tensor] from ``snapshot_weights``. Only
        needed in ``mode="full_ft"``.
    """
    if mode == "full_ft" and pre_weights is None:
        raise ValueError("mode='full_ft' requires pre_weights from snapshot_weights")

    baseline = prior_task_evaluator(model_post)
    rows = []
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    for i, comp in enumerate(components):
        if mode == "full_ft":
            snap = patch_component_full_ft(model_post, pre_weights, comp)
            try:
                acc = prior_task_evaluator(model_post)
            finally:
                restore_component_full_ft(model_post, comp, snap)
        elif mode == "lora":
            try:
                snap = patch_component_lora_zero(model_post, comp)
            except RuntimeError as e:
                # Component has no LoRA adapter (e.g., TG-LoRA skipped this
                # projection). Record NaN so the FVS aggregation knows to
                # treat it as "frozen, cannot cause forgetting by update".
                rows.append({**comp.to_dict(), "delta": float("nan"),
                             "acc_patched": float("nan"),
                             "note": "no-lora"})
                continue
            try:
                acc = prior_task_evaluator(model_post)
            finally:
                restore_component_lora(model_post, comp, snap)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        rows.append({**comp.to_dict(), "delta": acc - baseline, "acc_patched": acc})

        if i % 50 == 0:
            with Path(out_path).open("w") as f:
                json.dump({"baseline": baseline, "mode": mode, "rows": rows}, f, indent=2)

    with Path(out_path).open("w") as f:
        json.dump({"baseline": baseline, "mode": mode, "rows": rows}, f, indent=2)
    return {"baseline": baseline, "mode": mode, "rows": rows}
