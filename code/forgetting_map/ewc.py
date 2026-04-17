"""Elastic Weight Consolidation for sequential fine-tuning.

EWC adds a quadratic penalty to the loss that pulls parameters back toward
the values they held at the end of each previous task, weighted by the
diagonal Fisher information of that task. Formally:

    L_total = L_task + (lambda / 2) * sum_k sum_i F_k[i] * (theta[i] - theta_k*[i])^2

Where F_k is the diagonal Fisher for task k, and theta_k* are the parameters
at the end of task k. Reference: Kirkpatrick et al. 2017 (EWC).

This module provides:

  * ``compute_fisher_diagonal`` -- empirical Fisher from a task's dataloader
  * ``snapshot_params``         -- clone the trainable-parameter state
  * ``EWCState``                -- accumulator that fuses multiple prior tasks
  * ``EWCTrainer``              -- a thin ``trl.SFTTrainer`` subclass that
                                  adds the EWC penalty to ``compute_loss``

We accumulate Fishers across tasks (F_sum = F_1 + F_2 + ...) and keep a
single theta_star, set to the most-recent end-of-task snapshot. This is the
standard "online EWC" variant; it avoids memory growth linear in the number
of past tasks without materially hurting forgetting control in practice.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import torch
from trl import SFTTrainer


# ---------------------------------------------------------------------------
# Fisher + parameter snapshots
# ---------------------------------------------------------------------------
def snapshot_params(model) -> dict[str, torch.Tensor]:
    """Clone only the trainable parameters; EWC penalty applies to them."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@torch.enable_grad()
def compute_fisher_diagonal(
    model,
    dataloader: Iterable,
    max_samples: int = 1000,
) -> dict[str, torch.Tensor]:
    """Empirical Fisher diagonal over ``max_samples`` sequences.

    We use the *empirical* Fisher (gradient of the true-label NLL squared)
    rather than the true Fisher (expected over model samples), which is
    standard for EWC on classification / language modelling tasks and
    roughly 10x cheaper. See Kunstner et al. 2019 for discussion.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.train()
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}

    n_seen = 0
    for batch in dataloader:
        if n_seen >= max_samples:
            break
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        out = model(**batch)
        if getattr(out, "loss", None) is None:
            continue
        out.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        bsz = batch["input_ids"].shape[0] if "input_ids" in batch else 1
        n_seen += bsz

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.train(False)

    denom = max(1, n_seen)
    return {n: f / denom for n, f in fisher.items()}


# ---------------------------------------------------------------------------
# Online EWC state
# ---------------------------------------------------------------------------
@dataclass
class EWCState:
    """Online EWC: a single (cumulative Fisher, most-recent theta_star) pair."""
    fisher: dict[str, torch.Tensor] = field(default_factory=dict)
    theta_star: dict[str, torch.Tensor] = field(default_factory=dict)

    def consolidate(
        self,
        new_fisher: dict[str, torch.Tensor],
        new_theta_star: dict[str, torch.Tensor],
        gamma: float = 1.0,
    ) -> None:
        """Fuse a just-finished task's (F, theta*) into the accumulator.

        gamma < 1 gives exponential decay over tasks ("online EWC with
        forgetting factor"), gamma == 1 is vanilla online EWC.
        """
        for name, f_new in new_fisher.items():
            if name in self.fisher:
                self.fisher[name] = gamma * self.fisher[name] + f_new
            else:
                self.fisher[name] = f_new.clone()
        # theta_star always points to the MOST RECENT snapshot -- gradients
        # in the next task push theta back toward this anchor, weighted by
        # the (accumulated) Fisher.
        self.theta_star = {n: p.clone() for n, p in new_theta_star.items()}

    def is_empty(self) -> bool:
        return not self.fisher


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------
class EWCTrainer(SFTTrainer):
    """SFTTrainer that adds an EWC penalty to the supervised loss.

    Pass ``ewc_state`` (an ``EWCState`` instance carried across stages) and
    ``ewc_lambda`` (scaling coefficient -- standard values: 100-5000 for
    LoRA; higher for full fine-tuning).
    """

    def __init__(self, *args, ewc_state: EWCState | None = None,
                 ewc_lambda: float = 500.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._ewc_state = ewc_state or EWCState()
        self._ewc_lambda = ewc_lambda

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = super().compute_loss(model, inputs, return_outputs=True,
                                    num_items_in_batch=num_items_in_batch)
        loss, outputs = base
        if not self._ewc_state.is_empty():
            penalty = torch.zeros((), device=loss.device, dtype=loss.dtype)
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                F = self._ewc_state.fisher.get(n)
                theta = self._ewc_state.theta_star.get(n)
                if F is None or theta is None:
                    continue
                # F and theta were saved for the same name; their shapes must match
                if F.shape == p.shape and theta.shape == p.shape:
                    penalty = penalty + (F * (p - theta).pow(2)).sum()
            loss = loss + 0.5 * self._ewc_lambda * penalty
        return (loss, outputs) if return_outputs else loss
