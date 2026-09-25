"""Token-averaged cDPO, chosen NLL, and policy-to-reference KL."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _positions(mask: torch.Tensor):
    counts = mask[:, 1:].sum(-1)
    if torch.any(counts == 0):
        raise ValueError("Every completion must contain supervised tokens")
    indices = torch.arange(mask.shape[0], device=mask.device)[:, None].expand_as(mask[:, 1:])[
        mask[:, 1:]
    ]
    return indices, counts


def _means(values, indices, counts):
    sums = torch.zeros(len(counts), device=values.device, dtype=values.dtype)
    return sums.scatter_add(0, indices, values) / counts


def _selected_logps(logits, targets, chunk_size=128):
    def logp(block, labels):
        return -F.cross_entropy(block.float(), labels, reduction="none")

    blocks = []
    for start in range(0, len(logits), chunk_size):
        block, labels = logits[start : start + chunk_size], targets[start : start + chunk_size]
        blocks.append(
            checkpoint(logp, block, labels, use_reentrant=False)
            if block.requires_grad
            else logp(block, labels)
        )
    return torch.cat(blocks)


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    indices, counts = _positions(mask)
    selected = mask[:, 1:]
    values = _selected_logps(logits[:, :-1][selected], labels[:, 1:][selected])
    return _means(values, indices, counts)


def reference_kl(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    selected = mask[:, 1:]
    policy = policy_logits[:, :-1][selected]
    reference = reference_logits[:, :-1][selected].detach()
    indices, counts = _positions(mask)
    return _means(_selected_kl(policy, reference, chunk_size), indices, counts)


def _selected_kl(policy, reference, chunk_size):
    if chunk_size <= 0:
        raise ValueError("KL chunk size must be positive")

    def token_kl(p, q):
        log_p, log_q = F.log_softmax(p.float(), -1), F.log_softmax(q.float(), -1)
        return (log_p.exp() * (log_p - log_q)).sum(-1)

    blocks = []
    for start in range(0, len(policy), chunk_size):
        p, q = policy[start : start + chunk_size], reference[start : start + chunk_size]
        blocks.append(
            checkpoint(token_kl, p, q, use_reentrant=False) if p.requires_grad else token_kl(p, q)
        )
    return torch.cat(blocks)


def preference_loss(
    policy_logits, reference_logits, labels, mask, settings: dict
) -> tuple[torch.Tensor, dict]:
    indices, counts = _positions(mask)
    selected = mask[:, 1:]
    targets = labels[:, 1:][selected]
    selected_policy = policy_logits[:, :-1][selected]
    selected_reference = reference_logits[:, :-1][selected].detach()
    policy = _means(_selected_logps(selected_policy, targets), indices, counts)
    reference = _means(_selected_logps(selected_reference, targets), indices, counts)
    if len(policy) % 2:
        raise ValueError("A preference batch needs equally many chosen and rejected completions")
    size = len(policy) // 2
    margin = settings["beta"] * (
        policy[:size] - policy[size:] - reference[:size] + reference[size:]
    )
    epsilon = settings["label_smoothing"]
    dpo = -(1 - epsilon) * F.logsigmoid(margin) - epsilon * F.logsigmoid(-margin)
    nll = -policy[:size]
    kl = _means(
        _selected_kl(selected_policy, selected_reference, settings["kl_chunk_tokens"]),
        indices,
        counts,
    )
    paired_kl = (kl[:size] + kl[size:]) / 2
    loss = dpo + settings["nll_weight"] * nll + settings["kl_weight"] * paired_kl
    return loss.mean(), {
        "cdpo": dpo.detach().mean(),
        "chosen_nll": nll.detach().mean(),
        "reference_kl": paired_kl.detach().mean(),
    }
