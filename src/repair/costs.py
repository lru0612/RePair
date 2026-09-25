"""Token accounting and explicitly parameterized rollout-budget estimates."""

from __future__ import annotations

from collections import defaultdict


def summarize_usage(rows: list[dict], training_summaries: list[dict] = ()) -> dict:
    stages = defaultdict(
        lambda: {"calls": 0, "prefill_tokens": 0, "generated_tokens": 0, "missing_usage_calls": 0}
    )
    rollouts, candidates = defaultdict(lambda: [0, 0]), defaultdict(int)
    excluded_reader_calls = 0
    missing_rollout_usage = missing_candidate_usage = False
    for row in rows:
        if row["role"] == "reader":
            excluded_reader_calls += 1
            continue
        if row["stage"] not in ("collection", "induction", "reassignment", "repair", "validation"):
            continue
        stage = "failure_pool" if row["stage"] == "collection" else row["stage"]
        group = stages[stage]
        group["calls"] += 1
        if row.get("prompt_tokens") is None or row.get("completion_tokens") is None:
            group["missing_usage_calls"] += 1
            missing_rollout_usage |= row["stage"] == "collection" and row["role"] == "policy"
            missing_candidate_usage |= row["role"] == "repairer"
            continue
        group["prefill_tokens"] += row["prompt_tokens"]
        group["generated_tokens"] += row["completion_tokens"]
        if row["stage"] == "collection" and row["role"] == "policy":
            identity = row["identity"].rsplit(":", 1)[0]
            rollouts[identity][0] += row["prompt_tokens"]
            rollouts[identity][1] += row["completion_tokens"]
        if row["role"] == "repairer":
            # Candidate IDs include the assignment and sampling attempt.
            candidates[row["identity"]] += row["completion_tokens"]

    def mean(values):
        return sum(values) / len(values) if values else None

    return {
        "stages": dict(stages),
        "reader_calls_excluded": excluded_reader_calls,
        "mean_rollout_prefill_tokens": None
        if missing_rollout_usage
        else mean([value[0] for value in rollouts.values()]),
        "mean_rollout_generated_tokens": None
        if missing_rollout_usage
        else mean([value[1] for value in rollouts.values()]),
        "mean_candidate_generated_tokens": None
        if missing_candidate_usage
        else mean(list(candidates.values())),
        "accelerator_hours": sum(row["accelerator_hours"] for row in training_summaries),
        "complete_usage": not any(group["missing_usage_calls"] for group in stages.values()),
    }


def estimate_budget(inputs: dict) -> dict:
    required = ("rollouts", "mean_prefill_tokens", "mean_generated_tokens", "assumptions")
    if any(key not in inputs for key in required) or not str(inputs["assumptions"]).strip():
        raise ValueError(
            "A budget estimate needs rollouts, mean token counts, and written assumptions"
        )
    if any(inputs[key] < 0 for key in required[:-1]):
        raise ValueError("Budget inputs cannot be negative")
    return {
        "estimated": True,
        "assumptions": inputs["assumptions"],
        "rollouts": inputs["rollouts"],
        "prefill_tokens": inputs["rollouts"] * inputs["mean_prefill_tokens"],
        "generated_tokens": inputs["rollouts"] * inputs["mean_generated_tokens"],
        "includes_training_compute": False,
    }
