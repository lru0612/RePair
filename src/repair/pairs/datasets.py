"""Training views of accepted turns and rubric subsets."""

from collections import Counter

from repair.harness.context import parse_turn
from repair.io import fingerprint


def chosen_samples(pairs: list[dict]) -> list[dict]:
    return [
        {
            "id": row.get("id") or fingerprint(row),
            "prompt": row["prompt"],
            "completion": row["chosen"],
        }
        for row in pairs
    ]


def successful_turns(trajectories: list[dict], verdicts: list[dict]) -> list[dict]:
    accepted = {
        row["id"]
        for row in verdicts
        if row["correct"] is True and not row.get("judge_error", False)
    }
    samples = []
    for trajectory in trajectories:
        if trajectory["id"] not in accepted:
            continue
        for turn in trajectory["turns"]:
            if turn.get("status") != "ok":
                continue
            try:
                parsed = parse_turn(turn["completion"])
            except (ValueError, TypeError, KeyError):
                continue
            samples.append(
                {
                    "id": f"{trajectory['id']}:{turn['turn']}",
                    "query_id": trajectory["query_id"],
                    "prompt": turn["prompt"],
                    "completion": parsed.render(),
                }
            )
    return samples


def top_k_pairs(
    pairs: list[dict], assignments: list[dict], codebook: list, k: int
) -> tuple[list[dict], list[str]]:
    order = {rubric.id: number for number, rubric in enumerate(codebook)}
    identities = {(a["trajectory_id"], a["turn"], a["rubric_id"]) for a in assignments}
    if len(identities) != len(assignments):
        raise ValueError("Duplicate assignments would distort rubric frequencies")
    counts = Counter(row["rubric_id"] for row in assignments)
    if not set(counts) <= set(order):
        raise ValueError("Assignments reference an unknown rubric")
    selected = sorted(order, key=lambda name: (-counts[name], order[name]))[:k]
    return [pair for pair in pairs if pair["rubric_id"] in selected], selected
