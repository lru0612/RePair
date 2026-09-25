"""Final-answer grading and complete-sample evaluation summaries."""

from __future__ import annotations

import unicodedata
from collections import Counter

from repair.io import derive_seed
from repair.models import ModelError, parse_json_reply

JUDGE_PROMPT = """Decide whether the candidate answer is correct for the question and reference.
It must mean the same thing and identify the same answer. Ignore case, whitespace, diacritics,
nonsemantic punctuation, and common spelling or transliteration variants. Additional details must
be correct and consistent. Reject a different entity, ambiguity, or a substantive disagreement.
Allow a small numerical discrepancy. Judge only the final answer, not a reasoning trace.
Return JSON: {"correct": true or false}."""


def load_questions(rows: list[dict], *, expected: int | None = None) -> list[dict]:
    seen = set()
    for row in rows:
        if not all(
            isinstance(row.get(k), str) and row[k].strip() for k in ("id", "question", "answer")
        ):
            raise ValueError("Each question needs nonempty id, question, and answer strings")
        if row["id"] in seen:
            raise ValueError("Duplicate question identifier")
        seen.add(row["id"])
    if expected is not None and len(rows) != expected:
        raise ValueError(f"Expected {expected} questions, received {len(rows)}")
    return rows


def check_split_overlap(splits: dict[str, list[dict]]) -> None:
    ids, texts = {}, {}
    for split, rows in splits.items():
        for row in rows:
            normalized = " ".join(unicodedata.normalize("NFKC", row["question"]).casefold().split())
            if row["id"] in ids and ids[row["id"]] != split:
                raise ValueError("A question identifier appears in more than one split")
            if normalized in texts and texts[normalized] != split:
                raise ValueError("A question appears in more than one split")
            ids[row["id"]], texts[normalized] = split, split


def grade(trajectory: dict, reference: str, judge, *, seed: int, stage: str) -> dict:
    row = {"id": trajectory["id"], "correct": False, "judge_error": False}
    if trajectory["status"] != "completed" or not trajectory.get("answer"):
        return row
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {trajectory['question']}\n"
                f"Reference: {reference}\nCandidate: {trajectory['answer']}"
            ),
        },
    ]
    try:
        data = parse_json_reply(
            judge.generate(
                messages,
                settings={"temperature": 0.0, "top_p": 1.0},
                stage=stage,
                role="judge",
                identity=trajectory["id"],
                seed=derive_seed(seed, trajectory["id"], "judge"),
            )
        )
        if not isinstance(data, dict) or type(data.get("correct")) is not bool:
            raise ValueError("Invalid judge verdict")
        row["correct"] = data["correct"]
    except (ModelError, ValueError):
        row["judge_error"] = True
    return row


def summarize(
    questions: list[dict], trajectories: list[dict], verdicts: list[dict], samples: int
) -> dict:
    expected = {f"{q['id']}:{s}" for q in questions for s in range(samples)}
    by_id = {row["id"]: row for row in trajectories}
    judges = {row["id"]: row for row in verdicts}
    if len(by_id) != len(trajectories) or len(judges) != len(verdicts):
        raise ValueError("Duplicate rollout or judge records")
    if not set(by_id) <= expected or not set(judges) <= expected:
        raise ValueError("Unexpected evaluation sample")
    correct = {
        key: bool(
            key in by_id
            and by_id[key]["status"] == "completed"
            and judges.get(key, {}).get("correct") is True
            and not judges.get(key, {}).get("judge_error", False)
        )
        for key in expected
    }
    valid = [row for row in trajectories if row["status"] == "completed"]

    def mean(values):
        return sum(values) / len(values) if values else None

    tools = [Counter(t.get("action", {}).get("tool") for t in row["turns"]) for row in valid]
    reasoning = [len(t.get("reasoning", "")) for row in valid for t in row["turns"]]
    return {
        "runs_expected": len(expected),
        "runs_present": len(trajectories),
        "missing_runs": len(expected - set(by_id)),
        "acc": sum(correct.values()) / len(expected) if expected else None,
        "pass_at_3": (
            sum(any(correct[f"{q['id']}:{i}"] for i in range(samples)) for q in questions)
            / len(questions)
            if questions
            else None
        ),
        "behavior_runs": len(valid),
        "mean_turns": mean([len(row["turns"]) for row in valid]),
        "mean_search_calls": mean([count["search"] for count in tools]),
        "mean_document_calls": mean([count["get_document"] for count in tools]),
        "mean_reasoning_chars_per_turn": mean(reasoning),
        "natural_finish_rate": mean([int(row["natural_finish"]) for row in valid]),
        "judge_errors": sum(row.get("judge_error", False) for row in verdicts),
        "missing_verdicts": len(expected - set(judges)),
        "status_counts": dict(Counter(row["status"] for row in trajectories)),
    }
