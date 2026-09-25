"""Rubric validation and failure-pool selection."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from repair.harness.context import parse_turn


@dataclass(frozen=True)
class Rubric:
    id: str
    trigger: str
    hint: str
    criterion: str

    @classmethod
    def from_dict(cls, value: dict) -> Rubric:
        fields = {}
        for key in ("id", "trigger", "hint", "criterion"):
            text = value.get(key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Rubric {key} must be nonempty text")
            fields[key] = text.strip()
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", fields["id"]):
            raise ValueError("Rubric IDs must use lowercase kebab-case")
        return cls(**fields)

    def as_dict(self) -> dict:
        return asdict(self)


def read_codebook(rows: list[dict]) -> list[Rubric]:
    rubrics = [Rubric.from_dict(row) for row in rows]
    if not rubrics or len({r.id for r in rubrics}) != len(rubrics):
        raise ValueError("A codebook needs unique rubric IDs")
    return rubrics


def eligible_turns(trajectory: dict) -> list[dict]:
    eligible = []
    for turn in trajectory["turns"]:
        if turn.get("status") != "ok" or turn.get("forced_finish", False):
            continue
        try:
            parse_turn(turn["completion"])
        except (ValueError, TypeError, KeyError):
            continue
        eligible.append(turn)
    return eligible


def failure_pool(trajectories: list[dict], verdicts: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in verdicts}
    if len(by_id) != len(verdicts):
        raise ValueError("Duplicate judge records")
    return [
        trajectory
        for trajectory in trajectories
        if trajectory["status"] == "completed"
        and by_id.get(trajectory["id"], {}).get("correct") is False
        and not by_id[trajectory["id"]].get("judge_error", False)
        and not any(turn.get("status") == "refused" for turn in trajectory["turns"])
        and eligible_turns(trajectory)
    ]


def trajectory_view(trajectory: dict) -> dict:
    blocks = []
    for turn in trajectory["turns"]:
        blocks.append(
            f"[step {turn['turn']}]\n{turn['completion']}\n"
            f"Tool result:\n{turn.get('observation', '')}"
        )
    return {
        "question": trajectory["question"],
        "trajectory": "\n\n".join(blocks),
        "final_answer": trajectory["answer"],
        "valid_steps": ", ".join(str(t["turn"]) for t in eligible_turns(trajectory)),
        "agent_view": trajectory["turns"][-1]["prompt"]["user"],
    }


_STOP_WORDS = set(
    """
a an and are as at be between but by can could did do does during for from give had has have
how i identify if in into is it its list me my name not of on or over per please provide s t
that the their there these this those to under was were what when where which who whom whose
why will with you your find tell describe explain answer question following based according
""".split()
)
_QUOTED = re.compile(r'"([^"]{2,})"')
_NUMBER = re.compile(r"\b\d{3,}\b")
_IDENTIFIER = re.compile(r"\b(?=\w*[a-z])(?=\w*\d)\w{6,}\b", re.I)


def question_anchors(question: str) -> set[str]:
    words = re.findall(r"[A-Za-z][\w'-]*", question)
    result = {
        w.lower() for w in words if len(w) >= 3 and w[0].isupper() and w.lower() not in _STOP_WORDS
    }
    result.update(m.group(1).lower() for m in _QUOTED.finditer(question))
    result.update(_NUMBER.findall(question))
    return result


def specificity_error(text: str, anchors: set[str]) -> str | None:
    if _QUOTED.search(text):
        return "quoted_phrase"
    if _NUMBER.search(text):
        return "specific_number"
    if _IDENTIFIER.search(text):
        return "identifier"
    if any(re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", text.lower()) for anchor in anchors):
        return "question_anchor"
    return None
