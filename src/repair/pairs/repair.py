"""Candidate repair, local validation, and clean-prompt preference pairs."""

from __future__ import annotations

import re
import unicodedata

from repair.harness.context import Turn, parse_turn
from repair.io import derive_seed, fingerprint, stable_json
from repair.models import ModelError, parse_json_reply
from repair.pairs.prompts import FILTER_SYSTEM, REPAIR_GUIDANCE_TEMPLATE
from repair.pairs.schema import Rubric

_SCAFFOLD = re.compile(
    r"(?:as instructed|i was told|gold answer|ground.?truth|"
    r"rubric|repair guidance|system prompt|success.criteria)",
    re.I,
)


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return " ".join(
        re.sub(
            r"[^\w\s]", " ", "".join(c for c in normalized if not unicodedata.combining(c))
        ).split()
    )


def grounded(answer: str, prompt: str) -> bool:
    words = [word for word in re.findall(r"\w+", answer.casefold()) if len(word) >= 4]
    hits = sum(word in prompt.casefold() for word in words)
    return bool(words) and hits >= 1 and hits * 2 >= len(words)


def candidate_error(
    candidate: Turn, original: Turn, raw: str, prompt: dict, reference: str, limit: int
) -> str | None:
    if _SCAFFOLD.search(raw):
        return "scaffold_echo"
    if len(candidate.reasoning) > limit:
        return "reasoning_too_long"
    action = candidate.action
    visible = prompt["system"] + "\n" + prompt["user"]
    if action["tool"] == "finish":
        answer = action["args"]["exact_answer"]
        if normalize_answer(answer) != normalize_answer(reference):
            return "answer_mismatch"
        if not grounded(answer, visible):
            return "ungrounded_finish"
    else:
        gold = normalize_answer(reference)
        if len(gold) >= 4 and gold in normalize_answer(candidate.render()):
            return "answer_leak"
        old = original.action
        if action["tool"] == old["tool"] and action["args"] == old["args"]:
            return "identical_action"
        references = []
        if action["tool"] == "get_document":
            references.append(action["args"]["docid"])
        for update in candidate.state.get("satisfy", []):
            if isinstance(update, dict):
                references.extend(update.get("support") or [])
        if any(str(docid) not in visible for docid in references):
            return "unseen_document_id"
    return None


def both_yes(verdict: dict) -> bool:
    def yes(value) -> bool:
        return value is True or (type(value) is int and value == 1) or value == "yes"

    return (
        isinstance(verdict, dict)
        and yes(verdict.get("repairs_per_rubric"))
        and yes(verdict.get("natural"))
    )


def repair_assignment(
    trajectory: dict,
    assignment: dict,
    rubric: Rubric | None,
    *,
    repairer,
    validator,
    reference: str,
    config: dict,
) -> tuple[dict | None, list[dict]]:
    source = next(turn for turn in trajectory["turns"] if turn["turn"] == assignment["turn"])
    original = parse_turn(source["completion"])
    prompt = source["prompt"]
    system = prompt["system"]
    if rubric is not None:
        system += REPAIR_GUIDANCE_TEMPLATE.format(
            situation=f"\nWhat went wrong here: {rubric.trigger}\n", hint=rubric.hint
        )
    settings = config["repair"]
    decode = dict(settings["decode"])
    if config["experiment"]["repairer"] == "learner":
        rollout = config["rollout"]
        decode = {
            "thinking": True,
            "temperature": rollout["temperature"],
            "top_p": rollout["top_p"],
            "repetition_penalty": rollout["repetition_penalty"],
            "max_tokens": rollout["policy_max_tokens"],
        }
    identity = fingerprint(assignment)
    attempts = []
    for attempt in range(settings["candidates"]):
        try:
            reply = repairer.generate(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt["user"]},
                ],
                settings=decode,
                stage="repair",
                role="repairer",
                identity=f"{identity}:{attempt}",
                seed=derive_seed(config["seed"], identity, attempt),
            )
            if reply.finish_reason == "length":
                raise ValueError("truncated")
            candidate = parse_turn(reply.text)
            reason = candidate_error(
                candidate, original, reply.text, prompt, reference, settings["reasoning_chars"]
            )
            verdict = None
            if reason is None and rubric is not None:
                data = parse_json_reply(
                    validator.generate(
                        [
                            {"role": "system", "content": FILTER_SYSTEM},
                            {
                                "role": "user",
                                "content": stable_json(
                                    {
                                        "input": prompt,
                                        "original": source["completion"],
                                        "candidate": reply.text,
                                        "criterion": rubric.criterion,
                                    }
                                ),
                            },
                        ],
                        settings=settings["validator_decode"],
                        stage="validation",
                        role="validator",
                        identity=f"{identity}:{attempt}",
                        seed=derive_seed(config["seed"], identity, attempt, "validate"),
                    )
                )
                verdict = data
                if not both_yes(data):
                    reason = "validator_rejected"
            attempts.append(
                {
                    "assignment": identity,
                    "attempt": attempt,
                    "rejection": reason,
                    "verdict": verdict,
                }
            )
            if reason is None:
                return {
                    "id": identity,
                    "trajectory_id": trajectory["id"],
                    "query_id": trajectory["query_id"],
                    "turn": source["turn"],
                    "rubric_id": rubric.id if rubric else None,
                    "prompt": dict(prompt),
                    "chosen": candidate.render(),
                    "rejected": source["completion"],
                }, attempts
        except (ModelError, ValueError, TypeError, KeyError) as error:
            attempts.append(
                {
                    "assignment": identity,
                    "attempt": attempt,
                    "rejection": type(error).__name__,
                }
            )
    return None, attempts
