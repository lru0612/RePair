"""Bounded policy state and the assistant-turn format."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from repair.io import stable_json


@dataclass
class Turn:
    reasoning: str
    state: dict
    action: dict

    def render(self) -> str:
        return (
            f"<think>\n{self.reasoning.strip()}\n</think>\n"
            f"<state>\n{stable_json(self.state)}\n</state>\n"
            f"<action>\n{stable_json(self.action)}\n</action>"
        )


def parse_turn(raw: str) -> Turn:
    state_matches = list(re.finditer(r"<state>\s*(.*?)\s*</state>", raw, re.S))
    action_matches = list(re.finditer(r"<action>\s*(.*?)\s*</action>", raw, re.S))
    if len(state_matches) != 1 or len(action_matches) != 1:
        raise ValueError("A turn needs exactly one state and one action")
    state_match, action_match = state_matches[0], action_matches[0]
    if state_match.end() > action_match.start():
        raise ValueError("The state must precede the action")
    state = json.loads(state_match.group(1))
    action = json.loads(action_match.group(1))
    if not isinstance(state, dict) or not isinstance(action, dict):
        raise ValueError("State and action must be objects")
    for key in ("satisfy", "add_constraints", "drop_constraints"):
        if key in state and not isinstance(state[key], list):
            raise ValueError(f"state.{key} must be a list")
    if "notes" in state and state["notes"] is not None and not isinstance(state["notes"], str):
        raise ValueError("state.notes must be a string")
    tool = action.get("tool")
    args = action.get("args")
    required = {
        "search": ("query",),
        "get_document": ("docid", "goal"),
        "finish": ("exact_answer",),
    }
    if tool not in required or not isinstance(args, dict):
        raise ValueError("Unknown tool or invalid action arguments")
    if any(not isinstance(args.get(k), str) or not args[k].strip() for k in required[tool]):
        raise ValueError("Missing action argument")
    prefix = raw[: state_match.start()].strip()
    if prefix.startswith("<think>"):
        if not prefix.endswith("</think>"):
            raise ValueError("Unclosed reasoning block")
        prefix = prefix[len("<think>") : -len("</think>")].strip()
    elif prefix.endswith("</think>"):
        # Some chat templates place the opening tag in the generation prefix.
        prefix = prefix[: -len("</think>")].strip()
    if raw[action_match.end() :].strip():
        raise ValueError("Unexpected content after the action")
    return Turn(prefix, state, action)


def clip(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()[:limit]


@dataclass
class WorkingContext:
    question: str
    output_format: str
    max_evidence: int = 24
    evidence_chars: int = 300
    notes_chars: int = 400
    history_chars: int = 250
    constraints: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    seen_docids: set[str] = field(default_factory=set)
    notes: str = ""
    next_cid: int = 1

    def update(self, delta: dict) -> None:
        drops = {str(cid) for cid in delta.get("drop_constraints", [])}
        self.constraints = [c for c in self.constraints if c["cid"] not in drops]
        existing = {c["text"].strip().casefold() for c in self.constraints}
        for entry in delta.get("add_constraints", []):
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            if not text or text.casefold() in existing:
                continue
            cid = str(entry.get("cid") or f"c{self.next_cid}")
            while any(c["cid"] == cid for c in self.constraints):
                self.next_cid += 1
                cid = f"c{self.next_cid}"
            self.next_cid += 1
            self.constraints.append({"cid": cid, "text": text, "satisfied": False, "support": []})
            existing.add(text.casefold())
        by_id = {c["cid"]: c for c in self.constraints}
        for entry in delta.get("satisfy", []):
            if not isinstance(entry, dict) or entry.get("cid") not in by_id:
                continue
            supports = entry.get("support") or []
            if not isinstance(supports, list):
                continue
            support = [str(d) for d in supports if str(d) in self.seen_docids]
            if support:
                constraint = by_id[entry["cid"]]
                constraint["satisfied"] = True
                constraint["support"] = sorted(set(constraint["support"]) | set(support))
        if delta.get("notes") is not None:
            self.notes = delta["notes"][: self.notes_chars]

    def note_search(self, action: dict, results: list[dict]) -> None:
        self.seen_docids.update(str(hit["docid"]) for hit in results)
        for hit in results:
            if hit.get("parent_docid"):
                self.seen_docids.add(str(hit["parent_docid"]))
        self.record_action(action, stable_json(results))

    def note_document(self, action: dict, facts: list[str], observation: str) -> None:
        docid = action["args"]["docid"]
        self.seen_docids.add(docid)
        self.record_action(action, observation)
        for fact in facts:
            item = {"docid": docid, "fact": clip(fact, self.evidence_chars)}
            if item["fact"] and item not in self.evidence:
                self.evidence.append(item)
        cited = {d for c in self.constraints if c["satisfied"] for d in c["support"]}
        while len(self.evidence) > self.max_evidence:
            index = next(
                (i for i, item in enumerate(self.evidence) if item["docid"] not in cited), 0
            )
            self.evidence.pop(index)

    def record_action(self, action: dict, result: str) -> None:
        self.history.append(
            {
                "turn": len(self.history),
                "tool": action["tool"],
                "args": action["args"],
                "result": clip(result, self.history_chars),
            }
        )

    def render(self, last_observation: str = "") -> str:
        satisfied, unsatisfied = [], []
        for c in self.constraints:
            line = f"- [{c['cid']}] {c['text']}"
            if c["satisfied"]:
                line += " (support: " + " ".join(f"[{d}]" for d in c["support"]) + ")"
                satisfied.append(line)
            else:
                unsatisfied.append(line)
        history = [
            f"- t{a['turn']} {a['tool']} {stable_json(a['args'])} -> {a['result']}"
            for a in self.history
        ]
        evidence = [f"- [{item['docid']}] {item['fact']}" for item in self.evidence]
        fields = [
            ("GOAL", self.question),
            ("REQUIRED OUTPUT FORMAT (only when you FINISH)", self.output_format),
            ("SATISFIED CONSTRAINTS", "\n".join(satisfied) or "(none yet)"),
            ("UNSATISFIED CONSTRAINTS", "\n".join(unsatisfied) or "(none)"),
            ("ACTION HISTORY", "\n".join(history) or "(none yet)"),
            ("EVIDENCE", "\n".join(evidence) or "(none yet)"),
            ("NOTES", self.notes or "(none)"),
            ("LAST OBSERVATION", last_observation or "(none)"),
        ]
        return "\n\n".join(f"# {name}\n{body}" for name, body in fields) + (
            "\n\nDecide the next step. Emit a <state> constraint update then exactly one <action>."
        )

    def snapshot(self) -> dict[str, Any]:
        value = asdict(self)
        value["seen_docids"] = sorted(self.seen_docids)
        return value
