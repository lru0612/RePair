import copy
from pathlib import Path

import pytest

from repair.config import load_experiment
from repair.harness.context import Turn
from repair.models import Reply


@pytest.fixture
def recipe():
    return load_experiment(Path(__file__).parents[1] / "configs/experiments/A3_induced_14b.toml")


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def generate(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, Reply):
            return reply
        return Reply(reply, reply, "", "stop", 10, 8)


@pytest.fixture
def fake_client():
    return FakeClient


@pytest.fixture
def turn_factory():
    def make(tool="search", **args):
        if not args:
            args = {"query": "synthetic clue"}
        return Turn("Choose the next visible lead.", {}, {"tool": tool, "args": args})

    return make


@pytest.fixture
def trajectory(turn_factory):
    raw = turn_factory().render()
    return {
        "id": "q1:0",
        "query_id": "q1",
        "question": "Which fictional city is mentioned?",
        "sample": 0,
        "status": "completed",
        "answer": "Wrongplace",
        "natural_finish": True,
        "turns": [
            {
                "turn": 0,
                "status": "ok",
                "forced_finish": False,
                "prompt": {
                    "system": "Use the tools.",
                    "user": "Visible evidence names Northvale. [doc_a]",
                },
                "completion": raw,
                "reasoning": "Choose the next visible lead.",
                "action": {"tool": "search", "args": {"query": "synthetic clue"}},
                "observation": "No results.",
            }
        ],
    }
