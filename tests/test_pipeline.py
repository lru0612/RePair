import copy
import json
from pathlib import Path

from repair.config import load_experiment
from repair.experiment import Experiment
from repair.harness.context import Turn
from repair.harness.retrieval import build_index
from repair.io import read_jsonl, write_json, write_jsonl


class CharacterTokenizer:
    name_or_path = "synthetic-character-tokenizer"
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = "\n".join(message["content"] for message in messages) + "\nassistant\n"
        return self.encode(text) if tokenize else text

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def test_collection_to_preference_and_sft_views(tmp_path, fake_client):
    config = load_experiment(
        Path(__file__).parents[1] / "configs/experiments/A2_handwritten_14b.toml"
    )
    config["data"].update(warm_start_questions=1, training_questions=1, test_questions=1)
    paths = {"output_root": str(tmp_path / "outputs"), "index": str(tmp_path / "index")}
    for split in ("warm_start", "training", "test"):
        path = tmp_path / f"{split}.jsonl"
        write_jsonl(
            path, [{"id": split, "question": f"Synthetic {split} question", "answer": "Northvale"}]
        )
        paths[f"{split}_questions"] = str(path)
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(
        corpus, [{"docid": "doc_a", "title": "Example", "text": "Northvale river synthetic clue."}]
    )
    tokenizer = CharacterTokenizer()
    build_index(corpus, tmp_path / "index", tokenizer, config["retrieval"])
    names = ["query-churn", "search-without-read", "over-specified-query"]
    codebook = tmp_path / "codebook.json"
    write_json(
        codebook,
        {
            "rubrics": [
                {
                    "id": name,
                    "trigger": "A relevant result is unread.",
                    "hint": "Read the result.",
                    "criterion": "Read a visible result.",
                }
                for name in names
            ]
        },
    )
    paths["handwritten_codebook"] = str(codebook)
    clients = {
        "policy": fake_client(
            [
                Turn("Search.", {}, {"tool": "search", "args": {"query": "synthetic"}}).render(),
                Turn(
                    "Answer.", {}, {"tool": "finish", "args": {"exact_answer": "Wrongplace"}}
                ).render(),
            ]
        ),
        "reader": fake_client([]),
        "judge": fake_client([json.dumps({"correct": False})]),
        "reassignment": fake_client([json.dumps({"labels": [{"step": 1, "family": names[1]}]})]),
        "teacher": fake_client(
            [
                Turn(
                    "Read the result.",
                    {},
                    {
                        "tool": "get_document",
                        "args": {"docid": "doc_a::chunk00000", "goal": "find the location"},
                    },
                ).render(),
            ]
        ),
        "validator": fake_client([json.dumps({"repairs_per_rubric": 1, "natural": 1})]),
    }
    experiment = Experiment(config, {"paths": paths, "workers": 1})
    experiment.tokenizer = lambda: tokenizer
    experiment.client = lambda role, *args, **kwargs: clients[role]
    for stage in ("collect", "judge", "reassign", "repair"):
        experiment.execute(stage)
    pair_path = tmp_path / "outputs/A2/repair/pairs.jsonl"
    pairs = list(read_jsonl(pair_path))
    assert len(pairs) == 1
    assert pairs[0]["rubric_id"] == names[1]
    assert "Read the result." not in pairs[0]["prompt"]["system"]
    config_b3 = load_experiment(
        Path(__file__).parents[1] / "configs/experiments/B3_chosen_sft_14b.toml"
    )
    config_b3["experiment"]["upstream"] = "A2"
    view = Experiment(config_b3, {"paths": paths})
    view.execute("prepare-sft")
    samples = list(read_jsonl(tmp_path / "outputs/B3/prepare-sft/samples.jsonl"))
    assert samples[0]["completion"] == pairs[0]["chosen"]
    experiment.execute("repair")
    assert len(clients["teacher"].requests) == 1


def test_resuming_evaluation_preserves_expired_deadline(tmp_path, recipe, fake_client):
    from repair.harness.rollout import collect

    config = copy.deepcopy(recipe)
    config["rollout"]["evaluation_seconds"] = 1
    path = tmp_path / "trajectories.jsonl"
    write_json(path.with_suffix(".timing.json"), {"started_at": 0, "duration_seconds": 1})
    client = fake_client([])
    rows = collect(
        [{"id": "q1", "question": "Synthetic question"}],
        path,
        samples=3,
        workers=1,
        config=config,
        evaluation=True,
        policy=client,
        retrieval=None,
        tokenizer=CharacterTokenizer(),
        stage="evaluation",
    )
    assert len(rows) == 3 and all(row["status"] == "timeout" for row in rows)
    assert not client.requests
