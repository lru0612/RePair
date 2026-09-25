import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from repair.config import load_experiment
from repair.experiment import Experiment
from repair.harness.storage import decode_chunk, encode_chunk, encode_parent, parent_bytes
from repair.io import read_jsonl, write_json, write_jsonl
from repair.pairs.datasets import chosen_samples


def test_compressed_jsonl_and_minimal_preferences(tmp_path):
    row = {
        "query_id": "q",
        "rubric_id": "example-rule",
        "prompt": {"system": "system", "user": "question"},
        "chosen": "accepted",
        "rejected": "rejected",
    }
    path = tmp_path / "pairs.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    assert list(read_jsonl(path)) == [row]
    samples = chosen_samples([row])
    assert samples[0]["completion"] == "accepted" and samples[0]["id"]


@pytest.mark.parametrize("position", [0, 20000, 50000])
def test_dictionary_compression_preserves_unicode(position):
    text = "A synthetic document. 中文文字。\n" * 5000
    parent = text.encode()
    chunk = "Heading\n" + text[position : position + 1500]
    compressed_parent = encode_parent(text)
    assert parent_bytes(compressed_parent) == parent
    encoded, offset = encode_chunk(chunk, parent)
    assert decode_chunk(encoded, parent, offset) == chunk
    assert decode_chunk(chunk, None, None) == chunk


def test_bundled_pair_and_codebook_resolution(tmp_path, recipe):
    paths = {
        "output_root": str(tmp_path / "outputs"),
        "pairs_dir": str(tmp_path / "pairs"),
        "codebooks_dir": str(tmp_path / "codebooks"),
    }
    Path(paths["pairs_dir"]).mkdir()
    bundled = Path(paths["pairs_dir"]) / "A3.jsonl.gz"
    bundled.write_bytes(b"placeholder")
    experiment = Experiment(recipe, {"paths": paths})
    assert experiment.pair_data("A3") == bundled
    generated = tmp_path / "outputs/A3/repair/pairs.jsonl"
    write_jsonl(generated, [])
    assert experiment.pair_data("A3") == generated
    write_json(
        Path(paths["codebooks_dir"]) / "induced_14b.json",
        {
            "rubrics": [
                {
                    "id": "example-rule",
                    "trigger": "trigger",
                    "hint": "hint",
                    "criterion": "criterion",
                },
            ]
        },
    )
    assert experiment.codebook()[0].id == "example-rule"
    round_two = load_experiment(
        Path(__file__).parents[1] / "configs/experiments/A5_self_play_round2_14b.toml"
    )
    assert Experiment(round_two, {"paths": paths}).codebook()[0].id == "example-rule"


class CharacterTokenizer:
    name_or_path = "synthetic-character-tokenizer"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def test_compressed_index_matches_plain_index(tmp_path, recipe, fake_client):
    pytest.importorskip("bm25s")
    import numpy as np

    from repair.harness.retrieval import Retrieval, build_index

    tokenizer = CharacterTokenizer()
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(
        corpus,
        [
            {"docid": "a", "text": "Northvale has a river."},
            {"docid": "b", "text": "Westbridge has a mountain."},
        ],
    )
    index = tmp_path / "index"
    build_index(corpus, index, tokenizer, recipe["retrieval"])
    original = Retrieval(index, tokenizer, recipe["retrieval"], None)
    expected = original.search("Northvale river")
    original.close()
    db = sqlite3.connect(index / "documents.sqlite")
    parents = dict(db.execute("SELECT docid,text FROM parents"))
    chunks = list(db.execute("SELECT position,parent_docid,text FROM chunks"))
    db.execute("ALTER TABLE chunks ADD COLUMN dictionary_offset INTEGER")
    for key, value in parents.items():
        db.execute("UPDATE parents SET text=? WHERE docid=?", (encode_parent(value), key))
    for position, parent_id, text in chunks:
        encoded, offset = encode_chunk(text, parents[parent_id].encode())
        db.execute(
            "UPDATE chunks SET text=?, dictionary_offset=? WHERE position=?",
            (encoded, offset, position),
        )
    db.commit()
    db.close()
    sparse = index / "bm25"
    arrays = {
        key: np.load(sparse / f"{key}.csc.index.npy") for key in ("data", "indices", "indptr")
    }
    np.savez_compressed(sparse / "scores.npz", **arrays)
    write_json(sparse / "parameters.json", {"k1": 1.5, "b": 0.75, "method": "lucene"})
    with gzip.open(sparse / "vocabulary.json.gz", "wb") as stream:
        stream.write((sparse / "vocab.index.json").read_bytes())
    reader = fake_client(["- Northvale has a river."])
    restored = Retrieval(index, tokenizer, recipe["retrieval"], reader)
    try:
        assert restored.search("Northvale river") == expected
        observation, facts = restored.get_document(
            "a",
            "find a river",
            settings={"thinking": True, "temperature": 0},
            stage="evaluation",
            identity="q:0:0",
            seed=1,
            deadline=None,
        )
        assert facts == ["Northvale has a river."]
        assert "Northvale has a river." in reader.requests[0][0][1]["content"]
    finally:
        restored.close()
