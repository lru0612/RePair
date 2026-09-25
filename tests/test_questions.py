import base64
import hashlib
import json

import pytest

from repair.questions import CANARY, decrypt, write_questions


def obfuscate(text: str) -> str:
    data = text.encode("utf-8")
    key = hashlib.sha256(CANARY.encode("utf-8")).digest()
    key = key * (len(data) // len(key)) + key[: len(data) % len(key)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


def test_decrypt_inverts_the_benchmark_obfuscation():
    text = "Which author, born before 1950, wrote a story set in Zürich? " * 3
    assert decrypt(obfuscate(text)) == text


def test_questions_follow_the_split_order(tmp_path):
    source = {
        qid: {"id": qid, "question": f"question {qid}", "answer": f"answer {qid}"}
        for qid in ["1", "2", "3", "4"]
    }
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"warm_start": ["3"], "training": ["2", "1"], "test": ["4"]}))
    result = write_questions(splits, tmp_path / "questions", source)
    assert result["questions"] == {"warm_start": 1, "training": 2, "test": 1}
    lines = (tmp_path / "questions" / "training.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["2", "1"]
    assert lines[0] == '{"id":"2","question":"question 2","answer":"answer 2"}'


def test_unknown_identifiers_are_rejected(tmp_path):
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"warm_start": [], "training": ["9"], "test": []}))
    with pytest.raises(ValueError, match="training"):
        write_questions(splits, tmp_path / "questions", {})
