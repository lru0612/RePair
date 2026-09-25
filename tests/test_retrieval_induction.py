import pytest

from repair.io import write_jsonl


class CharacterTokenizer:
    name_or_path = "synthetic-character-tokenizer"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def test_chunk_boundaries_and_bm25_roundtrip(tmp_path, recipe):
    pytest.importorskip("bm25s")
    from repair.harness.retrieval import Retrieval, build_index, chunk_document

    tokenizer = CharacterTokenizer()
    settings = {**recipe["retrieval"], "chunk_tokens": 80, "chunk_overlap": 8}
    record = {"docid": "a", "title": "Example", "text": "# Place\n" + "Northvale river " * 12}
    chunks = list(chunk_document(record, tokenizer, settings))
    assert len(chunks) > 1
    assert all(len(tokenizer.encode(chunk["text"])) <= 80 for chunk in chunks)
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(
        corpus, [record, {"docid": "b", "text": "A different synthetic document about mountains."}]
    )
    destination = tmp_path / "index"
    build_index(corpus, destination, tokenizer, settings)
    retrieval = Retrieval(destination, tokenizer, settings, reader=None)
    try:
        hits = retrieval.search("Northvale")
        assert hits and hits[0]["parent_docid"] == "a"
        assert retrieval.search("qzxvnohit") == []
    finally:
        retrieval.close()


def test_weighted_clustering_has_twelve_groups(recipe):
    pytest.importorskip("sklearn")
    from repair.pairs.induction import cluster_issues

    findings = [
        {"id": f"issue-{i}", "hint": f"Read visible result group {i} and compare evidence."}
        for i in range(24)
    ]
    findings.extend([findings[0]] * 3)
    groups, stats = cluster_issues(findings, recipe["induction"], recipe["seed"])
    assert len(groups) == 12 and sum(map(len, groups)) == 27
    assert stats["issue_frequencies"]["issue-0"] == 4
