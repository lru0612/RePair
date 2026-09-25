"""Lossless storage for document text and precomputed BM25 scores."""

from __future__ import annotations

import gzip
import json
import zlib
from pathlib import Path


def encode_parent(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8"), level=9)


def parent_bytes(value: str | bytes) -> bytes:
    return value.encode("utf-8") if isinstance(value, str) else zlib.decompress(value)


def encode_chunk(text: str, parent: bytes) -> tuple[bytes, int]:
    content = text.encode("utf-8")
    offset = 0
    for position in (len(content) // 2, len(content) // 4, 0):
        anchor = content[position : position + 96]
        match = parent.find(anchor) if anchor else -1
        if match >= 0:
            offset = max(0, min(match - 16384, len(parent) - 32768))
            break
    compressor = zlib.compressobj(level=9, zdict=parent[offset : offset + 32768])
    return compressor.compress(content) + compressor.flush(), offset


def decode_chunk(value: str | bytes, parent: bytes | None, offset: int | None) -> str:
    if isinstance(value, str):
        return value
    if offset is None:
        return zlib.decompress(value).decode("utf-8")
    if parent is None:
        raise ValueError("A dictionary-compressed chunk needs its parent document")
    decoder = zlib.decompressobj(zdict=parent[offset : offset + 32768])
    return (decoder.decompress(value) + decoder.flush()).decode("utf-8")


def settings_path(directory: Path) -> Path:
    current = directory / "settings.json"
    return current if current.exists() else directory / "manifest.json"


def load_bm25(directory: Path, num_docs: int):
    import bm25s
    import numpy as np

    archive = directory / "scores.npz"
    if not archive.exists():
        return bm25s.BM25.load(str(directory), mmap=True)
    parameters = json.loads((directory / "parameters.json").read_text())
    if parameters.get("method") != "lucene":
        raise ValueError("The compressed score format requires Lucene BM25")
    result = bm25s.BM25(**parameters)
    with gzip.open(directory / "vocabulary.json.gz", "rt", encoding="utf-8") as stream:
        result.vocab_dict = json.load(stream)
    result.unique_token_ids_set = set(result.vocab_dict.values())
    with np.load(archive, allow_pickle=False) as arrays:
        result.scores = {name: arrays[name] for name in ("data", "indices", "indptr")}
    result.scores["num_docs"] = num_docs
    result.nonoccurrence_array = None
    return result
