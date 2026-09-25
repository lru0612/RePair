"""Heading-aware chunks, BM25 retrieval, and goal-conditioned reading."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Iterator

from repair.harness.prompts import READER_SYSTEM_PROMPT
from repair.harness.storage import decode_chunk, load_bm25, parent_bytes, settings_path
from repair.io import file_hash, read_jsonl, write_json
from repair.models import ChatClient


def chunk_document(record: dict, tokenizer, settings: dict) -> Iterator[dict]:
    docid, body = str(record["docid"]), str(record["text"])
    title = str(record.get("title", "")).strip()
    sections = re.split(r"(?m)(?=^\s{0,3}#{1,6}\s+|^\s*<h[1-6][ >])", body)
    number = 0
    for section in sections:
        if not section.strip():
            continue
        heading = section.splitlines()[0].strip()
        prefix = title + "\n\n" if title else ""
        if heading.startswith(("#", "<h")) and heading != title:
            prefix += heading + "\n\n"
            section = "\n".join(section.splitlines()[1:])
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        available = settings["chunk_tokens"] - len(prefix_ids)
        if available <= settings["chunk_overlap"]:
            raise ValueError("A document heading leaves insufficient room for chunk overlap")
        token_ids = tokenizer.encode(section, add_special_tokens=False)
        if not token_ids:
            token_ids = tokenizer.encode(heading, add_special_tokens=False)
        for start in range(0, len(token_ids), available - settings["chunk_overlap"]):
            end = min(start + available, len(token_ids))
            text = prefix + tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
            yield {
                "docid": f"{docid}::chunk{number:05d}",
                "parent_docid": docid,
                "text": text,
                "token_start": start,
                "token_end": end,
            }
            number += 1
            if end == len(token_ids):
                break


def build_index(corpus: Path, destination: Path, tokenizer, settings: dict) -> dict:
    import bm25s
    import numpy as np
    import Stemmer
    from bm25s.tokenization import Tokenized, Tokenizer

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("The index directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(destination / "documents.sqlite")
    db.execute("CREATE TABLE parents (docid TEXT PRIMARY KEY, text TEXT NOT NULL)")
    db.execute(
        "CREATE TABLE chunks (position INTEGER PRIMARY KEY, docid TEXT UNIQUE, parent_docid TEXT, text TEXT)"
    )
    sparse_tokenizer = Tokenizer(stopwords="en", stemmer=Stemmer.Stemmer("english"))
    corpus_ids, text_batch = [], []
    n_documents = n_chunks = 0

    def flush() -> None:
        ids = sparse_tokenizer.tokenize(
            text_batch, update_vocab=True, return_as="ids", show_progress=False, allow_empty=True
        )
        corpus_ids.extend(np.asarray(row, dtype=np.int32) for row in ids)
        text_batch.clear()
        db.commit()

    try:
        for document in read_jsonl(corpus):
            parent_text = str(document["text"])
            if document.get("title"):
                parent_text = f"{document['title']}\n\n{parent_text}"
            db.execute("INSERT INTO parents VALUES (?, ?)", (str(document["docid"]), parent_text))
            n_documents += 1
            for chunk in chunk_document(document, tokenizer, settings):
                db.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                    (n_chunks, chunk["docid"], chunk["parent_docid"], chunk["text"]),
                )
                text_batch.append(chunk["text"])
                n_chunks += 1
                if len(text_batch) >= 10000:
                    flush()
        if text_batch:
            flush()
    finally:
        db.close()
    if not n_chunks:
        raise ValueError("The corpus has no indexable chunks")
    retriever = bm25s.BM25(method="lucene", k1=settings["k1"], b=settings["b"])
    retriever.index(Tokenized(ids=corpus_ids, vocab=sparse_tokenizer.get_vocab_dict()))
    retriever.save(str(destination / "bm25"))
    manifest = {
        "documents": n_documents,
        "chunks": n_chunks,
        "corpus_hash": file_hash(corpus),
        "settings": settings,
        "tokenizer": tokenizer.name_or_path,
    }
    write_json(destination / "settings.json", settings)
    return manifest


class Retrieval:
    def __init__(self, directory: Path, tokenizer, settings: dict, reader: ChatClient):
        import bm25s
        import Stemmer

        self.bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        self.tokenizer, self.settings, self.reader = tokenizer, settings, reader
        self.db = sqlite3.connect(directory / "documents.sqlite", check_same_thread=False)
        self.lock = threading.Lock()
        self.size = self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        self.retriever = load_bm25(directory / "bm25", self.size)
        fields = {row[1] for row in self.db.execute("PRAGMA table_info(chunks)")}
        self.offset_column = "dictionary_offset" if "dictionary_offset" in fields else "NULL"
        manifest = json.loads(settings_path(directory).read_text())
        index_settings = manifest.get("settings", manifest)
        if any(
            index_settings.get(k) != settings[k]
            for k in ("k1", "b", "chunk_tokens", "chunk_overlap")
        ):
            raise ValueError("The index settings do not match this experiment")

    def close(self) -> None:
        self.db.close()

    def search(self, query: str) -> list[dict]:
        with self.lock:
            tokens = self.bm25s.tokenize(
                query, stopwords="en", stemmer=self.stemmer, show_progress=False
            )
            indices, scores = self.retriever.retrieve(
                tokens, k=min(self.settings["top_k"], self.size), show_progress=False
            )
            results = []
            parents = {}
            for index, score in zip(indices[0], scores[0]):
                if float(score) <= 0:
                    continue
                row = self.db.execute(
                    f"SELECT docid, parent_docid, text, {self.offset_column} FROM chunks WHERE position=?",
                    (int(index),),
                ).fetchone()
                if row[3] is not None and row[1] not in parents:
                    stored = self.db.execute(
                        "SELECT text FROM parents WHERE docid=?", (row[1],)
                    ).fetchone()
                    if stored is None:
                        raise ValueError("A retrieved chunk has no parent document")
                    parents[row[1]] = parent_bytes(stored[0])
                text = decode_chunk(row[2], parents.get(row[1]), row[3])
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                results.append(
                    {
                        "docid": row[0],
                        "parent_docid": row[1],
                        "snippet": self.tokenizer.decode(
                            ids[: self.settings["snippet_tokens"]], skip_special_tokens=True
                        ),
                    }
                )
        return results

    def get_document(
        self,
        docid: str,
        goal: str,
        *,
        settings: dict,
        stage: str,
        identity: str,
        seed: int,
        deadline: float | None,
    ) -> tuple[str, list[str]]:
        with self.lock:
            row = self.db.execute("SELECT text FROM parents WHERE docid=?", (docid,)).fetchone()
            if row is None:
                row = self.db.execute(
                    "SELECT p.text FROM parents p JOIN chunks c ON c.parent_docid=p.docid WHERE c.docid=?",
                    (docid,),
                ).fetchone()
        if row is None:
            return "Document not found.", []
        ids = self.tokenizer.encode(parent_bytes(row[0]).decode("utf-8"), add_special_tokens=False)
        document = self.tokenizer.decode(
            ids[: self.settings["document_tokens"]], skip_special_tokens=True
        )
        messages = [
            {"role": "system", "content": READER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"GOAL: {goal}\n\nDOCUMENT (docid {docid}):\n{document}\n\n"
                    f"Start with: The useful information in {docid} for goal {goal}:"
                ),
            },
        ]
        reply = self.reader.generate(
            messages,
            settings=settings,
            stage=stage,
            role="reader",
            identity=identity,
            seed=seed,
            deadline=deadline,
        )
        if reply.finish_reason == "length":
            raise ValueError("Reader generation was truncated")
        facts = [
            line.strip()[2:].strip().strip('"“”')
            for line in reply.content.splitlines()
            if line.strip().startswith("- ")
        ]
        return reply.content, facts
