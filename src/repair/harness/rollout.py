"""Structured search rollouts with deterministic retry seeds."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from repair.harness.context import WorkingContext, parse_turn
from repair.harness.prompts import (
    FORCE_FINISH_INSTRUCTION,
    OUTPUT_FORMAT,
    STRUCTURED_SYSTEM_PROMPT,
)
from repair.io import derive_seed, read_jsonl, stable_json, write_json
from repair.models import ChatClient, ContentRejected, ModelError


def run_query(
    query: dict,
    sample: int,
    *,
    policy: ChatClient,
    retrieval,
    tokenizer,
    config: dict,
    evaluation: bool,
    stage: str,
    deadline: float | None = None,
) -> dict:
    settings = config["rollout"]
    context = config["context"]
    state = WorkingContext(
        query["question"],
        OUTPUT_FORMAT,
        max_evidence=context["max_evidence"],
        evidence_chars=context["evidence_chars"],
        notes_chars=context["notes_chars"],
        history_chars=context["history_chars"],
    )
    identity = f"{query['id']}:{sample}"
    result = {
        "id": identity,
        "query_id": query["id"],
        "sample": sample,
        "question": query["question"],
        "turns": [],
        "answer": "",
        "status": "turn_limit",
        "natural_finish": False,
        "policy_generated_tokens": 0,
    }
    last_observation = ""
    for number in range(settings["max_turns"]):
        if deadline is not None and time.monotonic() >= deadline:
            result["status"] = "timeout"
            break
        forced = (
            result["policy_generated_tokens"] >= settings["force_finish_tokens"]
            or number == settings["max_turns"] - 1
        )
        user = state.render(last_observation)
        if forced:
            user += "\n\n" + FORCE_FINISH_INSTRUCTION
        messages = [
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompt_tokens = len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
            )
        )
        if prompt_tokens >= settings["context_tokens"]:
            result["status"] = "context_overflow"
            break
        turn = {
            "turn": number,
            "prompt": {"system": STRUCTURED_SYSTEM_PROMPT, "user": user},
            "completion": "",
            "forced_finish": forced,
            "status": "pending",
            "observation": "",
            "attempts": [],
        }
        result["turns"].append(turn)
        decode = {
            "thinking": True,
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "repetition_penalty": settings["repetition_penalty"],
        }
        if not evaluation:
            decode["max_tokens"] = settings["policy_max_tokens"]
        parsed = None
        penalties = [settings["repetition_penalty"], *settings["retry_penalties"]]
        for attempt, penalty in enumerate(penalties):
            seed = derive_seed(config["seed"], identity, number, attempt)
            decode["repetition_penalty"] = penalty
            try:
                reply = policy.generate(
                    messages,
                    settings=decode,
                    stage=stage,
                    role="policy",
                    identity=f"{identity}:{number}",
                    seed=seed,
                    deadline=deadline,
                )
            except ContentRejected:
                turn["status"] = "refused"
                break
            except TimeoutError:
                turn["status"] = "timeout"
                break
            except ModelError:
                turn["status"] = "transport_error"
                break
            generated = reply.completion_tokens
            if generated is None:
                generated = len(tokenizer.encode(reply.text, add_special_tokens=False))
            result["policy_generated_tokens"] += generated
            turn["completion"] = reply.text
            turn["attempts"].append(
                {
                    "seed": seed,
                    "repetition_penalty": penalty,
                    "finish_reason": reply.finish_reason,
                    "generated_tokens": generated,
                }
            )
            if reply.finish_reason == "length":
                turn["status"] = "truncated"
                continue
            try:
                parsed = parse_turn(reply.text)
            except (ValueError, TypeError, KeyError):
                turn["status"] = "invalid_action"
                break
            turn["status"] = "ok"
            break
        if parsed is None:
            result["status"] = turn["status"]
            break
        turn["reasoning"] = parsed.reasoning
        turn["action"] = parsed.action
        state.update(parsed.state)
        tool, args = parsed.action["tool"], parsed.action["args"]
        if tool == "finish":
            result["answer"] = args["exact_answer"]
            result["status"] = "completed"
            result["natural_finish"] = not forced
            break
        if forced:
            turn["status"] = "forced_finish_ignored"
            result["status"] = "forced_finish_ignored"
            break
        try:
            if tool == "search":
                hits = retrieval.search(args["query"])
                last_observation = stable_json(hits)
                state.note_search(parsed.action, hits)
            else:
                if args["docid"] not in state.seen_docids:
                    last_observation = "Document ID has not appeared in search results."
                    state.record_action(parsed.action, last_observation)
                else:
                    reader_decode = {"thinking": True, "temperature": 0.0}
                    if not evaluation:
                        reader_decode["max_tokens"] = settings["reader_max_tokens"]
                    last_observation, facts = retrieval.get_document(
                        args["docid"],
                        args["goal"],
                        settings=reader_decode,
                        stage=stage,
                        identity=f"{identity}:{number}",
                        deadline=deadline,
                        seed=derive_seed(config["seed"], identity, number, "reader"),
                    )
                    state.note_document(parsed.action, facts, last_observation)
        except (ModelError, ValueError, TimeoutError):
            turn["status"] = "tool_error"
            result["status"] = (
                "timeout" if deadline is not None and time.monotonic() >= deadline else "tool_error"
            )
            break
        turn["observation"] = last_observation
    return result


def collect(
    queries: list[dict],
    path: Path,
    *,
    samples: int,
    workers: int,
    config: dict,
    evaluation: bool,
    **kwargs,
) -> list[dict]:
    expected = {f"{q['id']}:{sample}" for q in queries for sample in range(samples)}
    existing = list(read_jsonl(path)) if path.exists() else []
    done = {row["id"] for row in existing}
    if len(done) != len(existing) or not done <= expected:
        raise ValueError("Existing rollouts have duplicate or unexpected sample identifiers")
    path.parent.mkdir(parents=True, exist_ok=True)
    seconds = config["rollout"]["evaluation_seconds"] if evaluation else 0
    deadline = None
    if seconds:
        timing_path = path.with_suffix(".timing.json")
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            if timing["duration_seconds"] != seconds:
                raise ValueError("The evaluation deadline differs from the existing job")
        else:
            timing = {"started_at": time.time(), "duration_seconds": seconds}
            write_json(timing_path, timing)
        remaining = timing["started_at"] + seconds - time.time()
        deadline = time.monotonic() + max(0, remaining)
    jobs = [(q, i) for q in queries for i in range(samples) if f"{q['id']}:{i}" not in done]
    with path.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_query,
                    q,
                    i,
                    config=config,
                    evaluation=evaluation,
                    deadline=deadline,
                    **kwargs,
                )
                for q, i in jobs
            ]
            for future in as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                existing.append(row)
    return sorted(existing, key=lambda row: (row["query_id"], row["sample"]))
