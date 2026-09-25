"""Stage execution and explicit dependencies between experiment arms."""

from __future__ import annotations

import json
import os
from pathlib import Path

from repair.config import output_root, public_config, require_path
from repair.evaluation import check_split_overlap, grade, load_questions, summarize
from repair.harness.storage import settings_path
from repair.io import file_hash, fingerprint, read_jsonl, write_json, write_jsonl
from repair.models import ChatClient, UsageLog
from repair.pairs.schema import failure_pool, read_codebook


def stages(config: dict) -> list[str]:
    experiment = config["experiment"]
    kind = experiment["kind"]
    if kind in ("cost", "budget"):
        return ["analyze"]
    if kind == "base":
        return ["evaluate"]
    if kind == "warm_start":
        return ["collect", "prepare-sft", "train", "evaluate"]
    if kind == "trajectory_sft":
        return ["collect", "judge", "prepare-sft", "train", "evaluate"]
    if kind == "chosen_sft":
        return ["prepare-sft", "train", "evaluate"]
    if kind == "top_k":
        return ["select", "train", "evaluate"]
    if kind == "no_rubric":
        return ["collect", "judge", "diagnose", "repair", "train", "evaluate"]
    result = ["collect", "judge"]
    if experiment["codebook"] == "induced":
        result.extend(["diagnose", "induce"])
    return result + ["reassign", "repair", "train", "evaluate"]


def ensure_manifest(directory: Path, definition: dict) -> None:
    path = directory / "manifest.json"
    digest = fingerprint(definition)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["fingerprint"] != digest:
            raise ValueError(
                f"Existing {directory.name} artifacts use different settings or inputs"
            )
    else:
        write_json(path, {"fingerprint": digest, "definition": definition})


class Experiment:
    def __init__(self, config: dict, runtime: dict, *, allow_model_override: bool = False):
        self.config, self.runtime = config, runtime
        self.arm = config["experiment"]
        self.root = output_root(runtime)
        self.directory = self.root / self.arm["id"]
        self.allow_model_override = allow_model_override
        self.clients = []

    def close(self) -> None:
        for client in self.clients:
            client.close()

    def model_path(self, identity: str) -> str:
        if identity.startswith("base_"):
            source = self.runtime.get("models", {}).get(identity)
            if not source:
                raise ValueError(f"Set models.{identity}")
            return source
        override = self.runtime.get("artifacts", {}).get(identity)
        path = Path(override) if override else self.root / identity / "training" / "model"
        if not path.is_dir():
            raise FileNotFoundError(f"Missing exported model for {identity}")
        return str(path)

    def tokenizer(self):
        from transformers import AutoTokenizer

        source = self.runtime.get("models", {}).get(f"base_{self.arm['size']}b")
        if not source:
            raise ValueError(f"Set models.base_{self.arm['size']}b")
        return AutoTokenizer.from_pretrained(source, use_fast=True)

    def client(self, role: str, usage: UsageLog, *, artifact: str = "") -> ChatClient:
        fallback = {
            "diagnoser": "teacher",
            "rewriter": "teacher",
            "reassignment": "teacher",
            "validator": "reader",
        }
        services = self.runtime.get("services", {})
        service = services.get(role, services.get(fallback.get(role, ""), {}))
        if not service:
            raise ValueError(f"Configure services.{role}")
        if role == "policy":
            if service.get("artifact") != artifact:
                raise ValueError(f"The policy service must declare artifact={artifact!r}")
        elif service.get("model") != self.config["roles"][role] and not self.allow_model_override:
            raise ValueError(
                f"Role {role} requires {self.config['roles'][role]}; "
                "use --allow-model-override to record a different experiment"
            )
        client = ChatClient(service, usage)
        self.clients.append(client)
        return client

    def questions(self, split: str) -> list[dict]:
        expected = {
            "warm_start": self.config["data"]["warm_start_questions"],
            "training": self.config["data"]["training_questions"],
            "test": self.config["data"]["test_questions"],
        }
        loaded = {}
        for name in expected:
            path = self.runtime.get("paths", {}).get(f"{name}_questions")
            if path and Path(path).exists():
                loaded[name] = load_questions(list(read_jsonl(path)), expected=expected[name])
        if split not in loaded:
            raise ValueError(f"Set paths.{split}_questions to the required question file")
        check_split_overlap(loaded)
        return loaded[split]

    def pool_directory(self) -> Path:
        return self.root / "pools" / self.arm["pool"]

    def pair_data(self, arm: str, *, accepted: bool = False) -> Path:
        filename = "accepted.jsonl" if accepted else "pairs.jsonl"
        generated = self.root / arm / "repair" / filename
        if generated.is_file():
            return generated
        directory = self.runtime.get("paths", {}).get("pairs_dir")
        if directory:
            for suffix in (".jsonl.gz", ".jsonl"):
                candidate = Path(directory) / f"{arm}{suffix}"
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(
            f"No preference dataset for {arm}; supply paths.pairs_dir or run repair"
        )

    def failures(self) -> list[dict]:
        pool = self.pool_directory()
        return failure_pool(
            list(read_jsonl(pool / "trajectories.jsonl")),
            list(read_jsonl(pool / "verdicts.jsonl")),
        )

    def codebook(self):
        mode = self.arm["codebook"]
        if mode == "handwritten":
            path = require_path(self.runtime, "handwritten_codebook")
        elif mode == "reuse":
            path = self.root / self.arm["upstream"] / "induce" / "codebook.json"
        else:
            path = self.directory / "induce" / "codebook.json"
        if not path.is_file() and mode != "handwritten":
            directory = self.runtime.get("paths", {}).get("codebooks_dir")
            if directory:
                size = 14 if mode == "reuse" else self.arm["size"]
                path = Path(directory) / f"induced_{size}b.json"
        value = json.loads(path.read_text())
        rubrics = read_codebook(value["rubrics"])
        if mode == "handwritten" and {r.id for r in rubrics} != {
            "query-churn",
            "search-without-read",
            "over-specified-query",
        }:
            raise ValueError("The handwritten arm requires the three specified rubric IDs")
        return rubrics

    def stage_directory(self, stage: str, inputs: list[Path] = ()) -> Path:
        directory = self.directory / stage
        ensure_manifest(
            directory,
            {
                "configuration": public_config(self.config, self.runtime),
                "inputs": {path.name: file_hash(path) for path in inputs},
                "model_override": self.allow_model_override,
            },
        )
        return directory

    def execute(self, stage: str, *, smoke: bool = False, resume: Path | None = None) -> dict:
        if stage not in stages(self.config):
            raise ValueError(f"Valid stages: {', '.join(stages(self.config))}")
        handlers = {
            "collect": self.collect,
            "judge": self.judge,
            "diagnose": self.diagnose,
            "induce": self.induce,
            "reassign": self.reassign,
            "repair": self.repair,
            "prepare-sft": self.prepare_sft,
            "select": self.select,
            "evaluate": self.evaluate,
            "analyze": self.analyze,
        }
        if stage == "train":
            return self.train(smoke=smoke, resume=resume)
        return handlers[stage]()

    def collect(self) -> dict:
        from repair.harness.retrieval import Retrieval
        from repair.harness.rollout import collect

        split = "warm_start" if self.arm["kind"] == "warm_start" else "training"
        queries = self.questions(split)
        directory = self.pool_directory()
        source = self.arm["starting_policy"]
        role = (
            "warm_teacher"
            if self.arm["kind"] == "warm_start"
            else "teacher"
            if self.arm["kind"] == "trajectory_sft"
            else "policy"
        )
        ensure_manifest(
            directory,
            {
                "queries": fingerprint(queries),
                "samples": self.arm["collection_samples"],
                "rollout": self.config["rollout"],
                "context": self.config["context"],
                "retrieval": self.config["retrieval"],
                "policy": source if role == "policy" else self.config["roles"][role],
                "role": role,
                "services": public_config(self.config, self.runtime)["services"],
                "index": file_hash(settings_path(require_path(self.runtime, "index"))),
            },
        )
        usage = UsageLog(directory / "usage.jsonl")
        policy = self.client(role, usage, artifact=source)
        reader = self.client("reader", usage)
        tokenizer = self.tokenizer()
        retrieval = Retrieval(
            require_path(self.runtime, "index"), tokenizer, self.config["retrieval"], reader
        )
        try:
            rows = collect(
                queries,
                directory / "trajectories.jsonl",
                samples=self.arm["collection_samples"],
                workers=self.runtime.get("workers", 4),
                config=self.config,
                evaluation=False,
                policy=policy,
                retrieval=retrieval,
                tokenizer=tokenizer,
                stage="collection",
            )
        finally:
            retrieval.close()
        return {"stage": "collect", "rollouts": len(rows)}

    def judge(self) -> dict:
        directory = self.pool_directory()
        trajectories = list(read_jsonl(directory / "trajectories.jsonl"))
        references = {q["id"]: q["answer"] for q in self.questions("training")}
        expected = {
            f"{query_id}:{sample}"
            for query_id in references
            for sample in range(self.arm["collection_samples"])
        }
        if {row["id"] for row in trajectories} != expected or len(trajectories) != len(expected):
            raise ValueError("Complete the collection before grading the training pool")
        self._grade(trajectories, references, directory, "collection")
        return {"stage": "judge", "rollouts": len(trajectories)}

    def _grade(self, trajectories: list[dict], references: dict, directory: Path, stage: str):
        ensure_manifest(
            directory / "grading",
            {
                "trajectories": fingerprint(trajectories),
                "references": fingerprint(references),
                "judge": self.runtime.get("services", {}).get("judge", {}).get("model"),
                "seed": self.config["seed"],
            },
        )
        path = directory / "verdicts.jsonl"
        previous = list(read_jsonl(path)) if path.exists() else []
        by_id = {row["id"]: row for row in previous}
        if len(by_id) != len(previous):
            raise ValueError("Duplicate existing verdicts")
        judge = self.client("judge", UsageLog(directory / "usage.jsonl"))
        for trajectory in trajectories:
            if trajectory["id"] not in by_id:
                by_id[trajectory["id"]] = grade(
                    trajectory,
                    references[trajectory["query_id"]],
                    judge,
                    seed=self.config["seed"],
                    stage=stage,
                )
                write_jsonl(path, by_id.values())
        return list(by_id.values())

    def diagnose(self) -> dict:
        from repair.pairs.induction import diagnose

        inputs = [self.pool_directory() / name for name in ("trajectories.jsonl", "verdicts.jsonl")]
        directory = self.stage_directory("diagnose", inputs)
        path = directory / "results.jsonl"
        rows = list(read_jsonl(path)) if path.exists() else []
        done = {row["trajectory_id"] for row in rows}
        client = self.client("diagnoser", UsageLog(directory / "usage.jsonl"))
        for failure in self.failures():
            if failure["id"] not in done:
                rows.append(
                    {
                        "trajectory_id": failure["id"],
                        "findings": diagnose(failure, client, self.config),
                    }
                )
                write_jsonl(path, rows)
        findings = [finding for row in rows for finding in row["findings"]]
        write_jsonl(directory / "findings.jsonl", findings)
        return {"stage": "diagnose", "findings": len(findings)}

    def induce(self) -> dict:
        from repair.pairs.induction import induce_codebook

        path = self.directory / "diagnose" / "findings.jsonl"
        directory = self.stage_directory("induce", [path])
        output = directory / "codebook.json"
        if output.exists():
            return {"stage": "induce", "reused": True}
        usage = UsageLog(directory / "usage.jsonl")
        value = induce_codebook(
            list(read_jsonl(path)),
            self.failures(),
            self.client("rewriter", usage),
            self.client("consolidator", usage),
            self.config,
        )
        value["expected_rubrics"] = self.arm["expected_rubrics"]
        value["matches_reported_size"] = len(value["rubrics"]) == self.arm["expected_rubrics"]
        write_json(output, value)
        return {
            "stage": "induce",
            "rubrics": len(value["rubrics"]),
            "matches_reported_size": value["matches_reported_size"],
        }

    def reassign(self) -> dict:
        from repair.pairs.induction import reassign

        directory = self.stage_directory(
            "reassign",
            [self.pool_directory() / name for name in ("trajectories.jsonl", "verdicts.jsonl")],
        )
        codebook = self.codebook()
        ensure_manifest(directory / "codebook", {"rubrics": [r.as_dict() for r in codebook]})
        path = directory / "results.jsonl"
        rows = list(read_jsonl(path)) if path.exists() else []
        done = {row["trajectory_id"] for row in rows}
        client = self.client("reassignment", UsageLog(directory / "usage.jsonl"))
        for failure in self.failures():
            if failure["id"] not in done:
                rows.append(
                    {
                        "trajectory_id": failure["id"],
                        "assignments": reassign(failure, codebook, client, self.config),
                    }
                )
                write_jsonl(path, rows)
        assignments = [assignment for row in rows for assignment in row["assignments"]]
        write_jsonl(directory / "assignments.jsonl", assignments)
        return {"stage": "reassign", "assignments": len(assignments)}

    def repair(self) -> dict:
        from repair.pairs.repair import repair_assignment
        from repair.training.data import encode_completion

        no_rubric = self.arm["kind"] == "no_rubric"
        input_path = self.directory / (
            "diagnose/findings.jsonl" if no_rubric else "reassign/assignments.jsonl"
        )
        directory = self.stage_directory(
            "repair",
            [
                input_path,
                *[
                    self.pool_directory() / name
                    for name in ("trajectories.jsonl", "verdicts.jsonl")
                ],
            ],
        )
        if no_rubric:
            assignments = [
                {"trajectory_id": row["trajectory_id"], "turn": row["turn"]}
                for row in read_jsonl(input_path)
            ]
            codebook = {}
        else:
            assignments = list(read_jsonl(input_path))
            codebook = {r.id: r for r in self.codebook()}
        failures = {row["id"]: row for row in self.failures()}
        references = {q["id"]: q["answer"] for q in self.questions("training")}
        usage = UsageLog(directory / "usage.jsonl")
        role = "policy" if self.arm["repairer"] == "learner" else "teacher"
        repairer = self.client(role, usage, artifact=self.arm["starting_policy"])
        validator = None if no_rubric else self.client("validator", usage)
        path = directory / "results.jsonl"
        previous = list(read_jsonl(path)) if path.exists() else []
        done = {row["assignment"] for row in previous}
        for assignment in assignments:
            identity = fingerprint(assignment)
            if identity in done:
                continue
            failure = failures[assignment["trajectory_id"]]
            pair, attempts = repair_assignment(
                failure,
                assignment,
                codebook.get(assignment.get("rubric_id")),
                repairer=repairer,
                validator=validator,
                reference=references[failure["query_id"]],
                config=self.config,
            )
            previous.append({"assignment": identity, "pair": pair, "attempts": attempts})
            write_jsonl(path, previous)
        accepted = [row["pair"] for row in previous if row["pair"] is not None]
        write_jsonl(directory / "accepted.jsonl", accepted)
        tokenizer = self.tokenizer()
        pairs = [
            row
            for row in accepted
            if max(
                len(encode_completion(tokenizer, row["prompt"], row[side]).input_ids)
                for side in ("chosen", "rejected")
            )
            <= self.config["training"]["max_length"]
        ]
        write_jsonl(directory / "pairs.jsonl", pairs)
        return {
            "stage": "repair",
            "accepted": len(accepted),
            "length_filtered": len(accepted) - len(pairs),
        }

    def prepare_sft(self) -> dict:
        from repair.pairs.datasets import chosen_samples, successful_turns

        if self.arm["kind"] == "chosen_sft":
            source = self.pair_data(self.arm["upstream"], accepted=True)
            inputs = [source]
            rows = chosen_samples(list(read_jsonl(source)))
        elif self.arm["kind"] == "trajectory_sft":
            pool = self.pool_directory()
            inputs = [pool / "trajectories.jsonl", pool / "verdicts.jsonl"]
            rows = successful_turns(
                list(read_jsonl(pool / "trajectories.jsonl")),
                list(read_jsonl(pool / "verdicts.jsonl")),
            )
        else:
            # Warm-start demonstrations include the main split and all three auxiliary corpora.
            inputs = [require_path(self.runtime, "warm_sft")]
            rows = list(read_jsonl(inputs[0]))
            sources = {row.get("dataset") for row in rows}
            if sources != {"browsecomp_plus", "patents", "web", "web_simple"}:
                raise ValueError("warm_sft must identify all four warm-start data sources")
            warm_ids = {q["id"] for q in self.questions("warm_start")}
            if any(
                row.get("query_id") not in warm_ids
                for row in rows
                if row["dataset"] == "browsecomp_plus"
            ):
                raise ValueError(
                    "Warm-start samples contain a question outside the warm-start split"
                )
        directory = self.stage_directory("prepare-sft", inputs)
        write_jsonl(directory / "samples.jsonl", rows)
        return {"stage": "prepare-sft", "samples": len(rows)}

    def select(self) -> dict:
        from repair.pairs.datasets import top_k_pairs

        source = self.root / self.arm["upstream"]
        pairs = list(read_jsonl(source / "repair" / "pairs.jsonl"))
        assignments = list(read_jsonl(source / "reassign" / "assignments.jsonl"))
        selected, names = top_k_pairs(pairs, assignments, self.codebook(), self.arm["top_k"])
        if self.arm["top_k"] == 1 and names != ["failed-search-repeat"]:
            raise ValueError("Top-1 differs from the specified failed-search-repeat rubric")
        directory = self.stage_directory(
            "select",
            [
                source / "repair" / "pairs.jsonl",
                source / "reassign" / "assignments.jsonl",
                source / "induce" / "codebook.json",
            ],
        )
        write_jsonl(directory / "pairs.jsonl", selected)
        write_json(directory / "selection.json", {"rubrics": names, "ranking": "assignment_count"})
        return {"stage": "select", "rubrics": names}

    def train(self, *, smoke: bool, resume: Path | None) -> dict:
        from repair.training.run import train

        if self.config["training"]["method"] == "sft":
            data = self.directory / "prepare-sft" / "samples.jsonl"
        else:
            data = (
                self.directory / "select" / "pairs.jsonl"
                if self.arm["kind"] == "top_k"
                else self.pair_data(self.arm["id"])
            )
        stage = "smoke-training" if smoke else "training"
        directory = self.stage_directory(stage, [data])
        summary = train(
            self.config,
            data,
            self.model_path(self.arm["starting_policy"]),
            directory,
            smoke=smoke,
            resume=resume,
        )
        return summary if int(os.environ.get("RANK", "0")) == 0 else {}

    def evaluate(self) -> dict:
        from repair.harness.retrieval import Retrieval
        from repair.harness.rollout import collect

        queries = self.questions("test")
        directory = self.stage_directory(
            "evaluation",
            [
                require_path(self.runtime, "test_questions"),
                settings_path(require_path(self.runtime, "index")),
            ],
        )
        usage = UsageLog(directory / "usage.jsonl")
        artifact = self.arm["starting_policy"] if self.arm["kind"] == "base" else self.arm["id"]
        policy = self.client("policy", usage, artifact=artifact)
        tokenizer = self.tokenizer()
        retrieval = Retrieval(
            require_path(self.runtime, "index"),
            tokenizer,
            self.config["retrieval"],
            self.client("reader", usage),
        )
        try:
            rows = collect(
                queries,
                directory / "trajectories.jsonl",
                samples=self.config["rollout"]["evaluation_samples"],
                workers=self.runtime.get("workers", 4),
                config=self.config,
                evaluation=True,
                policy=policy,
                retrieval=retrieval,
                tokenizer=tokenizer,
                stage="evaluation",
            )
        finally:
            retrieval.close()
        verdicts = self._grade(
            rows, {q["id"]: q["answer"] for q in queries}, directory, "evaluation"
        )
        report = summarize(queries, rows, verdicts, self.config["rollout"]["evaluation_samples"])
        write_json(directory / "summary.json", report)
        return report

    def analyze(self) -> dict:
        from repair.costs import estimate_budget, summarize_usage

        directory = self.stage_directory("analysis")
        if self.arm["kind"] == "budget":
            report = estimate_budget(
                json.loads(require_path(self.runtime, "budget_inputs").read_text())
            )
        else:
            source = self.root / self.arm["upstream"]
            paths = [self.root / "pools" / "pi0_14b_1" / "usage.jsonl"]
            paths.extend(
                source / stage / "usage.jsonl"
                for stage in ("diagnose", "induce", "reassign", "repair")
            )
            rows = [row for path in paths for row in read_jsonl(path)]
            training = json.loads((source / "training" / "training_summary.json").read_text())
            report = summarize_usage(rows, [training])
        write_json(directory / "summary.json", report)
        return report
