"""Command-line entry points for experiment stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repair.config import load_experiment, load_runtime, public_config
from repair.experiment import Experiment, stages


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="repair")
    commands = result.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="Validate and print an experiment without making model requests"
    )
    check.add_argument("--experiment", type=Path, required=True)
    check.add_argument("--runtime", type=Path)
    run = commands.add_parser("run", help="Run one stage of an experiment")
    run.add_argument("--experiment", type=Path, required=True)
    run.add_argument("--runtime", type=Path, required=True)
    run.add_argument("--stage", required=True)
    run.add_argument(
        "--smoke", action="store_true", help="One optimizer step; permits CPU or fewer GPUs"
    )
    run.add_argument("--resume", type=Path, help="An epoch checkpoint with optimizer state")
    run.add_argument("--allow-model-override", action="store_true")
    index = commands.add_parser("index", help="Build a BM25 index from external corpus JSONL")
    index.add_argument("--experiment", type=Path, required=True)
    index.add_argument("--corpus", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--tokenizer", required=True)
    questions = commands.add_parser(
        "questions", help="Recover the split questions from the official BrowseComp-Plus release"
    )
    questions.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    questions.add_argument("--output", type=Path, default=Path("data/questions"))
    return result


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "questions":
            from repair.questions import official_questions, write_questions

            result = write_questions(arguments.splits, arguments.output, official_questions())
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        config = load_experiment(arguments.experiment)
        if arguments.command == "check":
            runtime = load_runtime(arguments.runtime) if arguments.runtime else {}
            result = public_config(config, runtime)
            result["stages"] = stages(config)
        elif arguments.command == "index":
            from transformers import AutoTokenizer

            from repair.harness.retrieval import build_index

            result = build_index(
                arguments.corpus,
                arguments.output,
                AutoTokenizer.from_pretrained(arguments.tokenizer, use_fast=True),
                config["retrieval"],
            )
        else:
            if (arguments.smoke or arguments.resume) and arguments.stage != "train":
                raise ValueError("--smoke and --resume apply only to the train stage")
            experiment = Experiment(
                config,
                load_runtime(arguments.runtime),
                allow_model_override=arguments.allow_model_override,
            )
            try:
                result = experiment.execute(
                    arguments.stage, smoke=arguments.smoke, resume=arguments.resume
                )
            finally:
                experiment.close()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f"repair: {error}", file=sys.stderr)
        raise SystemExit(2) from None
