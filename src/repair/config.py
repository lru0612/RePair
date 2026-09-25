"""Load experiment settings separately from machine configuration."""

from __future__ import annotations

import copy
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from repair.io import fingerprint


def read_toml(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def load_experiment(path: str | Path) -> dict:
    config = read_toml(path)
    validate_experiment(config)
    return config


def load_runtime(path: str | Path) -> dict:
    path = Path(path).resolve()
    runtime = read_toml(path)
    root = path.parent.parent
    for group in ("paths", "models", "artifacts"):
        for name, value in runtime.get(group, {}).items():
            if not isinstance(value, str) or not value:
                continue
            expanded = os.path.expandvars(os.path.expanduser(value))
            if "$" in expanded:
                raise ValueError(f"Unresolved environment variable in {group}.{name}")
            if group == "models" and not expanded.startswith(("/", ".")):
                continue
            runtime[group][name] = str((root / expanded).resolve())
    return runtime


def require_path(runtime: dict, key: str) -> Path:
    value = runtime.get("paths", {}).get(key)
    if not value:
        raise ValueError(f"Set paths.{key} in the runtime configuration")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Missing input for paths.{key}: {path.name}")
    return path


def output_root(runtime: dict) -> Path:
    value = runtime.get("paths", {}).get("output_root")
    if not value:
        raise ValueError("Set paths.output_root in the runtime configuration")
    return Path(value)


def validate_experiment(config: dict) -> None:
    experiment = config["experiment"]
    if experiment["size"] not in (8, 14):
        raise ValueError("experiment.size must be 8 or 14")
    if experiment["kind"] in ("cost", "budget"):
        return
    settings = config["rollout"]
    if len(settings["retry_penalties"]) != settings["truncation_retries"]:
        raise ValueError("Provide one repetition penalty per truncation retry")
    if settings["force_finish_tokens"] >= settings["context_tokens"]:
        raise ValueError("The finish threshold must be below the context window")
    if config["retrieval"]["chunk_overlap"] >= config["retrieval"]["chunk_tokens"]:
        raise ValueError("Chunk overlap must be smaller than the chunk size")
    train = config.get("training", {})
    if not train:
        return
    if train["method"] not in ("dpo", "sft"):
        raise ValueError("training.method must be dpo or sft")
    if train["precision"] != "bf16" or train["fsdp"] != "full_shard":
        raise ValueError("The supported training recipe uses bf16 and FSDP full_shard")
    optimizer = "adafactor" if train["method"] == "dpo" else "adamw"
    if train["optimizer"] != optimizer:
        raise ValueError(f"The {train['method']} implementation uses {optimizer}")
    if train["method"] == "sft" and train["train_mode"] not in ("full", "lora"):
        raise ValueError("SFT train_mode must be full or lora")
    if not train["append_eos"] or not train["supervise_eos"]:
        raise ValueError("The training implementation appends and supervises EOS")
    expected = train["per_device_batch"] * train["world_size"] * train["gradient_accumulation"]
    if train["global_batch"] != expected:
        raise ValueError("global_batch must equal per_device_batch × world_size × accumulation")
    if train["method"] == "dpo":
        if train["train_mode"] != "full":
            raise ValueError("The preference recipe requires full-parameter training")
        if not train["append_eos"] or not train["supervise_eos"]:
            raise ValueError("Both completions must append and supervise EOS")
        if train["packing"]:
            raise ValueError("Preference pairs must not be packed")


def public_config(config: dict, runtime: dict) -> dict:
    result = copy.deepcopy(config)
    result["configuration_hash"] = fingerprint(config)
    result["services"] = {
        role: {
            "model": value.get("model", ""),
            "backend": value.get("backend", ""),
            "artifact": value.get("artifact", ""),
        }
        for role, value in runtime.get("services", {}).items()
    }
    return result


def validate_world_size(training: dict, *, smoke: bool = False) -> int:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != training["world_size"] and not smoke:
        raise ValueError(
            f"This recipe requires {training['world_size']} processes; got {world}. "
            "Use torchrun, or use --smoke for an explicitly reduced verification run."
        )
    return world
