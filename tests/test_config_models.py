from pathlib import Path

import httpx
import pytest

from repair.config import load_experiment, public_config, validate_experiment
from repair.experiment import ensure_manifest, stages
from repair.models import ChatClient, ModelError


def test_all_experiment_recipes_and_round_starts():
    root = Path(__file__).parents[1] / "configs/experiments"
    recipes = {c["experiment"]["id"]: c for c in map(load_experiment, root.glob("*.toml"))}
    assert len(recipes) == 18
    assert recipes["A4"]["experiment"]["starting_policy"] == "A1"
    assert recipes["A5"]["experiment"]["starting_policy"] == "A4"
    assert recipes["A6"]["roles"]["validator"] == "deepseek-v4-flash"
    assert recipes["B1"]["training"]["train_mode"] == "full"
    for arm in ("B2", "B3"):
        assert recipes[arm]["training"]["train_mode"] == "full"
        assert not any(key.startswith("lora_") for key in recipes[arm]["training"])
    for arm in ("A1", "A1_8b"):
        assert recipes[arm]["training"]["train_mode"] == "lora"
    assert recipes["A2"]["experiment"]["pool"] == recipes["A3"]["experiment"]["pool"]
    assert recipes["A6"]["experiment"]["pool"] == recipes["A7"]["experiment"]["pool"]
    assert stages(recipes["D2"]) == ["analyze"]
    for config in recipes.values():
        if config.get("training", {}).get("method") == "dpo":
            train = config["training"]
            assert (
                train["world_size"] * train["per_device_batch"] * train["gradient_accumulation"]
                == 64
            )
            assert train["supervise_eos"] and train["max_length"] == 20480


def test_sft_rejects_unknown_train_mode():
    config = load_experiment(
        Path(__file__).parents[1] / "configs/experiments/B3_chosen_sft_14b.toml"
    )
    config["training"]["train_mode"] = "frozen"
    with pytest.raises(ValueError, match="SFT train_mode must be full or lora"):
        validate_experiment(config)


def test_manifests_reject_stale_inputs(tmp_path):
    ensure_manifest(tmp_path, {"input": "one"})
    ensure_manifest(tmp_path, {"input": "one"})
    with pytest.raises(ValueError):
        ensure_manifest(tmp_path, {"input": "two"})


def test_explicit_backend_does_not_depend_on_localhost():
    client = ChatClient(
        {
            "model": "synthetic",
            "base_url": "https://api.example.com/v1",
            "backend": "vllm",
        }
    )
    try:
        payload = client.payload(
            [], {"thinking": True, "temperature": 1.0, "repetition_penalty": 1.2}, 123
        )
        assert payload["repetition_penalty"] == 1.2
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert "max_tokens" not in payload
    finally:
        client.close()


def test_errors_and_resolved_config_do_not_expose_secrets(recipe):
    client = ChatClient(
        {"model": "synthetic", "base_url": "https://api.example.com/v1", "backend": "vllm"}
    )
    client.http.close()
    client.http = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, text="secret request text")
        )
    )
    try:
        with pytest.raises(ModelError) as error:
            client.generate([], settings={}, stage="repair", role="repairer", identity="x")
        assert "secret request text" not in str(error.value)
        result = public_config(
            recipe,
            {
                "services": {
                    "teacher": {
                        "model": "synthetic",
                        "base_url": "secret address",
                        "api_key": "secret credential",
                    }
                }
            },
        )
        assert "secret" not in str(result)
    finally:
        client.close()
