from pathlib import Path

import pytest

from repair.config import load_experiment
from repair.io import write_jsonl
from repair.training.data import Collator, encode_completion, encode_pairs

torch = pytest.importorskip("torch")


@pytest.fixture
def tokenizer(tmp_path):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "<think>",
        "</think>",
        "<state>",
        "</state>",
        "<action>",
        "</action>",
        "system",
        "user",
        "assistant",
        "read",
        "search",
        "Northvale",
        "visible",
        "evidence",
        "Use",
        "tools",
        "answer",
        "first",
        "next",
        "wrong",
        "correct",
        "query",
        "doc_a",
        "tool",
        "args",
        "goal",
        "finish",
    ]
    backend = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=words[3:9],
    )
    result.chat_template = (
        "{% for message in messages %}{{ message['role'] + '\\n' + message['content'] + '\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}assistant\\n<think>\\n{% endif %}"
    )
    result.save_pretrained(tmp_path / "tokenizer")
    return result


def test_mask_eos_and_pair_length_boundary(tokenizer):
    prompt = {"system": "Use tools", "user": "visible evidence"}
    completion = "<think>read first</think><state>{}</state><action>finish</action>"
    encoded = encode_completion(tokenizer, prompt, completion + "[EOS][EOS]")
    assert encoded.input_ids[-1] == tokenizer.eos_token_id and encoded.mask[-1] == 1
    assert encoded.input_ids.count(tokenizer.eos_token_id) == 1
    assert encoded.mask[0] == 0 and sum(encoded.mask) > 1
    rows = [{"prompt": prompt, "chosen": completion, "rejected": completion}]
    kept, dropped = encode_pairs(rows, tokenizer, len(encoded.input_ids))
    assert len(kept) == 1 and dropped == 0
    kept, dropped = encode_pairs(rows, tokenizer, len(encoded.input_ids) - 1)
    assert kept == [] and dropped == 1
    batch = Collator(tokenizer.pad_token_id, True)(encode_pairs(rows, tokenizer, 512)[0])
    assert batch["input_ids"].shape[0] == 2
    assert not (batch["completion_mask"] & ~batch["attention_mask"].bool()).any()


def test_losses_match_direct_token_and_pair_averages(recipe):
    from repair.training.losses import preference_loss, reference_kl, sequence_logps

    torch.manual_seed(7)
    policy = torch.randn(4, 6, 11, requires_grad=True)
    reference = torch.randn(4, 6, 11, requires_grad=True)
    labels = torch.randint(0, 11, (4, 6))
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    logp, logq = policy.log_softmax(-1), reference.detach().log_softmax(-1)
    manual_p, manual_q, manual_kl = [], [], []
    for row in range(4):
        active = mask[row, 1:]
        manual_p.append(logp[row, :-1].gather(-1, labels[row, 1:, None]).squeeze(-1)[active].mean())
        manual_q.append(logq[row, :-1].gather(-1, labels[row, 1:, None]).squeeze(-1)[active].mean())
        manual_kl.append(
            (logp[row, :-1].exp() * (logp[row, :-1] - logq[row, :-1])).sum(-1)[active].mean()
        )
    p, q, kl = map(torch.stack, (manual_p, manual_q, manual_kl))
    assert torch.allclose(sequence_logps(policy, labels, mask), p)
    assert torch.allclose(reference_kl(policy, reference, mask, 2), kl, atol=1e-6)
    margin = 0.1 * (p[:2] - p[2:] - q[:2] + q[2:])
    expected = (
        -0.9 * torch.nn.functional.logsigmoid(margin)
        - 0.1 * torch.nn.functional.logsigmoid(-margin)
        - 0.1 * p[:2]
        + 0.2 * (kl[:2] + kl[2:]) / 2
    ).mean()
    settings = {**recipe["training"], "kl_chunk_tokens": 2}
    actual, _ = preference_loss(policy, reference, labels, mask, settings)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()
    assert reference.grad is None


@pytest.mark.training
@pytest.mark.parametrize(
    "recipe_name",
    [
        "A3_induced_14b.toml",
        "A1_warm_start_14b.toml",
        "B2_trajectory_sft_14b.toml",
        "B3_chosen_sft_14b.toml",
    ],
)
def test_small_model_update_save_reload_and_resume(tmp_path, tokenizer, recipe_name):
    config = load_experiment(Path(__file__).parents[1] / "configs/experiments" / recipe_name)
    method, mode = config["training"]["method"], config["training"]["train_mode"]
    if mode == "lora":
        pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    from repair.training.run import train

    torch.set_num_threads(1)
    source = tmp_path / "base"
    source.mkdir()
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=512,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    embedding_before = model.get_input_embeddings().weight.detach().clone()
    model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    config["training"].update(learning_rate=0.001, max_length=512)
    if mode == "lora":
        config["training"].update(lora_rank=2, lora_alpha=4)
    prompt = {"system": "Use tools", "user": "visible evidence"}
    chosen = "<think>read first</think><state>{}</state><action>read doc_a</action>"
    rejected = "<think>search next</think><state>{}</state><action>search wrong</action>"
    rows = (
        [{"prompt": prompt, "chosen": chosen, "rejected": rejected}] * 2
        if method == "dpo"
        else [{"prompt": prompt, "completion": chosen}] * 2
    )
    data_path = tmp_path / "samples.jsonl"
    write_jsonl(data_path, rows)
    summary = train(config, data_path, str(source), tmp_path / "output", smoke=True)
    assert summary["optimizer_steps"] == 1 and summary["smoke"]
    assert (tmp_path / "output/checkpoint-0001/training_state.pt").exists()
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path / "output/model")
    assert any(not torch.equal(value, loaded.state_dict()[name]) for name, value in before.items())
    assert all(torch.isfinite(value).all() for value in loaded.state_dict().values())
    assert summary["train_mode"] == mode
    checkpoint = tmp_path / "output/checkpoint-0001"
    assert (checkpoint / "adapter_config.json").exists() == (mode == "lora")
    if mode == "full":
        assert (checkpoint / "config.json").exists()
        assert not torch.equal(embedding_before, loaded.get_input_embeddings().weight)
    assert not (tmp_path / "output/model/training_state.pt").exists()
    resumed = train(
        config, data_path, str(source), tmp_path / "resumed", smoke=True, resume=checkpoint
    )
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "resumed/model")
    assert resumed["optimizer_steps"] == summary["optimizer_steps"]
    assert all(
        torch.equal(value, reloaded.state_dict()[name])
        for name, value in loaded.state_dict().items()
    )
