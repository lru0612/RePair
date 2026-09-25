"""Completion-only masks and whole-example length filtering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Encoded:
    input_ids: list[int]
    mask: list[int]


def encode_completion(tokenizer, prompt: dict, completion: str) -> Encoded:
    messages = [{"role": role, "content": prompt[role]} for role in ("system", "user")]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    if prefix.rstrip().endswith("<think>") and completion.lstrip().startswith("<think>"):
        completion = completion.lstrip()[len("<think>") :]
        if prefix.endswith("\n") and completion.startswith("\n"):
            completion = completion[1:]
    encoded = tokenizer(prefix + completion, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    mask = [int(end > len(prefix) and end > start) for start, end in encoded["offset_mapping"]]
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("The tokenizer must define EOS")
    while ids and ids[-1] == eos:
        ids.pop()
        mask.pop()
    ids.append(eos)
    mask.append(1)
    if not any(mask[:-1]):
        raise ValueError("An empty completion cannot be trained")
    return Encoded(ids, mask)


def encode_pairs(rows: list[dict], tokenizer, limit: int) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for row in rows:
        chosen = encode_completion(tokenizer, row["prompt"], row["chosen"])
        rejected = encode_completion(tokenizer, row["prompt"], row["rejected"])
        if max(len(chosen.input_ids), len(rejected.input_ids)) > limit:
            dropped += 1
            continue
        kept.append({"chosen": chosen, "rejected": rejected})
    return kept, dropped


def encode_sft(rows: list[dict], tokenizer, limit: int, packing: bool) -> tuple[list[Encoded], int]:
    samples, dropped = [], 0
    for row in rows:
        sample = encode_completion(tokenizer, row["prompt"], row["completion"])
        if len(sample.input_ids) > limit:
            dropped += 1
        else:
            samples.append(sample)
    if not packing:
        return samples, dropped
    packed, ids, mask = [], [], []
    for sample in samples:
        if ids and len(ids) + len(sample.input_ids) > limit:
            packed.append(Encoded(ids, mask))
            ids, mask = [], []
        ids.extend(sample.input_ids)
        mask.extend(sample.mask)
    if ids:
        packed.append(Encoded(ids, mask))
    return packed, dropped


class Collator:
    def __init__(self, pad_token_id: int, preference: bool):
        self.pad_token_id, self.preference = pad_token_id, preference

    def __call__(self, rows: list) -> dict:
        import torch

        samples = (
            [row["chosen"] for row in rows] + [row["rejected"] for row in rows]
            if self.preference
            else rows
        )
        width = max(len(sample.input_ids) for sample in samples)
        ids = torch.full((len(samples), width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for number, sample in enumerate(samples):
            length = len(sample.input_ids)
            ids[number, :length] = torch.tensor(sample.input_ids)
            attention[number, :length] = 1
            mask[number, :length] = torch.tensor(sample.mask, dtype=torch.bool)
        return {"input_ids": ids, "attention_mask": attention, "completion_mask": mask}
