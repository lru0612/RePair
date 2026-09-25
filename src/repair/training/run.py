"""Full-parameter preference training and full-parameter or LoRA SFT with torchrun."""

from __future__ import annotations

import functools
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

from repair.config import validate_world_size
from repair.io import read_jsonl, write_json
from repair.training.data import Collator, encode_pairs, encode_sft
from repair.training.losses import preference_loss, sequence_logps


def _distributed(settings: dict, smoke: bool):
    import torch
    import torch.distributed as dist

    world = validate_world_size(settings, smoke=smoke)
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif smoke and world == 1:
        device = torch.device("cpu")
    else:
        raise RuntimeError("Training requires CUDA; CPU is available only with --smoke")
    if world > 1:
        dist.init_process_group("nccl")
    return device, rank, world


def _wrap(model, device, world: int, dtype):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    model.to(device)
    if world == 1:
        return model
    layer_names = set(getattr(model, "_no_split_modules", []) or [])
    classes = {type(module) for module in model.modules() if type(module).__name__ in layer_names}
    if not classes:
        raise ValueError("The model must declare decoder layers in _no_split_modules")
    return FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=classes
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=device,
        sync_module_states=True,
        mixed_precision=MixedPrecision(
            param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=dtype
        ),
        limit_all_gathers=True,
    )


def _checkpoint(
    model,
    optimizer,
    scheduler,
    tokenizer,
    folder: Path,
    *,
    epoch: int,
    steps: int,
    rank: int,
    world: int,
):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )

    manager = (
        FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )
        if world > 1
        else nullcontext()
    )
    with manager:
        weights = model.state_dict()
        optim = FSDP.optim_state_dict(model, optimizer) if world > 1 else optimizer.state_dict()
    rng = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    all_rng = [None] * world
    if world > 1:
        dist.all_gather_object(all_rng, rng)
    else:
        all_rng[0] = rng
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if world > 1 else model
        unwrapped.save_pretrained(folder, state_dict=weights, safe_serialization=True)
        tokenizer.save_pretrained(folder)
        torch.save(
            {
                "optimizer": optim,
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "steps": steps,
                "rng": all_rng,
                "world_size": world,
            },
            folder / "training_state.pt",
        )
    if world > 1:
        dist.barrier()


def train(
    config: dict,
    data_path: Path,
    source: str,
    output: Path,
    *,
    smoke: bool = False,
    resume: Path | None = None,
) -> dict:
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import (
        Adafactor,
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    settings = config["training"]
    device, rank, world = _distributed(settings, smoke)
    random.seed(config["seed"] + rank)
    torch.manual_seed(config["seed"] + rank)
    preference = settings["method"] == "dpo"
    use_lora = settings["train_mode"] == "lora"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Completion masks require a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = list(read_jsonl(data_path))
    limit = settings["max_length"]
    if preference:
        dataset, dropped = encode_pairs(rows, tokenizer, limit)
    else:
        dataset, dropped = encode_sft(rows, tokenizer, limit, settings["packing"])
    if not dataset:
        raise ValueError("No examples remain after length filtering")
    model_source = str(resume) if resume and not use_lora else source
    load_kwargs = {"torch_dtype": dtype, "attn_implementation": settings["attention"]}
    policy = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
    policy.config.use_cache = False
    reference = None
    if preference:
        reference = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        reference.requires_grad_(False)
        reference.eval()
    elif use_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if resume:
            policy = PeftModel.from_pretrained(policy, resume, is_trainable=True)
        else:
            policy = get_peft_model(
                policy,
                LoraConfig(
                    r=settings["lora_rank"],
                    lora_alpha=settings["lora_alpha"],
                    lora_dropout=settings["lora_dropout"],
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=settings["lora_targets"],
                ),
            )
    if not use_lora:
        policy.requires_grad_(True)
    if settings["activation_checkpointing"]:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if use_lora:
            policy.enable_input_require_grads()
    policy = _wrap(policy, device, world, dtype)
    if reference is not None:
        reference = _wrap(reference, device, world, dtype)
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if preference:
        optimizer = Adafactor(
            parameters,
            lr=settings["learning_rate"],
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=settings["weight_decay"],
            clip_threshold=settings["adafactor_clip_threshold"],
        )
    else:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=config["seed"])
    loader = DataLoader(
        dataset,
        batch_size=settings["per_device_batch"],
        sampler=sampler,
        collate_fn=Collator(tokenizer.pad_token_id, preference),
        num_workers=0,
    )
    accumulation = settings["gradient_accumulation"]
    epochs = 1 if smoke else settings["epochs"]
    total_steps = math.ceil(len(loader) / accumulation) * epochs
    if smoke:
        total_steps = 1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0 if smoke else math.ceil(total_steps * settings["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    start_epoch, steps = 0, 0
    if resume:
        state_path = resume / "training_state.pt"
        if not state_path.exists():
            raise ValueError("A weights-only directory is not a training checkpoint")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["world_size"] != world:
            raise ValueError("Checkpoint world size differs from this training run")
        optim = (
            FSDP.optim_state_dict_to_load(policy, optimizer, state["optimizer"])
            if world > 1
            else state["optimizer"]
        )
        optimizer.load_state_dict(optim)
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, steps = state["epoch"], state["steps"]
        rng = state["rng"][rank]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        policy.train()
        for number, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            mask = batch.pop("completion_mask")
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
            ):
                logits = policy(**batch).logits
                if preference:
                    with torch.no_grad():
                        ref_logits = reference(**batch).logits
                    loss, metrics = preference_loss(
                        logits, ref_logits, batch["input_ids"], mask, settings
                    )
                    del ref_logits
                else:
                    loss = -sequence_logps(logits, batch["input_ids"], mask).mean()
                    metrics = {"nll": loss.detach()}
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            start_group = number // accumulation * accumulation
            divisor = min(accumulation, len(loader) - start_group)
            (loss / divisor).backward()
            del logits
            boundary = (number + 1) % accumulation == 0 or number + 1 == len(loader)
            if boundary:
                if world > 1:
                    policy.clip_grad_norm_(settings["max_grad_norm"])
                else:
                    torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if rank == 0:
                    write_json(
                        output / "progress.json",
                        {
                            "epoch": epoch + 1,
                            "step": steps,
                            "metrics": {name: float(value) for name, value in metrics.items()},
                        },
                    )
                if smoke:
                    break
        checkpoint_dir = output / f"checkpoint-{epoch + 1:04d}"
        _checkpoint(
            policy,
            optimizer,
            scheduler,
            tokenizer,
            checkpoint_dir,
            epoch=epoch + 1,
            steps=steps,
            rank=rank,
            world=world,
        )
    elapsed = time.monotonic() - started
    if start_epoch >= epochs:
        checkpoint_dir = resume
    del optimizer, policy, reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if rank == 0:
        final = output / "model"
        if not use_lora:
            # A checkpoint contains optimizer state; the exported model contains only inference files.
            import shutil

            final.mkdir(parents=True, exist_ok=True)
            for path in checkpoint_dir.iterdir():
                if path.name != "training_state.pt" and path.is_file():
                    shutil.copy2(path, final / path.name)
        else:
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype)
            merged = PeftModel.from_pretrained(base, checkpoint_dir).merge_and_unload()
            merged.save_pretrained(final, safe_serialization=True)
            tokenizer.save_pretrained(final)
        summary = {
            "method": settings["method"],
            "train_mode": settings["train_mode"],
            "smoke": smoke,
            "world_size": world,
            "epochs_completed": epochs,
            "optimizer_steps": steps,
            "examples_loaded": len(rows),
            "examples_dropped": dropped,
            "training_seconds": elapsed,
            "accelerator_hours": elapsed * world / 3600 if device.type == "cuda" else 0,
            "device": device.type,
        }
        write_json(output / "training_summary.json", summary)
    else:
        summary = {}
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return summary
