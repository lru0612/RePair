# Configuration details

Experiment TOML files contain complete method settings. Runtime TOML provides external inputs,
model locations, services, and request concurrency. No historical configuration overlays are loaded.

## Paper settings and implementation defaults

The supplied specification fixes the main recipe but leaves some engineering choices unspecified.
The following defaults are explicit in the source or recipes; they are not claimed as historical
experimental settings.

| Unspecified detail | Public implementation |
| --- | --- |
| Initial diagnostic finding limit | Eight findings per failed trajectory |
| Diagnostic/rewrite/consolidation decoding | Thinking enabled, temperature 1.0, top-p 0.95, 12288-token limit |
| Validator decoding | Thinking enabled, temperature 0, 4096-token limit |
| Truncation retry counting | Initial generation plus five retries |
| Retry repetition penalties | Initial 1.1; retries 1.1, 1.1, 1.2, 1.3, 1.4 |
| Equal-frequency rubric ranking | Original codebook order |
| Adafactor options | Fixed learning rate; relative_step=false, scale_parameter=false, warmup_init=false, clip_threshold=1 |
| Weight decay and gradient clipping | Weight decay 0; maximum gradient norm 1 |
| FSDP sharding | FULL_SHARD, use_orig_params=true, decoder-layer wrapping |
| Attention | PyTorch SDPA |
| SFT train mode | LoRA for A1/A1_8b warm start; full-parameter fine-tuning for B2/B3 |
| SFT optimizer | AdamW, betas (0.9, 0.999), epsilon 1e-8 |
| B2/B3 schedule | Cosine decay with 10% warmup |
| SFT process layout | Eight processes; accumulation 4 for warm start and 2 for B2/B3 |
| Checkpoints | Save complete training state after each epoch |
| Training framework | PyTorch distributed training with Transformers models and PEFT adapters |
| Distributed final batches | DistributedSampler pads to equal shard sizes; the last accumulation group may be smaller |
| Reader document window | First 24000 tokens of the parent document |
| Finish candidate matching | Equality after case, accent, whitespace, and punctuation normalization |
| Packing | Whole SFT examples concatenated into causal sequences without splitting examples |

The programmatic finish check is conservative. Semantic equivalence, spelling variants, and
numerical tolerance are handled by the final outcome judge. Finish grounding requires at least
half of the answer's words of length four or more, and at least one such word, to occur in the
visible input. An answer without any such words fails this candidate check.

The corpus manifest records the chunker and tokenizer settings. A tokenizer or corpus revision
can change chunk boundaries. The supplied fixed index and codebooks are loaded without rebuilding them.

## Service configuration

`backend = "vllm"` sends `chat_template_kwargs.enable_thinking` and repetition penalty regardless
of the service's hostname. `backend = "chat"` uses the configured `thinking_field` or an explicit
`thinking_always_on` declaration. Additional service-specific fields can be supplied under
`request_fields`; they cannot replace sampling settings, model identity, messages, or seed.

Diagnosis, rewriting, and reassignment fall back to `services.teacher`. Validation falls back
to `services.reader`. Each role can instead have its own service entry.

Before a policy collection or evaluation, set `services.policy.artifact` to the expected source:
`base_14b`, `base_8b`, `A1`, `A1_8b`, `A4`, or the arm being evaluated. Load the matching weights
into the service. This field prevents accidental stage selection with a declared wrong checkpoint;
it is not remote weight verification.

For evaluation, the client omits per-call `max_tokens`. Configure the server so that it does not
apply a smaller default generation cap. Transport retries are separate from the five truncation
retries, and every request with known usage is recorded.

Changing a teacher model requires `--allow-model-override`. This is recorded in the stage
manifest. A replacement service with a different model is a new configuration.

## Paths and continuation

Store runtime configuration in `configs/runtime.toml` to resolve `./data`, `./models`, and
`./outputs` relative to the repository root. Credentials are read only from named environment
variables and are excluded from resolved configuration output.

Outputs use stable query/sample identities. An existing stage with a different configuration
or input hash is rejected. Use a new output root for a new experiment. The evaluation timer is
persisted, so interruption and continuation do not reset the six-hour deadline.

Small verification runs use `--smoke`, write under `smoke-training`, perform one optimizer
step with no warmup, and may use CPU float32. They do not replace the paper artifacts under
`training/model`.
