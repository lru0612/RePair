# Experiment recipes

All settings below follow the experiment specification dated September 24, 2026.
Each row has a recipe under `configs/experiments/` and an entry script under `experiments/`.
A0_8b and A1_8b correspond to A0′ and A1′ in the experiment inventory.

| Arm | Experiment | Start | Collection | Codebook / repair |
| --- | --- | --- | --- | --- |
| A0 | 14B base evaluation | Qwen3-14B | None | None |
| A0_8b | 8B base evaluation | Qwen3-8B | None | None |
| A1 | 14B warm start | Qwen3-14B | External warm-start demonstrations | LoRA SFT, then merge |
| A1_8b | 8B warm start | Qwen3-8B | External warm-start demonstrations | LoRA SFT, then merge |
| A2 | 14B handwritten | A1 | Once per training question | Three handwritten rubrics; teacher repair |
| A3 | 14B induced | A1 | Shared with A2 | Independently induced 14B codebook; teacher repair |
| A4 | Self-play round 1 | A1 | Four times per training question | A3 codebook; A1 repairs |
| A5 | Self-play round 2 | A4 | Four fresh runs per training question | A3 codebook; A4 repairs |
| A6 | 8B handwritten | A1_8b | Once per training question | Same three handwritten rubrics; teacher repair |
| A7 | 8B induced | A1_8b | Shared with A6 | Independently induced 8B codebook; teacher repair |
| B1 | No-rubric repair | A1 | Shared with A2/A3 | Diagnose turns; no guidance or model validator |
| B2 | Full-trajectory SFT | A1 | Teacher solves each training question four times | Legal turns from correct trajectories |
| B3 | Chosen-only SFT | A1 | Accepted A3 repairs | Chosen completions only |
| C1 | Top-1 rubric | A1 | A3 pairs | Highest assignment count |
| C2 | Top-5 rubrics | A1 | A3 pairs | Five highest assignment counts |
| C3 | Top-10 rubrics | A1 | A3 pairs | Ten highest assignment counts |
| D1 | Token accounting | A3 artifacts | No new collection | Phase totals and accelerator hours |
| D2 | GRPO budget estimate | Explicit budget inputs | No GRPO run | Rollout-count × mean-token arithmetic |

The reported induced codebooks contain 25 rubrics for 14B and 14 for 8B. New induction may
produce a different number. The code records the difference and does not truncate the output
to manufacture the expected size. C1 checks that the top rubric is `failed-search-repeat`.

## Models

| Role | Model |
| --- | --- |
| Learner | Qwen3-14B or Qwen3-8B |
| Warm-start teacher | deepseek-v4-pro |
| Diagnosis, cluster rewriting, reassignment, teacher repair | kimi-k3 |
| Final codebook consolidation | gpt-5.6-sol |
| Reader and validator | deepseek-v4-flash |
| Outcome judge | gpt-oss-120b |

A6 uses **deepseek-v4-flash**, resolving the conflicting validator entries in the inventory.
B1 follows the full-parameter preference recipe in its optimization section.

## Training

Preference training uses cDPO with β=0.1 and label smoothing ε=0.1, chosen NLL weight 0.1,
and KL(policy‖reference) weight 0.2. Each completion's score and KL are averaged over its
supervised tokens; chosen and rejected KL terms receive equal weight. The prompt is masked,
and reasoning, state, action, and a single appended EOS are supervised.

The common preference recipe is full-parameter FSDP training on 16 H200 GPUs, one pair per
GPU, gradient accumulation 4, global batch 64, Adafactor, learning rate 1e-6, cosine decay,
10% warmup, bf16, seed 12345, and four epochs. Examples exceeding 20480 tokens on either
side, including the prompt and EOS, are dropped in full. The reference is frozen at the
round's start: A1 for A4, and A4 for A5.

Warm-start SFT uses LoRA rank/alpha 64/128 for 14B and 32/64 for 8B, dropout 0.05, and all
attention and MLP projections. Learning rate is 1e-4 with cosine decay and 10% warmup,
packing length 16384, global batch 32, and four epochs. Merge the adapter to obtain π0.
The demonstrations cover the 199-question warm-start split and the patents,
web, and web_simple auxiliary corpora.

B2 and B3 fine-tune all 14B model parameters starting from A1, using AdamW and FSDP
full sharding. They save full model checkpoints and export weights directly.
They use learning rate 1e-4, no packing,
length limit 20480, global batch 16, and four epochs. B3 applies its own SFT length check
to accepted chosen turns. It does not inherit rejection caused by the longer rejected side.

## Evaluation and cost

Evaluation uses the 166 held-out questions with three samples per question. Acc includes
all expected runs. Missing, timed-out, truncated, and otherwise failed runs are incorrect.
Pass@3 measures the fraction of questions with a correct sample. Behavior statistics use
completed trajectories, excluding execution failures.

Sampling jobs have a six-hour deadline. Resuming a job preserves its original deadline;
it does not grant another six hours. Per-call policy and reader generation caps are omitted
for evaluation. The context window, finish threshold, and turn limit still apply.

D1 counts input and generated tokens for collection, induction, reassignment, all repair
candidates, and validation. Collection includes the outcome judge and excludes the reader.
Unknown token usage is marked explicitly. Training duration is reported as accelerator-hours.

The supplied specification does not contain D2's full paper conversion formula or numerical
assumptions. The included calculator reports only explicitly supplied rollout-count × mean-token
estimates, with an assumptions field. It does not estimate optimizer work or claim a measured
GRPO run.
