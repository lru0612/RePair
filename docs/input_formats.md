# Input formats

Inputs are UTF-8 JSON or JSONL, with optional gzip compression for JSONL files. The examples
below are synthetic format examples. See [data contents](data.md) for the included inputs.
Model weights and raw trajectory collections are separate.

## Questions and corpus

Provide separate `warm_start_questions`, `training_questions`, and `test_questions` files.
The expected sizes are 199, 465, and 166. Question IDs must be unique within each split.
When split files are available, the loader checks IDs and normalized question text for overlap.

```json
{"id":"example_q","question":"Which fictional town is named in the document?","answer":"Northvale"}
```

The reference answer is used by the outcome judge and programmatic candidate checks. It is
not included in policy, diagnosis, reassignment, repair, reader, or validator requests.

Corpus JSONL uses one parent document per line:

```json
{"docid":"example_doc","title":"Example source","text":"# Location\nThe fictional town is Northvale."}
```

```bash
repair index \
  --experiment configs/experiments/A3_induced_14b.toml \
  --corpus data/corpus.jsonl --output data/index \
  --tokenizer Qwen/Qwen3-14B
```

The index stores parent texts, chunks, BM25 data, and a manifest. BM25 uses Lucene scoring,
k1=1.5, b=0.75, English stopwords, and stemming. Heading sections are token-windowed with
title prefixes, at most 1024 tokens per chunk, and 128-token overlap within a section.
The generated index records its actual document and chunk totals; it does not force the
reported corpus totals onto a different input corpus.

## Codebooks

Handwritten and generated codebooks have the same structure:

```json
{
  "rubrics": [
    {
      "id": "example-rule",
      "trigger": "A relevant result is visible but remains unread.",
      "hint": "Read the relevant result before reformulating the search.",
      "criterion": "The candidate reads a relevant visible document."
    }
  ]
}
```

The handwritten experiment requires the IDs `query-churn`, `search-without-read`, and
`over-specified-query`. Their full text is in `data/codebooks/handwritten.json`.
The synthetic rule above is not a replacement for that codebook.

Induction writes `outputs/<arm>/induce/codebook.json`. A4 and A5 read A3's fixed codebook;
A7 induces its own codebook. When no generated codebook exists, the reader uses the fixed
codebook under `paths.codebooks_dir`. Rubric IDs are stable lowercase kebab-case strings.

## Demonstrations and preferences

Warm-start JSONL contains complete assistant turns and their original system/user inputs:

```json
{"id":"example_turn","dataset":"browsecomp_plus","query_id":"example_q","prompt":{"system":"Use the tools.","user":"Read the visible evidence."},"completion":"<think>Read the result.</think>\n<state>{}</state>\n<action>{\"tool\":\"get_document\",\"args\":{\"docid\":\"example_doc\",\"goal\":\"find the location\"}}</action>"}
```

`dataset` identifies `browsecomp_plus`, `patents`, `web`, or `web_simple`. The complete warm-start
input must include all four sources. BrowseComp-Plus samples must belong to the warm-start split.
Auxiliary demonstrations must have no test-question overlap. The bundle includes prepared
demonstrations; raw auxiliary corpora are separate.

Use `--stage prepare-sft` to validate and copy the supplied or external demonstrations into an arm's
training view. The optional warm-start `collect` stage gathers teacher trajectories for the
main warm-start split only.

Generated pair JSONL contains:

```json
{"id":"example_pair","trajectory_id":"example_q:0","query_id":"example_q","turn":0,"rubric_id":"example-rule","prompt":{"system":"Use the tools.","user":"Read the visible evidence."},"chosen":"<think>Read first.</think><state>{}</state><action>{\"tool\":\"get_document\",\"args\":{\"docid\":\"example_doc\",\"goal\":\"find the location\"}}</action>","rejected":"<think>Search again.</think><state>{}</state><action>{\"tool\":\"search\",\"args\":{\"query\":\"example query\"}}</action>"}
```

`repair/accepted.jsonl` preserves accepted candidates before pair-length filtering.
`repair/pairs.jsonl` contains pairs within the preference length limit. B3 reads the former;
preference training and C1–C3 use the latter. The repaired completion is rendered in the
policy turn format; the rejected completion preserves the recorded response.

## Checkpoints and outputs

An epoch checkpoint contains model or adapter weights, tokenizer files, optimizer and scheduler
state, RNG states, and completed epoch/step counters. Resume with the same recipe and process count:

```bash
torchrun --nproc_per_node=16 -m repair run \
  --experiment configs/experiments/A3_induced_14b.toml \
  --runtime configs/runtime.toml --stage train \
  --resume outputs/A3/training/checkpoint-0002
```

`training/model` contains a complete inference model, including merged weights after LoRA SFT.
It is not a checkpoint for restoring optimizer state.

## Budget estimates

The D2 `budget_inputs` JSON file must provide:

```json
{"rollouts":4,"mean_prefill_tokens":100,"mean_generated_tokens":20,"assumptions":"Synthetic format example; not paper measurements."}
```

The calculator multiplies rollout count by each mean. Replace these synthetic values with
explicitly justified inputs before using the estimate.
