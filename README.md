# RePair

Code for **RePair: Rubric-Guided Turn-Level Preference Learning for Tool-Using Agents**.

RePair diagnoses failed tool-use trajectories, assigns general rubrics to individual turns,
samples and validates replacements, and trains on the resulting preferences. The repository
includes handwritten and induced codebook experiments, two rounds of learner-generated repair,
SFT controls, rubric-subset ablations, evaluation, and token accounting.

The experiment recipes follow the paper's experimental setup. Data is distributed separately
as `RePair-data.zip`. The archive contains the `data/` directory with the question splits
(identifiers only), warm-start demonstrations, fixed codebooks, A2–A7 preferences, and a
losslessly compressed retrieval index. Model weights and external model services are separate.

## Data attachment

Download `RePair-data.zip` from <DATA_URL> and place it next to the code directory.
BrowseComp-Plus asks that its questions and answers never appear as plain text online, so the
archive is password-protected and contains no question files. Extract it from the repository
root with the password `RePair-BrowseComp-Plus`, then recover the questions from the official
release:

```bash
unzip -P RePair-BrowseComp-Plus ../RePair-data.zip -d .
python -m pip install -e ".[questions]"
repair questions
```

This creates `data/` alongside `configs/`, `experiments/`, and `src/`. `repair questions`
downloads [Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus),
decrypts it with the benchmark's canary, and writes `data/questions/{warm_start,training,test}.jsonl`
in the order of `data/splits.json`. `sha256sum -c data/SHA256SUMS` checks every file,
the recovered questions included. The default paths in `configs/runtime.example.toml` then point to the extracted
files. See [data contents](docs/data.md). Please do not redistribute the extracted files.

## Installation

Use Python 3.10 or later. Training requires a CUDA-enabled PyTorch installation. Install
PyTorch for your NVIDIA driver and CUDA environment before installing the training extra.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/tested-constraints.txt -e ".[retrieval,induction,train,test]"
```

The main preference recipe uses 16 H200 GPUs, FSDP, and a global batch of 64.
CPU and smaller GPU runs are available for verification through `--smoke`; they are not
the paper training configuration. Model serving can use a separate environment.

## Configuration

Copy `configs/runtime.example.toml` to `configs/runtime.toml`. Supply local input paths,
model locations, and accessible model services. Relative paths resolve from the repository
root when the runtime file is stored in `configs/`. Export the credential variables named
in that file. The application does not load `.env` files automatically.

The role names in the recipes identify the paper's models. Endpoint availability and request
extensions depend on the service you provide. Set thinking controls explicitly. Policy
services must support the configured repetition penalty; unsupported settings are errors.
Using `--allow-model-override` records a different model configuration in the stage manifest.

```bash
repair check \
  --experiment configs/experiments/A3_induced_14b.toml \
  --runtime configs/runtime.toml
```

This checks settings and prints the available stages without calling a model or loading weights.
See [input formats](docs/input_formats.md), [configuration](docs/configuration.md), and
[experiment recipes](docs/experiments.md).
The [main-experiment input checklist](docs/main_experiment_inputs.md) lists shared inputs and
the artifacts needed for the five main experiment groups.

## Training from the included data

Set the runtime paths from `configs/runtime.example.toml`. To rebuild π0, run `prepare-sft`
and `train` for A1 or A1_8b using the included warm-start demonstrations. Alternatively, point
`artifacts.A1` and `artifacts.A1_8b` to existing merged models.

The A2–A7 `train` stages use the included preference files when no newly generated pair file
exists for that arm. A4 is trained before A5 because A5 starts from A4's exported model.
Evaluation uses the recovered test questions and the included index. See [data contents](docs/data.md)
for file formats and the scope of reproduction.

## Running an experiment

Each script in `experiments/` selects one recipe. Run stages in the order printed by `repair check`.
Inputs must exist before their dependent stages can run.

```bash
bash experiments/A3.sh --runtime configs/runtime.toml --stage collect
bash experiments/A3.sh --runtime configs/runtime.toml --stage judge
bash experiments/A3.sh --runtime configs/runtime.toml --stage diagnose
bash experiments/A3.sh --runtime configs/runtime.toml --stage induce
bash experiments/A3.sh --runtime configs/runtime.toml --stage reassign
bash experiments/A3.sh --runtime configs/runtime.toml --stage repair
```

A2 and A3 share the same 14B collection; A6 and A7 share the same 8B collection.
Self-play collections are separate. Stage manifests reject incompatible settings or inputs.
Completed collection and per-trajectory processing records are reused on subsequent invocations.

For preference training on two machines with eight GPUs each, run this on both machines with
the same `MASTER_ADDR` and `MASTER_PORT`, setting `NODE_RANK` to 0 or 1. Models, inputs, and
outputs must be accessible at the same paths on both machines.

```bash
torchrun \
  --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m repair run \
  --experiment configs/experiments/A3_induced_14b.toml \
  --runtime configs/runtime.toml --stage train
```

Load `outputs/A3/training/model` into your policy service, then set
`services.policy.artifact = "A3"` before evaluation:

```bash
bash experiments/A3.sh --runtime configs/runtime.toml --stage evaluate
```

The artifact field declares which checkpoint is served. The server must actually load that
checkpoint; the client cannot inspect remote weights. Serving, training, and stage selection
are explicit operations.

## Source layout

```text
src/repair/
  harness/      # Context, retrieval, and rollout
  pairs/        # Diagnosis, codebooks, repair, and training views
  training/     # Tokenization, losses, and distributed training
  evaluation.py
  costs.py
  experiment.py
  cli.py
```

Tests create synthetic questions, trajectories, and small models locally. The data bundle is
kept separate from the Python source distribution and wheel.

```bash
python -m pytest
ruff check src tests
python -m build
```

This release does not claim a rerun of the paper's experiments. GPU memory requirements,
multi-node training, and external service compatibility must be validated on the target setup.
