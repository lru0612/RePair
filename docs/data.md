# Data contents

Data is provided separately in `RePair-data.zip`. Extract it into the repository root so
that `data/` sits alongside `src/` and `configs/`. The code repository does not bundle these
data files.

The data bundle contains only inputs needed for training, retrieval, and evaluation:

```text
data/
  questions/
    warm_start.jsonl
    training.jsonl
    test.jsonl
  splits.json
  warm_start/
    sft.jsonl.gz
  pairs/
    A2.jsonl.gz
    A3.jsonl.gz
    A4.jsonl.gz
    A5.jsonl.gz
    A6.jsonl.gz
    A7.jsonl.gz
  codebooks/
    handwritten.json
    induced_14b.json
    induced_8b.json
  index/
    settings.json
    documents.sqlite
    bm25/
      parameters.json
      scores.npz
      vocabulary.json.gz
```

JSONL files may be gzip-compressed; the loader reads them directly. Gzip headers omit the
original filename and creation time.

## Retained fields

| Input | Fields |
| --- | --- |
| Question | `id`, `question`, `answer` |
| Split | Lists of question IDs under `warm_start`, `training`, and `test` |
| Warm-start sample | `dataset`, `query_id`, `prompt.system`, `prompt.user`, `completion` |
| Preference | `query_id`, `rubric_id`, `prompt.system`, `prompt.user`, `chosen`, `rejected` |
| Rubric | `id`, `trigger`, `hint`, `criterion` |

Question and rubric IDs preserve the associations needed for split checks and codebook use.
Dataset names identify the four warm-start sources. Generation timestamps, model annotations,
source paths, request details, processing statistics, and redundant parsed completions are
not retained. Task content, including dates that occur in questions or documents, is preserved.

Preferences retain their original text and order. They are not resampled or filtered during
metadata removal. Training applies the configured whole-pair length limit and builds its
own completion masks, including supervised EOS.

## Retrieval storage

The index preserves the original vocabulary, document ordering, sparse BM25 score arrays,
parent documents, and chunk text. NumPy arrays use lossless ZIP compression. Document text
uses zlib compression, with parent-text dictionaries for chunks. The loader decodes only
the documents needed by a request.

`documents.sqlite` contains only `parents` and `chunks` tables. Index settings and numerical
array types are retained because the retrieval implementation needs them. Source manifests,
creation information, analysis statistics, and source-location fields are omitted.

The sparse arrays are loaded into RAM from `scores.npz`; allow roughly two GiB of host memory
for the arrays, in addition to the vocabulary and the application's other memory needs.

## Reproduction scope

These inputs support warm-start training, training the six A2–A7 arms, and evaluating their
resulting models. Base or merged starting-model weights and accessible reader/judge services
are also required. Starting from supplied π0 weights avoids rebuilding the warm starts.

The bundle contains fixed, precomputed preferences. Removing metadata does not regenerate
them under the current model-service settings. In particular, the A6 data has not been
revalidated with the validator selected in the current recipe. The bundled inputs therefore
do not establish a rerun of every data-construction stage or guarantee identical historical
scores.

Raw learner collections, diagnostic findings, complete assignments, and candidate-validation
logs are outside this bundle. Reconstructing codebook induction and preference generation from
scratch requires running those stages with the corresponding starting models and services.
