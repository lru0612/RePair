# Validation

Validation date: September 24, 2026.

The automated suite passes 36 tests using synthetic inputs. It covers:

- All 18 experiment recipes and round-specific starting policies.
- Context bounds, eligible turns, candidate checks, and rubric reassignment.
- Prompt and reference-answer isolation, no-rubric repair, and assignment-count ranking.
- Truncation retries, preserved evaluation deadlines, and complete-sample metric denominators.
- BM25 index construction and retrieval, plus weighted TF-IDF/SVD/K-means clustering.
- Collection, grading, reassignment, pair construction, and chosen-SFT view generation.
- Completion masks, EOS, whole-pair length filtering, and numerical loss/gradient checks.
- One actual CPU optimizer update for both DPO and LoRA SFT using a tiny synthetic Qwen3 model,
  followed by checkpoint saving, adapter merging where applicable, and model reloading.
- Compressed JSONL input, bundled-data resolution, dictionary-compressed Unicode documents,
  and identical retrieval through plain and compressed index storage.

The test environment used Python 3.10.12, PyTorch 2.12.0+cu132 with CUDA unavailable,
Transformers 4.51.3, PEFT 0.15.2, and Accelerate 1.6.0. Additional tested versions are recorded
in `requirements/tested-constraints.txt`. Install PyTorch separately for the target hardware.

The source distribution and wheel build successfully. The wheel was installed in a new
virtual environment without the source tree or its optional training dependencies. The
installed command loaded and validated the A3 recipe from an unrelated working directory.
The source and package contents were checked for private paths, credentials, internal
service references, legacy imports, and unintended data or weight files.

The supplied data was checked separately. Every retained preference and demonstration
input/target was compared with its source text. Question identities and split isolation
were checked, including the auxiliary warm-start sources. All parent and chunk text passed
lossless round-trip checks, and the sparse score arrays were compared element by element.
Real-query checks matched the original index's document IDs, ranking, scores, and snippets.
The final copied files matched the validated staging files byte for byte. Record fields,
compression headers, and database tables were checked against the data-only allowlist.

The tests did not run 8B/14B training, a full-length 20480-token GPU batch, multi-node FSDP,
external model services, or the full benchmark. The data bundle contains existing inputs
with processing metadata removed; no new model-generated training data or weights were created.
