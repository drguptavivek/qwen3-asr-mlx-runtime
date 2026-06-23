# Evaluation Datasets

This runtime keeps evaluation data outside git. Use `eval/` for local dataset
exports; the repository ignores that directory.

## Raw Dataset Mirrors

Mirror the upstream Hugging Face dataset repos into the ignored local evaluation
folder when you want the original parquet/card files available offline:

```bash
uv run --with huggingface_hub hf download \
  ekacare/eka-medical-asr-evaluation-dataset \
  --repo-type dataset \
  --local-dir eval/hf-datasets/ekacare/eka-medical-asr-evaluation-dataset

uv run --with huggingface_hub hf download \
  argmaxinc/chime-6 \
  --repo-type dataset \
  --local-dir eval/hf-datasets/argmaxinc/chime-6
```

These mirrors are not used directly by the runtime. Export short WAV slices from
them or from the Hugging Face dataset API before running `perftest`.

## Recommended First Labeled Set: Eka Medical ASR

Use the Hugging Face dataset
[`ekacare/eka-medical-asr-evaluation-dataset`](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset)
for the first labeled assessment. It is directly aligned with medical ASR: short
16 kHz audio examples, ground-truth text, English and Hindi subsets, and medical
entity metadata.

The dataset card reports:

- license: MIT
- total samples: 3,939
- subsets: `en` with 3,619 test samples, `hi` with 320 test samples
- total file size: 281 MB
- fields including `audio`, `duration`, `text`, `audio_language`,
  `text_language`, `recording_context`, `type_concept`, and
  `medical_entities`

Export a small English smoke slice:

```bash
uv run --with datasets --with soundfile --with numpy \
  scripts/download-eval-dataset \
  ekacare/eka-medical-asr-evaluation-dataset \
  --subset en \
  --split test \
  --limit 25 \
  --streaming \
  --out eval/eka-medical-en
```

The checked local layout after export is:

```text
eval/eka-medical-en/
  audio/
  expected.jsonl
```

Export a small Hindi smoke slice:

```bash
uv run --with datasets --with soundfile --with numpy \
  scripts/download-eval-dataset \
  ekacare/eka-medical-asr-evaluation-dataset \
  --subset hi \
  --split test \
  --limit 25 \
  --streaming \
  --out eval/eka-medical-hi
```

Run Qwen3-ASR against the exported manifest:

```bash
scripts/qwen3-asr-mlx-perftest \
  --local-files-only \
  --manifest eval/eka-medical-en/expected.jsonl \
  --json > eval/eka-medical-en/qwen3-asr-0.6b.json
```

Score WER and CER:

```bash
scripts/qwen3-asr-mlx-score \
  --expected eval/eka-medical-en/expected.jsonl \
  --predictions eval/eka-medical-en/qwen3-asr-0.6b.json
```

Run the 1.7B accuracy candidate on the same slice:

```bash
scripts/qwen3-asr-mlx-perftest \
  --model Qwen/Qwen3-ASR-1.7B \
  --local-files-only \
  --manifest eval/eka-medical-en/expected.jsonl \
  --json > eval/eka-medical-en/qwen3-asr-1.7b.json
```

## Long-recording Stress Set: CHiME-6

Use
[`argmaxinc/chime-6`](https://huggingface.co/datasets/argmaxinc/chime-6)
as an optional long-recording, multi-speaker stress source. It is useful for VAD
boundary and batched ASR behavior, not as the default quick WER gate.

The dataset page reports:

- license: CC-BY-SA-4.0
- subset: `default`
- split: `test`
- rows: 2
- total file size: 589 MB
- modalities: audio and text
- row audio durations around 9.2k and 9.55k seconds
- fields including `audio`, `timestamps_start`, `timestamps_end`, `speakers`,
  `transcript`, and `word_speakers`

Export short clips for runtime stress:

```bash
uv run --with datasets --with soundfile --with numpy \
  scripts/download-eval-dataset \
  argmaxinc/chime-6 \
  --split test \
  --limit 1 \
  --clip-start 0 \
  --clip-duration 30 \
  --text-column transcript \
  --streaming \
  --out eval/chime6-smoke
```

For CHiME-6, the exported `text` field may be empty for clipped examples when
word timestamps are not available. When the exporter can infer only a rough text
slice, it writes `text_scope: "duration_proportional_approximation"`. Treat that
as a manual inspection aid, not a WER reference.

## Local Assessment Loop

Use the same exported manifest for both timing and quality:

```bash
scripts/qwen3-asr-mlx-perftest \
  --local-files-only \
  --decoder-mode sequential \
  --manifest eval/eka-medical-en/expected.jsonl \
  --json > eval/eka-medical-en/sequential.json

scripts/qwen3-asr-mlx-perftest \
  --local-files-only \
  --decoder-mode batched \
  --manifest eval/eka-medical-en/expected.jsonl \
  --json > eval/eka-medical-en/batched.json

scripts/qwen3-asr-mlx-score \
  --expected eval/eka-medical-en/expected.jsonl \
  --predictions eval/eka-medical-en/sequential.json

scripts/qwen3-asr-mlx-score \
  --expected eval/eka-medical-en/expected.jsonl \
  --predictions eval/eka-medical-en/batched.json
```

Report both quality and runtime:

- WER and CER from the score script
- total audio seconds
- run seconds
- realtime factor
- peak RSS
- MLX peak memory
- decoder mode
- model id
