# Performance

These numbers are from local Apple Silicon MLX smoke tests. They are useful for
runtime planning but are not a formal benchmark.

## Saved Utterance Smoke Set

Three VAD-sized WAVs totaling `10.1001s` of audio.

| Model | Intended role | Decoder mode | Total time | Realtime factor | Peak RSS | MLX peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-ASR-0.6B` | realtime efficiency default | `sequential` | `1.1157s` | `9.06x` | `2418.84 MB` | `3093.95 MB` |
| `Qwen/Qwen3-ASR-0.6B` | lowest-latency draft/speed mode | `batched` | `0.6936s` | `14.59x` | `2409.76 MB` | `3179.04 MB` |
| `Qwen/Qwen3-ASR-1.7B` | accuracy candidate | `sequential` | `2.1896s` | `4.61x` | `4914.04 MB` | `6609.91 MB` |
| `Qwen/Qwen3-ASR-1.7B` | heavier batched candidate | `batched` | `1.4309s` | `7.06x` | `4927.28 MB` | `6663.12 MB` |

## Run Your Own Perftest

Use `scripts/qwen3-asr-mlx-perftest` for local timing on one or more WAV/audio
files. The command loads the model once, transcribes the supplied files, and
reports load time, run time, realtime factor, token counts, audio embedding
shape, and memory telemetry.

One file:

```bash
scripts/qwen3-asr-mlx-perftest --local-files-only path/to/audio.wav
```

Multiple VAD-sized files:

```bash
scripts/qwen3-asr-mlx-perftest --local-files-only \
  path/to/utt-001.wav \
  path/to/utt-002.wav \
  path/to/utt-003.wav
```

Accuracy candidate:

```bash
scripts/qwen3-asr-mlx-perftest Qwen/Qwen3-ASR-1.7B --local-files-only \
  path/to/utt-001.wav \
  path/to/utt-002.wav
```

Batched decoder path:

```bash
scripts/qwen3-asr-mlx-perftest --local-files-only \
  --decoder-mode batched \
  path/to/utt-001.wav \
  path/to/utt-002.wav
```

Machine-readable output:

```bash
scripts/qwen3-asr-mlx-perftest --local-files-only --json path/to/audio.wav
```

Show decoded text in the table output:

```bash
scripts/qwen3-asr-mlx-perftest --local-files-only --show-text path/to/audio.wav
```

Important options:

| Option | Use |
| --- | --- |
| `--decoder-mode sequential` | Batches feature extraction/audio tower, then decodes each file with its own cache. This is the default for final transcripts. |
| `--decoder-mode batched` | Uses batched multimodal prefill and `BatchKVCache` continuation. Faster, but small wording drift is possible. |
| `--max-new-tokens 0` | Decode until EOS. This is the default. |
| `--json` | Emit one JSON result object for scripts and dashboards. |
| `--show-text` | Include transcript text in human-readable output. |
| `--local-files-only` | Use only locally cached model files. Omit it for first download. |

## Interpretation

- `0.6B` is the current realtime default.
- `1.7B` works through the same MLX path but roughly doubles latency and MLX
  peak memory on this smoke set.
- `batched` decode improves latency, but can produce small wording differences
  when logits are nearly tied.
- `sequential` decode should be used for final transcripts until a labeled
  evaluation proves batched decode is acceptable for the target product.

## Observed Transcript Differences

| Segment | 0.6B sequential | 1.7B sequential | Assessment |
| --- | --- | --- | --- |
| `segment_6_input.wav` | `Sorry.` | `Sorry.` | Equivalent. |
| `segment_2_input.wav` | `Chris tries there. Hey, Chris.` | `Chris Drysdale.` | 1.7B is shorter and more name-like, but needs ground truth before calling it correct. |
| `segment_1_input.wav` | `Possibly, maybe he wanna hang out with the cool guy.` | `Possibly, maybe he'd want to hang out with the cool guy.` | 1.7B is more grammatical; both preserve the same meaning. |

## Next Benchmark Gate

Add a labeled evaluation folder:

```text
eval/
  audio/
    sample-001.wav
  expected.jsonl
```

`expected.jsonl`:

```json
{"audio":"audio/sample-001.wav","text":"expected transcript"}
```

Then report:

- WER
- CER
- total latency
- realtime factor
- peak RSS
- MLX peak memory
- per-stage timings
