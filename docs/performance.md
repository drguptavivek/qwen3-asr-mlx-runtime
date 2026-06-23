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

## Multi-file Sample Smoke Set

Three longer WAV files totaling `39.2685s` of audio, using the optimized
combined checkpoint loader and `Qwen/Qwen3-ASR-0.6B`.

| Decoder mode | Audio files | Load time | Run time | Realtime factor | Realtime factor with load | Peak RSS | MLX peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential` | 3 | `2.9045s` | `3.3762s` | `11.63x` | `5.96x` | `2569.18 MB` | `3650.41 MB` |
| `batched` | 3 | `2.7501s` | `2.5568s` | `15.36x` | `6.99x` | `2560.80 MB` | `3435.69 MB` |

Sequential and batched decode produced matching text on this three-file smoke
set. Treat this as runtime evidence, not a quality guarantee; batched decode can
still drift when next-token logits are nearly tied.

## Run Your Own Perftest

Use `scripts/qwen3-asr-mlx-perftest` for local timing on one or more WAV/audio
files. The command loads the model once, transcribes the supplied files, and
reports load time, run time, realtime factor, token counts, audio embedding
shape, and memory telemetry.

The current runtime loads the Qwen3-ASR audio tower and MRoPE decoder through a
combined checkpoint loader. Safetensor shards are read once and partitioned into
audio and text weights in memory. The perftest JSON includes `load_profile` with
shard read time, weight load time, weight counts, and memory telemetry.

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

## Core MLX Runtime Optimizations

Implemented:

- combined checkpoint loader for audio tower and decoder, so safetensor shards
  are read once instead of once per component
- launcher package stamp, so repeat bridge/perftest runs skip dependency
  resolution when the managed venv already has the tested package set
- cached prompt token template
- fast contiguous audio embedding splice
- cached MRoPE one-step cos/sin tensors during generation
- cached decoder generation
- batched feature extraction and audio tower for multiple finalized utterances
- optional batched multimodal prefill and `BatchKVCache` continuation

Candidate future work:

- optional model quantization as a separate accuracy/latency mode
- MLX compilation experiments for stable-shape audio tower or decode kernels
- lower-level audio feature extraction replacement if CPU feature extraction
  becomes material for very small utterances
- labeled WER/CER test harness to ensure speed changes do not change quality

Measured launcher effect on the same managed venv:

| Command | Startup behavior | Wall time |
| --- | --- | ---: |
| first `--print-capabilities` after stamp change | validates package set and writes stamp | `0.27s` |
| repeated `--print-capabilities` | skips dependency resolution | `0.06s` |
