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
