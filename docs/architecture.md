# Architecture

The runtime is intentionally split from any app UI.

```text
caller app
  |
  | newline JSON over stdin/stdout
  v
qwen3-asr-mlx bridge
  |
  | finalized utterances from external VAD
  v
Qwen3ASRMLXRuntime
  |
  +-- Transformers/Qwen processor for audio features and tokenizer metadata
  +-- MLX Qwen3-ASR audio tower
  +-- audio embedding splice into text embeddings
  +-- Qwen3 MRoPE position generation
  +-- local MRoPE-aware Qwen decoder
  +-- cached greedy generation
```

## Runtime Responsibilities

The runtime owns:

- locating/downloading model snapshots
- loading config, processor, tokenizer metadata, and safetensors
- audio feature extraction through the official processor path
- MLX audio tower execution
- prompt construction and audio pad expansion
- Qwen3-ASR MRoPE positions
- decoder cache and greedy token generation
- batch preparation for VAD-finalized utterances
- JSONL bridge protocol

The caller owns:

- microphone capture
- VAD or endpointing
- diarization
- UI state
- persistence
- network or web API surface
- cancellation policy

## Why External VAD

Qwen3-ASR can run streaming-shaped transcription, but this bridge treats VAD as
an app concern. That keeps the runtime reusable:

- Swift apps can use CoreML or native audio endpointing.
- Web servers can use a separate VAD worker.
- Batch pipelines can pass files directly.

The bridge accepts finalized utterances via `end_utterance`, `vad_boundary`, or
`batch_transcribe`.

## Model Sizes

`Qwen/Qwen3-ASR-0.6B`:

- audio embedding width observed: `1024`
- best current default for realtime transcription

`Qwen/Qwen3-ASR-1.7B`:

- audio embedding width observed: `2048`
- heavier accuracy candidate

Both use the same protocol and runtime path.
