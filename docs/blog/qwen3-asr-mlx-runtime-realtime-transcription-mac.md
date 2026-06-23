# Qwen3-ASR MLX Runtime for Realtime Transcription on Mac

Realtime transcription is not only an ASR model problem. A usable transcription
system needs audio capture, speech boundary detection, batching, decoding,
latency monitoring, memory control, and a stable interface that an application
can call repeatedly. On a Mac, it also needs to use Apple Silicon efficiently
without making the application depend on a CUDA server.

Qwen3-ASR gave the community a strong open ASR model family. The remaining
implementation question for us was narrower and more operational: can we make
Qwen3-ASR work as a local MLX runtime on Apple Silicon, with a bridge that can
be called by a Swift app, a web backend, or another local process?

This article describes what was already available from Qwen, what was working
for realtime streaming ASR, what was missing for a Mac-native workflow, and what
we built in `qwen3-asr-mlx-runtime`.

## What Qwen Provided

Qwen released `Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B` as ASR models
for language identification and speech recognition. The model card describes
support for 52 languages and dialects, and lists both offline and streaming
inference as intended modes. It also describes the 1.7B model as the stronger
accuracy model and the 0.6B model as the better accuracy-efficiency trade-off.
The models are released under Apache-2.0 on Hugging Face.

The official user path was practical for many server deployments. Qwen provided
the `qwen-asr` Python package, a Transformers backend, a vLLM backend, a local
web UI, a streaming demo, and vLLM serving examples. The model card states that
`qwen-asr` provides two backends: Transformers and vLLM. It also states that
streaming inference is currently available with the vLLM backend.

For a CUDA server, this is a good starting point. A team can install
`qwen-asr[vllm]`, run vLLM serving, and expose transcription through an API.
For batch inference, the Transformers backend is also useful. The official
toolkit therefore covered model access, server-style deployment, and
streaming-shaped inference through vLLM.

The problem was different for a Mac desktop application.

## The Service Problem on Mac

A native Mac transcription app has a different pathway from a GPU server. It
records audio locally, detects speech boundaries locally, keeps user data on the
machine, and needs to return text without sending every utterance to a remote
service. The app should not require the user to run a CUDA container or a
separate Linux server.

Apple Silicon gives us MLX, which is well suited for local inference on Mac.
But Qwen3-ASR was not a standard text-only Qwen model. It needed more than
loading a decoder and calling `generate`.

The ASR path includes several specific steps:

- audio feature extraction
- Qwen3-ASR audio tower execution
- expansion of the `<|audio_pad|>` placeholder
- replacement of text token embeddings with audio embeddings
- Qwen3-ASR MRoPE positions
- decoder caching
- utterance-level batching
- a stable process boundary for non-Python apps

If any of these steps are skipped, the result may still run as code, but it
will not be a faithful Qwen3-ASR runtime. For a transcription product, that is
not enough. The runtime needs to reproduce the model pathway and expose it in a
form that the application can use safely.

## What Was Working For Realtime Streaming ASR

Before this work, the official streaming path was available through Qwen's vLLM
backend. This is appropriate when the deployment target is a server with vLLM
support. It gives a programme or product team a way to run streaming ASR through
an API-like service.

But it did not answer the local Mac question:

- There was no standard MLX Qwen3-ASR runtime for Apple Silicon.
- The standard `mlx-lm` Qwen3 text path did not directly expose the full
  multimodal ASR pathway as a ready-to-use ASR engine.
- A Swift or desktop app still needed a local bridge that could keep the model
  resident and accept finalized utterances.
- The system still needed a policy for VAD boundaries, batching, decoder mode,
  logging, and memory telemetry.

The difference matters in implementation. Realtime transcription is a service
pipeline. The model is only one component. The app needs a predictable boundary:
send audio or an utterance boundary, receive partial or final text, and collect
timing and memory indicators for monitoring.

## What We Built

We built `qwen3-asr-mlx-runtime`, a code-only Apache-2.0 runtime for Qwen3-ASR
on Apple Silicon. It is not an official Qwen project. It is a local MLX runtime
and newline-JSON bridge that can be used by other applications.

The public repository is:

<https://github.com/drguptavivek/qwen3-asr-mlx-runtime>

The runtime provides:

- an MLX implementation of the Qwen3-ASR audio tower path
- a local MRoPE-aware Qwen decoder wrapper
- audio embedding splice into the text embedding stream
- cached greedy generation
- VAD-sized utterance batching
- sequential and batched decoder modes
- structured profiling for latency and memory
- a newline-delimited JSON bridge for Swift, Node, Python, and other apps
- a submodule-friendly repository structure

The runtime does not include model weights. It downloads or uses cached
Hugging Face model snapshots under the user's configured model cache.

## The Runtime Contract

The bridge protocol is intentionally simple:

```text
caller app
  -> JSON line on stdin
qwen3-asr-mlx-runtime
  -> JSON line on stdout
diagnostics
  -> stderr
```

The app can send:

- `start`
- `transcribe`
- `batch_transcribe`
- `start_stream`
- `audio_chunk`
- `end_utterance`
- `vad_boundary`
- `flush`
- `stop`

The bridge returns:

- `ready`
- `transcript`
- `batch_transcript`
- `realtime_batch_final`
- `error`

This makes the runtime usable from a Swift app, a local web backend, or a
command-line tool. The caller owns microphone capture, VAD, diarization, UI,
and persistence. The runtime owns model loading, audio feature preparation,
audio tower execution, MRoPE positions, decoding, and telemetry.

This separation is important. VAD is not the same as ASR. Diarization is not the
same as ASR. Text cleanup is not the same as ASR. Keeping these boundaries clear
makes the runtime easier to test and reuse.

## Why VAD Boundary Batching

For live transcription, batching the whole recording is not useful. The user
needs text after each spoken segment. At the same time, running the model for
every tiny audio fragment wastes overhead and can make latency unstable.

We therefore used a VAD boundary model of operation. The caller sends speech
segments after endpointing. The runtime can process these finalized utterances
one by one or in a small micro-batch.

This is a practical compromise:

- VAD remains app-specific.
- The ASR runtime receives clean utterance units.
- Multiple finalized utterances can share feature extraction and audio tower
  work.
- The app can control latency through `batch_size` and flush timing.
- The transcript boundary remains meaningful for saving, editing, and review.

For a desktop transcription app, this is closer to the real service pathway
than unrestricted continuous decoding.

## The Main Technical Steps

The first task was to reproduce the Qwen3-ASR input pathway honestly.

We implemented and tested:

1. loading the Qwen3-ASR config, processor, tokenizer metadata, and safetensors
2. Whisper-style feature extraction through the Qwen processor path
3. `_get_feat_extract_output_lengths` parity for audio token counts
4. prompt construction with `<|audio_start|><|audio_pad|><|audio_end|>`
5. expansion of one audio placeholder into the required number of audio tokens
6. MLX execution of the Qwen3-ASR audio tower
7. splicing audio embeddings into the text embedding stream
8. Qwen3-ASR MRoPE position generation
9. a local Qwen3 decoder wrapper that consumes explicit MRoPE cos/sin tensors
10. cached greedy generation for repeated utterances
11. JSONL protocol responses with stage-level telemetry

The important point is that this was not a text-only Qwen port. It was an ASR
pathway port. The audio tower and MRoPE handling were required for correctness.

## Optimizations

The first working version was useful, but not yet a reusable realtime runtime.
We optimized the parts that matter for application use.

### 1. Keep the model resident

Model loading was kept out of the hot path. The app sends `start` once. The
bridge keeps the model loaded and processes subsequent utterances without
rebuilding the runtime.

This is essential for a live app. Loading a model for every utterance would make
the system unusable regardless of model speed.

### 2. Cache prompt token templates

The prompt has a stable prefix and suffix around the audio placeholder. We cache
this template and only expand the audio pad span according to the computed audio
token length.

This reduces repeated prompt work and makes input assembly predictable.

### 3. Use fast contiguous audio embedding splice

The audio embedding span is contiguous. Instead of doing slow element-wise
replacement, we splice the text prefix, audio embeddings, and text suffix as
three contiguous blocks.

This is simple and less error-prone.

### 4. Cache decoder generation

We added cached greedy decoding so that generated tokens use a decoder KV cache.
This avoids recomputing the whole prefix at every step.

### 5. Batch the audio tower

For multiple VAD-finalized utterances, the runtime can batch feature extraction
and the MLX audio tower. This is useful when several segments are ready close
together.

### 6. Add two decoder modes

The runtime exposes two modes:

- `sequential`: batch feature extraction and audio tower, then decode each
  utterance with its own cached decoder. This is the default for final text.
- `batched`: batch feature extraction, audio tower, multimodal prefill, and
  continuation decode using `BatchKVCache`. This reduces latency but may cause
  small wording differences when logits are close.

This gives the application a policy choice. It can use `sequential` for final
saved transcripts and `batched` for lower-latency draft output.

### 7. Add structured telemetry

The bridge returns timing and memory fields such as:

- feature extraction time
- audio tower time
- prompt assembly time
- prefill time
- continuation decode time
- total decode time
- generated tokens per second
- realtime factor
- RSS peak memory
- MLX peak, active, and cache memory

For a local transcription app, these are monitoring indicators. They help decide
which model should be the default, when batching is useful, and when the system
is too heavy for a particular Mac.

## Model Choice: 0.6B and 1.7B

We tested both released ASR models.

The 0.6B model is the realtime efficiency default. It is faster and uses much
less memory. It is the better starting point for live transcription on Mac.

The 1.7B model is the accuracy candidate. It produced cleaner wording on our
small smoke set, but it also used substantially more memory and took longer.
It should be offered as an opt-in mode until there is a labeled evaluation set.

Our measured runtime smoke numbers on three VAD-sized WAVs totaling `10.1001s`
of audio were:

| Model | Decoder mode | Total time | Realtime factor | Peak RSS | MLX peak |
| --- | --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-ASR-0.6B` | `sequential` | `1.1157s` | `9.06x` | `2418.84 MB` | `3093.95 MB` |
| `Qwen/Qwen3-ASR-0.6B` | `batched` | `0.6936s` | `14.59x` | `2409.76 MB` | `3179.04 MB` |
| `Qwen/Qwen3-ASR-1.7B` | `sequential` | `2.1896s` | `4.61x` | `4914.04 MB` | `6609.91 MB` |
| `Qwen/Qwen3-ASR-1.7B` | `batched` | `1.4309s` | `7.06x` | `4927.28 MB` | `6663.12 MB` |

These are smoke-test numbers, not a formal benchmark. They are still useful for
planning. They show that 1.7B is functional on the same runtime path, but about
twice as heavy in latency and memory on this test set.

The observed text differences were also useful:

| Segment | 0.6B sequential | 1.7B sequential | Assessment |
| --- | --- | --- | --- |
| `segment_6_input.wav` | `Sorry.` | `Sorry.` | Equivalent. |
| `segment_2_input.wav` | `Chris tries there. Hey, Chris.` | `Chris Drysdale.` | 1.7B is shorter and more name-like, but needs ground truth before calling it correct. |
| `segment_1_input.wav` | `Possibly, maybe he wanna hang out with the cool guy.` | `Possibly, maybe he'd want to hang out with the cool guy.` | 1.7B is more grammatical; both preserve the same meaning. |

The next fair quality step is WER/CER testing on labeled audio. Until that is
done, 1.7B should be called an accuracy candidate, not a proven accuracy
upgrade.

## Tested Package Versions

The working MLX stack was:

```text
mlx==0.31.2
mlx-lm==0.29.1
transformers==4.57.6
qwen-asr==0.0.6
```

The runtime launcher pins these versions. Users can install the same stack with:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install "mlx==0.31.2" "mlx-lm==0.29.1" "transformers==4.57.6" "qwen-asr==0.0.6"
```

## How Users Can Run It

Clone the runtime:

```bash
git clone https://github.com/drguptavivek/qwen3-asr-mlx-runtime.git
cd qwen3-asr-mlx-runtime
```

Start the bridge:

```bash
scripts/qwen3-asr-mlx-bridge
```

Or run only the capability check:

```bash
scripts/qwen3-asr-mlx-bridge --print-capabilities
```

A caller can then send JSON lines:

```json
{"type":"start"}
{"type":"transcribe","audio":"examples/audio/sample.wav","max_new_tokens":0}
{"type":"stop"}
```

For a parent application, the recommended pattern is a git submodule:

```bash
git submodule add https://github.com/drguptavivek/qwen3-asr-mlx-runtime.git Vendor/qwen3-asr-mlx-runtime
git submodule update --init --recursive
```

This keeps the runtime independently versioned and keeps the app pinned to a
known tested commit.

## What This Changes

Qwen3-ASR was already strong. The official package and vLLM path made it usable
for server deployments. The work here makes a different pathway available:
local Mac transcription using MLX, external VAD boundaries, small utterance
batches, and a bridge that any app can call.

This is useful for applications where local processing matters: desktop
transcription, privacy-sensitive note taking, field recording, offline review,
and developer tools that need a local ASR worker.

The main lesson is practical. A realtime transcription system needs more than a
model card. It needs a reproducible runtime, a clear boundary between VAD and
ASR, a model choice policy, latency and memory indicators, and a stable
interface for the application. `qwen3-asr-mlx-runtime` is one implementation of
that pathway for Apple Silicon Macs.

## Sources

- Qwen3-ASR model card: <https://huggingface.co/Qwen/Qwen3-ASR-0.6B>
- Qwen3-ASR upstream repository: <https://github.com/QwenLM/Qwen3-ASR>
- Qwen3-ASR technical report: <https://arxiv.org/abs/2601.21337>
- qwen3-asr-mlx-runtime: <https://github.com/drguptavivek/qwen3-asr-mlx-runtime>
