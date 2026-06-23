# JSONL Protocol

Protocol version: `qwen3-asr-mlx-jsonl-v1`

Transport is newline-delimited JSON:

- stdin: one request JSON object per line
- stdout: one response JSON object per line
- stderr: setup logs and diagnostics

## Audio Contract

- sample rate: `16000`
- channel layout: mono
- live chunks: PCM16 little-endian, base64 encoded
- file input: local WAV/audio path readable by the bridge process
- VAD: external; callers send utterance boundaries

## Requests

### `capabilities`

Does not load the model.

```json
{"type":"capabilities"}
```

### `start`

Loads the model and returns readiness.

```json
{"type":"start"}
```

### `transcribe`

Transcribes one audio file.

```json
{"type":"transcribe","audio":"examples/audio/sample.wav","max_new_tokens":0}
```

`max_new_tokens: 0` means decode until EOS.

### `batch_transcribe`

Transcribes multiple finalized utterances.

```json
{"type":"batch_transcribe","decoder_mode":"sequential","audio":["utt-1.wav","utt-2.wav"],"max_new_tokens":0}
```

`decoder_mode` can be `sequential` or `batched`.

### `start_stream`

Starts a live stream buffer.

```json
{"type":"start_stream","stream_id":"live","sample_rate":16000,"batch_size":2,"max_batch_delay_ms":120}
```

### `audio_chunk`

Appends base64 PCM16 to a stream.

```json
{"type":"audio_chunk","stream_id":"live","seq":1,"pcm16":"..."}
```

### `end_utterance` / `vad_boundary`

Queues a VAD-finalized utterance from the stream buffer:

```json
{"type":"end_utterance","stream_id":"live","utterance_id":"utt-1"}
```

Queues a VAD-finalized utterance from a file:

```json
{"type":"vad_boundary","stream_id":"live","utterance_id":"utt-2","audio":"utt-2.wav"}
```

### `flush`

Runs queued utterances as a micro-batch.

```json
{"type":"flush","decoder_mode":"sequential","max_new_tokens":0}
```

### `stop_stream`

Clears one stream buffer.

```json
{"type":"stop_stream","stream_id":"live"}
```

### `stop`

Stops the bridge.

```json
{"type":"stop"}
```

## Responses

### `capabilities`

```json
{"type":"capabilities","runtime":"qwen3-asr-mlx","protocol":"qwen3-asr-mlx-jsonl-v1","model":"Qwen/Qwen3-ASR-0.6B","sample_rate":16000,"decoder_modes":["sequential","batched"]}
```

### `ready`

```json
{"type":"ready","runtime":"qwen3-asr-mlx","protocol":"qwen3-asr-mlx-jsonl-v1","model":"Qwen/Qwen3-ASR-0.6B","model_dir":"...","use_cache":true,"load_seconds":2.8}
```

### `transcript`

```json
{"type":"transcript","audio":"sample.wav","text":"language English<asr_text>...","generated_tokens":13,"audio_embeddings_shape":[32,1024],"decode_seconds":0.4,"total_seconds":0.6,"profile":{}}
```

### `batch_transcript`

```json
{"type":"batch_transcript","count":2,"items":[{"audio":"utt-1.wav","text":"...","generated_tokens":13,"audio_embeddings_shape":[32,1024],"decode_seconds":0.4}],"decoder_mode":"sequential","profile":{}}
```

### `realtime_batch_final`

```json
{"type":"realtime_batch_final","reason":"flush","count":1,"items":[{"type":"final_result","stream_id":"live","utterance_id":"utt-1","text":"...","samples":25600}],"decoder_mode":"sequential","profile":{}}
```

### `error`

```json
{"type":"error","message":"..."}
```

## Profile Fields

Profile payloads may include:

- `feature_extract_seconds`
- `prompt_assembly_seconds`
- `audio_tower_seconds`
- `batch_assembly_seconds`
- `prefill_seconds`
- `continuation_seconds`
- `decode_seconds`
- `total_seconds`
- `generated_tokens_per_second`
- `realtime_factor`
- `rss_peak_mb`
- `mlx_peak_mb`
- `mlx_active_mb`
- `mlx_cache_mb`
