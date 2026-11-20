# OhanTTS

Checkpoint: `Metric-AI/OhanTTS-v0.1`.

## Quick start

OhanTTS-v0.1 is a gated model. Before launching the server, open the
[model page](https://huggingface.co/Metric-AI/OhanTTS-v0.1), click
"Agree and Access", and then sign in with `hf auth login`.

Run from the repository root after signing in to Hugging Face.

```bash
PYTHONPATH="$PWD" .venv/bin/sgl-omni serve \
  --model-path Metric-AI/OhanTTS-v0.1 \
  --port 8000
```

The command leaves serving settings at their defaults.

## Default parameters

### Server

| Setting | Default |
| --- | --- |
| Maximum running requests | 64 |
| CUDA graph batch-size cap | 64 |
| CUDA graphs | Enabled |
| Streaming window | 8 left-context / 20 emitted / 4 lookahead frames |
| Padding at utterance edges | 0.1 seconds |

Each full streaming packet emits 0.8 seconds of new audio. To change the
window, set `--tts_engine.factory.stream_left_context_frames`,
`--tts_engine.factory.stream_chunk_frames`, or
`--tts_engine.factory.stream_lookahead_frames` when starting the server.

## Request defaults

The reference audio and its transcript are required for this checkpoint. If
generation fields are omitted from a request, these values apply:

| Request field | Default |
| --- | --- |
| `temperature` | `0.0` (greedy) |
| `top_p` | `1.0` |
| `top_k` | No limit (`null` or `0`) |
| `repetition_penalty` | `1.1` |
| `max_new_tokens` | Estimated from text, up to 768 frames |
| `seed` | None |
| `language` | Armenian (`hy`) |

You can override these values per request. `top_p`, `top_k`, and `seed` affect
sampling only when `temperature` is above zero.

## Streaming request

Replace the reference WAV and its transcript with your own. The request
uploads the WAV and receives PCM audio as it is generated.

```bash
base64 -w0 reference.wav | jq -Rs --arg transcript 'Transcript of reference.wav' '{
  input: "Բարև ձեզ։",
  response_format: "pcm",
  stream: true,
  references: [{data: ., media_type: "audio/wav", text: $transcript}]
}' | curl -N --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' --data-binary @- -o ohan_tts.pcm
```

The PCM response is mono, signed 16-bit. Read its sample rate from the
`X-Sample-Rate` response header. For a complete WAV response, use
`"response_format": "wav"` and omit `"stream": true`.