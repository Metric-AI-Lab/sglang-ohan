# OhanTTS-Streaming

Model: [Metric-AI/OhanTTS-Streaming-v0.1].

## Quick start

Run from the repository root. If Hugging Face requires access, request it on
the model page and sign in with `hf auth login`.

Prepare the local model directory, then start SGLang-Omni.

```bash
PYTHONPATH="$PWD" .venv/bin/python -m sglang_omni.models.ohan_tts_streaming.prepare_model \
  artifacts/OhanTTS-Streaming

PYTHONPATH="$PWD" .venv/bin/sgl-omni serve \
  --model-path "$PWD/artifacts/OhanTTS-Streaming" \
  --port 8000 \
  --allowed-local-media-path /absolute/path/to/references
```

The command uses the default serving settings.

## Default parameters

### Server

| Setting | Default |
| --- | --- |
| Maximum running requests | 64 |
| CUDA graph batch-size cap | 64 |
| Decode CUDA graphs | Enabled |
| Prefill CUDA graphs | Disabled |
| Streaming window | 8 left-context / 20 emitted / 4 lookahead frames |
| Padding at utterance edges | 0.1 seconds |

Each full streaming window emits 0.8 seconds of new audio before edge padding.
The vocoder window can span word boundaries.

## Request defaults

Reference audio and its transcript are required.

| Request field | Default |
| --- | --- |
| `temperature` | `0.0` (greedy) |
| `top_p` | `1.0` |
| `top_k` | No limit (`null` or `0`) |
| `repetition_penalty` | `1.1` |
| `max_new_tokens` | Determined internally from the text and engine limits |
| `seed` | None |
| `language` | Omit for Armenian |

You can override the sampling settings per request. `top_p`, `top_k`, and
`seed` affect sampling only when `temperature` is above zero.

Send the full text in one request. The model appends each word and `<DPH>`
to the same KV cache. The learned duration head sets how many audio codec
frames to generate before the next word.

## Streaming request

Replace the reference URI and transcript with your own. The local file
must be under the directory passed to `--allowed-local-media-path`.

```bash
curl -N --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Բարև ձեզ։",
    "ref_audio": "file:///absolute/path/to/references/reference.wav",
    "ref_text": "Transcript of reference.wav",
    "response_format": "pcm",
    "stream": true
  }' -o ohan_tts_streaming.pcm
```

The PCM response is mono, signed 16-bit. Read its sample rate from the
`X-Sample-Rate` response header. For a complete WAV response, use
`"response_format": "wav"` and omit `"stream": true`.
