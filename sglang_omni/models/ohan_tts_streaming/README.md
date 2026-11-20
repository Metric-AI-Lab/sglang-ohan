# OhanTTS-Streaming

This is a separate SGLang model implementation for `Metric-AI/OhanTTS-Streaming-v0.1`.
It uses the model's `<DPH>` token and learned 128-class duration head to set
the number of codec frames. It does not use OmniVoice's rule-based duration
estimator. Repetition penalty defaults to `1.1`.

Prepare the local model directory, then start SGLang-Omni.

```bash
PYTHONPATH="$PWD" .venv/bin/python -m sglang_omni.models.ohan_tts_streaming.prepare_model \
  artifacts/OhanTTS-Streaming

PYTHONPATH="$PWD" .venv/bin/sgl-omni serve \
  --model-path "$PWD/artifacts/OhanTTS-Streaming" \
  --port 8000 \
  --allowed-local-media-path /path/to/references
```

Send the full target text in one request. The reference and first word are
processed together. Each next word is appended to the same KV cache after
the preceding word's codec frames. The cache is released when the request ends.
