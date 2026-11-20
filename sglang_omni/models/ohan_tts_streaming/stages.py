# SPDX-License-Identifier: Apache-2.0
"""Preprocessing and audio decoding stages for OhanTTS-Streaming."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping

import torch

from sglang_omni.models.higgs_tts.utils import load_audio_to_24k
from sglang_omni.models.ohan_tts.payload_types import OhanTTSState
from sglang_omni.models.ohan_tts.vocoder_scheduler import OhanTTSVocoderScheduler
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

STREAMING_MODEL_ID = "Metric-AI/OhanTTS-Streaming-v0.1"
BASE_MODEL_ID = "k2-fsa/OmniVoice"
MAX_WORD_FRAMES = 128
DEFAULT_REPETITION_PENALTY = 1.1

_FRONTENDS: dict[tuple[str, str, str], object] = {}
_FRONTENDS_LOCK = threading.Lock()


def _get_frontend(base_model_path: str, streaming_model_path: str, codec_device: str):
    from .frontend import OhanTTSStreamingFrontend

    key = (base_model_path, streaming_model_path, codec_device)
    with _FRONTENDS_LOCK:
        frontend = _FRONTENDS.get(key)
        if frontend is None:
            frontend = OhanTTSStreamingFrontend(
                base_model_path,
                streaming_model_path,
                codec_device=codec_device,
            )
            _FRONTENDS[key] = frontend
        return frontend


def _inputs(raw: object) -> dict[str, object]:
    if isinstance(raw, str):
        return {"text": raw}
    if not isinstance(raw, Mapping):
        raise TypeError("OhanTTS-Streaming inputs must be a string or object")
    values = dict(raw)
    references = values.get("references")
    if isinstance(references, list) and references and isinstance(references[0], Mapping):
        ref = references[0]
        values.setdefault("reference_text", ref.get("text"))
        if "reference_audio" not in values:
            values["reference_audio"] = ref.get("audio_path") or ref.get("path")
    return values


def create_preprocessing_executor(
    base_model_path: str = BASE_MODEL_ID,
    streaming_model_path: str = STREAMING_MODEL_ID,
    *,
    device: str | None = "cuda",
    gpu_id: int | None = 0,
) -> SimpleScheduler[StagePayload, StagePayload]:
    from sglang_omni.utils.device import resolve_concrete_device

    codec_device = str(resolve_concrete_device(device, gpu_id))
    frontend = _get_frontend(base_model_path, streaming_model_path, codec_device)

    def preprocess(payload: StagePayload) -> StagePayload:
        inputs = _inputs(payload.request.inputs or {})
        params = payload.request.params or {}
        text = str(inputs.get("text") or inputs.get("input") or "").strip()
        reference_audio = inputs.get("reference_audio")
        reference_text = inputs.get("reference_text")
        if reference_audio is None:
            raise ValueError("OhanTTS-Streaming requires reference audio")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError("OhanTTS-Streaming requires the reference transcript")
        waveform, sample_rate = load_audio_to_24k(reference_audio)
        prompt = frontend.prepare_prompt(
            text=text,
            reference_waveform=torch.from_numpy(waveform).float(),
            reference_sample_rate=sample_rate,
            reference_text=reference_text.strip(),
            language=str(inputs.get("language") or params.get("language") or "hy"),
            instructions=str(params.get("instructions") or "").strip() or None,
        )
        temperature = float(params.get("temperature", 0.0))
        top_p_value = params.get("top_p", 1.0)
        top_p = None if top_p_value is None else float(top_p_value)
        top_k_value = params.get("top_k")
        top_k = None if top_k_value in (None, 0) else int(top_k_value)
        repetition_penalty = float(
            params.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
        )
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if top_p is not None and (not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and > 0")

        payload.data = OhanTTSState(
            prompt_token_ids=prompt.prompt_token_ids,
            reference_codes_delayed=prompt.reference_codes or None,
            max_new_tokens=MAX_WORD_FRAMES * len(text.split()) + len(prompt.continuation_ids),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=int(params["seed"]) if params.get("seed") is not None else None,
            num_codebooks=8,
            codebook_size=1025,
            reference_rms=prompt.reference_rms,
            language=str(inputs.get("language") or params.get("language") or "hy"),
            sample_rate=prompt.sample_rate,
            repetition_penalty=repetition_penalty,
        ).to_dict()
        payload.data["continuation_ids"] = prompt.continuation_ids
        return payload

    return SimpleScheduler(preprocess)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_running_requests: int = 64,
    cuda_graph_max_bs: int = 64,
    total_gpu_memory_fraction: float = 0.25,
):
    from .engine_builder import OhanTTSStreamingEngineBuilder

    builder = OhanTTSStreamingEngineBuilder(
        max_new_tokens=4096,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        enable_async_decode=False,
        async_decode_min_batch_size=1,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        stream_chunk_frames=20,
        stream_lookahead_frames=4,
        stream_left_context_frames=8,
    )
    return builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides={"disable_cuda_graph": False, "disable_radix_cache": True, "chunked_prefill_size": -1},
    )


def create_vocoder_executor(
    base_model_path: str = BASE_MODEL_ID,
    streaming_model_path: str = STREAMING_MODEL_ID,
    *,
    device: str | None = "cuda",
    gpu_id: int | None = 0,
    codec_dtype: str = "float32",
    postprocess_output: bool = False,
    pad_duration: float = 0.1,
) -> OhanTTSVocoderScheduler:
    from sglang_omni.utils.device import resolve_concrete_device

    if codec_dtype != "float32":
        raise ValueError("OhanTTS-Streaming codec requires float32 output")
    codec_device = str(resolve_concrete_device(device, gpu_id))
    frontend = _get_frontend(base_model_path, streaming_model_path, codec_device)
    if postprocess_output:
        raise ValueError("OhanTTS-Streaming currently decodes without postprocessing")
    hop_length = int(frontend.model.audio_tokenizer.config.hop_length)
    return OhanTTSVocoderScheduler(
        frontend.frontend,
        sample_rate=int(frontend.model.sampling_rate),
        hop_length=hop_length,
        pad_duration=pad_duration,
    )


__all__ = [
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
