# SPDX-License-Identifier: Apache-2.0
"""Preprocessing, SGLang engine, and audio decode stages for OhanTTS."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping

import torch

from sglang_omni.models.higgs_tts.utils import load_audio_to_24k
from sglang_omni.models.ohan_tts.frontend import OhanTTSFrontend
from sglang_omni.models.ohan_tts.payload_types import OhanTTSState
from sglang_omni.models.ohan_tts.sampling import DEFAULT_REPETITION_PENALTY
from sglang_omni.models.ohan_tts.vocoder_scheduler import (
    OhanTTSVocoderScheduler,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


_FRONTENDS: dict[tuple[str, str], OhanTTSFrontend] = {}
_FRONTENDS_LOCK = threading.Lock()


def _get_frontend(model_path: str, codec_device: str) -> OhanTTSFrontend:
    key = (model_path, codec_device)
    with _FRONTENDS_LOCK:
        frontend = _FRONTENDS.get(key)
        if frontend is None:
            frontend = OhanTTSFrontend(model_path, codec_device=codec_device)
            _FRONTENDS[key] = frontend
        return frontend


def _normalize_request_inputs(raw: object) -> dict[str, object]:
    if isinstance(raw, str):
        return {"text": raw}
    if not isinstance(raw, Mapping):
        raise TypeError("TTS request inputs must be a string or object")
    inputs = dict(raw)
    references = inputs.get("references")
    if isinstance(references, list) and references:
        ref = references[0]
        if isinstance(ref, Mapping):
            if "reference_text" not in inputs and ref.get("text"):
                inputs["reference_text"] = ref["text"]
            if "reference_audio" not in inputs:
                audio = ref.get("audio_path") or ref.get("path")
                if audio is None and any(key in ref for key in ("bytes", "base64", "data")):
                    audio = {key: ref[key] for key in ("bytes", "base64", "data") if key in ref}
                inputs["reference_audio"] = audio
    return inputs


def create_preprocessing_executor(
    model_path: str = "Metric-AI/OhanTTS-v0.1",
    *,
    device: str | None = "cuda",
    gpu_id: int | None = 0,
    max_target_frames: int = 768,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """Prepare clone prefixes with the GPU codec used by original inference."""
    from sglang_omni.utils.device import resolve_concrete_device

    codec_device = str(resolve_concrete_device(device, gpu_id))
    frontend = _get_frontend(model_path, codec_device)

    def preprocess(payload: StagePayload) -> StagePayload:
        inputs = _normalize_request_inputs(payload.request.inputs or {})
        params = payload.request.params or {}
        text = str(inputs.get("input") or inputs.get("text") or "")
        reference = inputs.get("reference_audio")
        instructions = str(params.get("instructions") or "").strip()
        reference_text = inputs.get("reference_text")
        if reference is None:
            raise ValueError(
                "This OhanTTS checkpoint requires a reference "
                "audio clip and its transcript in references[0]"
            )
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError("references[0].text must be the reference audio transcript")
        waveform, sample_rate = load_audio_to_24k(reference)
        reference_waveform = torch.from_numpy(waveform).float()
        language = str(inputs.get("language") or params.get("language") or "hy")
        prompt_ids, ref_codes, frame_count, ref_rms = frontend.prepare_clone(
            text=text,
            reference_waveform=reference_waveform,
            reference_sample_rate=sample_rate,
            reference_text=reference_text.strip(),
            instructions=instructions or None,
            language=language,
            max_target_frames=max_target_frames,
        )
        temperature = float(params.get("temperature", 0.0))
        top_p_value = params.get("top_p", 1.0)
        top_p = None if top_p_value is None else float(top_p_value)
        top_k_value = params.get("top_k")
        top_k = None if top_k_value in (None, 0) else int(top_k_value)
        repetition_penalty = float(
            params.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
        )
        max_new_tokens_value = params.get("max_new_tokens")
        max_new_tokens = (
            int(frame_count)
            if max_new_tokens_value is None
            else int(max_new_tokens_value)
        )
        seed_value = params.get("seed")
        seed = None if seed_value is None else int(seed_value)
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite number >= 0")
        if top_p is not None and (not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("top_p must be null or a finite number in (0, 1]")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be null, 0, or a positive integer")
        if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be a finite number > 0")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        state = OhanTTSState(
            prompt_token_ids=prompt_ids,
            reference_codes_delayed=ref_codes or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            num_codebooks=8,
            codebook_size=1025,
            reference_rms=ref_rms,
            language=language,
            sample_rate=frontend.model.sampling_rate,
            repetition_penalty=repetition_penalty,
        )
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(preprocess)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_new_tokens: int = 768,
    max_running_requests: int = 64,
    cuda_graph_max_bs: int = 64,
    total_gpu_memory_fraction: float = 0.25,
    stream_chunk_frames: int = 20,
    stream_lookahead_frames: int = 4,
    stream_left_context_frames: int = 8,
    server_args_overrides: Mapping[str, object] | None = None,
):
    """Build the SGLang AR stage using the compatible multi-codebook runner."""
    from sglang_omni.models.ohan_tts.engine_builder import (
        OhanTTSEngineBuilder,
    )

    builder = OhanTTSEngineBuilder(
        max_new_tokens=max_new_tokens,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        enable_async_decode=False,
        async_decode_min_batch_size=1,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        stream_chunk_frames=stream_chunk_frames,
        stream_lookahead_frames=stream_lookahead_frames,
        stream_left_context_frames=stream_left_context_frames,
    )
    return builder.build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides={"disable_cuda_graph": False, **(server_args_overrides or {})},
    )


def create_vocoder_executor(
    model_path: str = "Metric-AI/OhanTTS-v0.1",
    *,
    device: str | None = "cuda",
    gpu_id: int | None = 0,
    codec_dtype: str = "float32",
    postprocess_output: bool = False,
    pad_duration: float = 0.1,
) -> OhanTTSVocoderScheduler:
    """Build streaming window decode and full-utterance fallback decoding."""
    from sglang_omni.utils.device import resolve_concrete_device

    if codec_dtype != "float32":
        raise ValueError("OhanTTS codec requires float32 output for NumPy decoding")
    codec_device = str(resolve_concrete_device(device, gpu_id))
    frontend = _get_frontend(model_path, codec_device)

    if postprocess_output:
        raise ValueError(
            "Streaming and full decode currently require postprocess_output=False "
            "to keep chunk amplitudes consistent."
        )
    hop_length = int(frontend.model.audio_tokenizer.config.hop_length)
    return OhanTTSVocoderScheduler(
        frontend,
        sample_rate=int(frontend.model.sampling_rate),
        hop_length=hop_length,
        pad_duration=pad_duration,
    )


__all__ = [
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
