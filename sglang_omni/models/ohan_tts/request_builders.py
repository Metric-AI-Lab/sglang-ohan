# SPDX-License-Identifier: Apache-2.0
"""SGLang scheduler adapters for fixed-length autoregressive codec generation."""

from __future__ import annotations

import time
from typing import Callable

from sglang_omni.models.higgs_tts.request_builders import (
    HiggsSGLangRequestData,
    apply_higgs_result,
    build_sglang_higgs_request,
)
from sglang_omni.models.ohan_tts.payload_types import OhanTTSState
from sglang_omni.proto import StagePayload


def make_scheduler_adapters(
    *,
    text_vocab_size: int = 151676,
    max_new_tokens_cap: int = 768,
    chunk_frames: int = 20,
    lookahead_frames: int = 4,
    left_context_frames: int = 8,
) -> tuple[
    Callable[[StagePayload], HiggsSGLangRequestData],
    Callable[[HiggsSGLangRequestData], StagePayload],
]:
    """Create request/result adapters, preserving the per-request frame limit."""

    def request_builder(payload: StagePayload) -> HiggsSGLangRequestData:
        state = OhanTTSState.from_dict(payload.data)
        state.max_new_tokens = min(int(state.max_new_tokens), max_new_tokens_cap)
        # Reuse the project's stable Req construction and ref-audio cache key,
        # then correct the text vocabulary size for OhanTTS's Qwen3 checkpoint.
        data = build_sglang_higgs_request(state, request_id=payload.request_id)
        data.req.vocab_size = text_vocab_size
        data.req.sampling_params.repetition_penalty = state.repetition_penalty
        data.engine_start_s = time.perf_counter()
        data.stage_payload = payload
        params = payload.request.params or {}
        if bool(params.get("stream", False)):
            data.stream_metadata = {
                "modality": "audio_codes",
                "stream": True,
                "num_codebooks": state.num_codebooks,
                "codebook_size": state.codebook_size,
                "ohan_tts_chunk_frames": chunk_frames,
                "ohan_tts_lookahead_frames": lookahead_frames,
                "ohan_tts_left_context_frames": left_context_frames,
                "reference_rms": state.reference_rms,
            }
        else:
            data.stream_metadata = None
        return data

    def result_adapter(data: HiggsSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = OhanTTSState.from_dict(payload.data)
        apply_higgs_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = time.perf_counter() - data.engine_start_s
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


__all__ = ["make_scheduler_adapters"]
