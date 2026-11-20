# SPDX-License-Identifier: Apache-2.0
"""SGLang engine settings for OhanTTS's fixed-length codec generation."""

from __future__ import annotations

from collections.abc import Callable

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.higgs_tts.engine_builder import HiggsTtsEngineBuilder
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.request_builders import HiggsSGLangRequestData
from sglang_omni.models.ohan_tts.request_builders import make_scheduler_adapters
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


class OhanTTSModelRunner(HiggsTTSModelRunner):
    """Flush aligned OhanTTS frames using the codec chunk cadence."""

    def before_prefill(self, forward_batch, schedule_batch, requests) -> None:
        super().before_prefill(forward_batch, schedule_batch, requests)
        for request in requests:
            self.model.set_request_repetition_penalty(
                request.request_id,
                request.data.req.sampling_params.repetition_penalty,
            )

    @staticmethod
    def initial_stream_flush_rows(data: HiggsSGLangRequestData) -> int:
        metadata = data.stream_metadata or {}
        return max(
            1,
            int(metadata.get("ohan_tts_chunk_frames", 20))
            + int(metadata.get("ohan_tts_lookahead_frames", 4)),
        )

    @staticmethod
    def next_stream_flush_rows(
        data: HiggsSGLangRequestData, flushed_rows: int
    ) -> int:
        metadata = data.stream_metadata or {}
        return int(flushed_rows) + max(
            1, int(metadata.get("ohan_tts_chunk_frames", 20))
        )


class OhanTTSEngineBuilder(HiggsTtsEngineBuilder):
    """Reuse Higgs' multi-codebook runner with OhanTTS request framing."""

    model_name = "OhanTTS"

    def __init__(
        self,
        *,
        max_new_tokens: int | None,
        max_running_requests: int,
        cuda_graph_max_bs: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        total_gpu_memory_fraction: float | None,
        stream_chunk_frames: int = 20,
        stream_lookahead_frames: int = 4,
        stream_left_context_frames: int = 8,
    ) -> None:
        super().__init__(
            max_new_tokens=max_new_tokens,
            max_running_requests=max_running_requests,
            cuda_graph_max_bs=cuda_graph_max_bs,
            enable_async_decode=enable_async_decode,
            async_decode_min_batch_size=async_decode_min_batch_size,
            total_gpu_memory_fraction=total_gpu_memory_fraction,
        )
        if stream_chunk_frames < 1 or stream_lookahead_frames < 0 or stream_left_context_frames < 0:
            raise ValueError("OhanTTS stream window sizes are invalid")
        self.stream_chunk_frames = stream_chunk_frames
        self.stream_lookahead_frames = stream_lookahead_frames
        self.stream_left_context_frames = stream_left_context_frames

    def make_model_runner(
        self,
        model_worker: ModelWorker,
        output_proc: SGLangOutputProcessor,
    ) -> OhanTTSModelRunner:
        return OhanTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: object) -> tuple[
        Callable[[StagePayload], HiggsSGLangRequestData],
        Callable[[HiggsSGLangRequestData], StagePayload],
    ]:
        del model
        return make_scheduler_adapters(
            max_new_tokens_cap=self.max_new_tokens,
            chunk_frames=self.stream_chunk_frames,
            lookahead_frames=self.stream_lookahead_frames,
            left_context_frames=self.stream_left_context_frames,
        )
