# SPDX-License-Identifier: Apache-2.0
"""SGLang engine builder for OhanTTS-Streaming."""

from sglang_omni.models.ohan_tts.engine_builder import (
    OhanTTSEngineBuilder,
    OhanTTSModelRunner,
)
from sglang_omni.models.ohan_tts.request_builders import make_scheduler_adapters
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend


class OhanTTSStreamingRunner(OhanTTSModelRunner):
    def before_prefill(self, forward_batch, schedule_batch, requests):
        super().before_prefill(forward_batch, schedule_batch, requests)
        for request in requests:
            self.model.set_continuation(
                request.request_id, request.data.stage_payload.data["continuation_ids"]
            )

    def before_decode(self, forward_batch, schedule_batch, requests, *, is_lookahead=False):
        super().before_decode(
            forward_batch, schedule_batch, requests, is_lookahead=is_lookahead
        )
        batch_size = int(forward_batch.batch_size)
        rows = self.model.cg_row_indices[:batch_size]
        self.model.active_decode_state[:batch_size] = self.model.decode_state[rows]

    def decode_pack_gpu(self, n_real):
        rows = self.model.cg_row_indices[:n_real]
        self.model.decode_state[rows] = self.model.active_decode_state[:n_real]
        return super().decode_pack_gpu(n_real)


class OhanTTSStreamingEngineBuilder(OhanTTSEngineBuilder):
    model_name = "OhanTTS-Streaming"
    supports_breakable_prefill_cuda_graph = False

    def generation_defaults(self, *, dtype):
        defaults = super().generation_defaults(dtype=dtype)
        defaults.pop("cuda_graph_bs_prefill", None)
        defaults["cuda_graph_backend_prefill"] = CudaGraphBackend.DISABLED
        return defaults

    def make_model_runner(self, model_worker, output_proc):
        return OhanTTSStreamingRunner(model_worker, output_proc)

    def make_adapters(self, model):
        return make_scheduler_adapters(
            text_vocab_size=model.config.text_config.vocab_size,
            max_new_tokens_cap=self.max_new_tokens,
            chunk_frames=self.stream_chunk_frames,
            lookahead_frames=self.stream_lookahead_frames,
            left_context_frames=self.stream_left_context_frames,
        )
