# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni integration for OhanTTS-Streaming."""

from transformers import AutoConfig

from sglang_omni.models.model_capabilities import ModelCapabilities

from .hf_config import OhanTTSStreamingConfig

AutoConfig.register(
    OhanTTSStreamingConfig.model_type, OhanTTSStreamingConfig, exist_ok=True
)

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=True,
    supports_cuda_graph=True,
    supports_torch_compile=False,
    supports_breakable_prefill_cuda_graph=False,
    supports_full_prefill_cuda_graph=False,
)

__all__ = ["OhanTTSStreamingConfig"]
