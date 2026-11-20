# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni integration for the OhanTTS streaming AR checkpoint."""

from transformers import AutoConfig

from sglang_omni.models.model_capabilities import ModelCapabilities

from . import config
from sglang_omni.models.ohan_tts.hf_config import LegacyOhanTTSConfig, OhanTTSConfig

try:
    AutoConfig.register(OhanTTSConfig.model_type, OhanTTSConfig)
except ValueError:
    # Another import path may already have registered the same config class.
    pass
AutoConfig.register(LegacyOhanTTSConfig.model_type, LegacyOhanTTSConfig, exist_ok=True)

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=True,
    supports_cuda_graph=False,
    supports_torch_compile=False,
    supports_breakable_prefill_cuda_graph=False,
    supports_full_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES", "config", "OhanTTSConfig"]
