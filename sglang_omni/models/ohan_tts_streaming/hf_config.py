# SPDX-License-Identifier: Apache-2.0
"""SGLang config for OhanTTS-Streaming."""

from __future__ import annotations

from collections.abc import Mapping

import transformers

from sglang_omni.models.higgs_tts.hf_config import (
    HiggsMultimodalQwen3Config,
    build_text_config,
)


class OhanTTSStreamingConfig(HiggsMultimodalQwen3Config):
    """Configuration for the OhanTTS-Streaming model."""

    model_type = "ohan_tts_streaming"
    has_no_defaults_at_init = True

    def __init__(
        self,
        llm_config: Mapping[str, object] | transformers.PretrainedConfig | None = None,
        text_config: Mapping[str, object] | transformers.PretrainedConfig | None = None,
        num_audio_codebook: int = 8,
        audio_vocab_size: int = 1025,
        duration_token_id: int = 151676,
        num_duration_classes: int = 128,
        **kwargs,
    ) -> None:
        raw_text_config = text_config if text_config is not None else llm_config
        if raw_text_config is None:
            raise ValueError("OhanTTS-Streaming requires llm_config/text_config")
        resolved_text_config = build_text_config(raw_text_config)
        super().__init__(
            audio_encoder_config={
                "encoder_type": "discrete",
                "num_codebooks": int(num_audio_codebook),
                "vocab_size": int(audio_vocab_size),
                "out_dim": int(resolved_text_config.hidden_size),
                "tie_word_embeddings": False,
            },
            text_config=resolved_text_config,
            audio_token_id=-100,
            **kwargs,
        )
        self.num_audio_codebook = int(num_audio_codebook)
        self.audio_vocab_size = int(audio_vocab_size)
        self.duration_token_id = int(duration_token_id)
        self.num_duration_classes = int(num_duration_classes)


__all__ = ["OhanTTSStreamingConfig"]
