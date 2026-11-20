# SPDX-License-Identifier: Apache-2.0
"""HF config for the OhanTTS streaming TTS model, backed by a Qwen3 decoder."""

from __future__ import annotations

from collections.abc import Mapping

import transformers

from sglang_omni.models.higgs_tts.hf_config import (
    HiggsMultimodalQwen3Config,
    build_text_config,
)


class OhanTTSConfig(HiggsMultimodalQwen3Config):
    """SGLang-facing config; the codec heads are eight parallel codebooks."""

    model_type = "ohan_tts"
    has_no_defaults_at_init = True

    def __init__(
        self,
        llm_config: Mapping[str, object] | transformers.PretrainedConfig | None = None,
        text_config: Mapping[str, object] | transformers.PretrainedConfig | None = None,
        num_audio_codebook: int = 8,
        audio_vocab_size: int = 1025,
        **kwargs,
    ) -> None:
        raw_text_config = text_config if text_config is not None else llm_config
        if raw_text_config is None:
            raise ValueError("OhanTTSConfig requires llm_config/text_config")
        resolved_text_config = build_text_config(raw_text_config)
        super().__init__(
            audio_encoder_config={
                "encoder_type": "discrete",
                "num_codebooks": int(num_audio_codebook),
                "vocab_size": int(audio_vocab_size),
                "out_dim": int(resolved_text_config.hidden_size),
                # OhanTTS keeps input embeddings and prediction heads separate.
                "tie_word_embeddings": False,
            },
            text_config=resolved_text_config,
            audio_token_id=-100,
            **kwargs,
        )
        self.num_audio_codebook = int(num_audio_codebook)
        self.audio_vocab_size = int(audio_vocab_size)


class LegacyOhanTTSConfig(OhanTTSConfig):
    """Load checkpoints saved before the model type was renamed."""

    model_type = "omnivoice_stream"
