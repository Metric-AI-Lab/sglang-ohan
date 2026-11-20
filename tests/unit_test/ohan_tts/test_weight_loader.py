# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.ohan_tts.weight_loader import map_checkpoint_name


def test_maps_qwen3_backbone_weights():
    assert map_checkpoint_name("model.llm.layers.0.self_attn.q_proj.weight") == (
        "backbone.model.layers.0.self_attn.q_proj.weight"
    )


def test_maps_untied_codec_embedding_and_prediction_head():
    assert map_checkpoint_name("model.audio_embeddings.weight") == (
        "multimodal_embedding.modality_embedding_0.weight"
    )
    assert map_checkpoint_name("model.audio_heads.weight") == "modality_head.weight"


def test_ignores_codec_decoder_and_tokenizer_weights():
    assert map_checkpoint_name("model.audio_tokenizer.acoustic_decoder.weight") is None
    assert map_checkpoint_name("model.text_tokenizer.embed_tokens.weight") is None
