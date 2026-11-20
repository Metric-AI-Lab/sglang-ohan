# SPDX-License-Identifier: Apache-2.0
"""Map the OhanTTS Hugging Face checkpoint into the SGLang model tree."""

from __future__ import annotations


_PREFIX_MAP = {
    "model.llm.embed_tokens.": "backbone.model.embed_tokens.",
    "model.llm.layers.": "backbone.model.layers.",
    "model.llm.norm.": "backbone.model.norm.",
    "model.audio_embeddings.": "multimodal_embedding.modality_embedding_0.",
    "model.audio_heads.": "modality_head.",
}


def map_checkpoint_name(name: str) -> str | None:
    """Return the SGLang parameter name, or ``None`` for frozen/non-AR weights."""
    for source_prefix, destination_prefix in _PREFIX_MAP.items():
        if name.startswith(source_prefix):
            return destination_prefix + name[len(source_prefix) :]
    # Text/audio tokenizers and the DAC codec are loaded by their own pipeline
    # stages, not into the autoregressive SGLang worker.
    return None
