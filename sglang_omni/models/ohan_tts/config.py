# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for the OhanTTS streaming AR checkpoint."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import EngineStageConfig, FactoryArgs, PipelineConfig, StageConfig

_PKG = "sglang_omni.models.ohan_tts"


class OhanTTSPipelineConfig(PipelineConfig):
    """Prepare a clone prompt, generate codec frames, then decode a WAV."""

    architecture: ClassVar[str] = "OhanTTSForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "OmniVoiceStreamForConditionalGeneration",
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(model_path="Metric-AI/OhanTTS-v0.1"),
            gpu=0,
            gpu_memory_fraction=0.05,
            next="tts_engine",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(model_path="Metric-AI/OhanTTS-v0.1"),
            gpu=0,
            gpu_memory_fraction=0.05,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(
                max_new_tokens=768,
                max_running_requests=64,
                cuda_graph_max_bs=64,
                stream_chunk_frames=20,
                stream_lookahead_frames=4,
                stream_left_context_frames=8,
            ),
            gpu=0,
            gpu_memory_fraction=0.25,
            next="vocoder",
            stream_to=["vocoder"],
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = OhanTTSPipelineConfig
