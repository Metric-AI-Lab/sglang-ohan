# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for OhanTTS-Streaming."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import EngineStageConfig, FactoryArgs, PipelineConfig, StageConfig

_PKG = "sglang_omni.models.ohan_tts_streaming"
_BASE_MODEL = "k2-fsa/OmniVoice"
_STREAMING_MODEL = "Metric-AI/OhanTTS-Streaming-v0.1"


class OhanTTSStreamingPipelineConfig(PipelineConfig):
    """SGLang pipeline for the separate learned-duration model."""

    architecture: ClassVar[str] = "OhanTTSStreamingForConditionalGeneration"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(
                base_model_path=_BASE_MODEL,
                streaming_model_path=_STREAMING_MODEL,
            ),
            gpu=0,
            gpu_memory_fraction=0.05,
            next="tts_engine",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(
                base_model_path=_BASE_MODEL,
                streaming_model_path=_STREAMING_MODEL,
            ),
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
                max_running_requests=64,
                cuda_graph_max_bs=64,
            ),
            gpu=0,
            gpu_memory_fraction=0.25,
            next="vocoder",
            stream_to=["vocoder"],
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = OhanTTSStreamingPipelineConfig
