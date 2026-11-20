# SPDX-License-Identifier: Apache-2.0
"""Per-request state for OhanTTS streaming TTS."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.scheduling.pipeline_state import wire
from sglang_omni.models.ohan_tts.sampling import DEFAULT_REPETITION_PENALTY


@dataclass
class OhanTTSState(HiggsTtsState):
    """Higgs-compatible scheduler state with OhanTTS clone metadata."""

    reference_rms: float | None = wire(None, codec="raw")
    language: str = wire("hy", codec="str_or")
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY


__all__ = ["OhanTTSState"]
