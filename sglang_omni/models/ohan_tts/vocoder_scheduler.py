# SPDX-License-Identifier: Apache-2.0
"""Incremental DAC decoding for OhanTTS's aligned codec frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from sglang_omni.models.ohan_tts.payload_types import OhanTTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase


@dataclass
class OhanTTSDecodeState:
    """Generated codec history and the next central frame range to emit."""

    code_chunks: list[torch.Tensor] = field(default_factory=list)
    total_frames: int = 0
    next_frame: int = 0
    reference_rms: float | None = None
    num_codebooks: int | None = None
    codebook_size: int | None = None
    chunk_frames: int | None = None
    lookahead_frames: int | None = None
    left_context_frames: int | None = None


class OhanTTSVocoderScheduler(
    StreamingVocoderBase[OhanTTSDecodeState, None]
):
    """Decode only completed central windows while retaining codec context."""

    def __init__(
        self,
        frontend,
        *,
        sample_rate: int,
        hop_length: int,
        pad_duration: float = 0.1,
    ) -> None:
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        self.frontend = frontend
        self.hop_length = int(hop_length)
        self.pad_samples = round(float(pad_duration) * int(sample_rate))
        super().__init__(
            self.decode_payload,
            sample_rate=int(sample_rate),
            stream_source_hint="OhanTTS",
        )

    def create_stream_state(self, request_id: str) -> OhanTTSDecodeState:
        del request_id
        return OhanTTSDecodeState()

    @staticmethod
    def _latch(state: OhanTTSDecodeState, key: str, value: int) -> None:
        current = getattr(state, key)
        if current is not None and current != value:
            raise ValueError(
                f"OhanTTS stream contract changed for {key}: {current} -> {value}"
            )
        setattr(state, key, value)

    def latch_stream_contract(
        self,
        request_id: str,
        state: OhanTTSDecodeState,
        source: StagePayload | Mapping[str, object],
        *,
        origin: str,
    ) -> None:
        del request_id
        if origin == "payload":
            if not isinstance(source, StagePayload) or not isinstance(source.data, dict):
                raise TypeError("OhanTTS initial stream payload must contain a state dict")
            reference_rms = source.data.get("reference_rms")
            if reference_rms is not None:
                state.reference_rms = float(reference_rms)
            self._latch(state, "num_codebooks", int(source.data.get("num_codebooks", 8)))
            self._latch(state, "codebook_size", int(source.data.get("codebook_size", 1025)))
            return

        if not isinstance(source, Mapping):
            raise TypeError("OhanTTS stream metadata must be a mapping")
        required = {
            "num_codebooks": "num_codebooks",
            "codebook_size": "codebook_size",
            "ohan_tts_chunk_frames": "chunk_frames",
            "ohan_tts_lookahead_frames": "lookahead_frames",
            "ohan_tts_left_context_frames": "left_context_frames",
        }
        for metadata_key, state_key in required.items():
            if metadata_key not in source:
                raise ValueError(f"OhanTTS stream metadata is missing {metadata_key!r}")
            self._latch(state, state_key, int(source[metadata_key]))
        reference_rms = source.get("reference_rms")
        if reference_rms is not None:
            state.reference_rms = float(reference_rms)
        if state.num_codebooks != 8:
            raise ValueError(f"Expected 8 OhanTTS codebooks, got {state.num_codebooks}")
        if state.chunk_frames < 1 or state.lookahead_frames < 0 or state.left_context_frames < 0:
            raise ValueError("OhanTTS stream window sizes are invalid")

    def validate_chunk(
        self, request_id: str, state: OhanTTSDecodeState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id
        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
        if codes.ndim != 2 or codes.shape[1] != state.num_codebooks:
            raise ValueError(
                f"Expected aligned codec frames [T, 8], got {tuple(codes.shape)}"
            )
        if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= 1024):
            raise ValueError("OhanTTS generated codec IDs must be in [0, 1023]")
        return codes.detach().to(device="cpu", dtype=torch.long).contiguous()

    def ingest(
        self, request_id: str, state: OhanTTSDecodeState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.code_chunks.append(codes)
        state.total_frames += int(codes.shape[0])

    def should_decode(
        self, state: OhanTTSDecodeState, *, is_final: bool
    ) -> bool:
        del is_final
        return state.total_frames - state.next_frame >= (
            int(state.chunk_frames) + int(state.lookahead_frames)
        )

    def decode_delta(
        self,
        request_id: str,
        state: OhanTTSDecodeState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        del request_id
        if state.total_frames <= state.next_frame:
            return None
        codes = torch.cat(state.code_chunks, dim=0)
        chunks: list[torch.Tensor] = []
        first_output = state.next_frame == 0

        while state.next_frame < state.total_frames:
            center_start = state.next_frame
            center_end = min(center_start + int(state.chunk_frames), state.total_frames)
            window_end = min(
                state.total_frames, center_end + int(state.lookahead_frames)
            )
            if not is_final and window_end - center_end < int(state.lookahead_frames):
                break

            window_start = max(0, center_start - int(state.left_context_frames))
            decoded = self.frontend.decode(
                codes[window_start:window_end],
                reference_rms=state.reference_rms,
                postprocess=False,
            ).reshape(-1)
            sample_start = self.pad_samples + (
                center_start - window_start
            ) * self.hop_length
            sample_count = (center_end - center_start) * self.hop_length
            central = decoded[sample_start : sample_start + sample_count]
            if central.numel() < sample_count:
                central = torch.nn.functional.pad(central, (0, sample_count - central.numel()))
            if central.numel():
                if first_output:
                    leading_silence = central.new_zeros(self.pad_samples)
                    central = torch.cat((leading_silence, central))
                    first_output = False
                chunks.append(central.float().cpu())
            state.next_frame = center_end

            if not is_final:
                break

        if not chunks:
            return None
        if is_final:
            chunks.append(chunks[-1].new_zeros(self.pad_samples))
        return torch.cat(chunks)

    def decode_payload(self, payload: StagePayload) -> StagePayload:
        state = OhanTTSState.from_dict(payload.data)
        raw_codes = state.output_codes_delayed
        if not raw_codes:
            raise RuntimeError("OhanTTS returned no codec frames")
        codes = torch.as_tensor(raw_codes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[1] != 8:
            raise ValueError(f"Expected completed codec frames [T, 8], got {tuple(codes.shape)}")
        audio = self.frontend.decode(
            codes,
            reference_rms=state.reference_rms,
            postprocess=False,
        )
        payload.data = {
            **state.to_dict(),
            "audio_data": audio.reshape(-1).tolist(),
            "sample_rate": self.sample_rate,
            "modality": "audio",
        }
        return payload

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: OhanTTSDecodeState,
    ) -> Mapping[str, object]:
        del request_id, payload, state
        return {"sample_rate": self.sample_rate, "modality": "audio"}


__all__ = ["OhanTTSVocoderScheduler"]
