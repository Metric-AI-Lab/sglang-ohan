"""Window and lookahead behavior for the OhanTTS streaming decoder."""

from __future__ import annotations

import torch

from sglang_omni.models.ohan_tts.vocoder_scheduler import (
    OhanTTSDecodeState,
    OhanTTSVocoderScheduler,
)


class _FrameIdentityFrontend:
    def decode(self, codes, *, reference_rms, postprocess):
        del reference_rms, postprocess
        return codes[:, 0].float().repeat_interleave(2)


def _scheduler() -> OhanTTSVocoderScheduler:
    return OhanTTSVocoderScheduler(
        _FrameIdentityFrontend(), sample_rate=24000, hop_length=2, pad_duration=0
    )


def _state() -> OhanTTSDecodeState:
    return OhanTTSDecodeState(
        num_codebooks=8,
        codebook_size=1025,
        chunk_frames=4,
        lookahead_frames=2,
        left_context_frames=8,
    )


def _frames(start: int, count: int) -> torch.Tensor:
    ids = torch.arange(start, start + count, dtype=torch.long)
    return ids[:, None].expand(-1, 8).contiguous()


def test_waits_for_lookahead_and_emits_only_central_frames() -> None:
    scheduler = _scheduler()
    state = _state()
    scheduler.ingest("r", state, _frames(0, 5))
    assert scheduler.decode_delta("r", state, is_final=False) is None

    scheduler.ingest("r", state, _frames(5, 1))
    first = scheduler.decode_delta("r", state, is_final=False)
    assert first is not None
    assert first.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]

    scheduler.ingest("r", state, _frames(6, 4))
    second = scheduler.decode_delta("r", state, is_final=False)
    assert second is not None
    assert second.tolist() == [4, 4, 5, 5, 6, 6, 7, 7]


def test_final_flush_emits_short_remainder_without_lookahead() -> None:
    scheduler = _scheduler()
    state = _state()
    scheduler.ingest("r", state, _frames(0, 3))
    final = scheduler.decode_delta("r", state, is_final=True)
    assert final is not None
    assert final.tolist() == [0, 0, 1, 1, 2, 2]

