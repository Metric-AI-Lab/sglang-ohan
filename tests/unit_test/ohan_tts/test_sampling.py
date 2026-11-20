"""Parity checks for the fixed-length codec repetition penalty."""

from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from sglang_omni.models.ohan_tts.model import OhanTTSSGLangModel
from sglang_omni.models.ohan_tts.sampling import (
    DEFAULT_REPETITION_PENALTY,
    penalize_repeated_codes,
)


def test_repetition_penalty_matches_sign_aware_evaluation_rule() -> None:
    logits = torch.tensor([[5.0, 4.9, -2.0, -1.0]])
    seen = torch.tensor([[True, False, True, False]])

    actual = penalize_repeated_codes(logits, seen)

    assert DEFAULT_REPETITION_PENALTY == 1.1
    assert actual.tolist()[0] == pytest.approx([5.0 / 1.1, 4.9, -2.0 * 1.1, -1.0])
    assert int(actual.argmax(dim=-1).item()) == 1


def test_repetition_penalty_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        penalize_repeated_codes(torch.zeros(2, 4), torch.zeros(2, 3, dtype=torch.bool))


def test_codec_selection_penalizes_seen_ids_and_masks_sentinel() -> None:
    model_state = SimpleNamespace(
        _seen_audio_codes=torch.tensor([[[True, False, False, False]]]),
        _request_repetition_penalties=torch.tensor([DEFAULT_REPETITION_PENALTY]),
        sampler_pool=SimpleNamespace(
            seeds=torch.tensor([-1]), step_count=torch.tensor([0])
        ),
        config=SimpleNamespace(audio_mask_id=3),
    )
    logits = torch.tensor([[[5.0, 4.9, 4.0, 99.0]]])

    codes = OhanTTSSGLangModel._select_codes(
        model_state,
        logits,
        torch.tensor([0]),
        gen_params=[SimpleNamespace(temperature=0.0, top_p=1.0, top_k=None)],
    )

    assert codes.tolist() == [[1]]
    assert model_state._seen_audio_codes.tolist() == [[[True, True, False, False]]]
