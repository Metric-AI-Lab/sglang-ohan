"""Integration contracts for the OhanTTS checkpoint adapter."""

from __future__ import annotations

import json

import pytest

from scripts import ohan_tts
from sglang_omni.models.ohan_tts.config import OhanTTSPipelineConfig
from sglang_omni.models.ohan_tts.payload_types import OhanTTSState
from sglang_omni.models.ohan_tts.request_builders import make_scheduler_adapters
from sglang_omni.models.ohan_tts.sampling import DEFAULT_REPETITION_PENALTY
from sglang_omni.proto import OmniRequest, StagePayload


def test_reference_codec_shares_gpu_with_vocoder() -> None:
    stages = {
        stage.name: stage
        for stage in OhanTTSPipelineConfig.model_fields["stages"].default
    }
    assert stages["preprocessing"].gpu == stages["vocoder"].gpu == 0
    assert stages["preprocessing"].process == stages["vocoder"].process
    assert stages["tts_engine"].next == "vocoder"
    window = stages["tts_engine"].factory
    assert (
        window.stream_left_context_frames,
        window.stream_chunk_frames,
        window.stream_lookahead_frames,
    ) == (8, 20, 4)


@pytest.mark.parametrize("config_field", ["model_config", "checkpoint_settings"])
def test_checkpoint_preparation_keeps_hub_weights(
    monkeypatch, tmp_path, config_field
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text(
        f"{config_field}:\n  postprocess_output: false\n"
    )
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"checkpoint weights")

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps({"llm_config": {"model_type": "qwen3", "hidden_size": 16}})
    )
    monkeypatch.setattr(
        ohan_tts,
        "snapshot_download",
        lambda repo: str(checkpoint if repo == "checkpoint-id" else base),
    )

    prepared = ohan_tts.prepare("checkpoint-id", tmp_path / "prepared")
    config = json.loads((prepared / "config.json").read_text())
    metadata = json.loads((prepared / "integration_metadata.json").read_text())
    assert config["architectures"] == ["OhanTTSForConditionalGeneration"]
    assert config["model_type"] == "ohan_tts"
    assert metadata["checkpoint"] == "checkpoint-id"
    assert (prepared / "model.safetensors").is_symlink()
    assert (prepared / "model.safetensors").resolve() == weights.resolve()
    assert ohan_tts.prepare("checkpoint-id", prepared) == prepared


def test_scheduler_request_uses_evaluation_repetition_penalty() -> None:
    request_builder, _ = make_scheduler_adapters()
    state = OhanTTSState(
        prompt_token_ids=[1, 2, 3], max_new_tokens=12, temperature=0.0
    )
    payload = StagePayload(
        request_id="test",
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )

    request = request_builder(payload)

    assert request.req.sampling_params.repetition_penalty == DEFAULT_REPETITION_PENALTY


def test_scheduler_request_preserves_overridden_generation_params() -> None:
    request_builder, _ = make_scheduler_adapters()
    state = OhanTTSState(
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=23,
        temperature=0.4,
        top_p=0.85,
        top_k=40,
        seed=123,
        repetition_penalty=1.25,
    )
    payload = StagePayload(
        request_id="test-override",
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )

    request = request_builder(payload)

    sampling = request.req.sampling_params
    assert sampling.max_new_tokens == 23
    assert sampling.temperature == 0.4
    assert sampling.top_p == 0.85
    assert sampling.top_k == 40
    assert sampling.sampling_seed == 123
    assert sampling.repetition_penalty == 1.25
