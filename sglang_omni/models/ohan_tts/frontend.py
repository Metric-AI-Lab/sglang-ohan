# SPDX-License-Identifier: Apache-2.0
"""Lightweight OhanTTS text/audio front end without a second Qwen3 model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer, HiggsAudioV2TokenizerModel

from omnivoice.models.omnivoice import OmniVoice as OhanTTSCore
from omnivoice.utils.duration import RuleDurationEstimator


class _TokenizerOnlyLanguageModel(nn.Module):
    """Satisfy OhanTTS construction without loading its unused Qwen3 LLM."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa")


class OhanTTSFrontend:
    """Tokenizer, reference codec and decoder helpers for a given device."""

    def __init__(self, model_path: str, *, codec_device: str = "cpu") -> None:
        base_path = Path(model_path)
        if not base_path.is_dir():
            base_path = Path(snapshot_download(model_path))
        config = AutoConfig.from_pretrained(base_path)
        self.model = OhanTTSCore(config, llm=_TokenizerOnlyLanguageModel())
        self.model.text_tokenizer = AutoTokenizer.from_pretrained(base_path)

        audio_path = base_path / "audio_tokenizer"
        self.model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
            audio_path,
            device_map=codec_device,
            # The decoder calls .numpy() on codec output; bf16 arrays
            # cannot be converted by NumPy, so keep the codec in fp32.
            dtype=torch.float32,
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(audio_path)
        self.model.sampling_rate = int(feature_extractor.sampling_rate)
        self.model.feature_extractor = feature_extractor
        self.model.duration_estimator = RuleDurationEstimator()
        self.model.eval()

    @torch.inference_mode()
    def prepare_clone(
        self,
        *,
        text: str,
        reference_waveform: torch.Tensor | None,
        reference_sample_rate: int | None,
        reference_text: str | None,
        language: str = "hy",
        max_target_frames: int = 768,
        instructions: str | None = None,
    ) -> tuple[list[int], list[list[int]], int, float | None]:
        """Return prompt IDs, reference frames [T,8], target frame count and RMS."""
        if not text.strip():
            raise ValueError("Speech request text is empty")
        if max_target_frames < 1:
            raise ValueError("max_target_frames must be positive")

        prompt = None
        if reference_waveform is not None:
            if reference_sample_rate is None:
                raise ValueError("reference sample rate is required")
            # Empty text disables the implicit Whisper-ASR download.
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=(reference_waveform, int(reference_sample_rate)),
                ref_text=reference_text or "",
                preprocess_prompt=True,
            )

        ref_tokens = prompt.ref_audio_tokens if prompt is not None else None
        ref_text = prompt.ref_text if prompt is not None else None
        frame_count = self.model._estimate_target_tokens(
            text.strip(),
            ref_text,
            ref_tokens.size(-1) if ref_tokens is not None else None,
        )
        frame_count = min(int(frame_count), int(max_target_frames))
        prepared = self.model._prepare_inference_inputs(
            text.strip(),
            frame_count,
            ref_text=ref_text,
            ref_audio_tokens=ref_tokens,
            lang=language,
            instruct=instructions,
            denoise=True,
        )

        input_ids = prepared["input_ids"][0]
        audio_mask = prepared["audio_mask"][0]
        prefix_length = input_ids.shape[-1] - frame_count
        prefix_ids = input_ids[:, :prefix_length].clone()
        prefix_audio_mask = audio_mask[:prefix_length]
        ref_positions = prefix_audio_mask.nonzero(as_tuple=True)[0]

        reference_codes: list[list[int]] = []
        if ref_positions.numel():
            codes = input_ids[:, ref_positions].transpose(0, 1).contiguous()
            reference_codes = codes.cpu().tolist()
            prefix_ids[:, ref_positions] = -100

        return (
            prefix_ids[0].cpu().tolist(),
            reference_codes,
            frame_count,
            prompt.ref_rms if prompt is not None else None,
        )

    @torch.inference_mode()
    def decode(
        self,
        codec_frames_t_c: torch.Tensor,
        *,
        reference_rms: float | None,
        postprocess: bool = True,
    ) -> torch.Tensor:
        from omnivoice.models.omnivoice import (
            OmniVoiceGenerationConfig as OhanTTSGenerationConfig,
        )

        generation_config = OhanTTSGenerationConfig(
            postprocess_output=postprocess,
        )
        waveform = self.model._decode_and_post_process(
            codec_frames_t_c.transpose(0, 1).contiguous(),
            reference_rms,
            generation_config,
        )
        return torch.from_numpy(waveform).to(torch.float32)


__all__ = ["OhanTTSFrontend"]
