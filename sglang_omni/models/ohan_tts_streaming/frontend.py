# SPDX-License-Identifier: Apache-2.0
"""Reference-audio preparation for OhanTTS-Streaming."""

from __future__ import annotations

from dataclasses import dataclass

import re

import torch
from transformers import AutoTokenizer

from sglang_omni.models.ohan_tts.frontend import OhanTTSFrontend


@dataclass(frozen=True)
class StreamingPrompt:
    prompt_token_ids: list[int]
    continuation_ids: list[int]
    reference_codes: list[list[int]]
    reference_rms: float | None
    sample_rate: int


class OhanTTSStreamingFrontend:
    """Build the trained ``word + <DPH>`` prompt without native length rules."""

    def __init__(
        self,
        base_model_path: str,
        streaming_model_path: str,
        *,
        codec_device: str = "cpu",
    ) -> None:
        self.frontend = OhanTTSFrontend(base_model_path, codec_device=codec_device)
        self.model = self.frontend.model
        self.model.text_tokenizer = AutoTokenizer.from_pretrained(streaming_model_path)
        tokenizer = self.model.text_tokenizer
        self.duration_token_id = tokenizer.convert_tokens_to_ids("<DPH>")
        if self.duration_token_id is None or tokenizer.encode(
            "<DPH>", add_special_tokens=False
        ) != [self.duration_token_id]:
            raise ValueError("OhanTTS-Streaming tokenizer must define <DPH>")

    @torch.inference_mode()
    def prepare_prompt(
        self,
        *,
        text: str,
        reference_waveform: torch.Tensor,
        reference_sample_rate: int,
        reference_text: str,
        language: str,
        instructions: str | None,
    ) -> StreamingPrompt:
        """Encode the reference once and queue all target text."""
        words = re.findall(r"\S+(?:\s+|$)", text)
        if "<DPH>" in text:
            raise ValueError("Target text must not contain <DPH>")
        if not words:
            raise ValueError("OhanTTS-Streaming requires target text")
        clone_prompt = self.model.create_voice_clone_prompt(
            ref_audio=(reference_waveform, int(reference_sample_rate)),
            ref_text=reference_text,
            preprocess_prompt=True,
        )
        reference_codes = clone_prompt.ref_audio_tokens
        prepared = self.model._prepare_inference_inputs(
            "",
            0,
            ref_text=clone_prompt.ref_text,
            ref_audio_tokens=reference_codes,
            lang=language,
            instruct=instructions,
            denoise=True,
        )
        input_ids = prepared["input_ids"][0].clone()
        audio_mask = prepared["audio_mask"][0]
        audio_positions = audio_mask.nonzero(as_tuple=True)[0]
        codes = input_ids[:, audio_positions].transpose(0, 1).contiguous()
        input_ids[:, audio_positions] = -100
        prompt_token_ids = input_ids[0].cpu().tolist()
        tokenizer = self.model.text_tokenizer
        chunks = [
            tokenizer.encode(
                f"<|text_start|>{word}<|text_end|>", add_special_tokens=False
            ) + [int(self.duration_token_id)]
            for word in words
        ]
        prompt_token_ids.extend(chunks[0])
        return StreamingPrompt(
            prompt_token_ids=prompt_token_ids,
            continuation_ids=[token for chunk in chunks[1:] for token in chunk],
            reference_codes=codes.cpu().tolist(),
            reference_rms=clone_prompt.ref_rms,
            sample_rate=int(self.model.sampling_rate),
        )

    def decode(self, codec_frames_t_c: torch.Tensor) -> torch.Tensor:
        return self.frontend.decode(
            codec_frames_t_c,
            reference_rms=None,
            postprocess=False,
        )


__all__ = ["OhanTTSStreamingFrontend", "StreamingPrompt"]
