# SPDX-License-Identifier: Apache-2.0
"""SGLang model implementation for OhanTTS-Streaming."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

import torch
from torch import nn
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

from sglang_omni.models.ohan_tts.model import OhanTTSSGLangModel


class OhanTTSStreamingSGLangModel(OhanTTSSGLangModel):
    """Append text tokens and codec frames to one SGLang sequence."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        hidden_size = int(config.text_config.hidden_size)
        self.duration_head = nn.Linear(
            hidden_size,
            int(getattr(config, "num_duration_classes", 128)),
            bias=True,
        )
        self.duration_token_id = int(getattr(config, "duration_token_id", 151676))
        pool_size = self.sampler_pool.max_batch_size
        device = self.sampler_pool.device
        for name, shape in (
            ("continuations", (pool_size, 4096)),
            ("continuation_lengths", (pool_size,)),
            # Text cursor, remaining audio frames, and whether to read text.
            ("decode_state", (pool_size, 3)),
            ("active_decode_state", (pool_size, 3)),
        ):
            self.register_buffer(
                name, torch.zeros(shape, dtype=torch.long, device=device),
                persistent=False,
            )

    def set_continuation(self, request_id, token_ids):
        row = self.acquire_row(request_id)
        length = len(token_ids)
        if length > self.continuations.shape[1]:
            raise ValueError("OhanTTS-Streaming continuation exceeds the token limit")
        self.continuations[row, :length] = self.continuations.new_tensor(token_ids)
        self.continuation_lengths[row] = length
        self.decode_state[row].zero_()

    def predict_duration(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return predicted frame counts (one through number of classes)."""
        return self.duration_head(hidden_states).argmax(dim=-1).to(torch.long) + 1

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor | None = None,
        omni_prefill_rids: list[str] | None = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        is_decode = self.is_decode_step(forward_batch)
        output = super().forward(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            omni_prefill_rids=omni_prefill_rids,
            **kwargs,
        )
        if is_decode or not omni_prefill_rids:
            return output

        lengths = getattr(forward_batch, "extend_seq_lens", None)
        if lengths is None:
            return output
        last_indices = torch.cumsum(lengths.to(input_ids.device), dim=0) - 1
        flat_input_ids = input_ids.reshape(-1)
        last_input_ids = flat_input_ids.index_select(0, last_indices.to(torch.long))
        duration_rows = torch.nonzero(
            last_input_ids == self.duration_token_id, as_tuple=False
        ).flatten()
        if duration_rows.numel() == 0:
            return output

        selected_hidden = output.hidden_states.index_select(0, duration_rows)
        predicted_frames = self.predict_duration(selected_hidden)
        for local_index, frame_count in zip(
            duration_rows.tolist(), predicted_frames.tolist(), strict=True
        ):
            req_id = omni_prefill_rids[local_index]
            row = self.acquire_row(req_id)
            remaining = int(frame_count) - 1
            self.decode_state[row, 1] = remaining
            if remaining == 0 and self.continuation_lengths[row] == 0:
                self.sampler_pool.generation_done[row] = True
        return output

    def decode_step_embeds_cg(self, input_ids, batch_size):
        embeddings = super().decode_step_embeds_cg(input_ids, batch_size)
        rows = self.cg_row_indices[:batch_size]
        cursor, _, reading_text = self.active_decode_state[:batch_size].unbind(-1)
        tokens = self.continuations[rows, cursor.clamp_max(self.continuations.shape[1] - 1)]
        text_embeddings = self.get_input_embeddings()(tokens)
        return torch.where(reading_text.bool().unsqueeze(-1), text_embeddings, embeddings)

    @torch.no_grad()
    def decode_codebooks_batch_cg(self, hidden_states_BD: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states_BD.shape[0]
        active = slice(0, batch_size)
        rows = self.cg_row_indices[active]
        state = self.active_decode_state[active]
        cursor, remaining, reading_text = state.unbind(-1)
        reading_text = reading_text.bool()
        tokens = self.continuations[rows, cursor.clamp_max(self.continuations.shape[1] - 1)]
        lengths = self.continuation_lengths[rows]
        active_rows = (rows != self.padding_row) & ~self.cg_active_generation_done[active]
        duration_step = reading_text & (tokens == self.duration_token_id)
        emit_audio = active_rows & (duration_step | (~reading_text & (remaining > 0)))
        next_cursor = cursor + (reading_text & active_rows).long()
        next_remaining = torch.where(
            duration_step & active_rows, self.predict_duration(hidden_states_BD), remaining
        ) - emit_audio.long()
        next_reading_text = active_rows & (
            (reading_text & ~duration_step) | (~reading_text & (remaining == 0))
        )

        seen = self._seen_audio_codes.index_select(0, rows)
        codes = self._select_codes(
            self.modality_head.generate(hidden_states_BD).float(),
            rows,
            temperature=self.cg_temperature[active],
            top_p=self.cg_top_p[active],
            top_k_buf=self.cg_top_k_buf[active],
            seeds=self.cg_active_seeds[active],
            step_count=self.cg_active_step_count[active],
        )
        updated_seen = self._seen_audio_codes.index_select(0, rows)
        self._seen_audio_codes.index_copy_(
            0, rows, torch.where(emit_audio[:, None, None], updated_seen, seen)
        )
        self.cg_active_last_codes[active] = torch.where(
            emit_audio[:, None], codes, self.cg_active_last_codes[active]
        )
        self.cg_codes_BN[active] = codes
        self.cg_active_delay_count[active] += emit_audio.long()
        self.cg_active_step_count[active] += emit_audio.long()
        self.cg_was_done[active] = ~emit_audio
        self.cg_active_generation_done[active] |= (
            emit_audio & (next_remaining == 0) & (next_cursor == lengths)
        )
        state.copy_(torch.stack((next_cursor, next_remaining, next_reading_text.long()), dim=-1))
        return hidden_states_BD.new_zeros(
            (batch_size, self.backbone.config.vocab_size), dtype=torch.float32
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load standard Ohan weights and the additional trained DPH head."""
        weights = list(weights)
        duration_weights: dict[str, torch.Tensor] = {}
        base_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            prefix = "duration_head."
            if name.startswith(prefix):
                duration_weights[name.removeprefix(prefix)] = tensor
            else:
                base_weights.append((name, tensor))

        loaded = super().load_weights(base_weights)
        parameters = dict(self.duration_head.named_parameters())
        for name, tensor in duration_weights.items():
            parameter = parameters.get(name)
            if parameter is None:
                raise ValueError(f"Unexpected duration-head parameter: {name}")
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for duration_head.{name}: expected "
                    f"{tuple(parameter.shape)}, got {tuple(tensor.shape)}"
                )
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(f"duration_head.{name}")

        required = {"weight", "bias"}
        if not required.issubset(duration_weights):
            missing = sorted(required - duration_weights.keys())
            raise RuntimeError(f"Missing OhanTTS-Streaming duration-head weights: {missing}")
        return loaded


EntryClass = OhanTTSStreamingSGLangModel

__all__ = ["EntryClass", "OhanTTSStreamingSGLangModel"]
