# SPDX-License-Identifier: Apache-2.0
"""SGLang model adapter for OhanTTS's fixed-length autoregressive decoder."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

import torch

from sglang_omni.models.higgs_tts.model import HiggsTTSModel
from sglang_omni.models.higgs_tts.sampler import K_MAX, sample_independent_batched
from sglang_omni.models.ohan_tts.sampling import (
    DEFAULT_REPETITION_PENALTY,
)
from sglang_omni.models.ohan_tts.weight_loader import map_checkpoint_name


class OhanTTSSGLangModel(HiggsTTSModel):
    """Qwen3 backbone with jointly predicted eight-codebook codec frames.

    One SGLang decode step predicts all eight codec IDs. Those IDs are summed
    through the fused multi-codebook embedding on the next step, matching the
    training/inference recurrence recorded in the checkpoint configuration. Sampling
    defaults to greedy selection and the evaluation's repetition penalty, with
    per-request sampling overrides.
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.register_buffer(
            "_seen_audio_codes",
            torch.zeros(
                self.sampler_pool.max_batch_size,
                self.num_codebooks,
                self.codebook_vocab_size,
                dtype=torch.bool,
                device=self.sampler_pool.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_request_repetition_penalties",
            torch.full(
                (self.sampler_pool.max_batch_size,),
                DEFAULT_REPETITION_PENALTY,
                dtype=torch.float32,
                device=self.sampler_pool.device,
            ),
            persistent=False,
        )

    def acquire_row(self, req_id: str) -> int:
        is_new = req_id not in self.rid_to_row
        row = super().acquire_row(req_id)
        if is_new:
            self._seen_audio_codes[row].zero_()
            self._request_repetition_penalties[row] = DEFAULT_REPETITION_PENALTY
        return row

    def release_row(self, req_id: str) -> None:
        row = self.rid_to_row.get(req_id)
        if row is not None:
            self._seen_audio_codes[row].zero_()
            self._request_repetition_penalties[row] = DEFAULT_REPETITION_PENALTY
        super().release_row(req_id)

    def set_request_repetition_penalty(self, req_id: str, penalty: float) -> None:
        row = self.acquire_row(req_id)
        self._request_repetition_penalties[row] = float(penalty)

    def _select_codes(
        self,
        logits: torch.Tensor,
        rows: torch.Tensor,
        *,
        gen_params=None,
        temperature: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
        top_k_buf: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        step_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_greedy_defaults = gen_params is None and temperature is None
        seen = self._seen_audio_codes.index_select(0, rows)
        penalties = self._request_repetition_penalties.index_select(0, rows).view(-1, 1, 1)
        penalized = torch.where(logits < 0, logits * penalties, logits / penalties)
        scores = torch.where(seen, penalized, logits)
        scores[..., self.config.audio_mask_id] = -float("inf")
        if gen_params is not None:
            temperature = torch.tensor(
                [float(params.temperature) for params in gen_params],
                dtype=torch.float32,
                device=logits.device,
            )
            top_p_values = [params.top_p for params in gen_params]
            if any(value is not None and float(value) < 1.0 for value in top_p_values):
                top_p = torch.tensor(
                    [1.0 if value is None else float(value) for value in top_p_values],
                    dtype=torch.float32,
                    device=logits.device,
                )
            else:
                top_p = None
            top_k_values = [params.top_k for params in gen_params]
            if any(value is not None and 0 < int(value) < K_MAX for value in top_k_values):
                top_k_buf = torch.tensor(
                    [K_MAX if value is None else int(value) for value in top_k_values],
                    dtype=torch.long,
                    device=logits.device,
                )
            else:
                top_k_buf = None
        if temperature is None:
            temperature = torch.zeros(rows.numel(), dtype=torch.float32, device=logits.device)
        if seeds is None:
            seeds = self.sampler_pool.seeds.index_select(0, rows)
        if step_count is None:
            step_count = self.sampler_pool.step_count.index_select(0, rows)
        if use_greedy_defaults or (
            gen_params is not None
            and all(float(params.temperature) <= 1e-5 for params in gen_params)
        ):
            codes = scores.argmax(dim=-1).to(torch.long)
        else:
            codes = sample_independent_batched(
                scores,
                temperature=temperature,
                top_p=top_p,
                top_k_buf=top_k_buf,
                seeds_B=seeds,
                step_B=step_count,
            ).to(torch.long)
        seen.scatter_(2, codes.unsqueeze(-1), True)
        self._seen_audio_codes.index_copy_(0, rows, seen)
        return codes

    @torch.no_grad()
    def decode_codebooks_batch(self, hidden_states_BD, req_ids, gen_params):
        if len(req_ids) != hidden_states_BD.shape[0] or len(req_ids) != len(gen_params):
            raise ValueError("SGLang request count does not match the decode batch")

        logits = self.modality_head.generate(hidden_states_BD).float()
        rows = torch.tensor(
            [self.acquire_row(req_id) for req_id in req_ids],
            dtype=torch.long,
            device=logits.device,
        )
        codes = self._select_codes(logits, rows, gen_params=gen_params)
        self.sampler_pool.last_codes[rows] = codes
        self.sampler_pool.delay_count[rows] += 1
        self.sampler_pool.step_count[rows] += 1
        for req_id, frame in zip(req_ids, codes, strict=True):
            self.output_codes.setdefault(req_id, []).append(frame.detach())

        return torch.zeros(
            (len(req_ids), self.backbone.config.vocab_size),
            dtype=torch.float32,
            device=codes.device,
        )

    @torch.no_grad()
    def decode_codebooks_batch_cg(self, hidden_states_BD: torch.Tensor) -> torch.Tensor:
        """CUDA-graph-safe frame sampling and recurrence state update."""
        batch_size = hidden_states_BD.shape[0]
        logits = self.modality_head.generate(hidden_states_BD).float()
        rows = self.cg_row_indices[:batch_size]
        codes = self._select_codes(
            logits,
            rows,
            temperature=self.cg_temperature[:batch_size],
            top_p=self.cg_top_p[:batch_size],
            top_k_buf=self.cg_top_k_buf[:batch_size],
            seeds=self.cg_active_seeds[:batch_size],
            step_count=self.cg_active_step_count[:batch_size],
        )
        self.cg_was_done[:batch_size] = self.cg_active_generation_done[:batch_size]
        self.cg_active_last_codes[:batch_size] = codes
        self.cg_active_delay_count[:batch_size] += 1
        self.cg_active_step_count[:batch_size] += 1
        self.cg_codes_BN[:batch_size] = codes
        return torch.zeros(
            (batch_size, self.backbone.config.vocab_size),
            dtype=torch.float32,
            device=codes.device,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load Qwen3 weights and both untied multi-codebook matrices."""
        backbone_weights: list[Tuple[str, torch.Tensor]] = []
        own_parameters = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()

        for source_name, tensor in weights:
            destination = map_checkpoint_name(source_name)
            if destination is None:
                continue
            if destination.startswith("backbone."):
                backbone_weights.append((destination.removeprefix("backbone."), tensor))
                continue
            if destination not in own_parameters:
                continue
            parameter = own_parameters[destination]
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {source_name} -> {destination}: "
                    f"expected {tuple(parameter.shape)}, got {tuple(tensor.shape)}"
                )
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(destination)

        self.backbone.load_weights(iter(backbone_weights))
        required = {
            "multimodal_embedding.modality_embedding_0.weight",
            "modality_head.weight",
        }
        if not required.issubset(loaded):
            raise RuntimeError(f"Missing OhanTTS codec weights: {sorted(required - loaded)}")
        return loaded


__all__ = ["OhanTTSSGLangModel"]
