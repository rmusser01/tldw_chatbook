"""OmniVoice diffusion-LM sampling loop — numpy port.

Faithful port of ``_generate_iterative`` and its helpers
(``_predict_tokens_with_scoring``, ``_filter_top_k``, ``_gumbel_sample``,
``_get_time_steps``) from k2-fsa/omnivoice ``omnivoice/models/omnivoice.py``
(Apache-2.0; https://github.com/k2-fsa/omnivoice). Nothing is taken from the
unlicensed AFun9/Omnivoice-onnx repository.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

AUDIO_VOCAB_SIZE = 1025
AUDIO_MASK_ID = 1024
NUM_CODEBOOK = 8


class OmniVoiceSamplingCancelled(RuntimeError):
    """Raised when a cooperative cancel is observed between steps."""


class LMBatchRunner(Protocol):
    """One batched LM forward pass (Task 6 adapts the ORT session to this)."""

    def run(
        self, input_ids: np.ndarray, audio_mask: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Run one forward pass over the conditional + unconditional batch.

        Args:
            input_ids: ``(B, 8, S)`` int64 token/codec ids.
            audio_mask: ``(B, 8, S)`` bool; True over audio (reference and
                target) positions. Adapters may collapse the codebook axis.
            attention_mask: ``(B, 1, S, S)`` bool attention mask.

        Returns:
            ``(B, 8, S, 1025)`` float32 logits.
        """
        ...


@dataclass(frozen=True)
class OmniVoiceSamplerConfig:
    """Upstream ``OmniVoiceGenerationConfig`` decoding defaults."""

    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    seed: int | None = None
    class_top_ratio: float = 0.1  # upstream hardcodes ratio=0.1 in _filter_top_k


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log_softmax (max-shifted)."""
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def _gumbel(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    # Upstream: -torch.log(-torch.log(u + 1e-10) + 1e-10)
    u = rng.uniform(size=shape)
    return -np.log(-np.log(u + 1e-10) + 1e-10)


def _build_attention_mask(seq_len: int, u_len: int) -> np.ndarray:
    """(2, 1, S, S) bool: cond row fully True; uncond row trailing block + pad diag.

    Upstream keeps the uncond tokens at the front and gives the pad tail a
    self-attention-only diagonal (``pad_diag = torch.arange(u_len, max_c_len)``)
    so padding never leaks into the token block and RoPE positions stay put.
    We keep the tokens at the tail — the same positions they occupy in the
    conditional row, with identical in-block relative distances — so the
    isolated diagonal belongs to the leading pad prefix instead.
    """
    mask = np.zeros((2, 1, seq_len, seq_len), dtype=np.bool_)
    mask[0, 0] = True
    mask[1, 0, seq_len - u_len:, seq_len - u_len:] = True
    idx = np.arange(seq_len - u_len)
    mask[1, 0, idx, idx] = True
    return mask


def run_diffusion_sampling(
    lm: LMBatchRunner,
    prompt_ids: np.ndarray,      # (8, S) int64 — target tail pre-filled with AUDIO_MASK_ID
    target_len: int,
    *,
    config: OmniVoiceSamplerConfig,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    num_codebook: int = NUM_CODEBOOK,
    prompt_audio_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run the diffusion unmasking loop; returns codes ``(8, target_len)`` int64.

    Port of upstream ``_generate_iterative`` for a single sequence (batch of 2
    = [conditional, unconditional]). ``lm.run`` receives ``input_ids`` and
    ``audio_mask`` shaped ``(2, num_codebook, S)`` and a ``(2, 1, S, S)`` bool
    ``attention_mask``; Task 6 adapts these to the ONNX session inputs.

    Args:
        lm: Batched forward pass returning ``(2, 8, S, 1025)`` float32 logits.
        prompt_ids: ``(8, S)`` prompt; the last ``target_len`` columns are the
            mask-filled target region.
        target_len: Number of target audio slots at the tail of ``prompt_ids``.
        config: Sampler configuration.
        cancel_check: Polled between steps; ``True`` aborts with
            :class:`OmniVoiceSamplingCancelled`.
        progress: Called once per completed step as ``progress(k, n)``.
        num_codebook: Number of audio codebook rows.
        prompt_audio_mask: Optional ``(8, S)`` bool audio mask of ``prompt_ids``
            (True over reference and target audio slots, as built by Task 4's
            ``build_prompt_inputs``). Mirrors upstream
            ``batch_audio_mask[i, :c_len] = inp["audio_mask"]``. When omitted,
            only the target tail is audio-marked (text-only prompts).

    Returns:
        Sampled codes ``(num_codebook, target_len)`` int64 — no mask ids remain.

    Raises:
        ValueError: If ``prompt_audio_mask`` does not match ``prompt_ids``.
        OmniVoiceSamplingCancelled: If ``cancel_check`` returns True between
            steps.
    """
    rng = np.random.default_rng(config.seed)
    prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
    seq_len = prompt_ids.shape[1]
    total = target_len * num_codebook

    # Timestep schedule (upstream _get_time_steps): t' = a·t / (1 + (a−1)·t)
    # over linspace(0, 1, n+1); each step unmasks ceil(total · Δt'), the final
    # step everything remaining (quota is clamped by what is still masked).
    t = np.linspace(0.0, 1.0, config.num_step + 1)
    a = config.t_shift
    ts = a * t / (1.0 + (a - 1.0) * t)
    per_step = np.ceil(total * np.diff(ts)).astype(np.int64)
    per_step[-1] = total

    attention = _build_attention_mask(seq_len, target_len)
    if prompt_audio_mask is None:
        audio_mask = np.zeros(prompt_ids.shape, dtype=np.bool_)
        audio_mask[:, seq_len - target_len:] = True
    else:
        prompt_audio_mask = np.asarray(prompt_audio_mask, dtype=np.bool_)
        if prompt_audio_mask.shape != prompt_ids.shape:
            raise ValueError(
                f"prompt_audio_mask must have shape {prompt_ids.shape}, "
                f"got {prompt_audio_mask.shape}"
            )
        audio_mask = prompt_audio_mask.copy()

    # B = 1 (single-flight). Batch of 2 = [conditional, unconditional].
    cond = prompt_ids.copy()
    tokens = cond[:, seq_len - target_len:].copy()
    # Uncond row: length-S array; trailing target_len positions mirror the
    # sampled tokens, the prefix is pad isolated by the attention mask's pad
    # diagonal (upstream pads with audio_mask_id — "Or any other tokens" —
    # and slices the trailing u_len positions of the prompt into its uncond
    # row, which is equivalent under relative RoPE attention).
    prefix_len = seq_len - target_len
    pad = np.full((num_codebook, prefix_len), AUDIO_MASK_ID, dtype=np.int64)
    uncond_audio_mask = np.zeros(prompt_ids.shape, dtype=np.bool_)
    uncond_audio_mask[:, prefix_len:] = audio_mask[:, prefix_len:]
    unmasked = np.zeros(tokens.shape, dtype=np.bool_)
    layer_penalty = (
        np.arange(num_codebook, dtype=np.float32)[:, None] * config.layer_penalty_factor
    )

    for step in range(config.num_step):
        if cancel_check is not None and cancel_check():
            raise OmniVoiceSamplingCancelled("omnivoice sampling cancelled between steps")

        uncond = np.concatenate([pad, tokens], axis=1)
        batch_ids = np.stack([cond, uncond], axis=0)              # (2, 8, S)
        batch_audio = np.stack([audio_mask, uncond_audio_mask], axis=0)
        logits = lm.run(batch_ids, batch_audio, attention)        # (2, 8, S, V)

        # CFG merge (upstream _predict_tokens_with_scoring), including the
        # SECOND log_softmax over the merged row — its per-position
        # normalization shifts confidence, which feeds position selection.
        cond_lp = _log_softmax(logits[0:1, :, -target_len:].astype(np.float32))
        if config.guidance_scale != 0:
            uncond_lp = _log_softmax(logits[1:2, :, -target_len:].astype(np.float32))
            merged = _log_softmax(
                cond_lp + config.guidance_scale * (cond_lp - uncond_lp)
            )
        else:
            merged = cond_lp
        merged[..., AUDIO_MASK_ID] = -np.inf

        if config.class_temperature > 0:
            # Upstream _filter_top_k: keep exactly ceil(ratio·V) entries, then
            # _gumbel_sample(logits, τ) = logits/τ + noise, argmax.
            k = max(1, math.ceil(config.class_top_ratio * AUDIO_VOCAB_SIZE))
            top_idx = np.argpartition(merged, -k, axis=-1)[..., -k:]
            keep = np.zeros(merged.shape, dtype=np.bool_)
            np.put_along_axis(keep, top_idx, True, axis=-1)
            filtered = np.where(keep, merged, -np.inf)
            choice = np.argmax(
                filtered / config.class_temperature + _gumbel(rng, filtered.shape),
                axis=-1,
            )
        else:
            choice = np.argmax(merged, axis=-1)
        conf = np.max(merged, axis=-1)

        scores = conf - layer_penalty
        if config.position_temperature > 0:
            # Upstream _gumbel_sample(scores, τ): scores/τ + noise (adding
            # τ·noise instead is order-equivalent but not the ported form).
            scores = scores / config.position_temperature + _gumbel(rng, scores.shape)
        scores = np.where(unmasked, -np.inf, scores).ravel()

        quota = int(min(per_step[step], int((~unmasked).sum())))
        if quota <= 0:
            if progress is not None:
                progress(step + 1, config.num_step)
            continue
        pick = np.argpartition(scores, -quota)[-quota:]

        flat_tokens, flat_unmasked = tokens.ravel(), unmasked.ravel()
        flat_tokens[pick] = choice.ravel()[pick]
        flat_unmasked[pick] = True
        tokens = flat_tokens.reshape(tokens.shape)
        unmasked = flat_unmasked.reshape(unmasked.shape)

        cond[:, -target_len:] = tokens  # uncond is rebuilt from pad+tokens next step
        if progress is not None:
            progress(step + 1, config.num_step)
        if unmasked.all():
            break

    return tokens.astype(np.int64)
