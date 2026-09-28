"""Diffusion sampling loop — mechanics against a deterministic fake LM."""

import numpy as np
import pytest

from tldw_chatbook.TTS.omnivoice_sampler import (
    OmniVoiceSamplerConfig,
    OmniVoiceSamplingCancelled,
    run_diffusion_sampling,
)

V, MASK, C = 1025, 1024, 8


class FakeLM:
    """Deterministic: every position wants token (position_index % V)."""

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.calls: list[tuple[tuple[int, ...], ...]] = []

    def run(self, input_ids, audio_mask, attention_mask):
        b, c, s = input_ids.shape
        self.calls.append((input_ids.shape, attention_mask.shape))
        # CFG contract: batch is 2B — first half cond, second half uncond
        assert b % 2 == 0
        want = np.tile(np.arange(s, dtype=np.float32) % V, (b, c, 1))
        logits = np.zeros((b, c, s, V), dtype=np.float32)
        np.put_along_axis(logits, (want % V).astype(np.int64)[..., None], 5.0, axis=-1)
        return logits


class RecordingLM(FakeLM):
    """FakeLM logits plus full capture of every call's batched inputs."""

    def __init__(self, seq_len: int):
        super().__init__(seq_len)
        self.batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def run(self, input_ids, audio_mask, attention_mask):
        self.batches.append(
            (input_ids.copy(), audio_mask.copy(), attention_mask.copy())
        )
        return super().run(input_ids, audio_mask, attention_mask)


class CfgNormalizationLM:
    """CFG fixture separating raw from re-normalized merged confidence.

    cond row (batch index 0): 5.0 at token 3 on the first two target slots,
    flat elsewhere. uncond row (batch index 1): 100.0 at token 12 on the last
    two target slots, flat elsewhere.

    The raw merge ``c_lp + g*(c_lp - u_lp)`` peaks on the ultra-uncond slots
    (≈ 3·c_lp − 2·u_lp ≈ 179 there, vs ≈ 8 on the sharp cond slots), but after
    upstream's second ``log_softmax`` over the merged row the sharp cond slots
    win (≈ 0 vs ≈ −6.9): the ultra-uncond rows are mass-dominated flats.
    Upstream ``_predict_tokens_with_scoring`` normalizes before scoring.
    """

    def __init__(self, target_len: int):
        self.target_len = target_len
        self.batches: list[np.ndarray] = []

    def run(self, input_ids, audio_mask, attention_mask):
        self.batches.append(input_ids.copy())
        b, c, s = input_ids.shape
        logits = np.zeros((b, c, s, V), dtype=np.float32)
        start = s - self.target_len
        logits[0, :, start : start + 2, 3] = 5.0
        logits[1, :, start + 2 :, 12] = 100.0
        return logits


def _prompt(target_len: int) -> np.ndarray:
    prefix = np.tile(np.arange(1, 6, dtype=np.int64), (C, 1))
    return np.concatenate([prefix, np.full((C, target_len), MASK, dtype=np.int64)], axis=1)


def test_all_positions_unmasked_with_expected_values() -> None:
    out = run_diffusion_sampling(FakeLM(15), _prompt(10), 10, config=OmniVoiceSamplerConfig(num_step=4))
    assert out.shape == (C, 10)
    assert not (out == MASK).any()
    # FakeLM wants token = global position % V; target starts at global pos 5
    assert (out[0] == (np.arange(5, 15) % V)).all()


def test_batch_is_cond_plus_uncond() -> None:
    lm = FakeLM(15)
    run_diffusion_sampling(lm, _prompt(10), 10, config=OmniVoiceSamplerConfig(num_step=2))
    ids_shape, mask_shape = lm.calls[0]
    assert ids_shape[0] == 2  # B=1 → 2B=2
    assert mask_shape == (2, 1, 15, 15)  # (2B, 1, S, S) block-diagonal


def test_uncond_attention_isolates_pad_prefix() -> None:
    # Upstream gives the uncond row's padding a self-attention-only diagonal
    # (pad_diag = torch.arange(u_len, max_c_len) for its front-padded layout);
    # with our pad-prefix layout the diagonal must cover the leading prefix.
    lm = RecordingLM(15)
    run_diffusion_sampling(lm, _prompt(10), 10, config=OmniVoiceSamplerConfig(num_step=2, seed=1))
    _, _, att = lm.batches[0]
    assert att[0, 0].all()  # cond row attends everywhere
    uncond = att[1, 0]
    assert uncond[5:, 5:].all()  # target block attends within itself...
    assert not uncond[5:, :5].any()  # ...never to the pad prefix
    assert not uncond[:5, 5:].any()
    assert (uncond[:5, :5] == np.eye(5, dtype=bool)).all()  # pads self-attend only


def test_step_schedule_counts() -> None:
    lm = RecordingLM(15)
    out = run_diffusion_sampling(
        lm, _prompt(10), 10, config=OmniVoiceSamplerConfig(num_step=4, seed=1)
    )
    committed = [int((ids[0, :, -10:] != MASK).sum()) for ids, _, _ in lm.batches]
    # t_shift=0.1 over linspace(0,1,5): ceil(total·Δt') per step → [3, 5, 12, ·],
    # final step fills everything remaining → cumulative [3, 8, 20, 80].
    assert committed == [0, 3, 8, 20]
    assert out.shape == (C, 10)
    assert not (out == MASK).any()


def test_uncond_row_mirrors_committed_tokens_next_step() -> None:
    lm = RecordingLM(15)
    run_diffusion_sampling(
        lm, _prompt(10), 10, config=OmniVoiceSamplerConfig(num_step=2, seed=3)
    )
    ids, audio, _ = lm.batches[1]
    # upstream writes sample_tokens into both the cond tail and the uncond row
    assert (ids[1, :, -10:] == ids[0, :, -10:]).all()
    assert (ids[1, :, :5] == MASK).all()  # pad prefix stays audio_mask_id (upstream pad)
    assert not audio[1, :, :5].any()  # uncond audio mask: target tail only
    assert audio[1, :, 5:].all()


def test_prompt_audio_mask_forwarded_to_cond_row() -> None:
    # Upstream: batch_audio_mask[i, :c_len] = inp["audio_mask"] — the cond row
    # must carry the prompt's own audio mask (reference slots included), not a
    # target-tail-only derivation.
    lm = RecordingLM(18)
    prompt = np.concatenate(
        [
            np.tile(np.arange(1, 6, dtype=np.int64), (C, 1)),  # style+text prefix
            np.tile(np.arange(7, 13, dtype=np.int64), (C, 1)),  # fake ref codes
            np.full((C, 7), MASK, dtype=np.int64),
        ],
        axis=1,
    )
    mask = np.zeros(prompt.shape, dtype=np.bool_)
    mask[:, 5:] = True  # ref + target are audio slots (Task 4 semantics)
    run_diffusion_sampling(
        lm, prompt, 7, prompt_audio_mask=mask, config=OmniVoiceSamplerConfig(num_step=2, seed=5)
    )
    _, audio, _ = lm.batches[0]
    assert (audio[0] == mask).all()  # cond: prompt's mask verbatim
    assert not audio[1, :, :11].any()  # uncond: only the trailing target slice...
    assert audio[1, :, 11:].all()  # ...of the prompt's mask (prefix 5+6 is pad)


def test_prompt_audio_mask_shape_validated() -> None:
    with pytest.raises(ValueError, match="audio_mask"):
        run_diffusion_sampling(
            FakeLM(15), _prompt(10), 10,
            prompt_audio_mask=np.zeros((C, 3), dtype=np.bool_),
            config=OmniVoiceSamplerConfig(num_step=2),
        )


def test_merged_confidence_is_renormalized() -> None:
    lm = CfgNormalizationLM(target_len=4)
    run_diffusion_sampling(
        lm, _prompt(4), 4,
        config=OmniVoiceSamplerConfig(num_step=2, position_temperature=0.0, seed=0),
    )
    # Step 1 commits ceil(32 · Δt'_0) = 3 of 32 slots. Re-normalized confidence
    # ranks the sharp cond slots above the ultra-uncond slots, so codebook 0's
    # tail entering step 2 is [3, 3, MASK, MASK]; the un-normalized variant
    # (raw merged max ≈ 179 on the ultra slots) would invert the pattern.
    tail = lm.batches[1][0, 0, -4:]
    assert tail[0] == 3
    assert tail[1] == 3
    assert tail[2] == MASK
    assert tail[3] == MASK


def test_progress_and_step_count() -> None:
    seen: list[tuple[int, int]] = []
    run_diffusion_sampling(
        FakeLM(15), _prompt(10), 10,
        config=OmniVoiceSamplerConfig(num_step=3),
        progress=lambda k, n: seen.append((k, n)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_cancellation_between_steps() -> None:
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(OmniVoiceSamplingCancelled):
        run_diffusion_sampling(
            FakeLM(15), _prompt(10), 10,
            config=OmniVoiceSamplerConfig(num_step=4),
            cancel_check=cancel,
        )


def test_seed_reproducibility_with_sampling_temperatures() -> None:
    cfg = OmniVoiceSamplerConfig(num_step=3, class_temperature=1.0, position_temperature=5.0, seed=7)
    a = run_diffusion_sampling(FakeLM(15), _prompt(8), 8, config=cfg)
    b = run_diffusion_sampling(FakeLM(15), _prompt(8), 8, config=cfg)
    assert (a == b).all()
