"""OmniVoice prompt construction + duration estimation.

Port map: k2-fsa/omnivoice (Apache-2.0) `_combine_text`,
`_prepare_inference_inputs`, and omnivoice/utils/duration.py.
"""

import numpy as np

from tldw_chatbook.TTS.omnivoice_prompt import (
    build_prompt_inputs,
    build_style_text,
    combine_text,
    estimate_target_frames,
)


class FakeTokenizer:
    """Maps each whitespace-split word to its length as an id — deterministic."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(w) % 1000 + 1 for w in text.split()]


def test_combine_text_joins_and_normalizes() -> None:
    assert combine_text("hello", "hi there") == "hi there hello"
    assert combine_text("a\r\nb") == "ab"
    assert combine_text("x（y）z") == "x(y)z"
    assert combine_text("a    b") == "a b"
    assert combine_text("你 好") == "你好"


def test_style_text_variants() -> None:
    assert build_style_text("en", None, has_reference=False) == (
        "<|lang_start|>en<|lang_end|><|instruct_start|>None<|instruct_end|>"
    )
    assert build_style_text(None, None, has_reference=False) == (
        "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
    )
    assert build_style_text("en", "warm", has_reference=True) == (
        "<|denoise|><|lang_start|>en<|lang_end|><|instruct_start|>warm<|instruct_end|>"
    )


def test_estimate_target_frames_fallback_anchor() -> None:
    # Upstream RuleDurationEstimator weights (omnivoice/utils/duration.py,
    # Apache-2.0): latin letter 1.0, space 0.2, punctuation 0.5.
    # Fallback ref "Nice to meet you." = 13 latin (1.0) + 3 spaces (0.2)
    # + 1 period (0.5) = weight 14.1; ref_frames 25.
    # "Hello world" weight = 10 latin (1.0) + 1 space (0.2) = 10.2
    # → raw 25 · 10.2 / 14.1 ≈ 18.0851.
    # 18.0851 < low_threshold 50 → boost: 50 · (18.0851/50)^(1/3) ≈ 35.6249 → 35.
    assert estimate_target_frames("Hello world") == 35


def test_estimate_target_frames_clone_ratio_and_speed() -> None:
    frames = estimate_target_frames("Hello world", "hi there", ref_frames=25)
    assert frames > 0
    faster = estimate_target_frames("Hello world", "hi there", ref_frames=25, speed=2.0)
    assert faster == max(1, frames // 2 + (1 if frames % 4 else 0)) or faster < frames


def test_build_prompt_inputs_shape_and_masks() -> None:
    tok = FakeTokenizer()
    out = build_prompt_inputs(tok, text="hello world", target_len=10)
    assert out.input_ids.shape == (8, out.input_ids.shape[1])
    assert out.input_ids.dtype == np.int64
    assert out.audio_mask.shape == out.input_ids.shape
    assert out.audio_mask.dtype == np.bool_
    # target region is masked audio positions at the tail
    assert out.audio_mask[:, -out.target_len :].all()
    assert not out.audio_mask[:, : out.prompt_len].any()
    assert (out.input_ids[:, -out.target_len :] == 1024).all()

    cloned = build_prompt_inputs(
        tok, text="hello", ref_codes=np.zeros((8, 7), dtype=np.int64), target_len=4
    )
    assert cloned.audio_mask[:, -11:-4].all()  # ref region is audio


def test_build_prompt_inputs_repeat_across_codebooks() -> None:
    tok = FakeTokenizer()
    out = build_prompt_inputs(tok, text="one two three", target_len=3)
    assert (out.input_ids[0] == out.input_ids[7]).all()
