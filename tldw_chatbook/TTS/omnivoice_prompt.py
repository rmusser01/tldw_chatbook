"""OmniVoice prompt construction and duration estimation.

Faithful numpy port of the Apache-2.0 upstream k2-fsa/omnivoice:
- omnivoice/models/omnivoice.py :: ``_combine_text``, ``_prepare_inference_inputs``
  (prompt layout), and ``_estimate_target_tokens`` (speed handling)
- omnivoice/utils/duration.py  :: ``RuleDurationEstimator`` — the phonetic
  weight table and Unicode range map are ported verbatim from upstream
  (Copyright 2026 Xiaomi Corp., Apache License 2.0;
  https://github.com/k2-fsa/omnivoice).

No code is taken from the unlicensed AFun9/Omnivoice-onnx repo.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

NUM_AUDIO_CODEBOOK = 8
AUDIO_VOCAB_SIZE = 1025
AUDIO_MASK_ID = 1024

_CJK_RANGE = r"[\u4e00-\u9fff]"
_CJK_SPACE_RE = re.compile(rf"(?<={_CJK_RANGE})\s+|\s+(?={_CJK_RANGE})")


def combine_text(text: str, ref_text: str = "") -> str:
    """Combine reference and target text the way upstream ``_combine_text`` does.

    Rules, in upstream order:

    1. ``ref_text.strip() + " " + text.strip()`` when cloning, else ``text.strip()``.
    2. Strip all CR/LF characters.
    3. Replace fullwidth Chinese parentheses with ASCII ones.
    4. Collapse runs of spaces/tabs into a single space.
    5. Remove whitespace adjacent to CJK ideographs.

    Args:
        text: The target text to synthesize.
        ref_text: The reference clip's transcript when cloning, else ``""``.

    Returns:
        The single normalized text segment the prompt encodes.
    """
    if ref_text:
        full = ref_text.strip() + " " + text.strip()
    else:
        full = text.strip()
    full = re.sub(r"[\r\n]+", "", full)
    full = full.replace("（", "(").replace("）", ")")
    full = re.sub(r"[ \t]+", " ", full)
    full = _CJK_SPACE_RE.sub("", full)
    return full


def build_style_text(
    lang: str | None, instruct: str | None, *, has_reference: bool
) -> str:
    """Build the style token segment (upstream ``_prepare_inference_inputs``).

    ``<|denoise|>`` is only emitted when reference audio is present, mirroring
    upstream's ``denoise and ref_audio_tokens is not None``.

    Args:
        lang: Language id, or ``None`` for language-agnostic mode.
        instruct: Optional voice-design instruction.
        has_reference: Whether reference audio codes follow in the prompt.

    Returns:
        The style-token string (``<|lang_start|>…`` / ``<|instruct_start|>…``).
    """
    parts: list[str] = []
    if has_reference:
        parts.append("<|denoise|>")
    parts.append(f"<|lang_start|>{lang or 'None'}<|lang_end|>")
    parts.append(f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>")
    return "".join(parts)


# --- duration estimation (RuleDurationEstimator port) ------------------------
#
# Ported verbatim from omnivoice/utils/duration.py (Apache-2.0): the weight of
# a character is the relative speaking time compared to a standard Latin
# letter (1.0 = one Latin character, ~40-50 ms).

_WEIGHTS = {
    # --- Logographic (1 char = full syllable/word) ---
    "cjk": 3.0,  # Chinese, Japanese Kanji, etc.
    # --- Syllabic / Blocks
    "hangul": 2.5,  # Korean Hangul
    "kana": 2.2,  # Japanese Hiragana/Katakana
    "ethiopic": 3.0,  # Amharic/Ge'ez
    "yi": 3.0,  # Yi script
    # --- Abugida (Consonant-Vowel complexes) ---
    "indic": 1.8,  # Hindi, Bengali, Tamil, etc.
    "thai_lao": 1.5,  # Thai, Lao
    "khmer_myanmar": 1.8,  # Khmer, Myanmar
    # --- Abjad (Consonant-heavy) ---
    "arabic": 1.5,  # Arabic, Persian, Urdu
    "hebrew": 1.5,  # Hebrew
    # --- Alphabet (Segmental) ---
    "latin": 1.0,  # English, Spanish, French, Vietnamese, etc. (Baseline)
    "cyrillic": 1.0,  # Russian, Ukrainian
    "greek": 1.0,  # Greek
    "armenian": 1.0,  # Armenian
    "georgian": 1.0,  # Georgian
    # --- Symbols & Misc ---
    "punctuation": 0.5,  # Pause capability
    "space": 0.2,  # Word boundary/Breath (0.05 / 0.22)
    "digit": 3.5,  # Numbers
    "mark": 0.0,  # Diacritics/Accents (Silent modifiers)
    "default": 1.0,  # Fallback for unknown scripts
}

# (End_Codepoint, Type_Key) pairs used for fast binary search (bisect),
# exactly as upstream maps Unicode blocks to weight keys.
_RANGES = [
    (0x02AF, "latin"),  # Latin (Basic, Supplement, Ext, IPA)
    (0x03FF, "greek"),  # Greek & Coptic
    (0x052F, "cyrillic"),  # Cyrillic
    (0x058F, "armenian"),  # Armenian
    (0x05FF, "hebrew"),  # Hebrew
    (0x077F, "arabic"),  # Arabic, Syriac, Arabic Supplement
    (0x089F, "arabic"),  # Arabic Extended-B (+ Syriac Supp)
    (0x08FF, "arabic"),  # Arabic Extended-A
    (0x097F, "indic"),  # Devanagari
    (0x09FF, "indic"),  # Bengali
    (0x0A7F, "indic"),  # Gurmukhi
    (0x0AFF, "indic"),  # Gujarati
    (0x0B7F, "indic"),  # Oriya
    (0x0BFF, "indic"),  # Tamil
    (0x0C7F, "indic"),  # Telugu
    (0x0CFF, "indic"),  # Kannada
    (0x0D7F, "indic"),  # Malayalam
    (0x0DFF, "indic"),  # Sinhala
    (0x0EFF, "thai_lao"),  # Thai & Lao
    (0x0FFF, "indic"),  # Tibetan (Abugida)
    (0x109F, "khmer_myanmar"),  # Myanmar
    (0x10FF, "georgian"),  # Georgian
    (0x11FF, "hangul"),  # Hangul Jamo
    (0x137F, "ethiopic"),  # Ethiopic
    (0x139F, "ethiopic"),  # Ethiopic Supplement
    (0x13FF, "default"),  # Cherokee
    (0x167F, "default"),  # Canadian Aboriginal Syllabics
    (0x169F, "default"),  # Ogham
    (0x16FF, "default"),  # Runic
    (0x171F, "default"),  # Tagalog (Baybayin)
    (0x173F, "default"),  # Hanunoo
    (0x175F, "default"),  # Buhid
    (0x177F, "default"),  # Tagbanwa
    (0x17FF, "khmer_myanmar"),  # Khmer
    (0x18AF, "default"),  # Mongolian
    (0x18FF, "default"),  # Canadian Aboriginal Syllabics Ext
    (0x194F, "indic"),  # Limbu
    (0x19DF, "indic"),  # Tai Le & New Tai Lue
    (0x19FF, "khmer_myanmar"),  # Khmer Symbols
    (0x1A1F, "indic"),  # Buginese
    (0x1AAF, "indic"),  # Tai Tham
    (0x1B7F, "indic"),  # Balinese
    (0x1BBF, "indic"),  # Sundanese
    (0x1BFF, "indic"),  # Batak
    (0x1C4F, "indic"),  # Lepcha
    (0x1C7F, "indic"),  # Ol Chiki (Santali)
    (0x1C8F, "cyrillic"),  # Cyrillic Extended-C
    (0x1CBF, "georgian"),  # Georgian Extended
    (0x1CCF, "indic"),  # Sundanese Supplement
    (0x1CFF, "indic"),  # Vedic Extensions
    (0x1D7F, "latin"),  # Phonetic Extensions
    (0x1DBF, "latin"),  # Phonetic Extensions Supplement
    (0x1DFF, "default"),  # Combining Diacritical Marks Supplement
    (0x1EFF, "latin"),  # Latin Extended Additional (Vietnamese)
    (0x309F, "kana"),  # Hiragana
    (0x30FF, "kana"),  # Katakana
    (0x312F, "cjk"),  # Bopomofo (Pinyin)
    (0x318F, "hangul"),  # Hangul Compatibility Jamo
    (0x9FFF, "cjk"),  # CJK Unified Ideographs (Main)
    (0xA4CF, "yi"),  # Yi Syllables
    (0xA4FF, "default"),  # Lisu
    (0xA63F, "default"),  # Vai
    (0xA69F, "cyrillic"),  # Cyrillic Extended-B
    (0xA6FF, "default"),  # Bamum
    (0xA7FF, "latin"),  # Latin Extended-D
    (0xA82F, "indic"),  # Syloti Nagri
    (0xA87F, "default"),  # Phags-pa
    (0xA8DF, "indic"),  # Saurashtra
    (0xA8FF, "indic"),  # Devanagari Extended
    (0xA92F, "indic"),  # Kayah Li
    (0xA95F, "indic"),  # Rejang
    (0xA97F, "hangul"),  # Hangul Jamo Extended-A
    (0xA9DF, "indic"),  # Javanese
    (0xA9FF, "khmer_myanmar"),  # Myanmar Extended-B
    (0xAA5F, "indic"),  # Cham
    (0xAA7F, "khmer_myanmar"),  # Myanmar Extended-A
    (0xAADF, "indic"),  # Tai Viet
    (0xAAFF, "indic"),  # Meetei Mayek Extensions
    (0xAB2F, "ethiopic"),  # Ethiopic Extended-A
    (0xAB6F, "latin"),  # Latin Extended-E
    (0xABBF, "default"),  # Cherokee Supplement
    (0xABFF, "indic"),  # Meetei Mayek
    (0xD7AF, "hangul"),  # Hangul Syllables
    (0xFAFF, "cjk"),  # CJK Compatibility
    (0xFDFF, "arabic"),  # Arabic Presentation Forms-A
    (0xFE6F, "default"),  # Variation Selectors
    (0xFEFF, "arabic"),  # Arabic Presentation Forms-B
    (0xFFEF, "latin"),  # Fullwidth Latin
]
_BREAKPOINTS = [r[0] for r in _RANGES]


@lru_cache(maxsize=4096)
def _char_weight(char: str) -> float:
    """Determine the phonetic weight of a single character (upstream order)."""
    code = ord(char)
    if (65 <= code <= 90) or (97 <= code <= 122):
        return _WEIGHTS["latin"]
    if code == 32:
        return _WEIGHTS["space"]

    # Ignore arabic Tatweel
    if code == 0x0640:
        return _WEIGHTS["mark"]

    category = unicodedata.category(char)

    if category.startswith("M"):
        return _WEIGHTS["mark"]

    if category.startswith(("P", "S")):
        return _WEIGHTS["punctuation"]

    if category.startswith("Z"):
        return _WEIGHTS["space"]

    if category.startswith("N"):
        return _WEIGHTS["digit"]

    # Binary search for Unicode block (category checks have already filtered
    # punctuation/symbols/whitespace/numbers out of these ranges).
    idx = bisect.bisect_left(_BREAKPOINTS, code)
    if idx < len(_RANGES):
        script_type = _RANGES[idx][1]
        return _WEIGHTS.get(script_type, _WEIGHTS["default"])

    # Handle upper planes (CJK Ext B/C/D, Historic scripts)
    if code > 0x20000:
        return _WEIGHTS["cjk"]

    return _WEIGHTS["default"]


def _text_weight(text: str) -> float:
    """Sum of normalized weights for a string (upstream ``calculate_total_weight``)."""
    return sum(_char_weight(ch) for ch in text)


def _estimate_duration(
    target_text: str,
    ref_text: str,
    ref_duration: float,
    low_threshold: float | None = 50,
    boost_strength: float = 3,
) -> float:
    """Upstream ``RuleDurationEstimator.estimate_duration`` (Apache-2.0 port)."""
    if ref_duration <= 0 or not ref_text:
        return 0.0

    ref_weight = _text_weight(ref_text)
    if ref_weight == 0:
        return 0.0

    speed_factor = ref_weight / ref_duration
    target_weight = _text_weight(target_text)

    estimated_duration = target_weight / speed_factor
    if low_threshold is not None and estimated_duration < low_threshold:
        alpha = 1.0 / boost_strength
        return low_threshold * (estimated_duration / low_threshold) ** alpha
    return estimated_duration


_DEFAULT_REF_TEXT = "Nice to meet you."
_DEFAULT_REF_FRAMES = 25


def estimate_target_frames(
    text: str, ref_text: str = "", ref_frames: int = 0, *, speed: float = 1.0
) -> int:
    """Estimate the number of target audio frames for ``text``.

    Port of upstream ``OmniVoice._estimate_target_tokens``: fall back to the
    built-in anchor when no usable reference is given, estimate by weight
    ratio, boost short estimates with a power curve, then divide by ``speed``.

    Args:
        text: The target text.
        ref_text: The reference transcript, when cloning.
        ref_frames: Number of reference codec frames, when cloning.
        speed: Speaking-rate multiplier (> 1 is faster, so fewer frames).

    Returns:
        The estimated number of target frames (at least 1).
    """
    if not ref_text or ref_frames <= 0:
        ref_text, ref_frames = _DEFAULT_REF_TEXT, _DEFAULT_REF_FRAMES
    est = _estimate_duration(text, ref_text, ref_frames)
    if speed > 0 and speed != 1.0:
        est = est / speed
    return max(1, int(est))


# --- prompt assembly -----------------------------------------------------------


class PromptTokenizer(Protocol):
    """The tokenizer surface prompt construction needs.

    ``tokenizers.Tokenizer`` satisfies it (``encode`` returns an ``Encoding``
    exposing ``.ids``); test fakes may return ``list[int]`` directly.
    """

    def encode(self, sequence: str, add_special_tokens: bool = ...) -> Any:
        """Encode ``sequence``; return ids or an object with ``.ids``."""
        ...


def _encode_ids(tokenizer: PromptTokenizer, text: str) -> list[int]:
    """Token ids for ``text``; unwraps a ``tokenizers.Encoding`` to ``.ids``."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return list(getattr(encoded, "ids", encoded))


@dataclass(frozen=True)
class OmniVoicePromptInputs:
    """Inference prompt for the OmniVoice ONNX graph (no torch dependency).

    Attributes:
        input_ids: (num_codebooks, S) int64 — style tokens ‖ text tokens ‖
            ref codes ‖ mask-filled target; the text segment is repeated on
            every codebook row, ref codes differ per row.
        audio_mask: (num_codebooks, S) bool — True only over the reference
            and target audio slots.
        prompt_len: Length of the style+text(+ref) prefix before the target.
        target_len: Number of mask-filled target slots at the tail.
    """

    input_ids: np.ndarray  # (8, S) int64
    audio_mask: np.ndarray  # (8, S) bool — True where positions are audio slots
    prompt_len: int
    target_len: int


def build_prompt_inputs(
    tokenizer: PromptTokenizer,
    *,
    text: str,
    ref_text: str = "",
    lang: str | None = None,
    instruct: str | None = None,
    ref_codes: np.ndarray | None = None,
    target_len: int,
    num_codebooks: int = NUM_AUDIO_CODEBOOK,
) -> OmniVoicePromptInputs:
    """Build ``input_ids`` and ``audio_mask`` for inference.

    Port of upstream ``OmniVoice._prepare_inference_inputs``. ``tokenizer`` is
    any object whose ``encode(text, add_special_tokens=False)`` returns token
    ids — either a ``list[int]`` or an object with ``.ids`` (the
    ``tokenizers`` library's ``Tokenizer`` returns an ``Encoding``).

    Args:
        tokenizer: Text tokenizer (see :class:`PromptTokenizer`).
        text: The target text to synthesize.
        ref_text: The reference transcript when cloning, else ``""``.
        lang: Language id, or ``None`` for language-agnostic mode.
        instruct: Optional voice-design instruction.
        ref_codes: ``(num_codebooks, T_ref)`` reference codes when cloning.
        target_len: Number of target audio frames to generate.
        num_codebooks: Codebook rows in the prompt.

    Returns:
        The ``input_ids``/``audio_mask`` arrays and segment lengths.

    Raises:
        ValueError: If ``ref_codes`` shape does not match ``num_codebooks``.
    """
    has_ref = ref_codes is not None
    ref: np.ndarray | None = None
    if has_ref:
        ref = np.asarray(ref_codes, dtype=np.int64)
        if ref.ndim != 2 or ref.shape[0] != num_codebooks:
            raise ValueError(
                f"ref_codes must have shape ({num_codebooks}, T), got {ref.shape}"
            )
    style_text = build_style_text(lang, instruct, has_reference=has_ref)
    style_ids = _encode_ids(tokenizer, style_text)

    # Upstream folds the reference transcript into the text segment.
    body = combine_text(text, ref_text)
    body_ids = _encode_ids(tokenizer, f"<|text_start|>{body}<|text_end|>")
    prefix = np.asarray(style_ids + body_ids, dtype=np.int64)

    rows = []
    for c in range(num_codebooks):
        row_parts = [prefix]
        if ref is not None:
            row_parts.append(ref[c])
        row_parts.append(np.full(target_len, AUDIO_MASK_ID, dtype=np.int64))
        rows.append(np.concatenate(row_parts))
    input_ids = np.stack(rows)

    prompt_len = len(prefix) + (ref.shape[1] if ref is not None else 0)
    audio_mask = np.zeros(input_ids.shape, dtype=np.bool_)
    audio_mask[:, len(prefix) :] = True
    return OmniVoicePromptInputs(
        input_ids=input_ids,
        audio_mask=audio_mask,
        prompt_len=prompt_len,
        target_len=target_len,
    )
