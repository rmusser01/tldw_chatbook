import os
import re
import pytest

from Tests.Chunking.conftest import requires_cached_hf_tokenizer as _tokenizer_cached
from hypothesis import given, strategies as st, settings as hyp_settings, HealthCheck

# --- Ported (chunking-engine-parity Task 4) -----------------------------------
# The 'tokens' strategy arm of this property test loads the real gpt2
# tokenizer from the HF hub; chatbook's network guard blocks the download
# and the engine surfaces TokenizerError, so it only runs where the
# tokenizer is already cached (env-dependent, see conftest).
# Spec §10.2: also run under the production sanitization path (test mode
# explicitly off) via the production_path marker; Tests/Chunking/conftest.py.
pytestmark = [
    pytest.mark.skipif(
        not _tokenizer_cached(),
        reason="requires a locally cached gpt2 tokenizer (HF download blocked by network guard)",
    ),
    pytest.mark.production_path,
]

from tldw_chatbook.Chunking.engine import Chunker


@pytest.fixture(autouse=True)
def testing_env():
    os.environ["TESTING"] = "true"
    yield
    os.environ.pop("TESTING", None)


# ------------------------- helper generators -------------------------

_EN_WORD = st.from_regex(r"[A-Za-z]{1,10}(?:[,.!?])?", fullmatch=True)


def _join_en_tokens(tokens: list[str]) -> str:
    # Join with variable spaces/newlines to stress offset mapping
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if i < len(tokens) - 1:
            # cycle through separators: space, double-space, newline, blank line
            sep = [" ", "  ", "\n", "\n\n  "][i % 4]
            out.append(sep)
    return "".join(out)


def _ja_token_strategy():

    # limited kana set + punctuation
    kana = "あいうえおかきくけこさしすせそたちつてとなにぬねのまみむめもやゆよらりるれろわをん"
    punct = "。！？"
    tok = st.text(alphabet=list(kana + punct), min_size=1, max_size=6)
    return st.lists(tok, min_size=5, max_size=60)


def _join_ja_tokens(tokens: list[str]) -> str:
    # Join without spaces; sprinkle punctuation via existing tokens
    return "".join(tokens)


def _th_token_strategy():

    # small Thai alphabet subset + punctuation
    thai = "กขคฆงจฉชซดตถทนบปผฝพฟมยรลวสหออะอิอึอือะเแโใไึ"
    punct = "!?"  # Thai uses these as well
    tok = st.text(alphabet=list(thai + punct), min_size=1, max_size=6)
    return st.lists(tok, min_size=5, max_size=60)


def _join_th_tokens(tokens: list[str]) -> str:
    return "".join(tokens)


def _text_for_language(lang: str):
    if lang == "en":
        return st.lists(_EN_WORD, min_size=10, max_size=80).map(_join_en_tokens)
    if lang == "ja":
        return _ja_token_strategy().map(_join_ja_tokens)
    if lang == "th":
        return _th_token_strategy().map(_join_th_tokens)
    # default English
    return st.lists(_EN_WORD, min_size=10, max_size=80).map(_join_en_tokens)


_LANGS = st.sampled_from(["en", "ja", "th"])  # representative locales
_METHODS = st.sampled_from(["words", "sentences", "paragraphs", "tokens"])  # safe subset


@hyp_settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    lang=_LANGS,
    method=_METHODS,
    max_size=st.integers(min_value=2, max_value=32),
    overlap=st.integers(min_value=0, max_value=8),
    data=st.data(),
)
def test_chunk_with_metadata_re_slices_source(lang: str, method: str, max_size: int, overlap: int, data):
    # Constrain overlap to < max_size for progress
    if overlap >= max_size:
        overlap = max_size - 1
        if overlap < 0:
            overlap = 0

    # Build randomized text for the chosen language
    text_strategy = _text_for_language(lang)
    text = data.draw(text_strategy)
    # Guard against pathological empties (shouldn't happen with our min sizes)
    if not text or not isinstance(text, str):
        return

    ck = Chunker()
    results = ck.chunk_text_with_metadata(
        text,
        method=method,
        max_size=max_size,
        overlap=overlap,
        language=lang,
        align_text_to_source=True,
    )

    # Property: every chunk's text equals text[start:end], and bounds are valid
    for res in results or []:
        s = getattr(res.metadata, "start_char", None)
        e = getattr(res.metadata, "end_char", None)
        assert isinstance(s, int) and isinstance(e, int)
        assert 0 <= s <= e <= len(text)
        assert res.text == text[s:e]
