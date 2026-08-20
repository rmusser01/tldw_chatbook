import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from tldw_chatbook.Chunking.engine import Chunker


class _FakeTokenizer:
    """Simple tokenizer that maps each codepoint to a token id and decodes back.

    - encode(text) -> List[int]: list of ord(c) for each character in text order
    - decode(token_ids) -> str: join of chr(id) for each id

    This makes token windows correspond to contiguous character spans, without
    offering offset mapping, which forces the strategy to use the rolling-pointer
    fallback span resolution we want to test.
    """

    def __init__(self, model_name: str = "fake"):
        self.model_name = model_name

    def encode(self, text: str):
        return [ord(c) for c in text]

    def decode(self, token_ids):

        return "".join(chr(i) for i in token_ids)


def _patch_strategy_for_fallback(chunker: Chunker):
    """Patch the token strategy to ensure offsets None path is exercised.

    - Inject a fake tokenizer so we don't rely on external packages.
    - Force _reconstruct_offsets_by_decoding to return None.
    """
    strat = chunker.get_strategy("tokens")
    # Inject our fake tokenizer instance directly
    setattr(strat, "_tokenizer", _FakeTokenizer())
    # Force offsets reconstruction to return None so the code path uses fallback
    setattr(strat, "_reconstruct_offsets_by_decoding", lambda ids, text: None)


def _assert_monotonic_spans(results, text_len):
    """Assert spans are monotonic by start and within bounds."""
    prev_start = -1
    for r in results:
        md = r.metadata
        assert 0 <= md.start_char <= text_len
        assert 0 <= md.end_char <= text_len
        assert md.end_char >= md.start_char
        # Starts should be strictly increasing across windows
        assert md.start_char > prev_start
        prev_start = md.start_char


@pytest.mark.parametrize(
    "max_size,overlap",
    [
        (3, 0),
        (3, 1),
        (4, 1),
        (4, 2),
        (5, 0),
        (5, 2),
        (6, 3),
        (7, 1),
        (8, 4),
        (9, 3),
    ],
)
def test_token_fallback_offsets_repeated_substrings(max_size, overlap):
    # Repeated substrings can trick naive find() into using the first occurrence
    text = "ababa ababa ababa"
    chunker = Chunker()
    _patch_strategy_for_fallback(chunker)

    results = chunker.chunk_text_with_metadata(
        text,
        method="tokens",
        max_size=max_size,
        overlap=overlap,
        align_text_to_source=True,
    )

    assert len(results) > 0
    _assert_monotonic_spans(results, len(text))
    # Ensure each chunk's text matches the source slice for its span
    for r in results:
        md = r.metadata
        assert r.text == text[md.start_char : md.end_char]


@settings(max_examples=25, deadline=None)
@given(
    max_size=st.integers(min_value=2, max_value=10),
    overlap=st.integers(min_value=0, max_value=9),
)
def test_token_fallback_offsets_repeated_substrings_property(max_size, overlap):
    assume(overlap < max_size)
    text = "ababa ababa ababa ababa"
    chunker = Chunker()
    _patch_strategy_for_fallback(chunker)
    results = chunker.chunk_text_with_metadata(
        text,
        method="tokens",
        max_size=max_size,
        overlap=overlap,
        align_text_to_source=True,
    )
    assert len(results) > 0
    _assert_monotonic_spans(results, len(text))
    for r in results:
        md = r.metadata
        assert r.text == text[md.start_char : md.end_char]


@pytest.mark.parametrize(
    "max_size,overlap",
    [
        (3, 0),
        (3, 1),
        (4, 1),
        (4, 2),
        (6, 2),
        (7, 3),
        (8, 4),
        (9, 1),
    ],
)
def test_token_fallback_offsets_unicode_cf_differences(max_size, overlap):
    # Include Cf characters (word joiner U+2060, variation selector U+FE0F)
    text = "alpha\u2060beta alpha\ufe0fbeta alpha beta"
    chunker = Chunker()
    _patch_strategy_for_fallback(chunker)

    results = chunker.chunk_text_with_metadata(
        text,
        method="tokens",
        max_size=max_size,
        overlap=overlap,
        align_text_to_source=True,
    )

    assert len(results) > 0
    _assert_monotonic_spans(results, len(text))
    for r in results:
        md = r.metadata
        assert r.text == text[md.start_char : md.end_char]


@settings(max_examples=25, deadline=None)
@given(
    max_size=st.integers(min_value=2, max_value=10),
    overlap=st.integers(min_value=0, max_value=9),
)
def test_token_fallback_offsets_unicode_cf_differences_property(max_size, overlap):
    assume(overlap < max_size)
    text = "alpha\u2060beta alpha\ufe0fbeta alpha\u2060beta"
    chunker = Chunker()
    _patch_strategy_for_fallback(chunker)
    results = chunker.chunk_text_with_metadata(
        text,
        method="tokens",
        max_size=max_size,
        overlap=overlap,
        align_text_to_source=True,
    )
    assert len(results) > 0
    _assert_monotonic_spans(results, len(text))
    for r in results:
        md = r.metadata
        assert r.text == text[md.start_char : md.end_char]


@pytest.mark.parametrize(
    "max_size,overlap",
    [
        (3, 0),
        (3, 1),
        (4, 1),
        (4, 2),
        (5, 2),
        (6, 3),
        (7, 1),
        (8, 4),
    ],
)
def test_token_fallback_offsets_zwj_sequences(max_size, overlap):
    # ZWJ sequence: woman technologist repeated, with ASCII around
    text = "👩‍💻👩‍💻👩‍💻 test 👩‍💻👩‍💻"
    chunker = Chunker()
    _patch_strategy_for_fallback(chunker)

    results = chunker.chunk_text_with_metadata(
        text,
        method="tokens",
        max_size=max_size,
        overlap=overlap,
        align_text_to_source=True,
    )

    assert len(results) > 0
    _assert_monotonic_spans(results, len(text))
    for r in results:
        md = r.metadata
        assert r.text == text[md.start_char : md.end_char]


@settings(max_examples=25, deadline=None)
@given(
    max_size=st.integers(min_value=2, max_value=10),
    overlap=st.integers(min_value=0, max_value=9),
)
def test_token_fallback_offsets_zwj_sequences_property(max_size, overlap):
    assume(overlap < max_size)
    text = "👩‍💻👩‍💻 test 👩‍💻👩‍💻👩‍💻 end"
    chunker = Chunker()
    _patch_strategy_for_fallback(chunker)
    results = chunker.chunk_text_with_metadata(
        text,
        method="tokens",
        max_size=max_size,
        overlap=overlap,
        align_text_to_source=True,
    )
    assert len(results) > 0
    _assert_monotonic_spans(results, len(text))
    for r in results:
        md = r.metadata
        assert r.text == text[md.start_char : md.end_char]
