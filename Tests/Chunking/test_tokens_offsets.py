import hashlib
import os
import sys
import tempfile
import types

import pytest


def _has_tiktoken():
    try:
        import tiktoken  # noqa: F401
    except Exception:
        return False

    network_enabled = os.getenv("ENABLE_NETWORK_TESTS", "").lower() in {"1", "true", "yes", "y", "on"}
    if network_enabled:
        return True

    cache_dir = os.getenv("TIKTOKEN_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.getenv("DATA_GYM_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
    if cache_dir == "":
        return False

    blob_url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    cache_key = hashlib.sha1(blob_url.encode()).hexdigest()  # nosec B324
    cache_path = os.path.join(cache_dir, cache_key)
    return os.path.exists(cache_path)


def test_tokens_offsets_tiktoken_monotonic_and_slice_match():

    if not _has_tiktoken():
        pytest.skip("tiktoken not available")

    from tldw_chatbook.Chunking.engine.strategies.tokens import (
        TokenChunkingStrategy,
    )

    text = "Hello,  world!\nHello world!  Goodbye.\n" "Repeated phrase. Repeated phrase. Repeated phrase."
    strat = TokenChunkingStrategy(tokenizer_name="gpt-3.5-turbo")
    results = strat.chunk_with_metadata(text, max_size=12, overlap=4)

    assert results, "No chunks returned"

    prev_start = -1
    prev_end = -1
    for r in results:
        s = r.metadata.start_char
        e = r.metadata.end_char
        # Bounds are valid and monotonic
        assert 0 <= s <= e <= len(text)
        assert s >= prev_start
        assert e >= prev_end
        # Slice matches decoded text
        assert text[s:e] == r.text
        # Token counts sane
        assert 0 < r.metadata.token_count <= 12
        prev_start, prev_end = s, e


def test_tokens_offsets_tiktoken_repeated_substrings_heavy_overlap():

    if not _has_tiktoken():
        pytest.skip("tiktoken not available")

    from tldw_chatbook.Chunking.engine.strategies.tokens import (
        TokenChunkingStrategy,
    )

    # Repeated patterns can confuse naive substring matching; ensure offsets handle it
    text = "foo bar foo bar foo bar foo bar\n" "foo bar foo bar foo bar foo bar\n" "foo bar foo bar foo bar"
    strat = TokenChunkingStrategy(tokenizer_name="gpt-3.5-turbo")
    results = strat.chunk_with_metadata(text, max_size=8, overlap=7)

    assert results and len(results) > 2
    for r in results:
        s = r.metadata.start_char
        e = r.metadata.end_char
        assert 0 <= s <= e <= len(text)
        # Exact slice match proves correct mapping even with repeats and heavy overlap
        assert text[s:e] == r.text


def test_tokens_offsets_tiktoken_unicode_emojis_multibyte():

    if not _has_tiktoken():
        pytest.skip("tiktoken not available")

    from tldw_chatbook.Chunking.engine.strategies.tokens import (
        TokenChunkingStrategy,
    )

    text = (
        "Start 😊😊 café café naïve 🚀 - dashes - and 🤝🏽 emoji with skin-tone.\n"
        "New line, tabs\t\t, and zero-width joiners: 👨‍👩‍👧‍👦 family."
    )
    import unicodedata as _ud

    def _strip_cf(s: str) -> str:
        return "".join(ch for ch in s if _ud.category(ch) != "Cf")

    strat = TokenChunkingStrategy(tokenizer_name="gpt-4")
    results = strat.chunk_with_metadata(text, max_size=20, overlap=10)

    import unicodedata as _ud

    assert results
    for r in results:
        s = r.metadata.start_char
        e = r.metadata.end_char
        assert 0 <= s <= e <= len(text)
        # Boundary should not split grapheme: no combining mark or joiner right after e
        if e < len(text):
            cat = _ud.category(text[e])
            assert cat not in ("Mn", "Me", "Cf"), f"Boundary splits cluster at pos {e}: U+{ord(text[e]):04X} ({cat})"


def test_tokens_offsets_transformers_path_via_mock():
    """Exercise the transformers offset_mapping logic via a mocked tokenizer."""
    from tldw_chatbook.Chunking.engine.strategies.tokens import (
        TokenChunkingStrategy,
    )

    text = "abcdef ghij klmno"

    class FakeHFTokenizer:
        def __call__(self, txt, add_special_tokens=False, return_offsets_mapping=False, **_: object):
            assert txt == text
            # char-level tokenization; optionally add specials (-1 at both ends)
            input_ids = list(range(len(txt)))
            offsets = [(i, i + 1) for i in range(len(txt))]
            if add_special_tokens:
                input_ids = [-1] + input_ids + [-2]
                offsets = [(0, 0)] + offsets + [(0, 0)]
            return {"input_ids": input_ids, "offset_mapping": offsets}

        def decode(self, token_ids):

            # map -1/-2 to empty, others index into original text
            out = []
            for tid in token_ids:
                if tid in (-1, -2):
                    continue
                out.append(text[tid])
            return "".join(out)

    # Wrap to look like our TransformersTokenizer wrapper (has .tokenizer attr)
    wrapper = types.SimpleNamespace(tokenizer=FakeHFTokenizer())

    strat = TokenChunkingStrategy(tokenizer_name="mock-hf")
    # Force our mocked wrapper
    strat._tokenizer = wrapper  # type: ignore[attr-defined]

    results = strat.chunk_with_metadata(text, max_size=5, overlap=2, add_special_tokens=True)

    assert results
    # Validate per-chunk spans map back to original text
    for r in results:
        s = r.metadata.start_char
        e = r.metadata.end_char
        assert 0 <= s <= e <= len(text)
        assert text[s:e] == r.text
        assert r.metadata.options.get("add_special_tokens") is True


def test_tokens_offsets_fallback_path():

    from tldw_chatbook.Chunking.engine.strategies.tokens import (
        TokenChunkingStrategy,
        FallbackTokenizer,
    )

    text = "Leading  spaces,\nmultiple\nlines,\tand\t punctuations!"

    # Force fallback tokenizer so we don't depend on external libs
    strat = TokenChunkingStrategy(tokenizer_name="gpt-3.5-turbo")
    fb = FallbackTokenizer("gpt-3.5-turbo")
    strat._tokenizer = fb  # type: ignore[attr-defined]

    results = strat.chunk_with_metadata(text, max_size=12, overlap=4)

    assert results
    prev_start = -1
    prev_end = -1
    ratio = fb.tokens_per_word.get(fb.model_name, fb.tokens_per_word["default"])
    for r in results:
        s = r.metadata.start_char
        e = r.metadata.end_char
        assert 0 <= s <= e <= len(text)
        assert s >= prev_start
        assert e >= prev_end
        # Words within slice match words in chunk text
        assert text[s:e].split() == r.text.split()
        # Token count approximates word_count * ratio
        expected = int(round(r.metadata.word_count * ratio))
        assert r.metadata.token_count == expected
        assert r.metadata.options.get("approximate") is True
        prev_start, prev_end = s, e


def test_tokenizer_override_resets_between_calls(monkeypatch):
    """Switching tokenizer names should re-attempt tokenizer initialization."""
    from tldw_chatbook.Chunking.engine import Chunker
    from tldw_chatbook.Chunking.engine.strategies import tokens as tokens_mod

    created = []

    class StubTokenizer:
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.available = True
            self._last_text = ""
            created.append(model_name)

        def encode(self, text: str):
            self._last_text = text
            return list(range(len(text)))

        def decode(self, token_ids, skip_special_tokens: bool = True):
            return "".join(self._last_text[i] for i in token_ids if i < len(self._last_text))

    monkeypatch.setattr(tokens_mod, "TiktokenTokenizer", StubTokenizer)

    chunker = Chunker()
    text = "abcde"
    chunker.chunk_text(text, method="tokens", max_size=2, overlap=0, tokenizer_name="tok-a")
    chunker.chunk_text(text, method="tokens", max_size=2, overlap=0, tokenizer_name="tok-b")

    assert created == ["tok-a", "tok-b"]


def test_tokens_chunk_preserves_trailing_newlines(monkeypatch):
    """Token chunking should not drop trailing newlines from decoded output."""
    from tldw_chatbook.Chunking.engine.strategies import tokens as tokens_mod
    from tldw_chatbook.Chunking.engine.strategies.tokens import TokenChunkingStrategy

    class StubTokenizer:
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.available = True
            self._last_text = ""

        def encode(self, text: str):
            self._last_text = text
            return list(range(len(text)))

        def decode(self, token_ids, skip_special_tokens: bool = True):
            return "".join(self._last_text[i] for i in token_ids if i < len(self._last_text))

    monkeypatch.setattr(tokens_mod, "TiktokenTokenizer", StubTokenizer)

    text = "line1\nline2\n"
    strat = TokenChunkingStrategy(tokenizer_name="tok-newline")
    chunks = strat.chunk(text, max_size=5, overlap=0)

    assert "".join(chunks) == text
    assert chunks[-1].endswith("\n")


def test_offset_reconstruction_fallback_logs_no_document_text():
    """TASK-19322 (ADR-029): the unlocatable piece is decoded user document
    text, so the fallback diagnostic must not echo its characters at any log
    level. It stays useful via piece length, scan position, and a short
    stable digest -- and the reconstruction behavior itself is unchanged."""
    from loguru import logger

    from tldw_chatbook.Chunking.engine.strategies.tokens import (
        TokenChunkingStrategy,
    )

    secret = "SECRET-DOCUMENT-CONTENT-19322"
    text = "The visible body shares nothing with the decoded token piece."

    class StubTokenizer:
        model_name = "stub-19322"
        available = True

        def encode(self, text: str):
            return [0]

        def decode(self, token_ids, skip_special_tokens: bool = True):
            # decoded_all != text forces the tolerant forward scan; the
            # single-token piece is absent from `text`, so the not-found
            # fallback (and its diagnostic) must fire.
            return secret

    strat = TokenChunkingStrategy(tokenizer_name="stub-19322")
    strat._tokenizer = StubTokenizer()  # type: ignore[attr-defined]

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="DEBUG")
    try:
        offsets = strat._reconstruct_offsets_by_decoding([0], text)
    finally:
        logger.remove(sink_id)

    # Behavior unchanged: the piece anchors at the current position.
    assert offsets == [(0, min(len(text), len(secret)))]

    fallback_messages = [
        m
        for m in messages
        if "Token piece not found in text during offset reconstruction" in m
    ]
    assert fallback_messages, (
        f"the not-found fallback must leave a debug trace: {messages}"
    )
    assert not any(secret in m for m in messages), (
        f"document text must not be echoed in any log record: {messages}"
    )
    # De-identified metadata keeps the diagnostic useful: length, position,
    # and a stable digest prefix for cross-record correlation.
    assert any(
        f"piece_len={len(secret)}" in m and "pos=0" in m and "piece_sha256=" in m
        for m in fallback_messages
    ), f"the diagnostic must keep piece_len/pos/piece_sha256: {fallback_messages}"
