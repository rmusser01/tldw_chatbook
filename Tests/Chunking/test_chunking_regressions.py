import sys
import types

import pytest

from tldw_chatbook.Chunking.engine import Chunker, ChunkingError
from tldw_chatbook.Chunking.engine.base import ChunkerConfig, ChunkingMethod
from tldw_chatbook.Chunking.engine.strategies.words import WordChunkingStrategy


def test_words_min_chunk_size_no_space_merge(monkeypatch):

    strategy = WordChunkingStrategy(language="zh")
    tokens = ["a", "b", "c", "d"]
    monkeypatch.setattr(strategy, "_tokenize_text", lambda text: tokens)
    text = "".join(tokens)
    chunks = strategy.chunk(text, max_size=3, overlap=0, min_chunk_size=2)
    assert chunks == [text]


def test_words_chunk_generator_matches_chunk_with_min_size(monkeypatch):

    strategy = WordChunkingStrategy(language="en")
    tokens = ["one", "two", "three", "four", "five"]
    monkeypatch.setattr(strategy, "_tokenize_text", lambda text: tokens)
    text = " ".join(tokens)
    chunks = strategy.chunk(text, max_size=3, overlap=0, min_chunk_size=3)
    gen_chunks = list(strategy.chunk_generator(text, max_size=3, overlap=0, min_chunk_size=3))
    assert gen_chunks == chunks


def test_sentence_strategy_language_reconfigure_via_chunker(monkeypatch):

    fake_pkg = types.ModuleType("pythainlp")
    fake_tokenize = types.ModuleType("pythainlp.tokenize")

    def sent_tokenize(text):

        return [text]

    fake_tokenize.sent_tokenize = sent_tokenize
    fake_pkg.tokenize = fake_tokenize
    monkeypatch.setitem(sys.modules, "pythainlp", fake_pkg)
    monkeypatch.setitem(sys.modules, "pythainlp.tokenize", fake_tokenize)

    ck = Chunker(config=ChunkerConfig(default_method=ChunkingMethod.SENTENCES, language="en"))
    ck.chunk_text("One. Two.", method="sentences", max_size=1, overlap=0, language="en")
    strategy = ck.get_strategy("sentences")
    assert not strategy.pythainlp_available

    ck.chunk_text("thai test.", method="sentences", max_size=1, overlap=0, language="th")
    assert strategy.pythainlp_available
    assert strategy._th_sent_tokenize is sent_tokenize


def test_hierarchical_sentences_respects_combine_short():

    ck = Chunker()
    text = "A. B. C. D."
    base = ck.chunk_text(
        text,
        method="sentences",
        max_size=2,
        overlap=0,
        combine_short=True,
        min_sentence_length=5,
    )
    hier = ck.chunk_text_hierarchical_flat(
        text,
        method="sentences",
        max_size=2,
        overlap=0,
        method_options={
            "combine_short": True,
            "min_sentence_length": 5,
        },
    )
    assert [item["text"] for item in hier] == base


def test_words_min_chunk_size_overlap_no_duplicates():

    strategy = WordChunkingStrategy(language="en")
    text = "one two three four five six seven"
    chunks = strategy.chunk(text, max_size=5, overlap=2, min_chunk_size=5)
    assert chunks == [text]


def test_hierarchical_words_min_chunk_size_overlap_matches_flat():

    ck = Chunker()
    text = "one two three four five six seven"
    base = ck.chunk_text(text, method="words", max_size=5, overlap=2, min_chunk_size=5)
    hier = ck.chunk_text_hierarchical_flat(
        text,
        method="words",
        max_size=5,
        overlap=2,
        method_options={"min_chunk_size": 5},
    )
    assert [item["text"] for item in hier] == base


def test_hierarchical_code_fence_long_marker_closes():

    ck = Chunker()
    text = "Intro\n\n````python\nprint('hi')\n````\n\nAfter."
    chunks = ck.chunk_text_hierarchical_flat(text, method="words", max_size=50, overlap=0)
    assert any(c.get("metadata", {}).get("paragraph_kind") == "code_fence" for c in chunks)
    assert any(
        c.get("metadata", {}).get("paragraph_kind") == "paragraph" and "After." in c.get("text", "")
        for c in chunks
    )


def test_token_chunk_decode_failure_falls_back():

    from tldw_chatbook.Chunking.engine.strategies.tokens import TokenChunkingStrategy

    class BadTokenizer:
        def encode(self, text):
            return list(range(len(text.split())))

        def decode(self, token_ids, skip_special_tokens=True):
            raise RuntimeError("decode failed")

    strategy = TokenChunkingStrategy()
    strategy._tokenizer = BadTokenizer()
    strategy._tokenizer_init_attempted = True

    text = "one two three four five six"
    chunks = strategy.chunk(text, max_size=4, overlap=0)
    assert chunks
    reconstructed = " ".join(chunks).split()
    assert reconstructed == text.split()


@pytest.mark.unit
def test_process_text_multi_level_fallback_offsets_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback offsets should remain within paragraph bounds even on mismatched chunks."""
    chunker = Chunker()
    text = "short paragraph"

    def _raise_chunking_error(*_args, **_kwargs):
        raise ChunkingError("forced failure")

    def _fake_chunk_text(*_args, **_kwargs):
        return ["this chunk is intentionally longer than the paragraph"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", _raise_chunking_error)
    monkeypatch.setattr(chunker, "chunk_text", _fake_chunk_text)

    rows = chunker.process_text(
        text,
        options={"method": "words", "max_size": 2, "overlap": 0, "multi_level": True},
    )
    assert rows
    for row in rows:
        md = row.get("metadata", {})
        start = md.get("start_offset")
        end = md.get("end_offset")
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start <= end <= len(text)
