import os
from typing import List

import pytest

from tldw_chatbook.Chunking.engine import Chunker


def _reconstruct_tokens_from_stream(chunks: List[str], max_overlap: int) -> List[str]:
    """Greedily reconstruct a token stream from streaming chunks.

    On each chunk, remove the longest prefix that matches the current suffix
    (up to max_overlap tokens) to deduplicate boundary overlap.
    """
    out: List[str] = []
    for ch in chunks:
        toks = ch.split()
        # find longest d <= max_overlap s.t. out[-d:] == toks[:d]
        d = 0
        L = min(max_overlap, len(toks), len(out))
        for k in range(L, 0, -1):
            if out[-k:] == toks[:k]:
                d = k
                break
        out.extend(toks[d:])
    return out


def _reconstruct_chars_from_stream(chunks: List[str], max_overlap: int) -> str:
    """Reconstruct a character stream from streaming chunks with overlap."""
    out: List[str] = []
    for ch in chunks:
        chars = list(ch)
        d = 0
        L = min(max_overlap, len(chars), len(out))
        for k in range(L, 0, -1):
            if out[-k:] == chars[:k]:
                d = k
                break
        out.extend(chars[d:])
    return "".join(out)


def test_chunk_file_stream_words_overlap(tmp_path):

    # Prepare a deterministic whitespace-normalized corpus
    words = [f"w{i:04d}" for i in range(1, 1200)]
    text = " ".join(words)
    p = tmp_path / "words.txt"
    p.write_text(text, encoding="utf-8")

    ck = Chunker()

    # overlap > 0
    chunks = list(ck.chunk_file_stream(p, method="words", max_size=25, overlap=5, language="en", buffer_size=1024))
    recon = _reconstruct_tokens_from_stream(chunks, max_overlap=5)
    assert recon == words, "Reconstructed token stream should match original (overlap>0)"

    # overlap == 0
    chunks0 = list(ck.chunk_file_stream(p, method="words", max_size=25, overlap=0, language="en", buffer_size=1024))
    recon0 = _reconstruct_tokens_from_stream(chunks0, max_overlap=0)
    assert recon0 == words, "Reconstructed token stream should match original (overlap=0)"


def test_chunk_file_stream_words_no_space_language(tmp_path, monkeypatch):
    from tldw_chatbook.Chunking.engine.strategies.words import WordChunkingStrategy

    monkeypatch.setattr(WordChunkingStrategy, "_tokenize_thai", lambda self, text: list(text))

    text = "abcdefg" * 700  # > 2048 chars to force multiple streaming flushes
    p = tmp_path / "nospace.txt"
    p.write_text(text, encoding="utf-8")

    ck = Chunker()
    chunks = list(
        ck.chunk_file_stream(
            p,
            method="words",
            max_size=32,
            overlap=5,
            language="th",
            buffer_size=1024,
        )
    )
    recon = _reconstruct_chars_from_stream(chunks, max_overlap=5)
    assert recon == text


def test_chunk_file_stream_sentences_overlap(tmp_path):

    # Build simple sentences
    sents = [f"This is sentence {i}." for i in range(1, 400)]
    text = " ".join(sents)
    p = tmp_path / "sents.txt"
    p.write_text(text, encoding="utf-8")

    ck = Chunker()

    # overlap one sentence
    chunks = list(ck.chunk_file_stream(p, method="sentences", max_size=8, overlap=1, language="en", buffer_size=2048))
    recon = _reconstruct_tokens_from_stream(chunks, max_overlap=50)  # sentence ~ few tokens
    # Compare after whitespace normalization via token lists
    original_tokens = text.split()
    assert recon == original_tokens, "Sentence stream should preserve content with dedup"


def test_chunk_file_stream_sentences_overlap_matches_full_chunking(tmp_path, monkeypatch):

    ck = Chunker()
    sents = [f"Sentence {i}." for i in range(1, 9)]
    text = " ".join(sents)
    part1 = " ".join(sents[:4]) + " "
    p = tmp_path / "sents_boundary.txt"
    p.write_text(text, encoding="utf-8")

    # Force early flushing so we hit a boundary split deterministically.
    monkeypatch.setattr(ck, "_estimate_stream_flush_threshold", lambda method, max_size: len(part1))

    chunks_stream = list(
        ck.chunk_file_stream(
            p,
            method="sentences",
            max_size=3,
            overlap=1,
            language="en",
            buffer_size=len(part1),
        )
    )
    chunks_full = ck.chunk_text(text, method="sentences", max_size=3, overlap=1, language="en")
    assert chunks_stream == chunks_full


def test_chunk_file_stream_sentences_no_overlap_matches_full_chunking(tmp_path, monkeypatch):

    ck = Chunker()
    sents = [f"Sentence {i}." for i in range(1, 9)]
    text = " ".join(sents)
    part1 = " ".join(sents[:4]) + " "
    p = tmp_path / "sents_boundary_no_overlap.txt"
    p.write_text(text, encoding="utf-8")

    # Force early flushing so we hit a boundary split deterministically.
    monkeypatch.setattr(ck, "_estimate_stream_flush_threshold", lambda method, max_size: len(part1))

    chunks_stream = list(
        ck.chunk_file_stream(
            p,
            method="sentences",
            max_size=3,
            overlap=0,
            language="en",
            buffer_size=len(part1),
        )
    )
    recon = _reconstruct_tokens_from_stream(chunks_stream, max_overlap=0)
    assert recon == text.split()


def test_structure_aware_code_fence_no_trailing_newline():

    ck = Chunker()
    src = "# Title\n\n" "```python\n" "print('hello')" "```\n" "Paragraph after.\n"
    # Intentionally no newline before the closing fence in middle of the string
    chunks = ck.chunk_text(src, method="structure_aware", max_size=3, overlap=0, language="en")
    # Ensure code fence is recognized and serialized back with fences present
    assert any("```python" in c and "```" in c for c in chunks), "Code fence should be preserved as a single block"
    assert any("print('hello')" in c for c in chunks), "Code content should be present"


def test_structure_aware_code_fence_long_marker():

    ck = Chunker()
    src = "# Title\n\n" "````python\n" "print('hello')\n" "````\n" "Paragraph after.\n"
    chunks = ck.chunk_text(src, method="structure_aware", max_size=1, overlap=0, language="en")
    assert any("print('hello')" in c for c in chunks), "Code content should be present"
    assert any("Paragraph after." in c for c in chunks), "Trailing paragraph should not be swallowed"
    assert not any("````python" in c for c in chunks), "Language tag should not include stray backticks"


def test_language_autodetect_thai():

    ck = Chunker()
    thai_text = "ภาษาไทยทดสอบ ทดสอบข้อความ เพื่อการตัดประโยคฯ"
    out = ck.process_text(
        thai_text,
        options={
            "method": "sentences",
            "max_size": 2,
            "overlap": 0,
            "language": "auto",
        },
    )
    assert out and isinstance(out, list)
    md = out[0].get("metadata", {})
    assert md.get("language") == "th", f"Expected Thai autodetect, got {md.get('language')}"


def test_language_autodetect_japanese_prefers_kana():

    ck = Chunker()
    japanese_text = "これはテストです。次の文です。"
    out = ck.process_text(
        japanese_text,
        options={
            "method": "sentences",
            "max_size": 2,
            "overlap": 0,
            "language": "auto",
        },
    )
    assert out and isinstance(out, list)
    md = out[0].get("metadata", {})
    assert md.get("language") == "ja", f"Expected Japanese autodetect, got {md.get('language')}"


@pytest.mark.asyncio
async def test_async_chunk_stream_sentences_overlap_boundary_matches_full():
    # --- Ported (chunking-engine-parity Task 4) -------------------------
    # async_chunker depends on server-only http_client/exceptions modules
    # and is deferred to sub-project #6 (spec §5.1 deferrals).
    pytest.importorskip(
        "tldw_chatbook.Chunking.engine.async_chunker",
        reason="async_chunker deferred to #6 (server http_client/exceptions deps)",
    )
    from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker

    part1 = " ".join([f"Sentence {i}." for i in range(1, 7)]) + " "
    part2 = " ".join([f"Sentence {i}." for i in range(7, 11)])
    full_text = part1 + part2

    async def text_stream():
        yield part1
        yield part2

    async with AsyncChunker() as chunker:
        chunks = [
            ch
            async for ch in chunker.chunk_stream(
                text_stream(),
                method="sentences",
                max_size=3,
                overlap=1,
                buffer_size=len(part1),
                language="en",
            )
        ]

    expected = Chunker().chunk_text(full_text, method="sentences", max_size=3, overlap=1, language="en")
    assert chunks == expected


@pytest.mark.asyncio
async def test_async_chunk_stream_sentences_overlap_matches_full():

    # --- Ported (chunking-engine-parity Task 4) -------------------------
    # async_chunker depends on server-only http_client/exceptions modules
    # and is deferred to sub-project #6 (spec §5.1 deferrals).
    pytest.importorskip(
        "tldw_chatbook.Chunking.engine.async_chunker",
        reason="async_chunker deferred to #6 (server http_client/exceptions deps)",
    )
    from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker

    sents = [f"Sentence {i}." for i in range(1, 9)]
    part1 = " ".join(sents[:4]) + " "
    part2 = " ".join(sents[4:])
    full_text = part1 + part2

    async def text_stream():
        yield part1
        yield part2

    expected = Chunker().chunk_text(full_text, method="sentences", max_size=3, overlap=1, language="en")

    async with AsyncChunker() as chunker:
        chunks = [
            ch
            async for ch in chunker.chunk_stream(
                text_stream(),
                method="sentences",
                max_size=3,
                overlap=1,
                buffer_size=len(part1),
                language="en",
            )
        ]

    assert chunks == expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_chunk_stream_overlap_clamps_to_max_size() -> None:
    # --- Ported (chunking-engine-parity Task 4) -------------------------
    # async_chunker depends on server-only http_client/exceptions modules
    # and is deferred to sub-project #6 (spec §5.1 deferrals).
    pytest.importorskip(
        "tldw_chatbook.Chunking.engine.async_chunker",
        reason="async_chunker deferred to #6 (server http_client/exceptions deps)",
    )
    from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker

    sents = [f"Sentence {i}." for i in range(1, 9)]
    part1 = " ".join(sents[:4]) + " "
    part2 = " ".join(sents[4:])
    full_text = part1 + part2

    async def text_stream():
        yield part1
        yield part2

    # chunk_text clamps overlap to max_size - 1
    expected = Chunker().chunk_text(full_text, method="sentences", max_size=2, overlap=1, language="en")

    async with AsyncChunker() as chunker:
        chunks = [
            ch
            async for ch in chunker.chunk_stream(
                text_stream(),
                method="sentences",
                max_size=2,
                overlap=5,
                buffer_size=len(part1),
                language="en",
            )
        ]

    assert chunks == expected


@pytest.mark.asyncio
async def test_async_chunk_stream_sentences_no_overlap_matches_full():
    # --- Ported (chunking-engine-parity Task 4) -------------------------
    # async_chunker depends on server-only http_client/exceptions modules
    # and is deferred to sub-project #6 (spec §5.1 deferrals).
    pytest.importorskip(
        "tldw_chatbook.Chunking.engine.async_chunker",
        reason="async_chunker deferred to #6 (server http_client/exceptions deps)",
    )
    from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker

    sents = [f"Sentence {i}." for i in range(1, 9)]
    part1 = " ".join(sents[:4]) + " "
    part2 = " ".join(sents[4:])
    full_text = part1 + part2

    async def text_stream():
        yield part1
        yield part2

    async with AsyncChunker() as chunker:
        chunks = [
            ch
            async for ch in chunker.chunk_stream(
                text_stream(),
                method="sentences",
                max_size=3,
                overlap=0,
                buffer_size=len(part1),
                language="en",
            )
        ]

    recon = _reconstruct_tokens_from_stream(chunks, max_overlap=0)
    assert recon == full_text.split()


@pytest.mark.asyncio
async def test_async_chunk_stream_words_no_space_language(monkeypatch):
    # --- Ported (chunking-engine-parity Task 4) -------------------------
    # async_chunker depends on server-only http_client/exceptions modules
    # and is deferred to sub-project #6 (spec §5.1 deferrals).
    pytest.importorskip(
        "tldw_chatbook.Chunking.engine.async_chunker",
        reason="async_chunker deferred to #6 (server http_client/exceptions deps)",
    )
    from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker
    from tldw_chatbook.Chunking.engine.strategies.words import WordChunkingStrategy

    monkeypatch.setattr(WordChunkingStrategy, "_tokenize_thai", lambda self, text: list(text))

    text = "abcdefg" * 200
    part1 = text[: len(text) // 2]
    part2 = text[len(text) // 2 :]

    async def text_stream():
        yield part1
        yield part2

    async with AsyncChunker() as chunker:
        chunks = [
            ch
            async for ch in chunker.chunk_stream(
                text_stream(),
                method="words",
                max_size=16,
                overlap=4,
                buffer_size=len(part1),
                language="th",
            )
        ]

    recon = _reconstruct_chars_from_stream(chunks, max_overlap=4)
    assert recon == text


@pytest.mark.asyncio
async def test_async_chunk_stream_overlap_no_tail_dup_on_boundary():
    # --- Ported (chunking-engine-parity Task 4) -------------------------
    # async_chunker depends on server-only http_client/exceptions modules
    # and is deferred to sub-project #6 (spec §5.1 deferrals).
    pytest.importorskip(
        "tldw_chatbook.Chunking.engine.async_chunker",
        reason="async_chunker deferred to #6 (server http_client/exceptions deps)",
    )
    from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker

    text = " ".join([f"Sentence {i}." for i in range(1, 7)])

    async def text_stream():
        yield text

    expected = Chunker().chunk_text(
        text,
        method="sentences",
        max_size=2,
        overlap=1,
        language="en",
    )

    async with AsyncChunker() as chunker:
        chunks = [
            ch
            async for ch in chunker.chunk_stream(
                text_stream(),
                method="sentences",
                max_size=2,
                overlap=1,
                buffer_size=len(text),
                language="en",
            )
        ]

    assert chunks == expected
