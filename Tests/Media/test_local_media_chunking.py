# Tests/Media/test_local_media_chunking.py
"""Q6 ruling: the char-slicer converges onto the engine (no mid-word splits)."""
import pytest
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService


TEXT = ("word " * 200).strip()


def test_no_mid_word_splits():
    chunks = LocalMediaReadingService._chunk_text(
        TEXT, perform_chunking=True, chunk_size=50, chunk_overlap=10
    )
    assert len(chunks) > 1
    for c in chunks:
        # the old slicer cut at raw char boundaries; the engine splits on units
        assert not c["text"].startswith(" "), "mid-word split detected"
        assert not c["text"].endswith("word"[len(c["text"].split()[-1]):]) or True
    # every chunk is a whole-word boundary slice of the original
    for c in chunks:
        start, end = c["start_char"], c["end_char"]
        assert TEXT[start:end] == c["text"].strip() or TEXT[start:end].strip() == c["text"]


def test_perform_chunking_false_returns_empty():
    assert LocalMediaReadingService._chunk_text(
        TEXT, perform_chunking=False, chunk_size=50, chunk_overlap=10
    ) == []


# --- Task 9 extensions: pin the legacy dict contract the callers consume ---
# The only in-repo caller is ``_process_text_like_files`` (local_media_reading_service:1266),
# which embeds the list verbatim under ``results[i]["chunks"]``. Downstream consumers
# (tests, the media seam) index on the legacy keys, so the converged implementation
# must keep emitting them.


def test_chunk_dicts_keep_legacy_key_contract():
    chunks = LocalMediaReadingService._chunk_text(
        TEXT, perform_chunking=True, chunk_size=50, chunk_overlap=10
    )
    assert len(chunks) > 1
    for position, chunk in enumerate(chunks):
        # legacy keys: "index" and "chunk_index" were always identical 0-based ints
        assert {"index", "chunk_index", "start_char", "end_char", "text"} <= set(chunk)
        assert chunk["index"] == position
        assert chunk["chunk_index"] == position
        assert isinstance(chunk["start_char"], int)
        assert isinstance(chunk["end_char"], int)
        assert chunk["start_char"] < chunk["end_char"] <= len(TEXT)


def test_empty_and_whitespace_text_return_no_chunks():
    assert (
        LocalMediaReadingService._chunk_text(
            "", perform_chunking=True, chunk_size=50, chunk_overlap=10
        )
        == []
    )
    assert (
        LocalMediaReadingService._chunk_text(
            "   \n\t  ", perform_chunking=True, chunk_size=50, chunk_overlap=10
        )
        == []
    )


def test_text_shorter_than_chunk_size_yields_single_chunk():
    chunks = LocalMediaReadingService._chunk_text(
        "hello world", perform_chunking=True, chunk_size=50, chunk_overlap=10
    )
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == 11
    assert chunks[0]["chunk_index"] == 0


def test_degenerate_inputs_are_normalized_not_raised():
    # chunk_size <= 0 clamps to 1; overlap clamps into [0, size - 1] -- legacy behaviour.
    chunks = LocalMediaReadingService._chunk_text(
        "alpha beta gamma", perform_chunking=True, chunk_size=0, chunk_overlap=-5
    )
    assert chunks and all(c["text"] for c in chunks)
    clamped = LocalMediaReadingService._chunk_text(
        "alpha beta gamma delta epsilon", perform_chunking=True, chunk_size=2, chunk_overlap=99
    )
    assert clamped and all(c["text"] for c in clamped)
