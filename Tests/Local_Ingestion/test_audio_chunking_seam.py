"""Contract tests for the audio processor's chunking seam (tasks 840, 952 review)."""

from __future__ import annotations

import inspect

import pytest


class TestAudioChunkingSeam:
    """The audio path must call the chunking service the way it is declared.

    Every audio and video ingest that reached chunking used to die with
    "'<=' not supported between instances of 'dict' and 'int'": an options dict
    was passed positionally into ``chunk_size``. A signature check is the guard
    that would have caught it, since a hand-written double would have been
    written to match the wrong call (task-840).
    """

    def test_the_service_signature_is_the_one_the_caller_assumes(self):
        from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

        params = inspect.signature(ChunkingService.chunk_text).parameters
        for name in ("content", "chunk_size", "chunk_overlap", "method"):
            assert name in params, f"chunk_text lost its {name} parameter"
        # The defect: a dict landing where an int is declared.
        assert params["chunk_size"].annotation in (int, "int")

    def test_chunk_text_returns_string_text_and_keeps_offsets(self):
        """Offsets must survive: the storage path otherwise re-derives them by
        summing lengths, which double-counts whenever chunks overlap."""
        from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor

        proc = LocalAudioProcessor.__new__(LocalAudioProcessor)

        class _Service:
            def chunk_text(self, content, chunk_size=400, chunk_overlap=100, method="words"):
                assert isinstance(chunk_size, int), "an options dict reached chunk_size"
                return [
                    {"text": "alpha beta", "start_char": 0, "end_char": 10, "chunk_index": 0},
                    {"text": "beta gamma", "start_char": 5, "end_char": 15, "chunk_index": 1},
                ]

        proc.chunking_service = _Service()
        out = proc._chunk_text("alpha beta gamma", method="words", max_size=2, overlap=1)

        assert [c["text"] for c in out] == ["alpha beta", "beta gamma"]
        assert all(isinstance(c["text"], str) for c in out)
        # Overlapping chunks: summing lengths would give 0 and 10, not 0 and 5.
        assert [c["start_char"] for c in out] == [0, 5]
        assert [c["end_char"] for c in out] == [10, 15]

    def test_a_string_chunker_result_still_works(self):
        from tldw_chatbook.Local_Ingestion.audio_processing import LocalAudioProcessor

        proc = LocalAudioProcessor.__new__(LocalAudioProcessor)

        class _StringService:
            def chunk_text(self, content, chunk_size=400, chunk_overlap=100, method="words"):
                return ["one two", "three four"]

        proc.chunking_service = _StringService()
        out = proc._chunk_text("one two three four", method="words")

        assert [c["text"] for c in out] == ["one two", "three four"]
        assert all("start_char" not in c for c in out), "offsets must not be invented"
