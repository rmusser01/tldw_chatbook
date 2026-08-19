# Tests/Chunking/test_chunk_lib_shim.py
"""Chunk_Lib shim contract (spec §6.2): legacy signatures + flat output shape."""
import pytest

from tldw_chatbook.Chunking import Chunk_Lib


def test_legacy_signature_improved_chunking_process():
    # Positional options dict, all legacy kwargs accepted (§6.1.1 callers).
    chunks = Chunk_Lib.improved_chunking_process(
        "Alpha beta gamma. Delta epsilon.",
        {"method": "words", "max_size": 3, "overlap": 1},
        tokenizer_name_or_path="gpt2",
        template=None,
        template_manager=None,
        llm_call_function_for_chunker=None,
        llm_api_config_for_chunker=None,
    )
    assert chunks, "expected chunks"
    first = chunks[0]
    assert first["text"], "chunk text must be top-level"
    assert isinstance(first["metadata"], dict)
    assert first["metadata"]["chunk_index"] == 1  # 1-based, legacy convention


def test_legacy_chunker_adapter():
    chunker = Chunk_Lib.Chunker(
        options={"method": "words", "max_size": 3, "overlap": 1},
        tokenizer_name_or_path="gpt2",
    )
    chunks = chunker.chunk_text("Alpha beta gamma delta.", method="words")
    assert chunks
    # chunk_text historically returned List[Union[str, dict]] — strings for
    # text methods, dicts for json/xml/ebook (§6.2). The adapter keeps that.
    assert isinstance(chunks[0], (str, dict))


def test_flat_contract_top_level_offsets():
    # §6.3.2: the flat per-chunk contract is what the DB seam reads.
    chunks = Chunk_Lib.improved_chunking_process(
        "One two three four. Five six seven eight.", {"method": "words", "max_size": 2}
    )
    assert all("start_char" in c and "end_char" in c for c in chunks), \
        "offsets must be top-level for _persist_chunks"
    assert all(c["word_count"] > 0 for c in chunks)


def test_module_level_chunk_xml_restored():
    assert callable(Chunk_Lib.chunk_xml)  # §7.1: name was gone, capability wasn't


def test_exception_aliases():
    from tldw_chatbook.Chunking.engine import (
        LanguageNotSupportedError, InvalidInputError, ChunkingError as EngineChunkingError,
    )
    assert Chunk_Lib.LanguageDetectionError is LanguageNotSupportedError
    assert Chunk_Lib.MemoryLimitError is InvalidInputError
    assert Chunk_Lib.ChunkingError is EngineChunkingError


def test_constants_reexported():
    assert Chunk_Lib.MAX_CHUNK_SIZE_WORDS == 10000
    assert Chunk_Lib.MAX_CHUNK_SIZE_PARAGRAPHS == 100
    assert Chunk_Lib.MAX_DOCUMENT_SIZE_MB == 100
    assert isinstance(Chunk_Lib.DEFAULT_CHUNK_OPTIONS, dict)
    assert callable(Chunk_Lib.ensure_nltk_data)


def test_tokens_no_silent_fallback():
    # Q2: the shim must raise if the engine would silently word-approximate.
    # Simulated by monkeypatching the engine tokenizer resolution to the
    # fallback and asserting the shim notices.
    from tldw_chatbook.Chunking.engine.strategies import tokens as tokens_mod
    monkeypatch_obj = pytest.MonkeyPatch()
    original = tokens_mod.TokenChunkingStrategy._resolve_tokenizer
    def fake_resolve(self):
        return tokens_mod.FallbackTokenizer("gpt2")
    monkeypatch_obj.setattr(tokens_mod.TokenChunkingStrategy, "_resolve_tokenizer", fake_resolve)
    try:
        with pytest.raises(Chunk_Lib.ChunkingError, match="tiktoken"):
            Chunk_Lib.improved_chunking_process(
                "one two three four five six", {"method": "tokens", "max_size": 3}
            )
    finally:
        monkeypatch_obj.setattr(tokens_mod.TokenChunkingStrategy, "_resolve_tokenizer", original)
        monkeypatch_obj.undo()
