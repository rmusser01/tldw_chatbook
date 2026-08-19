# Tests/Chunking/test_chunk_lib_shim.py
"""Chunk_Lib shim contract (spec §6.2): legacy signatures + flat output shape."""
import pytest

from tldw_chatbook.Chunking import Chunk_Lib


@pytest.fixture(autouse=True)
def _offline_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these tests off the optional GPT-2 download path.

    The repo's network guard fails any test that attempts a blocked socket;
    the legacy TokenBasedChunker (used by the rolling-summarize token
    counter) would otherwise try to load gpt2 from the HF hub.
    """
    from tldw_chatbook.Chunking import token_chunker

    monkeypatch.setattr(token_chunker, "get_safe_import", lambda _name: None)


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
    # (overlap=0: with the stock default overlap 200 >= max_size 3 the
    # legacy-parity overlap guard would raise first -- see
    # test_tokens_overlap_geq_max_size_raises -- masking the Q2 path.)
    from tldw_chatbook.Chunking.engine.strategies import tokens as tokens_mod
    monkeypatch_obj = pytest.MonkeyPatch()
    original = tokens_mod.TokenChunkingStrategy._resolve_tokenizer
    def fake_resolve(self):
        return tokens_mod.FallbackTokenizer("gpt2")
    monkeypatch_obj.setattr(tokens_mod.TokenChunkingStrategy, "_resolve_tokenizer", fake_resolve)
    try:
        with pytest.raises(Chunk_Lib.ChunkingError, match="tiktoken"):
            Chunk_Lib.improved_chunking_process(
                "one two three four five six",
                {"method": "tokens", "max_size": 3, "overlap": 0},
            )
    finally:
        monkeypatch_obj.setattr(tokens_mod.TokenChunkingStrategy, "_resolve_tokenizer", original)


# ---------------------------------------------------------------------------
# Review-fix regression tests (external review round 1: C1/C2, I1/I2, I4, I5).
#
# The tokens tests stub the ENGINE's tokenizer resolution (via the
# _resolve_tokenizer seam / TiktokenTokenizer) so no test touches the HF hub
# (the repo's network guard fails the test on any blocked socket attempt).
# ---------------------------------------------------------------------------

_TOKENS_OK = {"method": "tokens", "max_size": 4, "overlap": 0}
_TOKENS_TEXT = "one two three four five six seven eight"


class _StubRealTokenizer:
    """Network-free stand-in for a real (non-fallback) tokenizer."""

    def encode(self, text):
        return list(range(len(text.split())))

    def decode(self, ids, **_kwargs):
        return " ".join(f"w{i}" for i in ids)

    def count_tokens(self, text):
        return len(text.split())


def _patch_tokenizer_property(monkeypatch, factory):
    """Replace the engine strategy's ``tokenizer`` property.

    The engine's chunking path resolves the tokenizer through this property
    directly (not the ``_resolve_tokenizer`` seam), and the property consults
    tiktoken/transformers -- which needs the network. Pointing the property
    at ``factory`` stubs BOTH the shim's Q2 probe and the engine's actual
    chunking, keeping these tests network-free under the repo's guard.
    """
    from tldw_chatbook.Chunking.engine.strategies.tokens import TokenChunkingStrategy

    monkeypatch.setattr(
        TokenChunkingStrategy, "tokenizer", property(lambda self: factory())
    )


def test_rolling_summarize_payload_dict_via_improved_chunking_process():
    # C1: improved_chunking_process must honor the legacy payload-dict LLM
    # callback for rolling_summarize. It previously forwarded the callback
    # to engine process_text, whose strategy calls it positionally
    # (analyze-style) -- the payload-dict callback was never invoked and the
    # TypeError surfaced as ProcessingError("...provider call failed.").
    captured = []

    def fake_llm(payload):
        assert isinstance(payload, dict), (
            f"legacy payload-dict contract broken: got {type(payload).__name__}"
        )
        captured.append(payload)
        return "part summary"

    chunks = Chunk_Lib.improved_chunking_process(
        "Sentence one two three. Sentence four five six. " * 8,
        {
            "method": "rolling_summarize",
            "summarize_min_chunk_tokens": 10,
            "summarization_detail": 1.0,
        },
        llm_call_function_for_chunker=fake_llm,
        llm_api_config_for_chunker={},
    )
    assert captured, "payload-dict callback was never invoked"
    assert all("system_message" in p for p in captured)
    assert chunks and chunks[0]["text"]  # summary, not ProcessingError
    assert chunks[0]["metadata"]["chunk_method"] == "rolling_summarize"
    assert chunks[0]["metadata"]["chunk_index"] == 1


def test_improved_chunking_process_honors_template_kwarg():
    # C2: template=/template_manager= kwargs previously accepted but ignored
    # (the body never read them). The 'conversation' template's chunk stage
    # pins method='sentences'; without the template the default is 'words'.
    # Assert the template path is actually taken.
    chunks = Chunk_Lib.improved_chunking_process(
        "Introduction sentence one. Methods sentence here. Results are shown. "
        "Discussion follows. " * 6,
        {"max_size": 8, "overlap": 0},
        template="conversation",
    )
    assert chunks, "template chunking produced no chunks"
    assert all(c["metadata"]["chunk_method"] == "sentences" for c in chunks), (
        "template= kwarg was ignored: chunk_method would be the options "
        f"default, got {chunks[0]['metadata']['chunk_method']!r}"
    )


def test_tokens_poisoned_failure_cache_still_raises(monkeypatch):
    # I1: the Q2 check must catch a fallback that happens DURING the engine
    # call, not just a pre-call state. The engine's per-call strategy is an
    # ephemeral instance (tokenizer override in options), so inspecting a
    # cached get_strategy instance was dead code. Simulate a resolution that
    # degrades to the fallback AFTER a successful call (what a poisoned
    # class-level _failed_tokenizers set produces) by flipping the stub
    # between calls: first call resolves a real tokenizer, second call the
    # fallback -- the shim must raise.
    from tldw_chatbook.Chunking.engine.strategies import tokens as tokens_mod

    state = {"fallback": False}

    def factory():
        if state["fallback"]:
            return tokens_mod.FallbackTokenizer("gpt2")
        return _StubRealTokenizer()

    _patch_tokenizer_property(monkeypatch, factory)

    # Prime: resolution succeeds, chunks produced.
    assert Chunk_Lib.improved_chunking_process(_TOKENS_TEXT, _TOKENS_OK)
    # Degrade: the engine now falls back mid-call -> the shim must raise.
    state["fallback"] = True
    with pytest.raises(Chunk_Lib.ChunkingError, match="tiktoken"):
        Chunk_Lib.improved_chunking_process(_TOKENS_TEXT, _TOKENS_OK)
    with pytest.raises(Chunk_Lib.ChunkingError, match="tiktoken"):
        Chunk_Lib.Chunker(dict(_TOKENS_OK)).chunk_text(_TOKENS_TEXT, method="tokens")
    # Recovered: works again.
    state["fallback"] = False
    assert Chunk_Lib.improved_chunking_process(_TOKENS_TEXT, _TOKENS_OK)


def test_tokens_tiktoken_only_install_uses_engine(monkeypatch):
    # I2: on the Q2 target install (tiktoken present, transformers absent)
    # the legacy TokenBasedChunker seam is transformers-only and resolves the
    # word-approximation fallback -- keying the decision on it silently
    # degraded to word approximation and never used tiktoken. The shim must
    # probe the ENGINE's resolution and delegate to the engine when it
    # resolves a real tokenizer, even though the legacy seam is unusable.
    from tldw_chatbook.Chunking import token_chunker as legacy_token_chunker

    # Engine side: resolution succeeds (stand-in for tiktoken present).
    _patch_tokenizer_property(monkeypatch, _StubRealTokenizer)
    # Legacy seam: transformers-only -> word-approximation fallback.
    monkeypatch.setattr(
        legacy_token_chunker,
        "TransformersTokenizer",
        legacy_token_chunker.FallbackTokenizer,
    )

    # The shim delegates to the engine instead of word-approximating: chunk
    # texts are the stub's decode output (w0 w1 ...), NOT the source words a
    # legacy word-approximation would have produced.
    chunks = Chunk_Lib.improved_chunking_process(_TOKENS_TEXT, _TOKENS_OK)
    assert chunks, "tiktoken-only install produced no chunks"
    assert all(
        c["text"].split() and c["text"].split()[0].startswith("w") for c in chunks
    ), (
        "expected engine (tiktoken) chunking, got legacy word approximation: "
        f"{[c['text'] for c in chunks]}"
    )


def test_tokens_overlap_geq_max_size_raises():
    # I4: legacy chunk_by_tokens raised ValueError("Token overlap X must be
    # less than max_tokens Y") for overlap >= max_tokens; the shim wraps it
    # as ChunkingError. The old shim swallowed it and returned [] (zero
    # chunks, silently). Note overlap defaults to 200, so this is reachable
    # with stock defaults and any tokens max_size < 200.
    for overlap, max_size in ((5, 3), (200, 100), (3, 3)):
        with pytest.raises(Chunk_Lib.ChunkingError, match="must be less than"):
            Chunk_Lib.improved_chunking_process(
                "one two three four five",
                {"method": "tokens", "max_size": max_size, "overlap": overlap},
            )
        with pytest.raises(Chunk_Lib.ChunkingError, match="must be less than"):
            Chunk_Lib.Chunker(
                {"method": "tokens", "max_size": max_size, "overlap": overlap}
            ).chunk_text("one two three four five", method="tokens")


def test_flat_offsets_correct_for_overlapping_chunks():
    # I5: with overlap, a plain text.find(cursor) mislocates chunks whose
    # text starts BEFORE the cursor (returns -1) and can report
    # end_char > len(text). Offsets must stay within [0, len(text)] and each
    # synthesized span must contain the chunk's words.
    text = "One two three four. Five six seven eight."
    n = len(text)
    chunks = Chunk_Lib.improved_chunking_process(
        text, {"method": "words", "max_size": 2, "overlap": 1}
    )
    assert len(chunks) > 2, "expected overlapping chunks"
    for chunk in chunks:
        assert 0 <= chunk["start_char"] <= chunk["end_char"] <= n
        span_words = text[chunk["start_char"] : chunk["end_char"]].split()
        for word in chunk["text"].split():
            assert word in span_words, (
                f"chunk {chunk['text']!r} mapped to span "
                f"{text[chunk['start_char']:chunk['end_char']]!r}"
            )
