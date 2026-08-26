"""Contract tests for ChunkingService across every supported method."""

from __future__ import annotations

import pytest


class TestEveryChunkingMethodReturnsUsableText:
    """Every supported method must yield dicts whose ``text`` is a string.

    ``Chunker.chunk_text`` is not uniform: the text methods (words, sentences,
    paragraphs, tokens, semantic) return plain strings, while the
    structure-aware ones (json, xml, ebook_chapters) return dicts carrying their
    text alongside metadata. ``ChunkingService`` assumed strings and called
    ``.split()`` on them, so **every json and xml chunk request died** with
    "'dict' object has no attribute 'split'" (task-841).

    Found by exercising each method rather than by reading the dispatch, which
    looks uniform. This is the third instance in one session of a caller
    assuming a shape its callee does not produce -- see
    ``backlog/docs/lessons-testing-evidence.md``.
    """

    PROSE = (
        "The first sentence is here. The second follows it. A third arrives.\n\n"
        "A second paragraph begins. It has two sentences.\n\nThird paragraph."
    ) * 2

    CASES = [
        ("words", PROSE),
        ("sentences", PROSE),
        ("paragraphs", PROSE),
        ("tokens", PROSE),
        ("semantic", PROSE),
        ("json", '{"data": {"k1": "some text here", "k2": "more", "k3": "third"}}'),
        ("xml", "<root><item>alpha beta gamma</item><item>delta epsilon</item></root>"),
    ]

    @pytest.mark.parametrize(("method", "content"), CASES)
    def test_method_yields_string_text(self, method, content, monkeypatch):
        if method == "tokens":
            # This used to patch the legacy token_chunker.TransformersTokenizer
            # seam to force the word-approximation fallback. The Chunk_Lib shim
            # now routes 'tokens' through the ENGINE's tokenizer resolution
            # (Q2: refuse rather than silently word-approximate; engine-parity
            # task 3, review I2), so the legacy seam no longer decides the
            # path. Patch the ENGINE strategy's tokenizer property instead --
            # to a network-free stub of a REAL tokenizer -- so this test still
            # exercises the shape contract offline: the engine path produces
            # decoded-string chunks exactly like a real tokenizer would.
            from tldw_chatbook.Chunking.engine.strategies.tokens import (
                TokenChunkingStrategy,
            )

            class _StubTokenizer:
                def encode(self, text):
                    return list(range(len(text.split())))

                def decode(self, ids, **_kwargs):
                    return " ".join(f"w{i}" for i in ids)

                def count_tokens(self, text):
                    return len(text.split())

            monkeypatch.setattr(
                TokenChunkingStrategy,
                "tokenizer",
                property(lambda self: _StubTokenizer()),
            )
        elif method == "semantic":
            import tldw_chatbook.Chunking.Chunk_Lib as chunk_lib

            monkeypatch.setattr(chunk_lib, "_ensure_nltk", lambda: None)

        from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

        try:
            chunks = ChunkingService().chunk_text(
                content, chunk_size=20, chunk_overlap=5, method=method
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            # ``semantic`` needs NLTK's punkt tokeniser data, which is a runtime
            # download rather than a declared dependency. Skip when it is absent
            # rather than fail: the contract under test is the chunk SHAPE, and
            # pretending the data gap is a shape bug would bury both.
            if "punkt" in str(exc) or "NLTK" in str(exc):
                pytest.skip(f"{method} needs NLTK data that is not installed here")
            raise

        assert chunks, f"{method} produced no chunks at all"
        for chunk in chunks:
            assert isinstance(chunk, dict), f"{method} yielded {type(chunk).__name__}"
            assert isinstance(chunk.get("text"), str), (
                f"{method} chunk text is {type(chunk.get('text')).__name__}, "
                "not a string -- downstream .split() and storage both break"
            )
            assert chunk["text"].strip(), f"{method} yielded an empty chunk"

    def test_a_structured_chunk_without_a_text_field_is_not_dropped(self):
        """Serialise rather than discard: content still reaches the index."""
        from tldw_chatbook.RAG_Search.chunking_service import _chunk_to_text

        assert _chunk_to_text({"heading": "H", "body": "B"})
        assert _chunk_to_text({"text": "plain"}) == "plain"
        assert _chunk_to_text("already a string") == "already a string"


class TestSemanticChunkingWithoutNltkData:
    """Semantic chunking must degrade, not raise, when its corpus is absent.

    nltk's sentence tokeniser needs a corpus that is a runtime download rather
    than part of the package, so a machine can have nltk installed and still
    raise LookupError deep inside a chunking call -- the same call succeeding
    or failing purely on what happened to be cached (task-842).

    These tests simulate the missing corpus at its real seam: they replace
    ``nltk.tokenize.sent_tokenize`` (which ``_ensure_nltk`` re-imports on every
    call) with one that raises LookupError exactly as nltk does. Deliberately
    NOT done by pointing NLTK_DATA at an empty directory: ``nltk.data.path`` is
    built when nltk is first imported, so once any earlier test has imported
    nltk the env var has no effect and the test would silently pass against a
    fully working tokeniser on any developer machine that has the corpus.
    """

    @staticmethod
    def _simulate_missing_corpus(monkeypatch, *, downloadable=False):
        """Put Chunk_Lib in the state of a machine whose corpus is missing.

        Args:
            monkeypatch: pytest's monkeypatch fixture.
            downloadable: When True, the stub download provisions the corpus,
                so the re-probe succeeds; when False it is a no-op and no
                network is touched either way.

        Returns:
            The Chunk_Lib module, with its lazy-import latches reset.
        """
        import nltk.tokenize
        import tldw_chatbook.Chunking.Chunk_Lib as chunk_lib

        state = {"corpus_present": False}

        def fake_sent_tokenize(text, language="english"):
            if not state["corpus_present"]:
                raise LookupError("Resource 'punkt_tab' not found.")
            return [s.strip() for s in text.split(".") if s.strip()]

        def fake_download(_nltk):
            state["corpus_present"] = downloadable

        monkeypatch.setattr(nltk.tokenize, "sent_tokenize", fake_sent_tokenize)
        monkeypatch.setattr(chunk_lib, "_download_nltk_tokenizer_corpora", fake_download)
        # Reset every latch, and unbind whatever an earlier test left bound --
        # otherwise the module-level `sent_tokenize` could still be a working
        # tokeniser and the assertions would prove nothing.
        monkeypatch.setattr(chunk_lib, "NLTK_AVAILABLE", True)
        monkeypatch.setattr(chunk_lib, "nltk", None)
        monkeypatch.setattr(chunk_lib, "_nltk_tokenizer_unusable", False)
        monkeypatch.setattr(chunk_lib, "_nltk_data_ready", False)
        monkeypatch.setattr(
            chunk_lib, "sent_tokenize", chunk_lib._sent_tokenize_fallback
        )
        return chunk_lib

    def test_semantic_falls_back_instead_of_raising(self, monkeypatch):
        """A missing corpus yields simpler chunks, never a failed ingest.

        Args:
            monkeypatch: pytest's monkeypatch fixture.
        """
        self._simulate_missing_corpus(monkeypatch)

        from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

        prose = (
            "The first sentence is here. The second follows it. A third arrives. "
            "A fourth sentence appears. A fifth concludes."
        ) * 2
        chunks = ChunkingService().chunk_text(
            prose, chunk_size=20, chunk_overlap=5, method="semantic"
        )

        assert chunks, "semantic chunking produced nothing without punkt data"
        assert all(isinstance(c.get("text"), str) and c["text"].strip() for c in chunks)
        # It must still CHUNK. The old fallback split on newlines, so ordinary
        # single-paragraph prose came back as one chunk holding the whole
        # document -- technically "not raising" while silently doing nothing.
        assert len(chunks) > 1, (
            f"fallback returned {len(chunks)} chunk(s) for {len(prose.split())} "
            "words at chunk_size=20; it stopped chunking rather than degrading"
        )

    def test_the_corpus_download_still_runs_and_rebinds_the_real_tokenizer(
        self, monkeypatch
    ):
        """Downloading the corpus is the remediation; it must stay reachable.

        Probing before binding must not cost the download that made sentence
        chunking work in the first place. When the download provisions the
        corpus, the re-probe succeeds and the real tokeniser binds.

        Args:
            monkeypatch: pytest's monkeypatch fixture.
        """
        chunk_lib = self._simulate_missing_corpus(monkeypatch, downloadable=True)

        assert chunk_lib._ensure_nltk() is not None, "download path never ran"
        assert chunk_lib.sent_tokenize is not chunk_lib._sent_tokenize_fallback

    def test_an_unusable_tokenizer_is_latched_not_re_probed_every_call(
        self, monkeypatch
    ):
        """The verdict is cached, so the warning and download happen once.

        ``nltk`` stays None on this path, so without a latch every chunking
        call would re-probe, re-attempt a network download and re-log.

        Args:
            monkeypatch: pytest's monkeypatch fixture.
        """
        chunk_lib = self._simulate_missing_corpus(monkeypatch)

        attempts = []
        real_probe = chunk_lib._probe_sent_tokenize
        monkeypatch.setattr(
            chunk_lib,
            "_probe_sent_tokenize",
            lambda tokenize: (attempts.append(1), real_probe(tokenize))[1],
        )

        for _ in range(3):
            assert chunk_lib._ensure_nltk() is None
        chunk_lib.ensure_nltk_data()

        assert len(attempts) == 2, (
            f"probed {len(attempts)} times; expected one probe plus one "
            "re-probe after the download attempt, then a latched verdict"
        )
        assert chunk_lib._nltk_data_ready is False


# ---------------------------------------------------------------------------
# Phase B convergence (spec §6.3.1): all methods route through the engine
# (chunking-engine-parity task 7 -- the module's regex splitter is deleted
# and ChunkingService delegates to the Chunk_Lib shim for every method).
# ---------------------------------------------------------------------------
import pytest
from tldw_chatbook.RAG_Search import chunking_service
from tldw_chatbook.RAG_Search.chunking_service import ChunkingService, ChunkingError


def test_validation_messages_preserved():
    svc = ChunkingService()
    with pytest.raises(ChunkingError, match="max_words must be positive"):
        svc.chunk_text("text", chunk_size=0, chunk_overlap=0, method="words")
    with pytest.raises(ChunkingError, match="Overlap must be non-negative"):
        svc.chunk_text("text", chunk_size=10, chunk_overlap=-1, method="words")
    with pytest.raises(ChunkingError, match="Overlap must be less than max_words"):
        svc.chunk_text("text", chunk_size=10, chunk_overlap=10, method="words")


def test_all_methods_flat_contract():
    svc = ChunkingService()
    for method in ["words", "sentences", "paragraphs"]:
        chunks = svc.chunk_text(
            "One two three. Four five six. Seven eight nine ten.",
            chunk_size=4, chunk_overlap=1, method=method,
        )
        assert chunks
        for c in chunks:
            assert set(c) >= {"text", "start_char", "end_char", "word_count", "chunk_index"}


def test_ebook_chapters_no_whitelist():
    text = "# Chapter 1\n\nText one.\n\n# Chapter 2\n\nText two.\n"
    chunks = svc_ebook(text)
    assert len(chunks) >= 2


def svc_ebook(text):
    svc = ChunkingService()
    return svc.chunk_text(text, chunk_size=400, chunk_overlap=0, method="ebook_chapters")


def test_exceptions_are_aliases_of_the_engine_classes():
    """One exception tree: ``except chunking_service.ChunkingError`` and
    ``except Chunk_Lib.ChunkingError`` must catch the same objects."""
    from tldw_chatbook.Chunking import Chunk_Lib

    assert chunking_service.ChunkingError is Chunk_Lib.ChunkingError
    assert (
        chunking_service.InvalidChunkingMethodError
        is Chunk_Lib.InvalidChunkingMethodError
    )
    assert issubclass(
        chunking_service.InvalidChunkingMethodError, chunking_service.ChunkingError
    )


def test_module_level_improved_chunking_process_keeps_validation():
    """The options-dict entry point keeps the legacy size/overlap contract
    (the engine clamps instead of raising, so the wrapper must enforce it)."""
    with pytest.raises(ChunkingError, match="Overlap must be less than max_words"):
        chunking_service.improved_chunking_process(
            "text", {"method": "words", "max_size": 10, "overlap": 10}
        )


def test_module_level_improved_chunking_process_rejects_unknown_methods():
    with pytest.raises(chunking_service.InvalidChunkingMethodError):
        chunking_service.improved_chunking_process("text", {"method": "bogus"})
