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
    def test_method_yields_string_text(self, method, content):
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
