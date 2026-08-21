"""Task 5 (PR A) tests: ``Chunking/template_runtime.py`` — the one mapper,
the one resolver, and ``apply_template`` with flat-contract synthesis.

Spec: ``Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md``
§6.2 (the three functions), §6.3 (fencing), §6.4 (chunk-contract synthesis,
the measurement-corrected ruling). ACs 7-13.

Pinned values below were MEASURED against the vendored processor in this tree
(2026-08-21), not copied from the brief: the engine's sentences strategy
fills chunks sequentially ([s1, s2], [s3]) and joins sentences with a single
space, so the brief's illustrative pin was wrong. The pins here reflect
reality; see the task-5 report for the deviation note.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

import pytest

from tldw_chatbook.Chunking import template_runtime as tr
from tldw_chatbook.Chunking.engine.exceptions import TemplateError

# ---------------------------------------------------------------------------
# Fixtures: the SERVER FLAT shape (spec §4.1). The DB's chatbook-pipeline
# rows are converted in PR B; here templates are handed over as flat dicts.
# ---------------------------------------------------------------------------

FLAT = {
    "preprocessing": [
        {"operation": "normalize_whitespace", "config": {"max_line_breaks": 1}}
    ],
    "chunking": {"method": "sentences", "config": {"max_size": 2, "overlap": 0}},
    "postprocessing": [{"operation": "filter_empty", "config": {}}],
    "classifier": {"media_types": ["document"], "min_score": 0.5, "priority": 10},
    "metadata": {},
}

# Same, but the postprocess stage deletes a chunk ("Third one." is 10 chars,
# under min_length 12) so index alignment breaks and offsets are
# unsynthesizable (§6.4 ruling 3).
FLAT_DELETING = {
    "preprocessing": [
        {"operation": "normalize_whitespace", "config": {"max_line_breaks": 1}}
    ],
    "chunking": {"method": "sentences", "config": {"max_size": 2, "overlap": 0}},
    "postprocessing": [{"operation": "filter_empty", "config": {"min_length": 12}}],
    "metadata": {},
}

# No preprocessing at all: offsets are relative to the source text and
# ``offset_basis`` must be ``"source"`` (§6.4 ruling 2).
FLAT_NO_PRE = {
    "chunking": {"method": "words", "config": {"max_size": 3, "overlap": 0}},
    "postprocessing": [],
    "metadata": {},
}

# Chunking stage only — the contrast that proves AC 10's "differs from the
# chunking stage alone": without preprocessing the double space survives.
FLAT_CHUNK_ONLY = {
    "chunking": {"method": "sentences", "config": {"max_size": 2, "overlap": 0}},
    "metadata": {},
}

PINNED_TEXT = "First  sentence.\n\n\n\nSecond sentence here.  Third one."
# What normalize_whitespace(max_line_breaks=1) turns PINNED_TEXT into:
PINNED_PREPROCESSED = "First sentence.\nSecond sentence here. Third one."

WORDS_TEXT = "alpha beta gamma delta epsilon zeta eta."


# ---------------------------------------------------------------------------
# AC 7 — the ONE mapper, with the missing-`chunking` guard
# ---------------------------------------------------------------------------


class TestTemplateFromRecord:
    def test_mapper_guards_missing_chunking(self):
        with pytest.raises(TemplateError, match="chunking"):
            tr.template_from_record(
                {"name": "x", "template_json": '{"preprocessing": []}'}
            )

    def test_mapper_guards_missing_chunking_on_flat_dict(self):
        # A flat dict without `chunking` must raise TemplateError (clear
        # message), never KeyError (spec §4.3: the server's copies raise
        # KeyError into a generic handler — chatbook's does not).
        with pytest.raises(TemplateError, match="chunking"):
            tr.template_from_record({"name": "x", "preprocessing": []})

    def test_mapper_guards_missing_chunking_method(self):
        with pytest.raises(TemplateError, match="method"):
            tr.template_from_record(
                {"name": "x", "chunking": {"config": {"max_size": 2}}}
            )

    def test_mapper_guards_invalid_template_json(self):
        with pytest.raises(TemplateError, match="template_json"):
            tr.template_from_record({"name": "x", "template_json": "{not json"})

    def test_mapper_maps_flat_shape_to_engine_template(self):
        from tldw_chatbook.Chunking.engine.templates import (
            ChunkingTemplate as EngineChunkingTemplate,
        )

        mapped = tr.template_from_record(FLAT)
        assert isinstance(mapped, EngineChunkingTemplate)
        assert [stage.name for stage in mapped.stages] == [
            "preprocess",
            "chunk",
            "postprocess",
        ]
        assert mapped.base_method == "sentences"
        assert mapped.default_options == {"max_size": 2, "overlap": 0}
        assert mapped.stages[1].operations == [FLAT["chunking"]]
        assert mapped.stages[0].operations == FLAT["preprocessing"]
        assert mapped.stages[2].operations == FLAT["postprocessing"]

    def test_mapper_accepts_record_with_dict_template_json(self):
        record = {"name": "flat", "template_json": dict(FLAT)}
        mapped = tr.template_from_record(record)
        assert mapped.base_method == "sentences"


# ---------------------------------------------------------------------------
# AC 10 / 11 / 12 — apply_template
# ---------------------------------------------------------------------------


class TestApplyTemplate:
    def test_apply_runs_pre_and_post_pinned_exact_output(self):
        # AC 10: preprocessing AND postprocessing both run; the output is an
        # exact pinned value that differs from the chunking stage alone.
        out = tr.apply_template(FLAT, PINNED_TEXT)
        # Pinned (measured): normalize_whitespace collapses the quad newline
        # and the double spaces; sentences(max_size=2) fills sequentially.
        assert [c["text"] for c in out] == [
            "First sentence. Second sentence here.",
            "Third one.",
        ]
        # The chunking stage alone keeps the original double space (no
        # preprocessing) — the pinned contrast proving pre ran.
        chunk_only = tr.apply_template(FLAT_CHUNK_ONLY, PINNED_TEXT)
        assert [c["text"] for c in chunk_only] == [
            "First  sentence. Second sentence here.",
            "Third one.",
        ]
        # Postprocessing demonstrably ran: under FLAT_DELETING the filter
        # drops the 10-char chunk that FLAT (min_length 10) keeps.
        deleting = tr.apply_template(FLAT_DELETING, PINNED_TEXT)
        assert [c["text"] for c in deleting] == ["First sentence. Second sentence here."]

    def test_flat_contract_synthesized(self):
        # AC 11: full flat contract on every chunk.
        out = tr.apply_template(FLAT, PINNED_TEXT)
        assert [c["chunk_index"] for c in out] == [0, 1]  # 0-based top-level
        for c in out:
            assert {
                "text",
                "start_char",
                "end_char",
                "word_count",
                "chunk_index",
                "total_chunks",
                "metadata",
            } <= set(c)
            assert c["total_chunks"] == 2
            assert c["word_count"] == len(c["text"].split())
        assert [c["word_count"] for c in out] == [5, 2]
        # Pinned offsets, computed against the TRANSFORMED text (§6.4).
        assert (out[0]["start_char"], out[0]["end_char"]) == (0, 37)
        assert (out[1]["start_char"], out[1]["end_char"]) == (38, 48)
        assert (
            PINNED_PREPROCESSED[out[0]["start_char"] : out[0]["end_char"]]
            == "First sentence.\nSecond sentence here."
        )

    def test_offset_basis_preprocessed(self):
        out = tr.apply_template(FLAT, PINNED_TEXT)
        assert out[0]["metadata"]["offset_basis"] == "preprocessed:normalize_whitespace"
        assert out[1]["metadata"]["offset_basis"] == "preprocessed:normalize_whitespace"

    def test_offset_basis_source_when_no_preprocessing_rewrites(self):
        out = tr.apply_template(FLAT_NO_PRE, WORDS_TEXT)
        assert len(out) == 3
        for c in out:
            assert c["metadata"]["offset_basis"] == "source"
            # Words chunks are exact substrings: source-relative offsets.
            assert WORDS_TEXT[c["start_char"] : c["end_char"]] == c["text"]

    def test_unsynthesizable_offsets_omitted_never_none(self):
        # AC 12 (unit half): when postprocessing deletes/merges chunks the
        # offset keys are OMITTED entirely — never present-and-None.
        out = tr.apply_template(FLAT_DELETING, PINNED_TEXT)
        assert len(out) == 1
        for c in out:
            assert "start_char" not in c
            assert "end_char" not in c
        # Across every fixture shape: no chunk ever carries present-and-None.
        for template, text in (
            (FLAT, PINNED_TEXT),
            (FLAT_DELETING, PINNED_TEXT),
            (FLAT_NO_PRE, WORDS_TEXT),
            (FLAT_CHUNK_ONLY, PINNED_TEXT),
        ):
            for c in tr.apply_template(template, text):
                assert not ("start_char" in c and c["start_char"] is None)
                assert not ("end_char" in c and c["end_char"] is None)

    def test_apply_passes_options_through(self):
        # Runtime options reach the engine (method override measured to
        # win over the template's chunking.method). The words method plus
        # the template's max_size=2 yields 2-word chunks; FLAT's
        # filter_empty (min_length 10) then drops the 4-char "eta.".
        out = tr.apply_template(FLAT, WORDS_TEXT, {"method": "words"})
        assert [c["text"] for c in out] == [
            "alpha beta",
            "gamma delta",
            "epsilon zeta",
        ]

    def test_apply_empty_text_returns_empty_list(self):
        assert tr.apply_template(FLAT, "") == []
        assert tr.apply_template(FLAT, "   ") == []

    def test_apply_rejects_template_missing_chunking(self):
        with pytest.raises(TemplateError, match="chunking"):
            tr.apply_template({"preprocessing": []}, "some text")


# ---------------------------------------------------------------------------
# resolve_template (AC 8) — the ONLY name→template resolution
# ---------------------------------------------------------------------------


class _ConnDb:
    """Minimal DB handle shaped like the Media DB wrappers."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get_connection(self) -> sqlite3.Connection:
        return self._conn


def _templates_db() -> _ConnDb:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # The v6 table (PR B's task 8 changes the columns; name/template_json
    # are the stable ones the resolver is allowed to touch).
    conn.execute(
        """
        CREATE TABLE ChunkingTemplates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            template_json TEXT NOT NULL,
            is_system BOOLEAN DEFAULT 0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO ChunkingTemplates (name, template_json) VALUES (?, ?)",
        ("flat", json.dumps(FLAT)),
    )
    conn.execute(
        "INSERT INTO ChunkingTemplates (name, template_json) VALUES (?, ?)",
        ("broken", "{not json"),
    )
    return _ConnDb(conn)


class TestResolveTemplate:
    def test_resolves_name_to_template_dict(self):
        resolved = tr.resolve_template(_templates_db(), "flat")
        assert resolved is not None
        assert resolved["name"] == "flat"
        assert resolved["chunking"]["method"] == "sentences"
        assert resolved["preprocessing"][0]["operation"] == "normalize_whitespace"

    def test_unknown_name_returns_none(self):
        assert tr.resolve_template(_templates_db(), "nope") is None

    def test_corrupt_json_returns_none(self):
        assert tr.resolve_template(_templates_db(), "broken") is None

    def test_queries_only_the_stable_columns_no_deleted_filter(self):
        # Pre-flight ruling: query ONLY (name, template_json) — stable across
        # v6/v7 — with NO deleted filter yet (v6 has no deleted column; the
        # CRUD rewrite in PR B's task 8 adds it).
        db = _templates_db()
        statements: list[str] = []
        db.get_connection().set_trace_callback(statements.append)
        try:
            assert tr.resolve_template(db, "flat") is not None
        finally:
            db.get_connection().set_trace_callback(None)
        sql = " ".join(statements)
        # sqlite3's trace callback expands parameters; accept either the
        # placeholder or the bound literal.
        assert re.search(
            r"SELECT\s+name\s*,\s*template_json\s+FROM\s+ChunkingTemplates"
            r"\s+WHERE\s+name\s*=\s*(?:\?|'[^']*')",
            sql,
            re.IGNORECASE,
        ), sql
        assert "deleted" not in sql.lower(), sql


# ---------------------------------------------------------------------------
# Enumeration guards (AC 7 / AC 8 / AC 9) — grep-based, with self-checks
# ---------------------------------------------------------------------------

PKG_ROOT = Path(tr.__file__).resolve().parents[1]  # .../tldw_chatbook (the package)
# The vendored engine tree mirrors upstream and is fenced, not rewritten:
# upstream's own TemplateManager.load_template mapping lives inside it.
VENDORED_DIR_PARTS = ("Chunking", "engine")
VENDORED_SHIM_PARTS = ("Chunking", "_shims")

MAPPER_GUARD_ALLOWED = {"Chunking/template_runtime.py"}
# Legacy resolution slated for the PR B CRUD rewrite (spec AC 26 rewrites
# ChunkingInteropService; until then its get_template_by_name stays).
# DB/Client_Media_DB_v2.py (task-7): the v7 migration's idempotent-seed
# existence check ("a built-in name that already exists as a custom row is
# left alone") matches the regex but resolves no template — it never reads
# a template body, only name liveness.
RESOLVER_GUARD_ALLOWED = {
    "Chunking/template_runtime.py",
    "Chunking/chunking_interop_library.py",
    "DB/Client_Media_DB_v2.py",
}

_RE_VENDORED_TEMPLATES_IMPORT = re.compile(r"engine\.templates")
_RE_NAME_RESOLUTION = re.compile(
    r"FROM\s+ChunkingTemplates\s+WHERE\s+name", re.IGNORECASE
)
_RE_FENCED_CONSTRUCTION = re.compile(
    r"(?<![A-Za-z0-9_])Template(?:Manager|Classifier|Learner)\s*\("
)


def _production_py_files():
    for path in sorted(PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(PKG_ROOT)
        parts = rel.parts
        if parts[:2] == VENDORED_DIR_PARTS or parts[:2] == VENDORED_SHIM_PARTS:
            continue
        yield path, rel


def _scan(pattern: re.Pattern[str]) -> list[str]:
    hits = []
    for path, rel in _production_py_files():
        try:
            # errors="replace": a single non-UTF8 fixture under the package
            # must not blind the guard, and the patterns are ASCII.
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern.search(src):
            hits.append(rel.as_posix())
    return hits


class TestEnumerationGuards:
    def test_exactly_one_flat_mapper_in_production(self):
        # AC 7: only template_runtime.py touches the vendored templates
        # surface — a second flat→ChunkingTemplate mapper would have to
        # import it, and this guard would see that import.
        assert _scan(_RE_VENDORED_TEMPLATES_IMPORT) == sorted(MAPPER_GUARD_ALLOWED)

    def test_the_mapper_guard_can_see_what_it_guards(self):
        # Self-check: seed a second mapper under the package and prove the
        # guard goes red (finds it), then clean up.
        probe = PKG_ROOT / "Chunking" / f"_guard_probe_mapper_{uuid.uuid4().hex}.py"
        probe.write_text(
            "# Guard self-check probe; deleted by the test that wrote it.\n"
            "from tldw_chatbook.Chunking.engine.templates import ChunkingTemplate\n"
            "\n"
            "def rogue_flat_mapper(record):\n"
            '    return ChunkingTemplate(name=record.get("name", "rogue"))\n',
            encoding="utf-8",
        )
        try:
            hits = _scan(_RE_VENDORED_TEMPLATES_IMPORT)
            assert probe.relative_to(PKG_ROOT).as_posix() in hits
            assert set(hits) - MAPPER_GUARD_ALLOWED == {
                probe.relative_to(PKG_ROOT).as_posix()
            }
        finally:
            probe.unlink(missing_ok=True)

    def test_resolve_template_is_the_only_name_resolution(self):
        # AC 8: name→template resolution lives in template_runtime (plus the
        # documented legacy site the PR B rewrite removes).
        assert _scan(_RE_NAME_RESOLUTION) == sorted(RESOLVER_GUARD_ALLOWED)

    def test_the_resolver_guard_can_see_what_it_guards(self):
        probe = PKG_ROOT / "Chunking" / f"_guard_probe_resolver_{uuid.uuid4().hex}.py"
        probe.write_text(
            "# Guard self-check probe; deleted by the test that wrote it.\n"
            "def rogue_resolve(db, name):\n"
            "    return db.get_connection().execute(\n"
            '        "SELECT * FROM ChunkingTemplates WHERE name = ?", (name,)\n'
            "    ).fetchone()\n",
            encoding="utf-8",
        )
        try:
            hits = _scan(_RE_NAME_RESOLUTION)
            assert probe.relative_to(PKG_ROOT).as_posix() in hits
            assert set(hits) - RESOLVER_GUARD_ALLOWED == {
                probe.relative_to(PKG_ROOT).as_posix()
            }
        finally:
            probe.unlink(missing_ok=True)

    def test_no_production_module_constructs_fenced_classes(self):
        # AC 9 (source half): TemplateManager/TemplateClassifier/TemplateLearner
        # are vendored-but-unused. The lookbehind excludes the legacy
        # pydantic ChunkingTemplateManager (deleted wholesale in PR C).
        assert _scan(_RE_FENCED_CONSTRUCTION) == []


# ---------------------------------------------------------------------------
# AC 9 — fencing, observed the way the code cannot fake: the templates
# directory TemplateManager construction would mkdir does not exist.
# ---------------------------------------------------------------------------


class TestTemplateManagerFencing:
    def test_templates_directory_absent_with_positive_control(self):
        from tldw_chatbook.Chunking.engine.templates import TemplateManager

        probe = (
            Path(tr.__file__).resolve().parent / "engine" / "template_library"
        )
        assert not probe.exists(), (
            "engine/template_library exists: some code constructed TemplateManager"
        )
        stray = [p for p in PKG_ROOT.rglob("template_library") if p.is_dir()]
        assert stray == []

        # Positive control: constructing TemplateManager makes the probe
        # appear — proving the assertion above can actually fail.
        TemplateManager()
        assert probe.exists()
        try:
            shutil.rmtree(probe)
        finally:
            assert not probe.exists()


# ---------------------------------------------------------------------------
# AC 12 (integration half) — index template chunks through the RAG path,
# search with citations, no TypeError. The precedent is
# Tests/RAG/simplified/test_rag_service_basic.py (memory vector store +
# mock embeddings).
# ---------------------------------------------------------------------------


def _memory_rag_service():
    from tldw_chatbook.RAG_Search.simplified import create_rag_service_from_config
    from tldw_chatbook.RAG_Search.simplified.config import (
        ChunkingConfig,
        EmbeddingConfig,
        RAGConfig,
        SearchConfig,
        VectorStoreConfig,
    )

    config = RAGConfig(
        embedding=EmbeddingConfig(model="mock", device="cpu"),
        vector_store=VectorStoreConfig(
            type="memory",
            persist_directory=None,
            collection_name=f"tplrt_{uuid.uuid4().hex[:8]}",
            distance_metric="cosine",
        ),
        chunking=ChunkingConfig(chunk_size=100, chunk_overlap=20),
        search=SearchConfig(default_top_k=5, score_threshold=0.0, enable_cache=False),
    )
    return create_rag_service_from_config(config=config)


class TestRagIntegration:
    async def test_template_chunks_index_and_search_without_typeerror(self):
        from tldw_chatbook.RAG_Search.simplified.indexing_helpers import (
            generate_embeddings_batch,
            store_documents_batch,
        )

        service = _memory_rag_service()
        try:
            doc_a_chunks = tr.apply_template(FLAT, PINNED_TEXT)  # offsets present
            doc_b_chunks = tr.apply_template(
                FLAT_DELETING, PINNED_TEXT
            )  # offsets omitted
            documents = [
                {"id": "tpl_a", "title": "Template A", "content": PINNED_TEXT},
                {"id": "tpl_b", "title": "Template B", "content": PINNED_TEXT},
            ]
            all_chunks = doc_a_chunks + doc_b_chunks
            embeddings, failed = await generate_embeddings_batch(
                service, [c["text"] for c in all_chunks], show_progress=False
            )
            assert not failed
            doc_chunk_info = [
                {
                    "doc_idx": 0,
                    "chunk_start": 0,
                    "chunk_count": len(doc_a_chunks),
                    "chunks": doc_a_chunks,
                },
                {
                    "doc_idx": 1,
                    "chunk_start": len(doc_a_chunks),
                    "chunk_count": len(doc_b_chunks),
                    "chunks": doc_b_chunks,
                },
            ]
            results = await store_documents_batch(
                service, documents, doc_chunk_info, embeddings, time.time()
            )
            assert all(r.success for r in results), [r.error for r in results]

            # Search with citations: vector_store.py's citation builders call
            # int(metadata.get("chunk_start", 0)) — present-and-None offsets
            # raise TypeError HERE. Omitted keys must flow through as 0.
            found = await service.search(
                query="First sentence",
                search_type="semantic",
                include_citations=True,
            )
            assert found
            by_chunk = {}
            for chunk in doc_a_chunks:
                by_chunk[chunk["text"]] = chunk
            for chunk in doc_b_chunks:
                by_chunk[chunk["text"]] = chunk
            cited = 0
            for result in found:
                for citation in result.citations:
                    assert isinstance(citation.start_char, int)
                    assert isinstance(citation.end_char, int)
                    chunk = by_chunk.get(result.document)
                    if chunk is not None and "start_char" in chunk:
                        assert citation.start_char == chunk["start_char"]
                    cited += 1
            assert cited > 0
        finally:
            service.close()

    def test_citation_builder_tolerates_omitted_offset_keys(self):
        # The exact consumer the spec cites (vector_store.py:611,
        # ChromaVectorStore._create_citations_from_result) run directly on
        # metadata built the way indexing_helpers.py:263 builds it from a
        # chunk whose offset keys are OMITTED.
        from tldw_chatbook.RAG_Search.simplified.vector_store import (
            ChromaVectorStore,
            SearchResult,
        )

        omitted_chunk = next(iter(tr.apply_template(FLAT_DELETING, PINNED_TEXT)))
        assert "start_char" not in omitted_chunk
        metadata = {
            "doc_id": "tpl_b",
            "doc_title": "Template B",
            "chunk_index": 0,
            "chunk_start": omitted_chunk.get("start_char", 0),
            "chunk_end": omitted_chunk.get("end_char", len(omitted_chunk["text"])),
        }
        result = SearchResult(
            id="tpl_b_chunk_0", score=1.0, document=omitted_chunk["text"],
            metadata=metadata,
        )
        probe = ChromaVectorStore.__new__(ChromaVectorStore)
        citations = probe._create_citations_from_result(result, "First sentence")
        assert citations
        assert citations[0].start_char == 0
        assert isinstance(citations[0].end_char, int)

        # The hazard the omission rule prevents: a present-but-None offset
        # (what a naive "emit None" synthesis would store) is the TypeError.
        poisoned = SearchResult(
            id="tpl_b_chunk_0",
            score=1.0,
            document=omitted_chunk["text"],
            metadata={"doc_id": "tpl_b", "chunk_start": None, "chunk_end": None},
        )
        with pytest.raises(TypeError):
            probe._create_citations_from_result(poisoned, "First sentence")


# ---------------------------------------------------------------------------
# AC 13 — media navigation returns chunk-sized content for template chunks
# (Media/local_media_reading_service.py: chunk nodes carry start_char /
# end_char from UnvectorizedMediaChunks; :2442-2446 falls back to the whole
# document when they are NULL).
# ---------------------------------------------------------------------------


class _NavDb:
    """Real sqlite3 chunk rows + a content-bearing media row."""

    def __init__(self, content: str):
        self._content = content
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE UnvectorizedMediaChunks (
                media_id INTEGER,
                chunk_text TEXT,
                chunk_index INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                chunk_type TEXT,
                deleted INTEGER DEFAULT 0
            )
            """
        )

    def add_chunks(self, media_id: int, chunks: list[dict]) -> None:
        for i, chunk in enumerate(chunks):
            self._conn.execute(
                "INSERT INTO UnvectorizedMediaChunks "
                "(media_id, chunk_text, chunk_index, start_char, end_char, chunk_type, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, 0)",
                (
                    media_id,
                    chunk["text"],
                    chunk.get("chunk_index", i),
                    chunk.get("start_char"),
                    chunk.get("end_char"),
                    "template",
                ),
            )

    def get_connection(self) -> sqlite3.Connection:
        return self._conn

    def get_media_by_id(self, media_id, **kwargs):
        return {"id": media_id, "title": "Doc", "content": self._content}

    def get_media_read_it_later_state(self, media_id):
        return None


class TestMediaNavigation:
    def test_template_chunk_navigation_returns_chunk_sized_content(self):
        from tldw_chatbook.Media.local_media_reading_service import (
            LocalMediaReadingService,
        )

        # Source-basis template (no preprocessing rewrite): chunk offsets
        # index the stored content directly, so navigation is exact.
        chunks = tr.apply_template(FLAT_NO_PRE, WORDS_TEXT)
        assert len(chunks) == 3
        assert all(c["metadata"]["offset_basis"] == "source" for c in chunks)
        db = _NavDb(WORDS_TEXT)
        db.add_chunks(1, chunks)
        service = LocalMediaReadingService(db)

        for order, chunk in enumerate(chunks):
            nav = service.get_media_navigation_content(1, f"chunk-{order}")
            assert nav["content"] == chunk["text"]
            assert len(nav["content"]) < len(WORDS_TEXT)
            assert nav["target"]["target_start"] == chunk["start_char"]
            assert nav["target"]["target_end"] == chunk["end_char"]

    def test_preprocessed_basis_navigation_is_chunk_sized_with_documented_caveat(self):
        # §6.4 caveat: a preprocessing op rewrites the text, so offsets are
        # preprocessed-relative and navigation slices the STORED content at
        # those positions — chunk-sized (AC 13), though not byte-equal to
        # the chunk text. Consumers can test metadata.offset_basis to know.
        from tldw_chatbook.Media.local_media_reading_service import (
            LocalMediaReadingService,
        )

        chunks = tr.apply_template(FLAT, PINNED_TEXT)
        assert chunks[0]["metadata"]["offset_basis"] == "preprocessed:normalize_whitespace"
        db = _NavDb(PINNED_TEXT)
        db.add_chunks(3, chunks)
        service = LocalMediaReadingService(db)

        nav = service.get_media_navigation_content(3, "chunk-0")
        assert 0 < len(nav["content"]) < len(PINNED_TEXT)
        # The preprocessed first chunk maps onto the source's opening span.
        assert nav["content"].startswith("First")

    def test_null_offsets_still_fall_back_to_whole_document(self):
        # Contrast proving the test discriminates: chunk rows with NULL
        # offsets degrade navigation to the whole document (:2442-2446).
        from tldw_chatbook.Media.local_media_reading_service import (
            LocalMediaReadingService,
        )

        chunks = [
            {"text": "First sentence.", "chunk_index": 0},  # offsets absent → NULL
        ]
        db = _NavDb(PINNED_TEXT)
        db.add_chunks(2, chunks)
        service = LocalMediaReadingService(db)

        nav = service.get_media_navigation_content(2, "chunk-0")
        assert nav["content"] == PINNED_TEXT.strip()
