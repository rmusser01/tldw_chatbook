"""Task 10 (PR D): template resolution + precedence on the six ingest seams.

Spec §9.1/§9.2 (ACs 34-37) plus the AC-24 halves carried from Task 8's
review:

* **AC 34** -- ingest resolves picker/batch choice -> config
  ``[chunking] default_template`` -> plain options; re-chunk resolves
  stored per-media -> config default -> plain.
* **AC 35** -- a resolved template's chunk-stage options beat the ingest
  builder's DEFAULTS; only a user-changed form value beats the template
  (the inert-picker trap: ``_ingest_job_options`` always sets
  method/max_size/overlap, which the Chunker's explicit-wins merge would
  let override the template on every path).
* **AC 36** -- governance per media-type family: the same fixture under
  two different templates produces demonstrably different persisted chunk
  rows; the "None" default keeps today's plain-options output.
* **AC 37** -- an unresolvable template name FAILS the item with a NAMED
  error (never a silent fallback to plain options).
* **AC 24a** -- stored-invalid templates are listed WITH a validity flag
  (the data surface; where the flag renders is Task 12's UI).
* **AC 24b** -- the apply path refuses stored-invalid templates with the
  NAMED ``InvalidTemplateError``; the ingest resolution path refuses them
  the same way.

The governance tests run the REAL chunking seam end to end (parse ->
persist -> chunk rows) -- only optional-dependency EXTRACTION seams are
stubbed (pdf: pymupdf4llm absent; audio: ``_transcribe_audio``; image:
``process_image``), matching the established convention in
``test_book_ingestion_chunking.py``.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    parse_local_file_for_ingest,
    persist_parsed_media,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: Two governance templates whose chunk-stage options produce visibly
#: different chunk rows on any multi-word fixture.
TEMPLATE_TINY: Dict[str, Any] = {
    "name": "tiny-words",
    "chunking": {"method": "words", "config": {"max_size": 3, "overlap": 0}},
}
TEMPLATE_BIG: Dict[str, Any] = {
    "name": "big-words",
    "chunking": {"method": "words", "config": {"max_size": 12, "overlap": 0}},
}

#: Fixture body: 24 numbered words -> 8 tiny (3-word) / 2 big (12-word)
#: chunks with zero overlap.
_WORDS = [f"w{i:02d}" for i in range(1, 25)]
_FIXTURE_TEXT = " ".join(_WORDS)


@pytest.fixture()
def media_db(tmp_path: Path) -> MediaDatabase:
    db = MediaDatabase(tmp_path / "media.db", client_id="test-template-parity")
    yield db
    db.close_connection()


def _seed_template(db: MediaDatabase, body: Dict[str, Any]) -> None:
    """Create a template through the validated interop service."""
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )

    service = get_chunking_service(db)
    service.create_template(
        name=body["name"],
        description=f"Governance template {body['name']}",
        template_json={k: v for k, v in body.items() if k != "name"},
    )


def _seed_template_row_direct(
    db: MediaDatabase, name: str, body: Dict[str, Any], *, deleted: bool = False
) -> None:
    """Bypass validate-on-write (the only way a stored-invalid or
    soft-deleted row can exist: conversion-minted / soft-deleted rows)."""
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO ChunkingTemplates (uuid, name, description, "
            "template_json, is_builtin, version, deleted) "
            "VALUES (?, ?, ?, ?, 0, 1, ?)",
            (
                f"uuid-{name}",
                name,
                "direct seed",
                json.dumps(body),
                1 if deleted else 0,
            ),
        )


@pytest.fixture()
def template_db(media_db: MediaDatabase) -> MediaDatabase:
    _seed_template(media_db, TEMPLATE_TINY)
    _seed_template(media_db, TEMPLATE_BIG)
    return media_db


@pytest.fixture(autouse=True)
def _no_config_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Determinism: no test picks up the developer machine's config."""
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda section, key=None, default=None: (
            default if default is not None else None
        ),
    )


def _patch_config_default(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda section, key=None, default=None: (
            name if (section, key) == ("chunking", "default_template")
            else (default if default is not None else None)
        ),
    )


def _chunk_rows(db: MediaDatabase, media_id: int) -> list[str]:
    cursor = db.execute_query(
        "SELECT chunk_text FROM UnvectorizedMediaChunks WHERE media_id = ? "
        "AND deleted = 0 ORDER BY chunk_index",
        (media_id,),
    )
    return [row["chunk_text"] for row in cursor.fetchall()]


def _ingest(
    db: MediaDatabase,
    source: Path,
    chunk_options: Dict[str, Any] | None,
) -> tuple[int, list[str]]:
    payload = parse_local_file_for_ingest(
        source, {"chunk_options": chunk_options, "perform_analysis": False}
    )
    # overwrite=True: the same fixture is ingested twice (two templates);
    # historical duplicate-skip would return media_id=None on the second.
    media_id, _, _ = persist_parsed_media(
        payload, db, overwrite_existing=True, generate_embeddings=False
    )
    assert media_id is not None
    return media_id, _chunk_rows(db, media_id)


def _minimal_app(media_db: Any) -> Any:
    from tldw_chatbook.Library.library_ingest_jobs import (
        LibraryIngestJobRegistry,
    )
    from tldw_chatbook.app import TldwCli

    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app.media_db = media_db
    return app


def _submit_job(app: Any, source: str, detected_type: str, snapshot: Dict[str, Any]):
    return app.library_ingest_jobs.submit(
        source_path=source,
        detected_type=detected_type,
        ingest_options=snapshot,
    )


#: A fresh canvas snapshot for a plain local import (the shape
#: ``_build_ingest_options_snapshot`` emits; schema defaults seeded).
def _generic_snapshot(**overrides: Any) -> Dict[str, Dict[str, Any]]:
    generic: Dict[str, Any] = {
        "analyze": False,
        "overwrite_existing": False,
        "custom_prompt": "",
        "system_prompt": "",
        "generate_embeddings": True,
        "chunk": True,
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "encoding": "auto",
    }
    generic.update(overrides)
    return {"generic": generic}


# ---------------------------------------------------------------------------
# AC 34 + AC 37: resolve_ingest_template -- order per path, named refusal
# ---------------------------------------------------------------------------


class TestResolveIngestTemplate:
    def test_resolution_order_picker_then_config_then_none(
        self, template_db, monkeypatch
    ):
        from tldw_chatbook.Chunking.template_runtime import resolve_ingest_template

        # picker choice wins over the config default
        _patch_config_default(monkeypatch, "big-words")
        resolved = resolve_ingest_template(template_db, "tiny-words")
        assert resolved is not None and resolved["name"] == "tiny-words"

        # no picker choice -> the config default applies
        resolved = resolve_ingest_template(template_db, None)
        assert resolved is not None and resolved["name"] == "big-words"

        # neither -> None (plain options; today's behavior)
        monkeypatch.setattr(
            "tldw_chatbook.config.get_cli_setting",
            lambda section, key=None, default=None: (
                default if default is not None else None
            ),
        )
        assert resolve_ingest_template(template_db, None) is None
        assert resolve_ingest_template(template_db, "") is None

    def test_rechunk_order_per_media_then_config(
        self, template_db, monkeypatch
    ):
        from tldw_chatbook.Chunking.template_runtime import resolve_ingest_template

        _patch_config_default(monkeypatch, "big-words")
        # stored per-media choice wins for the re-chunk path
        resolved = resolve_ingest_template(
            template_db, picker_choice="tiny-words", per_media="big-words"
        )
        assert resolved is not None and resolved["name"] == "big-words"
        # empty per-media falls to the config default
        resolved = resolve_ingest_template(
            template_db, picker_choice=None, per_media=None
        )
        assert resolved is not None and resolved["name"] == "big-words"

    def test_unresolvable_picker_choice_raises_named_error(self, template_db):
        from tldw_chatbook.Chunking.template_runtime import (
            TemplateResolutionError,
            resolve_ingest_template,
        )

        with pytest.raises(TemplateResolutionError) as excinfo:
            resolve_ingest_template(template_db, "ghost-template")
        assert "ghost-template" in str(excinfo.value)

    def test_unresolvable_config_default_never_falls_through(
        self, template_db, monkeypatch
    ):
        from tldw_chatbook.Chunking.template_runtime import (
            TemplateResolutionError,
            resolve_ingest_template,
        )

        _patch_config_default(monkeypatch, "deleted-long-ago")
        with pytest.raises(TemplateResolutionError):
            resolve_ingest_template(template_db, None)

    def test_unresolvable_per_media_does_not_fall_through_to_config(
        self, template_db, monkeypatch
    ):
        from tldw_chatbook.Chunking.template_runtime import (
            TemplateResolutionError,
            resolve_ingest_template,
        )

        _patch_config_default(monkeypatch, "big-words")
        with pytest.raises(TemplateResolutionError):
            resolve_ingest_template(
                template_db, picker_choice=None, per_media="renamed-away"
            )

    def test_soft_deleted_choice_raises_named_error(self, media_db):
        """The deleted-filter re-assigned from Task 5: soft-deleted rows
        must NOT resolve at runtime."""
        from tldw_chatbook.Chunking.template_runtime import (
            TemplateResolutionError,
            resolve_ingest_template,
            resolve_template,
        )

        body = {k: v for k, v in TEMPLATE_TINY.items() if k != "name"}
        _seed_template_row_direct(
            media_db, "gone-soft", {**body, "name": "gone-soft"}, deleted=True
        )
        assert resolve_template(media_db, "gone-soft") is None
        with pytest.raises(TemplateResolutionError):
            resolve_ingest_template(media_db, "gone-soft")

    def test_stored_invalid_choice_refused_with_named_error(self, media_db):
        """AC-24b ingest half: a stored-invalid body is refused with the
        NAMED InvalidTemplateError -- never an unnamed engine error."""
        from tldw_chatbook.Chunking.chunking_interop_library import (
            InvalidTemplateError,
        )
        from tldw_chatbook.Chunking.template_runtime import resolve_ingest_template

        _seed_template_row_direct(
            media_db, "broken-body", {"name": "broken-body", "no_chunking": True}
        )
        with pytest.raises(InvalidTemplateError):
            resolve_ingest_template(media_db, "broken-body")

    def test_none_db_with_no_choice_is_plain_options(self):
        from tldw_chatbook.Chunking.template_runtime import resolve_ingest_template

        assert resolve_ingest_template(None, None) is None


# ---------------------------------------------------------------------------
# The chunking_service seam: template kwarg + the inert-key fix (§9.2)
# ---------------------------------------------------------------------------


class TestChunkingServiceTemplateKwarg:
    def test_template_kwarg_governs_module_wrapper(self):
        from tldw_chatbook.RAG_Search.chunking_service import (
            improved_chunking_process,
        )

        # No explicit max_size: the template's chunk-stage option governs.
        # (An EXPLICIT max_size beats the template -- that is the Chunker's
        # documented merge order, pinned by the next test.)
        chunks = improved_chunking_process(
            _FIXTURE_TEXT,
            {"method": "words", "overlap": 0},
            template=TEMPLATE_TINY,
        )
        assert [c["text"] for c in chunks] == [
            " ".join(_WORDS[i : i + 3]) for i in range(0, 24, 3)
        ]

    def test_explicit_max_size_beats_template_kwarg(self):
        from tldw_chatbook.RAG_Search.chunking_service import (
            improved_chunking_process,
        )

        chunks = improved_chunking_process(
            _FIXTURE_TEXT,
            {"method": "words", "max_size": 6, "overlap": 0},
            template=TEMPLATE_TINY,
        )
        assert [c["text"] for c in chunks] == [
            " ".join(_WORDS[i : i + 6]) for i in range(0, 24, 6)
        ]

    def test_template_key_inside_options_is_popped_and_forwarded(self):
        """The inert-key trap: a ``template`` key inside the options dict
        used to be ignored by the Chunker (it gates on the keyword)."""
        from tldw_chatbook.RAG_Search.chunking_service import (
            improved_chunking_process,
        )

        chunks = improved_chunking_process(
            _FIXTURE_TEXT,
            {
                "method": "words",
                "overlap": 0,
                "template": TEMPLATE_TINY,  # carries max_size 3
            },
        )
        assert len(chunks) == 8
        assert chunks[0]["text"] == "w01 w02 w03"

    def test_explicit_option_beats_template_key(self):
        from tldw_chatbook.RAG_Search.chunking_service import (
            improved_chunking_process,
        )

        chunks = improved_chunking_process(
            _FIXTURE_TEXT,
            {"method": "words", "max_size": 2, "overlap": 0, "template": TEMPLATE_BIG},
        )
        assert chunks[0]["text"] == "w01 w02"

    def test_chunke_service_chunk_text_accepts_template_kwarg(self):
        """The audio/video chunking seam (``ChunkingService.chunk_text``).

        The audio path calls this with the MATERIALIZED scalars (template
        values re-projected by the parse seam) plus the template dict, so
        both spellings agree. The explicit-scalar merge order is the
        Chunker's own (explicit beats template) and is pinned first.
        """
        from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

        service = ChunkingService()
        # Explicit scalar (50-word cap) beats the template's 3:
        chunks = service.chunk_text(
            _FIXTURE_TEXT,
            chunk_size=50,
            chunk_overlap=0,
            method="words",
            template=TEMPLATE_TINY,
        )
        assert len(chunks) == 1  # 24 words fit one 50-word chunk
        # The materialized audio call shape (scars from the template):
        chunks = service.chunk_text(
            _FIXTURE_TEXT,
            chunk_size=3,
            chunk_overlap=0,
            method="words",
            template=TEMPLATE_TINY,
        )
        assert [c["text"] for c in chunks] == [
            " ".join(_WORDS[i : i + 3]) for i in range(0, 24, 3)
        ]


# ---------------------------------------------------------------------------
# AC 35: precedence in _ingest_job_options (the inert-picker trap)
# ---------------------------------------------------------------------------


class TestIngestJobOptionsPrecedence:
    def test_template_beats_builder_defaults(self, template_db, tmp_path):
        app = _minimal_app(template_db)
        job = _submit_job(
            app,
            str(tmp_path / "doc.txt"),
            "plaintext",
            _generic_snapshot(chunk_template="tiny-words"),
        )
        options = app._ingest_job_options(job)
        chunk_options = options["chunk_options"]
        # The resolved template travels with the job...
        assert chunk_options["template"]["name"] == "tiny-words"
        # ...and the builder's DEFAULTS are stripped: the snapshot seeded
        # schema defaults (size 1000 / overlap 100), which must NOT ride
        # along as explicit options that would beat the template.
        assert "size" not in chunk_options
        assert "max_size" not in chunk_options
        assert "overlap" not in chunk_options
        assert "method" not in chunk_options

    def test_user_changed_values_beat_template(self, template_db, tmp_path):
        app = _minimal_app(template_db)
        job = _submit_job(
            app,
            str(tmp_path / "doc.txt"),
            "plaintext",
            _generic_snapshot(chunk_template="tiny-words", chunk_size=4, chunk_overlap=1),
        )
        chunk_options = app._ingest_job_options(job)["chunk_options"]
        assert chunk_options["template"]["name"] == "tiny-words"
        assert chunk_options["size"] == 4
        assert chunk_options["max_size"] == 4
        assert chunk_options["overlap"] == 1

    def test_pdf_method_injection_skipped_under_template(self, template_db, tmp_path):
        app = _minimal_app(template_db)
        job = _submit_job(
            app,
            str(tmp_path / "doc.pdf"),
            "pdf",
            _generic_snapshot(chunk_template="tiny-words"),
        )
        chunk_options = app._ingest_job_options(job)["chunk_options"]
        assert "method" not in chunk_options  # "words" is a builder default
        # and without a template the historical injection is byte-identical
        plain_job = _submit_job(
            app, str(tmp_path / "doc.pdf"), "pdf", _generic_snapshot()
        )
        plain = app._ingest_job_options(plain_job)["chunk_options"]
        assert plain == {"size": 1000, "max_size": 1000, "overlap": 100, "method": "words"}

    def test_audio_video_method_injection_skipped_under_template(
        self, template_db, tmp_path
    ):
        app = _minimal_app(template_db)
        job = _submit_job(
            app,
            str(tmp_path / "memo.wav"),
            "audio",
            _generic_snapshot(chunk_template="tiny-words"),
        )
        chunk_options = app._ingest_job_options(job)["chunk_options"]
        assert "method" not in chunk_options

    def test_ebook_user_changed_method_beats_template_default(
        self, template_db, tmp_path
    ):
        app = _minimal_app(template_db)
        snapshot = _generic_snapshot(chunk_template="tiny-words")
        snapshot["ebook"] = {"chunk_method": "paragraphs"}  # user-changed
        job = _submit_job(app, str(tmp_path / "book.epub"), "ebook", snapshot)
        chunk_options = app._ingest_job_options(job)["chunk_options"]
        assert chunk_options["method"] == "paragraphs"

        # the schema default ("chapters") is a builder default -> stripped
        snapshot_default = _generic_snapshot(chunk_template="tiny-words")
        snapshot_default["ebook"] = {"chunk_method": "chapters"}
        job = _submit_job(
            app, str(tmp_path / "book2.epub"), "ebook", snapshot_default
        )
        chunk_options = app._ingest_job_options(job)["chunk_options"]
        assert "method" not in chunk_options

    def test_none_default_options_byte_identical_to_today(self, tmp_path):
        """The 'None' default emits exactly today's chunk_options."""
        app = _minimal_app(None)
        job = _submit_job(
            app, str(tmp_path / "doc.txt"), "plaintext", _generic_snapshot()
        )
        chunk_options = app._ingest_job_options(job)["chunk_options"]
        assert chunk_options == {
            "size": 1000,
            "max_size": 1000,
            "overlap": 100,
        }

    def test_unresolvable_choice_fails_item_with_named_error(
        self, media_db, tmp_path
    ):
        from tldw_chatbook.Chunking.template_runtime import TemplateResolutionError

        app = _minimal_app(media_db)
        job = _submit_job(
            app,
            str(tmp_path / "doc.txt"),
            "plaintext",
            _generic_snapshot(chunk_template="ghost-template"),
        )
        with pytest.raises(TemplateResolutionError):
            app._ingest_job_options(job)

    def test_invalid_template_fails_item_with_named_error(
        self, media_db, tmp_path
    ):
        from tldw_chatbook.Chunking.chunking_interop_library import (
            InvalidTemplateError,
        )

        _seed_template_row_direct(
            media_db, "broken-body", {"name": "broken-body", "no_chunking": True}
        )
        app = _minimal_app(media_db)
        job = _submit_job(
            app,
            str(tmp_path / "doc.txt"),
            "plaintext",
            _generic_snapshot(chunk_template="broken-body"),
        )
        with pytest.raises(InvalidTemplateError):
            app._ingest_job_options(job)

    def test_config_default_resolves_without_picker_choice(
        self, template_db, tmp_path, monkeypatch
    ):
        _patch_config_default(monkeypatch, "big-words")
        app = _minimal_app(template_db)
        job = _submit_job(
            app, str(tmp_path / "doc.txt"), "plaintext", _generic_snapshot()
        )
        chunk_options = app._ingest_job_options(job)["chunk_options"]
        assert chunk_options["template"]["name"] == "big-words"


# ---------------------------------------------------------------------------
# AC 36: governance per media-type family -- real parse -> persist -> rows
# ---------------------------------------------------------------------------


class TestGovernancePlainText:
    def _text_source(self, tmp_path: Path) -> Path:
        source = tmp_path / "fixture.txt"
        source.write_text(_FIXTURE_TEXT, encoding="utf-8")
        return source

    def test_two_templates_different_persisted_rows(self, template_db, tmp_path):
        source = self._text_source(tmp_path)
        _, tiny_rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_TINY)}
        )
        _, big_rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_BIG)}
        )
        assert tiny_rows == [
            " ".join(_WORDS[i : i + 3]) for i in range(0, 24, 3)
        ]
        assert big_rows == [
            " ".join(_WORDS[i : i + 12]) for i in range(0, 24, 12)
        ]

    def test_builder_shaped_options_honor_template_despite_processor_defaults(
        self, template_db, tmp_path
    ):
        """The inert-picker trap end to end: the builder emits ONLY the
        template (defaults stripped); the parse must still produce
        template-sized chunks (3 words), not the tail's 500-word default."""
        source = self._text_source(tmp_path)
        _, rows = _ingest(template_db, source, {"template": dict(TEMPLATE_TINY)})
        assert len(rows) == 8

    def test_user_changed_size_beats_template_end_to_end(
        self, template_db, tmp_path
    ):
        source = self._text_source(tmp_path)
        _, rows = _ingest(
            template_db,
            source,
            {"template": dict(TEMPLATE_TINY), "size": 6, "max_size": 6, "overlap": 0},
        )
        assert rows == [" ".join(_WORDS[i : i + 6]) for i in range(0, 24, 6)]

    def test_none_default_byte_identical_to_today(self, media_db, tmp_path):
        """Plain options (no template) keep today's output exactly."""
        source = self._text_source(tmp_path)
        _, rows = _ingest(
            media_db,
            source,
            {"method": "words", "size": 5, "max_size": 5, "overlap": 0},
        )
        assert rows == [
            " ".join(_WORDS[i : i + 5]) for i in range(0, 24, 5)
        ]


class TestGovernancePdfFamily:
    """pdf -- the whole-dict pass-through family (pdf/document/ebook);
    extraction stubbed (pymupdf4llm absent), chunking REAL."""

    @pytest.fixture(autouse=True)
    def _stub_pdf_extraction(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        from tldw_chatbook.Local_Ingestion import PDF_Processing_Lib

        monkeypatch.setattr(
            PDF_Processing_Lib, "pymupdf4llm_parse_pdf", lambda _path: _FIXTURE_TEXT
        )
        # process_pdf's metadata step unconditionally calls pymupdf.open
        # (and its error handlers reference the exception classes), so the
        # absent library is replaced with a minimal stand-in -- extraction
        # of TEXT is the stubbed seam; chunking below it is fully real.
        class _Doc:
            metadata = {"title": "fixture", "author": "tester"}
            page_count = 1

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _FakePyMuPdf:
            class FileDataError(Exception):
                pass

            class EmptyFileError(Exception):
                pass

            @staticmethod
            def open(filename=None, **_kwargs):
                return _Doc()

        monkeypatch.setattr(PDF_Processing_Lib, "pymupdf", _FakePyMuPdf)

    def _pdf_source(self, tmp_path: Path) -> Path:
        source = tmp_path / "fixture.pdf"
        source.write_bytes(b"%PDF-1.4 fixture bytes; extraction is stubbed")
        return source

    def test_two_templates_different_persisted_rows(self, template_db, tmp_path):
        source = self._pdf_source(tmp_path)
        _, tiny_rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_TINY)}
        )
        _, big_rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_BIG)}
        )
        assert tiny_rows == [
            " ".join(_WORDS[i : i + 3]) for i in range(0, 24, 3)
        ]
        assert big_rows == [
            " ".join(_WORDS[i : i + 12]) for i in range(0, 24, 12)
        ]

    def test_template_beats_processor_setdefaults(self, template_db, tmp_path):
        """process_pdf setdefaults sentences/500/100; the template's
        3-word scheme must survive them."""
        source = self._pdf_source(tmp_path)
        _, rows = _ingest(template_db, source, {"template": dict(TEMPLATE_TINY)})
        assert len(rows) == 8
        assert rows[0] == "w01 w02 w03"


class TestGovernanceEbookFamily:
    """ebook (FB2: fully real extraction via stdlib XML + real chunking)."""

    @pytest.fixture(autouse=True)
    def _ebook_available(self, monkeypatch: pytest.MonkeyPatch):
        from tldw_chatbook.Local_Ingestion import Book_Ingestion_Lib

        monkeypatch.setattr(
            Book_Ingestion_Lib, "EBOOK_PROCESSING_AVAILABLE", True
        )

    def _fb2_source(self, tmp_path: Path) -> Path:
        body = "\n".join(
            f"    <p>{word} filler {i} tail</p>" for i, word in enumerate(_WORDS)
        )
        source = tmp_path / "fixture.fb2"
        source.write_text(
            f"""<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description><title-info><book-title>T</book-title></title-info></description>
  <body><section>
{body}
  </section></body>
</FictionBook>
""",
            encoding="utf-8",
        )
        return source

    def test_two_templates_different_persisted_rows(self, template_db, tmp_path):
        source = self._fb2_source(tmp_path)
        _, tiny_rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_TINY)}
        )
        _, big_rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_BIG)}
        )
        assert tiny_rows != big_rows
        # tiny (3-word) produces strictly more rows than big (12-word)
        assert len(tiny_rows) > len(big_rows) > 0
        # every tiny row is at most 3 words
        assert all(len(row.split()) <= 3 for row in tiny_rows)
        assert any(len(row.split()) > 3 for row in big_rows)

    def test_template_beats_ebook_processor_defaults(self, template_db, tmp_path):
        """process_fb2 setdefaults sentences/500/200; the template's
        words/3 scheme must survive them."""
        source = self._fb2_source(tmp_path)
        _, rows = _ingest(template_db, source, {"template": dict(TEMPLATE_TINY)})
        assert all(len(row.split()) <= 3 for row in rows)


class TestGovernanceAudioVideoFamily:
    """audio -- real processor, real chunking; only STT stubbed. Video's
    re-projection is pinned at the branch seam (the chunk site is the
    shared ``LocalAudioProcessor._chunk_text``)."""

    @pytest.fixture(autouse=True)
    def _stub_stt(self, monkeypatch: pytest.MonkeyPatch):
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor,
        )

        monkeypatch.setattr(
            LocalAudioProcessor,
            "_transcribe_audio",
            lambda self, audio_path, **kwargs: {
                "text": _FIXTURE_TEXT,
                "segments": [],
                "transcription_model": "stub",
                "transcription_provenance": None,
            },
        )

    def _audio_source(self, tmp_path: Path) -> Path:
        source = tmp_path / "fixture.wav"
        source.write_bytes(b"fake wav bytes; STT is stubbed")
        return source

    def test_two_templates_different_persisted_rows(self, template_db, tmp_path):
        source = self._audio_source(tmp_path)
        _, tiny_rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_TINY)}
        )
        _, big_rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_BIG)}
        )
        assert tiny_rows != big_rows
        assert len(tiny_rows) > len(big_rows) > 0

    def test_audio_reprojection_carries_template_scalars(
        self, template_db, tmp_path, monkeypatch
    ):
        """The widened key-by-key projection: template chunk-stage options
        arrive at the processor as the scalar kwargs."""
        from tldw_chatbook.Local_Ingestion import local_file_ingestion as lfi
        from tldw_chatbook.Local_Ingestion.audio_processing import (
            LocalAudioProcessor,
        )

        observed: Dict[str, Any] = {}

        class RecordingProcessor(LocalAudioProcessor):
            # Real class (real chunking); record the re-projected kwargs.
            def process_audio_files(self, *args, **kwargs):
                observed.update(kwargs)
                return super().process_audio_files(*args, **kwargs)

        monkeypatch.setattr(lfi, "LocalAudioProcessor", RecordingProcessor)
        source = self._audio_source(tmp_path)
        _, rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_TINY)}
        )
        assert observed["chunk_method"] == "words"
        assert observed["max_chunk_size"] == 3
        assert observed["chunk_overlap"] == 0
        assert observed["chunk_template"]["name"] == "tiny-words"
        assert len(rows) == 8

    def test_video_branch_passes_template_explicitly(
        self, tmp_path, monkeypatch
    ):
        """The :1309-1315 re-projection: a stub stands in for the video
        processor and records the call (signature-checked)."""
        from tldw_chatbook.Local_Ingestion import local_file_ingestion as lfi
        from tldw_chatbook.Local_Ingestion.video_processing import (
            LocalVideoProcessor,
        )

        recorded: Dict[str, Any] = {}

        class StubVideoProcessor:
            def __init__(self, media_db=None, transcription_runner=None):
                pass

            def process_videos(self, **kwargs):
                recorded.update(kwargs)
                return {
                    "results": [
                        {
                            "status": "Success",
                            "content": _FIXTURE_TEXT,
                            "metadata": {},
                            "chunks": [],
                            "analysis": "",
                        }
                    ],
                    "errors": [],
                }

        monkeypatch.setattr(lfi, "LocalVideoProcessor", StubVideoProcessor)
        source = tmp_path / "fixture.mp4"
        source.write_bytes(b"fake mp4 bytes")
        payload = parse_local_file_for_ingest(
            source,
            {"chunk_options": {"template": dict(TEMPLATE_TINY)}},
        )

        assert payload["content"] == _FIXTURE_TEXT
        # The re-projection no longer drops the template...
        assert recorded["chunk_template"]["name"] == "tiny-words"
        # ...and the template's chunk-stage options ride the scalars.
        assert recorded["chunk_method"] == "words"
        assert recorded["max_chunk_size"] == 3
        assert recorded["chunk_overlap"] == 0
        # The stub must not accept kwargs the real seam would reject.
        real_sig = inspect.signature(LocalVideoProcessor.process_videos)
        assert set(recorded) <= set(real_sig.parameters) or any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in real_sig.parameters.values()
        )


class TestImageSeamDocumented:
    """§9.2 image row: the branch passes ``chunk_options=None`` to
    ``process_image`` (documented unaffected); the OCR text chunks through
    the shared text tail, which the template governs."""

    @pytest.fixture(autouse=True)
    def _stub_image(self, monkeypatch: pytest.MonkeyPatch):
        from tldw_chatbook.Local_Ingestion import local_file_ingestion as lfi

        class StubProcessImage:
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {
                    "content": _FIXTURE_TEXT,
                    "title": "fixture",
                    "author": "Unknown",
                    "keywords": [],
                    "chunks": [],
                    "error": None,
                }

        stub = StubProcessImage()
        monkeypatch.setattr(lfi, "_ensure_process_image", lambda: stub)
        self.stub = stub

    def test_image_branch_unaffected_but_tail_honors_template(
        self, template_db, tmp_path
    ):
        source = tmp_path / "fixture.png"
        source.write_bytes(b"fake png bytes; OCR is stubbed")
        _, rows = _ingest(
            template_db, source, {"template": dict(TEMPLATE_TINY)}
        )
        # the branch still passes chunk_options=None to process_image
        assert self.stub.kwargs["chunk_options"] is None
        # ...and the shared tail produced template-sized chunks
        assert rows == [" ".join(_WORDS[i : i + 3]) for i in range(0, 24, 3)]


# ---------------------------------------------------------------------------
# Seam 6: server mode never carries a template
# ---------------------------------------------------------------------------


class TestServerModeStripsTemplate:
    def test_server_ingest_kwargs_never_carry_a_template(self, tmp_path):
        from tldw_chatbook.Library.server_ingest_request import (
            build_server_ingest_kwargs,
        )

        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4")
        kwargs = build_server_ingest_kwargs(
            str(source),
            options=_generic_snapshot(chunk_template="tiny-words"),
            title="t",
        )
        assert "chunk_template" not in kwargs
        assert "template" not in kwargs
        assert "chunking_template" not in kwargs

    def test_server_group_options_never_carry_a_template(self, tmp_path):
        from tldw_chatbook.Library.server_ingest_request import (
            build_server_ingest_kwargs,
        )

        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4")
        snapshot = _generic_snapshot(chunk_template="tiny-words")
        snapshot["pdf"] = {"engine": "pymupdf", "chunk_template": "tiny-words"}
        kwargs = build_server_ingest_kwargs(str(source), options=snapshot)
        assert "chunk_template" not in kwargs
        # the group's real options still travel ("engine" is not an alias)
        assert kwargs.get("engine") == "pymupdf"

    def test_server_generic_group_source_never_carries_a_template(
        self, tmp_path
    ):
        """A generic-group source (.txt) iterates the generic dict in the
        builder's group loop -- the strip must hold there too."""
        from tldw_chatbook.Library.server_ingest_request import (
            build_server_ingest_kwargs,
        )

        source = tmp_path / "notes.txt"
        source.write_text("plain text", encoding="utf-8")
        kwargs = build_server_ingest_kwargs(
            str(source), options=_generic_snapshot(chunk_template="tiny-words")
        )
        assert "chunk_template" not in kwargs
        assert "template" not in kwargs


# ---------------------------------------------------------------------------
# AC-24a/24b: listing flag data + named apply refusal
# ---------------------------------------------------------------------------


class _RecordingChunkingService:
    """Interop stand-in carrying one valid and one stored-invalid body."""

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}

    def get_all_templates(self, include_builtin: bool = True):
        return [dict(record) for record in self.records.values()]

    def get_template_by_name(self, name: str):
        record = self.records.get(name)
        return dict(record) if record else None

    def get_template_by_id(self, template_id: int):
        for record in self.records.values():
            if record["id"] == template_id:
                return dict(record)
        return None


def _admin_with_valid_and_invalid() -> Any:
    from tldw_chatbook.RAG_Admin.local_rag_admin_service import (
        LocalRAGAdminService,
    )

    service = _RecordingChunkingService()
    service.records["valid-one"] = {
        "id": 1,
        "name": "valid-one",
        "description": "",
        "template_json": {
            "chunking": {"method": "words", "config": {"max_size": 2, "overlap": 0}}
        },
        "tags": [],
        "is_builtin": False,
        "version": 1,
    }
    service.records["invalid-one"] = {
        "id": 2,
        "name": "invalid-one",
        "description": "",
        "template_json": {"not_a_chunking_block": True},
        "tags": [],
        "is_builtin": False,
        "version": 1,
    }
    return LocalRAGAdminService(None, chunking_service=service)


class TestA24TemplateHealthSurfaces:
    def test_listing_carries_validity_flag_data(self):
        service = _admin_with_valid_and_invalid()
        listed = {t["name"]: t for t in service.list_templates()}
        assert listed["valid-one"]["template_valid"] is True
        invalid = listed["invalid-one"]
        assert invalid["template_valid"] is False
        # the flag's evidence travels with the record (Task 12 renders it)
        assert invalid["template_validation_errors"]
        assert not listed["valid-one"].get("template_validation_errors")

    def test_apply_refuses_stored_invalid_with_named_error(self):
        from tldw_chatbook.Chunking.chunking_interop_library import (
            InvalidTemplateError,
        )

        service = _admin_with_valid_and_invalid()
        with pytest.raises(InvalidTemplateError) as excinfo:
            service.apply_template("invalid-one", text="alpha beta gamma")
        assert "invalid-one" in str(excinfo.value)

    def test_apply_still_applies_valid_templates(self):
        service = _admin_with_valid_and_invalid()
        result = service.apply_template(
            "valid-one", text="alpha beta gamma delta"
        )
        assert result["chunks"] == ["alpha beta", "gamma delta"]


# ---------------------------------------------------------------------------
# resolve_template deleted-filter (Task 5 SQL pin re-assignment)
# ---------------------------------------------------------------------------


class TestResolveTemplateDeletedFilter:
    def _db(self) -> Any:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE ChunkingTemplates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                template_json TEXT NOT NULL,
                tags TEXT,
                is_builtin BOOLEAN DEFAULT 0,
                version INTEGER DEFAULT 1,
                deleted BOOLEAN DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        body = json.dumps(
            {k: v for k, v in TEMPLATE_TINY.items() if k != "name"}
        )
        conn.execute(
            "INSERT INTO ChunkingTemplates (uuid, name, template_json) "
            "VALUES ('u1', 'live', ?)",
            (body,),
        )
        conn.execute(
            "INSERT INTO ChunkingTemplates (uuid, name, template_json, deleted) "
            "VALUES ('u2', 'dead', ?, 1)",
            (body,),
        )

        class _Db:
            def get_connection(self):
                return conn

        return _Db()

    def test_deleted_rows_do_not_resolve(self):
        from tldw_chatbook.Chunking.template_runtime import resolve_template

        db = self._db()
        assert resolve_template(db, "live") is not None
        assert resolve_template(db, "dead") is None

    def test_sql_filters_deleted(self):
        from tldw_chatbook.Chunking.template_runtime import resolve_template

        db = self._db()
        statements: list[str] = []
        db.get_connection().set_trace_callback(statements.append)
        try:
            resolve_template(db, "live")
        finally:
            db.get_connection().set_trace_callback(None)
        sql = " ".join(statements)
        assert "deleted = 0" in sql, sql
        # the SELECT list stays the v6/v7-stable pair
        assert "SELECT name, template_json FROM ChunkingTemplates" in " ".join(
            sql.split()
        )

class TestMaterializeTemplateChunkOptions:
    def test_materializes_defaults_and_size_alias(self):
        from tldw_chatbook.Chunking.template_runtime import (
            materialize_template_chunk_options,
        )

        options: Dict[str, Any] = {"overlap": 7}  # user-changed overlap wins
        materialize_template_chunk_options(options, TEMPLATE_TINY)
        assert options["method"] == "words"
        assert options["max_size"] == 3
        assert options["size"] == 3  # audio re-projection spelling
        assert options["overlap"] == 7  # setdefault: user value untouched

    def test_noop_without_chunking_block(self):
        from tldw_chatbook.Chunking.template_runtime import (
            materialize_template_chunk_options,
        )

        options: Dict[str, Any] = {}
        materialize_template_chunk_options(options, {"name": "empty"})
        assert options == {}
