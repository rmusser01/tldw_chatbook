"""Parse/persist split + spawn-safe pool worker entry point (F3 Task 2).

Covers:
  - ``parse_local_file_for_ingest`` (pre-DB half): returns a picklable
    payload, touches no database.
  - ``persist_parsed_media`` (post-parse half): the single
    ``add_media_with_keywords`` write.
  - ``ingest_local_file`` (compose): unchanged signature/return
    shape/error behavior, now built from the two halves above.
  - ``ingest_parse_worker.run_parse_job``/``classify_parse_failure``: the
    pool entry point, structured results, permanent-vs-retryable
    classification.
  - Import-weight guard: importing ``ingest_parse_worker`` (via its real
    dotted module path, matching how a spawned pool worker resolves it)
    must not pull in ``local_file_ingestion``.
  - One real ``multiprocessing.get_context("spawn").Pool(1)`` integration
    test exercising ``run_parse_job`` through an actual subprocess.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Local_Ingestion import Document_Processing_Lib as document_processing
from tldw_chatbook.Local_Ingestion.ingest_parse_worker import (
    classify_parse_failure,
    run_parse_job,
)
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    FileIngestionError,
    ingest_local_file,
    parse_local_file_for_ingest,
    persist_parsed_media,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_process_document_auto_falls_back_when_docling_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def broken_docling(*args, **kwargs):
        calls.append("docling")
        raise ImportError("broken transitive Docling dependency")

    def native_docx(*args, **kwargs):
        calls.append("native")
        return {
            "content": "Native content",
            "title": "sample.docx",
            "author": "tester",
            "metadata": {"processing_method": "python-docx"},
            "extraction_successful": True,
        }

    monkeypatch.setattr(document_processing, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(document_processing, "PYTHON_DOCX_AVAILABLE", True)
    monkeypatch.setattr(document_processing, "process_with_docling", broken_docling)
    monkeypatch.setattr(document_processing, "process_docx", native_docx)

    result = document_processing.process_document("sample.docx", processing_method="auto")

    assert result["extraction_successful"] is True
    assert result["content"] == "Native content"
    assert calls == ["docling", "native"]


def test_process_document_explicit_docling_preserves_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_calls: list[str] = []

    def broken_docling(*args, **kwargs):
        raise ImportError("broken transitive Docling dependency")

    def native_docx(*args, **kwargs):
        native_calls.append("native")
        return {"extraction_successful": True}

    monkeypatch.setattr(document_processing, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(document_processing, "PYTHON_DOCX_AVAILABLE", True)
    monkeypatch.setattr(document_processing, "process_with_docling", broken_docling)
    monkeypatch.setattr(document_processing, "process_docx", native_docx)

    result = document_processing.process_document("sample.docx", processing_method="docling")

    assert result["extraction_successful"] is False
    assert result["metadata"]["error"] == "broken transitive Docling dependency"
    assert native_calls == []


def test_process_document_auto_does_not_retry_broken_docling_without_native_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docling_calls: list[str] = []

    def broken_docling(*args, **kwargs):
        docling_calls.append("docling")
        raise ImportError("broken transitive Docling dependency")

    monkeypatch.setattr(document_processing, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(document_processing, "PYTHON_DOCX_AVAILABLE", False)
    monkeypatch.setattr(document_processing, "process_with_docling", broken_docling)

    result = document_processing.process_document("sample.docx", processing_method="auto")

    assert result["extraction_successful"] is False
    assert result["metadata"]["error"] == "No parser available for .docx"
    assert docling_calls == ["docling"]


# --- parse_local_file_for_ingest -------------------------------------------


def test_parse_returns_payload_without_touching_a_database(tmp_path: Path) -> None:
    """Parsing a txt file needs no media_db at all -- it isn't even a parameter."""
    source = tmp_path / "note.txt"
    source.write_text("Tides are driven by the moon's gravity.", encoding="utf-8")

    payload = parse_local_file_for_ingest(
        str(source),
        {"title": "Moon note", "author": "tester", "keywords": ["tides"]},
    )

    assert payload["title"] == "Moon note"
    assert payload["author"] == "tester"
    assert payload["media_type"] == "plaintext"
    assert payload["file_type"] == "plaintext"
    assert "moon's gravity" in payload["content"]
    assert payload["keywords"] == ["tides"]
    assert payload["analysis_content"] == ""
    assert payload["chunks"] is None
    assert payload["chunk_options"] is None
    assert payload["url"] == f"file://{source.absolute()}"
    assert payload["file_path"] == str(source)
    # Every value must be plain, picklable data -- this dict crosses a
    # process boundary as the pool's apply_async return value.
    import pickle

    pickle.dumps(payload)


def test_parse_pdf_success_with_present_but_none_error_key_does_not_raise(
    tmp_path: Path, monkeypatch
) -> None:
    """task-168(c): ``process_pdf``'s result dict always has an ``'error'``
    key (initialized to ``None`` and only ever overwritten on a real
    failure -- see ``PDF_Processing_Lib.process_pdf``'s initial ``result``
    dict). ``'error' in result`` is therefore ALWAYS ``True``, even on a
    clean success, so every real PDF parse incorrectly raised
    ``FileIngestionError``. The check must key off truthiness
    (``result.get('error')``) instead of key presence."""
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4 stub bytes, never actually parsed (process_pdf is mocked).")

    stub_result = {
        "status": "Success",
        "content": "Extracted PDF text.",
        "title": "Doc title",
        "author": "Some Author",
        "keywords": [],
        "chunks": [],
        "analysis": "",
        "metadata": {},
        "error": None,  # present, falsy -- the real process_pdf shape on success
    }
    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.local_file_ingestion.process_pdf",
        lambda **kwargs: stub_result,
    )

    payload = parse_local_file_for_ingest(str(source), {"perform_analysis": False})

    assert payload["content"] == "Extracted PDF text."
    assert payload["title"] == "Doc title"


def test_parse_missing_file_raises_filenotfounderror(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.txt"
    with pytest.raises(FileNotFoundError):
        parse_local_file_for_ingest(str(missing), {})


def test_parse_unsupported_extension_raises_with_prefix(tmp_path: Path) -> None:
    bad = tmp_path / "note.xyz"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(FileIngestionError, match="Unsupported file type"):
        parse_local_file_for_ingest(str(bad), {})


# --- persist_parsed_media ----------------------------------------------------


def test_persist_writes_payload_and_returns_media_id(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("Persisted content.", encoding="utf-8")
    payload = parse_local_file_for_ingest(str(source), {"title": "Persist me"})

    db = MediaDatabase(":memory:", client_id="test-parse-worker")
    media_id, media_uuid, message = persist_parsed_media(payload, db)

    assert isinstance(media_id, int)
    assert isinstance(media_uuid, str) and media_uuid
    assert isinstance(message, str)
    row = db.get_media_by_id(media_id)
    assert row is not None
    assert row["title"] == "Persist me"
    assert row["content"] == "Persisted content."
    assert row["type"] == "plaintext"


def test_persist_url_payload_writes_article_row_no_filesystem() -> None:
    """A URL-source payload (media_type=article, canonical url, URL string as
    file_path) persists to a media row without any filesystem access -- the
    payload's file_path is a URL that is not a real file, and persist never
    stats/opens it (it only forwards url/content/etc. to the DB)."""
    payload = {
        "file_type": "article",
        "media_type": "article",
        "title": "Kept article",
        "content": "Extracted article body.",
        "keywords": [],
        "url": "https://example.com/post",
        "analysis_content": "",
        "author": "Unknown",
        "chunks": None,
        "chunk_options": None,
        "file_path": "https://example.com/post",  # a URL, NOT a real file -> never accessed
    }

    db = MediaDatabase(":memory:", client_id="test-url-persist")
    media_id, media_uuid, message = persist_parsed_media(payload, db)

    assert isinstance(media_id, int)
    assert isinstance(media_uuid, str) and media_uuid
    row = db.get_media_by_id(media_id)
    assert row is not None
    assert row["url"] == "https://example.com/post"
    assert row["type"] == "article"


def test_persist_db_failure_is_wrapped_as_file_ingestion_error(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("content", encoding="utf-8")
    payload = parse_local_file_for_ingest(str(source), {})

    class _ExplodingDB:
        def add_media_with_keywords(self, **kwargs):
            raise RuntimeError("disk full")

    with pytest.raises(FileIngestionError, match="Failed to ingest plaintext file"):
        persist_parsed_media(payload, _ExplodingDB())


# --- ingest_local_file (compose) --------------------------------------------


def test_ingest_local_file_compose_end_to_end_unchanged(tmp_path: Path) -> None:
    """The historical single-function contract: same return shape, same DB effect."""
    source = tmp_path / "smoke-note.txt"
    source.write_text("Tides are driven by the moon's gravity.", encoding="utf-8")
    db = MediaDatabase(tmp_path / "smoke_media.db", client_id="parse-worker-smoke")

    result = ingest_local_file(
        file_path=source,
        media_db=db,
        title="Smoke note",
        author="tester",
        keywords=["smoke"],
        perform_analysis=False,
        chunk_options=None,
    )

    assert isinstance(result["media_id"], int)
    assert result["title"] == "Smoke note"
    assert result["author"] == "tester"
    assert result["file_type"] == "plaintext"
    assert result["file_path"] == str(source)
    assert result["content_length"] == len("Tides are driven by the moon's gravity.")
    assert result["chunks_created"] == 0
    assert result["keywords"] == ["smoke"]
    assert result["analysis"] == ""

    row = db.get_media_by_id(result["media_id"])
    assert row is not None
    assert row["title"] == "Smoke note"
    assert "moon's gravity" in row["content"]


def test_ingest_local_file_missing_file_raises(tmp_path: Path) -> None:
    db = MediaDatabase(tmp_path / "media.db", client_id="parse-worker-smoke")
    missing = tmp_path / "does-not-exist.txt"

    with pytest.raises((FileIngestionError, FileNotFoundError)):
        ingest_local_file(file_path=missing, media_db=db, perform_analysis=False)


# --- audio media_id regression lock (F3 media_db=None fix) -------------------


class _StubAudioProcessor:
    """Shape-accurate stand-in for ``LocalAudioProcessor``.

    Returns a single-result dict shaped exactly like
    ``LocalAudioProcessor.process_audio_files``/``_process_single_audio``
    build it, and -- critically -- mirrors the real class's internal
    ``_store_in_database`` step with the same ``if self.media_db and
    result["content"]`` gate and the same degraded write parameters
    (``url=result["input_ref"]``, a BARE path; title from the processor's
    own metadata; no keywords). That gate is what turns this stub into a
    regression tripwire: if the parse stage ever hands the processor a
    real ``media_db`` again, the degraded row gets written first and the
    pipeline's own ``add_media_with_keywords`` call collapses into the
    content-hash-dedup no-op -- exactly the old bug.
    """

    last_media_db: object = "UNSET"  # recorded at construction for assertion

    def __init__(self, media_db=None):
        type(self).last_media_db = media_db
        self.media_db = media_db

    def process_audio_files(self, *, inputs, custom_title=None, author=None, **kwargs):
        import time as _time

        input_item = inputs[0]
        content = "Transcribed words from the stub."
        result = {
            "status": "Success",
            "input_ref": input_item,  # bare path for local files, like the real class
            "processing_source": input_item,
            "media_type": "audio",
            "metadata": {"title": custom_title, "author": author},
            "content": content,
            "segments": [{"start": 0.0, "end": 1.0, "text": content}],
            "chunks": [],
            "analysis": "",
            "analysis_details": {},
            "error": None,
            "warnings": [],
        }
        # Mirror LocalAudioProcessor._store_in_database and its gate.
        if self.media_db and result["content"]:
            media_id, _, _ = self.media_db.add_media_with_keywords(
                url=result.get("input_ref", ""),
                title=result["metadata"].get("title", "Untitled"),
                media_type=result.get("media_type", "audio"),
                content=result.get("content", ""),
                author=result["metadata"].get("author", "Unknown"),
                ingestion_date=_time.strftime("%Y-%m-%d %H:%M:%S"),
                analysis_content=result.get("analysis"),
            )
            result["db_id"] = media_id
            result["db_message"] = "Stored successfully"
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [result],
        }


def test_ingest_local_file_audio_returns_real_media_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression lock for the F3 ``media_db=None`` fix (audio/video).

    Pre-split, ``ingest_local_file`` constructed
    ``LocalAudioProcessor(media_db)`` with the REAL database, whose
    internal ``_store_in_database`` wrote one degraded row first
    (bare-path URL, processor-metadata title, no keywords); the pipeline's
    own ``add_media_with_keywords`` call (``url="file://..."``) then hit
    the content-hash dedup fallback, matched that degraded row, took the
    "already exists, overwrite not enabled" branch, and returned
    ``(None, None, ...)`` -- so every real audio/video local ingest
    returned ``media_id=None``, and even ``app.py``'s
    ``get_media_by_url("file://...")`` recovery missed (the surviving
    row's URL was the bare path). Verified RED: temporarily rewiring the
    real ``media_db`` into the parse-stage processor construction makes
    this test fail with ``media_id=None`` (see f3-task-2-report.md).
    """
    from tldw_chatbook.Local_Ingestion import local_file_ingestion as lfi

    source = tmp_path / "voice-memo.mp3"
    source.write_bytes(b"\x00fake-mp3-bytes")

    _StubAudioProcessor.last_media_db = "UNSET"
    monkeypatch.setattr(lfi, "LocalAudioProcessor", _StubAudioProcessor)
    db = MediaDatabase(":memory:", client_id="audio-media-id-lock")

    result = ingest_local_file(
        file_path=source,
        media_db=db,
        title="Audio note",
        author="tester",
        keywords=["voices"],
        perform_analysis=False,
    )

    # The parse stage must never hand the processor a database.
    assert _StubAudioProcessor.last_media_db is None

    # The old bug's signature was media_id=None; it must be a real row id.
    media_id = result["media_id"]
    assert isinstance(media_id, int)

    row = db.get_media_by_id(media_id)
    assert row is not None
    assert row["title"] == "Audio note"
    assert row["author"] == "tester"
    assert row["type"] == "audio"
    assert row["url"] == f"file://{source.absolute()}"

    from tldw_chatbook.DB.Client_Media_DB_v2 import fetch_keywords_for_media

    assert fetch_keywords_for_media(db, media_id) == ["voices"]

    # app.py's re-ingest recovery path must also resolve now.
    assert db.get_media_by_url(f"file://{source.absolute()}") is not None


# --- classify_parse_failure ---------------------------------------------------


def test_classify_parse_failure_missing_file_is_permanent() -> None:
    assert classify_parse_failure(FileNotFoundError("File not found: x.txt")) is True


def test_classify_parse_failure_unsupported_type_is_permanent() -> None:
    assert classify_parse_failure(FileIngestionError("Unsupported file type: .xyz")) is True


def test_classify_parse_failure_generic_error_is_retryable() -> None:
    assert classify_parse_failure(RuntimeError("transient DB hiccup")) is False
    assert classify_parse_failure(FileIngestionError("Failed to ingest pdf file: boom")) is False


# --- run_parse_job (pool entry point) ----------------------------------------


def test_run_parse_job_ok_for_txt(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("hello worker", encoding="utf-8")

    result = run_parse_job(str(source), {"title": "Worker note"})

    assert set(result.keys()) == {"ok", "payload"}
    assert result["ok"] is True
    assert result["payload"]["title"] == "Worker note"
    assert result["payload"]["content"] == "hello worker"
    import pickle

    pickle.dumps(result)


def test_run_parse_job_missing_file_is_permanent(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.txt"

    result = run_parse_job(str(missing), {})

    assert result["ok"] is False
    assert result["permanent"] is True
    assert "not found" in result["error"].lower()


def test_run_parse_job_unsupported_extension_is_permanent(tmp_path: Path) -> None:
    bad = tmp_path / "note.xyz"
    bad.write_text("hello", encoding="utf-8")

    result = run_parse_job(str(bad), {})

    assert result["ok"] is False
    assert result["permanent"] is True
    assert result["error"].startswith("Unsupported file type")


def test_run_parse_job_corrupt_pdf_is_retryable_not_permanent(tmp_path: Path) -> None:
    pytest.importorskip("pymupdf")
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a real pdf, just garbage bytes" * 5)

    result = run_parse_job(str(corrupt), {"perform_analysis": False})

    assert result["ok"] is False
    assert result["permanent"] is False
    assert not result["error"].startswith("Unsupported file type")


def test_run_parse_job_never_raises_across_the_call(tmp_path: Path) -> None:
    """Structured-result contract: even a totally malformed options dict comes
    back as {"ok": False, ...}, never an exception escaping the call."""
    source = tmp_path / "note.txt"
    source.write_text("hello", encoding="utf-8")

    # `chunk_options` must be a dict/None -- feed it something that blows up
    # `chunk_options.get(...)` inside parsing (an AttributeError raised
    # outside parse's own try/except, exercising run_parse_job's own
    # catch-all rather than parse's FileIngestionError wrapping).
    result = run_parse_job(str(source), {"chunk_options": "not-a-dict"})

    assert result["ok"] is False
    assert isinstance(result["error"], str) and result["error"]
    assert isinstance(result["permanent"], bool)


# --- Import-weight guard ------------------------------------------------------


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ingest_parse_worker_import_excludes_local_file_ingestion(tmp_path: Path) -> None:
    """Resolving ingest_parse_worker by its real dotted path (exactly how a
    spawned pool worker unpickles ``run_parse_job``) must not drag
    ``local_file_ingestion`` (or its heavy transitive parse-chain deps)
    into ``sys.modules`` -- that import is deferred into ``run_parse_job``'s
    function body specifically so a freshly spawned worker process doesn't
    pay for it just to register the pool's target function.
    """
    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys

        import tldw_chatbook.Local_Ingestion.ingest_parse_worker  # noqa: F401

        guards = ("local_file_ingestion", "audio_processing", "video_processing",
                  "transcription_service", "torch", "docling", "nltk")
        loaded = sorted({
            m for m in sys.modules
            if any(m == f"tldw_chatbook.Local_Ingestion.{g}" or m.split(".")[0] == g for g in guards)
        })
        print(json.dumps({"loaded": loaded}))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["loaded"] == [], (
        f"importing ingest_parse_worker pulled in unexpected modules: {payload['loaded']}"
    )


# --- Real spawn-Pool integration ----------------------------------------------


@pytest.mark.integration
def test_run_parse_job_through_real_spawn_pool(tmp_path: Path) -> None:
    """One real subprocess round-trip: apply_async(run_parse_job) through an
    actual spawn-context Pool worker, matching production's pool shape.
    """
    source = tmp_path / "pool-note.txt"
    source.write_text("Parsed inside a real spawned worker process.", encoding="utf-8")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        async_result = pool.apply_async(run_parse_job, (str(source), {"title": "Pool note"}))
        result = async_result.get(timeout=120)

    assert result["ok"] is True
    assert result["payload"]["title"] == "Pool note"
    assert result["payload"]["content"] == "Parsed inside a real spawned worker process."
