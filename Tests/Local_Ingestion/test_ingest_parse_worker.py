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
