"""v52→v53: stored Safe exchange captures lose their per-turn history copy.

task-23026: before elision existed, every Safe (default-on) exchange capture
persisted the ENTIRE conversation so far — 21.33 MB measured for one 200-turn
conversation — with no retention path (the only purge is user-invoked and
Full-filtered). The migration applies the same trim the builder now performs
at capture time to every already-stored Safe blob, so existing databases
reclaim the space without the user knowing a manual purge exists.

Correctness bar (this DB's established one): atomic, re-enterable,
interrupt-safe (real-SIGKILL form, per
``Tests/DB/test_chachanotes_v47_messages_fts_backfill.py``), integrity_check
clean, and value-identical content for everything not deliberately trimmed.
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError
from tldw_chatbook.Chat.console_exchange_capture import (
    CAPTURE_SAFE_HISTORY_TAIL_ROWS,
    CaptureDetail,
    ExchangeCapture,
    build_request_capture,
    capture_from_storage,
    capture_to_blob,
    history_elision_marker,
    trim_safe_capture_blob,
)

SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME
_TAIL = CAPTURE_SAFE_HISTORY_TAIL_ROWS


def _version(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _filler(seed: str, chars: int) -> str:
    """Semi-incompressible prose-like filler (hex words): a repeated-word
    filler compresses ~100x under the blob's zlib, which made the ORIGINAL
    blobs unrealistically tiny and inverted every size comparison."""
    import hashlib

    words = []
    counter = 0
    while sum(len(w) + 1 for w in words) < chars:
        words.append(hashlib.sha256(f"{seed}:{counter}".encode()).hexdigest()[:10])
        counter += 1
    return " ".join(words)[:chars]


def _history_rows(count: int) -> list[dict]:
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"MIGRATION-HISTORY-{i:03d} " + _filler(f"row{i}", 180),
        }
        for i in range(count)
    ]


def _legacy_safe_blob(rows: int, *, seq: int = 0) -> bytes:
    """A pre-task-23026 Safe blob: ordinary history rows retained verbatim
    (built at Full to bypass today's elision, stamped Safe — exactly what
    the historical Safe builder produced for untagged rows)."""
    request, omitted = build_request_capture(
        {
            "api_endpoint": "openai",
            "system_message": "sys",
            "messages_payload": _history_rows(rows),
            "model": "gpt-4.1",
            "temp": 0.7,
            "api_key": "sk-SECRET",
        },
        capture_detail=CaptureDetail.FULL,
    )
    return capture_to_blob(
        ExchangeCapture(
            run_tag="legacy-run", seq=seq, created_at="2026-08-01T00:00:00Z",
            provider="openai", model="gpt-4.1", endpoint=None,
            request=request,
            response={"content": "pong", "tool_calls": [], "synthetic_fallback": False},
            status="complete", usage_json='{"input": 9}', omitted_keys=omitted,
            capture_detail=CaptureDetail.SAFE,
        )
    )


def _legacy_wire_blob(rows: int) -> bytes:
    return capture_to_blob(
        ExchangeCapture(
            run_tag="legacy-wire", seq=0, created_at="2026-08-01T00:00:00Z",
            provider="llama_cpp", model="m", endpoint=None,
            request={
                "model": "m",
                "wire_payload": {
                    "model": "m",
                    "messages": _history_rows(rows),
                    "stream": True,
                },
                "truncation_inventory": (),
            },
            response={"content": "ok"}, status="complete", usage_json=None,
            omitted_keys=(), capture_detail=CaptureDetail.SAFE,
        )
    )


def _full_blob(rows: int) -> bytes:
    request, omitted = build_request_capture(
        {"messages_payload": _history_rows(rows), "model": "m"},
        capture_detail=CaptureDetail.FULL,
    )
    return capture_to_blob(
        ExchangeCapture(
            run_tag="full-run", seq=0, created_at="2026-08-01T00:00:00Z",
            provider="openai", model="m", endpoint=None, request=request,
            response={"content": "pong"}, status="complete", usage_json=None,
            omitted_keys=omitted, capture_detail=CaptureDetail.FULL,
        )
    )


def _exchange_row(run_tag: str, seq: int, detail: str, blob: bytes) -> dict:
    return {
        "run_tag": run_tag,
        "seq": seq,
        "status": "complete",
        "abandoned": False,
        "capture_detail": detail,
        "capture_blob": blob,
        "created_at": "2026-08-01T00:00:00Z",
    }


def _seed_v52(db_path: Path) -> str:
    """Build a genuine v52 database holding every migration-relevant row
    class, seeded through production APIs. Returns the message id."""
    with chachanotes_db_at_version(db_path, 52, client_id="v53-seed") as db:
        conv_id = db.add_conversation({"title": "t"})
        message_id = db.add_message(
            {"conversation_id": conv_id, "sender": "user", "content": "hi"}
        )
        db.append_message_exchanges_local(
            message_id,
            [
                _exchange_row("oversized-safe", 0, "safe", _legacy_safe_blob(_TAIL + 12)),
                _exchange_row("small-safe", 0, "safe", _legacy_safe_blob(_TAIL)),
                _exchange_row("wire-safe", 0, "safe", _legacy_wire_blob(_TAIL + 6)),
                _exchange_row("full-verbatim", 0, "full", _full_blob(_TAIL + 12)),
                _exchange_row("corrupt", 0, "safe", b"not-a-zlib-blob"),
            ],
        )
    return message_id


def _blobs_by_run_tag(db: CharactersRAGDB, message_id: str) -> dict[str, bytes]:
    return {
        row["run_tag"]: bytes(row["capture_blob"])
        for row in db.get_message_exchanges(message_id)
    }


def test_v53_trims_safe_blobs_and_leaves_everything_else_value_identical(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "historical-v52.sqlite"
    message_id = _seed_v52(db_path)
    with chachanotes_db_at_version(db_path, 52, client_id="v53-before") as before_db:
        before = _blobs_by_run_tag(before_db, message_id)
    assert _version(db_path) == 52

    db = CharactersRAGDB(db_path, client_id="v53-upgrade")
    try:
        assert _version(db_path) == CharactersRAGDB._CURRENT_SCHEMA_VERSION == 53
        after = _blobs_by_run_tag(db, message_id)

        # 1. The oversized Safe blob was trimmed and still decodes under its
        #    declared provenance.
        assert after["oversized-safe"] != before["oversized-safe"]
        assert len(after["oversized-safe"]) < len(before["oversized-safe"])
        restored = capture_from_storage(after["oversized-safe"], "safe")
        original = capture_from_storage(before["oversized-safe"], "safe")
        payload = restored.request["messages_payload"]
        marker = history_elision_marker(payload)
        assert marker is not None
        assert marker["original_rows"] == _TAIL + 12
        kept = [row for row in payload if not history_elision_marker([row])]
        assert kept == original.request["messages_payload"][-_TAIL:]
        assert "MIGRATION-HISTORY-000 " not in json.dumps(payload)
        # Value-identical content for everything not deliberately trimmed.
        assert restored.response == original.response
        assert restored.status == original.status
        assert restored.usage_json == original.usage_json
        assert restored.run_tag == original.run_tag
        assert {
            k: v for k, v in restored.request.items() if k != "messages_payload"
        } == {
            k: v for k, v in original.request.items() if k != "messages_payload"
        }
        assert set(original.omitted_keys).issubset(set(restored.omitted_keys))
        assert "messages_payload.history" in restored.omitted_keys

        # 2. The llama.cpp wire-literal Safe blob was trimmed the same way.
        wire_restored = capture_from_storage(after["wire-safe"], "safe")
        wire = wire_restored.request["wire_payload"]
        assert history_elision_marker(wire["messages"]) is not None
        assert "MIGRATION-HISTORY-000 " not in json.dumps(wire)
        assert wire["stream"] is True
        assert "wire_payload.messages.history" in wire_restored.omitted_keys

        # 3. A short Safe blob, a Full blob, and an undecodable blob are
        #    byte-untouched — Full is the deliberate verbatim mode, and one
        #    corrupt row must never brick (or be "repaired" by) the upgrade.
        assert after["small-safe"] == before["small-safe"]
        assert after["full-verbatim"] == before["full-verbatim"]
        assert after["corrupt"] == before["corrupt"] == b"not-a-zlib-blob"

        # 4. The database is structurally sound.
        integrity = db.get_connection().execute("PRAGMA integrity_check").fetchone()
        assert tuple(integrity) == ("ok",)

        # 5. Re-entry converges: the trim is a fixed point on migrated blobs.
        assert trim_safe_capture_blob(after["oversized-safe"]) is None
        assert trim_safe_capture_blob(after["wire-safe"]) is None
    finally:
        db.close_connection()

    # 6. A second open performs no further blob writes.
    db_again = CharactersRAGDB(db_path, client_id="v53-again")
    try:
        assert _blobs_by_run_tag(db_again, message_id) == after
    finally:
        db_again.close_connection()


def test_v53_failure_mid_walk_rolls_back_blobs_and_version_together(
    tmp_path: Path, monkeypatch
) -> None:
    """Atomicity, in-process failure form (the media v8→v9 rollback
    precedent): a database error AFTER earlier successful blob rewrites
    must rewind the rewrites AND the version stamp, leaving a working v52
    database that a later clean open upgrades completely."""
    db_path = tmp_path / "historical-v52.sqlite"
    message_id = _seed_v52(db_path)
    with chachanotes_db_at_version(db_path, 52, client_id="v53-before") as before_db:
        before = _blobs_by_run_tag(before_db, message_id)

    import tldw_chatbook.Chat.console_exchange_capture as capture_module

    real_trim = capture_module.trim_safe_capture_blob
    calls = {"n": 0}

    def sabotaged_trim(blob: bytes):
        calls["n"] += 1
        if calls["n"] >= 2:
            # Not raised here (the per-row guard would skip it): RETURN a
            # value sqlite cannot bind, so the failure lands on the UPDATE
            # after at least one earlier row committed its rewrite into the
            # open transaction.
            return object()
        return real_trim(blob)

    monkeypatch.setattr(capture_module, "trim_safe_capture_blob", sabotaged_trim)
    with pytest.raises(SchemaError):
        CharactersRAGDB(db_path, client_id="v53-sabotaged")
    assert calls["n"] >= 2

    # Everything rolled back together: still v52, blobs byte-identical.
    assert _version(db_path) == 52
    with chachanotes_db_at_version(db_path, 52, client_id="v53-inspect") as db52:
        assert _blobs_by_run_tag(db52, message_id) == before

    # A later clean open completes the upgrade.
    monkeypatch.setattr(capture_module, "trim_safe_capture_blob", real_trim)
    recovered = CharactersRAGDB(db_path, client_id="v53-recovered")
    try:
        assert _version(db_path) == 53
        after = _blobs_by_run_tag(recovered, message_id)
        assert after["oversized-safe"] != before["oversized-safe"]
        assert tuple(
            recovered.get_connection()
            .execute("PRAGMA integrity_check")
            .fetchone()
        ) == ("ok",)
    finally:
        recovered.close_connection()


def test_v53_survives_sigkill_mid_migration(tmp_path: Path) -> None:
    """Interrupt-safety, real-kill form (the v47 backfill technique): a
    child process is SIGKILLed while the migration walks the blobs. The
    file must reopen (WAL recovery), still be v52-shaped or fully
    migrated — never in between — and a clean reopen must converge on the
    byte-identical end state of an uninterrupted control run."""
    db_path = tmp_path / "killed-v52.sqlite"
    message_id = _seed_v52(db_path)
    # Extra oversized Safe rows widen the kill window.
    with chachanotes_db_at_version(db_path, 52, client_id="v53-widen") as db52:
        db52.append_message_exchanges_local(
            message_id,
            [
                _exchange_row(f"widen-{i}", i, "safe", _legacy_safe_blob(_TAIL + 12, seq=i))
                for i in range(6)
            ],
        )
        before = _blobs_by_run_tag(db52, message_id)

    # Control arm: an identical database upgraded without interruption.
    control_path = tmp_path / "control-v52.sqlite"
    control_message_id = _seed_v52(control_path)
    with chachanotes_db_at_version(control_path, 52, client_id="v53-widen-c") as dbc:
        dbc.append_message_exchanges_local(
            control_message_id,
            [
                _exchange_row(f"widen-{i}", i, "safe", _legacy_safe_blob(_TAIL + 12, seq=i))
                for i in range(6)
            ],
        )
    control = CharactersRAGDB(control_path, client_id="v53-control")
    try:
        control_blobs = _blobs_by_run_tag(control, control_message_id)
    finally:
        control.close_connection()

    sentinel = tmp_path / "trim-progress"
    child_code = f"""
import sys, time
sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
import tldw_chatbook.Chat.console_exchange_capture as capture_module
real_trim = capture_module.trim_safe_capture_blob
def slow_trim(blob):
    with open({str(sentinel)!r}, "a") as handle:
        handle.write("r\\n")
    time.sleep(0.3)
    return real_trim(blob)
capture_module.trim_safe_capture_blob = slow_trim
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
CharactersRAGDB({str(db_path)!r}, client_id="v53-killed").close_connection()
print("done", flush=True)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if sentinel.exists() and sentinel.read_text().count("r") >= 2:
                break
            if child.poll() is not None:
                break
            time.sleep(0.02)
        assert child.poll() is None, (
            "child finished before it could be killed; widen the seed "
            f"(stdout={child.stdout.read()!r} stderr={child.stderr.read()!r})"
        )
        assert sentinel.exists() and sentinel.read_text().count("r") >= 2, (
            "no trim call observed before the deadline"
        )
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)

    # The kill landed mid-transaction: nothing may be half-applied.
    assert _version(db_path) == 52

    resumed = CharactersRAGDB(db_path, client_id="v53-after-kill")
    try:
        assert _version(db_path) == 53
        after = _blobs_by_run_tag(resumed, message_id)
        # Convergence with the uninterrupted control arm, byte for byte.
        assert after == control_blobs
        assert after["oversized-safe"] != before["oversized-safe"]
        integrity = (
            resumed.get_connection().execute("PRAGMA integrity_check").fetchone()
        )
        assert tuple(integrity) == ("ok",)
    finally:
        resumed.close_connection()


def test_fresh_database_lands_on_v53_with_no_exchange_rows(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.sqlite", client_id="v53-fresh")
    try:
        assert (
            db.get_connection()
            .execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (SCHEMA_NAME,),
            )
            .fetchone()[0]
            == 53
        )
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM message_exchanges")
            .fetchone()[0]
            == 0
        )
    finally:
        db.close_connection()
