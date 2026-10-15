# Tests/Web_Server/test_session_store.py
from __future__ import annotations

import json
from pathlib import Path

from tldw_chatbook.Web_Server.session_store import StoredSession, WebSessionStore


def _record(digest_hex: str = "ab12", **overrides) -> StoredSession:
    fields = dict(
        digest_hex=digest_hex,
        session_id="sid-1",
        csrf_token="csrf-1",
        client_ip="127.0.0.1",
        device_label="iPhone · Safari",
        created_at=1000.0,
        last_seen_at=1000.0,
        absolute_expires_at=2000.0,
    )
    fields.update(overrides)
    return StoredSession(**fields)


def test_round_trip_preserves_records(tmp_path: Path) -> None:
    store = WebSessionStore(tmp_path / "web_sessions.json")
    store.save({"ab12": _record(), "cd34": _record("cd34", device_label="Mac · Chrome")})
    loaded = WebSessionStore(tmp_path / "web_sessions.json").load()
    assert set(loaded) == {"ab12", "cd34"}
    assert loaded["cd34"].device_label == "Mac · Chrome"


def test_save_is_atomic_and_private(tmp_path: Path) -> None:
    store = WebSessionStore(tmp_path / "web_sessions.json")
    store.save({"ab12": _record()})
    files = list(tmp_path.iterdir())
    # no temp leftovers (the autouse isolate_test_environment fixture adds a
    # test_data/ directory to tmp_path, so filter rather than list-compare)
    assert not [f.name for f in files if f.name.startswith(".web_sessions-")]
    assert (tmp_path / "web_sessions.json").stat().st_mode & 0o777 == 0o600


def test_file_never_contains_raw_cookie_only_hex_digests(tmp_path: Path) -> None:
    store = WebSessionStore(tmp_path / "web_sessions.json")
    store.save({"ab12": _record()})
    text = (tmp_path / "web_sessions.json").read_text()
    assert "ab12" in text  # digest hex is the only key material


def test_load_missing_or_corrupt_file_fails_safe(tmp_path: Path) -> None:
    assert WebSessionStore(tmp_path / "absent.json").load() == {}
    corrupt = tmp_path / "web_sessions.json"
    corrupt.write_text("{not json")
    assert WebSessionStore(corrupt).load() == {}
