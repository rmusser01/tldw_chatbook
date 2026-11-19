"""TASK-21240 / TASK-21247 / TASK-21249: notes-sync boot gate + build-failure honesty.

TASK-21112 gated the runtime's START on configuration evidence; TASK-21247
gates its CONSTRUCTION on the same evidence (on_mount no longer touches the
lazy property unconditionally); TASK-21240 refines the evidence so the EMPTY
state DB an abandoned first-time setup leaves behind no longer re-arms the
gate forever; TASK-21249 makes a build failure inside the lazy property
distinguishable from feature-unavailable.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from tldw_chatbook.app import TldwCli
from tldw_chatbook.app_service_wiring import (
    ServiceWiringMixin,
    _notes_sync_boot_start_evidence,
    _notes_sync_state_db_has_configured_roots,
)

pytestmark = pytest.mark.unit


def _make_state_db(path: Path, *, with_root: bool) -> None:
    """A stand-in notes-sync state DB: schema-only, or with one root."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes_sync_roots (
                root_id TEXT PRIMARY KEY,
                note_scope_id TEXT NOT NULL,
                logical_folder_id TEXT,
                canonical_path TEXT NOT NULL,
                remote_origin_id TEXT,
                direction TEXT NOT NULL,
                state TEXT NOT NULL,
                cursor TEXT,
                last_status_code TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        if with_root:
            conn.execute(
                """
                INSERT INTO notes_sync_roots VALUES (
                    'root-1', 'note', 'folder-1', '/tmp/sync', NULL,
                    'bidirectional', 'active', NULL, NULL, 1, 1
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


class _AppDouble:
    """The two slots `_notes_sync_boot_start_evidence` reads."""

    def __init__(self, app_config: object) -> None:
        self.app_config = app_config


# --- TASK-21240: the state-DB evidence clause -----------------------------------


def test_state_db_evidence_false_when_file_absent_and_creates_nothing(tmp_path):
    missing = tmp_path / "tldw_chatbook_notes_sync_state.db"

    assert _notes_sync_state_db_has_configured_roots(missing) is False
    # TASK-21112 rule: the gate predicate stays side-effect free.
    assert not missing.exists()


def test_state_db_evidence_false_for_abandoned_setup_empty_store(tmp_path):
    """TASK-21240's exact scenario: schema-only DB from an abandoned setup."""
    empty = tmp_path / "empty.db"
    _make_state_db(empty, with_root=False)

    assert _notes_sync_state_db_has_configured_roots(empty) is False


def test_state_db_evidence_false_when_roots_table_missing(tmp_path):
    no_table = tmp_path / "other.db"
    conn = sqlite3.connect(no_table)
    conn.execute("CREATE TABLE something_else (x)")
    conn.commit()
    conn.close()

    assert _notes_sync_state_db_has_configured_roots(no_table) is False


def test_state_db_evidence_true_for_configured_root(tmp_path):
    configured = tmp_path / "configured.db"
    _make_state_db(configured, with_root=True)
    before = configured.read_bytes()

    assert _notes_sync_state_db_has_configured_roots(configured) is True
    # Side-effect free: a read-only probe must not so much as touch the file.
    assert configured.read_bytes() == before


# --- TASK-21247 / TASK-21240: the shared boot-evidence predicate ----------------


def test_boot_evidence_false_for_zero_profile_boot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.app_service_wiring.get_notes_sync_state_db_path",
        lambda: tmp_path / "none.db",
    )

    assert _notes_sync_boot_start_evidence(_AppDouble({})) is False


def test_boot_evidence_false_after_abandoned_first_time_setup(tmp_path, monkeypatch):
    """The 21240 AC: an abandoned setup's empty store does not arm the gate."""
    state = tmp_path / "state.db"
    _make_state_db(state, with_root=False)
    monkeypatch.setattr(
        "tldw_chatbook.app_service_wiring.get_notes_sync_state_db_path",
        lambda: state,
    )

    assert _notes_sync_boot_start_evidence(_AppDouble({})) is False


def test_boot_evidence_true_for_legacy_sync_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.app_service_wiring.get_notes_sync_state_db_path",
        lambda: tmp_path / "none.db",
    )
    legacy_config = {"notes": {"sync_directory": "/tmp/legacy"}}

    assert _notes_sync_boot_start_evidence(_AppDouble(legacy_config)) is True


def test_boot_evidence_true_for_on_disk_configured_root(tmp_path, monkeypatch):
    state = tmp_path / "state.db"
    _make_state_db(state, with_root=True)
    monkeypatch.setattr(
        "tldw_chatbook.app_service_wiring.get_notes_sync_state_db_path",
        lambda: state,
    )

    assert _notes_sync_boot_start_evidence(_AppDouble({})) is True


def test_boot_evidence_raises_for_corrupt_store_so_callers_fail_open(
    tmp_path, monkeypatch
):
    """A broken probe must propagate: on_mount and start() both fail OPEN."""
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not a database" * 100)
    monkeypatch.setattr(
        "tldw_chatbook.app_service_wiring.get_notes_sync_state_db_path",
        lambda: corrupt,
    )

    with pytest.raises(sqlite3.DatabaseError):
        _notes_sync_boot_start_evidence(_AppDouble({}))


# --- TASK-21249: build failure vs feature-unavailable ---------------------------


class _ExplodingOwnerApp(TldwCli):
    """TldwCli shape whose owner build raises from inside the property."""

    def _construct_notes_sync_runtime_owner(self):  # type: ignore[override]
        raise AttributeError("boom inside the build")

    def __init__(self) -> None:  # pragma: no cover - only the slots matter
        self._notes_sync_runtime_owner = None
        self._notes_sync_runtime_owner_lock = threading.Lock()


def test_safe_reader_records_build_failure_and_returns_none():
    app = _ExplodingOwnerApp()

    owner = app._read_notes_sync_runtime_owner_safe()

    assert owner is None
    assert isinstance(
        app.__dict__.get("_notes_sync_runtime_build_failure"), AttributeError
    )


def test_safe_reader_does_not_swallow_failure_as_feature_unavailable():
    """The 21249 AC: a build failure is distinguishable, not inert-cutover."""
    from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
        InertLastingSyncRuntime,
    )

    app = _ExplodingOwnerApp()
    app._read_notes_sync_runtime_owner_safe()

    build_failed = getattr(app, "_notes_sync_runtime_build_failure", None) is not None
    inert = (
        InertLastingSyncRuntime(status="failed", next_action="review_settings")
        if build_failed
        else InertLastingSyncRuntime()
    )
    snapshot = inert.snapshot()
    # The failed branch must NOT read as "not available yet".
    assert (snapshot.status, snapshot.next_action) != (
        "awaiting_cutover",
        "finish_upgrade",
    )


def test_safe_reader_is_memoized_until_a_double_is_installed():
    app = _ExplodingOwnerApp()

    assert app._read_notes_sync_runtime_owner_safe() is None
    # Second read must not re-raise (and not rebuild): failure is sticky.
    assert app._read_notes_sync_runtime_owner_safe() is None

    # Installing a double through the setter clears the failure: the reader
    # must not keep shadowing a substituted owner.
    double = object()
    app.notes_sync_runtime_owner = double  # type: ignore[assignment]
    assert app._read_notes_sync_runtime_owner_safe() is double


def test_concurrent_safe_readers_share_a_failed_construction():
    from concurrent.futures import ThreadPoolExecutor

    entered, release, second_lock_attempt = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    native_lock = threading.Lock()
    attempts = []

    class ObservedLock:
        def __enter__(self):
            if native_lock.locked():
                second_lock_attempt.set()
            native_lock.acquire()

        def __exit__(self, *_):
            native_lock.release()

    class App(_ExplodingOwnerApp):
        def _construct_notes_sync_runtime_owner(self):
            attempts.append(1)
            entered.set()
            assert release.wait(3)
            if len(attempts) == 1:
                raise ValueError("first construction failed")
            return object()

    app = App()
    app._notes_sync_runtime_owner_lock = ObservedLock()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(app._read_notes_sync_runtime_owner_safe)
        try:
            assert entered.wait(3)
            second = executor.submit(app._read_notes_sync_runtime_owner_safe)
            # The old reader passes the failure check before waiting here.
            # A serialized reader instead waits outside the property.
            second_lock_attempt.wait(0.2)
        finally:
            release.set()
        assert first.result(timeout=3) is None
        assert second.result(timeout=3) is None
    assert len(attempts) == 1


# --- TASK-21249(a): the construction read point ---------------------------------


class _RecordingOwnerApp(TldwCli):
    """Captures which app slots the construction actually read."""

    def _construct_notes_sync_runtime_owner(self):  # type: ignore[override]
        return {
            "user_id": self.notes_user_id,
            "config": self.app_config,
        }

    def __init__(self) -> None:  # pragma: no cover - only the slots matter
        self._notes_sync_runtime_owner = None
        self._notes_sync_runtime_owner_lock = threading.Lock()
        self.notes_user_id = "first-user"
        self.app_config = {"origin": "first"}


def test_owner_inputs_are_read_once_at_construction_not_at_use():
    """Moving first access later cannot rebuild the owner with new values.

    The property caches the built owner; values read inside the construct
    (the single documented read point) are frozen into it. A later slot
    mutation must not leak into what the app already holds.
    """
    app = _RecordingOwnerApp()
    app.notes_sync_runtime_owner  # first access builds (property getter)

    app.notes_user_id = "second-user"
    app.app_config = {"origin": "second"}

    owner = app.notes_sync_runtime_owner
    assert owner == {"user_id": "first-user", "config": {"origin": "first"}}


def test_mixin_safe_reader_lives_on_service_wiring_mixin():
    """The reader is part of the app's composition surface, not the screen's."""
    assert "_read_notes_sync_runtime_owner_safe" in vars(ServiceWiringMixin)
