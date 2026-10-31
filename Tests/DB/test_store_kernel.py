"""SQLiteStoreKernel contract tests (ADR-228, TASK-33132).

The kernel extracts the hardened Workspace/RAG lineage; these tests pin the
four load-bearing semantics: autocommit held connections, fail-loud nesting,
BaseException rollback (task-32801.5), and liveness replacement of closed
connections.
"""

from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.DB.store_kernel import SQLiteStoreKernel


class _KernelStore(SQLiteStoreKernel):
    """Minimal concrete store over a temporary file path."""

    _kernel_sqlite_owner = "tests.store_kernel"

    def __init__(self, db_path: str) -> None:
        self.is_memory_db = db_path == ":memory:"
        self.db_path_str = db_path
        self._init_store_kernel()
        with self.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kernel_probe "
                "(id INTEGER PRIMARY KEY, value TEXT)"
            )


def test_read_write_roundtrip_on_held_connection(tmp_path) -> None:
    store = _KernelStore(str(tmp_path / "probe.sqlite"))
    with store.transaction() as conn:
        conn.execute("INSERT INTO kernel_probe (value) VALUES ('first')")
    with store.connection() as conn:
        row = conn.execute(
            "SELECT value FROM kernel_probe WHERE id = 1"
        ).fetchone()
    assert row["value"] == "first"
    store.close()


def test_held_connection_is_true_autocommit(tmp_path) -> None:
    store = _KernelStore(str(tmp_path / "probe.sqlite"))
    with store.connection() as conn:
        # In autocommit mode a bare write commits immediately and no
        # transaction is open afterwards.
        conn.execute("INSERT INTO kernel_probe (value) VALUES ('bare')")
        assert conn.in_transaction is False
    store.close()


def test_nested_transaction_fails_loud_and_outer_rolls_back(tmp_path) -> None:
    store = _KernelStore(str(tmp_path / "probe.sqlite"))
    with pytest.raises(sqlite3.OperationalError):
        with store.transaction() as outer:
            outer.execute("INSERT INTO kernel_probe (value) VALUES ('outer')")
            with store.transaction() as inner:  # noqa: F841 - must raise
                pass
    # The outer block rolled back cleanly: the failure propagated through
    # its except path, so nothing landed.
    with store.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM kernel_probe").fetchone()[0]
    assert count == 0
    # And the thread's connection is still usable for a fresh write.
    with store.transaction() as conn:
        conn.execute("INSERT INTO kernel_probe (value) VALUES ('after')")
    with store.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM kernel_probe").fetchone()[0]
    assert count == 1
    store.close()


def test_base_exception_rolls_back_and_connection_survives(tmp_path) -> None:
    store = _KernelStore(str(tmp_path / "probe.sqlite"))

    class Escaped(BaseException):
        pass

    with pytest.raises(Escaped):
        with store.transaction() as conn:
            conn.execute("INSERT INTO kernel_probe (value) VALUES ('lost')")
            raise Escaped("task-32801.5 shape")

    # No transaction may remain open after the escaped BaseException...
    with store.connection() as conn:
        assert conn.in_transaction is False
        # ...so a later write rides its OWN transaction and survives close.
        conn.execute("INSERT INTO kernel_probe (value) VALUES ('survivor')")

    path = store.db_path_str
    store.close()
    reopened = _KernelStore(path)
    with reopened.connection() as conn:
        values = {r["value"] for r in conn.execute("SELECT value FROM kernel_probe")}
    assert values == {"survivor"}
    reopened.close()


def test_liveness_replaces_connection_closed_behind_our_back(tmp_path) -> None:
    store = _KernelStore(str(tmp_path / "probe.sqlite"))
    with store.transaction() as conn:
        conn.execute("INSERT INTO kernel_probe (value) VALUES ('kept')")
    victim = store._held_connection()
    victim.close()
    # Force the probe path by pretending the connection has been idle.
    store._thread_local.conn_last_used = 0.0
    revived = store._held_connection()
    assert revived is not victim
    with store.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM kernel_probe").fetchone()[0]
    assert count == 1
    store.close()


def test_kernel_wiring_is_participants_observable(tmp_path, monkeypatch) -> None:
    """The kernel's entry points fire the Backup_Recovery hooks (ADR-228).

    The kernel imports the ``_core_*`` helpers by name, so the pin patches
    the kernel module's own bindings and asserts each hook fires with this
    store — a refactor that drops the wiring fails here at the kernel seam,
    before any store-level suite notices indirectly.
    """
    import tldw_chatbook.DB.store_kernel as kernel_mod

    fired: list[tuple[str, object]] = []

    def recorder(name):
        def hook(*args):
            fired.append((name, args[0] if args else None))

        return hook

    monkeypatch.setattr(kernel_mod, "_core_access", recorder("access"))
    monkeypatch.setattr(
        kernel_mod, "_register_core_connection", recorder("register")
    )
    monkeypatch.setattr(
        kernel_mod, "_core_cached_connection",
        lambda repo, conn: recorder("cached")(repo) or conn,
    )

    class _ObservedStore(_KernelStore):
        _kernel_sqlite_owner = "tests.store_kernel"

    store = _ObservedStore(str(tmp_path / "hooks.sqlite"))
    store._held_connection()
    store.close()

    names = {name for name, _ in fired}
    assert {"access", "register", "cached"} <= names
    assert all(target is store for _, target in fired)


def _traced(store):
    statements: list[str] = []
    with store.connection() as conn:
        conn.set_trace_callback(statements.append)
    return statements


def test_kernel_default_begin_is_immediate(tmp_path) -> None:
    store = _KernelStore(str(tmp_path / "begin.sqlite"))
    statements = _traced(store)
    with store.transaction():
        pass
    assert any(s.strip().upper().startswith("BEGIN IMMEDIATE") for s in statements)
    store.close()


def test_kernel_deferred_begin_available(tmp_path) -> None:
    store = _KernelStore(str(tmp_path / "begin.sqlite"))
    statements = _traced(store)
    with store.transaction(immediate=False):
        pass
    assert any(s.strip().upper() == "BEGIN" for s in statements)
    assert not any("IMMEDIATE" in s.upper() for s in statements)
    store.close()


def test_workspace_transaction_default_is_deferred_begin(tmp_path) -> None:
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    store = WorkspaceDB(str(tmp_path / "workspace.sqlite"))
    statements = _traced(store)
    with store.transaction():
        pass
    assert any(s.strip().upper() == "BEGIN" for s in statements)
    assert not any("IMMEDIATE" in s.upper() for s in statements)
    store.close()


class _PoisonableConnection(sqlite3.Connection):
    """Live connection whose liveness probe can be made to fail on demand."""

    probe_poisoned = False

    def execute(self, sql, *args):  # type: ignore[override]
        if self.probe_poisoned and sql == "SELECT 1":
            raise sqlite3.OperationalError("poisoned probe on live handle")
        return super().execute(sql, *args)


def test_probe_failure_on_live_connection_propagates(tmp_path) -> None:
    """Failed probe on a still-live handle must raise, not replace (ADR-228).

    The reference port's asymmetry: a probe that fails while the handle is
    positively alive propagates the error and retains the cached
    connection; only a positively-closed handle is replaced.
    """

    class _PoisonableStore(_KernelStore):
        _kernel_sqlite_owner = "tests.store_kernel"

        def _open_connection(self):
            return sqlite3.connect(self.db_path_str, factory=_PoisonableConnection)

    store = _PoisonableStore(str(tmp_path / "live.sqlite"))
    conn = store._held_connection()
    assert isinstance(conn, _PoisonableConnection)
    conn.probe_poisoned = True
    store._thread_local.conn_last_used = 0.0  # force the probe path
    with pytest.raises(sqlite3.OperationalError):
        store._held_connection()
    assert store._thread_local.conn is conn
    # Unpoisoned, the retained handle serves statements normally.
    conn.probe_poisoned = False
    with store.connection() as recovered:
        assert recovered.execute("SELECT 1").fetchone()[0] == 1
    store.close()


def test_configure_connection_applies_on_revival(tmp_path) -> None:
    class _ConfiguredStore(_KernelStore):
        _kernel_sqlite_owner = "tests.store_kernel"

        def _configure_connection(self, conn):
            conn = super()._configure_connection(conn)
            # Per-connection only (never persisted in the file), so a fresh
            # connection to the same database reports the default, not 42.
            conn.execute("PRAGMA cache_size = -1234")
            return conn

    store = _ConfiguredStore(str(tmp_path / "revive.sqlite"))
    first = store._held_connection()
    assert first.execute("PRAGMA cache_size").fetchone()[0] == -1234
    first.close()
    store._thread_local.conn_last_used = 0.0
    revived = store._held_connection()
    assert revived is not first
    # Proves _configure_connection ran on the REVIVED connection: a fresh
    # connection would report SQLite's default (-2000), not the override.
    assert revived.execute("PRAGMA cache_size").fetchone()[0] == -1234
    assert revived.isolation_level is None
    store.close()


def test_workspace_explicit_immediate_wins_over_deferred_default(tmp_path) -> None:
    """Tri-state composition pin (PR-3112 review): explicit immediate=True
    on a False-default store still emits BEGIN IMMEDIATE — the write
    serialization registry_service depends on."""
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    store = WorkspaceDB(str(tmp_path / "workspace.sqlite"))
    statements = _traced(store)
    with store.transaction(immediate=True):
        pass
    assert any(s.strip().upper().startswith("BEGIN IMMEDIATE") for s in statements)
    store.close()
