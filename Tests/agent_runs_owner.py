"""Explicit real execution owners for file-backed AgentRuns test fixtures.

Same-process peer handles share the lease; the last fixture handle's collection
closes it. Process-contention tests use the production constructor directly.
"""

import weakref
from pathlib import Path
from threading import Condition, RLock

AGENT_RUNS_MIGRATIONS = Path(__file__).parents[1] / "tldw_chatbook/DB/migrations"

_owners = {}
_retiring = set()
_changed = Condition()
_construction_lock = RLock()


def _release(key):
    with _changed:
        owner, count = _owners[key]
        if count > 1:
            _owners[key] = owner, count - 1
            return
        _retiring.add(key)

    closed = False
    try:
        owner.close()
        closed = True
    finally:
        with _changed:
            if closed:
                del _owners[key]
            _retiring.remove(key)
            _changed.notify_all()


def owned_agent_runs_db(db_path, *args, **kwargs):
    """Construct a real store with an explicit shared fixture lease."""
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Utils.private_paths import lexical_path

    db_class = kwargs.pop("_db_class", AgentRunsDB)
    if db_path == ":memory:" or "execution_owner" in kwargs:
        return db_class(db_path, *args, **kwargs)
    path = lexical_path(db_path)

    # Native admission can collect an older DB on its holder thread. Keep
    # constructors serialized without making that finalizer wait for admission.
    with _construction_lock:
        owner = None
        with _changed:
            _changed.wait_for(lambda: path not in _retiring)
            if path in _owners:
                owner, count = _owners[path]
                _owners[path] = owner, count + 1
        if owner is None:
            owner = ConsoleExecutionOwner.acquire(path)
            with _changed:
                _owners[path] = owner, 1
        try:
            db = db_class(db_path, *args, execution_owner=owner, **kwargs)
            weakref.finalize(db, _release, path)
        except BaseException:
            _release(path)
            raise
        return db


def install_agent_runs_v18(connection):
    """Install the frozen v18 predecessor for forward migration fixtures."""
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    schema = next(schema for version, schema in _AGENT_RUNS_SCHEMA if version == 18)
    connection.execute("PRAGMA foreign_keys = ON")
    for sql in sorted(schema, key=lambda sql: not sql.startswith("CREATE TABLE")):
        if not sql.startswith("CREATE TABLE sqlite_sequence"):
            connection.execute(sql)
    connection.execute("INSERT INTO schema_version(version) VALUES (18)")
