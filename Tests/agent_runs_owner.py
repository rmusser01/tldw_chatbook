"""Explicit real execution owners for file-backed AgentRuns test fixtures.

Same-process peer handles share the lease; the last fixture handle's collection
closes it. Process-contention tests use the production constructor directly.
"""

import weakref
from pathlib import Path
from threading import RLock

_owners = {}
_lock = RLock()


def _release(key):
    with _lock:
        owner, count = _owners[key]
        if count == 1:
            owner.close()
            del _owners[key]
        else:
            _owners[key] = owner, count - 1


def owned_agent_runs_db(db_path, *args, **kwargs):
    """Construct a real store with an explicit shared fixture lease."""
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Utils.private_paths import lexical_path

    db_class = kwargs.pop("_db_class", AgentRunsDB)
    if db_path == ":memory:" or "execution_owner" in kwargs:
        return db_class(db_path, *args, **kwargs)
    path = lexical_path(db_path)
    with _lock:
        entry = _owners.get(path)
        if entry is None:
            owner, count = ConsoleExecutionOwner.acquire(Path(path)), 0
        else:
            owner, count = entry
        _owners[path] = owner, count + 1
        try:
            db = db_class(db_path, *args, execution_owner=owner, **kwargs)
        except BaseException:
            _release(path)
            raise
        weakref.finalize(db, _release, path)
        return db
