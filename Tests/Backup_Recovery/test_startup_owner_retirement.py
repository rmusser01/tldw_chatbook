"""Actual startup initializers must retire native resources on their worker."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


_SCRIPT = r"""
import concurrent.futures
import json
import sqlite3
import threading
from types import SimpleNamespace
from pathlib import Path
import sys

from tldw_chatbook import app as application, config
from tldw_chatbook.Backup_Recovery import storage_admission as storage

kind, failure = sys.argv[1:]
failure = failure == "failure"
observed = []
workers = []

def retain(db):
    connection = db.get_connection()
    connection.execute("CREATE TABLE startup_retirement_marker(value TEXT)")
    connection.execute("INSERT INTO startup_retirement_marker VALUES ('committed')")
    connection.commit()
    observed.append((db, connection))
    return db

app = SimpleNamespace(app_config={}, prompts_client_id="startup-test",
    _notify_rag_indexing_failure=lambda *a: None,
    _notify_rag_indexing_guidance=lambda *a: None)

if kind == "notes":
    lazy = application.get_chachanotes_db_lazy
    application.get_chachanotes_db_lazy = lambda: retain(lazy())
    constructor = application.NotesInteropService
    def notes(*args, **kwargs):
        result = constructor(*args, **kwargs)
        if failure:
            raise RuntimeError("injected notes service failure")
        return result
    application.NotesInteropService = notes
    initialize = lambda: application.TldwCli._init_notes_service(app, "startup-user")
elif kind == "prompts":
    original = application.prompts_interop.initialize_interop
    def prompts(*args, **kwargs):
        original(*args, **kwargs)
        retain(application.prompts_interop.get_db_instance())
        if failure:
            raise RuntimeError("injected prompts interop failure")
    application.prompts_interop.initialize_interop = prompts
    initialize = lambda: application.TldwCli._init_prompts_service(app)
elif kind == "media":
    constructor = application.MediaDatabase
    def media(*args, **kwargs):
        db = retain(constructor(*args, **kwargs))
        if failure:
            def prefetch(**kwargs):
                raise RuntimeError("injected media prefetch failure")
            db.get_distinct_media_types = prefetch
        return db
    application.MediaDatabase = media
    initialize = lambda: application.TldwCli._init_media_db(app)
else:
    assert kind == "notes_seed"
    original_seed = config.seed_builtin_content
    def seed(db):
        original_seed(db)
        retain(db)
        raise RuntimeError("injected notes seed failure")
    config.seed_builtin_content = seed
    initialize = lambda: application.TldwCli._init_notes_service(app, "startup-user")

def start():
    workers.append(threading.current_thread())
    initialize()

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    pool.submit(start).result(timeout=20)
assert len(observed) == 1, "actual database construction was not reached"
db, initial_connection = observed[0]
assert not workers[0].is_alive()
with storage._lock:
    leaked = [lease for lease in storage._live_leases if lease.resource_thread is workers[0]]
assert not leaked, "startup worker retained a native storage lease"
try:
    initial_connection.execute("SELECT 1")
except sqlite3.ProgrammingError as error:
    assert "closed database" in str(error)
else:
    raise AssertionError("startup native connection remains open")

if kind == "notes":
    assert (app.notes_service is None) == failure
    if not failure:
        assert app.notes_service.unified_db_template is db
elif kind == "prompts":
    assert app.prompts_service_initialized is not failure
    assert application.prompts_interop.get_db_instance() is db
elif kind == "media":
    assert (app.media_db is None) == failure
    if not failure:
        assert app.media_db is db
        assert app._media_types_for_ui[0] == "All Media"
    else:
        assert app._media_types_for_ui == ["Error: Exception fetching media types"]
else:
    assert app.notes_service is None
    assert config.chachanotes_db is None

# The same reusable owner can allocate a fresh connection on a later thread;
# committed initialization data survives both success and later init failures.
def later():
    current = threading.current_thread()
    assert current is not workers[0]
    connection = db.get_connection()
    try:
        assert connection is not initial_connection
        assert connection.execute("SELECT value FROM startup_retirement_marker").fetchone()[0] == "committed"
    finally:
        db.close_connection()
    with storage._lock:
        assert not [lease for lease in storage._live_leases if lease.resource_thread is current]
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    pool.submit(later).result(timeout=20)
print(json.dumps({"owner": kind, "failure": failure, "retired": True}))
"""


@pytest.mark.parametrize("kind", ["notes", "prompts", "media"])
@pytest.mark.parametrize("failure", [False, True])
def test_startup_worker_retires_native_handle_and_owner_reopens(
    tmp_path, kind, failure
):
    _run(tmp_path, kind, failure)


def test_failed_notes_seed_retires_constructed_owner_before_reset(tmp_path):
    _run(tmp_path, "notes_seed", True)


def _run(tmp_path, kind, failure):
    # Fix one selector before any app import; parent collection/per-test config
    # selectors must not be mistaken for one source generation in this child.
    root = tmp_path.resolve()
    for name in ("home", "config", "data"):
        (root / name).mkdir(mode=0o700)
    environment = os.environ.copy()
    environment.update(
        HOME=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        TLDW_CONFIG_PATH=str(root / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, kind, "failure" if failure else "success"],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    assert '"retired": true' in result.stdout
