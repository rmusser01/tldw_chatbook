"""Finite Home workers retire only their own native database handles."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

_SCRIPT = r"""
import asyncio
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
import sys
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Screens.home_screen import HomeScreen

route, outcome = sys.argv[1:]
root = Path.home()
template = config.get_chachanotes_db_lazy()
local = NotesInteropService(template.db_path.parent, "test", template)
scope = NotesScopeService(local, None)
db = local._get_db("test")
method = "count_notes" if route == "count" else "list_notes"
kwargs = dict(scope="local_note", user_id="test")
if route == "media":
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
    from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
    db.close_connection()
    db = MediaDatabase(root / "media.db", client_id="test")
    scope = MediaReadingScopeService(LocalMediaReadingService(db), None)
    method = "get_paginated_files"
    kwargs = dict(mode="local", page=1, results_per_page=1)
    call = scope.list_media_items
else:
    call = getattr(scope, method)
app_call = None
close = db.close_connection
if route in ("active", "flashcards", "reconcile", "migration"):
    from tldw_chatbook.app import TldwCli
    from loguru import logger
    app = SimpleNamespace(chachanotes_db=db, loguru_logger=logger)
    if route == "active":
        from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
        from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService
        from tldw_chatbook.Home.active_work_adapter import LocalNotificationHomeActiveWorkAdapter
        db.close_connection()
        db = SubscriptionsDB(root / "subscriptions.db")
        close = db.close
        adapter = LocalNotificationHomeActiveWorkAdapter(watchlist_service=LocalWatchlistsService(db_factory=lambda: db))
        target, method = adapter.watchlist_service, "list_home_run_snapshot"
        app_call = adapter.refresh_active_work_cache_async
    elif route == "flashcards":
        target, method = db, "count_due_flashcards"
        async def app_call():
            return await asyncio.to_thread(TldwCli._local_flashcards_due_count, app)
    else:
        from Tests.Chat.test_citation_trace_repository import _repository
        repository = _repository(db)
        if route == "reconcile":
            from Tests.Chat.test_citation_artifact_ownership import _persist, _owner_request, console_chatbook_artifact_payload
            from tldw_chatbook.Chatbooks import LocalChatbookService
            from tldw_chatbook.Chat.citation_artifact_ownership import CitationArtifactOwnershipCoordinator
            _persist(db, repository)
            service = LocalChatbookService(db_paths={}, registry_path=root / "chatbooks.json")
            coordinator = CitationArtifactOwnershipCoordinator(artifact_store=service, trace_repository=repository)
            service.set_citation_ownership_coordinator(coordinator)
            asyncio.run(service.create_chatbook(**console_chatbook_artifact_payload(title="Test", message_text="Answer [S1].", message_role="Assistant"), provenance_owner_request=_owner_request(repository)))
            app.citation_artifact_ownership_coordinator = coordinator
            target, method = coordinator, "reconcile_pending"
            app_call = lambda: TldwCli._reconcile_citation_artifact_ownership(app)
        else:
            from Tests.Chat.test_citation_legacy_migration import _conversation_with_messages, _write_sidecar, CODEC
            from tldw_chatbook.Chat.citation_legacy_migration import CitationLegacyMigrationService
            conversation, messages = _conversation_with_messages(db, 1)
            sidecar = root / "sidecar.json"
            _write_sidecar(sidecar, conversation, messages)
            migration = CitationLegacyMigrationService(db=db, repository=repository, sidecar_path=sidecar, fingerprint_codec=CODEC)
            app.citation_legacy_migration_service = migration
            target, method = migration, "migrate_idle_unit"
            app_call = lambda: TldwCli._migrate_legacy_citations_idle_unit(app)
else:
    target = db
original = getattr(target, method)
entered = threading.Event()
release = threading.Event()
observed = []
def query(*args, **kwargs):
    result = original(*args, **kwargs)
    connection = db.conn if route == "active" else db.get_connection()
    observed.append((connection, threading.current_thread()))
    entered.set()
    if outcome == "cancel":
        if not release.wait(10):
            raise RuntimeError("worker release timed out")
    if outcome == "error":
        raise RuntimeError("injected after real query")
    return result
setattr(target, method, query)
close()
closed_on_worker = []
if route == "active":
    def checked_close():
        close()
        if observed:
            connection, worker = observed[-1]
            if worker is threading.current_thread():
                try:
                    connection.execute("SELECT 1")
                except sqlite3.ProgrammingError as error:
                    assert "closed" in str(error)
                    closed_on_worker.append(worker)
                else:
                    raise AssertionError("native connection remains open")
    db.close = checked_close
template.close_connection()
home = SimpleNamespace(app_instance=SimpleNamespace(chachanotes_db=template))
async def invoke():
    if app_call is not None:
        return await app_call()
    return await HomeScreen._home_content_seam_call(home, call, **kwargs)
async def run():
    task = asyncio.create_task(invoke())
    if outcome == "cancel":
        while not entered.is_set():
            await asyncio.sleep(.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        with storage._lock:
            assert any(lease.resource_thread is observed[0][1] for lease in storage._live_leases)
        release.set()
    else:
        try:
            result = await task
        except RuntimeError:
            if outcome != "error" or route != "active":
                raise
            result = None
        if route in ("list", "count", "media"):
            assert (result is None) == (outcome == "error")
asyncio.run(run())
assert observed
for connection, worker in observed:
    with storage._lock:
        assert not [lease for lease in storage._live_leases if lease.resource_thread is worker], "worker retained lease"
    if route == "active":
        assert worker in closed_on_worker
        continue
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("native connection remains open")
setattr(target, method, original)
if route == "active":
    adapter._active_work_cache = None
asyncio.run(invoke())
print("retired and reopened")
"""

_GATE_SCRIPT = r"""
import asyncio
import sys
from types import SimpleNamespace
from tldw_chatbook import config
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Screens.home_screen import HomeScreen

template = config.get_chachanotes_db_lazy()
local = NotesInteropService(template.db_path.parent, "test", template)
acquired = []
def forbidden_acquisition(user_id):
    acquired.append(user_id)
    raise AssertionError("rejected request acquired database")
local._get_db = forbidden_acquisition
class Deny:
    def require_allowed(self, **kwargs):
        raise PermissionError("policy denied")
denied = sys.argv[1] == "denied"
scope = NotesScopeService(local, None, policy_enforcer=Deny() if denied else None)
kwargs = dict(scope="local_note", user_id="test" if denied else None)
async def direct():
    try:
        await scope.list_notes(**kwargs)
    except (PermissionError if denied else ValueError):
        pass
    else:
        raise AssertionError("original rejection was lost")
asyncio.run(asyncio.to_thread(lambda: asyncio.run(direct())))
home = SimpleNamespace(app_instance=SimpleNamespace(chachanotes_db=template))
assert asyncio.run(HomeScreen._home_content_seam_call(home, scope.list_notes, **kwargs)) is None
assert not acquired, "rejected request acquired database"
template.close_connection()
print("retired and reopened")
"""


@pytest.mark.parametrize("rejection", ["denied", "invalid"])
def test_notes_list_rejection_does_not_acquire_database(tmp_path, rejection):
    _run(tmp_path, rejection, "success", script=_GATE_SCRIPT)


_PRESERVE_SCRIPT = r"""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from tldw_chatbook.app import TldwCli
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.home_screen import HomeScreen

class CustomDatabase(CharactersRAGDB):
    pass

def worker():
    for db in (CharactersRAGDB(":memory:", client_id="test"),
               CustomDatabase(Path.home() / "custom.db", client_id="test")):
        connection = db.get_connection()
        try:
            result = TldwCli._local_flashcards_due_count(SimpleNamespace(chachanotes_db=db))
            assert result == 0
            assert connection.execute("SELECT 1").fetchone()[0] == 1
        finally:
            db.close_connection()
asyncio.run(asyncio.to_thread(worker))
class CustomOwner:
    is_memory_db = False
    def close_connection(self):
        raise AssertionError("custom owner must not be closed")
owner = CustomOwner()
home = SimpleNamespace(app_instance=SimpleNamespace(chachanotes_db=owner))
assert asyncio.run(HomeScreen._home_content_seam_call(home, lambda: [1])) == [1]
print("retired and reopened")
"""


def test_home_preserves_memory_subclass_and_custom_owners(tmp_path):
    _run(tmp_path, "preserve", "success", script=_PRESERVE_SCRIPT)


@pytest.mark.parametrize(
    "route",
    ["list", "count", "media", "active", "flashcards", "reconcile", "migration"],
)
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_home_worker_retirement(tmp_path, route, outcome):
    _run(tmp_path, route, outcome)


def _run(tmp_path, route, outcome, *, script=_SCRIPT):
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
        [sys.executable, "-c", script, route, outcome],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    assert "retired and reopened" in result.stdout
