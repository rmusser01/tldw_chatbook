"""Installed cache retirement keeps native ownership on its actual thread."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sqlite3, threading, time, sys
from pathlib import Path
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import _retire_current_thread_caches

route, outcome = sys.argv[1:]
root = Path.home()
if route == 'media':
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    db = MediaDatabase(root/'media.db', client_id='test')
    get, close = db.get_connection, db.close_connection
elif route == 'prompts':
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    db = PromptsDatabase(root/'prompts.db', client_id='test')
    get, close = db.get_connection, db.close_connection
elif route == 'workspace':
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    db = WorkspaceDB(root/'workspace.db')
    get, close = db._held_connection, db.close
elif route == 'file_notes':
    from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
    db = FileNotesReplica(root/'file_notes.db')
    get, close = db._get_connection, db.close
else:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    db = CharactersRAGDB(root/'notes.db', client_id='test')
    get, close = db.get_connection, db.close_connection
connection = get()
connection.execute('SELECT 1').fetchone()
extra = db._get_connection() if outcome == 'uncached' else None
operation = None
if outcome == 'operation':
    from tldw_chatbook.Backup_Recovery.participants import _core_operation
    operation = _core_operation(db)
    operation.__enter__()
if outcome == 'transaction':
    connection.execute('BEGIN')
foreign_entered, foreign_release = threading.Event(), threading.Event()
def worker():
    try:
        get().execute('SELECT 1').fetchone()
        foreign_entered.set()
        foreign_release.wait(5)
    finally:
        close()
thread = None
if outcome == 'foreign':
    thread = threading.Thread(target=worker)
    thread.start()
    assert foreign_entered.wait(5)
pause = storage._begin_local_pause()
try:
    _retire_current_thread_caches(pause)
    if operation is not None:
        assert connection.execute('SELECT 1').fetchone()[0] == 1
        assert not pause.drain(time.monotonic())
        operation.__exit__(None, None, None)
        operation = None
        _retire_current_thread_caches(pause)
    if outcome == 'transaction':
        assert connection.in_transaction
        assert connection.execute('SELECT 1').fetchone()[0] == 1
        assert not pause.drain(time.monotonic())
        connection.rollback()
        _retire_current_thread_caches(pause)
    with __import__('pytest').raises(sqlite3.ProgrammingError):
        connection.execute('SELECT 1')
    if extra is not None:
        assert extra.execute('SELECT 1').fetchone()[0] == 1
        assert not pause.drain(time.monotonic())
        extra.close()
    if outcome == 'foreign':
        assert not pause.drain(time.monotonic())
        assert thread.is_alive()
        foreign_release.set()
        thread.join(5)
    assert pause.drain(time.monotonic()+2)
    with __import__('pytest').raises(Exception, match='participant_runtime_coverage_incomplete'):
        pause.require_runtime_coverage()
finally:
    if operation is not None: operation.__exit__(None, None, None)
    if extra is not None: extra.close()
    foreign_release.set()
    if thread: thread.join(5)
    pause.resume()
assert get().execute('SELECT 1').fetchone()[0] == 1
close()
with __import__('pytest').raises(Exception, match='local_pause_inactive'):
    _retire_current_thread_caches(pause)
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route", ["notes", "media", "prompts", "workspace", "file_notes"]
)
def test_installed_cache_reopens_after_local_pause(tmp_path, route):
    _run(tmp_path, route, "success", script=_SCRIPT)


@pytest.mark.parametrize("outcome", ["transaction", "foreign", "operation"])
def test_live_borrowers_remain_blocking(tmp_path, outcome):
    _run(tmp_path, "notes", outcome, script=_SCRIPT)


def test_uncached_connection_remains_caller_owned(tmp_path):
    _run(tmp_path, "workspace", "uncached", script=_SCRIPT)
