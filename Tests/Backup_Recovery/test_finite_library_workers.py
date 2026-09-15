"""Finite Library workers release only their newly owned native caches."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio,sqlite3,threading,sys
from pathlib import Path
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker
route,outcome=sys.argv[1:]
root=Path.home()
if route=='media':
 from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
 db=MediaDatabase(root/'media.db',client_id='test')
 get,close=db.get_connection,db.close_connection
elif route=='prompts':
 from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
 db=PromptsDatabase(root/'prompts.db',client_id='test')
 get,close=db.get_connection,db.close_connection
elif route=='evals':
 from tldw_chatbook.DB.Evals_DB import EvalsDB
 db=EvalsDB(root/'evals.db',client_id='test')
 get,close=db.get_connection,db.close
elif route=='notifications':
 from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
 db=ClientNotificationsDB(root/'notifications.db')
 get,close=db._held_connection,db.close
elif route=='collections':
 from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
 db=LibraryCollectionsDB(root/'collections.db')
 get,close=db._held_connection,db.close
else:
 from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
 cls=type('CustomDB',(CharactersRAGDB,),{}) if outcome=='custom' else CharactersRAGDB
 db=cls(':memory:' if outcome=='memory' else root/'notes.db',client_id='test')
 get,close=db.get_connection,db.close_connection
main_connection=get()
main_connection.execute('SELECT 1')
observed=[]
worker_errors=[]
worker_done=threading.Event()
entered,release=threading.Event(),threading.Event()
def call():
 connection=get();observed.append(connection)
 connection.execute('SELECT 1')
 if outcome=='transaction':connection.execute('BEGIN')
 if outcome=='error':raise ValueError('injected worker error')
 if outcome=='cancel':
  entered.set();assert release.wait(10)
 return 'rows'
def worker():
 borrowed=get() if outcome=='borrowed' else None
 if outcome=='reopen':get().close()
 operation=None
 if outcome=='operation':
  from tldw_chatbook.Backup_Recovery.participants import _core_operation
  operation=_core_operation(db);operation.__enter__()
 try:
  if outcome=='error':
   with __import__('pytest').raises(ValueError,match='injected worker error'):
    run_finite_local_worker(call)
  else:assert run_finite_local_worker(call)=='rows'
  connection=observed[-1]
  if outcome in {'borrowed','memory','custom','transaction','operation'}:
   assert connection.execute('SELECT 1').fetchone()[0]==1
   if outcome=='transaction':assert connection.in_transaction;connection.rollback()
  else:
   with __import__('pytest').raises(sqlite3.ProgrammingError):connection.execute('SELECT 1')
  assert get().execute('SELECT 1').fetchone()[0]==1
 finally:
  if operation is not None:operation.__exit__(None,None,None)
  close()
def observed_worker():
 try:worker()
 except BaseException as error:worker_errors.append(error);raise
 finally:worker_done.set()
async def main():
 if outcome=='cancel':
  task=asyncio.create_task(asyncio.to_thread(observed_worker))
  assert await asyncio.to_thread(entered.wait,10)
  task.cancel()
  with __import__('pytest').raises(asyncio.CancelledError):await task
  assert observed and not release.is_set()
  release.set()
  assert await asyncio.to_thread(worker_done.wait,10)
  assert not worker_errors,worker_errors
 else:await asyncio.to_thread(observed_worker)
asyncio.run(main())
assert main_connection.execute('SELECT 1').fetchone()[0]==1
close()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["notes", "media", "prompts", "collections", "evals", "notifications"])
@pytest.mark.parametrize(
    "outcome",
    ["success", "error", "borrowed", "transaction", "operation", "reopen", "cancel"],
)
def test_finite_worker_native_cache_lifetime(tmp_path, route, outcome):
    _run(tmp_path, route, outcome, script=_SCRIPT)


@pytest.mark.parametrize("outcome", ["memory", "custom"])
def test_finite_worker_preserves_unowned_lifetimes(tmp_path, outcome):
    _run(tmp_path, "notes", outcome, script=_SCRIPT)
