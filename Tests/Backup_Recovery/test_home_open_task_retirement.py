"""Home open-task counters settle their own native worker caches."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio,sqlite3,threading,sys
from pathlib import Path
from types import SimpleNamespace
from tldw_chatbook.app import TldwCli
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Evaluations_Interop.local_evaluations_service import LocalEvaluationsService
route,outcome=sys.argv[1:]
root=Path.home()
if route=='evals':
 cls=type('CustomEvalsDB',(EvalsDB,),{}) if outcome=='custom' else EvalsDB
 db=cls(':memory:' if outcome=='memory' else root/'evals.db',client_id='test')
 getter_name='_get_connection';close=db.close
 app=SimpleNamespace(local_evaluation_service=LocalEvaluationsService(db))
 callback=lambda:TldwCli._local_eval_open_run_counts(app)
else:
 cls=type('CustomMediaDB',(MediaDatabase,),{}) if outcome=='custom' else MediaDatabase
 db=cls(':memory:' if outcome=='memory' else root/'media.db',client_id='test')
 getter_name='get_connection';close=db.close_connection
 app=SimpleNamespace(media_db=db)
 callback=lambda:TldwCli._local_read_later_count(app)
get=getattr(db,getter_name)
main_connection=get()
observed=[]
entered,release,done=threading.Event(),threading.Event(),threading.Event()
errors=[]
def observed_get():
 connection=get();observed.append(connection)
 if outcome=='error':raise ValueError('injected after native acquisition')
 if outcome=='cancel':entered.set();assert release.wait(10)
 return connection
setattr(db,getter_name,observed_get)
def worker():
 try:
  if outcome=='borrowed':get()
  callback()
  assert observed
  connection=observed[-1]
  if outcome in {'borrowed','memory','custom'}:
   assert connection.execute('SELECT 1').fetchone()[0]==1
  else:
   with __import__('pytest').raises(sqlite3.ProgrammingError):connection.execute('SELECT 1')
  assert get().execute('SELECT 1').fetchone()[0]==1
 except BaseException as error:errors.append(error);raise
 finally:close();done.set()
async def main():
 task=asyncio.create_task(asyncio.to_thread(worker))
 if outcome=='cancel':
  assert await asyncio.to_thread(entered.wait,10)
  task.cancel()
  with __import__('pytest').raises(asyncio.CancelledError):await task
  assert not done.is_set()
  release.set()
  assert await asyncio.to_thread(done.wait,10)
  assert not errors,errors
 else:await task
asyncio.run(main())
assert main_connection.execute('SELECT 1').fetchone()[0]==1
close()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["evals", "media"])
@pytest.mark.parametrize(
    "outcome", ["success", "error", "cancel", "borrowed", "memory", "custom"]
)
def test_home_open_task_counter_native_lifetime(tmp_path, route, outcome):
    _run(tmp_path, route, outcome, script=_SCRIPT)
