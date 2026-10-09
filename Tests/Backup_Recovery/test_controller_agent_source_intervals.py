"""Broad agent work keeps independent native source intervals and Notes cleanup."""

import pytest

from Tests.Backup_Recovery.test_activation_agents import _BRIDGE_WORKER
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = (
    _BRIDGE_WORKER.split("thread=threading.Thread")[0]
    .replace("str(base/'data')", "(base/'data').as_posix()")
    .replace(
        "from tldw_chatbook import config",
        "(base/'home'/'.config'/'tldw_cli').mkdir(parents=True,exist_ok=True)\nfrom tldw_chatbook import config",
    )
    + r"""
import asyncio, time
from types import SimpleNamespace
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.Chat.test_console_prompt_queue_coordinator import _arm_controller, SequencedGateway
notes=CharactersRAGDB(data/'notes.sqlite', 'controller-agent')
notes.close()
notes_connections=[]
entered,release=threading.Event(),threading.Event()
async def main():
 controller,_,_=_arm_controller(SequencedGateway())
 controller.store=SimpleNamespace(persistence=SimpleNamespace(db=notes))
 controller._agent_bridge=bridge
 def actual_reply():
  connection=notes.get_connection()
  assert connection.execute('SELECT 17').fetchone()[0]==17
  notes_connections.append(connection)
  reply()
  entered.set()
  if route=='cancel': assert release.wait(10)
  return outcomes[-1]
 pause=None
 waiter=None
 try:
  if route=='paused': pause=storage._begin_local_pause()
  waiter=asyncio.create_task(controller._run_maintenance_agent_call(actual_reply))
  if route=='paused':
   try: await waiter
   except PermissionError as exc: assert str(exc)=='agent_activation_required'
   else: raise AssertionError('fresh full agent admitted during pause')
   assert not outcomes and not entered.is_set() and not notes_connections
  else:
   while not entered.is_set():
    if waiter.done(): waiter.result()
    await asyncio.sleep(.005)
   assert outcomes[0].status==RUN_DONE and outcomes[0].final_text=='answer', outcomes[0]
   if route=='cancel':
    waiter.cancel();waiter.cancel()
    try: await waiter
    except asyncio.CancelledError: pass
    controller.maintenance_close_admission()
    assert not await controller.maintenance_drain(time.monotonic()+.02)
    pause=storage._begin_local_pause()
    assert not pause.drain(time.monotonic()+.02)
    release.set()
    assert await controller.maintenance_drain(time.monotonic()+5)
   else: assert (await waiter).status==RUN_DONE
  for connection in notes_connections+worker_connections:
   try: sqlite3.Connection.in_transaction.__get__(connection)
   except sqlite3.ProgrammingError as exc: assert 'closed' in str(exc)
   else: raise AssertionError('completed agent retained a new native connection')
  for connection in caller_connections: assert connection.execute('SELECT 1').fetchone()[0]==1
 finally:
  release.set()
  if pause is not None: pause.resume()
  if waiter is not None: await asyncio.gather(waiter,return_exceptions=True)
asyncio.run(main())
notes.close(); bridge_db.close(); registry.db.close(); db.close()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None: startup.close()
pause=storage._begin_local_pause()
try: assert pause.drain(time.monotonic()+1)
finally: pause.resume()
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["ordinary", "cancel", "paused"])
def test_full_agent_worker_independently_admits_config_runs_and_retires_notes(
    tmp_path, route
):
    _run(tmp_path, route, "approved", script=_SCRIPT, timeout=60)
