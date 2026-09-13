"""Merged startup workers retire their own native handles before becoming idle."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace

from loguru import logger
from tldw_chatbook import config
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import storage_admission as storage

route,outcome=sys.argv[1:]
root=Path.home()
db=config.get_chachanotes_db_lazy()
app=SimpleNamespace(chachanotes_db=db,loguru_logger=logger)
getter=db.get_connection
close=db.close_connection
if route in {'fts','persona'}:
 if route=='fts':
  target,method=db,'backfill_messages_fts'
  invoke=lambda: asyncio.to_thread(TldwCli._backfill_chachanotes_messages_fts,app)
 else:
  from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
  from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
  from tldw_chatbook.Character_Chat.local_character_persona_service import LocalCharacterPersonaService
  repository=ActorPackRepository(db)
  app.persona_actor_pack_coordinator=PersonaActorPackCoordinator(repository,LocalCharacterPersonaService(db,persona_store_path=root/'personas.json'))
  app.actor_pack_recovery_error='already_blocked'
  target,method=repository,'list_persona_intents'
  invoke=lambda: asyncio.to_thread(TldwCli.ensure_actor_pack_recovery,app)
elif route=='ingest':
 from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
 target,method=LibraryIngestJobsDB,'all_jobs'
 published=[]
 app._apply_ingest_job_restore=lambda store,plan:published.append(store)
 app.call_from_thread=lambda callback,*args:callback(*args)
 invoke=lambda:asyncio.to_thread(TldwCli._restore_ingest_jobs_off_thread,app)
elif route in {'collections','offline'}:
 from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
 from tldw_chatbook.Library.collections_capture_repository import CollectionsCaptureRepository
 from tldw_chatbook.Library.collections_offline_store import CollectionsOfflineStore
 db=LibraryCollectionsDB(root/'collections.db')
 repository=CollectionsCaptureRepository(db,authority_key='test')
 app.collections_capture_repository=repository if route=='collections' else None
 app.collections_offline_store=None
 if route=='collections':
  target,method=repository,'interrupt_stale_extractions'
 else:
  target=CollectionsOfflineStore(repository,data_root=root,authority_fingerprint='a'*64)
  app.collections_offline_store=target
  method='reconcile_batch'
 getter=db._held_connection
 close=db.close
 invoke=lambda:TldwCli._reconcile_collections_capture_startup(app)
else:
 from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
 from tldw_chatbook.Research_Workspace.source_operation_store import ResearchSourceOperationStore
 from tldw_chatbook.Research_Workspace.source_association import ResearchSourceAssociationScheduler
 from tldw_chatbook.Research_Workspace.source_readiness import ResearchSourceReadinessCoordinator
 from tldw_chatbook.Research_Workspace.local_adapter import LocalResearchWorkspaceAdapter
 from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
 db=WorkspaceDB(root/'workspaces.db')
 store=ResearchSourceOperationStore(db)
 if route=='association':
  target,method=store,'list_association_actionable'
  owner=ResearchSourceAssociationScheduler(coordinator=None,operation_store=store)
  invoke=owner.resume_incomplete
 elif route=='readiness':
  target,method=store,'list_readiness_actionable'
  owner=ResearchSourceReadinessCoordinator(operation_store=store,adapters={})
  invoke=owner.resume_incomplete
 else:
  target,method=LocalWorkspaceRegistryService(db),'list_quick_note_receipts'
  owner=LocalResearchWorkspaceAdapter(target,notes_user_id='test')
  invoke=lambda:owner._reconcile_quick_note_receipts(None,workspace_id=None)
 getter=db._held_connection
 close=db.close

entered=threading.Event()
release=threading.Event()
observed=[]
original=getattr(target,method)
def query(*args,**kwargs):
 result=original(*args,**kwargs)
 owner=args[0] if route=='ingest' else db
 conn=owner._get_connection() if route=='ingest' else getter()
 observed.append((owner,conn,threading.current_thread()))
 entered.set()
 if outcome=='cancel':
  if not release.wait(10):raise RuntimeError('release timed out')
 if outcome=='error':raise RuntimeError('injected after real query')
 return result
setattr(target,method,query)
close()

async def run():
 loop=asyncio.get_running_loop()
 loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
 borrowed=None
 if outcome=='borrowed':
  def acquire():
   conn=getter()
   conn.execute('BEGIN')
   observed.append((db,conn,threading.current_thread()))
   return conn
  borrowed=await asyncio.to_thread(acquire)
 task=asyncio.create_task(invoke())
 if outcome=='cancel':
  while not entered.is_set() and not task.done():await asyncio.sleep(.01)
  assert entered.is_set(),'worker did not reach actual query'
  task.cancel()
  try:await task
  except asyncio.CancelledError:pass
  with storage._lock:
   assert any(lease.resource_thread is observed[0][2] for lease in storage._live_leases)
  release.set()
 else:
  try:await task
  except RuntimeError:
   if outcome!='error':raise
  except sqlite3.OperationalError as error:
   # Transaction-owning startup methods retain their ordinary nested-BEGIN
   # refusal. The borrowed transaction must survive that refusal unchanged.
   assert outcome=='borrowed' and str(error)=='cannot start a transaction within a transaction'
 if borrowed is not None:
  def verify_borrowed():
   assert getter() is borrowed
   assert borrowed.in_transaction,'worker revoked caller transaction'
   assert borrowed.execute('SELECT 1').fetchone()[0]==1
   borrowed.rollback()
   close()
  await asyncio.to_thread(verify_borrowed)
 def verify_closed():
  for owner,connection,worker in observed:
   assert worker is threading.current_thread()
   try:connection.execute('SELECT 1')
   except sqlite3.ProgrammingError as error:assert 'closed' in str(error)
   else:raise AssertionError('worker native handle remains open')
 await asyncio.to_thread(verify_closed)
asyncio.run(run())
assert observed,'actual query did not run'
for owner,connection,worker in observed:
 with storage._lock:
  assert not [lease for lease in storage._live_leases if lease.resource_thread is worker], 'idle startup worker retained native lease'
if route=='ingest' and outcome=='success':
 assert len(published)==1
 assert published[0].all_jobs()==[]
 published[0].close()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "outcome,route",
    [
        (outcome, route)
        for outcome in ("success", "error", "cancel", "borrowed")
        for route in (
            "fts",
            "persona",
            "ingest",
            "collections",
            "offline",
            "association",
            "readiness",
            "receipts",
        )
        if (route, outcome) != ("ingest", "borrowed")
    ],
)
def test_startup_worker_retires_own_native_handle(tmp_path, route, outcome):
    _run(tmp_path, route, outcome, script=_SCRIPT, timeout=60)
