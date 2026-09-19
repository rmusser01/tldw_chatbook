"""Mounted Settings uses the actual local embedding recovery owner review."""

import pytest

from Tests.Backup_Recovery.test_activation_local_embedding import (
    _ISOLATED,
    _ISOLATED_CHILD,
    _TINY_MODEL,
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_UI = _ISOLATED_CHILD.split("if mode=='approve':", 1)[0] + r"""
import asyncio
import sqlite3
import threading
from contextlib import closing
from types import SimpleNamespace
from rich.markup import escape
from textual.app import App
from textual.widgets import Button
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
from tldw_chatbook.Embeddings import Embeddings_Lib as library
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing,recovery as projections
from tldw_chatbook.UI.Screens import settings_screen as settings_module
marked=model.with_name('[bold]local-bert');model.rename(marked);model=marked
os.environ.update(RAG_EMBEDDING_MODEL=str(model),RAG_DEVICE='cpu',RAG_PERSIST_DIR=str(Path(entry.data)/'vectors'))
if mode=='unavailable':os.environ['RAG_EMBEDDING_MODEL']=str(model.parent/'missing-model')
config=resolve_active_rag_config()
if mode!='unavailable':assert config.embedding.model==str(model)
before_config=Path(entry.config).read_bytes()
loads=[]
original_load=library._HuggingFaceEmbedder.__init__
def forbidden_load(*args,**kwargs):
 loads.append('model-load');raise AssertionError('review constructed a model')
library._HuggingFaceEmbedder.__init__=forbidden_load
original_backfill=indexing.backfill_semantic_index
original_rebuild=projections.rebuild_projection
original_reconcile=projections.reconcile_projection
def forbidden_projection(*args,**kwargs):
 loads.append('projection-work');raise AssertionError('model review started projection work')
indexing.backfill_semantic_index=settings_module.backfill_semantic_index=forbidden_projection
projections.rebuild_projection=projections.reconcile_projection=forbidden_projection
entered=threading.Event();release=threading.Event()
if mode=='late':
 original_preview=recovery.preview_local_embedding
 def delayed_preview(config):
  result=original_preview(config);entered.set();assert release.wait(10);return result
 recovery.preview_local_embedding=delayed_preview
if mode=='accepted-cancel':
 original_write=recovery._write
 def delayed_write(*args):
  result=original_write(*args);entered.set();assert release.wait(10);return result
 recovery._write=delayed_write
def index_contents():
 path=Path(entry.data)/'vectors'/'chroma.sqlite3'
 if not path.exists():return ((),())
 with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as connection:
  return (tuple(connection.execute('SELECT * FROM collections').fetchall()),tuple(connection.execute('SELECT * FROM embeddings').fetchall()))
instance=SimpleNamespace(app_config=installed_config.load_settings())
if mode=='rebuild':
 from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
 media=MediaDatabase(installed_config.get_media_db_path(),client_id='model-review-fixture')
 media.add_media_with_keywords(url='https://example.invalid/local',title='Local model content',media_type='document',content='local model content '*12,keywords=['local'],overwrite=True)
 instance.media_db=media
 instance.chachanotes_db=installed_config.get_chachanotes_db_lazy()
 assert instance.chachanotes_db is not None
class Host(App):
 def notify(self,message,**kwargs):
  if hasattr(self,'messages'):self.messages.append(str(message))
  return super().notify(message,**kwargs)
 def on_mount(self):
  self.messages=[]
  screen=SettingsScreen(instance)
  screen.active_category=SettingsCategoryId.LIBRARY_RAG.value
  self.push_screen(screen)
async def wait_for(predicate):
 async with asyncio.timeout(12):
  while not predicate():await asyncio.sleep(.02)
async def main():
 host=Host()
 async with host.run_test(size=(120,45)) as pilot:
  screen=host.screen
  await host.workers.wait_for_complete();await pilot.pause()
  before_index=index_contents()
  unrelated_inactive()
  assert screen.query('#settings-library-rag-model-review'),'local model recovery review is not reachable'
  if mode=='draft':screen._stage_library_rag_value('embedding_model','unsaved-model')
  screen.query_one('#settings-library-rag-model-review',Button).press()
  if mode in ('draft','late','unavailable','ordinary'):
   if mode=='late':
    assert await asyncio.to_thread(entered.wait,5)
    screen._stage_library_rag_value('embedding_model','changed-during-preview')
    release.set()
   await host.workers.wait_for_complete();await pilot.pause()
   assert host.screen is screen
   if mode!='ordinary':assert not store.allowed(witness['generation'],'models.artifacts')
   assert not loads
   assert not blocked_attempts()
   return
  await wait_for(lambda:isinstance(host.screen,ConfirmationDialog) and host.screen.query('#confirm-button'))
  dialog=host.screen
  assert escape(str(model)) in dialog.message
  assert witness['generation'] in dialog.message
  assert 'Files:' in dialog.message
  assert not loads
  assert not store.allowed(witness['generation'],'models.artifacts')
  if mode=='changed':
   path=model/'config.json';path.write_bytes(path.read_bytes()+b' ')
  elif mode=='selection':os.environ['RAG_EMBEDDING_MODEL']=str(model.parent/'different-model')
  elif mode=='navigation':
   screen._select_category(SettingsCategoryId.OVERVIEW.value)
   screen._select_category(SettingsCategoryId.LIBRARY_RAG.value)
  dialog.query_one('#cancel-button' if mode=='cancel' else '#confirm-button',Button).press()
  if mode=='accepted-cancel':
   assert await asyncio.to_thread(entered.wait,5)
   from tldw_chatbook.Backup_Recovery.control_records import admission_authority
   from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
   from textual.worker import WorkerCancelled
   worker=next(w for w in host.workers if w.group=='settings-local-model-confirm')
   worker.cancel();await host.pop_screen()
   try:
    with admission_authority(bootstrap.default_bootstrap_root()).maintenance(tuple(witness['namespaces']),.03):
     raise AssertionError('cancelled confirmation lost native model source hold')
   except AdmissionTimeout:pass
   release.set()
   try:await worker.wait()
   except WorkerCancelled:pass
   else:raise AssertionError('confirmation waiter did not cancel')
  elif mode=='navigation':
   # Category navigation intentionally cancels its old index-status worker.
   await pilot.pause()
   for worker in tuple(host.workers):
    if worker.group.startswith('settings-local-model-'):await worker.wait()
  else:await host.workers.wait_for_complete()
  await pilot.pause()
  assert store.allowed(witness['generation'],'models.artifacts')==(mode in ('approve','accepted-cancel','rebuild'))
  unrelated_inactive()
  assert Path(entry.config).read_bytes()==before_config
  assert not loads
  assert index_contents()==before_index,'model review changed collection or stored record content'
  if mode not in ('approve','accepted-cancel','rebuild'):
   assert not blocked_attempts()
   return
  if mode=='rebuild':
   assert indexing.get_shared_rag_service() is None,'model approval enabled RAG before its own review'
   library._HuggingFaceEmbedder.__init__=original_load
   builds=[]
   async def observed_backfill(*args,**kwargs):
    assert kwargs.get('reconcile_for_recovery') is True
    result=await original_backfill(*args,**kwargs);builds.append(result);return result
   indexing.backfill_semantic_index=settings_module.backfill_semantic_index=observed_backfill
   projections.rebuild_projection=original_rebuild
   projections.reconcile_projection=original_reconcile
   screen.query_one('#settings-library-rag-recovery-review',Button).press()
   await wait_for(lambda:isinstance(host.screen,ConfirmationDialog) and host.screen.query('#confirm-button'))
   host.screen.query_one('#confirm-button',Button).press()
   await host.workers.wait_for_complete();await pilot.pause()
   assert all(store.allowed(witness['generation'],owner) for owner in ('rag.definitions','rag.projections','db.rag_indexing'))
   assert not store.allowed(witness['generation'],'config')
   assert not store.allowed(witness['generation'],'db.chachanotes.primary')
   screen.query_one('#settings-library-rag-recovery-reconcile',Button).press()
   await wait_for(lambda:isinstance(host.screen,ConfirmationDialog) and host.screen.query('#confirm-button'))
   host.screen.query_one('#confirm-button',Button).press()
   await pilot.pause()
   await host.workers.wait_for_complete();await pilot.pause()
   assert builds,host.messages
   assert builds[-1]['projection'].ready,builds[-1]
   service=indexing._shared_service
   assert service is not None
   results=await service.search('local model content',search_type='semantic',include_citations=False)
   assert any('local model content' in result.document for result in results),results
   assert not store.allowed(witness['generation'],'config')
   assert not store.allowed(witness['generation'],'skills')
   assert not store.allowed(witness['generation'],'mcp.local')
   service.close();media.close_connection();instance.chachanotes_db.close_connection()
   assert not blocked_attempts()
   return
 library._HuggingFaceEmbedder.__init__=original_load
 wrapper=EmbeddingsServiceWrapper(str(model),device='cpu')
 assert wrapper.create_embedding('local model content').shape==(16,)
 wrapper.close()
 assert not blocked_attempts()
asyncio.run(main())
print('verified isolated local model')
"""


def test_settings_reviews_actual_isolated_local_model_without_loading(tmp_path):
    _run_ui(tmp_path, "approve")


@pytest.mark.parametrize(
    "mode", ["cancel", "changed", "selection", "draft", "navigation", "late", "accepted-cancel", "unavailable"]
)
def test_settings_model_review_boundaries(tmp_path, mode):
    _run_ui(tmp_path, mode)


def _run_ui(tmp_path, mode):
    script = "CHILD=" + repr(_UI) + "\n" + _ISOLATED.replace(
        "for mode in ('approve','reopen'):", "for mode in (sys.argv[1],):"
    )
    _run(tmp_path, mode, "local", script=script, timeout=60)


def test_settings_separate_rag_review_and_explicit_rebuild(tmp_path):
    _run_ui(tmp_path, "rebuild")


def test_ordinary_settings_has_no_recovery_approval(tmp_path):
    ordinary = _TINY_MODEL + r"""
import sys
from types import SimpleNamespace
from tldw_chatbook import config as installed_config
from tldw_chatbook.RAG_Search import model_recovery as recovery
mode='ordinary'
entry=SimpleNamespace(config=str(selector),data=str(data))
def unrelated_inactive():pass
""" + _UI[_UI.index("import asyncio\nimport sqlite3"):] + "\nprint('retired and reopened')\n"
    _run(
        tmp_path, "ordinary", "local",
        script=ordinary.replace("print('verified isolated local model')", ""), timeout=45,
    )
