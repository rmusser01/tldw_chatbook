"""Installed RAG effects require local owner review of restored sources."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
route,state=sys.argv[1:]
selector=Path(os.environ['TLDW_CONFIG_PATH']);base=selector.parent.parent;data=base/'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'+str(data)+'"\n[rag.service]\nfirst_run_import_done=true\n')
selector.chmod(0o600)
os.environ.update(RAG_EMBEDDING_MODEL='mock',RAG_PERSIST_DIR=str(data/'vectors'),HF_HUB_CACHE=str(data/'models'))
from tldw_chatbook.Backup_Recovery import bootstrap,storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore,bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,register_pending
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified import rag_factory
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing
media=MediaDatabase(data/'media.db',client_id='activation-rag')
media.add_media_with_keywords(url='https://example.invalid/a',title='Source',media_type='document',content='Authoritative source content retained through recovery. '*5,overwrite=True)
tracking=RAGIndexingDB(data/'tracking.db')
config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(data/'vectors')},'search':{'media_db_path':str(media.db_path)},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}})
service=rag_factory.create_rag_service(config=config)
entry=indexing.media_index_entry(dict(media.execute_query('SELECT * FROM Media').fetchone()))
from dataclasses import replace
entry=replace(entry,source_path=media.db_path)
service.vector_store.close();media.close_connection();tracking.close()
root=bootstrap.default_bootstrap_root()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None: startup.close()
authority=admission_authority(root);authority.register('profile',(selector.parent,data))
control=base/'operation';control.mkdir(mode=0o700)
owners=('config','rag.definitions','rag.projections','db.rag_indexing','models.artifacts')
if state!='ordinary':
 register_pending(root,'restore',('profile',),control,(selector,))
 with authority.maintenance(('profile',),3) as session:
  bind_activation(root,'restore',selector,'generation',owners,session=session)
 (root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
 activation=ActivationStore(control/'activation')
 if state=='approved':
  from tldw_chatbook.RAG_Search.activation import preview_recovery_review,approve_recovery_review
  approve_recovery_review(config,preview_recovery_review(config).fingerprint)
  assert not activation.allowed('generation','config')
  assert not activation.allowed('generation','models.artifacts')
  if route=='reranker_timeout':
   # Synthetic Event-based lifetime evidence only; not recovered model approval.
   activation.approve('generation','config')
   activation.approve('generation','models.artifacts')
 if state in ('config_only','shared'):
  activation.approve('generation','config')
 if state=='missing': (activation._generation('generation')/'required.json').unlink()
 if state=='shared':
  other=base/'unrelated.toml';other.write_text('[general]\n');other.chmod(0o600)
  os.environ['TLDW_CONFIG_PATH']=str(other)
if route=='hf_constructor': config.embedding.model='sentence-transformers/all-MiniLM-L6-v2'
denied=state not in ('ordinary','approved')
effects=[]
original_adopt=rag_factory.maybe_adopt_legacy_collection
def adopt(*args): effects.append('legacy-adoption');return original_adopt(*args)
rag_factory.maybe_adopt_legacy_collection=adopt
original_dimension=RAGService._get_embedding_dimension
def dimension(self): effects.append('model-dimension');return original_dimension(self)
RAGService._get_embedding_dimension=dimension
async def run():
 if route=='settings_review':
  from types import SimpleNamespace
  from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
  from tldw_chatbook.RAG_Search.activation import preview_recovery_review
  config.embedding.base_url='https://user:secret@example.invalid/provider?api_key=secret'
  review=preview_recovery_review(config)
  screens,actions=[],[]
  app=SimpleNamespace(push_screen=lambda screen,callback: screens.append((screen,callback)))
  owner=SimpleNamespace(app=app,_rag_recovery_review_worker=lambda fingerprint: actions.append(('approve',fingerprint)),_trigger_library_rag_index_backfill=lambda **kw: actions.append(('reconcile',kw)))
  SettingsScreen._show_rag_recovery_review(owner,review)
  screen,callback=screens.pop()
  assert 'secret' not in screen.message
  callback(False);assert not actions
  callback(True);assert actions==[('approve',review.fingerprint)]
  SettingsScreen.handle_library_rag_recovery_reconcile(owner,SimpleNamespace(stop=lambda:None))
  screen,callback=screens.pop();callback(False);assert len(actions)==1
  callback(True);assert actions[-1]==('reconcile',{'reconcile_for_recovery':True})
  assert not activation.allowed('generation','rag.projections')
  return True
 if route in ('review','review_changed'):
  import json
  from tldw_chatbook.RAG_Search.activation import preview_recovery_review,approve_recovery_review
  config.embedding.base_url='https://user:secret@example.invalid/provider?api_key=secret'
  review=preview_recovery_review(config)
  assert 'secret' not in json.dumps(review.to_dict())
  assert review.provider_host=='example.invalid'
  assert set(review.owners)==set(owners)-{'config','models.artifacts'}
  if route=='review_changed':
   selector.write_text(selector.read_text()+'\n# changed after displayed review\n')
   try: approve_recovery_review(config,review.fingerprint)
   except PermissionError as error: assert str(error)=='rag_recovery_review_changed'
   else: raise AssertionError('changed selector approved from stale preview')
  else:
   approve_recovery_review(config,review.fingerprint)
   assert all(activation.allowed('generation',owner) for owner in review.owners)
  assert not activation.allowed('generation','config')
  assert not activation.allowed('generation','models.artifacts')
  return True
 if route=='reranker_timeout':
  import threading,time
  from types import SimpleNamespace
  from tldw_chatbook.RAG_Search.reranker import CrossEncoderReranker,RerankingConfig
  reranker=CrossEncoderReranker(RerankingConfig(strategy='cross_encoder',timeout_seconds=.01))
  entered,release=threading.Event(),threading.Event()
  def predict(query,rows):
   entered.set();assert release.wait(5);return [.5]*len(rows)
  reranker._predict_scores_sync=predict
  task=asyncio.create_task(reranker.rerank('query',[SimpleNamespace(id='row',document='content',score=.5)]))
  while not entered.is_set():
   if task.done(): task.result();raise AssertionError('native reranker did not enter')
   await asyncio.sleep(.001)
  pause=storage._begin_local_pause()
  try:
   await asyncio.sleep(.03)
   assert not task.done(),'timeout released caller before native completion'
   assert not pause.drain(time.monotonic())
   release.set();outcome=await task
   assert outcome.failed==1
  finally: release.set();pause.resume()
  return True
 if route=='vector_constructor':
  from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
  return ChromaVectorStore(data/'newvectors')
 if route=='vector_collection': return service.vector_store.collection
 if route=='vector':
  service.vector_store.clear();return True
 if route=='reranker_local':
  from tldw_chatbook.RAG_Search.reranker import CrossEncoderReranker,RerankingConfig
  from huggingface_hub import constants
  constants.HF_HUB_CACHE=str(base/'ordinary-cache')
  model=data/'local-model';model.mkdir(exist_ok=True)
  from types import SimpleNamespace
  reranker=CrossEncoderReranker(RerankingConfig(model_name=str(model),strategy='cross_encoder'))
  def forbidden_predict(*args):
   effects.append('local-model-load');raise AssertionError('unreviewed local model execution')
  reranker._predict_scores_sync=forbidden_predict
  return await reranker.rerank('query',[SimpleNamespace(id='row',document='content',score=.5)])
 if route=='reranker':
  from tldw_chatbook.RAG_Search.reranker import PointwiseReranker,RerankingConfig
  return await PointwiseReranker(RerankingConfig()).rerank('query',[])
 if route in ('nested_pause','nested_cancel','nested_backfill'):
  import time
  service.vector_store.collection.count()
  entered,release=asyncio.Event(),asyncio.Event()
  original_chunk=service._chunk_document
  async def paused_chunk(*args,**kwargs):
   result=await original_chunk(*args,**kwargs)
   entered.set();await release.wait();return result
  service._chunk_document=paused_chunk
  call=(indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True) if route=='nested_backfill' else service.index_document('nested','Real nested native indexing content. '*20))
  work=asyncio.create_task(call)
  await asyncio.wait_for(entered.wait(),5)
  pause=storage._begin_local_pause()
  try:
   assert not pause.drain(time.monotonic())
   if route=='nested_cancel':
    work.cancel();await asyncio.sleep(.02);assert not work.done()
   release.set()
   try: result=await work
   except asyncio.CancelledError: assert route=='nested_cancel'
   else:
    if route=='nested_backfill':
     assert result['status']=='partial' and not result['projection'].ready,result
     assert result['errors']==['projection_owner_observation_unavailable'],result
    else: assert result.success,result
  finally: pause.resume()
  assert service.vector_store.collection.count()>0
  return True
 if route=='cli_review':
  from tldw_chatbook.RAG_Search.backfill import main
  return main(['--review-recovery'])
 if route=='factory': return rag_factory.create_rag_service(config=config)
 if route=='constructor': return RAGService(config)
 if route=='hf_constructor': return rag_factory.create_rag_service(config=config)
 if route=='shared_factory': return indexing.get_shared_rag_service()
 if route=='embedding': return await service.embeddings.create_embeddings_async(['source'])
 if route=='backfill': return await indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True)
 if route=='queue':
  worker=indexing.IngestionIndexer(rag_service=service,indexing_db=tracking)
  accepted=worker.submit(entry)
  if accepted: assert worker.wait_until_idle(10)
  worker.stop()
  return accepted
 raise AssertionError(route)
try: result=asyncio.run(run())
except (PermissionError,RuntimeError,ValueError) as error:
 assert denied or route=='hf_constructor',(route,state,error)
 assert 'activation' in str(error) or str(error)=='rag_model_setup_required',str(error)
else:
 if denied:
  assert result is None or result is False or route=='cli_review' and result==0 or route in ('review','review_changed','settings_review') and result is True,(route,state,result)
 else:
  assert route!='hf_constructor','restored constructor acquired an unqualified HF model'
  if route=='backfill': assert result['projection'].ready,result
  elif route=='queue': assert result
  else: assert result is not None
if denied: assert not effects,effects
if route=='hf_constructor': assert not effects,effects
if state=='approved':assert activation.allowed('generation','config')==(route=='reranker_timeout')
assert media.execute_query('SELECT content FROM Media').fetchone()[0].startswith('Authoritative')
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route,state",
    [
        ("factory", "denied"),
        ("constructor", "denied"),
        ("shared_factory", "denied"),
        ("embedding", "denied"),
        ("backfill", "denied"),
        ("queue", "denied"),
        ("factory", "config_only"),
        ("factory", "missing"),
        ("factory", "shared"),
        ("factory", "ordinary"),
        ("factory", "approved"),
        ("backfill", "approved"),
        ("hf_constructor", "approved"),
        ("cli_review", "denied"),
        ("review", "denied"),
        ("review_changed", "denied"),
        ("settings_review", "denied"),
        ("vector", "denied"),
        ("vector_constructor", "shared"),
        ("vector_collection", "denied"),
        ("reranker", "denied"),
        ("reranker_local", "shared"),
        ("reranker_timeout", "approved"),
        ("queue", "ordinary"),
        ("nested_pause", "approved"),
        ("nested_cancel", "approved"),
        ("nested_backfill", "approved"),
    ],
)
def test_rag_activation_before_actual_effects(tmp_path, route, state):
    _run(tmp_path, route, state, script=_SCRIPT)


def test_actual_app_drains_backfill_before_core_pause(tmp_path):
    from Tests.Backup_Recovery.test_runtime_startup_handoff import (
        _SCRIPT as runtime_script,
    )

    setup = r"""
    from pathlib import Path
    from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    from tldw_chatbook.RAG_Search.simplified.rag_factory import create_rag_service
    from tldw_chatbook.RAG_Search.ingestion_indexing import backfill_semantic_index, suppress_ingestion_indexing
    from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant as rag_lifetime
    media=app.media_db
    with suppress_ingestion_indexing():
        media.add_media_with_keywords(url='https://example.invalid/accepted-backfill',title='Accepted',media_type='document',content='Accepted Backfill must finish tracking before core pause. '*8,overwrite=True)
    tracking=RAGIndexingDB(Path(media.db_path).parent/'activation-indexing.db')
    config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(Path(media.db_path).parent/'activation-vectors')},'search':{'media_db_path':str(media.db_path),'chachanotes_db_path':str(app.chachanotes_db.db_path)},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}})
    rag=create_rag_service(config=config)
    rag_entered,rag_release=asyncio.Event(),asyncio.Event()
    original_chunk=rag._chunk_document
    async def blocked_chunk(*args,**kwargs):
        chunks=await original_chunk(*args,**kwargs)
        rag_entered.set();await rag_release.wait();return chunks
    rag._chunk_document=blocked_chunk
    backfill=asyncio.create_task(backfill_semantic_index(media_db=media,rag_service=rag,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True))
    await asyncio.wait_for(rag_entered.wait(),5)
    async def release_accepted_backfill():
        while not rag_lifetime._closed:
            await asyncio.sleep(.001)
        assert storage._pause is None
        assert not backfill.done()
        rag_release.set()
        result=await backfill
        assert result['status']=='ok' and result['projection'].ready,result
        assert tracking.get_indexed_items_by_type('media')
    finish_backfill=asyncio.create_task(release_accepted_backfill())
"""
    script = (
        runtime_script.replace(
            "    runtime = RuntimeMaintenance(app)",
            setup + "\n    runtime = RuntimeMaintenance(app)",
        )
        .replace(
            "        runtime.retire_local_caches()",
            "        await finish_backfill\n        assert rag.vector_store._client is None\n        runtime.retire_local_caches()",
        )
        .replace(
            "        assert not errors",
            "        assert tracking.get_indexed_items_by_type('media')\n        assert not errors",
        )
    )
    _run(tmp_path, "startup", "resume", script=script)


_LAZY_TRACKING = r"""
import asyncio, os, sqlite3, sys
from dataclasses import replace
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
route,state=sys.argv[1:]
selector=Path(os.environ['TLDW_CONFIG_PATH']);base=selector.parent.parent;data=base/'data'
protected=base/'restored'/'tracking.db';protected.parent.mkdir(mode=0o700)
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'+str(data)+'"\n[database]\nrag_indexing_db_path="'+str(protected)+'"\n[rag.service]\nfirst_run_import_done=true\n');selector.chmod(0o600)
os.environ.update(RAG_EMBEDDING_MODEL='mock',RAG_PERSIST_DIR=str(data/'vectors'))
from tldw_chatbook.Backup_Recovery import bootstrap,storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore,bind_activation,execution_allowed
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,register_pending
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_factory import create_rag_service
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing
from tldw_chatbook.RAG_Search.activation import RAGActivationRequired,preview_recovery_review,approve_recovery_review
from tldw_chatbook.config import get_rag_indexing_db_path
tracking=RAGIndexingDB(protected);tracking.close()
with sqlite3.connect(protected) as conn: assert conn.execute('PRAGMA journal_mode=DELETE').fetchone()[0]=='delete'
before=protected.read_bytes()
media=MediaDatabase(data/'media.db',client_id='lazy-tracking')
with indexing.suppress_ingestion_indexing():
 media.add_media_with_keywords(url='https://example.invalid/lazy',title='Lazy tracking',media_type='document',content='Lazy tracking owner must be approved before native opening. '*8,overwrite=True)
entry=replace(indexing.media_index_entry(dict(media.execute_query('SELECT * FROM Media').fetchone())),source_path=media.db_path)
config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(data/'vectors')},'search':{'media_db_path':str(media.db_path)},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}})
service=create_rag_service(config=config)
service.vector_store.close();media.close_connection()
root=bootstrap.default_bootstrap_root();startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup: startup.close()
if state!='ordinary':
 restored_selector=base/'restored.toml';restored_selector.write_text('[general]\n');restored_selector.chmod(0o600)
 authority=admission_authority(root);authority.register('restoredtracking',(restored_selector,protected))
 control=base/'operation';control.mkdir(mode=0o700)
 register_pending(root,'restore',('restoredtracking',),control,(restored_selector,))
 with authority.maintenance(('restoredtracking',),3) as session:
  bind_activation(root,'restore',restored_selector,'generation',('config','rag.definitions','rag.projections','db.rag_indexing'),session=session)
 (root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
 activation=ActivationStore(control/'activation')
 assert not execution_allowed(('db.rag_indexing',),protected)
 if state=='approved':
  # A configured external source also needs the existing native profile
  # enrollment. Owner approval does not broaden an unbound storage group.
  from tldw_chatbook.Backup_Recovery.control_records import bind_profile
  authority.register('ordinary',(selector.parent,data))
  bind_profile(root,selector,('ordinary','restoredtracking'),root/'admission')
  review=preview_recovery_review(config)
  assert str(protected) in review.sources
  assert set(review.owners)=={'rag.definitions','rag.projections','db.rag_indexing'}
  assert 'config_review_required' not in review.prerequisites
  approve_recovery_review(config,review.fingerprint)
  assert activation.allowed('generation','db.rag_indexing')
  assert not activation.allowed('generation','config')
  assert indexing._default_indexing_db() is not None
  assert not activation.allowed('generation','config')
assert get_rag_indexing_db_path()==protected
denied=state=='denied'
if route=='backfill':
 try: result=asyncio.run(indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=None,item_types=('media',),reconcile_for_recovery=True))
 except RAGActivationRequired: assert denied
 else: assert not denied and result['status']=='ok' and result['projection'].ready,result
else:
 worker=indexing.IngestionIndexer(rag_service=service)
 try:
  for attempt in range(2 if denied else 1):
   assert worker.submit(entry)
   assert worker.wait_until_idle(10)
  stats=worker.stats()
  if denied:
   assert stats['failed']==2 and stats['indexed']==0,stats
   assert not worker._indexing_db_resolved
  else: assert stats['indexed']==1 and stats['failed']==0,stats
 finally: worker.stop()
with sqlite3.connect(protected) as conn: mode=conn.execute('PRAGMA journal_mode').fetchone()[0]
if denied:
 assert mode=='delete',mode
 assert protected.read_bytes()==before
 assert not protected.with_name(protected.name+'-wal').exists()
 assert service.vector_store.collection.count()==0
else:
 assert mode=='wal',mode
 assert RAGIndexingDB(protected).get_indexed_items_by_type('media')
 assert service.vector_store.collection.count()>0
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["backfill", "queue"])
@pytest.mark.parametrize("state", ["denied", "ordinary", "approved"])
def test_lazy_tracking_owner_before_native_open(tmp_path, route, state):
    _run(tmp_path, route, state, script=_LAZY_TRACKING)
