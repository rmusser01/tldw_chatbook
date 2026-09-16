"""Actual shared factory, native Chroma and live source recovery qualification."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import asyncio, json, os, subprocess, sys
from pathlib import Path
from Tests.network_guard import install
install()
selector = Path(os.environ['TLDW_CONFIG_PATH'])
base = selector.parent.parent
data = base / 'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'+str(data)+'"\n[rag.service]\nprofile="hybrid_basic"\nfirst_run_import_done=true\n')
selector.chmod(0o600)
os.environ.update(RAG_EMBEDDING_MODEL='mock', RAG_PERSIST_DIR=str(data/'vectors'), RAG_CHUNK_SIZE='400', RAG_CHUNK_OVERLAP='0')
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing, recovery
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import _DeterministicEmbeddingFactory
media = MediaDatabase(data/'media.db', client_id='enhanced-recovery')
media.add_media_with_keywords(url='https://example.invalid/a',title='Source',media_type='document',content='Original authoritative searchable source text for actual enhanced recovery. '*5,keywords=[],overwrite=True)
tracking = RAGIndexingDB(data/'tracking.db')
service = indexing.get_shared_rag_service()
assert type(service) is EnhancedRAGServiceV2
assert type(service.embeddings.factory) is _DeterministicEmbeddingFactory
assert not service.enable_parent_retrieval
assert indexing.semantic_indexing_available()
assert indexing.get_shared_rag_service() is service
route = sys.argv[1]

async def build():
 return await indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True)
async def verify():
 return await recovery.reconcile_projection(service,media_db=media,item_types=('media',))
async def search():
 return await service.search('authoritative source',search_type='semantic',include_citations=False,score_threshold=-1)

if route == 'ordinary':
 first = asyncio.run(indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',)))
 second = asyncio.run(indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',)))
 assert first['indexed']==1 and second['skipped']==1
 assert asyncio.run(search())
 assert not asyncio.run(verify()).ready
 before = service.vector_store.recovery_snapshot()
 refused = asyncio.run(build())
 assert 'projection_legacy_source_scope_required' in refused['errors'],refused
 assert service.vector_store.recovery_snapshot()==before
 print('retired and reopened');sys.exit(0)

# Prepare only the lightweight unloaded wrapper in ordinary setup. Creating
# a new model wrapper after restoration separately requires model-owner review.
if route == 'unloaded_hf':
 from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import EmbeddingsServiceWrapper
 unloaded_embeddings = EmbeddingsServiceWrapper(model_name='sentence-transformers/all-MiniLM-L6-v2',device='cpu',cache_dir=str(data/'models'))
 assert not unloaded_embeddings.factory._cache

# Open the actual service/model before selecting the new local generation.
# This test does not qualify automatic model acquisition during construction.
service.vector_store.close(); media.close_connection(); tracking.close()
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None: startup.close()
authority = admission_authority(root)
authority.register('profile',(selector.parent,data))
control = base/'operation';control.mkdir(mode=0o700)
register_pending(root,'restore',('profile',),control,(selector,))
with authority.maintenance(('profile',),3) as session:
 bind_activation(root,'restore',selector,'generation',('config','rag.definitions','rag.projections','db.rag_indexing'),session=session)
(root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
for owner in ('config','rag.definitions','rag.projections','db.rag_indexing'): ActivationStore(control/'activation').approve('generation',owner)
try:
 asyncio.run(search())
except (RuntimeError,ValueError) as error:
 assert 'projection_' in str(error)
else: raise AssertionError('enhanced query returned before recovery reconciliation')

if route == 'parent':
 service.enable_parent_retrieval = True
 refused = asyncio.run(build())
 assert 'projection_parent_pipeline_unavailable' in refused['errors'],refused
 print('retired and reopened');sys.exit(0)

built = asyncio.run(build())
assert built['status']=='ok',built
assert built['projection'].ready,built
assert asyncio.run(search())
assert asyncio.run(service.search_with_context_expansion('source',search_type='semantic',include_citations=False,score_threshold=-1))
before = service.vector_store.recovery_snapshot()
assert len(before['rows']) >= 1

if route == 'fresh':
 service.close();media.close_connection();tracking.close()
 child = r"""
import asyncio, sys
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing, recovery
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
media=MediaDatabase(Path(sys.argv[1]),client_id='enhanced-reopen')
service=indexing.get_shared_rag_service()
assert type(service) is EnhancedRAGServiceV2
def forbidden(*args,**kwargs): raise AssertionError('automatic indexing or re-embedding')
service.index_batch_optimized=forbidden
service.embeddings.create_embeddings_async=forbidden
before=service.vector_store.recovery_snapshot()
observation=asyncio.run(recovery.reconcile_projection(service,media_db=media,item_types=('media',)))
assert observation.ready,observation
assert service.vector_store.recovery_snapshot()==before
# Queries may embed the query after proof; reconciliation itself may not.
del service.embeddings.create_embeddings_async
assert asyncio.run(service.search('source',search_type='semantic',include_citations=False,score_threshold=-1))
print('durable enhanced reopen')
"""
 result=subprocess.run([sys.executable,'-c',child,str(media.db_path)],capture_output=True,text=True,timeout=25)
 assert result.returncode==0,result.stderr[-5000:]
 assert 'durable enhanced reopen' in result.stdout
elif route == 'changed': media.execute_query('UPDATE Media SET content=?,version=version+1',('Changed contents retaining equal IDs and timestamps.',))
elif route == 'deleted': media.execute_query('UPDATE Media SET deleted=1,version=version+1')
elif route == 'target_only': service.vector_store.add(['media_999_chunk_0'],[[1.]*384],['Target only'],[{'doc_id':'media_999'}])
elif route == 'config': service.config.chunking.chunk_size += 1
elif route == 'profile_switch': service.switch_profile('hybrid_enhanced')
elif route == 'parent_config': service.config.chunking.enable_parent_retrieval = True
elif route == 'parent_switch': service.enable_parent_retrieval = True
elif route == 'unknown_model': service.embeddings.factory = object()
elif route == 'unloaded_hf':
 service.embeddings = unloaded_embeddings
 assert not service.embeddings.factory._cache
 assert 'projection_model_identity_unavailable' in asyncio.run(verify()).issues
 assert not service.embeddings.factory._cache
elif route == 'parent_write':
 result=asyncio.run(service.index_document_with_parents('parent-source','Parent source text requiring separate unqualified indexing. '*30,use_structural_chunking=False))
 # Installed parent API writes its rows but cannot construct IndexingResult
 # with metadata. Even this partial write must invalidate existing proof.
 assert not result.success and result.error,result
 assert service.vector_store.recovery_snapshot()!=before
elif route == 'other_owner':
 other=MediaDatabase(data/'other.db',client_id='other-owner')
 other.add_media_with_keywords(url='https://example.invalid/a',title='Source',media_type='document',content='Original authoritative searchable source text for actual enhanced recovery. '*5,keywords=[],overwrite=True)
 assert not asyncio.run(recovery.reconcile_projection(service,media_db=other,item_types=('media',))).ready
 refused=asyncio.run(indexing.backfill_semantic_index(media_db=other,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True))
 assert 'projection_legacy_source_scope_required' in refused['errors'],refused
 assert service.vector_store.recovery_snapshot()==before
 # Refusing a different source cannot invalidate the still-valid original
 # owner's durable proof; explicit recheck of that owner can reopen it.
 assert asyncio.run(verify()).ready
elif route == 'subclass':
 class Unqualified(EnhancedRAGServiceV2): pass
 service.__class__=Unqualified
elif route != 'positive': raise AssertionError(route)

if route not in ('positive','fresh','other_owner'):
 assert not asyncio.run(verify()).ready
 try: asyncio.run(search())
 except (RuntimeError,ValueError) as error: assert 'projection_' in str(error)
 else: raise AssertionError('enhanced query retained stale readiness')
 if route in ('changed','deleted'):
  repaired=asyncio.run(build())
  assert repaired['projection'].ready,repaired
  if route=='deleted': assert service.vector_store.collection.count()==0
 if route in ('target_only','parent_write'):
  current=service.vector_store.recovery_snapshot()
  refused=asyncio.run(build())
  assert 'projection_shared_source_scope_required' in refused['errors'],refused
  assert service.vector_store.recovery_snapshot()==current
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "route",
    [
        "positive",
        "fresh",
        "ordinary",
        "changed",
        "deleted",
        "target_only",
        "config",
        "profile_switch",
        "parent",
        "parent_config",
        "parent_switch",
        "parent_write",
        "other_owner",
        "unknown_model",
        "unloaded_hf",
        "subclass",
    ],
)
def test_actual_enhanced_factory_recovery(tmp_path, route):
    _run(tmp_path, route, "enhanced", script=_SCRIPT)
