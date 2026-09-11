"""Restored native projection queries require current local owner readiness."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import asyncio, os, sys, json, subprocess
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing
from tldw_chatbook.RAG_Search import recovery
selector=Path(os.environ['TLDW_CONFIG_PATH'])
base=selector.parent.parent
data=base/'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'+str(data)+'"\n')
selector.chmod(0o600)
media=MediaDatabase(data/'media.db',client_id='generation')
media.add_media_with_keywords(url='https://example.invalid/a',title='Source',media_type='document',content='Original authoritative searchable source text for recovery.',keywords=['fixture'],overwrite=True)
tracking=RAGIndexingDB(data/'indexing.db')
config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(data/'vectors')},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}})
config.search.media_db_path=str(media.db_path)
config.search.chachanotes_db_path=str(data/'notes.db')
config.search.prompts_db_path=str(data/'prompts.db')
service=RAGService(config)
indexing.semantic_indexing_available=lambda:True
async def build():
 return await indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True)
assert asyncio.run(build())['status']=='ok'
assert asyncio.run(service.search('source',include_citations=False))
second=RAGService(config)
assert asyncio.run(second.search('source',include_citations=False))
second.vector_store.close()
if sys.argv[1] in ('enhanced','expansion'):
 from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
 enhanced=EnhancedRAGServiceV2(config,enable_parent_retrieval=False,enable_reranking=False,enable_parallel_processing=False)
service.vector_store.close()
media.close_connection()
tracking.close()
root=bootstrap.default_bootstrap_root()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None: startup.close()
authority=admission_authority(root)
authority.register('profile',(selector.parent,data))
control=base/'operation';control.mkdir(mode=0o700)
register_pending(root,'restore',('profile',),control,(selector,))
with authority.maintenance(('profile',),3) as session:
 bind_activation(root,'restore',selector,'generation',('config',),session=session)
(root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
ActivationStore(control/'activation').approve('generation','config')
route=sys.argv[1]
if route=='independent':
 other=base/'unrelated.toml';other.write_text('[general]\n');other.chmod(0o600)
 os.environ['TLDW_CONFIG_PATH']=str(other)
if route=='unpaired':
 association=next(root.glob('activation-*.json'))
 association.unlink()
if route=='imported':
 service.vector_store.collection.modify(metadata={'ready':True,'generation':'generation','verified':True})
if route=='unsupported':
 service.embeddings.factory=object()
if route=='disjoint':
 ordinary=base/'ordinary';ordinary.mkdir(mode=0o700)
 other=ordinary/'config.toml';other.write_text('[general]\n');other.chmod(0o600)
 os.environ['TLDW_CONFIG_PATH']=str(other)
 ordinary_media=MediaDatabase(ordinary/'media.db',client_id='ordinary')
 ordinary_media.add_media_with_keywords(url='https://example.invalid/b',title='Independent',media_type='document',content='Independent ordinary source documents remain searchable.',keywords=['fixture'],overwrite=True)
 ordinary_config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(ordinary/'vectors')},'search':{'media_db_path':str(ordinary/'media.db'),'chachanotes_db_path':str(ordinary/'notes.db'),'prompts_db_path':str(ordinary/'prompts.db')},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}})
 ordinary_service=RAGService(ordinary_config)
 ordinary_service.vector_store.add(['ordinary_chunk_0'],[[1.]*384],['Independent ordinary source documents remain searchable.'],[{'doc_id':'ordinary'}])
 assert asyncio.run(ordinary_service.search('Independent',include_citations=False,score_threshold=-1))
 print('retired and reopened');sys.exit(0)
if route=='bare': del service.vector_store._projection_service
if route in ('enhanced','expansion'): service=enhanced
if route in ('uncached','enhanced','expansion'):
 async def unexpected_embedding(*args,**kwargs): raise AssertionError('unverified query invoked embedding')
 service.embeddings.create_embeddings_async=unexpected_embedding
positive_routes={'approved','changed','deleted','trashed','target_only','missing','corrupt','wrong_generation','model','config','new_generation','relaunch'}
if route in positive_routes:
 verified=asyncio.run(recovery.reconcile_projection(service,media_db=media,item_types=('media',)))
 assert verified.ready,verified
 assert asyncio.run(service.search('source',include_citations=False))
 assert service.vector_store.search([1.]*384)
 stored=service.vector_store.collection.get(include=['embeddings'])['embeddings'][0]
 assert service.vector_store.search_with_citations([float(v)+.03 for v in stored],'source')
 directory=ActivationStore(control/'activation')._generation('generation')
 ready=next(directory.glob('projection-*.json'))
 record=json.loads(ready.read_text())
 assert type(record['build']['model']) is str
 if route=='approved':
  print('retired and reopened');sys.exit(0)
 if route=='changed':
  media.execute_query('UPDATE Media SET content=?,version=version+1',('Changed authoritative contents despite equal IDs.',))
 if route=='deleted': media.execute_query('DELETE FROM Media')
 if route=='trashed': media.execute_query('UPDATE Media SET is_trash=1,version=version+1')
 if route=='target_only':
  service.vector_store.add(['media_999_chunk_0'],[[1.]*384],['target-only'],[{'doc_id':'media_999'}])
 if route=='missing': ready.unlink()
 if route=='corrupt': ready.write_text('{')
 if route=='wrong_generation':
  record['witnesses'][0]['generation']='imported-old-generation';ready.write_text(json.dumps(record))
 if route=='model': service.embeddings.factory=object()
 if route=='config': service.config.chunking.chunk_size+=1
 if route in ('new_generation','relaunch'):
  service.vector_store.close();media.close_connection();tracking.close()
  startup=storage._startups.pop((os.getpid(),str(root)),None)
  if startup is not None: startup.close()
 if route=='new_generation':
  control2=base/'operation2';control2.mkdir(mode=0o700)
  register_pending(root,'restore-again',('profile',),control2,(selector,))
  with authority.maintenance(('profile',),3) as session:
   bind_activation(root,'restore-again',selector,'generation-new',('config',),session=session)
  (root/('pending-'+bootstrap._key('restore-again')+'.json')).unlink()
  ActivationStore(control2/'activation').approve('generation-new','config')
 if route=='relaunch':
  child="""import asyncio, json, sys
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search import recovery
service=RAGService(RAGConfig.from_dict(json.loads(sys.argv[1])))
media=MediaDatabase(Path(sys.argv[2]),client_id='reopened-owner')
assert not hasattr(service,'_recovery_projection_build')
async def forbidden(*args,**kwargs): raise AssertionError('automatic embedding/rebuild')
service.index_batch_optimized=forbidden
observation=asyncio.run(recovery.reconcile_projection(service,media_db=media,item_types=('media',)))
assert observation.ready,observation
assert service.vector_store.search([1.]*384)
assert asyncio.run(service.search('source',include_citations=False))
print('reopened from durable proof')
"""
  result=subprocess.run([sys.executable,'-c',child,json.dumps(config.to_dict(),default=str),str(media.db_path)],capture_output=True,text=True,timeout=20)
  assert result.returncode==0,result.stderr
  assert 'reopened from durable proof' in result.stdout
  print('retired and reopened');sys.exit(0)
try:
 if route in ('direct','bare') or route in positive_routes: service.vector_store.search([1.]*384)
 elif route=='expansion': asyncio.run(service.search_with_context_expansion('source',include_citations=False))
 else: asyncio.run(service.search('uncached' if route=='uncached' else 'source',include_citations=False))
except (RuntimeError,ValueError) as error:
 assert 'projection_' in str(error),str(error)
else:
 raise AssertionError('restored projection returned results without local generation readiness')
assert not service.cache._cache
assert not second.cache._cache
assert media.execute_query('SELECT COUNT(*) FROM Media').fetchone()[0]>=0
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "route",
    [
        "service",
        "direct",
        "approved",
        "changed",
        "deleted",
        "trashed",
        "target_only",
        "missing",
        "corrupt",
        "wrong_generation",
        "model",
        "config",
        "new_generation",
        "relaunch",
        "independent",
        "unpaired",
        "imported",
        "unsupported",
        "bare",
        "uncached",
        "disjoint",
        "enhanced",
        "expansion",
    ],
)
def test_restored_projection_refuses_unverified_queries(tmp_path, route):
    _run(tmp_path, route, "owner", script=_SCRIPT)


def test_ordinary_ingestion_enrolls_independent_source(tmp_path):
    script = _SCRIPT.replace(
        "reconcile_for_recovery=True", "reconcile_for_recovery=False"
    )
    script = script.replace(
        "authority.register('profile',(selector.parent,data))",
        "authority.register('profile',(selector.parent,media.db_path))",
    )
    script = script.replace(
        "service=RAGService(config)",
        "config.search.media_db_path=str(base/'declared-media.db')\nconfig.search.chachanotes_db_path=str(base/'declared-notes.db')\nservice=RAGService(config)",
        1,
    )
    script = script.replace(
        "assert not second.cache._cache",
        """assert not second.cache._cache
try:
 asyncio.run(second.search('source',include_citations=False))
except (ValueError,RuntimeError) as error:
 assert 'projection_' in str(error)
else:
 raise AssertionError('second service lost actual collection source ownership')""",
    )
    _run(tmp_path, "independent", "owner", script=script)


@pytest.mark.parametrize("backend", ["memory", "chroma", "v2"])
def test_existing_ingestion_round_trip_under_fixed_private_config(tmp_path, backend):
    script = r"""
from pathlib import Path
import sys
from Tests.network_guard import install
install()
import pytest
from Tests.RAG.test_ingestion_indexing import TestEndToEndSemanticSearch
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
root=Path.home()
media=MediaDatabase(root/'media.db',client_id='generation-compatibility')
with pytest.MonkeyPatch.context() as patch:
 case=TestEndToEndSemanticSearch()
 if sys.argv[1]=='memory':
  case.test_semantic_search_returns_newly_ingested_document(media,root,patch)
 elif sys.argv[1]=='v2':
  case.test_v2_service_with_parallel_profile_indexes_and_searches(media,root,patch)
 else:
  case.test_chroma_round_trip_persists_ingested_document(media,root,patch)
print('retired and reopened')
"""
    _run(tmp_path, backend, "owner", script=script)


def test_installed_v2_reranking_retains_accepted_query(tmp_path):
    script = r"""
import asyncio,time
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install
install()
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(Path.home()/'vectors')}})
service=EnhancedRAGServiceV2(config,enable_parent_retrieval=False,enable_reranking=False,enable_parallel_processing=False)
service.vector_store.add(['a','b'],[[1.]*384,[.5]*384],['source a','source b'],[{'doc_id':'a'},{'doc_id':'b'}])
async def run():
 entered,release=asyncio.Event(),asyncio.Event()
 async def rerank(query,results):
  entered.set();await release.wait()
  return SimpleNamespace(results=results,degraded=False)
 service.reranker=SimpleNamespace(rerank=rerank)
 task=asyncio.create_task(service.search('source',include_citations=False,score_threshold=-1,rerank=True))
 try:
  await asyncio.wait_for(entered.wait(),5)
  participant._maintenance_close_admission()
  assert not await participant._maintenance_drain(time.monotonic()+.03),'installed V2 query escaped native accepted lifetime during reranking'
 finally:
  release.set()
  await task
  await participant._maintenance_resume()
 service.close()
asyncio.run(run())
print('retired and reopened')
"""
    _run(tmp_path, "v2", "owner", script=script)


_CORRECTION_RELAUNCH = r'''

import asyncio, os, sys, json, subprocess
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing
from tldw_chatbook.RAG_Search import recovery
selector=Path(os.environ['TLDW_CONFIG_PATH'])
base=selector.parent.parent
data=base/'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'+str(data)+'"\n')
selector.chmod(0o600)
media=MediaDatabase(data/'media.db',client_id='generation')
media.add_media_with_keywords(url='https://example.invalid/a',title='Source',media_type='document',content='Original authoritative searchable source text for recovery.',keywords=['fixture'],overwrite=True)
tracking=RAGIndexingDB(data/'indexing.db')
config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(data/'vectors')},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}})
config.search.media_db_path=str(base/'declared-media.db')
config.search.chachanotes_db_path=str(base/'declared-notes.db')
service=RAGService(config)
indexing.semantic_indexing_available=lambda:True
async def build():
 return await indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=False)
assert asyncio.run(build())['status']=='ok'
assert asyncio.run(service.search('source',include_citations=False))
second=RAGService(config)
assert asyncio.run(second.search('source',include_citations=False))
second.vector_store.close()
if sys.argv[1] in ('enhanced','expansion'):
 from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
 enhanced=EnhancedRAGServiceV2(config,enable_parent_retrieval=False,enable_reranking=False,enable_parallel_processing=False)
service.vector_store.close()
media.close_connection()
tracking.close()
root=bootstrap.default_bootstrap_root()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None: startup.close()
authority=admission_authority(root)
authority.register('profile',(selector.parent,media.db_path))
control=base/'operation';control.mkdir(mode=0o700)
register_pending(root,'restore',('profile',),control,(selector,))
with authority.maintenance(('profile',),3) as session:
 bind_activation(root,'restore',selector,'generation',('config',),session=session)
(root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
ActivationStore(control/'activation').approve('generation','config')
route=sys.argv[1]
if route=='independent':
 other=base/'unrelated.toml';other.write_text('[general]\n');other.chmod(0o600)
 os.environ['TLDW_CONFIG_PATH']=str(other)

child="""import asyncio,json,sys
from Tests.network_guard import install
install()
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
service=RAGService(RAGConfig.from_dict(json.loads(sys.argv[1])))
def forbidden(*args,**kwargs):
 raise AssertionError('query attempted model acquisition/embedding or rebuild')
service.embeddings.create_embeddings_async=forbidden
from tldw_chatbook.RAG_Search import recovery
recovery.rebuild_projection=forbidden
try:
 asyncio.run(service.search('source',include_citations=False))
except (ValueError,RuntimeError) as error:
 assert 'projection_' in str(error), str(error)
else:
 raise AssertionError('fresh owner lost actual indexed source dependency')
print('DEPENDENCY_REFUSED')
"""
result=subprocess.run([sys.executable,'-c',child,json.dumps(config.to_dict(),default=str)],capture_output=True,text=True,timeout=20)
assert result.returncode==0,result.stderr
assert 'DEPENDENCY_REFUSED' in result.stdout,result.stdout
print(result.stdout)
print('retired and reopened')
'''


def test_corrected_relaunch_source_dependency(tmp_path):
    _run(tmp_path, "independent", "owner", script=_CORRECTION_RELAUNCH)


_CORRECTION_PROMPTS = r"""

import asyncio, os, sys, json, subprocess
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing
from tldw_chatbook.RAG_Search import recovery
selector=Path(os.environ['TLDW_CONFIG_PATH'])
base=selector.parent.parent
data=base/'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'+str(data)+'"\n')
selector.chmod(0o600)
media=MediaDatabase(data/'media.db',client_id='generation')
media.add_media_with_keywords(url='https://example.invalid/a',title='Source',media_type='document',content='Original authoritative searchable source text for recovery.',keywords=['fixture'],overwrite=True)
tracking=RAGIndexingDB(data/'indexing.db')
config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(data/'vectors')},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}})
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
prompts=PromptsDatabase(base/'prompt-source.db',client_id='generation')
prompts.add_prompt(name='Unicorn source prompt',author='fixture',details='Unicorn source searchable prompt',system_prompt='unicorn source text')
config.search.prompts_db_path=prompts.db_path
service=RAGService(config)
indexing.semantic_indexing_available=lambda:True
async def build():
 return await indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True)
assert asyncio.run(build())['status']=='ok'
assert asyncio.run(service.search('source',include_citations=False))
cached=asyncio.run(service.search('unicorn',search_type='keyword',keyword_source_types=('prompt',),include_citations=False))
assert cached,cached
prompts.close_connection()
second=RAGService(config)
assert asyncio.run(second.search('source',include_citations=False))
second.vector_store.close()
if sys.argv[1] in ('enhanced','expansion'):
 from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
 enhanced=EnhancedRAGServiceV2(config,enable_parent_retrieval=False,enable_reranking=False,enable_parallel_processing=False)
service.vector_store.close()
media.close_connection()
tracking.close()
root=bootstrap.default_bootstrap_root()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None: startup.close()
authority=admission_authority(root)
authority.register('profile',(selector.parent,prompts.db_path))
control=base/'operation';control.mkdir(mode=0o700)
register_pending(root,'restore',('profile',),control,(selector,))
with authority.maintenance(('profile',),3) as session:
 bind_activation(root,'restore',selector,'generation',('config',),session=session)
(root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
ActivationStore(control/'activation').approve('generation','config')
route=sys.argv[1]
if route=='independent':
 other=base/'unrelated.toml';other.write_text('[general]\n');other.chmod(0o600)
 os.environ['TLDW_CONFIG_PATH']=str(other)

def forbidden(*args,**kwargs):
 raise AssertionError('query attempted embedding or rebuild')
service.embeddings.create_embeddings_async=forbidden
recovery.rebuild_projection=forbidden
try:
 asyncio.run(service.search('unicorn',search_type='keyword',keyword_source_types=('prompt',),include_citations=False))
except (ValueError,RuntimeError) as error:
 assert 'projection_' in str(error), str(error)
else:
 raise AssertionError('configured keyword source escaped projection gate')
assert not service.cache._cache
print('retired and reopened')
"""


def test_corrected_prompts_source_dependency(tmp_path):
    _run(tmp_path, "independent", "owner", script=_CORRECTION_PROMPTS)


_CORRECTION_KEYWORD_MEDIA = r"""

import asyncio, os, sys, json, subprocess
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing
from tldw_chatbook.RAG_Search import recovery
selector=Path(os.environ['TLDW_CONFIG_PATH'])
base=selector.parent.parent
data=base/'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'+str(data)+'"\n')
selector.chmod(0o600)
media=MediaDatabase(data/'media.db',client_id='generation')
media.add_media_with_keywords(url='https://example.invalid/a',title='Source',media_type='document',content='Original authoritative searchable source text for recovery.',keywords=['fixture'],overwrite=True)
tracking=RAGIndexingDB(data/'indexing.db')
config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(data/'vectors')},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}})
prompts=MediaDatabase(base/'keyword-source.db',client_id='keyword')
prompts.add_media_with_keywords(url='https://example.invalid/keyword',title='Unicorn source',media_type='document',content='Unicorn source searchable text',keywords=['fixture'],overwrite=True)
config.search.media_db_path=prompts.db_path
service=RAGService(config)
indexing.semantic_indexing_available=lambda:True
async def build():
 return await indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True)
assert asyncio.run(build())['status']=='ok'
assert asyncio.run(service.search('source',include_citations=False))
cached=asyncio.run(service.search('unicorn',search_type='keyword',keyword_source_types=('media',),include_citations=False))
assert cached,cached
prompts.close_connection()
from tldw_chatbook.RAG_Search.simplified.db_connection_pool import close_all_pools
close_all_pools()
second=RAGService(config)
assert asyncio.run(second.search('source',include_citations=False))
second.vector_store.close()
if sys.argv[1] in ('enhanced','expansion'):
 from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
 enhanced=EnhancedRAGServiceV2(config,enable_parent_retrieval=False,enable_reranking=False,enable_parallel_processing=False)
service.vector_store.close()
media.close_connection()
tracking.close()
root=bootstrap.default_bootstrap_root()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None: startup.close()
authority=admission_authority(root)
authority.register('profile',(selector.parent,prompts.db_path))
control=base/'operation';control.mkdir(mode=0o700)
register_pending(root,'restore',('profile',),control,(selector,))
with authority.maintenance(('profile',),3) as session:
 bind_activation(root,'restore',selector,'generation',('config',),session=session)
(root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
ActivationStore(control/'activation').approve('generation','config')
route=sys.argv[1]
if route=='independent':
 other=base/'unrelated.toml';other.write_text('[general]\n');other.chmod(0o600)
 os.environ['TLDW_CONFIG_PATH']=str(other)

def forbidden(*args,**kwargs):
 raise AssertionError('query attempted embedding or rebuild')
service.embeddings.create_embeddings_async=forbidden
recovery.rebuild_projection=forbidden
try:
 asyncio.run(service.search('unicorn',search_type='keyword',keyword_source_types=('media',),include_citations=False))
except (ValueError,RuntimeError) as error:
 assert 'projection_' in str(error), str(error)
else:
 raise AssertionError('configured keyword source escaped projection gate')
assert not service.cache._cache
print('retired and reopened')
"""


def test_corrected_keyword_media_source_dependency(tmp_path):
    _run(tmp_path, "independent", "owner", script=_CORRECTION_KEYWORD_MEDIA)


def test_unqualified_ordinary_backfill_preserves_query(tmp_path):
    script = _CORRECTION_RELAUNCH.split("second=RAGService(config)")[0]
    script = script.replace(
        "media=MediaDatabase",
        "from tldw_chatbook.Backup_Recovery import qualification\nqualification._qualified_identity=lambda *args:(False,'operation_not_qualified')\nmedia=MediaDatabase",
        1,
    )
    script += """
assert not (bootstrap.default_bootstrap_root()/'projection-dependencies').exists()
assert service._projection_indexed_source_paths == frozenset((media.db_path,))
print('retired and reopened')
"""
    _run(tmp_path, "independent", "owner", script=script)


@pytest.mark.parametrize("damage", ["missing", "corrupt", "unknown", "authority"])
def test_dependency_damage_and_unknown_control_still_refuse(tmp_path, damage):
    script = _CORRECTION_RELAUNCH.split("second=RAGService(config)")[0]
    script += r"""
from tldw_chatbook.RAG_Search.generation import _record_name
root=bootstrap.default_bootstrap_root()
record=root/'projection-dependencies'/_record_name(service.vector_store)
if sys.argv[1]=='missing':
 record.unlink()
elif sys.argv[1]=='corrupt':
 record.write_text('{')
elif sys.argv[1]=='unknown':
 unknown=root/'unknown.json';unknown.write_text('{"version":1}');unknown.chmod(0o600)
else:
 unknown=root/'profile-broken.json';unknown.write_text('{');unknown.chmod(0o600)
try:
 asyncio.run(service.search('source',include_citations=False))
except (ValueError,RuntimeError):
 pass
else:
 raise AssertionError('damaged dependency/control evidence was ignored')
assert not service.cache._cache
print('retired and reopened')
"""
    _run(tmp_path, damage, "owner", script=script)


def test_dependency_union_serializes_independent_indexers(tmp_path):
    script = _CORRECTION_RELAUNCH.split("second=RAGService(config)")[0]
    script += r'''
import time
service.vector_store.close()
media.close_connection()
tracking.close()
child="""import asyncio,json,sys,time
from pathlib import Path
from Tests.network_guard import install
install()
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search import generation,ingestion_indexing as indexing
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
config=RAGConfig.from_dict(json.loads(sys.argv[1]))
base=Path(sys.argv[2]);mode=sys.argv[3]
source=MediaDatabase(base/(mode+'.db'),client_id=mode)
source.add_media_with_keywords(url='https://example.invalid/'+mode,title=mode,media_type='document',content='actual source '+mode,overwrite=True)
tracking=RAGIndexingDB(base/(mode+'-tracking.db'))
service=RAGService(config)
indexing.semantic_indexing_available=lambda:True
if mode=='first':
 original=generation._publish_dependencies
 def hold(*args):
  (base/'entered').touch()
  deadline=time.monotonic()+10
  while not (base/'release').exists():
   assert time.monotonic()<deadline
   time.sleep(.01)
  return original(*args)
 generation._publish_dependencies=hold
try:
 result=asyncio.run(indexing.backfill_semantic_index(media_db=source,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=False))
except BlockingIOError:
 assert mode=='second'
 print('CONTENTION_REFUSED')
else:
 assert result['status']=='ok',result
 print('INDEXED')
"""
args=[sys.executable,'-c',child,json.dumps(config.to_dict(),default=str),str(base)]
with (base/'first.stderr.log').open('w+') as first_log, subprocess.Popen(args+['first'],stdout=subprocess.PIPE,stderr=first_log,text=True) as first:
 try:
  deadline=time.monotonic()+15
  while not (base/'entered').exists():
   assert first.poll() is None
   assert time.monotonic()<deadline
   time.sleep(.01)
  second=subprocess.run(args+['second'],capture_output=True,text=True,timeout=15)
  assert second.returncode==0,second.stderr
  assert 'CONTENTION_REFUSED' in second.stdout,second.stdout
 finally:
  (base/'release').touch()
 stdout,stderr=first.communicate(timeout=15)
 assert first.returncode==0,(base/'first.stderr.log').read_text()
 assert 'INDEXED' in stdout
second=subprocess.run(args+['second'],capture_output=True,text=True,timeout=15)
assert second.returncode==0,second.stderr
assert 'INDEXED' in second.stdout
from tldw_chatbook.RAG_Search.generation import _dependency_paths
with storage.acquire_storage(config.vector_store.persist_directory) as lease:
 paths=_dependency_paths(service.vector_store,lease)
assert paths=={media.db_path,base/'first.db',base/'second.db'},paths
print('retired and reopened')
'''
    _run(tmp_path, "independent", "owner", script=script)


@pytest.mark.parametrize("source", ["prompts", "media", "chachanotes"])
@pytest.mark.parametrize("restored", [False, True])
def test_relative_keyword_source_uses_installed_normalization(
    tmp_path, source, restored
):
    script = _CORRECTION_PROMPTS if source != "media" else _CORRECTION_KEYWORD_MEDIA
    field = "media" if source == "media" else "prompts"
    if source != "chachanotes":
        script = script.replace(
            f"config.search.{field}_db_path=prompts.db_path",
            f"os.chdir(base)\nconfig.search.{field}_db_path=Path(prompts.db_path).name",
        )
    if source == "chachanotes":
        # The real prompt query also checks a declared, not-yet-created notes
        # source using the same installed lexical validation semantics.
        script = script.replace(
            "service=RAGService(config)",
            "os.chdir(base)\n(base/'relative-notes').mkdir(mode=0o700)\nconfig.search.chachanotes_db_path=Path('relative-notes/notes.db')\nservice=RAGService(config)",
            1,
        )
        script = script.replace(
            "authority.register('profile',(selector.parent,prompts.db_path))",
            "assert not (base/'relative-notes/notes.db').exists()\nauthority.register('profile',(selector.parent,base/'relative-notes'))",
        )
    if not restored:
        script = script.split("prompts.close_connection()")[0]
        script += "assert cached\nprint('retired and reopened')\n"
    _run(tmp_path, "independent", "owner", script=script)
