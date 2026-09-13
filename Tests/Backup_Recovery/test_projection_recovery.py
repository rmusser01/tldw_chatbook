"""Owner provenance uses real source documents, indexing and native projection rows."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


def test_projection_ready_requires_all_owner_observations(tmp_path):
    script = """
from Tests import network_guard
network_guard.install()
from tldw_chatbook.RAG_Search.recovery import projection_ready
assert not projection_ready(source_digest="older", indexed_source_digest="newer", compatible=True, reconciled=True)
assert not projection_ready(source_digest="same", indexed_source_digest="same", compatible=True, reconciled=False)
assert not projection_ready(source_digest="", indexed_source_digest="", compatible=True, reconciled=True)
assert projection_ready(source_digest="same", indexed_source_digest="same", compatible=True, reconciled=True)
print("retired and reopened")
"""
    _run(tmp_path, "predicate", "owner", script=script)


_SCRIPT = r"""
import asyncio, json, sys
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing
from tldw_chatbook.RAG_Search import recovery
root=Path.home()
media=MediaDatabase(root/'media.db', client_id='projection-recovery')
media.add_media_with_keywords(url='https://example.invalid/a',title='Authoritative',media_type='document',content='Original authoritative text with enough words to create one complete chunk.',keywords=['fixture'],overwrite=True)
row=dict(media.execute_query('SELECT * FROM Media').fetchone())
identity=row['id']
tracking=RAGIndexingDB(root/'indexing.db')
config=RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(root/'vectors')},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}})
service=RAGService(config)
indexing.semantic_indexing_available=lambda: True
route=sys.argv[1]
async def run():
 async def verify():
  return await recovery.reconcile_projection(service, media_db=media, item_types=('media',))
 async def rebuild():
  return await indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True)
 missing=await verify()
 assert not missing.ready and 'projection_build_provenance_missing' in missing.issues
 if route=='ordinary':
  first=await indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',))
  second=await indexing.backfill_semantic_index(media_db=media,rag_service=service,indexing_db=tracking,item_types=('media',))
  assert first['indexed']==1 and second['skipped']==1
  assert not (await verify()).ready
  before=service.vector_store.recovery_snapshot()
  refused=await rebuild()
  assert 'projection_legacy_source_scope_required' in refused['errors']
  assert service.vector_store.recovery_snapshot()==before
  # Existing explicit collection deletion is the whole-collection repair action.
  service.clear_index()
  assert (await rebuild())['projection'].ready
  return
 if route=='imported':
  service.vector_store.collection.modify(metadata={'verified':True,'source':'built','embedding_model':'mock'})
  tracking.update_collection_state(service.vector_store.collection_name,1,1,{'verified':True,'source_digest':'forged','ready':True})
  assert not (await verify()).ready
  return
 if route=='conversion_mismatch':
  original_add=service.vector_store.add
  def wrong_vectors(ids,embeddings,documents,metadatas):
   original_add(ids,embeddings,documents,metadatas)
   service.vector_store.collection.update(ids=ids,embeddings=[[0.]*384 for _ in ids])
  service.vector_store.add=wrong_vectors
  refused=await rebuild()
  assert refused['status']=='partial' and not refused['projection'].ready
  return
 built=await rebuild()
 assert built['status']=='ok',built
 original=built['projection']
 assert original.ready,original
 assert (await verify()).ready
 assert await recovery.recheck_projection(original,service,media_db=media,item_types=('media',))
 if route=='unknown_model':
  service.embeddings.factory=object()
  assert 'projection_model_identity_unavailable' in (await verify()).issues
  return
 if route=='native_deleted':
  service.vector_store.delete_collection(service.vector_store.collection_name)
  unavailable=await verify()
  assert not unavailable.ready and unavailable.issues
  return
 if route=='tracking_failure':
  tracking.mark_items_indexed=lambda *args: (_ for _ in ()).throw(RuntimeError('synthetic tracking failure'))
  failed=await rebuild()
  assert failed['status']=='partial' and not failed['projection'].ready
  assert not (await verify()).ready
  return
 if route=='source_during_build':
  original_mark=tracking.mark_items_indexed
  def mark_then_change(*args):
   original_mark(*args)
   media.execute_query('UPDATE Media SET content=?,version=version+1 WHERE id=?',('Changed while accepted rebuild is completing.',identity))
  tracking.mark_items_indexed=mark_then_change
  refused=await rebuild()
  assert 'projection_source_changed_during_build' in refused['errors']
  assert not (await verify()).ready
  return
 if route=='other_source':
  other=MediaDatabase(root/'other-media.db',client_id='other-source')
  other.add_media_with_keywords(url='https://example.invalid/a',title='Authoritative',media_type='document',content=row['content'],keywords=['fixture'],overwrite=True)
  before=service.vector_store.recovery_snapshot()
  assert not (await recovery.reconcile_projection(service,media_db=other,item_types=('media',))).ready
  refused=await indexing.backfill_semantic_index(media_db=other,rag_service=service,indexing_db=tracking,item_types=('media',),reconcile_for_recovery=True)
  assert 'projection_legacy_source_scope_required' in refused['errors']
  assert service.vector_store.recovery_snapshot()==before
  other.close_connection()
  return
 if route=='during_verify':
  original_snapshot=service.vector_store.recovery_snapshot
  changed=False
  def snapshot_then_change(**kwargs):
   nonlocal changed
   before=original_snapshot(**kwargs)
   if not changed:
    changed=True
    first=service.vector_store.collection.get(include=['embeddings'])
    service.vector_store.collection.update(ids=[first['ids'][0]],embeddings=[[0.]*384])
   return before
  service.vector_store.recovery_snapshot=snapshot_then_change
  assert 'projection_changed_during_verification' in (await verify()).issues
  return
 if route=='foreign_scope':
  service.vector_store.add(['external_chunk_0'],[[1.]*384],['unowned content'],[{'doc_id':'external'}])
  before=service.vector_store.recovery_snapshot()
  refused=await rebuild()
  assert refused['status']=='partial' and not refused['projection'].ready
  assert service.vector_store.recovery_snapshot()==before
  return
 if route=='shared_source':
  other=MediaDatabase(root/'foreign.db',client_id='foreign-source')
  for suffix in ('alpha','bravo'):
   other.add_media_with_keywords(url='https://example.invalid/foreign-'+suffix,title='Foreign '+suffix,media_type='document',content='Foreign authoritative source '+suffix+' content which must survive until review.',keywords=[],overwrite=True)
  foreign=dict(other.execute_query('SELECT * FROM Media ORDER BY id DESC LIMIT 1').fetchone())
  entry=indexing.media_index_entry(foreign)
  assert entry.document['id'] != 'media_'+str(identity)
  ordinary=await service.index_batch_optimized([entry.document],show_progress=False)
  assert ordinary[0].success and not ordinary[0].error
  before=service.vector_store.recovery_snapshot()
  assert len(before['rows'])==2
  refused=await rebuild()
  assert 'projection_shared_source_scope_required' in refused['errors'],refused
  assert not refused['projection'].ready
  assert service.vector_store.recovery_snapshot()==before
  other.close_connection()
  return
 if route=='positive':
  persisted=tracking.get_collection_state(original.record_key)
  assert persisted['metadata']['source_digest']==original.source_digest
  service.close()
  replacement=RAGService(config)
  assert not (await recovery.reconcile_projection(replacement,media_db=media,item_types=('media',))).ready
  replacement.close()
  return
 if route=='changed':
  # Deliberately retain ID and last_modified; timestamp tracking alone would skip.
  media.execute_query('UPDATE Media SET content=?,version=version+1 WHERE id=?',('Changed authoritative text of equal identity and retained modification timestamp.',identity))
 elif route=='deleted':
  media.execute_query('UPDATE Media SET deleted=1,version=version+1 WHERE id=?',(identity,))
 elif route=='trashed':
  media.execute_query('UPDATE Media SET is_trash=1,version=version+1 WHERE id=?',(identity,))
 elif route=='orphan':
  service.vector_store.add(['media_orphan_chunk_0'],[[1.]*384],['untracked orphan'],[{'doc_id':'media_orphan','source_type':'media'}])
 elif route=='vector':
  first=service.vector_store.collection.get(include=['embeddings'])
  service.vector_store.collection.update(ids=[first['ids'][0]],embeddings=[[0.]*384])
 elif route=='tiny_vector':
  first=service.vector_store.collection.get(include=['embeddings'])
  vector=list(first['embeddings'][0]);vector[0]+=5e-7
  service.vector_store.collection.update(ids=[first['ids'][0]],embeddings=[vector])
 elif route=='config':
  service.config.chunking.chunk_size+=1
 elif route=='model':
  service.embeddings.factory.dimension+=1
 assert not (await verify()).ready
 assert not await recovery.recheck_projection(original,service,media_db=media,item_types=('media',))
 if route=='orphan':
  before=service.vector_store.recovery_snapshot()
  refused=await rebuild()
  assert 'projection_shared_source_scope_required' in refused['errors'],refused
  assert service.vector_store.recovery_snapshot()==before
 elif route in {'changed','deleted','trashed'}:
  repaired=await rebuild()
  assert repaired['projection'].ready,repaired
  assert (await verify()).ready
  if route in {'deleted','trashed'}: assert service.vector_store.collection.count()==0
asyncio.run(run())
service.close();tracking.close();media.close_connection()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "positive",
        "ordinary",
        "imported",
        "changed",
        "deleted",
        "trashed",
        "orphan",
        "vector",
        "tiny_vector",
        "conversion_mismatch",
        "config",
        "model",
        "unknown_model",
        "native_deleted",
        "tracking_failure",
        "source_during_build",
        "other_source",
        "during_verify",
        "foreign_scope",
        "shared_source",
    ],
)
def test_owner_reconciles_actual_sources_and_native_build(tmp_path, route):
    _run(tmp_path, route, "provenance", script=_SCRIPT)


def test_recovery_reconciles_notes_and_complete_conversation_beyond_500_messages(
    tmp_path,
):
    script = r"""
import asyncio
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search import ingestion_indexing as indexing, recovery
root=Path.home()
db=CharactersRAGDB(root/'notes.db','recovery-fixture')
note=db.add_note('Kept note','Authoritative note body has enough words to be indexed completely.')
conversation=db.add_conversation({'title':'Complete transcript'})
for number in range(501):
 db.add_message({'conversation_id':conversation,'sender':'user','content':f'Message {number} must remain authoritative.'})
tracking=RAGIndexingDB(root/'indexing.db')
service=RAGService(RAGConfig.from_dict({'embedding':{'model':'mock'},'vector_store':{'type':'chroma','persist_directory':str(root/'vectors')},'chunking':{'chunk_size':400,'chunk_overlap':0,'min_chunk_size':1}}))
indexing.semantic_indexing_available=lambda:True
async def run():
 summary=await indexing.backfill_semantic_index(chachanotes_db=db,rag_service=service,indexing_db=tracking,item_types=('note','conversation'),reconcile_for_recovery=True)
 assert summary['projection'].ready,summary
 rows=service.vector_store.collection.get(include=['documents'])
 assert any('Message 500' in text for text in rows['documents'])
 prior=summary['projection']
 db.add_message({'conversation_id':conversation,'sender':'user','content':'One additional message after the verified build.'})
 assert not await recovery.recheck_projection(prior,service,chachanotes_db=db,item_types=('note','conversation'))
asyncio.run(run())
service.close();tracking.close();db.close()
print('retired and reopened')
"""
    _run(tmp_path, "conversation", "complete", script=script)
