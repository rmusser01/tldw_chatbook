"""Installed Chroma borrowers settle before releasing their storage holds."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_NATIVE = r"""
import asyncio, sys, threading, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
from tldw_chatbook.RAG_Search.simplified import collection_indexes
from chromadb.api.shared_system_client import SharedSystemClient

route = sys.argv[1]
path = Path.home() / 'vectors'
before = set(storage._live_leases)
if route == 'unsupported_version':
    import chromadb
    chromadb.__version__ = 'unqualified-test-version'
if route == 'paused':
    pause = storage._begin_local_pause()
    try:
        try:
            ChromaVectorStore(path)
        except RecoveryRequired:
            pass
        else:
            raise AssertionError('paused constructor created root')
        assert not path.exists()
    finally:
        pause.resume()
else:
    store = ChromaVectorStore(path, collection_name='owned')
    store.add(['one'], [[1.0, 0.0]], ['retained'], [{'doc_id': 'one'}])
    client, system = store._client, store._client._system
    held = set(storage._live_leases) - before
    assert held, 'resident native client has no storage lease'
    if route == 'unsupported_version':
        store.close()
        assert store._client is None, 'successful ordinary close lost lazy reopen'
        assert held <= storage._live_leases, 'unsupported version claimed native retirement'
        assert store.search([1.0, 0.0], top_k=1)[0].document == 'retained'
        store.close()
    elif route == 'close_error':
        original_stop = system.stop
        def stop():
            raise RuntimeError('controlled native stop failure')
        system.stop = stop
        store.close()
        assert client._closed
        assert store._client is client, 'ambiguous close discarded client evidence'
        assert held <= storage._live_leases, 'ambiguous native stop released admission'
        store.close()
        assert held <= storage._live_leases, 'idempotent client close forged retirement'
        system.stop = original_stop
        original_stop()
    else:
        assert collection_indexes.list_indexes(path)[0]['count'] == 1
        assert store.search([1.0, 0.0], top_k=1)[0].document == 'retained'
        store.close()
        assert not system._running
        assert not (held & storage._live_leases)
        assert str(path) not in SharedSystemClient._identifier_to_system
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route", ["paused", "resident", "close_error", "unsupported_version"]
)
def test_native_admission_and_positive_retirement(tmp_path, route):
    _run(tmp_path, route, "native", script=_NATIVE)


_EXTERNAL_CLOSE = r"""
import asyncio, sys, threading, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
import chromadb
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore

route = sys.argv[1]
path = Path.home() / 'vectors'
before = set(storage._live_leases)
store = ChromaVectorStore(path, collection_name='owned')
store.add(['one'], [[1., 0.]], ['preserved'], [{'doc_id': 'one'}])
client, system = store.client, store.client._system
server = client._server
external = chromadb.PersistentClient(path=str(path), settings=system.settings)
assert external._system is system
held = set(storage._live_leases) - before
store.close()
assert held <= storage._live_leases and system._running
original_stop = server.stop
entered, release = threading.Event(), threading.Event()
closer = None
if route == 'failed':
    def stop():
        raise RuntimeError('native component failed before releasing bindings')
    server.stop = stop
    try:
        external.close()
    except RuntimeError:
        pass
    else:
        raise AssertionError('injected final native stop did not fail')
elif route == 'stopping':
    def stop():
        entered.set()
        assert release.wait(5)
        original_stop()
    server.stop = stop
    closer = threading.Thread(target=external.close)
    closer.start()
    assert entered.wait(5)
elif route == 'success':
    external.close()
assert system._running is (route == 'running')
assert ('bindings' in vars(server)) is (route != 'success')
async def run():
    participant._maintenance_close_admission()
    try:
        retired = await participant._maintenance_drain(time.monotonic() + 2)
        assert retired is (route == 'success'), 'passive system flag forged native retirement'
        if route != 'success':
            assert held <= storage._live_leases
        else:
            assert not (held & storage._live_leases)
        if route == 'stopping':
            release.set()
            await asyncio.to_thread(closer.join, 5)
            assert not closer.is_alive()
            assert 'bindings' not in vars(server)
            assert await participant._maintenance_drain(time.monotonic() + 2)
            assert not (held & storage._live_leases)
    finally:
        release.set()
        if closer is not None:
            await asyncio.to_thread(closer.join, 5)
        await participant._maintenance_resume()
asyncio.run(run())
if route == 'failed':
    server.stop = original_stop
    original_stop()
elif route == 'running':
    external.close()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["running", "stopping", "failed", "success"])
def test_untracked_final_close_requires_native_bindings_retirement(tmp_path, route):
    _run(tmp_path, route, "external", script=_EXTERNAL_CLOSE)


_DRAIN = r"""
import asyncio, sys, threading, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
from tldw_chatbook.RAG_Search.simplified import collection_indexes
from chromadb.api.shared_system_client import SharedSystemClient

route = sys.argv[1]
path = Path.home() / 'vectors'
first = ChromaVectorStore(path, collection_name='owned')
first.add(['one'], [[1., 0.]], ['first'], [{'doc_id': 'one'}])
second = ChromaVectorStore(path, collection_name='owned')
assert second.client._system is first.client._system
system = first.client._system
entered, release = threading.Event(), threading.Event()
if route == 'native':
    original = first.collection.add
    def add(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    first.collection.add = add
    call = lambda: first.add(['two'], [[0., 1.]], ['second'], [{'doc_id': 'two'}])
else:
    original = collection_indexes._client
    def client(root):
        value = original(root)
        entered.set()
        assert release.wait(5)
        return value
    collection_indexes._client = client
    call = lambda: collection_indexes.list_indexes(path)

async def run():
    task = asyncio.create_task(asyncio.to_thread(call))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        participant._maintenance_close_admission()
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        assert not await participant._maintenance_drain(time.monotonic()+.03), 'retired running native call'
        assert system._running
        try:
            second.search([1., 0.])
        except RecoveryRequired:
            pass
        else:
            raise AssertionError('cached native query entered after closure')
        release.set()
        assert await participant._maintenance_drain(time.monotonic()+5)
        assert system._running is False
        assert first._client is None and second._client is None
        assert str(path) not in SharedSystemClient._identifier_to_system
        await participant._maintenance_resume()
        found = first.search([1., 0.], top_k=10)
        assert {row.document for row in found} == ({'first','second'} if route == 'native' else {'first'})
        assert first.client._system is not system
    finally:
        release.set()
        await participant._maintenance_resume()
        first.close()
        second.close()
asyncio.run(run())
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["native", "finite"])
def test_waiter_cancel_does_not_retire_native_or_other_borrowers(tmp_path, route):
    _run(tmp_path, route, "drain", script=_DRAIN)


_INDEX = r"""
import asyncio, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.ingestion_indexing import index_entries, IndexEntry
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB

route = sys.argv[1]
store = ChromaVectorStore(Path.home()/'vectors', collection_name='owned')
db = RAGIndexingDB(Path.home()/'indexing.db')
entered, release = asyncio.Event(), asyncio.Event()
native_entered, native_release = threading.Event(), threading.Event()
class Cache:
    async def clear_async(self):
        if route == 'cache':
            entered.set()
            await release.wait()
class Service:
    vector_store = store
    cache = Cache()
    async def index_batch_optimized(self, documents, **kwargs):
        await RAGService._store_chunks(self, ['media_1_chunk_0'], [[1.,0.]], ['preserved content'], [{'doc_id':'media_1'}])
        return [SimpleNamespace(doc_id='media_1', success=True, chunks_created=1)]
if route == 'native':
    original = store.add
    def add(*args, **kwargs):
        native_entered.set()
        assert native_release.wait(5)
        return original(*args, **kwargs)
    store.add = add
entry = IndexEntry(item_id='1', item_type='media', last_modified=datetime.now(timezone.utc),
                   document={'id':'media_1','content':'preserved content','metadata':{}})
async def run():
    task = asyncio.create_task(index_entries(Service(), db, [entry]))
    try:
        if route == 'native':
            assert await asyncio.to_thread(native_entered.wait, 5)
        else:
            await asyncio.wait_for(entered.wait(), 5)
        participant._maintenance_close_admission()
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done(), 'cancelled waiter abandoned accepted indexing bookkeeping'
        assert not await participant._maintenance_drain(time.monotonic()+.02)
        release.set()
        native_release.set()
        try: await task
        except asyncio.CancelledError: pass
        assert db.get_indexed_item_info('1', 'media')['chunk_count'] == 1
        assert await participant._maintenance_drain(time.monotonic()+5)
        await participant._maintenance_resume()
        assert store.search([1.,0.])[0].document == 'preserved content'
    finally:
        release.set()
        native_release.set()
        await participant._maintenance_resume()
        await asyncio.gather(task, return_exceptions=True)
        store.close()
        db.close()
asyncio.run(run())
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["native", "cache"])
def test_indexing_retains_native_cache_and_durable_bookkeeping(tmp_path, route):
    _run(tmp_path, route, "index", script=_INDEX)


_QUEUED = r"""
import asyncio, time, threading
from pathlib import Path
from types import SimpleNamespace
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.RAG_Search.ingestion_indexing import IngestionIndexer, media_index_entry
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.Backup_Recovery import storage_admission as storage

store = ChromaVectorStore(Path.home()/'vectors', collection_name='owned')
class Service:
    vector_store = store
    cache = SimpleNamespace(clear=lambda: None)
    async def index_batch_optimized(self, documents, **kwargs):
        await RAGService._store_chunks(self, ['media_1_chunk_0'], [[1.,0.]], ['queued content'], [{'doc_id':'media_1'}])
        return [SimpleNamespace(doc_id='media_1', success=True, chunks_created=1)]
indexer = IngestionIndexer(rag_service=Service(), indexing_db_path=Path.home()/'indexing.db')
entered, release = threading.Event(), threading.Event()
original = indexer._process_batch
async def delayed(batch):
    entered.set()
    assert await asyncio.to_thread(release.wait, 5)
    await original(batch)
indexer._process_batch = delayed
entry = media_index_entry({'id':1,'content':'queued content','title':'Queued','deleted':0,'is_trash':0})
async def run():
    try:
        assert indexer.submit(entry)
        assert await asyncio.to_thread(entered.wait, 5)
        participant._maintenance_close_admission()
        assert not indexer.submit(entry), 'paused indexer admitted new root'
        assert not await participant._maintenance_drain(time.monotonic()+.02), 'forgot queued indexing acceptance'
        release.set()
        assert await participant._maintenance_drain(time.monotonic()+5)
        assert indexer.stats()['indexed'] == 1
        assert not [lease for lease in storage._live_leases if lease.resource_thread is indexer._thread]
        await participant._maintenance_resume()
        assert store.search([1.,0.])[0].document == 'queued content'
    finally:
        release.set()
        await participant._maintenance_resume()
        indexer.stop()
        store.close()
asyncio.run(run())
print('retired and reopened')
"""


def test_queued_ingestion_settles_before_native_retirement(tmp_path):
    _run(tmp_path, "queue", "drain", script=_QUEUED)


_SLOW_CLOSE = r"""
import asyncio, threading, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
from tldw_chatbook.Backup_Recovery import storage_admission as storage
store = ChromaVectorStore(Path.home()/'vectors', collection_name='owned')
store.add(['one'], [[1.,0.]], ['kept'], [{'doc_id':'one'}])
system = store.client._system
original = system.stop
entered, release = threading.Event(), threading.Event()
def stop():
    entered.set()
    assert release.wait(5)
    original()
system.stop = stop
async def run():
    try:
        participant._maintenance_close_admission()
        drain = asyncio.create_task(participant._maintenance_drain(time.monotonic()+.03))
        assert await asyncio.to_thread(entered.wait, 5)
        assert await drain is False
        assert system._running
        release.set()
        assert await participant._maintenance_drain(time.monotonic()+5)
        assert not system._running
    finally:
        release.set()
        await participant._maintenance_resume()
        store.close()
asyncio.run(run())
print('retired and reopened')
"""


def test_native_close_respects_drain_deadline_without_abandoning_stop(tmp_path):
    _run(tmp_path, "slow", "close", script=_SLOW_CLOSE)


_SERVICE = r"""
import asyncio, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
config = RAGConfig()
config.embedding.model = 'mock'
config.vector_store.type = 'chroma'
config.vector_store.persist_directory = Path.home()/'vectors'
service = RAGService(config)
async def run():
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        indexed = await service.index_document('source', 'The retained source describes quokka habitats.', title='Habitat')
        assert indexed.success
        system = service.vector_store.client._system
        original = service.embeddings.create_embeddings_async
        async def embedding(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)
        service.embeddings.create_embeddings_async = embedding
        task = asyncio.create_task(service.search('quokka habitats', search_type='semantic', include_citations=False, score_threshold=-1))
        await asyncio.wait_for(entered.wait(), 5)
        participant._maintenance_close_admission()
        assert not await participant._maintenance_drain(time.monotonic()+.03)
        release.set()
        result = await task
        assert any('retained source' in row.document for row in result)
        assert await participant._maintenance_drain(time.monotonic()+5)
        assert not system._running
        await participant._maintenance_resume()
        service.cache.clear()
        result = await service.search('quokka habitats', search_type='semantic', include_citations=False, score_threshold=-1)
        assert any('retained source' in row.document for row in result)
        reopened = service.vector_store.client._system
        assert reopened is not system
        service.close()
        assert not reopened._running
    finally:
        release.set()
        await participant._maintenance_resume()
        service.close()
asyncio.run(run())
print('retired and reopened')
"""


def test_actual_service_search_continues_to_native_after_intake_closes(tmp_path):
    _run(tmp_path, "service", "search", script=_SERVICE)


@pytest.mark.asyncio
async def test_empty_lifetime_drains_immediately_at_deadline():
    import time

    from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import ChromaLifetime

    owner = ChromaLifetime()
    owner._maintenance_close_admission()
    assert await owner._maintenance_drain(time.monotonic())
    await owner._maintenance_resume()
