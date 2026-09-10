"""Finite collection helpers release real private Chroma clients."""

import importlib.util

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import sys
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.RAG_Search.simplified import collection_indexes as module
import chromadb
from chromadb.config import Settings
from chromadb.api.shared_system_client import SharedSystemClient

route, outcome = sys.argv[1:]
settings = Settings(anonymized_telemetry=False, allow_reset=True)
original_client = module._client

def invoke(path):
    if route == 'adopt':
        return module.adopt_legacy_collection(path, 'legacy', 'adopted', {'source': 'test'})
    if route == 'delete':
        return module.delete_index(path, 'legacy')
    return module.list_indexes(path)

for shared in (False, True):
    path = Path.home() / ('shared' if shared else 'sole')
    resident = chromadb.PersistentClient(path=str(path), settings=settings)
    collection = resident.create_collection('legacy', embedding_function=None)
    collection.add(ids=['one'], embeddings=[[1.0, 0.0]], documents=['kept'])
    if outcome == 'empty':
        resident.delete_collection('legacy')
    if not shared:
        resident.close()
    identifier = str(path)
    baseline = SharedSystemClient._identifier_to_refcount.get(identifier, 0)
    created = []
    def client(directory):
        value = original_client(directory)
        created.append((value, value._system))
        if outcome == 'error':
            def failed_list(*args, **kwargs):
                raise RuntimeError('controlled operation failure')
            value.list_collections = failed_list
        return value
    module._client = client
    try:
        result = invoke(path)
        if route == 'list':
            assert result == [] if outcome != 'success' else result[0]['count'] == 1
        else:
            assert result is (outcome == 'success')
        assert SharedSystemClient._identifier_to_refcount.get(identifier, 0) == baseline
        assert created[0][0]._closed
        if shared:
            assert resident._system is created[0][1]
            assert resident.heartbeat() > 0
        else:
            assert identifier not in SharedSystemClient._identifier_to_system
            assert created[0][1]._running is False
        with chromadb.PersistentClient(path=str(path), settings=settings) as reopened:
            names = {c.name for c in reopened.list_collections()}
            expected = set() if outcome == 'empty' or (route == 'delete' and outcome == 'success') else {'adopted' if route == 'adopt' and outcome == 'success' else 'legacy'}
            assert names == expected
            if names:
                row = reopened.get_collection(next(iter(names)), embedding_function=None).get(include=['documents', 'embeddings'])
                assert row['ids'] == ['one'] and row['documents'] == ['kept']
                assert row['embeddings'].tolist() == [[1.0, 0.0]]
    finally:
        for value, _ in created:
            value.close()
        resident.close()
    assert identifier not in SharedSystemClient._identifier_to_system
print('retired and reopened')
"""


@pytest.mark.skipif(
    importlib.util.find_spec("chromadb") is None,
    reason="Chroma optional dependency unavailable",
)
@pytest.mark.parametrize("route", ["adopt", "list", "delete"])
@pytest.mark.parametrize("outcome", ["success", "empty", "error"])
def test_real_clients_release_only_their_own_shared_engine_reference(
    tmp_path, route, outcome
):
    _run(tmp_path, route, outcome, script=_SCRIPT)


_COMPATIBILITY = r"""
import sys
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.RAG_Search.simplified import collection_indexes as module
route, outcome = sys.argv[1:]
messages = []
sink = module.logger.add(lambda message: messages.append(str(message)))
class LegacyClient:
    def list_collections(self):
        return []
client = LegacyClient()
if outcome == 'close_error':
    def close():
        raise RuntimeError('CLOSE_SECRET_SENTINEL_7654321')
    client.close = close
module._client = lambda path: client
if route == 'adopt':
    result = module.adopt_legacy_collection(Path.home(), 'old', 'new', {})
    assert result is False
elif route == 'delete':
    assert module.delete_index(Path.home(), 'old') is False
else:
    assert module.list_indexes(Path.home()) == []
assert any('chroma_retirement_unqualified' in message for message in messages)
assert all('CLOSE_SECRET_SENTINEL_7654321' not in message for message in messages)
module.logger.remove(sink)
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["adopt", "list", "delete"])
@pytest.mark.parametrize("outcome", ["missing_close", "close_error"])
def test_unsupported_or_failed_close_is_explicit_without_changing_results(
    tmp_path, route, outcome
):
    _run(tmp_path, route, outcome, script=_COMPATIBILITY)
