"""One authorized read-only retained-cache diagnostic; no semantic claim or model."""
import asyncio
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from Tests import conftest as _test_bootstrap  # noqa: F401
from Tests.UI.app_factory import _build_test_app, drain_created_dirs, drain_created_runtimes
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery.isolated_restore import verified_data_identity
from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB, index_store_path
from tldw_chatbook.Workspaces.project_index.allocation import allocation_gate, _inventory, existing_namespace_paths
from tldw_chatbook.Workspaces.project_index.runtime import close_project_index_service, get_project_index_service, project_index_key

SDD = Path(__file__).resolve().parent
BASE = SDD.parents[1] / 'k5-scale'
PROFILE = BASE / 'stable-profile'
OUTPUT = SDD / 'task4-retained-read-diagnostic-01.json'

def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start

def blocked(*args, **kwargs):
    raise AssertionError('read_diagnostic_http_attempted')

def observe(service, key, launch):
    path = index_store_path(service._factory.data_root, key)
    group = path.parent.parent
    assert str(group).casefold() == str(Path(launch['cache'])).casefold()
    assert path.exists() and group.exists()
    db_hash_before = digest(path)
    status_before, status_seconds = timed(lambda: service.status(key))
    semantic_before, semantic_seconds = timed(lambda: service.semantic_state(key))
    with service._store(key, cleanup_only=True) as store:
        assert store.db_path == path
        with allocation_gate(group, existing_only=True):
            inventory, inventory_seconds = timed(lambda: _inventory(group))
            peers, paths_seconds = timed(lambda: existing_namespace_paths(group))
            peer_results = []
            for peer in peers:
                result, seconds = timed(lambda p=peer: WorkspaceProjectIndexDB._probe_peer(p, group, gated=True))
                peer_results.append({'schema_version': result[0], 'has_semantics': result[1], 'seconds': seconds})
        start = time.perf_counter()
        with service._read_fenced(store, key) as authority:
            assert authority is not None
            fenced_entry_seconds = time.perf_counter() - start
            start_sql = time.perf_counter()
            with store.connection() as conn:
                rows = conn.execute(
                    'SELECT c.id,f.path,f.identity,f.content_sha256,c.line_start,c.line_end,'
                    'length(CAST(c.text AS BLOB)) AS byte_count '
                    'FROM chunks c JOIN files f ON f.path=c.path '
                    'WHERE c.id>? ORDER BY c.id LIMIT 32', (0,)
                ).fetchall()
                head_seconds = time.perf_counter() - start_sql
                ids = tuple(row[0] for row in rows)
                assert 0 < len(ids) <= 32
                slots = ','.join('?' for _ in ids)
                start_text = time.perf_counter()
                texts = conn.execute(f'SELECT id,text FROM chunks WHERE id IN ({slots})', ids).fetchall()
                text_seconds = time.perf_counter() - start_text
                assert len(texts) == len(ids)
                text_bytes = sum(len(row[1].encode('utf-8')) for row in texts)
            fenced_total_seconds = time.perf_counter() - start
    status_after, _ = timed(lambda: service.status(key))
    semantic_after, _ = timed(lambda: service.semantic_state(key))
    with allocation_gate(group, existing_only=True):
        inventory_after = _inventory(group)
    db_hash_after = digest(path)
    assert status_before == status_after and semantic_before == semantic_after
    assert inventory == inventory_after and db_hash_before == db_hash_after
    return {
        'status_state': status_before.state, 'status_indexed': status_before.indexed,
        'semantic_state': semantic_before.state, 'semantic_revision': semantic_before.semantic_revision,
        'status_equal_after': True, 'semantic_equal_after': True,
        'db_sha256_before_after_equal': True, 'inventory_equal_after': True,
        'db_sha256': db_hash_before,
        'group_file_count': len(inventory), 'group_file_bytes': sum(inventory.values()),
        'peer_count': len(peers), 'peer_results': peer_results,
        'page_heads': len(rows), 'page_texts': len(texts), 'page_text_utf8_bytes': text_bytes,
        'seconds': {'status': status_seconds, 'semantic_state': semantic_seconds,
                    'inventory': inventory_seconds, 'namespace_paths': paths_seconds,
                    'fenced_entry': fenced_entry_seconds, 'page_heads': head_seconds,
                    'page_text': text_seconds, 'fenced_total': fenced_total_seconds},
        'limitations': 'Read-only heads/text probe outside semantic claim; no write admission or append measured.'
    }

async def main():
    preflight = json.loads((PROFILE / 'preflight-2.json').read_text(encoding='utf-8'))
    data = config.get_user_data_dir()
    assert str(data) == preflight['data']
    assert list(verified_data_identity()) == preflight['verified_identity']
    launch = json.loads((BASE / 'nested-launch.json').read_text(encoding='utf-8'))
    app = _build_test_app(user_data_dir=data)
    try:
        binding = app.workspace_registry_service.get_runtime_binding(launch['key']['binding_id'])
        assert binding is not None and binding.locator == launch['corpus']
        service = get_project_index_service(app)
        key = project_index_key(app, launch['key']['workspace_id'], binding.binding_id, Path(launch['corpus']))
        assert asdict(key) == launch['key']
        result = await asyncio.to_thread(observe, service, key, launch)
    finally:
        await close_project_index_service(app)
        await drain_created_runtimes()
        drain_created_dirs()
    OUTPUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'phase': 'complete', 'output': str(OUTPUT), 'seconds': result['seconds']}), flush=True)

if __name__ == '__main__':
    import httpx
    import requests
    with (patch.object(requests.sessions.Session, 'request', blocked),
          patch.object(httpx.Client, 'send', blocked),
          patch.object(httpx.AsyncClient, 'send', blocked)):
        asyncio.run(main())
