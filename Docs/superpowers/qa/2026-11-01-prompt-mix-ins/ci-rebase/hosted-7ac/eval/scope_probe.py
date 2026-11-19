import os
import pytest
INJECT = r"""
from tldw_chatbook.Backup_Recovery import capture_service as _observed_capture
from dataclasses import asdict as _asdict
_original_review_digest = _observed_capture._review_digest
_scope_number = 0
def _observe_review(inventory, *args):
    global _scope_number
    digest = _original_review_digest(inventory, *args)
    rows = []
    for item in inventory.items:
        metadata = _asdict(item.metadata) if item.metadata else None
        if metadata:
            metadata.pop('mtime_ns', None)
            metadata.pop('mode', None)
        rows.append({'owner': item.owner, 'id': item.logical_id,
                     'path_hash': hashlib.sha256(str(item.path).encode()).hexdigest(),
                     'status': item.status, 'dependencies': item.dependencies,
                     'shared_group': item.shared_group, 'metadata': metadata})
    _scope_number += 1
    record = {'n':_scope_number, 'digest':digest, 'inventory_digest':inventory.scope_digest,
              'complete': inventory.complete, 'issues':inventory.issues, 'rows':rows}
    with Path(os.environ['EVAL_SCOPE_EVIDENCE']).open('a') as output:
        output.write(json.dumps(record, sort_keys=True) + '\n')
    return digest
_observed_capture._review_digest = _observe_review
"""
def pytest_collection_modifyitems(items):
    module = next(item.module for item in items if 'test_eval_retained_definitions' in item.nodeid)
    marker = 'from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app'
    assert module._RETAIN_PUBLIC.count(marker) == 1
    module._RETAIN_PUBLIC = module._RETAIN_PUBLIC.replace(marker, INJECT + '\n' + marker)
