"""Installed fixed control authority is an explicit, verified capture exclusion."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import json, os, subprocess, sys
from pathlib import Path
from threading import Event
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture, capture
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.capture import compare_scope

route = sys.argv[1]
source = Path(os.environ['TLDW_CONFIG_PATH'])
data = Path(os.environ['XDG_DATA_HOME']) / 'fixture'
source.write_text('[paths]\ndata_dir='+json.dumps(str(data))+'\n')
source.chmod(0o600)
external_config_neighbor = source.parent / 'recovery-bootstrap'
external_config_neighbor.mkdir(mode=0o700)
(external_config_neighbor/'unrelated.txt').write_text('custom config neighbor')
data.mkdir(mode=0o700)
outside = data/'keep.txt'
outside.write_text('selected external data root remains untouched')
root = bootstrap.default_bootstrap_root()
assert root.parent != source.parent
options = {'allow_partial':True, 'staging_parent':Path.home()}
before = None
if route == 'absent':
    before = preview_capture((source,), options=options)
    assert not root.exists(), 'preview created local authority'
if route == 'empty':
    root.mkdir(parents=True, mode=0o700)
else:
    admission_authority(root)
if route == 'symlink':
    moved = Path.home()/'moved-control'
    root.rename(moved)
    root.symlink_to(moved, target_is_directory=True)
elif route == 'missing-registry':
    (root/'admission'/'registry.json').unlink()
elif route == 'unsafe':
    root.chmod(0o755)
elif route == 'unknown-record':
    (root/'unexpected.json').write_text('{"version":1}')
elif route == 'unknown-siblings':
    (root.parent/'unrecognized-store').write_text('unclassified durable state')
    lookalike = data/'default_user'/'recovery-bootstrap'
    lookalike.mkdir(parents=True, mode=0o700)
    (lookalike/'not-control.txt').write_text('ordinary unknown owner')
elif route == 'capture':
    producer = """
import os
from pathlib import Path
from Tests import network_guard
network_guard.install()
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
_populate_required_dependencies(preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),),options={'allow_partial':True}))
"""
    prepared = subprocess.run([sys.executable,'-c',producer],capture_output=True,text=True,timeout=30)
    assert prepared.returncode == 0, prepared.stderr[-3000:]
preserved = {p:(p.read_bytes(),p.stat().st_mode,p.stat().st_mtime_ns) for p in (source, outside, external_config_neighbor/'unrelated.txt')}
preview = preview_capture((source,), options=options)
controls = [item for item in preview.items if item.path == root]
assert len(controls) == 1, controls
control = controls[0]
assert control.owner == 'recovery.control', control
if route in {'symlink','missing-registry','empty','unsafe','unknown-record'}:
    assert control.status == 'unavailable', control
    assert not preview.complete
else:
    assert control.status == 'intentionally_excluded', control
assert not any(item.owner == 'unknown' and item.path == root for item in preview.items)
assert not any(item.path == external_config_neighbor for item in preview.items), 'custom config parent scanned by name'
if route == 'unknown-siblings':
    unknowns = {item.path for item in preview.items if item.owner == 'unknown'}
    assert root.parent/'unrecognized-store' in unknowns
    assert lookalike in unknowns
    assert not preview.complete
if route == 'absent':
    assert compare_scope(before, preview) == ('scope_changed',)
if route == 'capture':
    result = capture((source,),preview.scope_digest,Path.home()/'backup.tldw-backup.zip',options=options,cancel=Event())
    manifest = json.loads(result.manifest_bytes)
    assert manifest['consistency'] == 'partial', 'unrelated owners are not qualified here'
    ids = [item.logical_id for item in result.inventory.items]
    assert len(ids) == len(set(ids)), 'discovery and actual-session controls have duplicate IDs'
    rows = [item for item in result.inventory.items if item.owner == 'recovery.control']
    assert len(rows) == 2 and all(item.path == root and item.status == 'intentionally_excluded' for item in rows)
    assert not any(row['owner_id'] == 'recovery.control' for row in manifest['files'])
assert all((p.read_bytes(),p.stat().st_mode,p.stat().st_mtime_ns) == original for p,original in preserved.items())
print('retired and reopened')
'''


@pytest.mark.parametrize(
    "route",
    [
        "initialized",
        "absent",
        "unknown-siblings",
        "symlink",
        "missing-registry",
        "empty",
        "unsafe",
        "unknown-record",
        "capture",
    ],
)
def test_public_inventory_recognizes_only_intact_fixed_control(tmp_path, route):
    _run(tmp_path, route, "success", script=_SCRIPT)
