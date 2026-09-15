"""Explicit recovery commands use the actual byte-only service operations."""

import pytest

_EXTRACT = r"""
import builtins, json, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from Tests.Backup_Recovery.test_inert_extraction import _archive
from tldw_chatbook.Backup_Recovery.launcher import recovery_main

decision, destination_kind = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('broken = [')
before = selector.read_bytes()
archive, payloads = _archive(Path.home())
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name not in {'tldw_chatbook.app', 'tldw_chatbook.config'}, name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
builtins.input = lambda prompt: decision
destination = Path.home() / 'manual'
if destination_kind == 'existing':
    destination.mkdir(mode=0o700)
    (destination / 'keep').write_bytes(b'original')
result = recovery_main(['--control-root', str(Path.home() / 'control'),
    'extract', str(archive.path), '--group', 'unsupported', '--destination', str(destination)])
if destination_kind == 'existing':
    assert result == 1
    assert (destination / 'keep').read_bytes() == b'original'
elif decision == 'extract':
    assert result == 0
    report = json.loads((destination / 'inert-mapping.json').read_text())
    assert {row['logical_id'] for row in report['files']} == {'db', 'config'}
    for row in report['files']:
        assert (destination / row['output']).read_bytes() == payloads[row['logical_id']]
else:
    assert result == 0
    assert not destination.exists()
assert selector.read_bytes() == before
assert 'tldw_chatbook.app' not in sys.modules
assert 'tldw_chatbook.config' not in sys.modules
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "decision,destination",
    [("extract", "absent"), ("cancel", "absent"), ("extract", "existing")],
)
def test_cli_extracts_only_reviewed_inert_groups_before_normal_startup(
    tmp_path, decision, destination
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, decision, destination, script=_EXTRACT)
