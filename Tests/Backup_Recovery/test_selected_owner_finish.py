"""Fresh Finish preserves an absent redundant history root through activation."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_selected_owner_absence import _history_service_script

_INTERRUPT = r"""
    from tldw_chatbook.Backup_Recovery import control_records
    from tldw_chatbook.Backup_Recovery.journal import Journal
    original_pair_member = control_records._publish_activation_record
    def stop_after_association(root, parent, name, *args, **kwargs):
        result = original_pair_member(root, parent, name, *args, **kwargs)
        if BOUNDARY == 'association' and name.startswith('activation-'):
            os._exit(91)
        return result
    control_records._publish_activation_record = stop_after_association
    original_append = Journal._append
    def stop_after_record(self, parent, event, evidence):
        result = original_append(self, parent, event, evidence)
        if event == BOUNDARY:
            os._exit(91)
        return result
    Journal._append = stop_after_record
"""

_FINISH = r"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from Tests import network_guard
network_guard.install()
child = subprocess.run([sys.executable, '-c', CHILD], capture_output=True,
                       text=True, timeout=90, check=False)
assert child.returncode == 91, child.stderr[-5000:] + child.stdout[-1000:]

# A new process and service acquire native authority from the existing journal.
from keyring.backends.null import Keyring
from tldw_chatbook.Backup_Recovery import bootstrap, credentials
from tldw_chatbook.Backup_Recovery.admission import Admission
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
store = KeyringServerCredentialStore(keyring_backend=Keyring())
credentials._credential_store = lambda: store
source = Path(os.environ['TLDW_CONFIG_PATH'])
root = bootstrap.default_bootstrap_root()
evidence = json.loads((Path.home() / 'history-publication-evidence.json').read_text())
history = Path(evidence['history_path'])
service = RecoveryService(Path.home() / 'recovery-control')
try:
    pending = service.pending_operations()
    assert len(pending) == 1, pending
    operation = pending[0]['operation_id']
    journal = _journal(service.control_root, operation)
    with journal._locked(exclusive=False) as parent:
        events = [row.event for row in journal._records(parent)]
    assert ('activation_recorded' in events) == (BOUNDARY != 'association')
    assert ('committed' in events) == (BOUNDARY == 'committed')
    assert bool(tuple(root.glob('activation-update-*.json'))) == (BOUNDARY == 'association')
    assert not bootstrap.startup_permission(source, root)[0]
    def unchanged():
        assert not history.exists()
        assert hashlib.sha256((root / 'admission' / 'registry.json').read_bytes()).hexdigest() == evidence['registry']
        for name, before in evidence['preserved'].items():
            path = Path(name)
            info = path.stat()
            assert [hashlib.sha256(path.read_bytes()).hexdigest(), info.st_dev, info.st_ino] == before
    unchanged()
    result = service.wait(service.start_recovery(
        operation, action='finish', rollback_password=b'rollback-test-password'))
    assert result['state'] == 'succeeded', dict(result)
    unchanged()
    assert service.pending_operations() == ()
    assert not tuple(root.glob('activation-update-*.json'))
    assert bootstrap.startup_permission(source, root)[0]
    profiles = bootstrap._records(root)[1]
    assert {row['selector']: [row['namespaces'], row['roots']] for row in profiles} == evidence['profiles']
    registry = bootstrap._registry(root)
    binding = bootstrap._binding(source, profiles, registry)
    assert binding is not None and str(history) in binding['roots']
    assert any(entry['roots'] == [str(history)] and evidence['history_inode_token'] in entry['historical']
               for entry in registry.values())
    with Admission.open_existing(root / 'admission').normal(tuple(binding['namespaces'])):
        assert not history.exists()
    # Real installed producer readback, in another fresh process, must also
    # traverse normal startup/admission without creating the retired history.
    reopened = subprocess.run([sys.executable, '-c', READBACK], capture_output=True,
                              text=True, timeout=30, check=False)
    assert reopened.returncode == 0, reopened.stderr[-4000:]
    unchanged()
    with journal._locked(exclusive=False) as parent:
        events = [row.event for row in journal._records(parent)]
    assert events.count('activation_recorded') == events.count('committed') == 1
    assert not network_guard.blocked_attempts()
finally:
    service.close()
print('retired and reopened')
"""

_READBACK = r"""
import os
import tomllib
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.profile_paths import database_path
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
selector = Path(os.environ['TLDW_CONFIG_PATH'])
assert bootstrap.startup_permission(selector, bootstrap.default_bootstrap_root())[0]
store = PromptsDatabase(database_path(tomllib.loads(selector.read_text()), 'prompts_db_path'),
                        'history-finish-readback')
try:
    assert store.get_prompt_by_name('prepare')['user_prompt'] == 'Saved selected prompt content'
    assert store.get_prompt_by_name('newer') is None
finally:
    store.close()
assert not network_guard.blocked_attempts()
"""


@pytest.mark.parametrize(
    "boundary", ["association", "activation_recorded", "committed"]
)
def test_fresh_finish_preserves_retired_history_namespace(tmp_path, boundary):
    child = _history_service_script()
    marker = "    result = succeeded(service, service.start_restore(inspection, plan,"
    assert child.count(marker) == 1
    child = child.replace(marker, _INTERRUPT + "\n" + marker)
    child = child.replace("BOUNDARY", repr(boundary))
    script = (
        _FINISH.replace("BOUNDARY", repr(boundary))
        .replace("CHILD", repr(child))
        .replace("READBACK", repr(_READBACK))
    )
    _run(tmp_path, "history-finish", boundary, script=script, timeout=150)
