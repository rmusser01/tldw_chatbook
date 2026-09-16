"""A fresh process finishes selective Settings with its activation still pending."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_selective_restore_service import _SELECTED_SERVICE

_INTERRUPT = r'''
    import hashlib
    from tldw_chatbook.Backup_Recovery import activation
    from tldw_chatbook.Backup_Recovery.journal import Journal
    evidence = {
        str(path): [hashlib.sha256(content).hexdigest(), device, inode]
        for path, (content, device, inode) in preserved.items()
    }
    proof = Path.home() / 'preserved-evidence.json'
    proof.write_text(json.dumps(evidence))
    proof.chmod(0o600)
    original_bind = activation.bind_activation
    def stop_after_pair(*args, **kwargs):
        result = original_bind(*args, **kwargs)
        if BOUNDARY == 'paired':
            os._exit(91)
        return result
    activation.bind_activation = stop_after_pair
    original_append = Journal._append
    def stop_after_record(self, parent, event, evidence):
        result = original_append(self, parent, event, evidence)
        if event == BOUNDARY:
            os._exit(91)
        return result
    Journal._append = stop_after_record
'''


_FINISH = r'''
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from Tests import network_guard
network_guard.install()
child = subprocess.run([sys.executable, '-c', CHILD], capture_output=True,
                       text=True, timeout=90, check=False)
assert child.returncode == 91, child.stderr[-5000:] + child.stdout[-1000:]

# No runtime object or native lease crosses the process boundary.
from keyring.backends.null import Keyring
from tldw_chatbook.Backup_Recovery import bootstrap, credentials
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
store = KeyringServerCredentialStore(keyring_backend=Keyring())
credentials._credential_store = lambda: store
selector = Path(os.environ['TLDW_CONFIG_PATH'])
service = RecoveryService(Path.home() / 'recovery-control')
try:
    pending = service.pending_operations()
    assert len(pending) == 1, pending
    operation = pending[0]['operation_id']
    journal = _journal(service.control_root, operation)
    with journal._locked(exclusive=False) as parent:
        events = [row.event for row in journal._records(parent)]
    assert ('activation_recorded' in events) == (BOUNDARY == 'activation_recorded')
    assert 'committed' not in events
    assert not bootstrap.startup_permission(selector, bootstrap.default_bootstrap_root())[0]
    evidence = json.loads((Path.home() / 'preserved-evidence.json').read_text())
    def unchanged():
        for name, before in evidence.items():
            path = Path(name)
            info = path.stat()
            assert [hashlib.sha256(path.read_bytes()).hexdigest(), info.st_dev, info.st_ino] == before
    unchanged()
    result = service.wait(service.start_recovery(
        operation, action='finish', rollback_password=b'rollback-test-password'))
    assert result['state'] == 'succeeded', dict(result)
    assert tomllib.loads(selector.read_text())['general']['default_theme'] == 'textual-dark'
    unchanged()
    assert service.pending_operations() == ()
    assert bootstrap.startup_permission(selector, bootstrap.default_bootstrap_root())[0]
    with journal._locked(exclusive=False) as parent:
        events = [row.event for row in journal._records(parent)]
    assert events.count('activation_recorded') == events.count('committed') == 1
    assert not network_guard.blocked_attempts()
finally:
    service.close()
print('retired and reopened')
'''


@pytest.mark.parametrize("boundary", ["paired", "activation_recorded"])
def test_fresh_process_finishes_settings_after_pending_activation(tmp_path, boundary):
    marker = "    result = succeeded(service, service.start_restore(inspection, plan,"
    assert _SELECTED_SERVICE.count(marker) == 1
    child = _SELECTED_SERVICE.replace(marker, _INTERRUPT + "\n" + marker)
    child = child.replace("ENCRYPTED", "False").replace("SCENARIO", repr("settings"))
    child = child.replace("BOUNDARY", repr(boundary))
    script = _FINISH.replace("BOUNDARY", repr(boundary)).replace("CHILD", repr(child))
    _run(tmp_path, "settings-finish", boundary, script=script, timeout=150)
