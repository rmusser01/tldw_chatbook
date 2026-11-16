"""Explicit startup restrictions never widen fresh activation decisions."""

from textwrap import indent

import pytest

from Tests.Backup_Recovery.test_activation_control_observation import _SETUP
from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = (
    _SETUP
    + r"""
import json
from tldw_chatbook.Backup_Recovery import activation, generation_witnesses
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

consumer, behavior = route.split(':')
path = db.db_path
before_leases = set(storage._live_leases)
before_pending = set(storage._pending_acquisitions)
lease = storage.acquire_storage(path)
native_root, names = lease.execution_context(path)
assert native_root == root and 'profile' in names
hold = storage._holds[lease._key]
original_public_startup = bootstrap.startup_permission
original_activation = activation.activation_permission
calls = []
effects = []
refused = None
pair = root / ('activation-' + bootstrap._key(str(selector)) + '.json')
assert pair.exists()
before_header = path.read_bytes()[:16]
assert before_header == b'SQLite format 3\x00'

def restriction(actual_selector, actual_root, /):
    assert (actual_selector, actual_root) == (selector, root)
    calls.append((actual_selector, actual_root))
    if behavior == 'deny':
        return False, 'explicit_refusal'
    if behavior == 'raise':
        raise RuntimeError('explicit_refusal')
    if behavior == 'pending':
        register_pending(root, 'restriction-restore', ('profile',), control, (selector,))
    elif behavior == 'pair_loss':
        pair.unlink()
    else:
        raise AssertionError(behavior)
    return True, 'custom_allow_cannot_override_current_controls'

def old_activation(owner, *, config_selector=None, bootstrap_root=None,
                   namespaces=None, ordinary_only=False):
    calls.append(owner)
    return original_activation(owner, config_selector=config_selector,
                               bootstrap_root=bootstrap_root, namespaces=namespaces,
                               ordinary_only=ordinary_only)

try:
    if behavior == 'old_signature':
        activation.activation_permission = old_activation
        with activation.execution_scope(('config', 'config'), path, retained=lease) as allowed:
            assert allowed
            assert path.read_bytes()[:16] == before_header
            effects.append('actual_payload')
        assert effects == ['actual_payload']
        assert calls == ['config', 'config'], 'duplicate owners must remain separate checks'
    elif consumer == 'execution':
        with activation.execution_scope(
            ('config', 'db.agent_runs'), path, retained=lease,
            startup_restriction=restriction,
        ) as allowed:
            if allowed:
                effects.append('protected_payload')
        assert not allowed
    else:
        assert consumer == 'witness'
        try:
            generation_witnesses._witnesses(path, lease, startup_restriction=restriction)
            effects.append('published_witness')
        except (ValueError, RuntimeError) as error:
            refused = error
        assert refused is not None
        if behavior == 'raise':
            assert type(refused) is RuntimeError and str(refused) == 'explicit_refusal'
        else:
            assert type(refused) is ValueError and str(refused) == 'projection_generation_unavailable'
    if behavior != 'old_signature':
        assert calls == [(selector, root)]
        assert not effects
    assert bootstrap.startup_permission is original_public_startup
    assert path.read_bytes()[:16] == before_header
    assert lease.execution_context(path) == (native_root, names)
    assert set(storage._live_leases) == before_leases | {lease}
    assert set(storage._pending_acquisitions) == before_pending
    try:
        with authority.maintenance(names, .02):
            raise AssertionError('restriction lost actual native exclusion')
    except AdmissionTimeout:
        pass
finally:
    activation.activation_permission = original_activation
    lease.close()
assert set(storage._live_leases) == before_leases
assert set(storage._pending_acquisitions) == before_pending
assert hold.stop.is_set() and not hold.thread.is_alive() and hold.error is None
with authority.maintenance(names, 2):
    pass
assert not blocked_attempts()
print(json.dumps({'consumer': consumer, 'restriction': behavior,
                  'effects': effects, 'physical_retirement': True}))
print('retired and reopened')
"""
)
_SCRIPT = (
    "from Tests.windows_private_fixture_runner import user_fixture_default_owner\n"
    "with user_fixture_default_owner():\n" + indent(_SCRIPT, "    ")
)


@pytest.mark.parametrize("consumer", ["execution", "witness"])
@pytest.mark.parametrize("change", ["pending", "pair_loss"])
def test_custom_allow_still_observes_changed_controls(tmp_path, consumer, change):
    _run(tmp_path, consumer + ":" + change, "approved", script=_SCRIPT)


@pytest.mark.parametrize("behavior", ["deny", "raise"])
def test_witness_restriction_refuses_without_publication(tmp_path, behavior):
    _run(tmp_path, "witness:" + behavior, "approved", script=_SCRIPT)


def test_default_execution_preserves_existing_activation_callback_signature(tmp_path):
    _run(tmp_path, "execution:old_signature", "approved", script=_SCRIPT)
