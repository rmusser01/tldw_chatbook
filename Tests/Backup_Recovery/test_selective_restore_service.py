"""Selected service replacement and rollback preserve actual unselected stores."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SELECTED_SERVICE = r'''
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from threading import Event
from Tests import network_guard
network_guard.install()
os.environ['PYTHON_KEYRING_BACKEND'] = 'keyring.backends.null.Keyring'
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap, credentials
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.data_groups import group_for_owner
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback, execute_rollback
from tldw_chatbook.Backup_Recovery.profile_paths import database_path
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore

# This fixture has a known-empty credential store. The production default
# correctly rejects the null backend as an unavailable OS credential store.
credential_store = KeyringServerCredentialStore(keyring_backend=Keyring())
credentials._credential_store = lambda: credential_store

scenario = SCENARIO
group = 'settings' if scenario == 'settings' else 'prompts'
source = Path(os.environ['TLDW_CONFIG_PATH'])
data = Path(os.environ['XDG_DATA_HOME']) / 'fixture'
data.mkdir(mode=0o700)
source_prompt = None
if scenario == 'absent':
    source_store = Path.home() / 'archive-source'
    source_store.mkdir(mode=0o700)
    default_prompt = database_path({'paths': {'data_dir': str(data)}}, 'prompts_db_path')
    source_prompt = source_store / default_prompt.name

def set_config(theme, prompt=None):
    raw = ('[general]\nusers_name="default_user"\ndefault_theme=' + json.dumps(theme)
           + '\n[paths]\ndata_dir=' + json.dumps(str(data)) + '\n')
    if prompt is not None:
        raw += '[database]\nprompts_db_path=' + json.dumps(str(prompt)) + '\n'
    source.write_text(raw)
    source.chmod(0o600)

set_config('textual-dark', source_prompt)
admission_authority(bootstrap.default_bootstrap_root())
assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
producer = """
import json
import os
import sys
import tomllib
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.profile_paths import database_path
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
selector = Path(os.environ['TLDW_CONFIG_PATH'])
if sys.argv[1] == 'prepare':
    from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
    from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
    baseline = preview_capture((selector,), options={'allow_partial': True})
    _populate_required_dependencies(baseline)
path = database_path(tomllib.loads(selector.read_text()), 'prompts_db_path')
assert path.is_file(), 'readback must never create an absent target'
store = PromptsDatabase(path, 'selected-restore-fixture')
try:
    if sys.argv[1] == 'read':
        for name, expected in json.loads(sys.argv[2]).items():
            row = store.get_prompt_by_name(name)
            assert (row['user_prompt'] if row is not None else None) == expected, name
    else:
        store.add_prompt(sys.argv[2], 'fixture', 'Description', user_prompt=sys.argv[3])
finally:
    store.close()
assert not network_guard.blocked_attempts()
"""

def prompt_process(*args):
    result = subprocess.run([sys.executable, '-c', producer, *args],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-4000:] + result.stdout[-1000:]

def read_prompts(expected):
    prompt_process('read', json.dumps(expected))

def succeeded(service, operation):
    result = service.wait(operation)
    assert result['state'] == 'succeeded', dict(result)
    return result

def observed(path):
    info = path.stat()
    return path.read_bytes(), info.st_dev, info.st_ino

def assert_preserved():
    assert all(observed(path) == before for path, before in preserved.items())

saved_text = 'Saved selected prompt content'
newer_text = 'Newer selected prompt after backup'
edited_text = 'Prompt edited after first restoration'
prompt_process('prepare', 'prepare', saved_text)
read_prompts({'prepare': saved_text, 'newer': None})
password = b'selected-test-password' if ENCRYPTED else None
destination = Path.home() / (group + ('.tldw-backup.zip.age' if ENCRYPTED else '.tldw-backup.zip'))
options = {'data_groups': (group,), 'encrypted': ENCRYPTED}
service = RecoveryService(Path.home() / 'recovery-control')
try:
    preview = service.preview_backup((source,), options=options)
    assert preview.complete, preview.issues
    succeeded(service, service.start_backup((source,), preview.scope_digest, destination,
                                            options=options, password=password))
    inspection = service.start_inspection(destination, password=password)
    succeeded(service, inspection)
    assert service.summary(inspection)['group_scope']['effective_groups'] == (group,)
    # Capture initializes authority but never enrolls a profile. First confirmed
    # replacement must prove and bind the current local configuration itself.
    assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
    if scenario == 'absent':
        # Preserve the original source DB outside the target profile. The new
        # local selector names a destination that has never held a database.
        set_config('textual-dark')
        expected_prompt = database_path(tomllib.loads(source.read_text()), 'prompts_db_path')
        assert expected_prompt != source_prompt and not expected_prompt.exists()
    else:
        prompt_process('add', 'newer', newer_text)
        read_prompts({'prepare': saved_text, 'newer': newer_text})
        if scenario == 'settings':
            set_config('textual-light')
    target = service.preview_backup((source,), options={})
    # Unused installed databases remain absent; use the real full inventory
    # throughout and let the selected-scope policy decide eligibility.
    preserved = {
        row.path: observed(row.path)
        for row in target.items
        if row.status == 'included' and row.path is not None
        and group_for_owner(row.owner) not in {None, group}
    }
    assert next(row.path for row in target.items if row.owner == 'db.media.primary') in preserved
    prompt_row = next(row for row in target.items if row.owner == 'db.prompts.primary')
    prompt_path = prompt_row.path
    if scenario == 'absent':
        assert prompt_row.status == 'missing_required' and prompt_path == expected_prompt
        assert all(not os.path.lexists(str(prompt_path) + suffix)
                   for suffix in ('', '-wal', '-shm', '-journal'))
        preserved[source_prompt] = observed(source_prompt)
    if scenario == 'settings':
        assert prompt_path in preserved and source not in preserved
    else:
        assert source in preserved and prompt_path not in preserved
    local_config = source.read_bytes()
    profile = service.summary(inspection)['profile_ids'][0]
    plan = service.preview_restore(inspection, mode='replace', profile_bases={},
                                   target_configs={profile: source}, external_destinations={},
                                   profile_names={}, target=target, data_groups=(group,))
    assert plan.effective_groups == (group,)
    if scenario == 'settings':
        assert source in {path for _, path in plan.restore}
        assert not ({path for _, path in plan.restore} & set(preserved))
    else:
        assert {path for _, path in plan.restore} == {prompt_path}
    assert not plan.retire
    result = succeeded(service, service.start_restore(inspection, plan,
                                                       rollback_password=b'rollback-test-password'))
    assert_preserved()
    if scenario == 'settings':
        assert tomllib.loads(source.read_text())['general']['default_theme'] == 'textual-dark'
        read_prompts({'prepare': saved_text, 'newer': newer_text})
    else:
        read_prompts({'prepare': saved_text, 'newer': None})
    assert_preserved()
    if scenario == 'absent':
        prompt_process('add', 'after-restore', edited_text)
        read_prompts({'prepare': saved_text, 'after-restore': edited_text})
    operation = result['result']['journal_operation_id']
    current = service.preview_backup((source,), options={})
    rollback = preview_rollback(operation, control_root=service.control_root,
                                old_password=b'rollback-test-password', target=current, cancel=Event())
    assert rollback.effective_groups == (group,)
    assert rollback.requested_groups == plan.requested_groups
    assert rollback.required_groups == plan.required_groups
    if scenario == 'absent':
        assert rollback.restore == ()
        assert {path for _, path in rollback.retire} == {prompt_path}
    elif scenario == 'settings':
        assert source in {path for _, path in rollback.restore}
    else:
        assert {path for _, path in rollback.restore} == {prompt_path}
    assert not ({path for _, path in (*rollback.restore, *rollback.retire)} & set(preserved))
    rollback_result = service.wait(service.start_rollback(
        operation, rollback, old_password=b'rollback-test-password',
        new_password=b'second-safety-password'))
    if rollback_result['state'] == 'recovery_required' and rollback_result['review_issues']:
        # Settings imports may leave explicit redacted credential markers. Use
        # the normal review -> Abort untouched preparation -> reviewed retry flow.
        assert scenario == 'settings', dict(rollback_result)
        issues = tuple(rollback_result['review_issues'])
        assert all(code.startswith('credential_') for code in issues), issues
        assert_preserved()
        before_abort = source.read_bytes()
        pending = rollback_result['result']['journal_operation_id']
        aborted = succeeded(service, service.start_recovery(pending, action='abort'))
        assert aborted['result']['aborted']
        assert source.read_bytes() == before_abort
        assert_preserved()
        current = service.preview_backup((source,), options={})
        reviewed = service.preview_rollback(
            operation, old_password=b'rollback-test-password', target=current,
            acknowledged_credential_issues=issues)
        assert reviewed.effective_groups == rollback.effective_groups
        assert set(reviewed.acknowledged_credential_issues) == set(issues)
        rollback_result = service.wait(service.start_rollback(
            operation, reviewed, old_password=b'rollback-test-password',
            new_password=b'second-safety-password'))
    assert rollback_result['state'] == 'succeeded', dict(rollback_result)
    rollback_operation = rollback_result['result']['journal_operation_id']
    assert_preserved()
    if scenario == 'settings':
        assert source.read_bytes() == local_config
        assert tomllib.loads(source.read_text())['general']['default_theme'] == 'textual-light'
        read_prompts({'prepare': saved_text, 'newer': newer_text})
    elif scenario == 'prompts':
        # SQLite snapshots may update header counters; verify restored records
        # through the installed store. Unselected files remain byte/identity exact.
        read_prompts({'prepare': saved_text, 'newer': newer_text})
    else:
        # Readback intentionally stays closed while absent, so it cannot create
        # an empty DB and invalidate the authenticated rollback-to-absence proof.
        assert all(not os.path.lexists(str(prompt_path) + suffix)
                   for suffix in ('', '-wal', '-shm', '-journal'))
        absent_target = service.preview_backup((source,), options={})
        absent_row = next(row for row in absent_target.items if row.owner == 'db.prompts.primary')
        assert absent_row.path == prompt_path and absent_row.status == 'missing_required'
        undo = preview_rollback(rollback_operation, control_root=service.control_root,
                                old_password=b'second-safety-password', target=absent_target, cancel=Event())
        assert undo.effective_groups == (group,)
        assert undo.requested_groups == rollback.requested_groups
        assert undo.required_groups == rollback.required_groups
        assert {path for _, path in undo.restore} == {prompt_path}
        assert not undo.retire
        assert not ({path for _, path in undo.restore} & set(preserved))
        execute_rollback(rollback_operation, control_root=service.control_root,
                         old_password=b'second-safety-password', new_password=b'third-safety-password',
                         cancel=Event(), approved_plan=undo)
        read_prompts({'prepare': saved_text, 'after-restore': edited_text})
    assert_preserved()
    assert bootstrap.startup_permission(source, bootstrap.default_bootstrap_root())[0]
    assert not network_guard.blocked_attempts()
finally:
    service.close()
print('retired and reopened')
'''


def _service_case(tmp_path, encrypted, scenario):
    script = _SELECTED_SERVICE.replace("ENCRYPTED", repr(encrypted)).replace(
        "SCENARIO", repr(scenario)
    )
    _run(tmp_path, "selected-service", "success", script=script, timeout=180)


@pytest.mark.parametrize("encrypted", [False, True])
def test_prompt_backup_replacement_and_rollback_preserve_other_groups(
    tmp_path, encrypted
):
    _service_case(tmp_path, encrypted, "prompts")


@pytest.mark.parametrize("encrypted", [False, True])
def test_settings_backup_replacement_and_rollback_preserve_unselected_stores(
    tmp_path, encrypted
):
    _service_case(tmp_path, encrypted, "settings")


@pytest.mark.parametrize("encrypted", [False, True])
def test_prompt_restore_to_absence_then_rollback_and_undo_preserve_other_groups(
    tmp_path, encrypted
):
    _service_case(tmp_path, encrypted, "absent")
