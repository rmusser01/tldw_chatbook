"""Restore selects whole available groups and reviews target dependents."""

import hashlib
import json
from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_archive_reader import manifest
from Tests.Backup_Recovery.test_first_binding_absent_sqlite import (
    absent_database as absent_database,  # noqa: PLC0414 - shared pytest fixture
)
from tldw_chatbook.Backup_Recovery.archive_models import ArchiveManifest
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem


def _document(owners=("config", "db.prompts.primary", "db.media.primary")):
    doc = manifest()
    file = doc["files"][0]
    root = doc["directories"][0]
    doc["owners"] = [{"owner_id": owner, "schema_version": 1, "capabilities": []} for owner in owners]
    doc["files"], doc["directories"], doc["producer_inventory"], doc["dependency_groups"] = [], [], [], []
    for owner in owners:
        key = "profile:profile:" + owner
        root_id = "root:" + owner
        dependencies = [] if owner == "config" else ["profile:profile:config"]
        doc["files"].append({**file, "logical_id": key, "owner_id": owner, "root_id": root_id, "parent_id": root_id, "payload": "payload/" + owner})
        doc["directories"].append({**root, "logical_id": root_id, "root_id": root_id, "synthetic": True})
        doc["producer_inventory"].extend([
            {"logical_id": key, "owner_id": owner, "status": "included", "dependencies": dependencies},
            {"logical_id": root_id, "owner_id": owner, "status": "included_directory", "dependencies": []},
        ])
        doc["dependency_groups"].append({"group_id": "group:" + hashlib.sha256(key.encode()).hexdigest(), "members": sorted([key, *dependencies]), "complete": True})
    return ArchiveManifest.model_validate_json(json.dumps(doc))


def test_restore_selection_excludes_unselected_roots_and_marks_config_support():
    from tldw_chatbook.Backup_Recovery.restore_groups import (
        resolve_archive_groups,
        selected_archive_ids,
    )

    doc = _document()
    scope = resolve_archive_groups(doc, ("prompts",))
    assert scope.effective_groups == ("prompts",)
    assert scope.support_ids == ("profile:profile:config",)
    assert selected_archive_ids(doc, scope, retain_config=True) == frozenset({
        "profile:profile:db.prompts.primary", "root:db.prompts.primary"
    })


def test_isolated_selection_keeps_support_config_for_the_new_profile():
    from tldw_chatbook.Backup_Recovery.restore_groups import (
        resolve_archive_groups,
        selected_archive_ids,
    )

    doc = _document()
    scope = resolve_archive_groups(doc, ("prompts",))
    assert "profile:profile:config" in selected_archive_ids(doc, scope, retain_config=False)


def test_selective_archive_defaults_to_saved_groups_without_selecting_settings():
    from tldw_chatbook.Backup_Recovery.archive_models import GroupScope
    from tldw_chatbook.Backup_Recovery.restore_groups import (
        available_group_ids,
        resolve_archive_groups,
    )

    doc = _document(("config", "db.prompts.primary")).model_copy(update={
        "format_version": 2,
        "group_scope": GroupScope(requested_groups=("prompts",), effective_groups=("prompts",), support_ids=("profile:profile:config",)),
    })
    assert available_group_ids(doc) == ("prompts",)
    scope = resolve_archive_groups(doc, None)
    assert scope.requested_groups is None
    assert scope.effective_groups == ("prompts",)


def test_target_reverse_dependency_adds_required_group_to_review():
    from tldw_chatbook.Backup_Recovery.restore_groups import resolve_archive_groups

    media = StorageItem("db.media.primary", "local:media", None, "included", ())
    workspace = StorageItem("db.workspaces", "local:workspace", None, "included", (media.logical_id,))
    target = Inventory((media, workspace), True, "local", ())
    doc = _document(("config", "db.media.primary", "db.workspaces"))
    scope = resolve_archive_groups(doc, ("library",), target=target)
    assert scope.effective_groups == ("library", "workspaces")
    assert scope.required_groups == ("workspaces",)


def test_unavailable_reverse_dependency_refuses_instead_of_retiring_unselected_data():
    from tldw_chatbook.Backup_Recovery.restore_groups import resolve_archive_groups

    media = StorageItem("db.media.primary", "local:media", None, "included", ())
    projection = StorageItem("rag.projections", "local:projection", None, "included", (media.logical_id,))
    target = Inventory((media, projection), True, "local", ())
    with pytest.raises(ValueError, match="required_target_group_unavailable:retrieval"):
        resolve_archive_groups(_document(), ("library",), target=target)


def test_absent_target_dependency_does_not_require_restoring_an_unselected_group():
    from tldw_chatbook.Backup_Recovery.restore_groups import resolve_archive_groups

    prompts = StorageItem("db.prompts.primary", "local:prompts", None, "included", ())
    collections = StorageItem("db.library_collections", "local:collections", None,
                              "missing_required", (prompts.logical_id,))
    scope = resolve_archive_groups(_document(("config", "db.prompts.primary")),
                                   ("prompts",), target=Inventory((prompts, collections), False,
                                                                 "local", ("missing_required",)))
    assert scope.effective_groups == ("prompts",)


def test_target_config_dependency_does_not_implicitly_select_every_group():
    from tldw_chatbook.Backup_Recovery.restore_groups import resolve_archive_groups

    config = StorageItem("config", "local:config", None, "included", ())
    media = StorageItem("db.media.primary", "local:media", None, "included", (config.logical_id,))
    scope = resolve_archive_groups(_document(), ("settings",), target=Inventory((config, media), True, "local", ()))
    assert scope.effective_groups == ("settings",)


def test_unknown_atomic_dependency_group_adds_linked_group_to_review():
    from tldw_chatbook.Backup_Recovery.archive_models import DependencyGroup
    from tldw_chatbook.Backup_Recovery.restore_groups import resolve_archive_groups

    doc = _document()
    doc = doc.model_copy(update={"dependency_groups": (*doc.dependency_groups,
        DependencyGroup(group_id="legacy-cohort", members=(
            "profile:profile:db.prompts.primary", "profile:profile:db.media.primary"
        ), complete=True),
    )})
    scope = resolve_archive_groups(doc, ("prompts",))
    assert scope.effective_groups == ("library", "prompts")
    assert scope.required_groups == ("library",)


def test_modified_writer_group_does_not_gain_directed_semantics():
    from tldw_chatbook.Backup_Recovery.restore_groups import resolve_archive_groups

    doc = _document()
    groups = list(doc.dependency_groups)
    groups[1] = groups[1].model_copy(update={"members": (
        "profile:profile:config", "profile:profile:db.prompts.primary",
        "profile:profile:db.media.primary",
    )})
    doc = doc.model_copy(update={"dependency_groups": tuple(groups)})
    assert resolve_archive_groups(doc, ("prompts",)).required_groups == ("library",)


def test_explicit_destinations_cannot_omit_an_independent_root_in_selected_group(tmp_path):
    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    document = _document(("config", "ui.state")).model_dump(mode="json")
    archive = sealed(tmp_path, data=b"hello", mutate=lambda doc: (doc.clear(), doc.update(document)))
    with pytest.raises(ValueError, match="data_group_destination_missing"):
        plan_restore(archive, mode="isolated", target=None, data_groups=("settings",),
                     destinations={"root:config": tmp_path / "config"})


def test_empty_deferred_directory_keeps_setup_destination_visible(tmp_path):
    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    document = _document(("config", "tts.voices")).model_dump(mode="json")
    document["files"] = [row for row in document["files"] if row["owner_id"] == "config"]
    document["producer_inventory"] = [row for row in document["producer_inventory"]
                                      if row["logical_id"] != "profile:profile:tts.voices"]
    document["dependency_groups"] = document["dependency_groups"][:1]
    document["directories"][1]["synthetic"] = False
    archive = sealed(tmp_path, data=b"hello", mutate=lambda doc: (doc.clear(), doc.update(document)))
    service = RecoveryService(tmp_path / "control")
    try:
        operation = service.start_inspection(archive.path, password=None)
        assert service.wait(operation)["state"] == "succeeded"
        summary = service.summary(operation)
        assert summary["setup_destination_required"]
        assert summary["setup_group_ids"] == ("audio",)
    finally:
        service.close()


@pytest.mark.parametrize("groups", [(), ("not-a-group",), ("prompts", "prompts"), ("models",)])
def test_restore_refuses_invalid_or_unavailable_selection(groups):
    from tldw_chatbook.Backup_Recovery.restore_groups import resolve_archive_groups

    with pytest.raises(ValueError):
        resolve_archive_groups(_document(), groups)


@pytest.fixture
def absent_prompt_restore_case(absent_database, tmp_path, monkeypatch):
    """Archive a real closed Prompts store, then discover its actual absence."""
    import zipfile
    from threading import Event
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import bootstrap, restore_plan
    from tldw_chatbook.Backup_Recovery.archive_reader import acquire
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads

    selector, config, baseline, unselected = absent_database
    selected = next(row for row in baseline.items if row.owner == "db.prompts.primary")
    owners = {owner.owner_id: owner for owner in install_adapters()}
    content = {"config": selector.read_bytes(), selected.owner: selected.path.read_bytes()}
    document = _document(("config", selected.owner)).model_dump(mode="json")
    for row in document["files"]:
        payload = content[row["owner_id"]]
        row.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                   relative_path=selector.name if row["owner_id"] == "config" else selected.path.name)
    for row in document["owners"]:
        row["schema_version"] = max(owners[row["owner_id"]].schema_policy().versions)
    source = tmp_path / "selected-prompts.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("manifest.json", json.dumps(document))
        for row in document["files"]:
            archive.writestr(row["payload"], content[row["owner_id"]])
    archive = acquire(source, tmp_path / "selected-acquired", ArchiveLimits(), None, Event())
    selected.path.unlink()
    assert not any(selected.path.with_name(selected.path.name + suffix).exists()
                   for suffix in ("", "-wal", "-shm", "-journal"))
    configured = {**config, DISCOVERY_CONTEXT_KEY: DiscoveryContext(
        selector, baseline.items[0].logical_id.split(":")[1])}
    with _preview_reads():
        target = classify_entries(tuple(row for owner in ("config", selected.owner, unselected.owner)
                                        for row in owners[owner].discover(configured)))
    assert target.issues == ("missing_required",)
    assert next(row for row in target.items if row.owner == selected.owner).status == "missing_required"
    root = tmp_path / "absence-bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    admission_authority(root)
    relation = restore_plan.RetainedConfig(
        "profile:profile:config", baseline.items[0].logical_id, selector,
        restore_plan.retained_config_observation(selector),
    )

    def review(observed=target):
        return restore_plan.plan_restore(
            archive, mode="replace", destinations={"root:" + selected.owner: selected.path.parent},
            target=observed, retained_configs=(relation,), data_groups=("prompts",),
        )

    return SimpleNamespace(
        review=review, target=target, path=selected.path, selector=selector, source=source,
        unchanged={path: (path.read_bytes(), path.stat().st_ino) for path in (selector, source)},
        unselected=unselected,
    )


def test_selected_absent_sqlite_plans_creation_without_changing_inventory(absent_prompt_restore_case):
    case = absent_prompt_restore_case
    plan = case.review()
    assert plan.restore == (("profile:profile:db.prompts.primary", case.path),)
    assert not plan.retire
    assert plan.target is case.target and not plan.target.complete
    assert (case.unselected.logical_id, case.unselected.path) in plan.preserve
    assert all((path.read_bytes(), path.stat().st_ino) == before for path, before in case.unchanged.items())


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_selected_absent_sqlite_refuses_arrival_before_review(absent_prompt_restore_case, suffix):
    case = absent_prompt_restore_case
    case.path.with_name(case.path.name + suffix).write_bytes(b"new local state")
    with pytest.raises(ValueError, match="target_owner_unclassified|target_changed"):
        case.review()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_selected_absent_sqlite_recheck_refuses_late_companion(absent_prompt_restore_case, suffix):
    from tldw_chatbook.Backup_Recovery.restore_plan import recheck_targets

    case = absent_prompt_restore_case
    plan = case.review()
    case.path.with_name(case.path.name + suffix).write_bytes(b"arrived after review")
    with pytest.raises(ValueError, match="target_changed"):
        recheck_targets(plan)
    assert all((path.read_bytes(), path.stat().st_ino) == before for path, before in case.unchanged.items())


@pytest.mark.parametrize("owner,status", [
    ("unknown", "missing_required"), ("ui.state", "missing_required"),
    ("db.media.primary", "missing_required"), ("db.prompts.primary", "unavailable"),
])
def test_selected_absence_does_not_borrow_another_owner_or_unavailable_status(
    absent_prompt_restore_case, owner, status,
):
    case = absent_prompt_restore_case
    target = replace(case.target, items=tuple(
        replace(row, owner=owner, status=status) if row.path == case.path else row
        for row in case.target.items))
    with pytest.raises(ValueError, match="target_owner_unclassified|target_owner_mismatch"):
        case.review(target)


def _personal_context_archive(tmp_path, data):
    from Tests.Backup_Recovery.test_restore_plan import sealed

    document = _document(("config", "personal_context.repository")).model_dump(mode="json")
    for row in document["owners"]:
        if row["owner_id"] == "personal_context.repository":
            row["schema_version"] = 9
    for row in document["files"]:
        row.update(
            relative_path="config.toml"
            if row["owner_id"] == "config"
            else "tldw_chatbook_personal_context.db",
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    return sealed(
        tmp_path, data=data, mutate=lambda doc: (doc.clear(), doc.update(document))
    )


@pytest.mark.parametrize("mode", ["replace", "isolated"])
def test_restore_plan_refuses_an_unreadable_personal_context_copy(tmp_path, mode):
    """TASK-40135/40136: every restore now publishes the repository, so its plan
    reads the archived copy first; one that is not a database refuses by name
    instead of planning an outcome it cannot state."""
    from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed
    from tldw_chatbook.Backup_Recovery.restore_groups import available_group_ids
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    data = b'[general]\nusers_name="default_user"\n'
    archive = _personal_context_archive(tmp_path, data)
    assert "personal_context" in available_group_ids(verify_sealed(archive))
    restored = tmp_path / "restored"
    with pytest.raises(ValueError, match="^personal_context_validation_unavailable$"):
        plan_restore(
            archive,
            mode="isolated",
            destinations={
                "root:config": restored / "config",
                "root:personal_context.repository": restored / "data",
                "profile:profile:paths.data_dir": restored / "data",
            },
            target=None,
            profile_names={"profile": "restored"},
        ) if mode == "isolated" else _replace_plan(tmp_path, archive, data)


def _replace_plan(tmp_path, archive, data):
    from tldw_chatbook.Backup_Recovery.models import FileMetadata
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    live = tmp_path / "live"
    live.mkdir(mode=0o700)
    profile = hashlib.sha256(str(live / "config.toml").encode()).hexdigest()[:24]
    items = []
    for owner, name in (
        ("config", "config.toml"),
        ("personal_context.repository", "tldw_chatbook_personal_context.db"),
    ):
        path = live / name
        path.write_bytes(data)
        path.chmod(0o600)
        key = "profile:" + profile + ":" + owner
        items.append(
            StorageItem(
                owner, key, path, "included",
                () if owner == "config" else ("profile:" + profile + ":config",),
                metadata=FileMetadata(
                    1, key, name, None, "file", 0o600, path.stat().st_mtime_ns, "private"
                ),
            )
        )
    return plan_restore(
        archive,
        mode="replace",
        destinations={"root:config": live, "root:personal_context.repository": live},
        target=Inventory(tuple(items), True, "independent-local-review", ()),
        profile_names={"profile": "default_user"},
    )


# TASK-40135: real backups and replace restores of a profile holding Personal
# Context, through RecoveryService in a scratch child. Both "devices" are
# in-process SecureKeyring dicts; the child never reaches an OS keyring.
_PERSONAL_CONTEXT_RESTORE = r'''
import json
import os
import tomllib
from pathlib import Path
from threading import Event
from Tests import network_guard
network_guard.install()
os.environ['PYTHON_KEYRING_BACKEND'] = 'keyring.backends.null.Keyring'
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from Tests.Personal_Context.test_key_protector import SecureKeyring
devices = {'a': SecureKeyring(), 'b': SecureKeyring()}
current = {}

def use(name):
    current['device'] = devices[name]

use('a')
from Tests.Architecture.test_backup_owner_inventory import _device_only_record
from tldw_chatbook.Backup_Recovery import bootstrap, credentials, publication
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Personal_Context.key_protector import (
    KeyringProfileKeyProtector, ProfileKeyMaterial, ProfileLockedError)
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import PersonalContextService
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore

credentials._credential_store = lambda: KeyringServerCredentialStore(
    keyring_backend=current['device'])
SCENARIO, MODE = SCENARIO_VALUE, MODE_VALUE
source = Path(os.environ['TLDW_CONFIG_PATH'])
SERVICE = 'tldw_chatbook.personal_context'
GROUP = ('personal_context',)
ROLLBACK = b'rollback-test-password'

PREPARE = """
import os
from pathlib import Path
from Tests import network_guard
network_guard.install()
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
selector = Path(os.environ['TLDW_CONFIG_PATH'])
_populate_required_dependencies(preview_capture((selector,), options={'allow_partial': True}))
assert not network_guard.blocked_attempts()
"""

def set_config(name):
    import subprocess
    import sys
    data = Path(os.environ['XDG_DATA_HOME']) / name
    data.mkdir(mode=0o700, exist_ok=True)
    source.write_text('[general]\nusers_name="default_user"\n[paths]\ndata_dir='
                      + json.dumps(str(data)) + '\n')
    source.chmod(0o600)
    root = user_data_dir(tomllib.loads(source.read_text()))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The installed required stores, as a real profile has them.
    done = subprocess.run([sys.executable, '-c', PREPARE], capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-4000:]
    return root / 'tldw_chatbook_personal_context.db'

def repository(path):
    return PersonalContextRepository(
        path, key_protector=KeyringProfileKeyProtector(current['device']))

def records(path):
    return sorted(row.record_id for row in repository(path).list_records())

def account(path):
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute('SELECT key_account FROM profile_meta').fetchone()[0]

def tables(path):
    """Every row of every table, without ``profile_meta.key_account``."""
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(path)) as connection:
        dump = {}
        for (name,) in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name"):
            cursor = connection.execute(f'SELECT * FROM "{name}"')
            columns = [column[0] for column in cursor.description]
            dump[name] = sorted(
                tuple(v for c, v in zip(columns, row)
                      if (name, c) != ('profile_meta', 'key_account'))
                for row in cursor)
        return dump

STATUS = """
import os, sys
os.environ['PYTHON_KEYRING_BACKEND'] = 'keyring.backends.null.Keyring'
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Personal_Context.bootstrap import bootstrap_personal_context_service
# As production starts it: the default protector, here with no secure keyring,
# as an isolated copy (and a device without one) always has.
status = bootstrap_personal_context_service(db_path=sys.argv[1]).status()
print(status.locked, status.reason_code)
"""

def status(path):
    """A fresh start (another process, null keyring, default protector) reads the
    locked reason from the copy alone."""
    import subprocess
    import sys
    done = subprocess.run([sys.executable, '-c', STATUS, str(path)], capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-4000:]
    return done.stdout.split()[-2:]

def succeeded(operation):
    result = service.wait(operation)
    assert result['state'] == 'succeeded', dict(result)
    return result

RETENTION = []

def reviewed(start, acknowledged=(), state='succeeded'):
    """Run one operation; an unacknowledged credential review aborts and retries."""
    result = service.wait(start(acknowledged))
    new = set(result['review_issues']) - set(acknowledged)
    if result['kind'] == 'restore' and 'credential_isolated_retention_required' in new:
        # TASK-41124: whether the review may say this restore installs the key.
        RETENTION.append(result['result'].get('personal_context_keys_installed', False))
    if result['state'] != 'succeeded' and new:
        pending = result['result'].get('journal_operation_id')
        if pending:
            assert succeeded(service.start_recovery(pending, action='abort'))['result']['aborted']
        return reviewed(start, tuple(sorted(set(acknowledged) | new)), state)
    assert result['state'] == state, dict(result)
    return result

def plan_for(acknowledged=()):
    return service.preview_restore(
        inspection, mode='replace', profile_bases={}, target_configs={profile: source},
        external_destinations={}, profile_names={},
        target=service.preview_backup((source,), options={}), data_groups=GROUP,
        acknowledged_credential_issues=acknowledged)

admission_authority(bootstrap.default_bootstrap_root())
live = set_config('fixture')
original = repository(live)
PersonalContextService(original).create_profile()
original.commit_record_version(_device_only_record(original), expected_version_id=None)
original_account = account(live)
# TASK-40139 AC#7: this device's link keys (device wrapping key, a staged
# integrity key, a dataset staging key) must never reach an archive.
LINK = 'tldw_chatbook.personal_context.link'
CANARIES = {('device-wrapping-rsa-v1', 'rsa-private-pkcs8-v1:LINKCANARYWRAP'),
            ('staged-binding', 'staged-integrity-v1:LINKCANARYSTAGED'),
            ('storage-binding', 'dataset-staging-v1:LINKCANARYSTORAGE')}
for name, value in CANARIES:
    current['device'].values[(LINK, name)] = value
service = RecoveryService(Path.home() / 'recovery-control')
try:
    destination = Path.home() / 'pc.tldw-backup.zip.age'
    password = b'pc-test-password'

    def backup(acknowledged):
        options = {'data_groups': GROUP, 'encrypted': True, 'credential_mode': MODE,
                   'acknowledged_credential_issues': acknowledged}
        preview = service.preview_backup((source,), options=options)
        return service.start_backup((source,), preview.scope_digest, destination,
                                    options=options, password=password)

    reviewed(backup)
    archived = tables(live)
    inspection = service.start_inspection(destination, password=password)
    succeeded(inspection)
    import zipfile
    with zipfile.ZipFile(service.inspection(inspection).path) as sealed:
        members = b''.join(sealed.read(name) for name in sealed.namelist())
    assert b'personal_context' in members  # the repository and its record are there
    # Positive control: the scan sees captured keyring values (the profile key).
    assert (current['device'].values[(SERVICE, original_account)].encode()
            in members) is (MODE == 'include')
    assert not [value for _, value in CANARIES if value.encode() in members]
    assert LINK.encode() not in members
    profile = service.summary(inspection)['profile_ids'][0]
    assert 'personal_context' in {
        row['group_id'] for row in service.summary(inspection)['data_groups']}
    if SCENARIO.startswith('isolated'):
        # TASK-40136: an isolated copy is published locked, keyring untouched.
        if SCENARIO == 'isolated_elsewhere':
            use('b')  # no account on this device opens the copy
        before = dict(current['device'].values)
        (Path.home() / 'isolated').mkdir(mode=0o700)
        base = Path.home() / 'isolated' / 'copy'
        result = reviewed(lambda ack: service.start_restore(inspection, service.preview_restore(
            inspection, mode='isolated', profile_bases={profile: base},
            external_destinations={}, profile_names={profile: 'restored'}, target=None,
            data_groups=GROUP, acknowledged_credential_issues=ack)))
        (copy,) = [p for p in base.rglob('tldw_chatbook_personal_context.db')]
        reason = ('personal_context_key_not_in_backup' if SCENARIO == 'isolated_elsewhere'
                  else 'personal_context_keyring_unavailable')
        assert account(copy) == 'recovery:setup_required:' + reason
        assert current['device'].values == before
        assert tables(copy) == archived
        assert status(copy) == ['True', reason]
        assert account(live) == original_account
        print('retired and reopened')
        raise SystemExit(0)
    if SCENARIO == 'no_keyring':
        # TASK-40136 AC#1/#6: no secure keyring (the production store refuses),
        # then a second restore from the same archive once one is there.
        from tldw_chatbook.runtime_policy.server_credentials import CredentialStoreUnavailable
        store = credentials._credential_store

        def no_store():
            raise CredentialStoreUnavailable('no secure keyring')

        credentials._credential_store = no_store
        before = dict(current['device'].values)
        locked = 'personal_context_restore_locked:personal_context_key_install_unavailable:'
        assert any(i.startswith(locked) for i in plan_for().issues)
        reviewed(lambda ack: service.start_restore(
            inspection, plan_for(ack), rollback_password=ROLLBACK))
        reason = 'personal_context_key_install_unavailable'
        assert account(live) == 'recovery:setup_required:' + reason
        assert current['device'].values == before
        assert tables(live) == archived
        assert status(live) == ['True', reason]
        assert RETENTION == [False], RETENTION  # a locked key is not installed
        credentials._credential_store = store
        assert 'personal_context_restore_readable:' + next(
            key for key, path in plan_for().restore if path == live) in plan_for().issues
        reviewed(lambda ack: service.start_restore(
            inspection, plan_for(ack), rollback_password=ROLLBACK))
        assert account(live).startswith('personal-context:recovery_')
        assert records(live) == ['record-device-only']
        assert RETENTION == [False, True], RETENTION
        print('retired and reopened')
        raise SystemExit(0)
    if SCENARIO == 'other_device':
        # The relocation advice: another device (empty keyring), another path.
        use('b')
        target = set_config('elsewhere')
        assert target != live and not target.exists()
    else:
        target = live
        # A record made after the backup: the restore replaces it.
        from tldw_profile_core import PreferencePayload, SemanticKey
        original.commit_record_version(
            _device_only_record(original).model_copy(update={
                'record_id': 'record-after-backup',
                'version_id': 'record-after-backup-v1',
                'payload': PreferencePayload(
                    subject='response.length', polarity='like', value='v'),
                'semantic_key': SemanticKey(
                    namespace='preference', subject='response.length'),
            }), expected_version_id=None)
        assert records(live) == ['record-after-backup', 'record-device-only']
    before = dict(current['device'].values)
    plan = plan_for()
    assert [i for i in plan.issues if i.startswith('personal_context_restore_')] == [
        'personal_context_restore_readable:' + next(
            key for key, path in plan.restore if path == target)], plan.issues
    assert target in {path for _, path in plan.restore}
    if SCENARIO == 'finalize_fails':
        # Review r14pcc-rev2: a finalize failure inside replace() rolls back on
        # its own; the account this restore installed must go with it.
        finalize = publication.finalize_candidate

        def refused(*args, **kwargs):
            raise ValueError('injected finalize failure')

        publication.finalize_candidate = refused
        reviewed(lambda ack: service.start_restore(
            inspection, plan_for(ack), rollback_password=ROLLBACK), state='failed')
        publication.finalize_candidate = finalize
        assert account(live) == original_account
        assert records(live) == ['record-after-backup', 'record-device-only']
        assert {k for k in current['device'].values if k[0] == SERVICE} - set(before) == set()
        assert {k: v for k, v in current['device'].values.items() if k in before} == before
        print('retired and reopened')
        raise SystemExit(0)
    if SCENARIO in ('finish', 'rollback'):
        # AC#9: interrupted after the key step, before publication completes.
        class Stop(Exception):
            pass

        publish = publication.publish_candidate

        def stopped(*args, **kwargs):
            raise Stop('interrupted after the key step')

        publication.publish_candidate = stopped
        pending = reviewed(lambda ack: service.start_restore(
            inspection, plan_for(ack), rollback_password=ROLLBACK),
            state='recovery_required')['result']['journal_operation_id']
        publication.publish_candidate = publish
        installed = {k for k in current['device'].values if k[0] == SERVICE} - set(before)
        assert len(installed) == 1, installed
        assert account(live) == original_account
        succeeded(service.start_recovery(
            pending, action='finish' if SCENARIO == 'finish' else 'rollback',
            rollback_password=ROLLBACK))
        named = account(live)
        assert (SERVICE, named) in current['device'].values
        if SCENARIO == 'finish':
            assert {(SERVICE, named)} == installed
            assert records(live) == ['record-device-only']
        else:
            assert named == original_account
            assert records(live) == ['record-after-backup', 'record-device-only']
            # Spec Q4: the rollback deletes the account this restore created.
            assert not installed & set(current['device'].values), installed
        assert {k: v for k, v in current['device'].values.items() if k in before} == before
        print('retired and reopened')
        raise SystemExit(0)
    restored = reviewed(lambda ack: service.start_restore(
        inspection, plan_for(ack), rollback_password=ROLLBACK))
    assert 'personal_context_key_unverified' not in restored['issues']
    # TASK-41124: the only retained credential is the key this restore installs.
    assert RETENTION == ([True] if MODE == 'include' else []), RETENTION
    # AC#1/#2/#4/#11: readable, from a new account; every earlier entry unchanged.
    restored_account = account(target)
    assert restored_account.startswith('personal-context:recovery_')
    assert restored_account != original_account
    assert {k: v for k, v in current['device'].values.items() if k in before} == before
    assert records(target) == ['record-device-only']
    if SCENARIO == 'other_device':
        assert {k for k in current['device'].values if k[0] == SERVICE} == {
            (SERVICE, restored_account)}
        print('retired and reopened')
        raise SystemExit(0)
    # AC#7/#8: the restored profile links (its own account is replaced) or is
    # removed (its own account is deleted); a rollback still opens the originals.
    protector = KeyringProfileKeyProtector(current['device'])
    if SCENARIO == 'link':
        protector.replace(restored_account, ProfileKeyMaterial(b'l' * 32, b'k' * 32, 2))
    elif SCENARIO == 'remove':
        PersonalContextService(repository(target)).remove_local_profile(
            confirm_only_copy=True)
        assert (SERVICE, restored_account) not in current['device'].values
    assert current['device'].values[(SERVICE, original_account)] == before[
        (SERVICE, original_account)]
    operation = restored['result']['journal_operation_id']

    def rollback(acknowledged):
        current_target = service.preview_backup((source,), options={})
        plan = service.preview_rollback(
            operation, old_password=ROLLBACK, target=current_target,
            acknowledged_credential_issues=acknowledged)
        return service.start_rollback(operation, plan, old_password=ROLLBACK,
                                      new_password=b'second-safety-password')

    reviewed(rollback)
    assert account(live) == original_account
    assert records(live) == ['record-after-backup', 'record-device-only']
    assert not network_guard.blocked_attempts()
finally:
    service.close()
print('retired and reopened')
'''


@pytest.mark.parametrize(
    ("scenario", "mode"),
    [
        ("link", "include"),
        ("remove", "include"),
        ("other_device", "include"),
        ("link", "exclude"),
        ("finish", "include"),
        ("rollback", "include"),
        ("no_keyring", "include"),
        ("isolated", "include"),
        ("isolated_elsewhere", "exclude"),
        ("finalize_fails", "include"),
    ],
)
def test_replace_restore_makes_personal_context_readable_and_rolls_back(
    tmp_path, scenario, mode
):
    """TASK-40135 AC#1, #2, #4, #7-#11: a replace restore (with the key in the
    backup, or same-device without it) opens and lists the restored records from
    a new account, leaves every earlier entry unchanged, and a rollback after the
    restored profile links or is removed opens the originals. ``other_device`` is
    the portable relocation advice: another keyring, another data path, a backup
    with credentials, then Replace. ``finish``/``rollback``: interrupted after
    the key step, either recovery leaves a reference to an account that exists
    and opens it. TASK-40136: ``no_keyring`` (AC#1, #2, #6) and the two isolated
    cases (AC#1, #2) end locked with their reason, keyring and rows untouched."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = _PERSONAL_CONTEXT_RESTORE.replace("SCENARIO_VALUE", repr(scenario)).replace(
        "MODE_VALUE", repr(mode)
    )
    _run(tmp_path, "personal-context", "success", script=script, timeout=300)
