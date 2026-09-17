"""Absent unselected SQL locators do not grant data-read or publication scope."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import (
    UNBOUND_NAMESPACE,
    admission_authority,
)
from tldw_chatbook.Backup_Recovery.inventory import classify_entries
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
from tldw_chatbook.Backup_Recovery.preserved_groups import check_preserved_group_paths
from tldw_chatbook.Backup_Recovery.profile_paths import database_path
from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan
from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads


def observed(path):
    info = path.stat()
    return path.read_bytes(), info.st_dev, info.st_ino


@pytest.fixture
def absent_scope(tmp_path, monkeypatch):
    import hashlib

    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

    config_parent = tmp_path / "config"
    config_parent.mkdir(mode=0o700)
    selector = config_parent / "config.toml"
    absent_parent = tmp_path / "absent-sql"
    absent_parent.mkdir(mode=0o700)
    scheduler = absent_parent / "scheduler.db"
    data = {
        "general": {"users_name": "Local"},
        "paths": {"data_dir": str(tmp_path / "data")},
        "database": {"scheduled_tasks_db_path": str(scheduler)},
        "ui": {"theme": "old"},
    }
    selector.write_text(toml.dumps(data))
    selector.chmod(0o600)
    prompt = database_path(data, "prompts_db_path")
    prompt.parent.mkdir(parents=True, mode=0o700)
    store = PromptsDatabase(prompt, "absent-scope-test")
    try:
        store.add_prompt("retained", "fixture", "Retained prompt", user_prompt="keep")
    finally:
        store.close()
    profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    configured = {**data, DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, profile)}
    owners = {owner.owner_id: owner for owner in install_adapters()}
    with _preview_reads():
        entries = tuple(
            row
            for owner in ("config", "db.prompts.primary", "db.scheduled_tasks")
            for row in owners[owner].discover(configured)
        )
    target = classify_entries(entries)
    assert target.issues == ("missing_required",)
    missing = next(row for row in entries if row.owner == "db.scheduled_tasks")
    assert missing.status == "missing_required" and missing.path == scheduler
    plan = RestorePlan(
        "a" * 64,
        "replace",
        (("profile:source:config", selector),),
        (),
        tuple((row.logical_id, row.path) for row in entries if row.owner != "config"),
        "b" * 64,
        target=target,
        requested_groups=("settings",),
        effective_groups=("settings",),
    )
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    authority = admission_authority(root)
    authority.register("settings.config", (selector,))
    authority.register("prompts", (prompt,))
    assert bootstrap._records(root)[1] == []
    return SimpleNamespace(
        plan=plan, data=data, selector=selector, prompt=prompt, scheduler=scheduler,
        root=root, authority=authority, owners=owners,
        names=(UNBOUND_NAMESPACE, "settings.config", "prompts"),
    )


def check(case, data=None, names=None):
    before = observed(case.selector), observed(case.prompt)
    registry = bootstrap._registry(case.root)
    records = bootstrap._records(case.root)
    try:
        with (
            case.authority.maintenance(names or case.names, 3) as session,
            session._discovery_reads(),
        ):
            check_preserved_group_paths(case.plan, data or case.data, case.selector)
    finally:
        assert (observed(case.selector), observed(case.prompt)) == before
        assert bootstrap._registry(case.root) == registry
        assert bootstrap._records(case.root) == records


def test_unchanged_absent_sqlite_needs_no_parent_registration(absent_scope):
    data = deepcopy(absent_scope.data)
    data["ui"]["theme"] = "new"
    check(absent_scope, data)
    assert all(
        Path(path).is_file()
        for row in bootstrap._registry(absent_scope.root).values()
        for path in row["roots"]
    )


def test_moved_absent_locator_refuses_even_when_both_paths_are_absent(absent_scope):
    data = deepcopy(absent_scope.data)
    data["database"]["scheduled_tasks_db_path"] = str(
        absent_scope.scheduler.with_name("different.db")
    )
    with pytest.raises(ValueError, match="preserved_group_path_changed:automation"):
        check(absent_scope, data)


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_absent_sqlite_primary_or_companion_arrival_refuses(absent_scope, suffix):
    arrived = Path(str(absent_scope.scheduler) + suffix)
    arrived.write_bytes(b"unreviewed arrival")
    arrived.chmod(0o600)
    before = observed(arrived)
    with pytest.raises(ValueError, match="preserved_group_"):
        check(absent_scope)
    assert observed(arrived) == before


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_arrival_after_real_owner_discovery_refuses(absent_scope, monkeypatch, suffix):
    owner = absent_scope.owners["db.scheduled_tasks"]
    discover = type(owner).discover
    arrived = Path(str(absent_scope.scheduler) + suffix)
    observations = []

    def discover_then_arrive(self, configured):
        result = discover(self, configured)
        if self.owner_id == owner.owner_id:
            assert result[0].status == "missing_required"
            arrived.write_bytes(b"arrived after the real absence observation")
            arrived.chmod(0o600)
            observations.append(observed(arrived))
        return result

    monkeypatch.setattr(type(owner), "discover", discover_then_arrive)
    with pytest.raises(ValueError, match="preserved_group_"):
        check(absent_scope)
    assert observations == [observed(arrived)]


@pytest.mark.parametrize("history", [False, True], ids=["current-root", "historical-file"])
def test_absent_locator_requires_existing_namespace_and_capture_only_reuses_it(
    absent_scope, tmp_path, history
):
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names

    case = absent_scope
    if history:
        case.scheduler.write_bytes(b"former namespace object")
        case.scheduler.chmod(0o600)
        case.authority.register("foreign-owner", (case.scheduler,))
        relocated = tmp_path / "relocated-owner.db"
        relocated.write_bytes(b"current namespace object")
        relocated.chmod(0o600)
        case.authority.remap("foreign-owner", (relocated,), 3)
        case.scheduler.unlink()
    else:
        # A real foreign directory scope is the conflict under test. Ordinary
        # config/Prompt authority above remains separate exact-file scopes.
        case.authority.register("foreign-owner", (case.scheduler.parent,))
    with pytest.raises(ValueError, match="preserved_group_native_scope_required"):
        check(case)
    registry = bootstrap._registry(case.root)
    names = _capture_names(case.authority, case.plan.target, include_absent_sqlite=True)
    assert "foreign-owner" in names
    assert bootstrap._registry(case.root) == registry
    check(case, names=names)


def test_sqlite_read_outside_held_file_scopes_still_refuses(absent_scope, tmp_path):
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

    outside = tmp_path / "outside.db"
    store = PromptsDatabase(outside, "outside-held-scope")
    store.close()
    before = observed(outside)
    with (
        absent_scope.authority.maintenance(absent_scope.names, 3) as session,
        session._discovery_reads(),
        pytest.raises(bootstrap.RecoveryRequired, match="capture_source_outside_scope"),
    ):
        connect_private_sqlite("recovery.core.prompts", outside, read_only=True)
    assert observed(outside) == before


_FULL_TARGET = r'''
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.capture_service import _capture_names, preview_capture
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.preserved_groups import check_preserved_group_paths
from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan
source = Path(os.environ['TLDW_CONFIG_PATH'])
data = Path(os.environ['XDG_DATA_HOME']) / 'fixture'
data.mkdir(mode=0o700)
source.write_text('[general]\nusers_name="default_user"\n[paths]\ndata_dir=' + json.dumps(str(data)) + '\n')
source.chmod(0o600)
producer = """
from pathlib import Path
import os
from Tests.Backup_Recovery.test_capture_service import _populate_required_dependencies
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
_populate_required_dependencies(preview_capture((Path(os.environ['TLDW_CONFIG_PATH']),), options={'allow_partial': True}))
"""
result = subprocess.run([sys.executable, '-c', producer], capture_output=True, text=True, timeout=30)
assert result.returncode == 0, result.stderr[-4000:]
authority = admission_authority(bootstrap.default_bootstrap_root())
target = preview_capture((source,), options={})
assert target.issues == ('missing_required',), target.issues
assert {'db.evals', 'db.library_collections', 'db.scheduled_tasks', 'db.subscriptions'} <= {
    row.owner for row in target.items if row.status == 'missing_required'
}
plan = RestorePlan('a' * 64, 'replace', (('profile:source:config', source),), (),
    tuple((row.logical_id, row.path) for row in target.items if row.path is not None and row.path != source),
    'b' * 64, target=target, requested_groups=('settings',), effective_groups=('settings',))
names = _capture_names(authority, target, include_absent_sqlite=True)
registry = bootstrap._registry(bootstrap.default_bootstrap_root())
assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
declared_directories = {str(row.path) for row in target.items if row.status == 'included_directory'}
assert all(Path(path).is_file() or path in declared_directories
           for entry in registry.values() for path in entry['roots'])
assert str(data) not in {path for entry in registry.values() for path in entry['roots']}
def observed(path):
    info = path.stat()
    return path.read_bytes(), info.st_dev, info.st_ino
preserved = {row.path: observed(row.path) for row in target.items if row.status == 'included'}
final = tomllib.loads(source.read_text())
final['general']['default_theme'] = 'textual-light'
with authority.maintenance(names, 3) as session:
    with session._discovery_reads():
        check_preserved_group_paths(plan, final, source)
assert all(observed(path) == before for path, before in preserved.items())
assert bootstrap._registry(bootstrap.default_bootstrap_root()) == registry
assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
assert not network_guard.blocked_attempts()
print('retired and reopened')
'''


def test_full_discovered_target_preserves_preferences_without_binding_absent_stores(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, "absent-settings", "success", script=_FULL_TARGET, timeout=60)
