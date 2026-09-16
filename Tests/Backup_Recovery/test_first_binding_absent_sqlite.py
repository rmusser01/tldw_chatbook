"""First binding distinguishes absent installed databases from uncertain owners."""

import hashlib
from contextlib import contextmanager
from dataclasses import replace
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery.inventory import classify_entries
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
from tldw_chatbook.Backup_Recovery.profile_paths import database_path, user_data_dir
from tldw_chatbook.Backup_Recovery.replacement import _first_user_data_root
from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads


@pytest.fixture
def absent_database(tmp_path):
    import toml

    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

    selector = tmp_path / "config.toml"
    data = {"paths": {"data_dir": str(tmp_path / "data")}}
    selector.write_text(toml.dumps(data))
    selector.chmod(0o600)
    prompt = database_path(data, "prompts_db_path")
    prompt.parent.mkdir(mode=0o700, parents=True)
    store = PromptsDatabase(prompt, "first-binding-absence")
    store.close()
    profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    configured = {**data, DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, profile)}
    owners = {owner.owner_id: owner for owner in install_adapters()}
    with _preview_reads():
        rows = tuple(
            row
            for name in ("config", "db.prompts.primary", "db.scheduled_tasks")
            for row in owners[name].discover(configured)
        )
    inventory = classify_entries(rows)
    assert inventory.issues == ("missing_required",)
    absent = next(row for row in rows if row.owner == "db.scheduled_tasks")
    assert absent.status == "missing_required" and not absent.path.exists()
    return selector, data, inventory, absent


def test_first_binding_keeps_installed_data_root_with_absent_sqlite(absent_database):
    selector, data, inventory, _ = absent_database
    assert _first_user_data_root(selector, inventory) == user_data_dir(data)
    assert not inventory.complete


@pytest.mark.parametrize("arrival", ["primary", "-wal", "-shm", "-journal", "symlink"])
def test_first_binding_refuses_residual_or_new_sqlite_state(absent_database, arrival):
    selector, _, inventory, absent = absent_database
    absent.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if arrival == "symlink":
        absent.path.symlink_to(absent.path.with_name("missing-target"))
    else:
        path = (
            absent.path
            if arrival == "primary"
            else absent.path.with_name(absent.path.name + arrival)
        )
        path.write_bytes(b"unreviewed database state")
        path.chmod(0o600)
    with pytest.raises(ValueError, match="replacement_current_scope_unavailable"):
        _first_user_data_root(selector, inventory)


@pytest.mark.parametrize(
    "issue",
    [
        "unsupported",
        "unavailable",
        "dependency_unavailable",
        "shared_identity_mismatch",
    ],
)
def test_first_binding_refuses_other_incomplete_scope(absent_database, issue):
    selector, _, inventory, _ = absent_database
    inventory = replace(inventory, issues=tuple(sorted({*inventory.issues, issue})))
    with pytest.raises(ValueError, match="replacement_current_scope_unavailable"):
        _first_user_data_root(selector, inventory)


def test_first_binding_refuses_non_sqlite_missing_owner(absent_database):
    selector, _, inventory, absent = absent_database
    inventory = classify_entries(
        tuple(
            replace(row, owner="ui.state") if row == absent else row
            for row in inventory.items
        )
    )
    with pytest.raises(ValueError, match="replacement_current_scope_unavailable"):
        _first_user_data_root(selector, inventory)


@pytest.fixture
def external_absent_binding(tmp_path, monkeypatch, request):
    """Installed owner rows with native binding and no broad container snapshot."""
    import tomllib

    import toml

    from tldw_chatbook.Backup_Recovery import bootstrap, inventory, restore_plan
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

    config_parent = tmp_path / "config"
    config_parent.mkdir(mode=0o700)
    selector = config_parent / "config.toml"
    shared = tmp_path / "shared-databases"
    shared.mkdir(mode=0o700)
    absent_parent = shared
    if getattr(request, "param", None) == "uncovered":
        absent_parent = tmp_path / "uncovered-databases"
        absent_parent.mkdir(mode=0o700)
    absent = absent_parent / "scheduler.db"
    prompt = shared / "prompts.db"
    selector.write_text(
        toml.dumps(
            {
                "paths": {"data_dir": str(tmp_path / "unused-data")},
                "database": {
                    "prompts_db_path": str(prompt),
                    "scheduled_tasks_db_path": str(absent),
                },
            }
        )
    )
    selector.chmod(0o600)
    store = PromptsDatabase(prompt, "first-binding-companion-race")
    store.close()
    owners = {owner.owner_id: owner for owner in install_adapters()}
    profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]

    def discover_installed_rows(selectors):
        assert selectors == (selector,)
        configured = {
            **tomllib.loads(selector.read_text()),
            DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, profile),
        }
        return classify_entries(
            tuple(
                row
                for name in ("config", "db.prompts.primary", "db.scheduled_tasks")
                for row in owners[name].discover(configured)
            )
        )

    # Bound the owner catalog only. The actual preflight, target rechecks,
    # container proofs, maintenance session, and bind_profile remain installed.
    monkeypatch.setattr(inventory, "discover", discover_installed_rows)
    with _preview_reads():
        target = discover_installed_rows((selector,))
    assert target.issues == ("missing_required",)
    assert (
        next(row for row in target.items if row.owner == "db.scheduled_tasks").path
        == absent
    )
    selected = next(row for row in target.items if row.owner == "db.prompts.primary")
    plan = restore_plan.RestorePlan(
        archive_digest="native-first-binding-component",
        mode="replace",
        restore=((selected.logical_id, prompt),),
        retire=(),
        preserve=tuple(
            (row.logical_id, row.path)
            for row in target.items
            if row.logical_id != selected.logical_id and row.path is not None
        ),
        target_fingerprint="",
        target=target,
    )
    plan = replace(
        plan,
        target_fingerprint=restore_plan._fingerprint(restore_plan._paths(plan), target),
    )
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    authority = admission_authority(root)
    authority.register("shared-databases", (shared,))
    return selector, absent, plan, root


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_first_binding_rechecks_absent_companions_under_native_hold(
    external_absent_binding, monkeypatch, suffix
):
    from tldw_chatbook.Backup_Recovery import bootstrap, replacement
    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE

    selector, absent, plan, root = external_absent_binding
    companion = absent.with_name(absent.name + suffix)
    native_maintenance = Admission.maintenance
    entered = []

    @contextmanager
    def arrive_before_maintenance(self, names, *args, **kwargs):
        assert UNBOUND_NAMESPACE in names and "shared-databases" in names
        assert not companion.exists() and not absent.exists()
        companion.write_bytes(b"residual state after initial absence proof")
        companion.chmod(0o600)
        with native_maintenance(self, names, *args, **kwargs) as session:
            entered.append(True)
            yield session

    monkeypatch.setattr(Admission, "maintenance", arrive_before_maintenance)
    with pytest.raises(ValueError, match="^replacement_current_scope_unavailable$"):
        replacement._ensure_first_bindings(plan, (selector,), root, Event())
    assert entered == [True]
    assert not bootstrap._records(root)[1]
    assert companion.read_bytes() == b"residual state after initial absence proof"
    assert not absent.exists()


def test_first_binding_accepts_unchanged_absence_in_held_shared_root(
    external_absent_binding,
):
    from tldw_chatbook.Backup_Recovery import bootstrap, replacement

    selector, absent, plan, root = external_absent_binding
    replacement._ensure_first_bindings(plan, (selector,), root, Event())
    records = bootstrap._records(root)[1]
    assert len(records) == 1 and records[0]["selector"] == str(selector)
    assert str(absent.parent) in records[0]["roots"]
    assert not absent.exists()


@pytest.mark.parametrize("external_absent_binding", ["uncovered"], indirect=True)
def test_first_binding_refuses_absent_sqlite_outside_held_roots(
    external_absent_binding,
):
    from tldw_chatbook.Backup_Recovery import bootstrap, replacement

    selector, absent, plan, root = external_absent_binding
    with pytest.raises(ValueError, match="^replacement_current_scope_changed$"):
        replacement._ensure_first_bindings(plan, (selector,), root, Event())
    assert not bootstrap._records(root)[1]
    assert not absent.exists()
