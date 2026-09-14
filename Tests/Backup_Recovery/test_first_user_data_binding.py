"""First binding retains only the proven installed profile data container."""

import subprocess
import sys
from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_default_service_container import (
    default_container as default_container,  # noqa: PLC0414 - shared native fixture
)
from tldw_chatbook.Backup_Recovery import bootstrap, replacement
from tldw_chatbook.Backup_Recovery.archive_reader import verify_sealed
from tldw_chatbook.Backup_Recovery.models import StorageItem
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Utils.platform_files import os


@pytest.fixture
def data_binding(default_container, tmp_path, monkeypatch):
    service, archive, config_parent, target, _ = default_container
    base = tmp_path / "data"
    base.mkdir(mode=0o700)
    data = base / "selected"
    data.mkdir(mode=0o700)
    selector = config_parent / "config.toml"
    selector.write_text(
        f'[general]\nusers_name="selected"\n[paths]\ndata_dir="{base.as_posix()}"\n'
    )
    selector.chmod(0o600)
    source = data / "mcp_server_targets.json"
    source.write_bytes(b'{"version":1,"targets":[]}')
    source.chmod(0o600)
    target = replace(
        target,
        items=(
            *target.items,
            StorageItem("config", "profile:test:config", selector, "included", ()),
            StorageItem(
                "mcp.targets",
                "profile:test:mcp.targets",
                source,
                "included",
                ("profile:test:config",),
            ),
        ),
    )
    plan = plan_restore(
        archive, mode="replace", destinations={"root": config_parent}, target=target
    )
    # Discovery is independently covered by native lifecycle acceptance. This
    # fixture supplies a complete finite local inventory to actual native binding.
    monkeypatch.setattr(
        replacement, "_first_binding_inventory", lambda plan, path: target
    )
    root = bootstrap.default_bootstrap_root()
    from tldw_chatbook.Backup_Recovery.service_storage import work_root

    protected = (service.control_root, work_root(service.control_root) / "candidate")
    document = verify_sealed(archive)
    return selector, data, plan, root, protected, document


def _bind(case):
    selector, _, plan, root, protected, document = case
    replacement._ensure_first_bindings(
        plan, (selector,), root, Event(), protected=protected, document=document
    )


def test_first_binding_allows_runtime_sibling_without_config_control_authority(
    data_binding,
):
    from tldw_chatbook.Backup_Recovery.storage_admission import _scope

    selector, data, _, root, _, _ = data_binding
    config_before = selector.read_bytes()
    marker = root / "unbound-owner"
    control_before = marker.read_bytes()
    parent_before = os.stat(selector.parent)
    _bind(data_binding)
    names = _scope(root, selector, data / "mcp_server_targets.json.tmp")
    record = bootstrap._records(root)[1][0]
    assert names == tuple(record["namespaces"])
    assert str(data) in record["roots"]
    assert str(data.parent) not in record["roots"]
    assert str(selector.parent) not in record["roots"]
    for forbidden in (root / "forbidden.tmp", data.parent / "another-profile.tmp"):
        with pytest.raises(
            bootstrap.RecoveryRequired, match="storage_scope_not_enrolled"
        ):
            _scope(root, selector, forbidden)
    assert selector.read_bytes() == config_before
    assert marker.read_bytes() == control_before
    parent_after = os.stat(selector.parent)
    assert (parent_before.st_dev, parent_before.st_ino, parent_before.st_mode) == (
        parent_after.st_dev,
        parent_after.st_ino,
        parent_after.st_mode,
    )


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "public",
        "missing",
        "alias",
        "foreign",
        "historical",
        "historical_inode",
        "proposed",
        "empty",
    ],
)
def test_first_data_container_refuses_unverified_ownership(
    data_binding, tmp_path, case
):
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    selector, data, plan, root, protected, _ = data_binding
    authority = admission_authority(root)
    names = _capture_names(authority, plan.target)
    registry = bootstrap._registry(root)
    if case == "unknown":
        (data / "unknown.txt").write_bytes(b"external")
    elif case == "public":
        if sys.platform == "win32":
            subprocess.run(  # nosec B603 B607
                ["icacls", str(data), "/grant", "*S-1-1-0:(R)"],
                check=True,
                capture_output=True,
            )
        else:
            os.chmod(data, 0o755)
        assert os.stat(data).st_mode & 0o044
    elif case == "missing":
        (data / "mcp_server_targets.json").unlink()
        data.rmdir()
    elif case == "alias":
        moved = data.with_name("moved")
        data.rename(moved)
        data.symlink_to(moved, target_is_directory=True)
    elif case == "foreign":
        registry["foreign"] = {
            "roots": [str(data)],
            "historical": [],
            "pending": None,
            "proposed": [],
        }
    elif case == "historical":
        registry["foreign"] = {
            "roots": [str(tmp_path / "elsewhere")],
            "historical": ["path:" + str(data / "retired.json")],
            "pending": None,
            "proposed": [],
        }
    elif case == "historical_inode":
        info = os.stat(data / "mcp_server_targets.json")
        registry["foreign"] = {
            "roots": [str(tmp_path / "elsewhere")],
            "historical": [f"inode:{info.st_dev}:{info.st_ino}"],
            "pending": None,
            "proposed": [],
        }
    elif case == "proposed":
        registry["foreign"] = {
            "roots": [str(tmp_path / "elsewhere")],
            "historical": [],
            "pending": "operation",
            "proposed": [str(data / "new.json")],
        }
    elif case == "empty":
        (data / "mcp_server_targets.json").unlink()
        plan = replace(
            plan,
            target=replace(
                plan.target,
                items=tuple(
                    item for item in plan.target.items if item.owner != "mcp.targets"
                ),
            ),
        )
    with pytest.raises((OSError, ValueError)):
        replacement._first_user_data_container(
            selector, plan.target, names, registry, (), (selector,), (root, *protected)
        )
    assert not bootstrap._records(root)[1]


def test_data_container_recheck_refuses_new_child_before_binding(
    data_binding, monkeypatch
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery.admission import Admission

    _, data, _, root, _, _ = data_binding
    original = Admission.maintenance

    @contextmanager
    def arriving_child(self, *args, **kwargs):
        with original(self, *args, **kwargs) as session:
            (data / "late-external.txt").write_bytes(b"external arrival")
            yield session

    monkeypatch.setattr(Admission, "maintenance", arriving_child)
    with pytest.raises(ValueError):
        _bind(data_binding)
    assert not bootstrap._records(root)[1]
    assert (data / "late-external.txt").read_bytes() == b"external arrival"


def test_first_binding_does_not_expand_an_existing_exact_file_binding(data_binding):
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    selector, data, plan, root, _, _ = data_binding
    names = tuple(
        name
        for name in _capture_names(admission_authority(root), plan.target)
        if name != "bootstrap.unbound"
    )
    bind_profile(root, selector, names, root / "admission")
    before = bootstrap._records(root)
    registry = bootstrap._registry(root)
    _bind(data_binding)
    assert bootstrap._records(root) == before
    assert bootstrap._registry(root) == registry
    assert str(data) not in before[1][0]["roots"]


@pytest.mark.parametrize("change", ["selector", "directory", "cancel"])
def test_first_data_binding_rechecks_identity_selection_and_cancellation(
    data_binding, monkeypatch, change
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery.admission import Admission

    selector, data, plan, root, protected, document = data_binding
    original = Admission.maintenance
    cancel = Event()

    @contextmanager
    def changed(self, *args, **kwargs):
        with original(self, *args, **kwargs) as session:
            if change == "selector":
                selector.write_bytes(selector.read_bytes() + b"\n# external write\n")
            elif change == "directory":
                data.rename(data.with_name("retained"))
                data.mkdir(mode=0o700)
            else:
                cancel.set()
            yield session

    monkeypatch.setattr(Admission, "maintenance", changed)
    with pytest.raises((ValueError, InterruptedError)):
        replacement._ensure_first_bindings(
            plan, (selector,), root, cancel, protected=protected, document=document
        )
    assert not bootstrap._records(root)[1]


def test_config_only_profile_does_not_require_a_new_default_data_container(
    data_binding, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.restore_plan import _fingerprint, _paths

    selector, data, plan, root, protected, document = data_binding
    (data / "mcp_server_targets.json").unlink()
    data.rmdir()
    target = replace(
        plan.target,
        items=tuple(item for item in plan.target.items if item.owner != "mcp.targets"),
    )
    plan = replace(plan, target=target)
    plan = replace(plan, target_fingerprint=_fingerprint(_paths(plan), target))
    monkeypatch.setattr(
        replacement, "_first_binding_inventory", lambda plan, path: target
    )
    replacement._ensure_first_bindings(
        plan, (selector,), root, Event(), protected=protected, document=document
    )
    assert len(bootstrap._records(root)[1]) == 1
    assert str(data) not in bootstrap._records(root)[1][0]["roots"]
    assert not data.exists()


def test_existing_inventory_directory_needs_no_duplicate_data_namespace(
    data_binding, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.restore_plan import _fingerprint, _paths

    selector, data, plan, root, protected, document = data_binding
    target = replace(
        plan.target,
        items=(
            *plan.target.items,
            StorageItem(
                "mcp.targets",
                "profile:test:mcp.container",
                data,
                "included_directory",
                (),
            ),
        ),
    )
    plan = replace(plan, target=target)
    plan = replace(plan, target_fingerprint=_fingerprint(_paths(plan), target))
    authority = admission_authority(root)
    authority.register("existing-data", (data,))
    _capture_names(authority, target)
    before = bootstrap._registry(root)
    monkeypatch.setattr(
        replacement, "_first_binding_inventory", lambda plan, path: target
    )
    replacement._ensure_first_bindings(
        plan, (selector,), root, Event(), protected=protected, document=document
    )
    assert bootstrap._registry(root) == before
    assert str(data) in bootstrap._records(root)[1][0]["roots"]


@pytest.mark.parametrize("case", ["empty", "new_child", "unknown_owner"])
def test_installed_empty_actor_scaffold_is_exact_observation_only(data_binding, case):
    import tomllib

    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )

    selector, data, plan, root, protected, _ = data_binding
    scaffold = data / "actor_pack_imports"
    scaffold.mkdir(mode=0o700)
    config = tomllib.loads(selector.read_text())
    config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(selector, "test")
    adapter = next(
        owner
        for owner in recovery_adapters()
        if owner.owner_id == "actor_packs.import_staging"
    )
    rows = adapter.discover(config)
    assert (
        len(rows) == 1
        and rows[0].status == "unused"
        and rows[0].metadata.kind == "directory"
    )
    if case == "unknown_owner":
        rows = (replace(rows[0], owner="unknown.scaffold"),)
    target = replace(plan.target, items=(*plan.target.items, *rows))
    authority = admission_authority(root)
    names = _capture_names(authority, target)
    if case == "new_child":
        (scaffold / "pending.zip").write_bytes(b"new staged payload")
    args = (
        selector,
        target,
        names,
        bootstrap._registry(root),
        (),
        (selector,),
        (root, *protected),
    )
    if case == "empty":
        _, observed = replacement._first_user_data_container(*args)
        assert scaffold in {row[0] for row in observed}
        assert not tuple(scaffold.iterdir())
    else:
        with pytest.raises(ValueError):
            replacement._first_user_data_container(*args)


@pytest.mark.parametrize(
    "case", ["shared", "different_selection", "changed_tree", "existing_shared"]
)
def test_same_call_shared_data_requires_each_local_selection_and_footprint(
    data_binding, tmp_path, monkeypatch, case
):
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.storage_admission import _scope

    first, data, plan, root, protected, document = data_binding
    parent = tmp_path / "second-config"
    parent.mkdir(mode=0o700)
    second = parent / "config.toml"
    second.write_bytes(first.read_bytes())
    second.chmod(0o600)
    if case == "different_selection":
        second.write_text(
            second.read_text().replace(
                'users_name="selected"', 'users_name="different"'
            )
        )
    other_inventory = replace(
        plan.target,
        items=tuple(
            replace(item, path=second, logical_id="profile:second:config")
            if item.owner == "config"
            else item
            for item in plan.target.items
            if item.owner != "ui.state"
        ),
    )
    monkeypatch.setattr(
        replacement,
        "_first_binding_inventory",
        lambda plan, selector: plan.target if selector == first else other_inventory,
    )
    if case == "changed_tree":
        original = replacement._first_user_data_container

        def changed(selector, *args, **kwargs):
            if selector == second:
                (data / "late-foreign.txt").write_bytes(b"new unowned child")
            return original(selector, *args, **kwargs)

        monkeypatch.setattr(replacement, "_first_user_data_container", changed)
    if case == "existing_shared":
        authority = admission_authority(root)
        names = tuple(
            name
            for name in _capture_names(authority, other_inventory)
            if name != "bootstrap.unbound"
        )
        bind_profile(root, second, names, root / "admission")
    if case == "changed_tree":
        with pytest.raises(ValueError):
            replacement._ensure_first_bindings(
                plan,
                (first, second),
                root,
                Event(),
                protected=protected,
                document=document,
            )
        assert not bootstrap._records(root)[1]
        return
    replacement._ensure_first_bindings(
        plan, (first, second), root, Event(), protected=protected, document=document
    )
    records = {row["selector"]: row for row in bootstrap._records(root)[1]}
    assert len(records) == 2
    if case == "existing_shared":
        assert all(str(data) not in row["roots"] for row in records.values())
        for selector in (first, second):
            with pytest.raises(
                bootstrap.RecoveryRequired, match="storage_scope_not_enrolled"
            ):
                _scope(root, selector, data / "mcp_server_targets.json.tmp")
    elif case == "different_selection":
        assert str(data) in records[str(first)]["roots"]
        assert str(data) not in records[str(second)]["roots"]
    else:
        first_names = _scope(root, first, data / "mcp_server_targets.json.tmp")
        second_names = _scope(root, second, data / "mcp_server_targets.json.tmp")
        registry = bootstrap._registry(root)
        shared = {
            name for name in first_names if registry[name]["roots"] == [str(data)]
        }
        assert len(shared) == 1 and shared <= set(second_names)

        from threading import Thread

        from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

        authority = admission_authority(root)
        refusals = []

        def competing_maintenance():
            try:
                with authority.maintenance(second_names, 0.1):
                    refusals.append("incorrectly admitted")
            except AdmissionTimeout:
                refusals.append("native shared lease retained")

        with authority.normal(first_names):
            worker = Thread(target=competing_maintenance)
            worker.start()
            worker.join(5)
            assert not worker.is_alive()
            assert refusals == ["native shared lease retained"]
        with authority.maintenance(second_names, 1):
            pass
