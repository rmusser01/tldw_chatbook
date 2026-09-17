"""Live recovery records are preserved without hiding payload changes."""

from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery import bootstrap, inventory
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
    register_pending,
)
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore, recheck_targets


@pytest.fixture
def reviewed_control(tmp_path, monkeypatch):
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    authority = admission_authority(root)
    config = tmp_path / "current.toml"
    config.write_text('[general]\nusers_name="original"\n')
    config.chmod(0o600)
    authority.register("current", (config,))
    item = inventory._fixed_control_exclusion()[0]
    assert item.status == "intentionally_excluded"
    return root, config, item


def reviewed_plan(tmp_path, item, config):
    return plan_restore(
        sealed(tmp_path),
        mode="isolated",
        destinations={"root": tmp_path / "new"},
        target=Inventory(
            (item, StorageItem("config", "current", config, "included", ())),
            True,
            "current",
            (),
        ),
    )


def test_local_binding_and_pending_records_do_not_change_preserved_payload_plan(
    tmp_path, reviewed_control
):
    root, config, item = reviewed_control
    plan = reviewed_plan(tmp_path, item, config)
    bind_profile(root, config, ("current",), root / "admission")
    recheck_targets(plan)
    register_pending(root, "replacement", ("current",), tmp_path / "control", (config,))
    recheck_targets(plan)
    assert dict(plan.preserve)[item.logical_id] == root
    assert config.read_text() == '[general]\nusers_name="original"\n'


@pytest.mark.parametrize("change", ["payload", "unknown_record", "identity", "alias"])
def test_live_control_exception_keeps_payload_and_authority_checks(
    tmp_path, reviewed_control, change
):
    root, config, item = reviewed_control
    plan = reviewed_plan(tmp_path, item, config)
    if change == "payload":
        config.write_text('[general]\nusers_name="changed"\n')
    elif change == "unknown_record":
        (root / "unknown.json").write_text("{}")
    else:
        saved = root.with_name("saved-bootstrap")
        root.rename(saved)
        if change == "alias":
            root.symlink_to(saved, target_is_directory=True)
        else:
            import shutil

            shutil.copytree(saved, root)
    with pytest.raises(ValueError, match="target_changed"):
        recheck_targets(plan)


@pytest.mark.parametrize(
    "field,value", [("owner", "ui.state"), ("logical_id", "unrecognized")]
)
def test_ordinary_preserved_directory_does_not_get_control_semantics(
    tmp_path, reviewed_control, field, value
):
    root, config, item = reviewed_control
    plan = reviewed_plan(tmp_path, replace(item, **{field: value}), config)
    bind_profile(root, config, ("current",), root / "admission")
    with pytest.raises(ValueError, match="target_changed"):
        recheck_targets(plan)


@pytest.mark.parametrize(
    "change", ["activity", "marker", "sibling", "control", "control-work"]
)
def test_service_control_activity_preserves_actual_directory_identity(
    tmp_path, monkeypatch, change
):
    from tldw_chatbook.Backup_Recovery import service_storage

    config = tmp_path / "selected.toml"
    config.write_text("[general]\n")
    monkeypatch.setattr(service_storage, "default_config_path", lambda: config)
    control = service_storage.default_control_root()
    work = service_storage.ensure_storage(control)
    item = inventory._service_control_exclusion()[0]
    plan = reviewed_plan(tmp_path, item, config)
    if change == "activity":
        (control / "operation-test").mkdir(mode=0o700)
        (work / "inspection-test").mkdir(mode=0o700)
        recheck_targets(plan)
        assert dict(plan.preserve)[item.logical_id] == control.parent
        return
    if change == "marker":
        (control / "service.json").write_text("{}")
    elif change == "sibling":
        (control.parent / "unknown").mkdir(mode=0o700)
    else:
        import shutil

        child = control.parent / change
        saved = tmp_path / "saved-child"
        child.rename(saved)
        shutil.copytree(saved, child)
        service_storage.verify_default_storage()
    with pytest.raises(ValueError, match="target_changed"):
        recheck_targets(plan)
