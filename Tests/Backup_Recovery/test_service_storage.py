"""Recognized private recovery storage stays outside portable data."""

import pytest


def test_custom_work_storage_survives_reopening_and_is_disjoint(tmp_path):
    from tldw_chatbook.Backup_Recovery.service_storage import ensure_storage

    control = tmp_path / "custom-control"
    work = ensure_storage(control)
    candidate = work / "pending-candidate"
    candidate.write_bytes(b"journal-bound pending work")
    assert ensure_storage(control) == work
    assert control not in work.parents and work not in control.parents
    assert candidate.read_bytes() == b"journal-bound pending work"


def test_nonprivate_existing_work_storage_is_refused_without_chmod(tmp_path):
    from tldw_chatbook.Backup_Recovery.service_storage import ensure_storage, work_root

    control = tmp_path / "custom-control"
    work = work_root(control)
    work.mkdir(mode=0o755)
    before = work.stat().st_mode
    with pytest.raises(ValueError, match="recovery_control_not_private"):
        ensure_storage(control)
    assert work.stat().st_mode == before


@pytest.mark.parametrize(
    "damage", ["missing_marker", "invalid_marker", "unknown_child"]
)
def test_default_service_storage_damage_blocks_inventory(tmp_path, monkeypatch, damage):
    from tldw_chatbook.Backup_Recovery import inventory, service_storage

    config = tmp_path / "app-config" / "config.toml"
    monkeypatch.setattr(service_storage, "default_config_path", lambda: config)
    control = service_storage.default_control_root()
    service_storage.ensure_storage(control)
    assert inventory._service_control_exclusion()[0].status == "intentionally_excluded"
    marker = control / "service.json"
    if damage == "missing_marker":
        marker.unlink()
    elif damage == "invalid_marker":
        marker.write_text('{"version":1,"kind":"unrecognized"}')
    else:
        (control.parent / "unclassified").write_bytes(b"keep this visible")
    assert inventory._service_control_exclusion()[0].status == "unavailable"
    if damage != "unknown_child":
        with pytest.raises(ValueError, match="recovery_control_unverified"):
            service_storage.ensure_storage(control)


@pytest.mark.parametrize(
    "case",
    ["valid", "control", "outside", "wrong_owner", "wrong_id", "unavailable", "marker"],
)
def test_only_recognized_default_service_work_is_internal_to_preserved_control(
    tmp_path, monkeypatch, case
):
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery import inventory, service_storage

    config = tmp_path / "app-config" / "config.toml"
    monkeypatch.setattr(service_storage, "default_config_path", lambda: config)
    control = service_storage.default_control_root()
    work = service_storage.ensure_storage(control)
    item = inventory._service_control_exclusion()[0]
    candidate = work / "inspection" / "candidate"
    if case == "control":
        candidate = control / "candidate"
    elif case == "outside":
        candidate = tmp_path / "candidate"
    elif case == "wrong_owner":
        item = replace(item, owner="config")
    elif case == "wrong_id":
        item = replace(item, logical_id="different")
    elif case == "unavailable":
        item = replace(item, status="unavailable")
    elif case == "marker":
        (control / "service.json").write_text("{}")
    assert service_storage.is_private_service_work(candidate, item) is (case == "valid")


@pytest.mark.parametrize("case", ["valid", "alias", "unsafe_parent", "payload"])
def test_staging_preserves_control_and_checks_actual_work_ancestry(
    tmp_path, monkeypatch, case
):
    import json
    from dataclasses import replace
    from pathlib import Path
    from threading import Event

    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery import inventory, service_storage
    from tldw_chatbook.Backup_Recovery.models import Inventory
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    config = tmp_path / "app-config" / "config.toml"
    monkeypatch.setattr(service_storage, "default_config_path", lambda: config)
    control = service_storage.default_control_root()
    work = service_storage.ensure_storage(control)
    item = inventory._service_control_exclusion()[0]
    parent = work / "inspection"
    if case == "alias":
        actual = tmp_path / "actual"
        actual.mkdir(mode=0o700)
        parent.symlink_to(actual, target_is_directory=True)
    else:
        parent.mkdir(mode=0o700)
        if case == "unsafe_parent":
            parent.chmod(0o777)
    if case == "payload":
        item = replace(
            item, logical_id="preserved", owner="ui.state", status="included_directory"
        )
    archive = sealed(tmp_path)
    destination = tmp_path / "restored"
    plan = plan_restore(
        archive,
        mode="isolated",
        destinations={"root": destination},
        target=Inventory((item,), True, "target", ()),
    )
    marker = (control / "service.json").read_bytes()
    if case == "valid":
        candidate = stage_restore(archive, plan, parent / "candidate", Event())
        descriptor = json.loads((candidate / "candidate.json").read_bytes())
        record = next(r for r in descriptor["artifacts"] if r["logical_id"] == "file")
        assert Path(record["candidate"]).read_bytes() == b"durable"
    else:
        error = "staging_target_alias" if case == "payload" else "destination_alias"
        with pytest.raises(ValueError, match=error):
            stage_restore(archive, plan, parent / "candidate", Event())
        assert not (parent / "candidate").exists()
    assert (control / "service.json").read_bytes() == marker
    assert dict(plan.preserve)[item.logical_id] == control.parent
    assert not destination.exists()
