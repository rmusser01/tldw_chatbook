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
