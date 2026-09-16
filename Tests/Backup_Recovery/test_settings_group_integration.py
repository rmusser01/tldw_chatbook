"""Settings path preservation runs at real review and staging boundaries."""

from copy import deepcopy
from threading import Event

import pytest
import toml

from Tests.Backup_Recovery.test_preserved_group_paths import (
    preserved_case as preserved_case,  # noqa: PLC0414 - pytest fixture reuse
)


@pytest.mark.parametrize("boundary", ["preview", "stage"])
@pytest.mark.parametrize("move_data", [False, True])
def test_settings_group_checks_unselected_database_locations(preserved_case, tmp_path, boundary, move_data):
    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE
    from tldw_chatbook.Backup_Recovery.destinations import check_config_destinations
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    template, data, selector, store, authority = preserved_case
    incoming = deepcopy(data)
    incoming["ui"]["theme"] = "new"
    if move_data:
        incoming["paths"]["data_dir"] += "-foreign"
    payload = toml.dumps(incoming).encode()

    def config_document(doc):
        doc["owners"][0]["owner_id"] = "config"
        doc["files"][0].update(owner_id="config", logical_id="profile:profile:config", relative_path="config.toml")
        doc["directories"][0]["synthetic"] = True
        doc["producer_inventory"] = [
            {"logical_id": "root", "owner_id": "config", "status": "included_directory", "dependencies": []},
            {"logical_id": "profile:profile:config", "owner_id": "config", "status": "included", "dependencies": []},
        ]
        doc["dependency_groups"][0]["members"] = ["profile:profile:config"]

    archive = sealed(tmp_path, data=payload, mutate=config_document)
    plan = plan_restore(archive, mode="replace", target=template.target,
                        data_groups=("settings",), destinations={"root": selector.parent},
                        profile_names={"profile": "Local"})
    before = store.read_bytes(), store.stat().st_ino, selector.read_bytes(), selector.stat().st_ino

    def check():
        if boundary == "preview":
            check_config_destinations(archive, plan)
        else:
            with authority.maintenance(("local", UNBOUND_NAMESPACE), 3) as session:
                stage_restore(archive, plan, tmp_path / "candidate", Event(), session=session)

    if move_data:
        with pytest.raises(ValueError, match="preserved_group_path_changed:prompts"):
            check()
    else:
        check()
    assert (store.read_bytes(), store.stat().st_ino, selector.read_bytes(), selector.stat().st_ino) == before


@pytest.mark.parametrize("change", ["preferences", "data_root", "username", "database_override"])
def test_settings_snapshot_preflight_checks_exact_historical_locations(
    preserved_case, tmp_path, change
):
    """Exercise preflight on sealed bytes; snapshot authentication is separate.

    The typed source identifies this boundary's snapshot branch. No journal,
    staging, publication, or native rollback authority is simulated here.
    """
    from dataclasses import replace

    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.destinations import check_config_destinations
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        LocalSnapshotSource,
        plan_restore,
    )

    template, data, selector, store, _ = preserved_case
    incoming = deepcopy(data)
    incoming["ui"]["theme"] = "historical"
    if change == "data_root":
        incoming["paths"]["data_dir"] += "-historical"
    elif change == "username":
        incoming["general"]["users_name"] = "Historical"
    elif change == "database_override":
        incoming["database"] = {"prompts_db_path": str(store.with_name("historical.db"))}
    payload = b"# Preserve the exact historical representation.\n" + toml.dumps(incoming).encode()

    def config_document(doc):
        doc["owners"][0]["owner_id"] = "config"
        doc["files"][0].update(owner_id="config", logical_id="profile:profile:config", relative_path="config.toml")
        doc["directories"][0]["synthetic"] = True
        doc["producer_inventory"] = [
            {"logical_id": "root", "owner_id": "config", "status": "included_directory", "dependencies": []},
            {"logical_id": "profile:profile:config", "owner_id": "config", "status": "included", "dependencies": []},
        ]
        doc["dependency_groups"][0]["members"] = ["profile:profile:config"]

    archive = sealed(tmp_path, data=payload, mutate=config_document)
    plan = plan_restore(
        archive,
        mode="replace",
        target=template.target,
        data_groups=("settings",),
        destinations={"root": selector.parent},
        profile_names={"profile": "Local"},
    )
    plan = replace(
        plan,
        local_snapshot=LocalSnapshotSource(tmp_path / "history", "prior-operation", "c" * 64),
        # Ordinary import naming must neither hide a historical username change
        # nor alter a preference-only snapshot's currently valid data locator.
        profile_names=(("profile", "OrdinaryRestoreOnly" if change == "preferences" else "Local"),),
    )
    before = (
        store.read_bytes(), store.stat().st_ino,
        selector.read_bytes(), selector.stat().st_ino,
        archive.path.read_bytes(),
    )
    if change == "preferences":
        check_config_destinations(archive, plan)
    else:
        with pytest.raises(ValueError, match="preserved_group_path_changed:prompts"):
            check_config_destinations(archive, plan)
    assert (
        store.read_bytes(), store.stat().st_ino,
        selector.read_bytes(), selector.stat().st_ino,
        archive.path.read_bytes(),
    ) == before
