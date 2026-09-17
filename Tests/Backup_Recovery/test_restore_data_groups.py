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
