"""Settings-only review preserves custom Retrieval storage without publishing it."""

import hashlib
import json
import zipfile
from copy import deepcopy

import pytest
import toml

from Tests.Backup_Recovery.test_restore_data_groups import _document
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext


def test_retrieval_remains_required_when_its_source_group_changes():
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Backup_Recovery.restore_groups import resolve_archive_groups

    config = StorageItem("config", "local:config", None, "included", ())
    media = StorageItem(
        "db.media.primary", "local:media", None, "included", (config.logical_id,)
    )
    projection = StorageItem(
        "rag.projections",
        "local:projection",
        None,
        "included",
        (config.logical_id, media.logical_id),
    )
    scope = resolve_archive_groups(
        _document(("config", "db.media.primary", "rag.projections")),
        ("library",),
        target=Inventory((config, media, projection), True, "observed", ()),
    )
    assert scope.effective_groups == ("library", "retrieval")
    assert scope.required_groups == ("retrieval",)


@pytest.mark.parametrize("retire", [False, True])
def test_excluded_projection_gets_no_unselected_retirement_authority(tmp_path, retire):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Backup_Recovery.restore_groups import (
        check_preserved_group_effects,
        resolve_archive_groups,
    )

    item = StorageItem(
        "rag.projections",
        "local:projection",
        tmp_path / "projection",
        "intentionally_excluded",
        (),
    )
    scope = resolve_archive_groups(_document(("config",)), ("settings",))
    plan = SimpleNamespace(
        target=Inventory((item,), True, "observed", ()),
        restore=(),
        retire=((item.logical_id, item.path),) if retire else (),
    )
    if retire:
        with pytest.raises(ValueError, match="unreviewed_group_effect:retrieval"):
            check_preserved_group_effects(plan, scope)
    else:
        check_preserved_group_effects(plan, scope)


def test_omitted_live_projection_cannot_escape_selected_group_boundary(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.rag_inventory import _Projections
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    live = tmp_path / "live"
    live.mkdir(mode=0o700)
    config, research = live / "config.toml", live / "research.db"
    for path in (config, research):
        path.write_bytes(b"opaque planner fixture")
        path.chmod(0o600)
    vectors = live / "chromadb"
    vectors.mkdir(mode=0o700)
    (vectors / "chroma.sqlite3").write_bytes(b"opaque planning fixture")
    configured = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(config, "profile")}
    omitted = tuple(
        replace(
            item,
            status="intentionally_excluded",
            dependencies=(*item.dependencies, "profile:profile:research.local"),
        )
        for item in _Projections("rag.projections")._tree(configured, vectors)
    )
    target = Inventory(
        (
            StorageItem("config", "profile:profile:config", config, "included", ()),
            StorageItem(
                "research.local",
                "profile:profile:research.local",
                research,
                "included",
                (),
            ),
            *omitted,
        ),
        True,
        "observed",
        (),
    )
    document = _document(("config", "research.local")).model_dump(mode="json")
    owners = {owner.owner_id: owner for owner in install_adapters()}
    for row in document["owners"]:
        row["schema_version"] = max(owners[row["owner_id"]].schema_policy().versions)
    for row in document["files"]:
        row["relative_path"] = (
            "config.toml" if row["owner_id"] == "config" else "research.db"
        )
    archive = sealed(
        tmp_path,
        data=b"hello",
        mutate=lambda value: (value.clear(), value.update(document)),
    )
    with pytest.raises(
        ValueError,
        match="unreviewed_group_effect:retrieval|required_target_group_unavailable:retrieval",
    ):
        plan_restore(
            archive,
            mode="replace",
            target=target,
            destinations={"root:config": live, "root:research.local": live},
            profile_names={"profile": "Local"},
            data_groups=("settings", "research"),
        )


@pytest.mark.parametrize(
    "change", ["settings", "projection_location", "retrieval_selected"]
)
def test_settings_review_limits_projection_relocation_to_selected_payloads(
    tmp_path, monkeypatch, change
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads

    local = tmp_path / "local"
    local.mkdir(mode=0o700)
    vectors = local / "custom-vectors"
    vectors.mkdir(mode=0o700)
    projection = vectors / "chroma.sqlite3"
    # Review observes owner topology; it never opens or publishes engine data.
    projection.write_bytes(b"untouched opaque projection planning fixture")
    projection.chmod(0o600)
    selector = local / "config.toml"
    data = {
        "general": {"users_name": "Local", "default_theme": "textual-dark"},
        "paths": {"data_dir": str(local / "data")},
        "AppRAGSearchConfig": {
            "rag": {"vector_store": {"persist_directory": str(vectors)}}
        },
    }
    selector.write_text(toml.dumps(data))
    selector.chmod(0o600)
    profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    configured = {**data, DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, profile)}
    owners = {owner.owner_id: owner for owner in install_adapters()}
    with _preview_reads():
        target = classify_entries(
            tuple(
                item
                for name in ("config", "rag.projections")
                for item in owners[name].discover(configured)
            )
        )
    assert target.complete, target.issues
    before = {
        path: (path.read_bytes(), path.stat().st_dev, path.stat().st_ino)
        for path in (selector, projection)
    }
    incoming = deepcopy(data)
    incoming["general"]["default_theme"] = "textual-light"
    if change == "projection_location":
        incoming["AppRAGSearchConfig"]["rag"]["vector_store"]["persist_directory"] += (
            "-other"
        )
    contents = {
        "config": toml.dumps(incoming).encode(),
        "rag.projections": b"saved projection",
    }
    document = _document(tuple(contents)).model_dump(mode="json")
    for row in document["owners"]:
        row["schema_version"] = max(owners[row["owner_id"]].schema_policy().versions)
    for row in document["files"]:
        content = contents[row["owner_id"]]
        row.update(
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            relative_path="config.toml"
            if row["owner_id"] == "config"
            else "chroma.sqlite3",
        )
    archive = tmp_path / "incoming.zip"
    with zipfile.ZipFile(archive, "w") as container:
        container.writestr("manifest.json", json.dumps(document))
        for row in document["files"]:
            container.writestr(row["payload"], contents[row["owner_id"]])
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    admission_authority(root)
    service = RecoveryService(tmp_path / "recovery-control")
    try:
        inspection = service.start_inspection(archive, password=None)
        assert service.wait(inspection)["state"] == "succeeded"

        def review():
            return service.preview_restore(
                inspection,
                mode="replace",
                profile_bases={},
                target_configs={"profile": selector},
                external_destinations={},
                target=target,
                profile_names={},
                data_groups=("settings", "retrieval")
                if change == "retrieval_selected"
                else ("settings",),
            )

        if change == "retrieval_selected":
            with pytest.raises(
                ValueError, match="owner_relocation_unverified:rag.projections"
            ):
                review()
        elif change == "projection_location":
            with pytest.raises(
                ValueError, match="preserved_group_path_changed:retrieval"
            ):
                review()
        else:
            plan = review()
            assert {path for _, path in plan.restore} == {selector}
            assert {vectors, projection} <= {path for _, path in plan.preserve}
            assert not plan.retire
        assert {
            path: (path.read_bytes(), path.stat().st_dev, path.stat().st_ino)
            for path in before
        } == before
    finally:
        service.close()
