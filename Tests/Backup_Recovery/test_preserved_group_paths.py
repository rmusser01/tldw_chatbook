"""Final Settings must still locate actual data in unselected owner groups."""

import hashlib
import os
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_file_inventory import (
    installed_model as installed_model,  # noqa: PLC0414 - installed native model fixture
)
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import (
    UNBOUND_NAMESPACE,
    admission_authority,
)
from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    DiscoverySelections,
    Inventory,
)
from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
from tldw_chatbook.Backup_Recovery.profile_paths import database_path, user_data_dir
from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan
from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads


@pytest.fixture
def preserved_case(tmp_path, monkeypatch):
    # Windows Path.home() follows USERPROFILE, not the per-case HOME alone.
    monkeypatch.setenv("USERPROFILE", os.environ["HOME"])
    root = tmp_path / "local"
    root.mkdir(mode=0o700)
    selector = root / "config.toml"
    selector.write_text('[general]\nusers_name="Local"\n')
    selector.chmod(0o600)
    data = {
        "general": {"users_name": "Local"},
        "paths": {"data_dir": str(root / "data")},
        "ui": {"theme": "old"},
    }
    profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    configured = {**data, DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, profile)}
    store = database_path(data, "prompts_db_path")
    store.parent.mkdir(parents=True, mode=0o700)
    with closing(sqlite3.connect(store)) as connection, connection:
        connection.execute("CREATE TABLE retained_content(value TEXT)")
        connection.execute("INSERT INTO retained_content VALUES ('unchanged')")
    store.chmod(0o600)
    owners = {owner.owner_id: owner for owner in install_adapters()}
    with _preview_reads():
        items = tuple(
            item
            for name in ("config", "db.prompts.primary")
            for item in owners[name].discover(configured)
        )
    assert all(item.status == "included" for item in items)
    target = Inventory(items, True, "observed", ())
    plan = RestorePlan(
        "a" * 64,
        "replace",
        (("profile:source:config", selector),),
        (),
        tuple((item.logical_id, item.path) for item in items if item.owner != "config"),
        "b" * 64,
        target=target,
        requested_groups=("settings",),
        effective_groups=("settings",),
    )
    control = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: control)
    authority = admission_authority(control)
    authority.register("local", (root,))
    return plan, data, selector, store, authority


def check(plan, data, selector):
    from tldw_chatbook.Backup_Recovery.preserved_groups import (
        check_preserved_group_paths,
    )

    return check_preserved_group_paths(plan, data, selector)


@pytest.mark.parametrize("change", ["username", "data_root", "database_override"])
def test_settings_refuses_stranding_unselected_real_database(preserved_case, change):
    plan, data, selector, store, _ = preserved_case
    before = store.read_bytes(), store.stat().st_ino, selector.read_bytes()
    final = deepcopy(data)
    if change == "username":
        final["general"]["users_name"] = "Imported"
    elif change == "data_root":
        final["paths"]["data_dir"] += "-other"
    else:
        final["database"] = {"prompts_db_path": str(store.with_name("foreign.db"))}
    with (
        _preview_reads(),
        pytest.raises(ValueError, match="preserved_group_path_changed:prompts"),
    ):
        check(plan, final, selector)
    assert (store.read_bytes(), store.stat().st_ino, selector.read_bytes()) == before


def test_nonlocation_settings_change_preserves_unselected_database(preserved_case):
    plan, data, selector, store, _ = preserved_case
    final = deepcopy(data)
    final["ui"]["theme"] = "new"
    before = store.read_bytes(), selector.read_bytes()
    with _preview_reads():
        check(plan, final, selector)
    assert (store.read_bytes(), selector.read_bytes()) == before


def test_explicit_locator_can_preserve_database_across_username_change(preserved_case):
    plan, data, selector, store, _ = preserved_case
    final = deepcopy(data)
    final["general"]["users_name"] = "Imported"
    final["database"] = {"prompts_db_path": str(store)}
    with _preview_reads():
        check(plan, final, selector)


def test_selected_group_can_move_under_new_settings(preserved_case):
    plan, data, selector, _, _ = preserved_case
    plan = replace(plan, effective_groups=("settings", "prompts"))
    final = deepcopy(data)
    final["general"]["users_name"] = "Imported"
    with _preview_reads():
        check(plan, final, selector)


def test_path_proof_requires_real_read_scope(preserved_case):
    plan, data, selector, _, _ = preserved_case
    with pytest.raises(ValueError, match="preserved_group_read_scope_required"):
        check(plan, data, selector)


@pytest.mark.parametrize("held", [False, True])
def test_native_path_proof_requires_unselected_owner_namespace(preserved_case, held):
    plan, data, selector, store, authority = preserved_case
    names = (UNBOUND_NAMESPACE, "local") if held else (UNBOUND_NAMESPACE,)
    before = store.read_bytes(), store.stat().st_ino
    with authority.maintenance(names, 3) as session, session._discovery_reads():
        if held:
            check(plan, data, selector)
        else:
            with pytest.raises(
                ValueError, match="preserved_group_native_scope_required"
            ):
                check(plan, data, selector)
    assert (store.read_bytes(), store.stat().st_ino) == before


def test_settings_proof_rejects_injected_discovery_context(preserved_case):
    plan, data, selector, _, _ = preserved_case
    final = {**data, DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "forged")}
    with (
        _preview_reads(),
        pytest.raises(ValueError, match="preserved_group_config_invalid"),
    ):
        check(plan, final, selector)


def test_data_directory_resolution_uses_actual_installed_user_sanitization(
    preserved_case,
):
    plan, data, selector, _, _ = preserved_case
    final = deepcopy(data)
    final["general"]["users_name"] = "Other User"
    assert user_data_dir(final).name == "Other_User"
    with (
        _preview_reads(),
        pytest.raises(ValueError, match="preserved_group_path_changed:prompts"),
    ):
        check(plan, final, selector)


def test_native_path_proof_rejects_another_admission_authority(
    preserved_case, tmp_path
):
    plan, data, selector, _, _ = preserved_case
    foreign = admission_authority(tmp_path / "different-bootstrap")
    foreign.register("local", (selector.parent,))
    with (
        foreign.maintenance((UNBOUND_NAMESPACE, "local"), 3) as session,
        session._discovery_reads(),
        pytest.raises(ValueError, match="preserved_group_native_scope_required"),
    ):
        check(plan, data, selector)


@pytest.mark.parametrize(
    "change", ["none", "missing_destination", "existing_destination"]
)
def test_settings_preserves_unselected_database_absence(preserved_case, change):
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries

    plan, data, selector, store, _ = preserved_case
    foreign = store.with_name("other-profile.db")
    if change == "existing_destination":
        foreign.write_bytes(store.read_bytes())
        foreign.chmod(0o600)
    store.unlink()
    profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    owner = next(
        row for row in install_adapters() if row.owner_id == "db.prompts.primary"
    )
    configured = {**data, DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, profile)}
    with _preview_reads():
        missing = owner.discover(configured)
    config = next(item for item in plan.target.items if item.owner == "config")
    target = classify_entries((config, *missing))
    assert target.issues == ("missing_required",)
    plan = replace(plan, target=target)
    final = deepcopy(data)
    if change != "none":
        final["database"] = {"prompts_db_path": str(foreign)}
    with _preview_reads():
        if change == "none":
            check(plan, final, selector)
        else:
            with pytest.raises(
                ValueError, match="preserved_group_path_changed:prompts"
            ):
                check(plan, final, selector)
    assert not store.exists()


@pytest.mark.parametrize("payload_selected", [False, True])
def test_real_model_recipe_and_selected_payload_remain_reachable(
    preserved_case, installed_model, payload_selected
):
    plan, _, selector, _, _ = preserved_case
    store, descriptor, configured, owner = installed_model
    data = {
        key: value for key, value in configured.items() if key != DISCOVERY_CONTEXT_KEY
    }
    profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    configured = {
        **data,
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(
            selector,
            profile,
            DiscoverySelections(
                model_ids=(descriptor.model_id,) if payload_selected else ()
            ),
        ),
    }
    with _preview_reads():
        items = owner.discover(configured)
    assert all(
        item.status not in {"unsupported", "unavailable", "missing_required"}
        for item in items
    )
    config = next(item for item in plan.target.items if item.owner == "config")
    plan = replace(plan, target=Inventory((config, *items), True, "models", ()))
    payload = store.artifact_path(descriptor.reference) / "model.onnx"
    before = payload.read_bytes(), payload.stat().st_ino
    with _preview_reads():
        check(plan, data, selector)
    assert (payload.read_bytes(), payload.stat().st_ino) == before
    final = deepcopy(data)
    final["general"]["users_name"] = "Other"
    with (
        _preview_reads(),
        pytest.raises(ValueError, match="preserved_group_path_changed:models"),
    ):
        check(plan, final, selector)


@pytest.mark.parametrize("temporary_selected", [False, True])
def test_generated_tree_reachability_preserves_reviewed_selection(
    preserved_case, temporary_selected
):
    plan, data, selector, _, _ = preserved_case
    root = user_data_dir(data) / "generated_images"
    for group in ("saved", "temp"):
        folder = root / group
        folder.mkdir(parents=True, mode=0o700)
        (folder / "asset.png").write_bytes(b"inert local asset")
    owner = next(
        row for row in install_adapters() if row.owner_id == "generation.assets"
    )
    profile = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    configured = {
        **data,
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(
            selector, profile, DiscoverySelections(temporary_media=temporary_selected)
        ),
    }
    with _preview_reads():
        items = owner.discover(configured)
    config = next(item for item in plan.target.items if item.owner == "config")
    plan = replace(plan, target=Inventory((config, *items), True, "generated", ()))
    with _preview_reads():
        check(plan, data, selector)


@pytest.fixture
def normalized_case(preserved_case):
    import asyncio

    import toml

    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.profile_paths import default_config_path
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

    plan, data, selector, _, _ = preserved_case
    data = deepcopy(data)
    core_path = selector.parent / "conversations.db"
    tts_path = selector.parent / "voices.db"
    core = CharactersRAGDB(core_path, "preserved-settings-fixture")
    core.close()
    workspace = WorkspaceDB(
        database_path(data, "workspaces_db_path"), "preserved-settings-fixture"
    )
    workspace.close()
    # Normal startup creates this shared config parent. Its absence makes
    # installed tokenizers/voice discovery unavailable, not simply unused.
    default_config_path().parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    async def seed_tts():
        repository = TTSProfileRepository(tts_path)
        await repository.open()
        await repository.close()

    asyncio.run(seed_tts())
    data["database"] = {
        "chachanotes_db_path": str(core_path),
        "tts_profiles_db_path": str(tts_path),
    }
    selector.write_text(toml.dumps(data))
    with _preview_reads():
        target = discover((selector,))
    assert not set(target.issues) - {"missing_required", "dependency_unavailable"}, (
        target.issues,
        [
            (item.owner, item.status, str(item.path))
            for item in target.items
            if item.status == "unavailable"
        ],
    )
    plan = replace(
        plan,
        target=target,
        preserve=tuple(
            (item.logical_id, item.path)
            for item in target.items
            if item.path is not None and item.path != selector
        ),
    )
    return plan, data, selector


def test_full_discovered_shared_cohorts_allow_preference_only_settings(normalized_case):
    plan, data, selector = normalized_case
    shared = {
        item.owner: item.shared_group
        for item in plan.target.items
        if item.owner
        in {"db.chachanotes.primary", "tts.profile_store", "tts.references"}
        and item.status == "included"
    }
    assert set(shared) == {
        "db.chachanotes.primary",
        "tts.profile_store",
        "tts.references",
    }
    assert all(":profile:" not in label for label in shared.values())
    final = deepcopy(data)
    final["ui"]["theme"] = "new"
    with _preview_reads():
        check(plan, final, selector)


@pytest.mark.parametrize(
    "setting,group",
    [("chachanotes_db_path", "conversations"), ("tts_profiles_db_path", "audio")],
)
def test_full_discovered_cohort_still_refuses_locator_change(
    normalized_case, setting, group
):
    plan, data, selector = normalized_case
    final = deepcopy(data)
    final["database"][setting] = str(selector.parent / "foreign.db")
    with (
        _preview_reads(),
        pytest.raises(ValueError, match="preserved_group_path_changed:" + group),
    ):
        check(plan, final, selector)


def test_full_discovered_multiple_profiles_allow_unchanged_shared_paths(
    normalized_case,
):
    import toml

    from tldw_chatbook.Backup_Recovery.inventory import discover

    plan, data, selector = normalized_case
    second = selector.with_name("second.toml")
    second.write_text(toml.dumps(data))
    second.chmod(0o600)
    with _preview_reads():
        target = discover((selector, second))
    aliases = [item for item in target.items if item.owner == "db.prompts.primary"]
    assert len(aliases) == 2 and len({row.shared_group for row in aliases}) == 1
    assert aliases[0].shared_group is not None
    plan = replace(plan, target=target)
    with _preview_reads():
        check(plan, data, selector)


@pytest.mark.parametrize("change", ["identity", "contents"])
def test_normalized_cohort_still_requires_unchanged_physical_evidence(
    normalized_case, change
):
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery.restore_plan import (
        _fingerprint,
        _paths,
        recheck_targets,
    )

    plan, data, selector = normalized_case
    plan = replace(plan, target_fingerprint=_fingerprint(_paths(plan), plan.target))
    with _preview_reads():
        check(plan, data, selector)
    source = Path(data["database"]["chachanotes_db_path"])
    if change == "identity":
        replacement = source.with_name("different-identity.db")
        replacement.write_bytes(source.read_bytes())
        replacement.chmod(0o600)
        replacement.replace(source)
    else:
        with sqlite3.connect(source) as connection:
            connection.execute("CREATE TABLE unexpected_local_change(value TEXT)")
    with pytest.raises(ValueError, match="target_changed"):
        recheck_targets(plan)
