"""Selection behavior for installed whole-owner backup groups."""

from dataclasses import replace
from importlib import import_module

import pytest

from tldw_chatbook.Backup_Recovery.models import FileMetadata, StorageItem


def _api():
    return import_module("tldw_chatbook.Backup_Recovery.data_groups")


def _item(
    owner, local="", *, profile="one", status="included", dependencies=(), **kwargs
):
    logical_id = f"profile:{profile}:{owner}" + (f":{local}" if local else "")
    return StorageItem(owner, logical_id, None, status, dependencies, **kwargs)


def test_everything_keeps_durable_groups_and_existing_optional_exclusions():
    items = (
        _item("config"),
        _item("db.prompts.primary", status="unused"),
        _item("db.media.primary"),
        _item("models.artifacts", status="intentionally_excluded"),
        _item("runtime.instance_lock", status="intentionally_excluded"),
        _item("unknown", status="unsupported"),
    )
    result = _api().resolve_inventory_groups(items, None)
    assert result.requested_groups is None
    assert set(result.effective_groups) == {"settings", "prompts", "library", "models"}
    assert set(result.member_ids) == {
        "profile:one:config",
        "profile:one:db.prompts.primary",
        "profile:one:db.media.primary",
        "profile:one:models.artifacts",
    }
    assert result.required_groups == result.support_ids == ()


@pytest.mark.parametrize(
    "selection",
    [(), [], "prompts", ("prompts", "prompts"), ("missing",), (1,), (None,)],
)
def test_invalid_group_selection_refuses(selection):
    with pytest.raises(ValueError, match="invalid_backup_groups"):
        _api().resolve_inventory_groups((_item("db.prompts.primary"),), selection)


def test_exact_group_selection_includes_every_owner_member_across_profiles():
    items = (
        _item("db.prompts.primary"),
        _item("chat.prompts", "first"),
        _item("chat.prompts", "second"),
        _item("db.prompts.primary", profile="two"),
        _item("db.media.primary"),
    )
    result = _api().resolve_inventory_groups(items, ("prompts",))
    assert result.requested_groups == result.effective_groups == ("prompts",)
    assert set(result.member_ids) == {
        "profile:one:db.prompts.primary",
        "profile:one:chat.prompts:first",
        "profile:one:chat.prompts:second",
        "profile:two:db.prompts.primary",
    }


def test_config_dependency_is_whole_owner_support_without_settings_selection():
    config = _item("config")
    companion = _item("config", "companion", dependencies=(config.logical_id,))
    items = (
        _item("db.prompts.primary", dependencies=(config.logical_id,)),
        config,
        companion,
        _item("config.history"),
        _item("ui.state"),
        _item("config", profile="unrelated"),
    )
    result = _api().resolve_inventory_groups(items, ("prompts",))
    assert result.effective_groups == ("prompts",)
    assert result.member_ids == ("profile:one:db.prompts.primary",)
    assert set(result.support_ids) == {
        "profile:one:config",
        "profile:one:config:companion",
    }


def test_explicit_settings_promotes_config_support_to_selected_members():
    config = _item("config")
    items = (
        _item("db.prompts.primary", dependencies=(config.logical_id,)),
        config,
        _item("config.history"),
        _item("ui.state"),
    )
    result = _api().resolve_inventory_groups(items, ("prompts", "settings"))
    assert set(result.member_ids) == {
        "profile:one:config",
        "profile:one:config.history",
        "profile:one:ui.state",
        "profile:one:db.prompts.primary",
    }
    assert result.support_ids == result.required_groups == ()


def test_shared_chachanotes_owners_and_related_assets_cannot_be_split():
    owners = (
        "db.chachanotes.primary",
        "study.local",
        "quiz.local",
        "notes.sync_bindings",
        "chat.attachments",
        "notes.file_notes",
        "personas",
        "persona.assets",
        "persona.visual_identity",
        "persona.visual_identity_builtin",
        "recovered.media",
    )
    items = tuple(_item(owner) for owner in owners) + (_item("db.media.primary"),)
    result = _api().resolve_inventory_groups(items, ("conversations",))
    assert {
        item.owner for item in items if item.logical_id in result.member_ids
    } == set(owners)


def test_forward_dependency_cycles_close_entire_required_groups_globally():
    media = _item("db.media.primary", dependencies=("profile:one:db.prompts.primary",))
    prompts = _item("db.prompts.primary", dependencies=(media.logical_id,))
    items = (
        media,
        prompts,
        _item("chat.prompts", "template"),
        _item("chat.prompts", "template", profile="two"),
        _item("writing.local"),
    )
    result = _api().resolve_inventory_groups(items, ("library",))
    assert set(result.effective_groups) == {"library", "prompts"}
    assert result.required_groups == ("prompts",)
    assert set(result.member_ids) == {
        "profile:one:db.media.primary",
        "profile:one:db.prompts.primary",
        "profile:one:chat.prompts:template",
        "profile:two:chat.prompts:template",
    }


def test_shared_store_adds_other_group_and_all_its_declared_members():
    items = (
        _item("db.prompts.primary", shared_group="shared:approved"),
        _item("db.media.primary", shared_group="shared:approved"),
        _item("db.library_ingest_jobs"),
    )
    result = _api().resolve_inventory_groups(items, ("prompts",))
    assert result.required_groups == ("library",)
    assert set(result.member_ids) == {
        "profile:one:db.prompts.primary",
        "profile:one:db.media.primary",
        "profile:one:db.library_ingest_jobs",
    }


def test_tree_topology_closes_declared_root_and_parent_edges_without_paths():
    root = _item("db.media.primary", "root", status="included_directory")
    parent = _item("chat.prompts", "parent", status="included_directory")
    child = _item(
        "writing.local",
        "child",
        metadata=FileMetadata(
            1,
            root.logical_id,
            "folder/child.txt",
            parent.logical_id,
            "file",
            0o600,
            0,
            "private",
        ),
    )
    result = _api().resolve_inventory_groups((root, parent, child), ("writing",))
    assert set(result.required_groups) == {"library", "prompts"}
    assert set(result.member_ids) == {
        "profile:one:db.media.primary:root",
        "profile:one:chat.prompts:parent",
        "profile:one:writing.local:child",
    }


def test_archive_synthetic_roots_are_selected_through_their_installed_owner():
    root = StorageItem(
        "chat.prompts", "root:archive-hash", None, "included_directory", ()
    )
    result = _api().resolve_inventory_groups(
        (root, _item("chat.prompts", "child")), ("prompts",)
    )
    assert set(result.member_ids) == {
        "root:archive-hash",
        "profile:one:chat.prompts:child",
    }


@pytest.mark.parametrize("edge", ["dependency", "root", "parent"])
def test_missing_selected_dependency_or_topology_edge_refuses(edge):
    metadata = None
    dependencies = ("absent",) if edge == "dependency" else ()
    if edge != "dependency":
        metadata = FileMetadata(
            1,
            "absent" if edge == "root" else "profile:one:chat.prompts",
            "child.txt",
            "absent" if edge == "parent" else None,
            "file",
            0o600,
            0,
            "private",
        )
    item = _item("chat.prompts", dependencies=dependencies, metadata=metadata)
    with pytest.raises(ValueError, match="dependency_unavailable"):
        _api().resolve_inventory_groups((item,), ("prompts",))


@pytest.mark.parametrize("status", ["unused", "intentionally_excluded"])
def test_unused_or_excluded_members_do_not_follow_stale_dependencies(status):
    item = _item("db.prompts.primary", status=status, dependencies=("absent",))
    result = _api().resolve_inventory_groups((item,), ("prompts",))
    assert result.effective_groups == ("prompts",)
    assert result.member_ids == ("profile:one:db.prompts.primary",)


def test_unselected_broken_dependencies_remain_for_inventory_classification():
    items = (
        _item("db.prompts.primary"),
        _item("db.media.primary", dependencies=("absent",)),
    )
    result = _api().resolve_inventory_groups(items, ("prompts",))
    assert result.member_ids == ("profile:one:db.prompts.primary",)


def test_unknown_included_dependency_refuses_without_guessing_group_authority():
    unknown = _item("future.owner")
    items = (_item("db.prompts.primary", dependencies=(unknown.logical_id,)), unknown)
    with pytest.raises(ValueError, match="unsupported_group_owner"):
        _api().resolve_inventory_groups(items, ("prompts",))


def test_unknown_blocking_dependency_is_retained_as_support_for_classifier():
    unknown = _item("future.owner", status="unsupported")
    items = (_item("db.prompts.primary", dependencies=(unknown.logical_id,)), unknown)
    result = _api().resolve_inventory_groups(items, ("prompts",))
    assert result.support_ids == ("profile:one:future.owner",)


def test_duplicate_logical_inventory_ids_refuse():
    item = _item("db.prompts.primary")
    with pytest.raises(ValueError, match="duplicate_logical_id"):
        _api().resolve_inventory_groups((item, item), ("prompts",))


def test_resolution_is_stable_after_excluding_unselected_rows_and_reordering():
    config = _item("config")
    items = (
        config,
        _item("db.prompts.primary", dependencies=(config.logical_id,)),
        _item("chat.prompts", status="unused"),
        _item("db.media.primary"),
    )
    before = _api().resolve_inventory_groups(items, ("prompts",))
    excluded = tuple(
        replace(item, status="intentionally_excluded")
        if item.logical_id not in before.member_ids + before.support_ids
        else item
        for item in reversed(items)
    )
    assert _api().resolve_inventory_groups(excluded, ("prompts",)) == before


@pytest.mark.parametrize("selection", [("prompts",), ("library",)])
def test_repeated_tree_edges_expand_each_whole_group_only_once(monkeypatch, selection):
    """Measure actual set-operation inputs, independent of machine speed.

    Each file repeats its parent in dependencies and topology, as real owner
    trees do. Instrumented sets preserve ordinary set behavior while recording
    how much group data the resolver submits to difference operations.
    """
    scanned_members = 0

    class ObservedSet(set):
        def __sub__(self, other):
            nonlocal scanned_members
            scanned_members += len(self)
            return super().__sub__(other)

    config = _item("config")
    root = _item("chat.prompts", "root", status="included_directory")
    files = tuple(
        _item(
            "chat.prompts",
            f"file-{index}",
            dependencies=(root.logical_id, config.logical_id),
            metadata=FileMetadata(
                1,
                root.logical_id,
                f"file-{index}.txt",
                root.logical_id,
                "file",
                0o600,
                0,
                "private",
            ),
        )
        for index in range(256)
    )
    items = (
        config,
        root,
        *files,
        _item("db.media.primary", dependencies=(root.logical_id,)),
    )
    # Observe this module's ordinary set constructors only. Production gets no
    # counters, alternate algorithms, filesystem dependencies, or test hooks.
    monkeypatch.setattr(_api(), "set", ObservedSet, raising=False)

    result = _api().resolve_inventory_groups(items, selection)

    assert len(result.member_ids) == 257 + (selection == ("library",))
    assert result.support_ids == (config.logical_id,)
    assert scanned_members <= 2 * len(items)


def test_every_durable_installed_adapter_has_a_named_selectable_group():
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    nonselectable = {
        "cache.model_catalog",
        "runtime.instance_lock",
        "runtime.config_lock",
        "runtime.chatbook_scratch",
        "research.paste_staging",
        "actor_packs.import_staging",
        "collections.archives",
        "runtime.crash_forensics",
        "runtime.scheduler_heartbeat",
        "notes.sync_state",
        "diagnostics.logs",
    }
    for adapter in install_adapters():
        group_id = _api().group_for_owner(adapter.owner_id)
        if adapter.owner_id in nonselectable:
            assert group_id is None
        else:
            assert group_id in {group.group_id for group in _api().BACKUP_GROUPS}, (
                adapter.owner_id
            )
    assert _api().group_for_owner("future.owner") is None
