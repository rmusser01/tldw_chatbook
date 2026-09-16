"""Preserved dependencies must be reviewed before replacement staging starts."""

from dataclasses import replace
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery.models import FileMetadata, Inventory, StorageItem
from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan


def dependency_plan(tmp_path):
    core = StorageItem(
        "db.chachanotes.primary", "core", tmp_path / "core.db", "included", ("assets",)
    )
    root = StorageItem(
        "persona.visual_identity_builtin",
        "assets",
        tmp_path / "assets",
        "included_directory",
        ("core", "asset"),
        metadata=FileMetadata(1, "assets", "", None, "directory", 0o700, 0, "private"),
    )
    asset = StorageItem(
        "persona.visual_identity_builtin",
        "asset",
        tmp_path / "assets/image.png",
        "included",
        ("assets", "core"),
        metadata=FileMetadata(
            1, "assets", "image.png", "assets", "file", 0o600, 0, "private"
        ),
    )
    unrelated = StorageItem("ui.state", "ui", tmp_path / "ui.toml", "included", ())
    return RestorePlan(
        "digest",
        "replace",
        (("core", core.path),),
        (),
        tuple((i.logical_id, i.path) for i in (root, asset, unrelated)),
        "fingerprint",
        target=Inventory((core, root, asset, unrelated), True, "scope", ()),
    )


@pytest.mark.parametrize(
    "selected,expected",
    [((), ("asset", "assets")), (("assets",), ("asset",)), (("assets", "asset"), ())],
)
def test_required_dependency_closure_is_exact(tmp_path, selected, expected):
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        required_rollback_dependencies,
    )

    plan = replace(dependency_plan(tmp_path), safety_scope=selected)
    assert required_rollback_dependencies(plan) == expected


def test_missing_safety_scope_refuses_before_native_qualification(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import qualification

    def unexpected(*a, **kw):
        pytest.fail("native qualification is later than plan dependency preflight")

    monkeypatch.setattr(qualification, "_release_for_roots", unexpected)
    assert qualification.replacement_capability(
        dependency_plan(tmp_path), control_root=tmp_path / "control"
    ) == (False, "rollback_dependency_selection_required")


def test_missing_safety_scope_refuses_before_workspace_or_archive_read(tmp_path):
    from tldw_chatbook.Backup_Recovery.archive_models import SealedArchive
    from tldw_chatbook.Backup_Recovery.staging import stage_restore

    archive = SealedArchive(tmp_path / "nonexistent.zip", "digest", b"{}")
    work = tmp_path / "candidate-work"
    with pytest.raises(ValueError, match="^rollback_dependency_selection_required$"):
        stage_restore(archive, dependency_plan(tmp_path), work, Event())
    assert not work.exists()


def test_exact_explicit_scope_keeps_original_qualification(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import qualification

    monkeypatch.setattr(
        qualification,
        "_release_for_roots",
        lambda *a, **kw: (False, "original_native_reason"),
    )
    plan = replace(dependency_plan(tmp_path), safety_scope=("assets", "asset"))
    assert qualification.replacement_capability(
        plan, control_root=tmp_path / "control"
    ) == (False, "original_native_reason")


def test_preserved_alias_of_mutated_file_is_already_in_originals(tmp_path):
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        required_rollback_dependencies,
    )

    plan = dependency_plan(tmp_path)
    core = replace(plan.target.items[0], dependencies=("alias",))
    alias = replace(core, logical_id="alias", dependencies=())
    plan = replace(
        plan,
        target=Inventory((core, alias), True, "scope", ()),
        preserve=(("alias", alias.path),),
    )
    assert required_rollback_dependencies(plan) == ()
