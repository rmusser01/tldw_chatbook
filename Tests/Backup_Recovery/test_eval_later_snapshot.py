"""Authenticated later snapshots preserve current independently owned YAML."""

import zipfile
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery import test_eval_rollback_retention
from tldw_chatbook.Backup_Recovery.later_rollback import preview_rollback
from tldw_chatbook.Backup_Recovery.models import Inventory
from tldw_chatbook.Backup_Recovery.plan_records import load_plan

rolled_back_eval = test_eval_rollback_retention.rolled_back_eval


def test_later_snapshot_preserves_current_eval_with_new_inventory_id(
    rolled_back_eval, tmp_path
):
    from tldw_chatbook.Backup_Recovery import archive_reader, recovery_copies
    from tldw_chatbook.Backup_Recovery.activation import activation_permission
    from tldw_chatbook.Backup_Recovery.journal import Journal
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter

    journal, selected, selector, context = rolled_back_eval
    original = load_plan(journal)
    current = b"tasks:\n  current: true\n"
    selected.write_bytes(current)
    target = replace(
        original.target,
        items=tuple(
            replace(item, logical_id="profile:profile:eval.definitions:current")
            if item.path == selected
            else item
            for item in original.target.items
        ),
    )
    approved = preview_rollback(
        journal.operation_id,
        control_root=journal.root.parent,
        old_password=b"old",
        target=target,
        cancel=Event(),
        acknowledged_credential_issues=("credential_format_unreadable",),
    )
    assert approved.safety_scope == ("profile:profile:eval.definitions:current",)
    assert (approved.safety_scope[0], selected) in approved.preserve
    assert selected not in dict(approved.restore).values()
    assert selected.read_bytes() == current
    operation = recovery_copies.rollback(
        journal.operation_id,
        control_root=journal.root.parent,
        old_password=b"old",
        new_password=b"new",
        cancel=Event(),
        approved_plan=approved,
    )
    assert selected.read_bytes() == current
    new_journal = Journal(journal.root.parent, operation)
    with new_journal._locked(exclusive=False) as parent:
        rows = new_journal._records(parent)
    assert rows[-1].event == "committed"
    assert not Path(rows[0].evidence["descriptor"]["path"]).parent.exists()
    copy = next(
        item
        for item in recovery_copies.list_recovery_copies(journal.root.parent)
        if item.operation_id == operation
    )
    archive = archive_reader.acquire(
        copy.path, tmp_path / "readback", ArchiveLimits(), b"new", Event()
    )
    doc = archive_reader.verify_sealed(archive)
    with zipfile.ZipFile(archive.path) as packed:
        row = next(item for item in doc.files if item.owner_id == "eval.definitions")
        assert packed.read(row.payload) == current
    assert not activation_permission("eval.definitions", config_selector=selector)
    items = _DefinitionsAdapter().discover(context)
    assert any(item.path == selected and item.status == "included" for item in items)
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    authority = admission_authority(bootstrap.default_bootstrap_root())
    names = _capture_names(authority, Inventory(items, True, "owned-component", ()))
    stage = tmp_path / "recaptured"
    stage.mkdir(mode=0o700)
    retained = next(item for item in items if item.path == selected)
    with (
        authority.maintenance(names, 3) as session,
        session.capture_scope((selected,), stage),
    ):
        _DefinitionsAdapter().capture(retained, stage / "retained.yaml", Event())
    assert (stage / "retained.yaml").read_bytes() == current
    assert not activation_permission("eval.definitions", config_selector=selector)


@pytest.mark.parametrize(
    "change", ["removed", "ambiguous", "owner", "dependency", "missing", "alias"]
)
def test_later_snapshot_requires_exact_current_preserved_owner(
    rolled_back_eval, change
):
    journal, selected, _, _ = rolled_back_eval
    target = load_plan(journal).target
    item = next(item for item in target.items if item.path == selected)
    if change == "removed":
        target = replace(
            target, items=tuple(row for row in target.items if row != item)
        )
    elif change == "ambiguous":
        target = replace(
            target,
            items=(
                *target.items,
                replace(item, logical_id=item.logical_id + ":duplicate"),
            ),
        )
    elif change in {"owner", "dependency"}:
        changed = replace(
            item,
            **({"owner": "ui.state"} if change == "owner" else {"dependencies": ()}),
        )
        target = replace(
            target, items=tuple(changed if row == item else row for row in target.items)
        )
    elif change == "missing":
        selected.unlink()
    else:
        actual = selected.with_suffix(".actual")
        selected.rename(actual)
        selected.symlink_to(actual)
    with pytest.raises(
        (ValueError, FileNotFoundError),
        match="local_snapshot_preservation_unverified|No such file|destination_alias",
    ):
        preview_rollback(
            journal.operation_id,
            control_root=journal.root.parent,
            old_password=b"old",
            target=target,
            cancel=Event(),
            acknowledged_credential_issues=("credential_format_unreadable",),
        )


@pytest.mark.parametrize("proof", ["ordinary", "changed", "restore_preserved"])
def test_partial_group_requires_authenticated_preserved_source(
    rolled_back_eval, tmp_path, proof
):
    from tldw_chatbook.Backup_Recovery import archive_reader, recovery_copies
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    journal, selected, _, _ = rolled_back_eval
    target = load_plan(journal).target
    approved = preview_rollback(
        journal.operation_id,
        control_root=journal.root.parent,
        old_password=b"old",
        target=target,
        cancel=Event(),
        acknowledged_credential_issues=("credential_format_unreadable",),
    )
    entry = next(
        item
        for item in recovery_copies.list_recovery_copies(journal.root.parent)
        if item.operation_id == journal.operation_id
    )
    archive = archive_reader.acquire(
        entry.path, tmp_path / "source", ArchiveLimits(), b"old", Event()
    )
    snapshot = approved.local_snapshot
    destinations = {**dict(approved.destinations), **dict(approved.selectors)}
    if proof == "ordinary":
        snapshot = None
    elif proof == "changed":
        snapshot = replace(snapshot, rollback_digest="0" * 64)
    else:
        doc = archive_reader.verify_sealed(archive)
        row = next(item for item in doc.files if item.owner_id == "eval.definitions")
        destinations[row.root_id] = selected.parent
    with pytest.raises(
        ValueError,
        match="dependency_group_incomplete|local_snapshot_source_changed|local_snapshot_preservation_unverified",
    ):
        plan_restore(
            archive,
            mode="replace",
            destinations=destinations,
            target=target,
            safety_scope=approved.safety_scope,
            local_snapshot=snapshot,
            profile_names=dict(approved.profile_names),
            acknowledged_credential_issues=approved.acknowledged_credential_issues,
        )
