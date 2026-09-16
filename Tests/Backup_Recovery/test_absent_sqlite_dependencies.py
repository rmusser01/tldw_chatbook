"""Absent databases can depend on each other without authorizing missing live data."""

from dataclasses import replace

import pytest

from Tests.Backup_Recovery import test_first_binding_absent_sqlite
from tldw_chatbook.Backup_Recovery.inventory import classify_entries
from tldw_chatbook.Backup_Recovery.models import StorageItem
from tldw_chatbook.Backup_Recovery.replacement import _first_binding_scope_available

absent_database = test_first_binding_absent_sqlite.absent_database


@pytest.fixture
def absent_dependency(absent_database):
    _, _, inventory, scheduler = absent_database
    prompt = next(row for row in inventory.items if row.owner == "db.prompts.primary")
    prompt.path.unlink()
    target = classify_entries(
        tuple(
            replace(row, status="missing_required", metadata=None)
            if row == prompt
            else replace(row, dependencies=(*row.dependencies, prompt.logical_id))
            if row == scheduler
            else row
            for row in inventory.items
        )
    )
    assert target.issues == ("dependency_unavailable", "missing_required")
    return target, prompt, scheduler


def test_absent_sqlite_chain_allows_first_binding_without_marking_complete(
    absent_dependency,
):
    target, _, _ = absent_dependency
    assert _first_binding_scope_available(target)
    assert not target.complete


@pytest.mark.parametrize(
    "change", ["live_dependent", "unknown", "unused", "non_sqlite"]
)
def test_absence_does_not_authorize_other_unavailable_dependencies(
    absent_dependency, change
):
    target, prompt, scheduler = absent_dependency
    rows = list(target.items)
    if change == "live_dependent":
        scheduler.path.write_bytes(b"existing database must not lose a dependency")
        rows = [
            replace(row, status="included")
            if row.logical_id == scheduler.logical_id
            else row
            for row in rows
        ]
    elif change == "unknown":
        rows = [
            replace(row, dependencies=(*row.dependencies, "unknown"))
            if row.logical_id == scheduler.logical_id
            else row
            for row in rows
        ]
    elif change == "unused":
        rows = [
            replace(row, status="unused")
            if row.logical_id == prompt.logical_id
            else row
            for row in rows
        ]
    else:
        rows = [
            replace(row, owner="ui.state")
            if row.logical_id == prompt.logical_id
            else row
            for row in rows
        ]
    assert not _first_binding_scope_available(classify_entries(tuple(rows)))


def test_absent_dependency_chain_does_not_hide_unrelated_broken_edge(absent_dependency):
    target, prompt, _ = absent_dependency
    live = prompt.path.with_name("live-state.json")
    live.write_bytes(b"{}")
    row = StorageItem(
        "ui.state", "profile:local:ui.state", live, "included", (prompt.logical_id,)
    )
    assert not _first_binding_scope_available(classify_entries((*target.items, row)))


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_absent_dependency_chain_still_refuses_arrivals(absent_dependency, suffix):
    target, prompt, _ = absent_dependency
    prompt.path.with_name(prompt.path.name + suffix).write_bytes(b"unreviewed")
    assert not _first_binding_scope_available(target)
