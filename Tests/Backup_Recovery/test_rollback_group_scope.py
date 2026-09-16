"""Selected rollback preserves reviewed scope and proves absent SQL footprints."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import later_rollback, publication, restore_plan
from tldw_chatbook.Backup_Recovery.inventory import classify_entries
from tldw_chatbook.Backup_Recovery.models import StorageItem


@pytest.fixture
def scope_case(tmp_path):
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters

    install_adapters()
    live = tmp_path / "live"
    live.mkdir(mode=0o700)
    rows = []
    config_id = "profile:local:config"
    for owner in ("config", "db.prompts.primary", "db.scheduled_tasks"):
        path = live / (owner + ".data")
        status = "missing_required" if owner == "db.scheduled_tasks" else "included"
        if status == "included":
            path.write_bytes(b"reviewed local data")
            path.chmod(0o600)
        rows.append(
            StorageItem(
                owner,
                "profile:local:" + owner,
                path,
                status,
                () if owner == "config" else (config_id,),
            )
        )
    inventory = classify_entries(tuple(rows))
    plan = restore_plan.RestorePlan(
        "a" * 64,
        "replace",
        (("profile:archive:db.prompts.primary", rows[1].path),),
        (),
        ((rows[0].logical_id, rows[0].path), (rows[2].logical_id, rows[2].path)),
        "",
        target=inventory,
        requested_groups=("prompts",),
        effective_groups=("prompts",),
    )
    return replace(
        plan,
        target_fingerprint=restore_plan._fingerprint(
            restore_plan._paths(plan), inventory
        ),
    )


def test_selected_current_target_accepts_unrelated_absent_sqlite(scope_case):
    assert later_rollback._snapshot_target_available(scope_case, scope_case.target)
    assert scope_case.target.complete is False


def test_current_selected_database_may_be_absent_after_retirement(scope_case):
    prompt = scope_case.target.items[1]
    prompt.path.unlink()
    target = classify_entries(
        tuple(
            replace(row, status="missing_required") if row == prompt else row
            for row in scope_case.target.items
        )
    )
    assert later_rollback._snapshot_target_available(scope_case, target)
    assert target.items[1].status == "missing_required"


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_current_absence_rejects_primary_or_companion_arrival(scope_case, suffix):
    absent = scope_case.target.items[2]
    Path(str(absent.path) + suffix).write_bytes(b"unreviewed arrival")
    assert not later_rollback._snapshot_target_available(scope_case, scope_case.target)


@pytest.mark.parametrize(
    "issue",
    [
        "unsupported",
        "unavailable",
        "dependency_unavailable",
        "undeclared_alias",
        "unsupported_owner",
    ],
)
def test_selected_scope_keeps_full_inventory_refusals(scope_case, issue):
    target = replace(scope_case.target, issues=(*scope_case.target.issues, issue))
    assert not later_rollback._snapshot_target_available(scope_case, target)


def test_effective_closure_cannot_expand_after_original_review(scope_case):
    prompt, scheduler = scope_case.target.items[1:]
    scheduler.path.write_bytes(b"new dependency")
    target = classify_entries(
        tuple(
            replace(row, dependencies=(*row.dependencies, scheduler.logical_id))
            if row == prompt
            else replace(row, status="included")
            if row == scheduler
            else row
            for row in scope_case.target.items
        )
    )
    assert target.complete
    assert not later_rollback._snapshot_target_available(scope_case, target)


@pytest.fixture
def reverse_dependency(scope_case):
    prompt = scope_case.target.items[1]
    path = prompt.path.with_name("library-collections.data")
    path.write_bytes(b"Library collection referencing a saved prompt")
    path.chmod(0o600)
    collection = StorageItem(
        "db.library_collections",
        "profile:local:db.library_collections",
        path,
        "included",
        (scope_case.target.items[0].logical_id, prompt.logical_id),
    )
    return collection, classify_entries((*scope_case.target.items, collection))


def test_current_rollback_refuses_new_library_dependency_on_selected_prompts(
    scope_case,
    reverse_dependency,
):
    _, target = reverse_dependency
    assert not later_rollback._snapshot_target_available(scope_case, target)


def test_historical_scope_refuses_unreviewed_library_dependency(
    scope_case,
    reverse_dependency,
):
    collection, target = reverse_dependency
    original = replace(
        scope_case,
        target=target,
        preserve=(*scope_case.preserve, (collection.logical_id, collection.path)),
    )
    prepared = SimpleNamespace(artifacts=[], directory_metadata=[])
    proof = SimpleNamespace(coverage={}, safety_sources=[])
    assert not later_rollback._historical_snapshot_target_available(
        original, prepared, proof, None
    )


@pytest.mark.parametrize(
    "status", ["unused", "intentionally_excluded", "missing_required"]
)
def test_reverse_group_check_ignores_inactive_dependent_rows(
    reverse_dependency, status
):
    from tldw_chatbook.Backup_Recovery.restore_groups import required_target_groups

    collection, target = reverse_dependency
    target = replace(
        target,
        items=tuple(
            replace(row, status=status) if row == collection else row
            for row in target.items
        ),
    )
    assert required_target_groups(target, ("prompts",)) == frozenset()


def test_reverse_group_check_does_not_select_config_dependents(reverse_dependency):
    from tldw_chatbook.Backup_Recovery.restore_groups import required_target_groups

    _, target = reverse_dependency
    assert required_target_groups(target, ("settings",)) == frozenset()


def test_legacy_plan_still_requires_complete_target(scope_case):
    legacy = replace(scope_case, requested_groups=None, effective_groups=())
    assert not later_rollback._snapshot_target_available(legacy, legacy.target)


@pytest.fixture
def absent_history(scope_case):
    prompt = scope_case.target.items[1]
    target = replace(
        scope_case.target,
        items=tuple(
            replace(row, status="missing_required") if row == prompt else row
            for row in scope_case.target.items
        ),
    )
    original = replace(scope_case, target=target)
    incoming_id = original.restore[0][0]
    artifact = SimpleNamespace(
        logical_id=incoming_id,
        target=str(prompt.path),
        action="publish",
        previous=None,
        previous_metadata=None,
        candidate=SimpleNamespace(kind="file"),
    )
    prepared = SimpleNamespace(artifacts=[artifact], directory_metadata=[])
    proof = SimpleNamespace(coverage={}, safety_sources=[])
    document = SimpleNamespace(
        files=[SimpleNamespace(logical_id=incoming_id, owner_id=prompt.owner)]
    )
    return original, prepared, proof, document


def test_historical_absence_joins_source_owner_and_destination_not_profile_id(
    absent_history,
    monkeypatch,
):
    original, prepared, proof, document = absent_history
    assert prepared.artifacts[0].logical_id != original.target.items[1].logical_id

    def no_live_paths(*args, **kwargs):
        raise AssertionError("Historical target paths must not be observed")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", no_live_paths)
        patch.setattr(Path, "resolve", no_live_paths)
        patch.setattr(later_rollback.os, "stat", no_live_paths)
        assert later_rollback._historical_snapshot_target_available(
            original, prepared, proof, document
        )


@pytest.mark.parametrize("live_dependent", [False, True])
def test_historical_absent_dependency_chain_uses_only_recorded_scope(
    absent_history, monkeypatch, live_dependent
):
    original, prepared, proof, document = absent_history
    prompt, scheduler = original.target.items[1:]
    target = replace(
        original.target,
        issues=("dependency_unavailable", "missing_required"),
        items=tuple(
            replace(
                row,
                dependencies=(*row.dependencies, prompt.logical_id),
                status="included" if live_dependent else "missing_required",
            )
            if row == scheduler
            else row
            for row in original.target.items
        ),
    )
    original = replace(original, target=target)

    def no_live_paths(*args, **kwargs):
        raise AssertionError("Historical target paths must not be observed")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", no_live_paths)
        patch.setattr(Path, "resolve", no_live_paths)
        patch.setattr(later_rollback.os, "stat", no_live_paths)
        assert later_rollback._historical_snapshot_target_available(
            original, prepared, proof, document
        ) is (not live_dependent)


@pytest.mark.parametrize(
    "damage",
    [
        "previous",
        "action",
        "candidate",
        "owner",
        "destination",
        "mapping",
        "coverage",
        "safety",
    ],
)
def test_selected_historical_absence_requires_exact_original_receipt(
    absent_history, damage
):
    original, prepared, proof, document = absent_history
    artifact = prepared.artifacts[0]
    if damage == "previous":
        artifact.previous = SimpleNamespace(kind="file")
    elif damage == "action":
        artifact.action = "retire"
    elif damage == "candidate":
        artifact.candidate = SimpleNamespace(kind="directory")
    elif damage == "owner":
        document.files[0].owner_id = "db.media.primary"
    elif damage == "destination":
        artifact.target += ".different"
    elif damage == "mapping":
        original = replace(original, restore=())
    elif damage == "coverage":
        proof.coverage[artifact.logical_id] = original.target.items[1].logical_id
    elif damage == "safety":
        original = replace(
            original, safety_scope=(original.target.items[1].logical_id,)
        )
    assert not later_rollback._historical_snapshot_target_available(
        original, prepared, proof, document
    )


def test_historical_unselected_absence_must_be_preserved(absent_history):
    original, prepared, proof, document = absent_history
    original = replace(original, preserve=original.preserve[:1])
    assert not later_rollback._historical_snapshot_target_available(
        original, prepared, proof, document
    )


def test_historical_active_rows_do_not_reobserve_now_removed_files(
    scope_case, monkeypatch
):
    scope_case.target.items[1].path.unlink()
    prepared = SimpleNamespace(artifacts=[], directory_metadata=[])
    proof = SimpleNamespace(coverage={}, safety_sources=[])
    with monkeypatch.context() as patch:
        patch.setattr(
            Path, "stat", lambda *args, **kwargs: pytest.fail("live history read")
        )
        assert later_rollback._historical_snapshot_target_available(
            scope_case, prepared, proof, None
        )


@pytest.fixture
def preview_scope(scope_case, tmp_path, monkeypatch):
    """Bound only proof inputs; exercise preview's real final plan and fingerprint.

    These orchestration cases make no native or archive authentication claim.
    The native retirement lifecycle and public service cases cover those seams.
    """
    from tldw_chatbook.Backup_Recovery import destinations

    root = tmp_path / "bootstrap"
    monkeypatch.setattr(
        later_rollback.bootstrap, "default_bootstrap_root", lambda: root
    )
    original = [scope_case]
    prepared = SimpleNamespace(
        publication=SimpleNamespace(bootstrap_root=str(root)),
        safety_sources=[],
        artifacts=[],
    )
    proof = SimpleNamespace(safety_sources=[], model_dump=dict)
    rows = [SimpleNamespace(event="prepared", evidence={})]
    journal = SimpleNamespace(
        root=tmp_path / "control" / "operation",
        operation_id="operation",
        _locked=lambda **kwargs: nullcontext(None),
        _records=lambda parent: rows,
    )
    document = SimpleNamespace(files=[], directories=[], profile_ids=[])
    archive = SimpleNamespace(digest="a" * 64, config_checks=[])

    def check_config_destinations(received_archive, plan, *, session=None):
        assert received_archive is archive
        archive.config_checks.append((plan, session))

    monkeypatch.setattr(
        destinations, "check_config_destinations", check_config_destinations
    )
    monkeypatch.setattr(later_rollback, "load_plan", lambda journal: original[0])
    monkeypatch.setattr(
        later_rollback,
        "_Prepared",
        SimpleNamespace(model_validate=lambda data: prepared),
    )
    monkeypatch.setattr(
        later_rollback.archive_reader, "verify_sealed", lambda archive: document
    )
    monkeypatch.setattr(
        later_rollback, "_verify_snapshot_source", lambda *args: (journal, proof)
    )
    monkeypatch.setattr(later_rollback, "_preserved_snapshot_members", lambda *args: {})
    monkeypatch.setattr(
        later_rollback, "_created_config_file_targets", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        later_rollback,
        "_created_destination_target",
        lambda *args, **kwargs: (args[4], {}, {}),
    )

    def plan_snapshot(*args, **kwargs):
        return replace(
            original[0],
            requested_groups=None,
            effective_groups=(),
            required_groups=(),
            local_snapshot=kwargs["local_snapshot"],
        )

    monkeypatch.setattr(later_rollback, "plan_restore", plan_snapshot)
    return original, journal, proof, archive


@pytest.mark.parametrize("selection", ["prompts", "required", "settings"])
def test_preview_carries_groups_through_two_snapshot_generations_and_plan_digest(
    preview_scope,
    selection,
):
    original, journal, proof, archive = preview_scope
    if selection == "required":
        original[0] = replace(
            original[0],
            effective_groups=("automation", "prompts"),
            required_groups=("automation",),
        )
    elif selection == "settings":
        config = original[0].target.items[0]
        original[0] = replace(
            original[0],
            requested_groups=("settings",),
            effective_groups=("settings",),
            restore=((config.logical_id, config.path),),
            preserve=tuple(
                (row.logical_id, row.path) for row in original[0].target.items[1:]
            ),
        )
    for _ in range(2):
        reviewed = later_rollback._preview(
            journal, proof, archive, original[0].target, ()
        )
        assert reviewed.requested_groups == original[0].requested_groups
        assert reviewed.effective_groups == original[0].effective_groups
        assert reviewed.required_groups == original[0].required_groups
        if selection == "settings":
            assert archive.config_checks[-1] == (reviewed, None)
        else:
            assert archive.config_checks == []
        restore_plan.recheck_targets(reviewed)
        assert publication._plan_digest(reviewed) != publication._plan_digest(
            replace(
                reviewed, requested_groups=None, effective_groups=(), required_groups=()
            )
        )
        original[0] = reviewed


@pytest.mark.parametrize("arrival", [None, "-wal", "-shm", "-journal", "unheld"])
def test_current_absence_is_rechecked_inside_actual_native_scope(
    scope_case,
    tmp_path,
    monkeypatch,
    arrival,
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import (
        UNBOUND_NAMESPACE,
        admission_authority,
    )

    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    authority = admission_authority(root)
    authority.register("local", (scope_case.target.items[0].path.parent,))
    assert later_rollback._snapshot_target_available(scope_case, scope_case.target)
    absent = scope_case.target.items[2]
    if arrival is not None and arrival != "unheld":
        Path(str(absent.path) + arrival).write_bytes(b"arrived after preview")
    names = (
        (UNBOUND_NAMESPACE,) if arrival == "unheld" else (UNBOUND_NAMESPACE, "local")
    )
    with authority.maintenance(names, 3) as session:
        assert later_rollback._snapshot_target_available(
            scope_case, scope_case.target, session=session
        ) is (arrival is None)


def test_final_known_absence_retirement_cannot_expand_reviewed_groups(
    preview_scope, monkeypatch
):
    original, journal, proof, archive = preview_scope
    unselected = original[0].target.items[2]
    unselected.path.write_bytes(b"unselected content")
    target = classify_entries(
        tuple(
            replace(row, status="included") if row == unselected else row
            for row in original[0].target.items
        )
    )
    original[0] = replace(original[0], target=target)
    known = later_rollback._known_absences

    def widened(plan, *args, **kwargs):
        result = known(plan, *args, **kwargs)
        return replace(result, retire=((unselected.logical_id, unselected.path),))

    monkeypatch.setattr(later_rollback, "_known_absences", widened)
    with pytest.raises(ValueError, match="unreviewed_group_effect:automation"):
        later_rollback._preview(journal, proof, archive, target, ())
