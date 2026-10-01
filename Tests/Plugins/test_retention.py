"""Retention selects only eligible immutable material, never protected authority."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.fixture
def retention_case():
    now = datetime(2026, 9, 16, tzinfo=UTC)
    rows = tuple(
        {
            "installation_id": "one",
            "revision_digest": key,
            "created_at": now - timedelta(days=days),
            "current": current,
        }
        for key, days, current in (
            ("current", 0, True),
            ("old", 1, False),
            ("recent", 2, False),
            ("leased", 40, False),
            ("recovery", 45, False),
            ("expired", 50, False),
        )
    )
    return SimpleNamespace(
        revisions=rows,
        protected=frozenset({"current", "leased", "recovery"}),
        now=now,
        expired_unleased="expired",
    )


def test_retention_never_removes_recovery_material(retention_case):
    from tldw_chatbook.Plugins.retention import retention_candidates

    case = retention_case
    candidates = retention_candidates(case.revisions, case.protected, case.now)
    assert not (set(candidates) & case.protected)
    assert case.expired_unleased in candidates
    assert not set(candidates) & {"old", "recent"}


def test_retention_count_and_age_are_per_installation(retention_case):
    from tldw_chatbook.Plugins.retention import retention_candidates

    case = retention_case
    rows = tuple(
        {
            "installation_id": owner,
            "revision_digest": f"{owner}-{index}",
            "created_at": case.now - timedelta(days=index),
            "current": index == 0,
        }
        for owner in ("a", "b")
        for index in range(4)
    )
    assert set(retention_candidates(rows, frozenset(), case.now)) == {"a-3", "b-3"}
    expired = (
        {
            "installation_id": "c",
            "revision_digest": "ancient",
            "created_at": case.now - timedelta(days=31),
            "current": False,
        },
    )
    assert retention_candidates(expired, frozenset(), case.now) == ("ancient",)


def test_rolling_transition_retention_keeps_current_recovery_and_stales_original(
    plugin_stack, native_package, monkeypatch
):
    from Tests.Plugins.test_coordinator import reviewed
    from tldw_chatbook.Plugins import authority_store

    stack = plugin_stack
    monkeypatch.setattr(authority_store, "MAX_TRANSITIONS", 3)
    first = None
    for _ in range(6):
        review = reviewed(stack, native_package())
        first = first or review
        assert stack.call(
            lambda review=review: stack.coordinator.commit(review, review.operation_id)
        ).committed
    inventory = stack.call(lambda: stack.authority.list_transitions(limit=50, offset=0))
    assert len(inventory) <= 3
    assert len(stack.call(stack.coordinator.published_snapshot)["installations"]) == 6
    assert all(
        item.phase == "complete" for item in stack.call(stack.coordinator.recover)
    )
    with pytest.raises(ValueError, match="stale"):
        stack.call(lambda: stack.coordinator.commit(first, first.operation_id))


@pytest.mark.asyncio
async def test_actual_revision_compaction_precedes_package_removal(
    native_console, native_package
):
    rig = native_console
    installed = await rig.install()
    roots = []
    for index in range(4):
        root = native_package()
        file = root / "skills/review/SKILL.md"
        file.write_text(file.read_text() + f"\nRevision {index}\n")
        review = await rig.service.review_revision(installed.installation_id, root)
        assert (await rig.service.apply_revision(review, review.operation_id)).committed
        snapshot = await rig.service._call(rig.service._coordinator.published_snapshot)
        roots = [row["materialized_identity"] for row in snapshot["revisions"]]
    result = await rig.service.retain_revisions(installed.installation_id)
    assert result.committed and not result.cleanup_pending
    snapshot = await rig.service._call(rig.service._coordinator.published_snapshot)
    assert len(snapshot["revisions"]) == 3
    from pathlib import Path

    assert sum(Path(root).exists() for root in roots) == 3
    assert all(
        receipt.phase == "complete"
        for receipt in await rig.service._call(rig.service._coordinator.recover)
    )


def test_quota_counts_unowned_staging_and_refuses_protected_capacity(
    plugin_stack, monkeypatch
):
    from tldw_chatbook.Plugins import retention

    root = plugin_stack.owner.root
    (root / "staging").mkdir()
    (root / "staging" / "unowned").write_bytes(b"x" * 100)
    monkeypatch.setattr(retention, "MANAGED_BYTES", 100)
    assert retention.managed_usage(root) == 100
    with pytest.raises(OSError, match="quota"):
        retention.require_capacity(root, 1)
    assert (root / "staging" / "unowned").exists()


@pytest.mark.parametrize(
    "state", ["owned", "unknown", "active_producer", "recovery", "young", "replaced"]
)
def test_staging_cleanup_requires_original_reconciled_terminal_owner(
    plugin_stack, tmp_path, state
):
    from dataclasses import replace

    from tldw_chatbook.Plugins.retention import StagingCustody, cleanup_staging

    stack = plugin_stack
    root = stack.owner.root / "staging" / "stage"
    root.mkdir(parents=True)
    (root / "package").write_text("owned stage bytes")
    info = root.stat()
    now = datetime.now(UTC)
    custody = StagingCustody(
        "stage",
        "acquisition-request",
        now - timedelta(hours=25),
        info.st_dev,
        info.st_ino,
        True,
        True,
        False,
    )
    if state == "unknown":
        custody = replace(custody, reconciled=False)
    elif state == "active_producer":
        custody = replace(custody, terminal=False)
    elif state == "recovery":
        custody = replace(custody, recovery_referenced=True)
    elif state == "young":
        custody = replace(custody, created_at=now)
    elif state == "replaced":
        root.rename(root.with_name("original"))
        root.mkdir()
        (root / "new").write_text("new owner")
        with pytest.raises(ValueError, match="identity"):
            stack.call(lambda: cleanup_staging(stack.owner, custody, now))
        assert (root / "new").read_text() == "new owner"
        return
    assert stack.call(lambda: cleanup_staging(stack.owner, custody, now)) == (
        state == "owned"
    )
    assert root.exists() != (state == "owned")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cleanup_state", ["original", "replaced_leaf", "missing_anchor", "missing_leaf"]
)
async def test_retention_restart_uses_original_authenticated_cleanup_identity(
    native_console, native_package, monkeypatch, cleanup_state
):
    from pathlib import Path

    from tldw_chatbook.Plugins import retention
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService

    rig = native_console
    installed = await rig.install()
    original_remove = retention.remove_revision_root

    def fail_cleanup(*args, **kwargs):
        if kwargs.get("remove", True):
            raise OSError("controlled removal failure")
        return original_remove(*args, **kwargs)

    monkeypatch.setattr(retention, "remove_revision_root", fail_cleanup)
    for index in range(3):
        root = native_package()
        file = root / "skills/review/SKILL.md"
        file.write_text(file.read_text() + f"\nCleanup revision {index}\n")
        review = await rig.service.review_revision(installed.installation_id, root)
        result = await rig.service.apply_revision(review, review.operation_id)
    assert result.committed and result.cleanup_pending
    evidence = await rig.service._call(
        lambda: next(
            item
            for item in retention.transition_inventory(
                rig.service._coordinator.authority
            )
            if item.snapshot["operation_result"]["kind"] == "retain"
        )
    )
    row = evidence.snapshot["operation_result"]["retired_revisions"][0]
    path = Path(row["materialized_identity"])
    assert path.exists()
    await rig.service.aclose()
    if cleanup_state == "replaced_leaf":
        path.rename(path.with_name(path.name + "-original"))
        path.mkdir()
        (path / "new-owner").write_text("must survive")
    moved_anchor = path.parent.with_name(path.parent.name + "-moved")
    if cleanup_state == "missing_anchor":
        path.parent.rename(moved_anchor)
    elif cleanup_state == "missing_leaf":
        import shutil

        shutil.rmtree(path)
    monkeypatch.setattr(retention, "remove_revision_root", original_remove)
    restarted = PluginService(
        rig.service.profile_root,
        workspace_lookup=rig.registry.get_workspace,
        marker_store_factory=lambda _: FilePluginMarkerStore(
            rig.service.profile_root.parent / "marker"
        ),
        accept_reduced_protection=True,
    )
    try:
        await restarted.unlock("test passphrase")
        pending = await restarted.lookup_operation(evidence.new.operation_id)
        assert pending.committed and pending.cleanup_pending == (
            cleanup_state != "missing_leaf"
        )
        assert path.exists() == (cleanup_state in {"original", "replaced_leaf"})
        result = await restarted.retain_revisions(installed.installation_id)
        assert result.operation_id == evidence.new.operation_id
        assert result.committed and result.cleanup_pending == (
            cleanup_state in {"replaced_leaf", "missing_anchor"}
        )
        assert path.exists() == (cleanup_state == "replaced_leaf")
        if cleanup_state == "replaced_leaf":
            assert (path / "new-owner").read_text() == "must survive"
        if cleanup_state == "missing_anchor":
            assert (moved_anchor / path.name).is_dir()
            assert (
                await restarted._call(
                    lambda: restarted._coordinator.authority.verify_transition(
                        evidence.new.operation_id
                    )
                )
            ).committed
            moved_anchor.rename(path.parent)
            assert not (
                await restarted.retain_revisions(installed.installation_id)
            ).cleanup_pending
            assert not path.exists()
    finally:
        await restarted.aclose()


def test_update_storage_link_refuses_before_creating_external_directories(
    plugin_stack, native_package, tmp_path
):
    from Tests.Plugins.test_coordinator import reviewed

    stack = plugin_stack
    initial = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(initial, initial.operation_id))
    external = tmp_path / "external-storage"
    external.mkdir()
    (stack.owner.root / "packages-revisions").symlink_to(
        external, target_is_directory=True
    )
    package = native_package()
    skill = package / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nchanged\n")
    review = stack.call(
        lambda: stack.coordinator.review_revision(
            initial.installation_id,
            __import__(
                "tldw_chatbook.Plugins.inspection", fromlist=["inspect_package"]
            ).inspect_package(package),
        )
    )
    with pytest.raises((OSError, ValueError)):
        stack.call(
            lambda: stack.coordinator.apply_revision(review, review.operation_id)
        )
    assert list(external.iterdir()) == []


@pytest.mark.asyncio
async def test_original_revocation_request_lookup_survives_restart_without_mutation(
    native_console,
    native_package,
    monkeypatch,
):
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.revocation import RevocationTarget
    from tldw_chatbook.Plugins.service import PluginService

    rig = native_console
    installed = await rig.install()
    request = rig.service.begin_disable(
        RevocationTarget(installed.installation_id, None, True)
    )
    completed = await rig.service.finish_revocation(request)
    assert completed.committed and completed.request_id == request.request_id
    await rig.service.aclose()
    restarted = PluginService(
        rig.service.profile_root,
        workspace_lookup=rig.registry.get_workspace,
        marker_store_factory=lambda _: FilePluginMarkerStore(
            rig.service.profile_root.parent / "marker"
        ),
        accept_reduced_protection=True,
    )
    try:
        await restarted.unlock("test passphrase")
        before = await restarted._call(restarted._coordinator.authority.load_marker)
        found = await restarted.lookup_operation(request.request_id)
        assert found.committed and found.operation_id == completed.operation_id
        assert found.request_id == request.request_id
        missing = await restarted.lookup_operation("lr1." + "0" * 32 + "." + "1" * 32)
        assert missing.phase == "unavailable_or_expired" and not missing.committed
        assert (
            await restarted._call(restarted._coordinator.authority.load_marker)
            == before
        )
        from tldw_chatbook.Plugins import authority_store

        monkeypatch.setattr(authority_store, "MAX_TRANSITIONS", 6)
        for _ in range(8):
            review = await restarted.review_install(
                native_package(), selection=("skill:review",), workspace_id=None
            )
            await restarted.commit(review, review.operation_id)
        marker = await restarted._call(restarted._coordinator.authority.load_marker)
        expired = await restarted.lookup_operation(request.request_id)
        assert expired.phase == "unavailable_or_expired" and not expired.committed
        assert (
            await restarted._call(restarted._coordinator.authority.load_marker)
            == marker
        )
    finally:
        await restarted.aclose()


@pytest.mark.asyncio
async def test_install_uninstall_cycles_retire_only_removed_revision_observations(
    native_console, native_package, monkeypatch
):
    from tldw_chatbook.Plugins import authority_store

    rig = native_console
    kept = await rig.install()
    monkeypatch.setattr(authority_store, "MAX_TRANSITIONS", 6)
    for _ in range(5):
        review = await rig.service.review_install(
            native_package(), selection=("skill:review",), workspace_id=None
        )
        await rig.service.commit(review, review.operation_id)
        assert (await rig.service.uninstall(review.installation_id)).committed

    def inspect_metadata():
        coordinator = rig.service._coordinator
        with coordinator.registry.transaction() as cursor:
            rows = cursor.execute(
                "SELECT operation_id FROM receipts WHERE operation_id LIKE 'revision:%'"
            ).fetchall()
            count = cursor.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        inventory = coordinator.authority.list_transitions(limit=50, offset=0)
        return [row[0] for row in rows], count, len(inventory)

    observed, receipt_count, transitions = await rig.service._call(inspect_metadata)
    assert observed == ["revision:" + kept.installation_id]
    assert transitions <= 6 and receipt_count <= transitions + 1
    assert rig.service.capture_maximum("workspace-a")["available_skills"]


@pytest.mark.parametrize("cut", ["hints", "certificate", "intent", "snapshot"])
def test_coordinator_pruning_recovers_each_durable_retirement_boundary(
    plugin_stack, native_package, monkeypatch, cut
):
    from Tests.Plugins.test_coordinator import reviewed
    from tldw_chatbook.Plugins import retention

    stack = plugin_stack
    for _ in range(3):
        review = reviewed(stack, native_package())
        stack.call(
            lambda review=review: stack.coordinator.commit(review, review.operation_id)
        )
    current = stack.call(stack.authority.load_marker)
    original_snapshot = stack.call(stack.authority.verify_current)
    oldest = min(
        stack.call(lambda: retention.transition_inventory(stack.authority)),
        key=lambda item: item.new.generation,
    )
    old_snapshot_path = stack.call(lambda: stack.authority._snapshot_path(oldest.new))
    observed = stack.registry.operation_observed_at
    monkeypatch.setattr(
        stack.registry,
        "operation_observed_at",
        lambda _: datetime.now(UTC) - timedelta(days=31),
    )
    crossed = []
    if cut == "hints":
        original = stack.registry.forget_operation_hints

        def interrupt(ids):
            original(ids)
            crossed.append("hints")
            raise OSError("controlled durable retirement cut")

        monkeypatch.setattr(stack.registry, "forget_operation_hints", interrupt)
    else:
        original = stack.authority._unlink_artifact

        def interrupt(path):
            original(path)
            purpose = {
                "certificates": "certificate",
                "intents": "intent",
                "snapshots": "snapshot",
            }.get(path.parent.name)
            if purpose == cut:
                crossed.append(purpose)
                raise OSError("controlled durable retirement cut")

        monkeypatch.setattr(stack.authority, "_unlink_artifact", interrupt)
    with pytest.raises(OSError, match="controlled durable retirement cut"):
        stack.call(lambda: retention.prune_transition_history(stack.coordinator))
    assert crossed == [cut]
    if cut == "hints":
        monkeypatch.setattr(stack.registry, "forget_operation_hints", original)
    else:
        monkeypatch.setattr(stack.authority, "_unlink_artifact", original)

    def reopen_owned_storage():
        from tldw_chatbook.Plugins.authority_store import (
            FilePluginMarkerStore,
            PluginAuthorityStore,
        )
        from tldw_chatbook.Plugins.coordinator import PluginCoordinator
        from tldw_chatbook.Plugins.registry import PluginRegistry
        from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

        root = stack.owner.root.parent
        stack.registry.close()
        stack.owner.close()
        stack.owner = PluginRuntimeOwner(root / "plugins")
        assert stack.owner.try_acquire()
        stack.registry = PluginRegistry(
            stack.owner.root / "registry.sqlite3", owner=stack.owner
        )
        stack.authority = PluginAuthorityStore(
            root / "trust" / "plugins",
            FilePluginMarkerStore(root / "marker"),
            accept_reduced_protection=True,
        )
        stack.authority.unlock("test passphrase")
        stack.coordinator = PluginCoordinator(
            stack.registry, stack.authority, stack.owner
        )
        return stack.coordinator.recover()

    receipts = stack.call(reopen_owned_storage)
    observed = stack.registry.operation_observed_at
    monkeypatch.setattr(
        stack.registry,
        "operation_observed_at",
        lambda _: datetime.now(UTC) - timedelta(days=31),
    )
    assert all(item.phase == "complete" for item in receipts), receipts
    assert stack.call(stack.authority.load_marker) == current
    assert stack.call(stack.coordinator.published_snapshot) == original_snapshot
    stack.call(lambda: retention.prune_transition_history(stack.coordinator))
    assert not old_snapshot_path.exists()
    assert all(
        item.phase == "complete" for item in stack.call(stack.coordinator.recover)
    )
    assert stack.call(stack.coordinator.published_snapshot) == original_snapshot
    monkeypatch.setattr(stack.registry, "operation_observed_at", observed)


def test_coordinator_pruner_protects_live_aborted_review_and_refuses_only_protected_capacity(
    plugin_stack, native_package, monkeypatch
):
    from Tests.Plugins.test_coordinator import reviewed
    from tldw_chatbook.Plugins import authority_store, retention

    stack = plugin_stack
    aborted = reviewed(stack, native_package())

    def interrupt(phase):
        if phase == "prepared":
            raise OSError("controlled prepared abort")

    stack.coordinator.progress = interrupt
    with pytest.raises(OSError, match="controlled prepared abort"):
        stack.call(lambda: stack.coordinator.commit(aborted, aborted.operation_id))
    stack.coordinator.progress = None
    assert stack.call(stack.coordinator.recover)[0].phase == "aborted"
    monkeypatch.setattr(authority_store, "MAX_TRANSITIONS", 3)
    for _ in range(2):
        review = reviewed(stack, native_package())
        assert stack.call(
            lambda review=review: stack.coordinator.commit(review, review.operation_id)
        ).committed
    before = stack.call(stack.authority.load_marker)
    stack.call(lambda: retention.prune_transition_history(stack.coordinator))
    inventory = stack.call(lambda: retention.transition_inventory(stack.authority))
    assert len(inventory) == 3 and any(
        item.new.operation_id == aborted.operation_id for item in inventory
    )
    blocked = reviewed(stack, native_package())
    with pytest.raises(ValueError, match="capacity"):
        stack.call(lambda: stack.coordinator.commit(blocked, blocked.operation_id))
    assert stack.call(stack.authority.load_marker) == before
    assert len(stack.call(lambda: retention.transition_inventory(stack.authority))) == 3
    assert all(
        item.phase in {"complete", "aborted"}
        for item in stack.call(stack.coordinator.recover)
    )
    original_retry = stack.call(
        lambda: stack.coordinator.commit(aborted, aborted.operation_id)
    )
    assert original_retry.phase == "aborted" and not original_retry.committed
    assert stack.call(stack.authority.load_marker) == before


@pytest.fixture
def orphan_snapshot_case(plugin_stack, native_package):
    from Tests.Plugins.test_coordinator import reviewed
    from tldw_chatbook.Plugins.retention import transition_inventory

    stack = plugin_stack
    bootstrap = stack.call(stack.authority.load_marker)
    for _ in range(3):
        review = reviewed(stack, native_package())
        stack.call(
            lambda review=review: stack.coordinator.commit(review, review.operation_id)
        )
    evidence = sorted(
        stack.call(lambda: transition_inventory(stack.authority)),
        key=lambda item: item.new.generation,
    )
    first = evidence[0]
    stack.call(lambda: stack.registry.forget_operation_hints((first.new.operation_id,)))
    for purpose in ("committed", "prepared"):
        stack.call(
            lambda purpose=purpose: stack.authority._unlink_artifact(
                stack.authority._operation_path(purpose, first.new.operation_id)
            )
        )
    assert all(
        item.phase == "complete" for item in stack.call(stack.coordinator.recover)
    )
    return stack, bootstrap, evidence


def _extra_snapshot(stack, generation, nonce, *, legacy_id=None):
    import copy
    import hashlib

    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    def write():
        snapshot = copy.deepcopy(stack.authority.verify_current())
        result = snapshot["operation_result"]
        result.pop("operation_id")
        nonce_digest = hashlib.sha256(nonce.encode()).hexdigest()
        operation_id = legacy_id or stack.authority.issue_operation_id(
            generation, nonce_digest, result, nonce_digest
        )
        result["operation_id"] = operation_id
        marker = PluginMarker(
            generation=generation,
            operation_id=operation_id,
            recovery_snapshot_digest=snapshot_digest(snapshot),
        )
        stack.authority._save_snapshot(snapshot, marker)
        return marker, snapshot

    return stack.call(write)


def test_orphan_snapshot_cleanup_protects_bootstrap_current_referenced_and_newer(
    orphan_snapshot_case,
):
    stack, bootstrap, evidence = orphan_snapshot_case
    current = stack.call(stack.authority.load_marker)
    newer, _ = _extra_snapshot(stack, current.generation + 1, "future")
    same_generation, _ = _extra_snapshot(stack, current.generation, "same-generation")
    stack.call(
        lambda: stack.authority.retire_orphan_snapshots(
            protected_operation_ids=frozenset({evidence[0].new.operation_id})
        )
    )
    assert stack.authority._snapshot_path(evidence[0].new).exists()
    stack.call(stack.authority.retire_orphan_snapshots)
    assert not stack.authority._snapshot_path(evidence[0].new).exists()
    for marker in (bootstrap, evidence[1].new, current, newer, same_generation):
        assert (
            stack.call(lambda marker=marker: stack.authority.verify_snapshot(marker))
            is not None
        )
    assert stack.call(stack.authority.load_marker) == current


@pytest.mark.parametrize(
    "damage",
    [
        "ciphertext",
        "filename",
        "link",
        "directory",
        "unexpected",
        "unknown_legacy",
        "issued_generation",
        "changed_identity",
        "changed_bytes",
    ],
)
def test_orphan_snapshot_inventory_refuses_unknown_invalid_or_changed_material(
    orphan_snapshot_case, monkeypatch, damage
):
    import json
    import os

    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    stack, _, evidence = orphan_snapshot_case
    marker, snapshot = _extra_snapshot(
        stack,
        1,
        "extra",
        legacy_id="unknown-legacy" if damage == "unknown_legacy" else None,
    )
    path = stack.authority._snapshot_path(marker)
    if damage == "ciphertext":
        value = json.loads(path.read_text())
        value["blob"]["ciphertext"] = "not-encrypted"
        path.write_text(json.dumps(value))
    elif damage == "filename":
        path.rename(path.with_name("f" * 64 + ".json"))
    elif damage == "link":
        path.unlink()
        path.symlink_to(stack.authority._snapshot_path(evidence[1].new))
    elif damage == "directory":
        path.unlink()
        path.mkdir()
    elif damage == "unexpected":
        path.rename(path.with_name("unexpected.json"))
    elif damage == "issued_generation":
        path.unlink()
        wrong = PluginMarker(
            generation=2,
            operation_id=marker.operation_id,
            recovery_snapshot_digest=snapshot_digest(snapshot),
        )
        stack.call(lambda: stack.authority._save_snapshot(snapshot, wrong))
    elif damage in {"changed_identity", "changed_bytes"}:
        original = stack.authority._unlink_artifact

        def replace_before_unlink(candidate, **kwargs):
            if damage == "changed_identity":
                replacement = candidate.with_suffix(".replacement")
                replacement.write_bytes(candidate.read_bytes())
                replacement.chmod(0o600)
                os.replace(replacement, candidate)
            else:
                candidate.write_bytes(candidate.read_bytes() + b" ")
            return original(candidate, **kwargs)

        monkeypatch.setattr(stack.authority, "_unlink_artifact", replace_before_unlink)
    before = stack.call(stack.authority.load_marker)
    with pytest.raises((ValueError, OSError)):
        stack.call(stack.authority.retire_orphan_snapshots)
    assert stack.authority._snapshot_path(evidence[0].new).exists()
    assert stack.call(stack.authority.load_marker) == before


def test_snapshot_capacity_refuses_allocation_and_overflow_but_preserves_exact_retry(
    orphan_snapshot_case, monkeypatch
):
    from tldw_chatbook.Plugins import authority_store

    stack, _, evidence = orphan_snapshot_case
    monkeypatch.setattr(authority_store, "MAX_TRANSITIONS", 3)
    for index in range(4):
        marker, snapshot = _extra_snapshot(stack, 4, f"future-{index}")
    assert len(list((stack.authority.store_dir / "snapshots").iterdir())) == 8
    stack.call(lambda: stack.authority._save_snapshot(snapshot, marker))
    with pytest.raises(ValueError, match="capacity"):
        _extra_snapshot(stack, 4, "over-capacity")
    extra = stack.authority.store_dir / "snapshots" / ("f" * 64 + ".json")
    extra.write_bytes(stack.authority._snapshot_path(marker).read_bytes())
    extra.chmod(0o600)
    with pytest.raises(ValueError, match="excessive snapshot"):
        stack.call(stack.authority.retire_orphan_snapshots)
    assert stack.authority._snapshot_path(evidence[0].new).exists()
    extra.unlink()
    stack.call(stack.authority.retire_orphan_snapshots)
    assert not stack.authority._snapshot_path(evidence[0].new).exists()


def test_orphan_snapshot_fixed_legacy_membership_is_exact(plugin_stack, native_package):
    from Tests.Plugins.test_coordinator import reviewed
    from Tests.Plugins.test_recovery import (
        _legacy_fixture_commit,
        _legacy_fixture_namespace,
    )

    stack = plugin_stack
    _legacy_fixture_namespace(stack)
    legacy = _legacy_fixture_commit(stack, native_package(), "legacy-one")
    assert all(
        item.phase == "complete" for item in stack.call(stack.coordinator.recover)
    )
    fixed = stack.call(stack.authority.verify_legacy_cutover)
    review = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    stack.call(
        lambda: stack.registry.forget_operation_hints((legacy.new.operation_id,))
    )
    for purpose in ("committed", "prepared"):
        stack.call(
            lambda purpose=purpose: stack.authority._unlink_artifact(
                stack.authority._operation_path(purpose, legacy.new.operation_id)
            )
        )
    assert all(
        item.phase == "complete" for item in stack.call(stack.coordinator.recover)
    )
    stack.call(stack.authority.retire_orphan_snapshots)
    assert not stack.authority._snapshot_path(legacy.new).exists()
    marker, _ = _extra_snapshot(stack, 1, "mismatched-legacy", legacy_id="legacy-one")
    with pytest.raises(ValueError, match="unknown legacy"):
        stack.call(stack.authority.retire_orphan_snapshots)
    assert stack.authority._snapshot_path(marker).exists()
    assert stack.call(stack.authority.verify_legacy_cutover) == fixed
