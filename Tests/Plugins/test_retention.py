"""Retention selects only eligible immutable material, never protected authority."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest


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
@pytest.mark.parametrize("replace_root", [False, True])
async def test_retention_restart_uses_original_authenticated_cleanup_identity(
    native_console, native_package, monkeypatch, replace_root
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
    if replace_root:
        path.rename(path.with_name(path.name + "-original"))
        path.mkdir()
        (path / "new-owner").write_text("must survive")
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
        assert pending.committed and pending.cleanup_pending
        assert path.exists()  # read-only lookup never performs cleanup
        result = await restarted.retain_revisions(installed.installation_id)
        assert result.operation_id == evidence.new.operation_id
        assert result.committed and result.cleanup_pending == replace_root
        assert path.exists() == replace_root
        if replace_root:
            assert (path / "new-owner").read_text() == "must survive"
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
