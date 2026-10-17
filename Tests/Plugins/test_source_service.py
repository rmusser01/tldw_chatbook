"""Source discovery and reviewed authority through the actual plugin worker."""

import json
import shutil
from pathlib import Path

import pytest


async def test_update_candidate_cannot_be_submitted_for_a_different_installation(
    source_stack, native_package
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    roots = [native_package(), native_package()]
    installed = []
    for index, root in enumerate(roots):
        operation = f"target-{index}"
        snapshot = await service.acquire_snapshot(
            select_local_source(root),
            operation,
            await service.acquisition_destination(operation),
        )
        review = await sources.review_acquired(
            snapshot, selection=("skill:review",), workspace_id=None
        )
        assert (await service.commit(review, review.operation_id)).committed
        installed.append(review.installation_id)
    skill = roots[0] / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nTargeted update.\n")
    candidate = (await sources.check_updates(installed[0]))[0]
    before = await service._call(service._coordinator.published_snapshot)
    with pytest.raises(ValueError, match="candidate_review_required"):
        await sources.review_update(installed[1], candidate)
    assert await service._call(service._coordinator.published_snapshot) == before
    update = await sources.review_update(installed[0], candidate)
    assert (await service.apply_revision(update, update.operation_id)).committed


@pytest.mark.parametrize("phase", ["quota", "size"])
async def test_refresh_tolerates_actually_vanished_owned_transient_file(
    source_stack, marketplace, monkeypatch, phase
):
    import inspect

    from tldw_chatbook.Plugins import acquisition
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    sources = SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    assert (await sources.refresh(identity)).refresh_error is None
    original_quota = acquisition._Producer.quota
    original_lstat = Path.lstat
    observed = {"path": None, "removed": False}

    def with_transient(producer, *args, **kwargs):
        if observed["path"] is None:
            observed["path"] = producer.staging / "owned-transient"
            observed["path"].write_bytes(b"temporary producer bytes")
        return original_quota(producer, *args, **kwargs)

    def remove_after_enumeration(path, *args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        if path == observed["path"] and caller == phase and not observed["removed"]:
            path.unlink()
            observed["removed"] = True
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(acquisition._Producer, "quota", with_transient)
    monkeypatch.setattr(Path, "lstat", remove_after_enumeration)
    result = await sources.refresh(identity)
    assert observed["removed"]
    assert result.refresh_error is None


@pytest.mark.parametrize(
    "fault", ["permission", "special", "link", "missing_leaf", "unrelated", "final"]
)
async def test_refresh_transient_exception_keeps_other_storage_refusals(
    source_stack, marketplace, monkeypatch, fault
):
    import inspect
    import os

    from tldw_chatbook.Plugins import acquisition
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    sources = SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    good = await sources.refresh(identity)
    assert good.refresh_error is None
    original_quota, original_lstat = acquisition._Producer.quota, Path.lstat
    observed = {"path": None, "fault": False}

    def with_fault(producer, *args, **kwargs):
        if fault == "final" and not kwargs.get("force", False):
            return original_quota(producer, *args, **kwargs)
        if observed["path"] is None:
            if fault == "missing_leaf":
                hidden = producer.staging.with_name(
                    producer.staging.name + "-test-held"
                )
                producer.staging.rename(hidden)
                observed["fault"] = True
                try:
                    return original_quota(producer, *args, **kwargs)
                finally:
                    hidden.rename(producer.staging)
            parent = (
                producer.owner.root / "cache"
                if fault == "unrelated"
                else producer.staging
            )
            parent.mkdir(exist_ok=True)
            observed["path"] = parent / "owned-fault"
            if fault == "special":
                os.mkfifo(observed["path"])
                observed["fault"] = True
            elif fault == "link":
                observed["path"].symlink_to(marketplace)
                observed["fault"] = True
            else:
                observed["path"].write_bytes(b"owned test bytes")
        return original_quota(producer, *args, **kwargs)

    def fault_after_enumeration(path, *args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        expected = "size" if fault == "unrelated" else "quota"
        if path == observed["path"] and caller == expected and not observed["fault"]:
            observed["fault"] = True
            if fault == "permission":
                raise PermissionError(13, "owned test refusal")
            path.unlink()
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(acquisition._Producer, "quota", with_fault)
    monkeypatch.setattr(Path, "lstat", fault_after_enumeration)
    result = await sources.refresh(identity)
    assert observed["fault"]
    assert result.refresh_error == "catalog_refresh_failed"
    assert result.digest == good.digest and result.entries == good.entries


@pytest.fixture(autouse=True)
def private_host_auth(monkeypatch, tmp_path):
    import os

    from tldw_chatbook.Plugins import acquisition

    monkeypatch.setattr(acquisition, "HOST_GIT_SYSTEM", tmp_path / "host-git-system")
    monkeypatch.setattr(acquisition, "HOST_SSH_SYSTEM", tmp_path / "host-ssh-system")
    for key in tuple(os.environ):
        if key.startswith(("GIT_", "SSH_")) or "proxy" in key.lower():
            monkeypatch.delenv(key)


@pytest.fixture
async def source_stack(tmp_path):
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService

    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    await service.bootstrap("test passphrase")
    try:
        yield service
    finally:
        await service.aclose()


@pytest.fixture
def marketplace(tmp_path, native_package):
    root = tmp_path / "marketplace"
    root.mkdir()
    shutil.copytree(native_package(), root / "plugins/review")
    path = root / ".agents/plugins/marketplace.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "plugins": [
                    {
                        "name": "review",
                        "source": {"source": "local", "path": "./plugins/review"},
                    }
                ]
            }
        )
    )
    return root


async def test_empty_profile_refresh_and_review_are_separate(source_stack, marketplace):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    destination = await service.acquisition_destination("plain-catalog-control")
    with pytest.raises(ValueError, match="package_inspection_rejected"):
        await service.acquire_snapshot(
            select_local_source(marketplace), "plain-catalog-control", destination
        )
    identity = await sources.add(select_local_source(marketplace))
    summary = sources.list_sources()[0]
    assert summary.source_id == identity and not summary.cached
    assert summary.locator == select_local_source(marketplace)
    assert summary.refresh_error == "not_refreshed"
    result = await sources.refresh(identity)
    assert result.entries[0].name == "review"
    assert sources.search("review", offset=0) == result.entries
    assert (
        await service._call(
            lambda: service._coordinator.registry.list_installations(limit=50, offset=0)
        )
        == ()
    )
    review = await sources.review_entry(
        identity,
        result.entries[0].identity,
        selection=("skill:review",),
        workspace_id=None,
    )
    receipt = await service.commit(review, review.operation_id)
    assert receipt.committed
    state = await service._call(service._coordinator.published_snapshot)
    assert not state["installations"][0]["activation_default"]
    assert state["revisions"][0]["catalog_binding"]["source_id"] == identity
    await sources.remove(identity)
    after = await service._call(service._coordinator.published_snapshot)
    assert state == after
    with pytest.raises(ValueError, match="source_removed"):
        await sources.check_updates(review.installation_id)


async def test_failed_refresh_keeps_last_cache_and_bounds_pages(
    source_stack, marketplace
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    sources = SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    good = await sources.refresh(identity)
    (marketplace / ".agents/plugins/marketplace.json").write_text("broken")
    stale = await sources.refresh(identity)
    assert stale.entries == good.entries and stale.digest == good.digest
    assert stale.refresh_error == "catalog_refresh_failed"
    assert sources.list_sources()[0].refresh_error == "catalog_refresh_failed"
    assert sources.search("review", offset=0) == good.entries
    for offset, limit in ((-1, 50), (0, 51), (True, 50)):
        with pytest.raises(ValueError):
            sources.search("", offset=offset, limit=limit)


async def test_source_tamper_and_cache_spoof_never_select_another_origin(
    source_stack, marketplace
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    good = await sources.refresh(identity)

    def damage():
        registry = service._coordinator.registry
        row = registry.list_marketplace_sources()[0]
        body = json.loads(row["binding_json"])
        body["locator"]["repository"] = "/private/unselected"
        with registry.transaction() as cursor:
            cursor.execute(
                "UPDATE marketplace_sources SET binding_json=?", (json.dumps(body),)
            )
        return row

    original = await service._call(damage)
    with pytest.raises(ValueError, match="authentication"):
        await sources.refresh(identity)
    await service._call(
        lambda: service._coordinator.registry.write_marketplace_source(
            **{
                key: original[key]
                for key in (
                    "source_id",
                    "binding_json",
                    "binding_mac",
                    "snapshot_json",
                    "refresh_error",
                )
            }
        )
    )
    assert (await sources.refresh(identity)).entries == good.entries


async def test_producer_result_substitution_is_rejected_at_public_review(
    source_stack, marketplace, monkeypatch
):
    from tldw_chatbook.Plugins import source_service
    from tldw_chatbook.Plugins.git_source import select_local_source

    service, sources = source_stack, source_service.SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    entry = (await sources.refresh(identity)).entries[0]
    original = source_service.acquire_snapshot

    async def forged(*args, **kwargs):
        result = await original(*args, **kwargs)
        return result.model_copy(update={"origin": "https://unselected.example"})

    monkeypatch.setattr(source_service, "acquire_snapshot", forged)
    with pytest.raises(ValueError, match="acquisition_result"):
        await sources.review_entry(
            identity, entry.identity, selection=("skill:review",), workspace_id=None
        )
    monkeypatch.setattr(source_service, "acquire_snapshot", original)
    review = await sources.review_entry(
        identity, entry.identity, selection=("skill:review",), workspace_id=None
    )
    assert (await service.commit(review, review.operation_id)).committed


async def test_catalog_recovery_uses_retained_artifacts_after_source_removal_and_boot_change(
    source_stack, marketplace, monkeypatch
):
    from tldw_chatbook.Plugins import acquisition
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    entry = (await sources.refresh(identity)).entries[0]
    old_cache = await service._call(
        lambda: sources._binding(sources._row(identity)).current
    )
    await sources.refresh(identity)
    review = await sources.review_entry(
        identity, entry.identity, selection=("skill:review",), workspace_id=None
    )
    await service.commit(review, review.operation_id)
    await sources.remove(identity)
    shutil.rmtree(marketplace)
    monkeypatch.setattr(
        acquisition, "boot_identity", lambda: "changed-boot-for-readonly-test"
    )
    with pytest.raises(ValueError, match="owner_changed"):
        await sources.cleanup()
    assert Path(old_cache.materialized_identity).exists()

    def lose_projection():
        with service._coordinator.registry.transaction() as cursor:
            cursor.execute("DELETE FROM selections")

    await service._call(lose_projection)
    receipts = await service._call(service._coordinator.recover)
    assert receipts and all(receipt.committed for receipt in receipts)
    state = await service._call(service._coordinator.published_snapshot)
    binding = state["revisions"][0]["catalog_binding"]
    artifact = (
        Path(binding["catalog"]["materialized_identity"]) / binding["catalog_path"]
    )
    artifact.chmod(0o600)
    artifact.write_text('{"plugins":[]}')
    receipts = await service._call(service._coordinator.recover)
    assert receipts[0].phase == "recovery_required"


async def test_cache_retirement_protects_installed_and_ordinary_candidates(
    source_stack, marketplace, native_package
):
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Plugins.acquisition import cleanup_acquisitions
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    ordinary = await service.acquire_snapshot(
        select_local_source(native_package()),
        "ordinary-held",
        await service.acquisition_destination("ordinary-held"),
    )
    identity = await sources.add(select_local_source(marketplace))
    entry = (await sources.refresh(identity)).entries[0]
    old = await service._call(lambda: sources._binding(sources._row(identity)).current)
    await sources.refresh(identity)
    await sources.cleanup()
    assert Path(old.materialized_identity).exists()
    removed = await service._call(
        lambda: cleanup_acquisitions(
            service._coordinator.owner,
            service._coordinator.authority,
            now=datetime.now(UTC) + timedelta(hours=25),
        )
    )
    assert old.operation_id in removed
    assert Path(ordinary.materialized_identity).exists()
    review = await sources.review_entry(
        identity, entry.identity, selection=("skill:review",), workspace_id=None
    )
    await service.commit(review, review.operation_id)
    current = review.inspection.catalog_binding.catalog
    await sources.remove(identity)
    await sources.cleanup()
    await service._call(
        lambda: cleanup_acquisitions(
            service._coordinator.owner,
            service._coordinator.authority,
            now=datetime.now(UTC) + timedelta(hours=25),
        )
    )
    assert Path(current.materialized_identity).exists()


async def test_reviewed_update_preserves_origin_and_rejects_catalog_redirect(
    source_stack, marketplace
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    entry = (await sources.refresh(identity)).entries[0]
    review = await sources.review_entry(
        identity, entry.identity, selection=("skill:review",), workspace_id=None
    )
    await service.commit(review, review.operation_id)
    file = marketplace / "plugins/review/skills/review/SKILL.md"
    file.write_text(file.read_text() + "\nUpdated checklist.\n")
    candidate = (await sources.check_updates(review.installation_id))[0]
    before = await service._call(service._coordinator.published_snapshot)
    assert (
        before["installations"][0]["revision_digest"]
        == review.inspection.effective_digest
    )
    assert candidate.effective_digest != review.inspection.effective_digest
    update = await sources.review_update(review.installation_id, candidate)
    assert update.kind == "update"
    assert (await service.apply_revision(update, update.operation_id)).committed
    (marketplace / ".agents/plugins/marketplace.json").write_text(
        json.dumps(
            {
                "plugins": [
                    {
                        "name": "review",
                        "source": {
                            "source": "url",
                            "url": "https://unselected.example/repo.git",
                        },
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="update_origin_changed"):
        await sources.check_updates(review.installation_id)


async def test_import_commit_refuses_mutated_preview_then_copies_only_reviewed_catalog(
    source_stack, marketplace
):
    from tldw_chatbook.Plugins.imports import inspect_app_import
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    (marketplace / "auth.json").write_text('{"secret":"private-token"}')
    preview = inspect_app_import("codex", marketplace)
    file = marketplace / "plugins/review/skills/review/SKILL.md"
    file.write_text(file.read_text() + "\nChanged after preview.\n")
    with pytest.raises(ValueError, match="preview_changed"):
        await sources.commit_import(preview)
    result = await sources.commit_import(inspect_app_import("codex", marketplace))
    assert len(result.source_ids) == 1
    snapshot = await sources.refresh(result.source_ids[0])
    assert snapshot.entries[0].name == "review"
    binding = await service._call(
        lambda: sources._binding(sources._row(result.source_ids[0]))
    )
    copied = Path(binding.locator.repository)
    assert not (copied / "auth.json").exists()
    assert not (await service._call(service._coordinator.published_snapshot))[
        "installations"
    ]


async def test_catalog_plugin_admission_rechecks_retained_catalog_bytes(
    source_stack, marketplace
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    entry = (await sources.refresh(identity)).entries[0]
    review = await sources.review_entry(
        identity, entry.identity, selection=("skill:review",), workspace_id=None
    )
    await service.commit(review, review.operation_id)
    trust = await service.review_trust(review.installation_id)
    await service.commit(trust, trust.operation_id)
    active = await service.review_activation(
        review.installation_id, workspace_id=None, intent="enabled"
    )
    await service.commit(active, active.operation_id)
    maximum = service.capture_maximum(None)
    assert len(maximum["available_skills"]) == 1
    admitted = await service.admit(maximum, "catalog-run")
    assert len(admitted["available_skills"]) == 1
    binding = review.inspection.catalog_binding
    artifact = Path(binding.catalog.materialized_identity) / binding.catalog_path
    artifact.chmod(0o600)
    artifact.write_text('{"plugins":[]}')
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    with pytest.raises(PluginUnavailable):
        await service.admit(maximum, "tampered-catalog-run")


async def test_direct_git_acquired_review_update_and_recovery_keep_original_origin(
    source_stack, native_package, tmp_path
):
    from Tests.Plugins.acquisition_process_support import git, repository
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    root = repository(native_package(), tmp_path / "direct-git")
    locator = select_local_source(root)
    acquired = await service.acquire_snapshot(
        locator, "direct", await service.acquisition_destination("direct")
    )
    with pytest.raises(ValueError, match="acquisition_result"):
        await sources.review_acquired(
            acquired.model_copy(update={"commit": "a" * 40}),
            selection=("skill:review",),
            workspace_id=None,
        )
    review = await sources.review_acquired(
        acquired, selection=("skill:review",), workspace_id=None
    )
    assert (await service.commit(review, review.operation_id)).committed
    file = root / "skills/review/SKILL.md"
    file.write_text(file.read_text() + "\nDirect update.\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "update fixture")
    candidate = (await sources.check_updates(review.installation_id))[0]
    assert candidate.acquisition_reference.repository == str(root)
    assert candidate.acquisition_reference.commit != acquired.commit
    update = await sources.review_update(review.installation_id, candidate)
    assert (await service.apply_revision(update, update.operation_id)).committed

    def lose_projection():
        with service._coordinator.registry.transaction() as cursor:
            cursor.execute("DELETE FROM selections")

    await service._call(lose_projection)
    assert all(
        row.committed for row in await service._call(service._coordinator.recover)
    )
    state = await service._call(service._coordinator.published_snapshot)
    assert all("catalog_binding" not in row for row in state["revisions"])
    assert any(
        row["acquisition_reference"]["commit"] == candidate.acquisition_reference.commit
        for row in state["revisions"]
    )
    await service.aclose()
    registry = tmp_path / "profile/plugins/registry.sqlite3"
    for member in (
        registry,
        registry.with_name(registry.name + "-wal"),
        registry.with_name(registry.name + "-shm"),
    ):
        member.unlink(missing_ok=True)
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService

    reopened = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    try:
        await reopened.unlock("test passphrase")
        assert (await reopened._call(reopened._coordinator.published_snapshot))[
            "revisions"
        ] == state["revisions"]
        altered = (
            Path(candidate.acquisition_reference.materialized_identity)
            / "skills/review/SKILL.md"
        )
        altered.chmod(0o600)
        altered.write_text("changed retained direct bytes")
        assert (await reopened._call(reopened._coordinator.recover))[
            0
        ].phase == "recovery_required"
    finally:
        await reopened.aclose()


async def test_locked_cache_is_display_only_and_tampered_oversized_rows_are_bounded(
    source_stack, marketplace, monkeypatch
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    cached = await sources.refresh(identity)
    await service._call(service._coordinator.authority.lock)
    assert (await SourceService(service).load_cached())[0].entries == cached.entries
    with pytest.raises(ValueError):
        await sources.refresh(identity)
    with monkeypatch.context() as locked:

        def forbidden(*args, **kwargs):
            raise AssertionError("display attempted authentication or source stat")

        locked.setattr(
            service._coordinator.authority, "verify_marketplace_source", forbidden
        )
        locked.setattr(
            "tldw_chatbook.Plugins.source_service._validate_locator", forbidden
        )
        display = SourceService(service)
        await display.load_cached()
        assert display.list_sources()[0].source_id == identity
    await service.unlock("test passphrase")

    def damage_cache():
        with service._coordinator.registry.transaction() as cursor:
            cursor.execute(
                "UPDATE marketplace_sources SET snapshot_json=?",
                ("x" * (5 * 1024 * 1024 + 1),),
            )

    await service._call(damage_cache)
    assert await sources.load_cached() == ()
    assert sources.list_sources()[0].refresh_error == "cache_unavailable"
    assert (await sources.refresh(identity)).entries == cached.entries


async def test_first_failed_refresh_has_no_successful_cache_but_empty_catalog_does(
    source_stack, marketplace
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    sources = SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    catalog = marketplace / ".agents/plugins/marketplace.json"
    catalog.write_text("invalid")
    await sources.refresh(identity)
    summary = sources.list_sources()[0]
    assert not summary.cached and summary.fetched_at is None
    assert summary.refresh_error == "catalog_refresh_failed"
    catalog.write_text('{"plugins":[]}')
    await sources.refresh(identity)
    summary = sources.list_sources()[0]
    assert summary.cached and summary.fetched_at and summary.refresh_error is None


async def test_source_limit_is_independent_of_installations(source_stack, marketplace):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    sources = SourceService(source_stack)
    for _ in range(50):
        await sources.add(select_local_source(marketplace))
    with pytest.raises(ValueError, match="sources_limit"):
        await sources.add(select_local_source(marketplace))
    assert not (await source_stack._call(source_stack._coordinator.published_snapshot))[
        "installations"
    ]


async def test_selected_remote_entry_requires_exact_displayed_origin_before_network(
    source_stack, marketplace, native_package, tmp_path
):
    import os
    import shlex

    from Tests.hooks_v2_process_support import child_argv
    from Tests.Plugins.acquisition_process_support import https_git, repository
    from tldw_chatbook.Plugins.git_source import (
        parse_source_locator,
        select_local_source,
    )
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    repository(native_package(), tmp_path / "remote-package")
    with https_git(tmp_path) as (origin, ca, requests):
        helper = "!" + shlex.join(
            child_argv(
                "print('username=fixture\\npassword=private-credential-sentinel')"
            )
        )
        (Path(os.environ["HOME"]) / ".gitconfig").write_text(
            "[credential]\n helper = "
            + json.dumps(helper)
            + "\n[http]\n sslCAInfo = "
            + json.dumps(str(ca))
            + "\n"
        )
        locator = parse_source_locator(
            origin + "/remote-package/.git#ref=refs/heads/main"
        )
        path = marketplace / ".agents/plugins/marketplace.json"
        path.write_text(
            json.dumps(
                {
                    "plugins": [
                        {
                            "name": "review",
                            "source": {
                                "source": "url",
                                "url": locator.url,
                                "ref": locator.ref,
                            },
                        }
                    ]
                }
            )
        )
        identity = await sources.add(select_local_source(marketplace))
        entry = (await sources.refresh(identity)).entries[0]
        for selected in (
            None,
            parse_source_locator("https://unselected.invalid/repo.git"),
        ):
            with pytest.raises(ValueError, match="origin_review_required"):
                await sources.review_entry(
                    identity,
                    entry.identity,
                    selection=("skill:review",),
                    workspace_id=None,
                    selected_locator=selected,
                )
        assert requests == []
        review = await sources.review_entry(
            identity,
            entry.identity,
            selection=("skill:review",),
            workspace_id=None,
            selected_locator=locator,
        )
        assert (
            requests and (await service.commit(review, review.operation_id)).committed
        )
        assert review.inspection.catalog_binding.package.origin == origin


async def test_imported_package_review_and_discard_keep_copied_local_provenance(
    source_stack, native_package
):
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Plugins.acquisition import cleanup_acquisitions
    from tldw_chatbook.Plugins.imports import inspect_app_import
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    root = native_package()
    imported = await sources.commit_import(inspect_app_import("codex", root))
    package = imported.packages[0]
    assert package.repository != str(root) and package.commit is None
    review = await sources.review_acquired(
        package, selection=("skill:review",), workspace_id=None
    )
    await sources.discard_candidate(package)
    with pytest.raises(ValueError):
        await service.commit(review, review.operation_id)
    removed = await service._call(
        lambda: cleanup_acquisitions(
            service._coordinator.owner,
            service._coordinator.authority,
            now=datetime.now(UTC) + timedelta(hours=25),
        )
    )
    assert package.operation_id in removed
    imported = await sources.commit_import(inspect_app_import("codex", root))
    package = imported.packages[0]
    review = await sources.review_acquired(
        package, selection=("skill:review",), workspace_id=None
    )
    await service.commit(review, review.operation_id)
    await sources.discard_candidate(package)
    await service._call(
        lambda: cleanup_acquisitions(
            service._coordinator.owner,
            service._coordinator.authority,
            now=datetime.now(UTC) + timedelta(hours=25),
        )
    )
    assert Path(package.materialized_identity).exists()


async def test_same_byte_rebind_is_explicit_stable_and_recovers_after_registry_loss(
    source_stack, marketplace, tmp_path
):
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    first = await sources.add(select_local_source(marketplace))
    entry = (await sources.refresh(first)).entries[0]
    review = await sources.review_entry(
        first, entry.identity, selection=("skill:review",), workspace_id=None
    )
    await service.commit(review, review.operation_id)
    same = (await sources.check_updates(review.installation_id))[0]
    assert same.effective_digest == review.inspection.effective_digest
    with pytest.raises(ValueError, match="already current"):
        await sources.review_update(review.installation_id, same)
    await sources.discard_candidate(same)
    await sources.remove(first)
    with pytest.raises(ValueError, match="source_removed"):
        await sources.check_updates(review.installation_id)
    second = await sources.add(select_local_source(marketplace))
    replacement = (await sources.refresh(second)).entries[0]
    before = await service._call(service._coordinator.published_snapshot)
    rebound = await sources.review_rebind(
        review.installation_id, second, replacement.identity
    )
    assert rebound.inspection.content_digest == review.inspection.content_digest
    assert rebound.inspection.effective_digest != review.inspection.effective_digest
    assert await service._call(service._coordinator.published_snapshot) == before
    assert (await service.apply_revision(rebound, rebound.operation_id)).committed
    candidate = (await sources.check_updates(review.installation_id))[0]
    assert candidate.catalog_binding.source_id == second
    await sources.discard_candidate(candidate)
    expected = await service._call(service._coordinator.published_snapshot)
    assert {row["catalog_binding"]["source_id"] for row in expected["revisions"]} == {
        first,
        second,
    }
    await service.aclose()
    registry = tmp_path / "profile/plugins/registry.sqlite3"
    for member in (
        registry,
        registry.with_name(registry.name + "-wal"),
        registry.with_name(registry.name + "-shm"),
    ):
        member.unlink(missing_ok=True)
    reopened = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    try:
        await reopened.unlock("test passphrase")
        recovered = await reopened._call(reopened._coordinator.published_snapshot)
        assert recovered["revisions"] == expected["revisions"]
        assert await SourceService(reopened).load_cached() == ()
        with pytest.raises(ValueError, match="source_removed"):
            await SourceService(reopened).check_updates(review.installation_id)
    finally:
        await reopened.aclose()


async def test_cancelled_refresh_retains_actual_materializer_and_does_not_publish_cache(
    source_stack, marketplace, monkeypatch
):
    import asyncio
    import threading

    from tldw_chatbook.Plugins import package_files
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    original = package_files.materialize_package
    entered, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(package_files, "materialize_package", blocked)
    task = asyncio.create_task(sources.refresh(identity))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service._coordinator.owner.acquisition_tasks
        with pytest.raises(ValueError, match="dependencies_unsettled"):
            await sources.cleanup()
        assert sources.search("", offset=0) == ()
    finally:
        release.set()
        for _ in range(500):
            if not service._coordinator.owner.acquisition_tasks:
                break
            await asyncio.sleep(0.01)
    assert not service._coordinator.owner.acquisition_tasks
    monkeypatch.setattr(package_files, "materialize_package", original)
    assert (await sources.refresh(identity)).entries


async def test_import_partial_failure_is_per_item_but_authority_failure_is_terminal(
    source_stack, marketplace
):
    from tldw_chatbook.Plugins.imports import inspect_app_import
    from tldw_chatbook.Plugins.source_service import SourceService

    broken = marketplace / ".claude-plugin/marketplace.json"
    broken.parent.mkdir()
    broken.write_text("malformed selected catalog")
    preview = inspect_app_import("codex", marketplace)
    assert len(preview.candidates) == len(preview.failures) == 1
    assert preview.failures[0].path == ".claude-plugin/marketplace.json"
    sources = SourceService(source_stack)
    await source_stack._call(source_stack._coordinator.authority.lock)
    with pytest.raises(ValueError):
        await sources.commit_import(preview)
    await source_stack.unlock("test passphrase")
    result = await sources.commit_import(preview)
    assert len(result.source_ids) == 1 and result.failures == preview.failures
    assert not (await source_stack._call(source_stack._coordinator.published_snapshot))[
        "installations"
    ]


async def test_import_base_dependency_survives_restart_until_explicit_discard(
    source_stack, native_package, tmp_path
):
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Plugins.acquisition import cleanup_acquisitions
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.imports import inspect_app_import
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Plugins.source_service import SourceService

    sources = SourceService(source_stack)
    package = (
        await sources.commit_import(inspect_app_import("codex", native_package()))
    ).packages[0]
    base = Path(package.repository)
    await source_stack.aclose()
    reopened = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    try:
        await reopened.unlock("test passphrase")
        sources = SourceService(reopened)
        await sources.cleanup()
        assert base.exists()
        unknown = tmp_path / "profile/plugins/staging/unknown"
        unknown.mkdir()
        with pytest.raises(ValueError, match="dependencies_unknown"):
            await sources.cleanup()
        assert base.exists() and unknown.exists()
        unknown.rmdir()
        await sources.discard_candidate(package)
        await sources.cleanup()
        await reopened._call(
            lambda: cleanup_acquisitions(
                reopened._coordinator.owner,
                reopened._coordinator.authority,
                now=datetime.now(UTC) + timedelta(hours=25),
            )
        )
        assert not base.exists() and not Path(package.materialized_identity).exists()
    finally:
        await reopened.aclose()


@pytest.mark.parametrize(
    "dialect,manifest",
    [("cursor", ".cursor-plugin/plugin.json"), ("openai", ".codex-plugin/plugin.json")],
)
async def test_catalog_executable_overlay_is_retained_for_review_update_and_recovery(
    source_stack, marketplace, dialect, manifest
):
    from Tests.Plugins.test_codex_adapter import write_json, write_skill
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    package = marketplace / "plugins/review"
    shutil.rmtree(package)
    write_json(package, manifest, {"name": "review"})
    write_skill(package, "alternate/example/SKILL.md")
    catalog = marketplace / ".agents/plugins/marketplace.json"
    catalog.write_text(
        json.dumps(
            {
                "plugins": [
                    {
                        "name": "review",
                        "source": "./plugins/review",
                        "skills": "./alternate",
                    }
                ]
            }
        )
    )
    service, sources = source_stack, SourceService(source_stack)
    identity = await sources.add(select_local_source(marketplace))
    entry = (await sources.refresh(identity)).entries[0]
    review = await sources.review_entry(
        identity, entry.identity, selection=("skill:example",), workspace_id=None
    )
    assert review.inspection.dialect == dialect
    assert set(review.inspection.inventory) == {"skill:example"}
    assert (await service.commit(review, review.operation_id)).committed
    catalog.write_text(
        json.dumps(
            {
                "plugins": [
                    {"name": "review", "source": "./plugins/review", "skills": []}
                ]
            }
        )
    )
    candidate = (await sources.check_updates(review.installation_id))[0]
    assert not candidate.inventory
    with pytest.raises(ValueError, match="candidate_review_required"):
        await sources.review_update(
            review.installation_id,
            candidate.model_copy(update={"overlay_identities": ()}),
        )
    update = await sources.review_update(review.installation_id, candidate)
    assert (await service.apply_revision(update, update.operation_id)).committed
    state = await service._call(service._coordinator.published_snapshot)

    def lose_projection():
        with service._coordinator.registry.transaction() as cursor:
            cursor.execute("DELETE FROM selections")

    await service._call(lose_projection)
    receipts = await service._call(service._coordinator.recover)
    assert receipts and all(receipt.committed for receipt in receipts)
    assert (await service._call(service._coordinator.published_snapshot))[
        "revisions"
    ] == state["revisions"]


@pytest.mark.parametrize("outcome", ["success", "failure"])
@pytest.mark.parametrize("change", ["remove", "supersede"])
async def test_refresh_publication_refuses_removed_or_superseded_source(
    source_stack, marketplace, monkeypatch, outcome, change
):
    import asyncio
    import threading
    from datetime import UTC, datetime, timedelta

    from tldw_chatbook.Plugins.acquisition import cleanup_acquisitions
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    sources = SourceService(source_stack)
    locator = select_local_source(marketplace)
    identity = await sources.add(locator)
    assert (await sources.refresh(identity)).digest
    original = sources._acquire
    entered, release = threading.Event(), threading.Event()
    captured = []
    first = True
    hidden = marketplace.with_name("held-marketplace")
    if outcome == "failure":
        marketplace.rename(hidden)

    async def held(*args, **kwargs):
        nonlocal first
        if not first:
            return await original(*args, **kwargs)
        first = False
        try:
            result = await original(*args, **kwargs)
            captured.append(result[0])
        except (ValueError, OSError):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            raise
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return result

    monkeypatch.setattr(sources, "_acquire", held)
    pending = asyncio.create_task(sources.refresh(identity))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if outcome == "failure":
            hidden.rename(marketplace)
        if change == "remove":
            await sources.remove(identity)
            assert sources.list_sources() == ()
            replacement = await sources.add(locator)
            assert replacement != identity
            assert (await sources.refresh(replacement)).digest
        else:
            assert (await sources.refresh(identity)).digest
        expected = sources.list_sources()
        rows = await source_stack._call(sources._rows)
    finally:
        release.set()
        if hidden.exists():
            hidden.rename(marketplace)
    with pytest.raises(ValueError, match="source_changed"):
        await pending
    assert bool(captured) == (outcome == "success")
    assert sources.list_sources() == expected
    assert await source_stack._call(sources._rows) == rows
    await source_stack._call(
        lambda: cleanup_acquisitions(
            source_stack._coordinator.owner,
            source_stack._coordinator.authority,
            now=datetime.now(UTC) + timedelta(hours=25),
        )
    )
    assert all(not Path(item.materialized_identity).exists() for item in captured)


@pytest.mark.parametrize("mutation", ["content", "excluded", "catalog"])
async def test_import_refuses_changed_then_restored_captured_bytes(
    source_stack, marketplace, monkeypatch, mutation
):
    from tldw_chatbook.Plugins import package_files, source_service
    from tldw_chatbook.Plugins.imports import inspect_app_import
    from tldw_chatbook.Plugins.source_service import SourceService

    sources = SourceService(source_stack)
    preview = inspect_app_import("codex", marketplace)
    control = await sources.commit_import(preview)
    assert len(control.source_ids) == 1
    before = sources.list_sources()
    staging = source_stack._coordinator.owner.root / "staging"
    retained = set(staging.iterdir())
    target = marketplace / "plugins/review"
    skill = target / "skills/review/SKILL.md"
    approved = skill.read_bytes()
    original = package_files.capture_package
    changed = []

    def altered_capture(root, *args, **kwargs):
        if Path(root) != target:
            return original(root, *args, **kwargs)
        excluded = target / "auth.json"
        if mutation == "excluded":
            excluded.write_bytes(b"owned excluded credential sentinel")
        else:
            skill.write_bytes(approved + b"\nUNREVIEWED-CAPTURE-SENTINEL\n")
        try:
            result = original(root, *args, **kwargs)
            changed.append(result.digest)
            return result
        finally:
            skill.write_bytes(approved)
            excluded.unlink(missing_ok=True)

    original_read = source_service.read_selected_file

    def changed_catalog(root, path, **kwargs):
        if root != marketplace:
            return original_read(root, path, **kwargs)
        catalog = root / path
        approved_catalog = catalog.read_bytes()
        document = json.loads(approved_catalog)
        document["name"] = "unreviewed catalog metadata"
        catalog.write_text(json.dumps(document))
        try:
            result = original_read(root, path, **kwargs)
            changed.append(result)
            return result
        finally:
            catalog.write_bytes(approved_catalog)

    if mutation == "catalog":
        monkeypatch.setattr(source_service, "read_selected_file", changed_catalog)
    else:
        monkeypatch.setattr(package_files, "capture_package", altered_capture)
    with pytest.raises(ValueError, match="import_preview_changed"):
        await sources.commit_import(preview)
    assert changed and inspect_app_import("codex", marketplace) == preview
    assert sources.list_sources() == before
    assert set(staging.iterdir()) == retained


async def test_import_local_read_failure_stays_per_item_when_preview_is_unchanged(
    source_stack, marketplace, monkeypatch
):
    from tldw_chatbook.Plugins import source_service
    from tldw_chatbook.Plugins.imports import inspect_app_import
    from tldw_chatbook.Plugins.source_service import SourceService

    sources = SourceService(source_stack)
    preview = inspect_app_import("codex", marketplace)
    assert len((await sources.commit_import(preview)).source_ids) == 1
    before = sources.list_sources()
    original = source_service.read_selected_file
    failed = []

    def unavailable_during_read(root, path, **kwargs):
        if root != marketplace:
            return original(root, path, **kwargs)
        catalog = root / path
        hidden = catalog.with_suffix(".held")
        catalog.rename(hidden)
        try:
            return original(root, path, **kwargs)
        except FileNotFoundError:
            failed.append(path)
            raise
        finally:
            hidden.rename(catalog)

    monkeypatch.setattr(source_service, "read_selected_file", unavailable_during_read)
    result = await sources.commit_import(preview)
    assert failed and inspect_app_import("codex", marketplace) == preview
    assert not result.source_ids and not result.packages
    assert len(result.failures) == 1
    assert result.failures[0].code == "import_item_unavailable"
    assert sources.list_sources() == before


async def test_persisted_reference_discard_uses_retained_review_and_rechecks_identity(
    source_stack, native_package
):
    from dataclasses import replace

    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service = source_stack
    op = "persisted-display-candidate"
    snapshot = await service.acquire_snapshot(
        select_local_source(native_package()),
        op,
        await service.acquisition_destination(op),
    )
    sources = SourceService(service)
    rows = await sources.list_persisted_candidates()
    row = next(item for item in rows if item["operation_id"] == op)
    assert row["reason"] is None
    review = await sources.review_candidate_discard(row["reference"])
    with pytest.raises(ValueError):
        await sources.discard_candidate_reference(replace(review, token="forged"))
    assert Path(snapshot.materialized_identity).exists()
    await sources.discard_candidate_reference(review)
    assert Path(
        snapshot.materialized_identity
    ).exists(), "reference discard is not physical cleanup"
    after = next(
        item
        for item in await sources.list_persisted_candidates()
        if item["operation_id"] == op
    )
    assert after["reason"] == "reference_already_released"


async def test_selected_format_authenticates_unchanged_full_producer(
    source_stack, native_package
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service = source_stack
    op = "format-reviewed"
    snapshot = await service.acquire_snapshot(
        select_local_source(native_package()),
        op,
        await service.acquisition_destination(op),
    )
    sources = SourceService(service)
    inspected = await sources.inspect_acquired(snapshot, dialect="portable")
    assert inspected.dialect == "portable"
    forged = snapshot.model_copy(
        update={
            "inspection": snapshot.inspection.model_copy(
                update={"source_identity": "forged"}
            )
        }
    )
    with pytest.raises(ValueError):
        await sources.inspect_acquired(forged, dialect="portable")
    review = await sources.review_acquired(
        snapshot, dialect="portable", selection=("skill:review",), workspace_id=None
    )
    assert (await service.commit(review, review.operation_id)).committed
    assert not (await service.management_snapshot(None))[0].trusted


async def test_retained_root_review_keeps_candidate_inventory_and_discard_safe(
    source_stack, native_package
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    snapshots = []
    for name in ("installed-shared", "unsubmitted-draft"):
        snapshots.append(
            await service.acquire_snapshot(
                select_local_source(native_package()),
                name,
                await service.acquisition_destination(name),
            )
        )
    installed = await sources.review_acquired(
        snapshots[0], selection=("skill:review",), workspace_id=None
    )
    await service.commit(installed, installed.operation_id)
    root_review = await service.review_data_creation(installed.installation_id)
    draft = await sources.review_acquired(
        snapshots[1], selection=("skill:review",), workspace_id=None
    )
    rows = await sources.list_persisted_candidates()
    assert (
        next(row for row in rows if row["operation_id"] == "installed-shared")["reason"]
        == "acquisition_reference_in_use"
    )
    await sources.discard_candidate(snapshots[1])
    rows = await sources.list_persisted_candidates()
    assert (
        next(row for row in rows if row["operation_id"] == "unsubmitted-draft")[
            "reason"
        ]
        == "reference_already_released"
    )
    protected = next(row for row in rows if row["operation_id"] == "installed-shared")
    with pytest.raises(ValueError, match="acquisition_reference_in_use"):
        await sources.review_candidate_discard(protected["reference"])
    assert await service._call(
        lambda: service._coordinator._reviews.get(root_review.token) is root_review
    )
    with pytest.raises(ValueError):
        await service.commit(draft, draft.operation_id)
    assert Path(snapshots[0].materialized_identity).exists()


async def test_unknown_review_custody_blocks_discard_before_mutation(
    source_stack, native_package
):
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.source_service import SourceService

    service, sources = source_stack, SourceService(source_stack)
    operation = "unknown-review-custody"
    snapshot = await service.acquire_snapshot(
        select_local_source(native_package()),
        operation,
        await service.acquisition_destination(operation),
    )
    review = await sources.review_acquired(
        snapshot, selection=("skill:review",), workspace_id=None
    )
    await service._call(lambda: service._coordinator._reviews.update(unknown=object()))
    try:
        rows = await sources.list_persisted_candidates()
        assert rows[0]["reason"] == "acquisition_dependencies_unknown"
        with pytest.raises(ValueError, match="acquisition_dependencies_unknown"):
            await sources.discard_candidate(snapshot)
        assert await service._call(
            lambda: service._coordinator._reviews.get(review.token) is review
        )
    finally:
        await service._call(lambda: service._coordinator._reviews.pop("unknown"))
