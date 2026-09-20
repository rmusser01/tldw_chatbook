"""Exact handoffs through the mounted Library and real Chatbook registry."""

import threading
from dataclasses import replace

import pytest

from Tests.Chatbooks.test_artifact_registry_snapshot import seed_registry
from Tests.private_profile import private_profile_test
from Tests.UI.test_library_artifacts_canvas import artifact_library
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.Library.library_artifacts_catalog import LibraryArtifactsCatalog
from tldw_chatbook.Library.library_artifacts_state import ArtifactKey
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

CHANNEL = HandoffChannel.ARTIFACT_CHATBOOK_TARGET
pytestmark = pytest.mark.ui


async def settled(pilot, condition):
    for _ in range(150):
        if condition():
            return
        await pilot.pause(0.02)
    raise AssertionError("Artifact navigation did not settle")


async def registry(screen, pilot, tmp_path):
    service = LocalChatbookService(registry_path=tmp_path / "registered.json")
    await seed_registry(service, 45)
    screen.app_instance.local_chatbook_service = service
    controller = screen._artifacts_controller
    controller.resume()
    await settled(
        pilot, lambda: not controller.loading and controller.detail is not None
    )
    return controller, screen.app_instance.pending_handoffs


@pytest.mark.asyncio
@private_profile_test
async def test_exact_target_beyond_first_page_is_consumed_only_after_applied(
    tmp_path, request, monkeypatch
):
    async with artifact_library(tmp_path) as (screen, pilot):
        controller, store = await registry(screen, pilot, tmp_path)
        await screen._select_library_rail_row("artifacts-chatbooks")
        await settled(
            pilot,
            lambda: (
                controller.page is not None
                and controller.page.scope.view == "chatbooks"
            ),
        )
        controller.request_scope(
            replace(controller.scope, query="excluding-restored-filter")
        )
        await settled(
            pilot, lambda: controller.page is not None and controller.page.total == 0
        )
        target = ArtifactKey("chatbook", 1)
        entered, release = threading.Event(), threading.Event()
        original = LibraryArtifactsCatalog.locate

        def delayed(catalog, scope, key):
            if key == target:
                entered.set()
                assert release.wait(10)
            return original(catalog, scope, key)

        monkeypatch.setattr(LibraryArtifactsCatalog, "locate", delayed)
        store.stage(CHANNEL, "local:chatbook:1")
        try:
            screen.apply_navigation_context({"mode": "artifacts-all"})
            await settled(pilot, entered.is_set)
            assert store.exact_revision_status(CHANNEL, 1) == "in_flight"
            assert controller.selected != target
            release.set()
            await settled(
                pilot,
                lambda: (
                    controller.detail is not None and controller.detail.key == target
                ),
            )
            assert controller.page.start == 40
            assert controller.page.total == 45
            assert controller.selected == target
            assert controller.scope.view == "chatbooks"
            assert store.exact_revision_status(CHANNEL, 1) == "settled"
        finally:
            release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_missing_exact_target_clears_previous_detail_and_claim(tmp_path, request):
    async with artifact_library(tmp_path) as (screen, pilot):
        controller, store = await registry(screen, pilot, tmp_path)
        assert controller.detail is not None
        store.stage(CHANNEL, "local:chatbook:9999")
        screen.apply_navigation_context({"mode": "artifacts-all"})
        await settled(pilot, lambda: "missing" in controller.error)
        assert controller.selected is None
        assert controller.detail is None
        assert controller.page is None
        assert store.exact_revision_status(CHANNEL, 1) == "settled"


@pytest.mark.asyncio
@private_profile_test
async def test_failed_exact_source_retry_retains_target_and_then_applies(
    tmp_path, request, monkeypatch
):
    async with artifact_library(tmp_path) as (screen, pilot):
        controller, store = await registry(screen, pilot, tmp_path)
        target = ArtifactKey("chatbook", 1)
        original = LibraryArtifactsCatalog.locate
        calls = []

        def fail_once(catalog, scope, key):
            calls.append(key)
            if len(calls) == 1:
                raise RuntimeError("Source temporarily unavailable")
            return original(catalog, scope, key)

        monkeypatch.setattr(LibraryArtifactsCatalog, "locate", fail_once)
        store.stage(CHANNEL, "local:chatbook:1")
        screen.apply_navigation_context({"mode": "artifacts-all"})
        await settled(pilot, lambda: bool(controller.error))
        assert store.has_pending(CHANNEL)
        assert controller.selected != target
        controller.action("retry")
        await settled(
            pilot,
            lambda: controller.detail is not None and controller.detail.key == target,
        )
        assert calls == [target, target]
        assert controller.page.start == 40
        assert controller.selected == target
        assert store.exact_revision_status(CHANNEL, 1) == "settled"
