"""Settings Change Review shows real readiness and recoverable local outcomes."""

import asyncio
import threading

import pytest
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from Tests.UI.test_settings_workspace_memory_confirmation import _press
from tldw_chatbook.Widgets.Settings_Widgets.workspace_change_review import (
    WorkspaceChangeReviewPanel,
)
from tldw_chatbook.Workspaces.change_review_consent import RootReadinessState


async def _visible(host, pilot, suffix):
    async with asyncio.timeout(5):
        while not host.screen.query_one(
            "#settings-workspace-change-review-" + suffix
        ).display:
            await pilot.pause(0.05)
    await _settle(host, pilot)


def _receipt(host, needle):
    receipt = host.screen.query_one("#settings-workspace-change-review-result", Static)
    assert needle in str(receipt.renderable)
    _assert_painted(host.screen, receipt)
    _assert_painted(host.screen, host.focused)
    assert " ".join(str(receipt.renderable).split()) in " ".join(
        _painted(host, receipt).split()
    )
    assert "/private/synthetic-root" not in str(receipt.renderable)


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_change_review_keyboard_conflict_preparation_and_retry(
    request, tmp_path, monkeypatch, theme, size
):
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-review", name="Review workspace")
    root = tmp_path / "workspace"
    root.mkdir()
    registry.add_folder_binding("ws-review", root)
    service = app.change_review_consent_service
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def initialize(path):
        calls.append(path)
        entered.set()
        if not release.wait(20):
            raise RuntimeError("test never released initializer")
        if len(calls) == 1:
            raise RuntimeError("preparation failed at /private/synthetic-root")

    monkeypatch.setattr(service, "_initialize_root", initialize)
    host = _StyledDestinationHarness(app, "settings")
    host.theme = theme
    try:
        async with host.run_test(size=size) as pilot:
            await _category(host, pilot, "Workspaces")
            await _press(host, pilot, "#settings-workspace-row-ws-review")
            panel = host.screen.query_one(WorkspaceChangeReviewPanel)
            registry.set_change_review_enabled("ws-review", True)
            await _press(host, pilot, "#settings-workspace-change-review-toggle")
            assert registry.change_review_enabled("ws-review")
            _receipt(host, "changed elsewhere")
            await _press(host, pilot, "#settings-workspace-change-review-toggle")
            assert not registry.change_review_enabled("ws-review")
            _receipt(host, "existing history is retained")
            await _press(host, pilot, "#settings-workspace-change-review-toggle")
            assert await asyncio.to_thread(entered.wait, 2)
            await _visible(host, pilot, "preparing")
            _receipt(host, "enabled")
            rename = host.screen.query_one("#settings-workspace-rename-input", Input)
            rename.focus()
            await pilot.press("home", "shift+end", *"Draft name")
            await _settle(host, pilot)
            release.set()
            await _visible(host, pilot, "failed")
            assert host.screen.query_one("#settings-workspace-rename-input") is rename
            assert rename.value == "Draft name" and host.focused is rename
            _assert_painted(host.screen, rename)
            assert "Draft name" in _painted(host, rename)
            original_retry = service.retry_failed_roots

            def fail_retry(_workspace_id):
                raise RuntimeError("failure at /private/synthetic-root")

            monkeypatch.setattr(service, "retry_failed_roots", fail_retry)
            await _press(host, pilot, "#settings-workspace-change-review-retry")
            _receipt(host, "could not be retried")
            assert len(calls) == 1
            monkeypatch.setattr(service, "retry_failed_roots", original_retry)
            # The panel callback resolves the service method at activation time.
            release.clear()
            entered.clear()
            await _press(host, pilot, "#settings-workspace-change-review-retry")
            assert await asyncio.to_thread(entered.wait, 2)
            await _visible(host, pilot, "preparing")
            _receipt(host, "Retry scheduled for 1")
            assert host.focused.id == "settings-workspace-change-review-toggle"
            assert len(calls) == 2
            assert original_retry("ws-review") == 0
            release.set()
            await _visible(host, pilot, "ready")
            _receipt(host, "Retry scheduled for 1")
            assert host.screen.query_one("#settings-workspace-rename-input") is rename
            assert rename.value == "Draft name"
            assert (
                service.status("ws-review").roots[0].state is RootReadinessState.READY
            )
            assert host.screen.query_one(WorkspaceChangeReviewPanel) is panel
            await _press(host, pilot, "#settings-workspace-change-review-toggle")
            assert not registry.change_review_enabled("ws-review")
            assert not host.screen.query_one(
                "#settings-workspace-change-review-ready"
            ).display
            _receipt(host, "existing history is retained")
    finally:
        release.set()
        service.shutdown()


@pytest.mark.parametrize("navigation", ["workspace", "modal"])
@private_profile_test
async def test_slow_readiness_observation_cannot_publish_after_navigation(
    request, monkeypatch, navigation
):
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-review", name="Review workspace")
    registry.create_workspace(workspace_id="ws-other", name="Other workspace")
    host = _StyledDestinationHarness(app, "settings")
    entered = threading.Event()
    release = threading.Event()
    try:
        async with host.run_test(size=(170, 48)) as pilot:
            await _category(host, pilot, "Workspaces")
            await _press(host, pilot, "#settings-workspace-row-ws-review")
            panel = host.screen.query_one(WorkspaceChangeReviewPanel)
            observed = panel._status
            reads = []
            paints = []

            def held_read():
                reads.append(True)
                entered.set()
                assert release.wait(10)

            monkeypatch.setattr(panel, "_read_status", held_read)
            monkeypatch.setattr(panel, "_paint_status", lambda: paints.append(True))
            panel._refresh_preparing()
            assert await asyncio.to_thread(entered.wait, 2)
            if navigation == "workspace":
                # Do not wait for all workers while this read is deliberately held.
                button = host.screen.query_one("#settings-workspace-row-ws-other")
                button.focus()
                await pilot.press("enter")
                await pilot.pause(0.2)
                assert host.screen._settings_selected_workspace_id == "ws-other"
            else:
                await host.push_screen(ModalScreen())
                await pilot.pause()
            release.set()
            await _settle(host, pilot)
            assert reads == [True] and paints == []
            assert panel._status is observed
            if navigation == "modal":
                await host.pop_screen()
                await _settle(host, pilot)
            else:
                assert host.screen.query_one(WorkspaceChangeReviewPanel) is not panel
                assert not registry.change_review_enabled("ws-other")
    finally:
        release.set()


@private_profile_test
async def test_queued_toggle_preserves_the_activated_consent_intent(request):
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-review", name="Review workspace")
    registry.set_change_review_enabled("ws-review", True)
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        await _category(host, pilot, "Workspaces")
        await _press(host, pilot, "#settings-workspace-row-ws-review")
        panel = host.screen.query_one(WorkspaceChangeReviewPanel)
        toggle = host.screen.query_one("#settings-workspace-change-review-toggle")
        toggle.focus()
        await _settle(host, pilot)
        assert str(toggle.label) == "Disable change review"
        toggle.press()  # Queued; do not yield before replacing the observation.
        registry.set_change_review_enabled("ws-review", False)
        panel._status = app.change_review_consent_service.status("ws-review")
        panel._paint_status()
        assert str(toggle.label) == "Enable change review"
        await _settle(host, pilot)
        assert not registry.change_review_enabled("ws-review")
        _receipt(host, "changed elsewhere")
