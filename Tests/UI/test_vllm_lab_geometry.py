"""Production-stylesheet geometry and keyboard matrix for Lab's vLLM pane."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from textual.geometry import Region
from textual.widget import Widget
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)
from tldw_chatbook.UI.LLM_Management.vllm_connection import (
    VllmActivityEvent,
    VllmConnectionOwner,
    VllmConnectionSnapshot,
    VllmProbeResult,
)
from tldw_chatbook.UI.LLM_Management.vllm_profiles import (
    VllmProfileDocumentV1,
    profile_from_draft,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup import (
    VllmConnectionTarget,
    VllmIssue,
    VllmLaunchDraft,
    VllmMode,
    VllmModelSource,
    VllmPreflightResult,
    VllmReadinessState,
    launch_snapshot_from_draft,
    semantic_fingerprint,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup_view import VllmSetupView
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _no_splash(monkeypatch):
    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)


VLLM_GEOMETRY_STATES = (
    "setup_incomplete",
    "preflight_ready",
    "launching",
    "loading",
    "ready",
    "failed",
    "dirty_restart",
    "profile_management",
)

VLLM_GEOMETRY_SIZES = ((80, 24), (100, 30), (120, 40))

_COMMON_LOCAL_TAB_IDS = (
    "vllm-start-local-button",
    "vllm-connect-existing-button",
    "vllm-profile-select",
    "vllm-profile-name",
    "vllm-profile-create-button",
    "vllm-profile-save-button",
    "vllm-profile-rename-button",
    "vllm-profile-duplicate-button",
    "vllm-profile-delete-button",
    "vllm-python-environment",
    "vllm-hugging-face-source-button",
    "vllm-local-model-source-button",
    "vllm-hf-model",
    "vllm-bind-address",
    "vllm-port",
    "vllm-activity-toggle",
    "vllm-advanced-toggle",
)

_STATE_TAB_IDS = {
    "setup_incomplete": ("vllm-check-setup",) + _COMMON_LOCAL_TAB_IDS,
    "preflight_ready": ("vllm-check-setup", "vllm-start") + _COMMON_LOCAL_TAB_IDS,
    "launching": ("vllm-stop",) + _COMMON_LOCAL_TAB_IDS,
    "loading": ("vllm-stop",) + _COMMON_LOCAL_TAB_IDS,
    "ready": ("vllm-stop", "vllm-use-console")
    + _COMMON_LOCAL_TAB_IDS
    + ("vllm-make-default",),
    "failed": ("vllm-recovery-primary",) + _COMMON_LOCAL_TAB_IDS,
    "dirty_restart": ("vllm-stop", "vllm-restart") + _COMMON_LOCAL_TAB_IDS,
    "profile_management": (
        "vllm-check-setup",
        *_COMMON_LOCAL_TAB_IDS[:12],
        "vllm-local-model-directory",
        "vllm-browse-local-model-directory-button",
        *_COMMON_LOCAL_TAB_IDS[13:],
    ),
}


async def _mounted_vllm_view(app, pilot) -> tuple[LLMScreen, VllmSetupView]:
    """Mount the real Models screen and select vLLM through its catalog row."""

    assert app.CSS_PATH == TldwCli.CSS_PATH
    screen = LLMScreen(app)
    await app.push_screen(screen)
    for _ in range(40):
        await pilot.pause()
        if screen.query(LLMManagementWindow):
            break
    else:
        raise AssertionError("Models body did not mount")

    row = screen.query_one("#lab-models-row-vllm", Button)
    row.press()
    for _ in range(40):
        await pilot.pause()
        views = tuple(screen.query(VllmSetupView))
        if views:
            await pilot.pause()
            return screen, views[0]
    raise AssertionError("vLLM setup view did not mount")


def _base_draft() -> VllmLaunchDraft:
    return VllmLaunchDraft(
        mode=VllmMode.LOCAL,
        python_environment="python",
        model_source=VllmModelSource.HUGGING_FACE,
        model_value="org/model",
    )


def _state_projection(state: str):
    draft = _base_draft()
    token_owner = VllmConnectionOwner()
    token = token_owner.begin(draft, runtime_owner="chatbook")
    snapshot = launch_snapshot_from_draft(draft, generation=token.generation)
    preflight = VllmPreflightResult(
        generation=token.generation,
        fingerprint=token.fingerprint,
        issues=(),
        cli_path=Path("/safe/vllm"),
    )
    connection = VllmConnectionSnapshot(
        current_token=token,
        state=VllmReadinessState.NOT_CONFIGURED,
        launch_snapshot=None,
        target=None,
        issue=None,
    )
    runtime_active = False
    current_snapshot = None

    if state == "setup_incomplete":
        readiness = VllmReadinessState.NOT_CONFIGURED
        preflight = None
    elif state == "preflight_ready":
        readiness = VllmReadinessState.READY_TO_START
        connection = replace(connection, state=readiness)
    elif state in {"launching", "loading"}:
        readiness = (
            VllmReadinessState.LAUNCHING
            if state == "launching"
            else VllmReadinessState.LOADING_MODEL
        )
        connection = replace(connection, state=readiness, launch_snapshot=snapshot)
        runtime_active = True
        current_snapshot = snapshot
        preflight = None
    elif state == "ready":
        readiness = VllmReadinessState.READY
        claim = ServerLaunchClaim(provider="vllm", authority="chatbook-vllm")
        assert token_owner.bind_launch_claim(token, claim)
        target = VllmConnectionTarget(
            provider_key="vllm",
            api_url="http://127.0.0.1:8000/v1/chat/completions",
            model_id="chatbook-vllm",
            runtime_owner="chatbook",
            generation=token.generation,
            credential_source="none",
        )
        result = VllmProbeResult(
            token=token,
            state=readiness,
            target=target,
            issue=None,
            activity=(VllmActivityEvent("ready", "under_1s"),),
        )
        assert token_owner.settle(token, result)
        connection = token_owner.snapshot()
        runtime_active = True
        current_snapshot = snapshot
        preflight = None
    elif state == "failed":
        readiness = VllmReadinessState.NEEDS_ATTENTION
        issue = VllmIssue("model_missing", "model")
        connection = replace(
            connection,
            state=readiness,
            issue=issue,
            activity=(VllmActivityEvent("model_missing", "1_to_4s"),),
        )
        preflight = None
    elif state == "dirty_restart":
        readiness = VllmReadinessState.READY_TO_START
        current = draft
        draft = replace(draft, port=8001)
        preflight = VllmPreflightResult(
            generation=token.generation,
            fingerprint=semantic_fingerprint(draft),
            issues=(),
            cli_path=Path("/safe/vllm"),
        )
        connection = replace(connection, state=readiness)
        runtime_active = True
        current_snapshot = launch_snapshot_from_draft(
            current, generation=token.generation
        )
    elif state == "profile_management":
        readiness = VllmReadinessState.NOT_CONFIGURED
        draft = replace(
            draft,
            model_source=VllmModelSource.LOCAL_DIRECTORY,
            model_value="/private/local-model",
        )
        preflight = None
    else:  # pragma: no cover - the parameter list is the closed state set
        raise AssertionError(f"Unknown state: {state}")

    profile = profile_from_draft("Default vLLM", draft)
    profiles = VllmProfileDocumentV1(1, 0, profile.profile_id, (profile,))
    return {
        "draft": draft,
        "state": readiness,
        "preflight": preflight,
        "connection": connection,
        "current_launch_snapshot": current_snapshot,
        "profiles": profiles,
        "runtime_active": runtime_active,
    }


def _visible_focusables(view: Widget, compositor) -> tuple[Widget, ...]:
    return tuple(
        widget
        for widget in view.query("*").results(Widget)
        if widget.can_focus
        and not widget.disabled
        and widget in compositor.visible_widgets
    )


def _owner_region(widget: Widget, view: VllmSetupView) -> Region:
    parent = widget.parent
    if parent is None or parent is view.screen:
        return view.content_region
    return parent.content_region


async def _wait_for_profile_mutation_idle(screen: LLMScreen, pilot) -> None:
    for _ in range(4):
        await pilot.pause()
    for _ in range(80):
        worker = screen._vllm_profile_worker
        if worker is None or worker.is_finished:
            await pilot.pause()
            worker = screen._vllm_profile_worker
            if worker is None or worker.is_finished:
                return
        await pilot.pause()
    raise AssertionError("profile mutation did not settle")


@pytest.mark.parametrize("size", VLLM_GEOMETRY_SIZES)
@pytest.mark.parametrize("state", VLLM_GEOMETRY_STATES)
async def test_every_visible_focusable_is_inside_its_owner(size, state):
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen, view = await _mounted_vllm_view(app, pilot)
        view.apply_state(**_state_projection(state))
        await pilot.pause()

        expected_class = {
            (80, 24): "vllm-compact",
            (100, 30): "vllm-medium",
            (120, 40): "vllm-wide",
        }[size]
        assert view.has_class(expected_class), (
            f"{size} body width {view.size.width} did not select {expected_class}"
        )
        for widget in _visible_focusables(view, app.screen._compositor):
            owner = _owner_region(widget, view)
            assert widget.region.intersection(owner) == widget.region, (
                f"{state} {size}: {widget.id} escaped {widget.parent.id}: "
                f"widget={widget.region}, owner={owner}"
            )
            assert widget.region.intersection(view.content_region) == widget.region, (
                f"{state} {size}: {widget.id} escaped the active vLLM viewport: "
                f"widget={widget.region}, viewport={view.content_region}"
            )

        inspector = screen.query_one("#lab-inspector")
        if size != (120, 40):
            assert inspector.display is False
            handle = screen.query_one("#lab-inspector-handle")
            assert handle.display
            assert handle in app.screen._compositor.visible_widgets

        if size == (80, 24):
            assert screen.query_one("#lab-rail").display is False
            reopen = screen.query_one("#lab-rail-open", Button)
            assert reopen.display
            assert reopen in app.screen._compositor.visible_widgets
            painted = "\n".join(
                strip.text for strip in app.screen._compositor.render_strips()
            )
            assert str(view.query_one(f"#{_STATE_TAB_IDS[state][0]}").label) in painted
            assert "more below" in painted


@pytest.mark.parametrize("size", VLLM_GEOMETRY_SIZES)
async def test_profile_delete_confirmation_is_contained_and_keyboard_cancelable(size):
    """Removing the real modal or clipping either action must fail this contract."""

    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen, view = await _mounted_vllm_view(app, pilot)
        screen._apply_vllm_view_state()
        await _wait_for_profile_mutation_idle(screen, pilot)
        delete = view.query_one("#vllm-profile-delete-button", Button)
        delete.scroll_visible()
        await pilot.pause()
        path = screen._vllm_profile_repository.path
        before = path.read_bytes() if path.exists() else None

        delete.press()
        for _ in range(40):
            await pilot.pause()
            if isinstance(app.screen, ConfirmationDialog):
                break
        else:
            raise AssertionError("profile deletion confirmation did not mount")

        dialog = app.screen
        pane = dialog.query_one("#confirmation-dialog")
        focusables = _visible_focusables(dialog, dialog._compositor)
        assert tuple(widget.id for widget in focusables) == (
            "cancel-button",
            "confirm-button",
        )
        for widget in focusables:
            owner = widget.parent.content_region
            assert widget.region.intersection(owner) == widget.region
            assert widget.region.intersection(pane.content_region) == widget.region
        copy = " ".join(
            str(widget.renderable) for widget in dialog.query("Label, Static")
        )
        assert "Delete selected vLLM profile?" in copy
        assert "PROFILE_SECRET_CANARY" not in copy
        cancel = dialog.query_one("#cancel-button", Button)
        cancel.focus()
        await pilot.press("tab")
        assert app.focused.id == "confirm-button"
        await pilot.press("tab")
        assert app.focused.id == "cancel-button"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen
        assert app.focused is delete
        after = path.read_bytes() if path.exists() else None
        assert after == before


@pytest.mark.parametrize("size", VLLM_GEOMETRY_SIZES)
@pytest.mark.parametrize("state", VLLM_GEOMETRY_STATES)
async def test_complete_tab_walk_stays_in_active_vllm_provider(size, state):
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        _, view = await _mounted_vllm_view(app, pilot)
        view.apply_state(**_state_projection(state))
        await pilot.pause()

        expected = _STATE_TAB_IDS[state]
        first = view.query_one(f"#{expected[0]}")
        first.focus()
        await pilot.pause()
        visited = [app.focused.id]
        for _ in range(len(expected) - 1):
            await pilot.press("tab")
            visited.append(app.focused.id)
        assert tuple(visited) == expected

        await pilot.press("tab")
        assert app.focused.id == expected[0]
        assert view in app.focused.ancestors_with_self
        hidden_provider_ids = {
            widget.id
            for pane in app.query(".llm-view")
            if pane.id != "llm-view-vllm"
            for widget in pane.query("*").results(Widget)
            if widget.id
        }
        assert hidden_provider_ids.isdisjoint(visited)


async def test_background_projection_preserves_focus_but_explicit_transition_moves_it():
    app = _build_test_app()
    async with app.run_test(size=(100, 30)) as pilot:
        screen, view = await _mounted_vllm_view(app, pilot)
        draft = _base_draft()
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        screen._vllm_draft = draft
        screen._vllm_preflight = VllmPreflightResult(
            generation=token.generation,
            fingerprint=token.fingerprint,
            issues=(),
            cli_path=Path("/safe/vllm"),
        )
        screen._settle_vllm_state(
            token,
            VllmReadinessState.READY_TO_START,
            activity_code="checking",
        )

        profile_name = view.query_one("#vllm-profile-name", Input)
        profile_name.focus()
        screen._apply_vllm_view_state(focus=False)
        await pilot.pause()
        assert app.focused is profile_name

        screen._apply_vllm_view_state(focus=True)
        await pilot.pause()
        assert app.focused.id == "vllm-start"


async def test_provider_child_has_no_bracket_or_digit_bindings():
    keys = {binding.key for binding in LLMManagementWindow.BINDINGS}
    assert keys.isdisjoint({"[", "]", *(str(number) for number in range(1, 10))})
