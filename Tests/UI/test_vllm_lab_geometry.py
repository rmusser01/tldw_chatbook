"""Production-stylesheet geometry and keyboard matrix for Lab's vLLM pane."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

import pytest
from textual.geometry import Region
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select

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
    "checking",
    "preflight_ready",
    "launching",
    "loading",
    "ready",
    "failed",
    "dirty_restart",
    "profile_management",
    "existing_discovery",
    "existing_ready",
)

VLLM_GEOMETRY_SIZES = ((80, 24), (100, 30), (120, 40))

_PROFILE_TAB_IDS = (
    "vllm-start-local-button",
    "vllm-connect-existing-button",
    "vllm-profile-select",
    "vllm-profile-name",
    "vllm-profile-create-button",
    "vllm-profile-save-button",
    "vllm-profile-rename-button",
    "vllm-profile-duplicate-button",
    "vllm-profile-delete-button",
)

_COMMON_LOCAL_TAB_IDS = _PROFILE_TAB_IDS + (
    "vllm-python-environment",
    "vllm-browse-python-environment",
    "vllm-hugging-face-source-button",
    "vllm-local-model-source-button",
    "vllm-hf-model",
    "vllm-bind-address",
    "vllm-port",
    "vllm-activity-toggle",
    "vllm-advanced-toggle",
)

_COMMON_EXISTING_TAB_IDS = (
    "vllm-start-local-button",
    "vllm-connect-existing-button",
    "vllm-profile-select",
    "vllm-existing-server-url",
    "vllm-existing-model",
    "vllm-activity-toggle",
)

_STATE_TAB_IDS = {
    "setup_incomplete": ("vllm-check-setup",) + _COMMON_LOCAL_TAB_IDS,
    "checking": ("vllm-cancel-check",) + _COMMON_LOCAL_TAB_IDS,
    "preflight_ready": ("vllm-check-setup", "vllm-start") + _COMMON_LOCAL_TAB_IDS,
    "launching": ("vllm-stop",) + _COMMON_LOCAL_TAB_IDS,
    "loading": ("vllm-stop",) + _COMMON_LOCAL_TAB_IDS,
    "ready": ("vllm-stop", "vllm-use-console")
    + _COMMON_LOCAL_TAB_IDS
    + ("vllm-make-default",),
    "failed": ("vllm-recovery-primary",) + _COMMON_LOCAL_TAB_IDS,
    "dirty_restart": ("vllm-check-setup", "vllm-stop", "vllm-restart")
    + _COMMON_LOCAL_TAB_IDS,
    "profile_management": (
        "vllm-check-setup",
        *_PROFILE_TAB_IDS,
        "vllm-python-environment",
        "vllm-browse-python-environment",
        "vllm-hugging-face-source-button",
        "vllm-local-model-source-button",
        "vllm-local-model-directory",
        "vllm-browse-local-model-directory-button",
        "vllm-bind-address",
        "vllm-port",
        "vllm-activity-toggle",
        "vllm-advanced-toggle",
    ),
    "existing_discovery": ("vllm-check-setup",) + _COMMON_EXISTING_TAB_IDS,
    "existing_ready": ("vllm-use-console",)
    + _COMMON_EXISTING_TAB_IDS
    + ("vllm-make-default",),
}

_STATE_OUTCOME_COPY = {
    "setup_incomplete": (
        ("vllm-readiness-state", "Setup incomplete"),
        ("vllm-check-model", "choose a model"),
    ),
    "checking": (
        ("vllm-readiness-state", "Checking setup"),
        ("vllm-check-environment", "checking"),
    ),
    "preflight_ready": (
        ("vllm-readiness-state", "Ready to start"),
        ("vllm-check-environment", "Python resolved"),
    ),
    "launching": (("vllm-readiness-state", "Launching process"),),
    "loading": (("vllm-readiness-state", "Loading model"),),
    "ready": (
        ("vllm-readiness-state", "Ready at"),
        ("vllm-check-model", "exact selection verified"),
    ),
    "failed": (
        ("vllm-readiness-state", "Needs attention"),
        ("vllm-activity-summary", "Expected chat model is unavailable"),
    ),
    "dirty_restart": (
        ("vllm-next-restart-state", "Modified for next restart"),
        ("vllm-next-restart-changes", "Port"),
    ),
    "profile_management": (
        ("vllm-mode-summary", "Start on this computer"),
        ("vllm-start-blocker", "Check setup before Start"),
    ),
    "existing_discovery": (
        ("vllm-readiness-state", "Setup incomplete"),
        ("vllm-credential-status", "not configured"),
        ("vllm-existing-model-help", "Select a returned model"),
        ("vllm-check-model", "choose one returned model"),
        ("vllm-check-network", "API reachable"),
    ),
    "existing_ready": (
        ("vllm-readiness-state", "Ready · Existing vLLM server"),
        ("vllm-existing-model-help", "exact fresh verification"),
        ("vllm-console-scope-copy", "Session only"),
    ),
}


async def _mounted_vllm_view(app, pilot) -> tuple[LLMScreen, VllmSetupView]:
    """Mount the real Models screen and select vLLM through its catalog row."""

    assert app.CSS_PATH == TldwCli.CSS_PATH
    screen = LLMScreen(app)
    # Geometry states are projected explicitly below. Keep asynchronous profile
    # storage recovery from focusing/scrolling a different row mid-measurement.
    screen._vllm_profiles_loaded = True
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


def _close_geometry_app_databases(app: TldwCli) -> None:
    """Close factory-app databases retained across repeated matrix mounts.

    A real process owns one ``TldwCli`` and the operating system releases its
    process-lifetime SQLite handles at exit.  This matrix deliberately mounts
    dozens of independent apps in one pytest process, so it must close those
    owners explicitly rather than make the suite-level descriptor sentinel
    account for test-only multiplicity.
    """

    orchestrator = getattr(app, "evaluation_orchestrator", None)
    if orchestrator is not None:
        orchestrator.close()
    subscriptions = getattr(app, "subscriptions_db", None)
    if subscriptions is not None:
        subscriptions.close_all_connections()
    for attribute in ("local_library_collections_db", "local_workspace_db"):
        database = getattr(app, attribute, None)
        if database is not None:
            database.close()


@asynccontextmanager
async def _run_vllm_geometry_app(
    size: tuple[int, int],
) -> AsyncIterator[tuple[TldwCli, Pilot]]:
    """Mount one production-styled app and settle its test-only DB owners."""

    app = _build_test_app()
    try:
        async with app.run_test(size=size) as pilot:
            yield app, pilot
    finally:
        _close_geometry_app_databases(app)


def _base_draft() -> VllmLaunchDraft:
    return VllmLaunchDraft(
        mode=VllmMode.LOCAL,
        python_environment="python",
        model_source=VllmModelSource.HUGGING_FACE,
        model_value="org/model",
    )


def _state_projection(state: str):
    local_profile_draft = _base_draft()
    draft = local_profile_draft
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
    discovered_model_ids: tuple[str, ...] = ()

    if state == "setup_incomplete":
        draft = replace(draft, model_value="")
        token_owner = VllmConnectionOwner()
        token = token_owner.begin(draft, runtime_owner="chatbook")
        connection = token_owner.snapshot()
        readiness = VllmReadinessState.NOT_CONFIGURED
        preflight = None
    elif state == "checking":
        readiness = VllmReadinessState.CHECKING
        connection = token_owner.snapshot()
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
    elif state in {"existing_discovery", "existing_ready"}:
        selected = "" if state == "existing_discovery" else "org/model-b"
        draft = replace(
            local_profile_draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
            existing_model_id=selected,
        )
        token_owner = VllmConnectionOwner()
        token = token_owner.begin(draft, runtime_owner="external")
        discovered_model_ids = (
            ("org/model-a", "org/model-b")
            if state == "existing_discovery"
            else ("org/model-b",)
        )
        target = (
            VllmConnectionTarget(
                provider_key="vllm",
                api_url="http://127.0.0.1:8000/v1/chat/completions",
                model_id=selected,
                runtime_owner="external",
                generation=token.generation,
                credential_source="none",
            )
            if selected
            else None
        )
        readiness = (
            VllmReadinessState.READY
            if target is not None
            else VllmReadinessState.NOT_CONFIGURED
        )
        result = VllmProbeResult(
            token=token,
            state=readiness,
            target=target,
            issue=None,
            activity=(
                VllmActivityEvent(
                    "ready" if target is not None else "models_discovered",
                    "under_1s",
                ),
            ),
            discovered_model_ids=discovered_model_ids,
        )
        assert token_owner.settle(token, result)
        connection = token_owner.snapshot()
        preflight = None
    else:  # pragma: no cover - the parameter list is the closed state set
        raise AssertionError(f"Unknown state: {state}")

    profile = profile_from_draft("Default vLLM", local_profile_draft)
    profiles = VllmProfileDocumentV1(1, 0, profile.profile_id, (profile,))
    return {
        "draft": draft,
        "state": readiness,
        "preflight": preflight,
        "connection": connection,
        "current_launch_snapshot": current_snapshot,
        "profiles": profiles,
        "runtime_active": runtime_active,
        "discovered_model_ids": discovered_model_ids,
        "credential_configured": False,
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
    async with _run_vllm_geometry_app(size) as (app, pilot):
        screen, view = await _mounted_vllm_view(app, pilot)
        view.apply_state(**_state_projection(state))
        await pilot.pause()

        for widget_id, expected_copy in _STATE_OUTCOME_COPY[state]:
            widget = view.query_one(f"#{widget_id}", Label)
            assert expected_copy in str(widget.renderable), (
                f"{state}: {widget_id} did not prove {expected_copy!r}"
            )
        first_action = view.query_one(f"#{_STATE_TAB_IDS[state][0]}", Button)
        assert first_action.display and not first_action.disabled
        if state.startswith("existing_"):
            model_select = view.query_one("#vllm-existing-model", Select)
            assert model_select.display and not model_select.disabled
            assert model_select.value == (
                Select.NULL if state == "existing_discovery" else "org/model-b"
            )

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

    async with _run_vllm_geometry_app(size) as (app, pilot):
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
    async with _run_vllm_geometry_app(size) as (app, pilot):
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
    async with _run_vllm_geometry_app((100, 30)) as (app, pilot):
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
