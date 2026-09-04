from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Input, Label, TextArea

from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.UI.LLM_Management.vllm_connection import (
    VllmActivityEvent,
    VllmConnectionOwner,
    VllmProbeResult,
)
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.LLM_Management.vllm_setup import (
    VllmConnectionTarget,
    VllmIssue,
    VllmLaunchDraft,
    VllmMode,
    VllmModelSource,
    VllmReadinessState,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup_view import VllmSetupView
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from Tests.UI.app_factory import _build_test_app


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _no_splash(monkeypatch):
    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)


class _VllmHost(App[None]):
    def compose(self) -> ComposeResult:
        yield VllmSetupView(id="vllm-setup")


async def test_initial_vllm_setup_is_guided_and_blocks_start():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        copy = " ".join(str(label.renderable) for label in view.query(Label))
        buttons = " ".join(str(button.label) for button in view.query(Button))
        assert "Set up vLLM" in copy
        assert "Start on this computer" in buttons
        assert "Connect to existing server" in buttons
        assert app.query_one("#vllm-port", Input).value == "8000"
        assert app.query_one("#vllm-start-button", Button).disabled
        assert "GGUF" not in copy
        assert "checkpoint" not in copy.lower()


async def test_source_specific_controls_and_mode_drafts_are_preserved():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        hf_input = app.query_one("#vllm-hf-model", Input)
        local_input = app.query_one("#vllm-local-model-directory", Input)
        assert hf_input.display
        assert not local_input.display

        hf_input.value = "org/kept-model"
        await pilot.pause()
        await pilot.click("#vllm-connect-existing-button")
        assert app.query_one("#vllm-existing-server-url", Input).display
        await pilot.click("#vllm-start-local-button")
        assert app.query_one("#vllm-hf-model", Input).value == "org/kept-model"

        await pilot.click("#vllm-local-model-source-button")
        assert local_input.display
        assert not hf_input.display
        assert app.query_one(
            "#vllm-browse-local-model-directory-button", Button
        ).display


async def test_preflight_blocker_is_adjacent_and_start_enables_only_for_current_success():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NEEDS_ATTENTION,
            preflight=None,
        )
        assert app.query_one("#vllm-start-button", Button).disabled
        assert "Check setup" in str(
            app.query_one("#vllm-start-blocker", Label).renderable
        )


async def test_lifecycle_projection_survives_remount_and_process_exit():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        view.project_lifecycle(active=True)
        assert not app.query_one("#vllm-stop-button", Button).disabled
        view.project_lifecycle(active=False, status="process exited")
        assert app.query_one("#vllm-stop-button", Button).disabled is False


def _ready_result(token) -> VllmProbeResult:
    return VllmProbeResult(
        token=token,
        state=VllmReadinessState.READY,
        target=VllmConnectionTarget(
            provider_key="vllm",
            api_url="http://127.0.0.1:8000/v1/chat/completions",
            model_id="chatbook-vllm",
            runtime_owner=token.runtime_owner,
            generation=token.generation,
            credential_source="none",
        ),
        issue=None,
        activity=(VllmActivityEvent("ready", "under_1s"),),
    )


async def test_mounted_activity_renders_ready_and_expands_bounded_failure():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="/private/PATH_CANARY/bin/python",
            model_source=VllmModelSource.LOCAL_DIRECTORY,
            model_value="/private/MODEL_SOURCE_CANARY",
            raw_arguments="--flag COMMAND_CANARY",
        )
        owner = VllmConnectionOwner()
        token = owner.begin(draft, runtime_owner="chatbook")
        assert owner.settle(token, _ready_result(token))
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.READY,
            preflight=None,
            connection=owner.snapshot(),
        )
        assert "Ready at" in str(
            app.query_one("#vllm-readiness-state", Label).renderable
        )

        token = owner.begin(draft, runtime_owner="chatbook")
        failure = VllmProbeResult(
            token=token,
            state=VllmReadinessState.NEEDS_ATTENTION,
            target=None,
            issue=VllmIssue("model_missing", "model"),
            activity=(VllmActivityEvent("model_missing", "1_to_4s"),),
        )
        assert owner.settle(token, failure)
        view.apply_state(
            draft=draft,
            state=failure.state,
            preflight=None,
            connection=owner.snapshot(),
        )
        assert not app.query_one("#vllm-activity-details", Collapsible).collapsed
        assert not app.query_one("#vllm-retry-button", Button).disabled
        visible = " ".join(str(label.renderable) for label in view.query(Label))
        assert "Expected chat model is unavailable" in visible
        assert not any(
            canary in visible
            for canary in ("PATH_CANARY", "MODEL_SOURCE_CANARY", "COMMAND_CANARY")
        )


async def _mount_vllm_screen(
    app, pilot
) -> tuple[LLMScreen, LLMManagementWindow, VllmSetupView]:
    screen = LLMScreen(app)
    await app.push_screen(screen)
    for _ in range(8):
        await pilot.pause()
    window = screen.query_one(LLMManagementWindow)
    window.active_view = "vllm"
    for _ in range(12):
        await pilot.pause()
        views = list(screen.query(VllmSetupView))
        if views:
            return screen, window, views[0]
    raise AssertionError("vLLM setup view did not mount")


async def test_mounted_draft_edit_fences_old_readiness_generation():
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        assert app._vllm_connection_owner is screen._vllm_owner
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._vllm_draft = draft
        screen._apply_vllm_view_state()

        view.query_one("#vllm-hf-model", Input).value = "org/changed-model"
        await pilot.pause()

        snapshot = screen._vllm_owner.snapshot()
        assert snapshot.generation == token.generation + 1
        assert snapshot.target is None
        assert snapshot.activity[-1].code == "target_changed"
        assert screen._vllm_owner.settle(token, _ready_result(token)) is False

        current_draft = screen._vllm_draft
        token = screen._vllm_owner.begin(current_draft, runtime_owner="chatbook")
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._apply_vllm_view_state()
        view.query_one("#vllm-raw-arguments", TextArea).text = "--enable-prefix-caching"
        await pilot.pause()
        raw_edit = screen._vllm_owner.snapshot()
        assert raw_edit.generation == token.generation + 1
        assert raw_edit.activity[-1].code == "target_changed"


async def test_mounted_recomposition_and_detach_invalidate_readiness_generation():
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        draft = screen._vllm_draft
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        assert screen._vllm_owner.settle(token, _ready_result(token))

        await screen.recompose()
        for _ in range(8):
            await pilot.pause()
        recomposed = screen._vllm_owner.snapshot()
        assert recomposed.generation == token.generation + 1
        assert recomposed.target is None
        assert recomposed.activity[-1].code == "recomposed"

        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        assert screen._vllm_owner.settle(token, _ready_result(token))
        await app.pop_screen()
        await pilot.pause()
        detached = screen._vllm_owner.snapshot()
        assert detached.generation == token.generation + 1
        assert detached.target is None
        assert detached.activity[-1].code == "screen_detached"
