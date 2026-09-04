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
    VllmPreflightResult,
    VllmReadinessState,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup_view import VllmSetupView
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
    clear_server_process,
    publish_server_process,
    release_server_claim,
    reserve_server_launch,
)
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


class _RunningProcess:
    pid = 12345

    def __init__(self) -> None:
        self.running = True

    def poll(self):
        return None if self.running else 0

    def terminate(self) -> None:
        self.running = False

    def wait(self, timeout=None) -> int:
        self.running = False
        return 0

    def kill(self) -> None:
        self.running = False


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


async def test_lifecycle_projection_enables_stop_only_while_runtime_is_active():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        view.project_lifecycle(active=True)
        assert not app.query_one("#vllm-stop-button", Button).disabled
        view.project_lifecycle(active=False, status="process exited")
        assert app.query_one("#vllm-stop-button", Button).disabled


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


def _bind_local_claim(owner: VllmConnectionOwner, token) -> ServerLaunchClaim:
    claim = ServerLaunchClaim(provider="vllm", authority="chatbook-vllm")
    assert owner.bind_launch_claim(token, claim)
    return claim


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
        _bind_local_claim(owner, token)
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
        claim = _bind_local_claim(screen._vllm_owner, token)
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
        assert screen._vllm_owner.bind_launch_claim(token, claim)
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
        claim = _bind_local_claim(screen._vllm_owner, token)
        assert screen._vllm_owner.settle(token, _ready_result(token))

        await screen.recompose()
        for _ in range(8):
            await pilot.pause()
        recomposed = screen._vllm_owner.snapshot()
        assert recomposed.generation == token.generation + 1
        assert recomposed.target is None
        assert recomposed.activity[-1].code == "recomposed"

        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        await app.pop_screen()
        await pilot.pause()
        detached = screen._vllm_owner.snapshot()
        assert detached.generation == token.generation + 1
        assert detached.target is None
        assert detached.activity[-1].code == "screen_detached"


async def test_stop_before_process_publication_settles_cancel_and_retry_refuses_claim():
    """Catch a cancelled reservation leaving Retry indefinitely loading."""

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert claim is not None
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        screen._vllm_claim = claim
        screen._vllm_draft = draft
        screen._settle_vllm_state(
            token,
            VllmReadinessState.LAUNCHING,
            activity_code="launch_reserved",
        )
        screen._apply_vllm_view_state()

        await screen._on_vllm_stop_requested(VllmSetupView.StopRequested())
        stopped = screen._vllm_owner.snapshot()
        assert stopped.state is VllmReadinessState.NEEDS_ATTENTION
        assert stopped.issue == VllmIssue("cancelled", "process")

        screen._on_vllm_retry_requested(VllmSetupView.RetryRequested())
        await pilot.pause()
        retried = screen._vllm_owner.snapshot()
        assert retried.state is VllmReadinessState.NEEDS_ATTENTION
        assert retried.issue == VllmIssue("cancelled", "process")
        assert screen._vllm_probe_worker is None
        assert release_server_claim(app, "vllm", claim)


async def test_live_owned_claim_keeps_stop_enabled_across_edit_and_screen_replacement():
    """Catch editable connection state hiding an app-owned live process."""

    app = _build_test_app()
    process = _RunningProcess()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert claim is not None
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        assert publish_server_process(app, "vllm", claim, process)
        screen._vllm_claim = claim
        screen._vllm_draft = draft
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._apply_vllm_view_state()
        assert not view.query_one("#vllm-stop-button", Button).disabled

        view.query_one("#vllm-hf-model", Input).value = "org/edited-model"
        await pilot.pause()
        assert screen._vllm_owner.snapshot().state is (
            VllmReadinessState.NOT_CONFIGURED
        )
        assert not view.query_one("#vllm-stop-button", Button).disabled

        await app.pop_screen()
        await pilot.pause()
        replacement = LLMScreen(app)
        assert replacement._vllm_claim is claim
        await app.push_screen(replacement)
        for _ in range(8):
            await pilot.pause()
        replacement_window = replacement.query_one(LLMManagementWindow)
        replacement_window.active_view = "vllm"
        for _ in range(12):
            await pilot.pause()
            replacement_views = list(replacement.query(VllmSetupView))
            if replacement_views:
                break
        else:
            raise AssertionError("replacement vLLM setup view did not mount")
        replacement._apply_vllm_view_state()
        assert not replacement_views[0].query_one(
            "#vllm-stop-button", Button
        ).disabled

        process.running = False
        assert clear_server_process(app, "vllm", claim, process)


async def test_preflight_issue_settles_owner_view_and_recovery(monkeypatch):
    """Catch a preflight failure existing only in the mounted view."""

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        failure = VllmPreflightResult(
            generation=token.generation,
            fingerprint=token.fingerprint,
            issues=(VllmIssue("python_unavailable", "python_environment"),),
        )
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
            lambda candidate, generation: failure,
        )

        await screen._run_vllm_preflight_generation(token, draft)

        snapshot = screen._vllm_owner.snapshot()
        assert snapshot.state is VllmReadinessState.NEEDS_ATTENTION
        assert snapshot.issue == VllmIssue(
            "python_unavailable", "python_environment"
        )
        assert snapshot.activity[-1].code == "preflight_failed"
        assert "Needs attention" in str(
            view.query_one("#vllm-readiness-state", Label).renderable
        )
        assert not view.query_one("#vllm-retry-button", Button).disabled

        retry = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        success = VllmPreflightResult(
            generation=retry.generation,
            fingerprint=retry.fingerprint,
            issues=(),
        )
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
            lambda candidate, generation: success,
        )
        await screen._run_vllm_preflight_generation(retry, draft)
        recovered = screen._vllm_owner.snapshot()
        assert recovered.state is VllmReadinessState.READY_TO_START
        assert recovered.issue is None
        assert "Ready to start" in str(
            view.query_one("#vllm-readiness-state", Label).renderable
        )


async def test_vllm_handoff_intents_are_secret_free_exact_and_strict():
    """A looser value type or copied extras would cross the screen boundary."""

    from tldw_chatbook.UI.Navigation.vllm_handoff import (
        VllmConsoleIntent,
        VllmDefaultIntent,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        HandoffValueError,
        PendingHandoffStore,
    )

    target = VllmConnectionTarget(
        provider_key="vllm",
        api_url="http://127.0.0.1:8000/v1/chat/completions",
        model_id="chatbook-vllm",
        runtime_owner="chatbook",
        generation=7,
        credential_source="environment",
    )

    assert VllmConsoleIntent.from_target(target) == VllmConsoleIntent(
        api_url="http://127.0.0.1:8000/v1/chat/completions",
        model_id="chatbook-vllm",
        generation=7,
    )
    assert VllmDefaultIntent.from_target(target) == VllmDefaultIntent(
        api_url="http://127.0.0.1:8000/v1/chat/completions",
        model_id="chatbook-vllm",
        generation=7,
    )
    assert set(VllmConsoleIntent.__slots__) == {"api_url", "model_id", "generation"}
    assert set(VllmDefaultIntent.__slots__) == {"api_url", "model_id", "generation"}

    for intent_type in (VllmConsoleIntent, VllmDefaultIntent):
        for invalid in (
            {"api_url": "http://user:secret@127.0.0.1:8000/v1", "generation": 7},
            {"api_url": "http://127.0.0.1:8000/v1?secret=yes", "generation": 7},
            {"api_url": "http://127.0.0.1:8000/v1#secret", "generation": 7},
            {"model_id": "/private/model", "generation": 7},
            {"generation": True},
        ):
            values = {
                "api_url": "http://127.0.0.1:8000/v1/chat/completions",
                "model_id": "chatbook-vllm",
                "generation": 7,
            }
            values.update(invalid)
            with pytest.raises((TypeError, ValueError)):
                intent_type(**values)

    mutable_extra = type(
        "MutableVllmIntent",
        (),
        {
            "api_url": target.api_url,
            "model_id": target.model_id,
            "generation": target.generation,
            "extras": [],
        },
    )()
    with pytest.raises(HandoffValueError):
        PendingHandoffStore().stage(
            HandoffChannel.VLLM_CONSOLE,
            mutable_extra,
        )


async def test_handoff_buttons_enable_only_for_current_verified_target():
    """Stale readiness must never leave either cross-screen action enabled."""

    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        use = app.query_one("#vllm-use-in-console-button", Button)
        default = app.query_one("#vllm-make-default-button", Button)
        assert use.disabled and default.disabled

        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        owner = VllmConnectionOwner()
        token = owner.begin(draft, runtime_owner="chatbook")
        _bind_local_claim(owner, token)
        assert owner.settle(token, _ready_result(token))
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.READY,
            preflight=None,
            connection=owner.snapshot(),
        )
        assert not use.disabled and not default.disabled

        owner.invalidate("target_changed")
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            connection=owner.snapshot(),
        )
        assert use.disabled and default.disabled


async def test_vllm_handoff_stages_only_current_target_and_uses_normal_navigation(
    monkeypatch,
):
    """A stale target or failed dispatch must not survive as a pending handoff."""

    from tldw_chatbook.Constants import TAB_CHAT, TAB_SETTINGS
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        _bind_local_claim(screen._vllm_owner, token)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        seen: list[NavigateToScreen] = []
        original_post_message = screen.post_message
        monkeypatch.setattr(
            screen,
            "post_message",
            lambda message: seen.append(message) or True,
        )

        screen._on_vllm_use_in_console_requested(
            VllmSetupView.UseInConsoleRequested()
        )
        screen._on_vllm_make_default_requested(VllmSetupView.MakeDefaultRequested())

        assert [(message.screen_name, message.screen_context) for message in seen] == [
            (TAB_CHAT, {}),
            (
                TAB_SETTINGS,
                {"category": "providers-models"},
            ),
        ]
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_DEFAULT)

        screen._vllm_owner.invalidate("target_changed")
        app.pending_handoffs.clear_pending(HandoffChannel.VLLM_CONSOLE)
        screen._on_vllm_use_in_console_requested(
            VllmSetupView.UseInConsoleRequested()
        )
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        monkeypatch.setattr(screen, "post_message", original_post_message)

        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        _bind_local_claim(screen._vllm_owner, token)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        monkeypatch.setattr(screen, "post_message", lambda _message: False)
        screen._on_vllm_use_in_console_requested(
            VllmSetupView.UseInConsoleRequested()
        )
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        monkeypatch.setattr(screen, "post_message", original_post_message)
