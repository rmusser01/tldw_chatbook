"""TASK-2031 (live-UAT defect): provider/model chips must refresh on Apply.

The session model popover's Apply updated the session and the left-rail
Model summary, but the status chips kept showing the OLD provider/model
until a session/tab switch — the user watches "Provider: Anthropic" while
the run is actually served by the newly-applied provider.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Select, Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
)
from tldw_chatbook.Chat.console_settings_apply import (
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSurface,
    ConsoleSettingsSubmission,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover


class _ConsoleHarness(ConsolidatedCSSApp):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


async def _wait_for(pilot, predicate, what: str, timeout: float = 8.0):
    import time as _t

    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.asyncio
async def test_popover_apply_refreshes_the_provider_chip():
    """Applying new session settings must refresh the provider chip
    without a session switch — the tick's control-bar sync path."""
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "local-model",
            },
            "vllm": {
                "api_url": "http://127.0.0.1:9098",
                "model": "served-model",
            },
        },
        "providers": {
            "llama_cpp": ["local-model"],
            "vLLM": ["served-model"],
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    harness = _ConsoleHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        chat_screen = harness.screen_stack[-1]
        assert isinstance(chat_screen, ChatScreen)
        chat_screen._ensure_console_chat_controller()

        from textual.css.query import NoMatches

        def chip_text() -> str:
            try:
                chip = chat_screen.query_one("#console-provider-chip", Static)
            except NoMatches:
                return ""
            return str(chip.renderable)

        await _wait_for(pilot, chip_text, "initial provider chip")

        store = chat_screen._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        settings = chat_screen._session._ensure_active_console_session_settings()
        next_settings = replace(
            settings,
            provider="vllm",
            model="served-model",
            source="user",
        )
        draft = chat_screen._console_settings_initial_draft(
            next_settings,
            store.session_context_policy_overrides(session_id),
            exposed_fields=QUICK_MODEL_DEFAULT_FIELDS,
        )
        submission = ConsoleSettingsSubmission(
            submission_id="provider-chip-refresh",
            action=ConsoleSettingsAction.APPLY_TO_CHAT,
            surface=ConsoleSettingsSurface.QUICK_POPOVER,
            origin=store.capture_console_settings_origin(session_id),
            draft=draft,
            user_display_name_override=None,
            default_field_mask=frozenset(),
        )
        live_commit = store.commit_console_settings_live(submission)
        chat_screen._apply_console_model_popover_result(
            ConsoleSettingsCommittedSubmission(
                submission=submission,
                live_commit=live_commit,
            )
        )
        # The regular tick calls this; the chip must be fresh WITHOUT a
        # session switch.
        chat_screen._sync_console_control_bar()
        await pilot.pause()

        await _wait_for(
            pilot,
            lambda: "vllm" in chip_text().lower(),
            f"chip to show the applied provider (still: {chip_text()!r})",
        )


@pytest.mark.asyncio
async def test_model_apply_popover_commits_selected_provider_and_model_once() -> None:
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="local-model",
        base_url="http://127.0.0.1:9099",
    )
    origin = ConsoleSettingsOrigin("session-a", None, 0)
    draft = ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        field_drafts=tuple(
            ConsoleSettingsFieldDraft(
                name=name,
                effective_value=getattr(settings, name),
                profile_override=getattr(settings, name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for name in ("temperature", "streaming")
        ),
        model_drafts=(),
        endpoint_draft=None,
    )
    config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings": {
            "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
            "vllm": {"api_url": "http://127.0.0.1:9098"},
        },
    }
    submissions: list[ConsoleSettingsSubmission] = []

    def rebase(state: ConsoleSettingsDraftState, **kwargs) -> ConsoleSettingsDraftState:
        return ConsoleChatController.rebase_console_settings_draft(
            object(), state, **kwargs
        )

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return ConsoleSettingsLiveCommit(
            submission_id=submission.submission_id,
            session_id=origin.session_id,
            persisted_conversation_id=None,
            conversation_binding_revision=0,
            generation_revision=1,
            context_policy_revision=1,
            settings=submission.draft.settings,
            context_policy_overrides=submission.draft.context_policy_overrides,
        )

    results: list[object] = []
    harness = ConsolidatedCSSApp()
    modal = ConsoleModelPopover(
        origin=origin,
        app_config=config,
        initial_draft=draft,
        providers_models={
            "llama_cpp": ["local-model"],
            "vllm": ["served-model"],
        },
        scope_copy="Applies to this conversation",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=rebase,
        live_committer=commit,
        default_readiness_resolver=lambda _provider, _model: (
            ConsoleSettingsReadiness("Ready", "Ready.", True)
        ),
    )
    async with harness.run_test(size=(100, 38)) as pilot:
        await harness.push_screen(modal, callback=results.append)
        modal.query_one("#console-popover-provider", Select).value = "vllm"
        await pilot.pause()
        picker = modal.query_one("#console-popover-model-search")
        picker.set_model_value("served-model")
        picker.post_message(picker.ModelSelected("served-model"))
        await pilot.pause()
        await pilot.click("#console-popover-apply")
        await pilot.pause()

    assert len(submissions) == 1
    assert len(results) == 1
    assert isinstance(results[0], ConsoleSettingsCommittedSubmission)
    assert results[0].live_commit.settings.provider == "vllm"
    assert results[0].live_commit.settings.model == "served-model"
    assert results[0].submission.default_field_mask == frozenset()
    assert QUICK_MODEL_DEFAULT_FIELDS == frozenset({"temperature", "streaming"})


@pytest.mark.asyncio
async def test_model_apply_exact_origin_is_captured_before_catalog_await(
) -> None:
    """A tab switch during catalog loading must not retarget the popover."""
    store = ConsoleChatStore()
    origin_settings = ConsoleSessionSettings(provider="llama_cpp", model="origin")
    origin = store.create_session(settings=origin_settings)
    pushed: list[tuple[object, object]] = []
    context_calls: list[tuple[str, ConsoleSessionSettings, str]] = []

    async def delayed_catalog(*_args, **_kwargs):
        store.create_session(
            settings=ConsoleSessionSettings(provider="vllm", model="background")
        )
        return {"llama_cpp": ["origin"]}

    async def effective_thinking_policy(session_id: str) -> str:
        assert session_id == origin.id
        return "keep"

    def context_state_for_session(
        session_id: str,
        *,
        settings: ConsoleSessionSettings,
        thinking_history_effective_policy: str,
    ) -> None:
        context_calls.append(
            (session_id, settings, thinking_history_effective_policy)
        )

    fake = SimpleNamespace(
        _console_setup_modal_blocking=lambda: False,
        _ensure_console_chat_store=lambda: store,
        _ensure_console_chat_controller=lambda: SimpleNamespace(
            rebase_console_settings_draft=lambda state, **_kwargs: state,
            effective_thinking_history_policy_for_session=(
                effective_thinking_policy
            ),
        ),
        _session=SimpleNamespace(
            _ensure_active_console_session_settings=lambda: store.session_settings(
                origin.id
            )
        ),
        _providers_models_for_console_settings=delayed_catalog,
        _provider_readiness_app_config=lambda: {},
        _console_context_control_state_for_session=context_state_for_session,
        _console_settings_initial_draft=ChatScreen._console_settings_initial_draft,
        _console_default_readiness=lambda _provider, _model: (
            ConsoleSettingsReadiness("Ready", "Ready.", True)
        ),
        _commit_console_settings_submission_live=lambda _submission: None,
        _apply_console_model_popover_result=lambda _result: None,
        app=SimpleNamespace(
            push_screen=lambda modal, callback: pushed.append((modal, callback))
        ),
    )

    await ChatScreen.action_open_console_model_popover(fake)

    assert store.active_session_id != origin.id
    modal, _callback = pushed[0]
    assert isinstance(modal, ConsoleModelPopover)
    assert modal._origin.session_id == origin.id
    assert modal._draft.settings == origin_settings
    assert context_calls == [(origin.id, origin_settings, "keep")]
