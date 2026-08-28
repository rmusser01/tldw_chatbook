"""Mounted contracts for current-conversation context controls."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Button, Input, OptionList, Select, Static

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCompactionMode,
    ContextCompactionRepresentation,
)
from tldw_chatbook.Chat.console_context_repository import ConsoleMemoryRecord
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleModelDraft,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsTransfer,
)
from tldw_chatbook.Widgets.Console.console_context_controls import (
    build_console_context_control_state,
)
from tldw_chatbook.Widgets.Console.console_model_popover import (
    ConsoleModelPopover,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleSettingsModal,
    ConsoleSettingsResult,
)


def _settings() -> ConsoleSessionSettings:
    return ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        max_tokens=4_000,
    )


def _memory() -> ConsoleMemoryRecord:
    return ConsoleMemoryRecord(
        memory_id="memory-1",
        conversation_id="conversation-1",
        boundary_message_id="message-4",
        captured_leaf_message_id="message-8",
        lineage_json='["message-1", "message-4", "message-8"]',
        summary_text="The user chose the local-first deployment plan.",
        provider="llama_cpp",
        model="model-a",
        prompt_id="console.rewind_summarize",
        prompt_revision=2,
        prompt_digest="prompt-digest",
        selected_units_json='["message-1", "message-4"]',
        summarized_prefix_digest="prefix-digest",
        input_tokens=12_000,
        output_tokens=700,
        before_tokens=52_000,
        after_tokens=24_000,
        created_at="2026-08-10T20:00:00+00:00",
    )


def _state(*, memory: ConsoleMemoryRecord | None = None):
    return build_console_context_control_state(
        settings=_settings(),
        estimate=ConsoleSettingsContextEstimate(
            used_tokens=42_000,
            token_limit=100_000,
            label="42,000 / 100,000 tokens",
        ),
        overrides=ConsoleContextPolicyOverrides(),
        conversation_tokens=32_000,
        request_overhead_tokens=10_000,
        safety_margin_tokens=2_000,
        active_memory=memory,
    )


def _draft(
    settings: ConsoleSessionSettings,
    *,
    context_state=None,
    temperature_provenance: ConsoleSettingsFieldProvenance = (
        ConsoleSettingsFieldProvenance.INHERITED
    ),
) -> ConsoleSettingsDraftState:
    state = context_state or _state()
    return ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=state.overrides,
        field_drafts=(
            ConsoleSettingsFieldDraft(
                name="temperature",
                effective_value=settings.temperature,
                profile_override=settings.temperature,
                provenance=temperature_provenance,
                dirty=False,
            ),
            ConsoleSettingsFieldDraft(
                name="streaming",
                effective_value=settings.streaming,
                profile_override=settings.streaming,
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            ),
        ),
        model_drafts=(),
        endpoint_draft=None,
    )


def _rebase_quick_draft(
    state: ConsoleSettingsDraftState,
    *,
    provider: str,
    model: str | None,
    app_config,
    exposed_fields: frozenset[str],
) -> ConsoleSettingsDraftState:
    return ConsoleChatController.rebase_console_settings_draft(
        object(),
        state,
        provider=provider,
        model=model,
        app_config=app_config,
        exposed_fields=exposed_fields,
    )


def _accept_live_submission(
    submission: ConsoleSettingsSubmission,
) -> ConsoleSettingsLiveCommit:
    return ConsoleSettingsLiveCommit(
        submission_id=submission.submission_id,
        session_id=submission.origin.session_id,
        persisted_conversation_id=submission.origin.persisted_conversation_id,
        conversation_binding_revision=(
            submission.origin.conversation_binding_revision
        ),
        generation_revision=1,
        context_policy_revision=1,
        settings=submission.draft.settings,
        context_policy_overrides=submission.draft.context_policy_overrides,
    )


def _popover(
    *,
    settings: ConsoleSessionSettings | None = None,
    initial_draft: ConsoleSettingsDraftState | None = None,
    providers_models=None,
    context_state=None,
    app_config=None,
    scope_copy: str = "Applies to this conversation",
    durability_copy: str = "Saved with the conversation after its first message",
    live_committer=_accept_live_submission,
) -> ConsoleModelPopover:
    session_settings = settings or _settings()
    controls = context_state or _state()
    config = (
        app_config
        if app_config is not None
        else {
            "chat_defaults": {
                "provider": session_settings.provider,
                "model": session_settings.model,
            },
            "api_settings": {session_settings.provider: {}},
        }
    )
    return ConsoleModelPopover(
        origin=ConsoleSettingsOrigin("session-a", None, 0),
        app_config=config,
        initial_draft=(
            initial_draft
            if initial_draft is not None
            else _draft(session_settings, context_state=controls)
        ),
        providers_models=(
            providers_models
            if providers_models is not None
            else {session_settings.provider: [session_settings.model]}
        ),
        context_state=controls,
        scope_copy=scope_copy,
        durability_copy=durability_copy,
        draft_rebaser=_rebase_quick_draft,
        live_committer=live_committer,
    )


class _ContextHarness(App[None]):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self) -> None:
        super().__init__()
        self.result = None
        self.capture_count = 0
        self.reset_calls = 0
        self.undo_calls: list[tuple[str, int]] = []
        self.reset_all_calls = 0

    def capture(self, result) -> None:
        self.result = result
        self.capture_count += 1

    def reset_current(self) -> tuple[str, int]:
        self.reset_calls += 1
        return "memory-1", 2

    def undo_current(self, memory_id: str, revision: int) -> bool:
        self.undo_calls.append((memory_id, revision))
        return True

    def reset_all(self) -> int:
        self.reset_all_calls += 1
        return 3


@pytest.mark.asyncio
async def test_quick_popover_separates_request_conversation_and_policy() -> None:
    app = _ContextHarness()
    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            _popover(),
            callback=app.capture,
        )
        assert "~42,000 / 94,000 safe input" in str(
            app.screen.query_one("#console-popover-request-usage", Static).renderable
        )
        assert "~32,000 / 84,000 max tokens" in str(
            app.screen.query_one(
                "#console-popover-conversation-usage", Static
            ).renderable
        )
        assert "4,000 tokens for the next reply" in str(
            app.screen.query_one("#console-popover-response-max", Static).renderable
        )
        assert "Automatic may add one extra model call" in str(
            app.screen.query_one(
                "#console-popover-compaction-help",
                Static,
            ).renderable
        )
        assert not app.screen.query("#console-popover-custom-budget")
        app.screen.query_one(
            "#console-popover-compaction-mode", Select
        ).value = ContextCompactionMode.AUTOMATIC.value
        await pilot.click("#console-popover-apply")
        await pilot.pause()

    assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
    assert app.result.submission.action is ConsoleSettingsAction.APPLY_TO_CHAT
    assert (
        app.result.submission.draft.context_policy_overrides.compaction_mode
        is ContextCompactionMode.AUTOMATIC
    )


@pytest.mark.parametrize(
    "durability_copy",
    (
        "Saved with the conversation after its first message",
        "Temporary until this chat is promoted",
    ),
)
@pytest.mark.asyncio
async def test_popover_main_and_defaults_actions_expose_exact_scope(
    durability_copy: str,
) -> None:
    app = _ContextHarness()
    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            _popover(
                scope_copy="Applies to this conversation",
                durability_copy=durability_copy,
            ),
            callback=app.capture,
        )
        await pilot.pause()

        main_actions = list(app.screen.query("#console-popover-main-actions Button"))
        assert [str(button.label) for button in main_actions] == [
            "Cancel",
            "Full settings…",
            "Defaults…",
            "Apply to this chat",
        ]
        assert str(
            app.screen.query_one("#console-popover-scope", Static).renderable
        ) == "Applies to this conversation"
        assert str(
            app.screen.query_one("#console-popover-durability", Static).renderable
        ) == durability_copy
        assert str(
            app.screen.query_one(
                "#console-popover-temperature-provenance", Static
            ).renderable
        ) == "Inherited"

        await pilot.click("#console-popover-defaults")
        await pilot.pause()

        assert not app.screen.query_one("#console-popover-main-actions").display
        assert app.screen.query_one("#console-popover-default-actions").display
        assert [
            str(button.label)
            for button in app.screen.query("#console-popover-default-actions Button")
        ] == [
            "Save as model default",
            "Make default for new chats",
            "Back",
        ]
        assert "Compaction stays with this chat." in str(
            app.screen.query_one(
                "#console-popover-defaults-compaction-scope", Static
            ).renderable
        )
        assert "Temperature + Streaming" in str(
            app.screen.query_one(
                "#console-popover-save-model-default-copy", Static
            ).renderable
        )
        assert "eligible new chats" in str(
            app.screen.query_one(
                "#console-popover-make-new-chat-default-copy", Static
            ).renderable
        )


@pytest.mark.asyncio
async def test_popover_defaults_disable_blocked_new_chat_provider() -> None:
    app = _ContextHarness()
    settings = ConsoleSessionSettings(provider="openai", model="m")
    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            _popover(
                settings=settings,
                providers_models={"openai": ["m"]},
                app_config={
                    "api_settings": {
                        "openai": {"credential_source": "none"}
                    }
                },
            )
        )
        await pilot.click("#console-popover-defaults")
        await pilot.pause()

        make_default = app.screen.query_one(
            "#console-popover-make-new-chat-default", Button
        )
        reason = str(
            app.screen.query_one(
                "#console-popover-new-chat-default-block", Static
            ).renderable
        )
        assert make_default.disabled
        assert "Unavailable" in reason
        assert "missing api key" in reason.lower()


@pytest.mark.asyncio
async def test_popover_defaults_disable_console_unsupported_provider() -> None:
    app = _ContextHarness()
    settings = ConsoleSessionSettings(provider="local_transformers", model="m")
    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            _popover(
                settings=settings,
                providers_models={"local_transformers": ["m"]},
                app_config={"api_settings": {"local_transformers": {}}},
            )
        )
        await pilot.click("#console-popover-defaults")
        await pilot.pause()

        make_default = app.screen.query_one(
            "#console-popover-make-new-chat-default", Button
        )
        reason = str(
            app.screen.query_one(
                "#console-popover-new-chat-default-block", Static
            ).renderable
        )
        assert make_default.disabled
        assert "Unavailable" in reason
        assert "not available in Console yet" in reason


@pytest.mark.asyncio
async def test_popover_equal_numeric_temperature_text_is_still_an_edit() -> None:
    app = _ContextHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        temperature=0.7,
    )
    popover = _popover(settings=settings)
    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(popover)
        temperature = popover.query_one("#console-popover-temperature", Input)
        initial = next(
            field
            for field in popover._draft.field_drafts
            if field.name == "temperature"
        )
        assert not initial.dirty
        assert str(
            popover.query_one(
                "#console-popover-temperature-provenance", Static
            ).renderable
        ) == "Inherited"

        temperature.value = "0.70"
        await pilot.pause()

        edited = next(
            field
            for field in popover._draft.field_drafts
            if field.name == "temperature"
        )
        assert edited.dirty
        assert edited.provenance is ConsoleSettingsFieldProvenance.EXPLICIT
        assert str(
            popover.query_one(
                "#console-popover-temperature-provenance", Static
            ).renderable
        ) == "Edited"


@pytest.mark.asyncio
async def test_popover_carried_source_uses_explicit_keyed_draft_provenance() -> None:
    app = _ContextHarness()
    current_settings = ConsoleSessionSettings(
        provider="vllm",
        model="model-b",
        temperature=0.2,
    )
    source_settings = replace(
        current_settings,
        provider="llama_cpp",
        model="model-a",
    )
    intermediate_settings = replace(
        current_settings,
        provider="ollama",
        model="model-c",
    )
    explicit = ConsoleSettingsFieldDraft(
        name="temperature",
        effective_value=0.2,
        profile_override=0.2,
        provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
        dirty=True,
    )
    carried = replace(
        explicit,
        provenance=ConsoleSettingsFieldProvenance.CARRIED,
    )
    streaming = ConsoleSettingsFieldDraft(
        name="streaming",
        effective_value=current_settings.streaming,
        profile_override=current_settings.streaming,
        provenance=ConsoleSettingsFieldProvenance.INHERITED,
        dirty=False,
    )
    draft = ConsoleSettingsDraftState(
        settings=current_settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        field_drafts=(carried, streaming),
        model_drafts=(
            ConsoleModelDraft(
                provider="llama_cpp",
                model="model-a",
                settings=source_settings,
                field_drafts=(explicit, streaming),
                endpoint_draft=None,
            ),
            ConsoleModelDraft(
                provider="ollama",
                model="model-c",
                settings=intermediate_settings,
                field_drafts=(carried, streaming),
                endpoint_draft=None,
            ),
        ),
        endpoint_draft=None,
    )
    async with app.run_test(size=(90, 34)):
        popover = _popover(
            settings=current_settings,
            initial_draft=draft,
            providers_models={"vllm": ["model-b"]},
        )
        await app.push_screen(popover)

        assert str(
            popover.query_one(
                "#console-popover-temperature-provenance", Static
            ).renderable
        ) == "Edited — carried from llama_cpp/model-a"


@pytest.mark.parametrize(
    ("button_id", "expected_action"),
    (
        ("#console-popover-apply", ConsoleSettingsAction.APPLY_TO_CHAT),
        (
            "#console-popover-save-model-default",
            ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
        ),
        (
            "#console-popover-make-new-chat-default",
            ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
        ),
    ),
)
@pytest.mark.asyncio
async def test_popover_enter_activates_each_committing_action_once(
    button_id: str,
    expected_action: ConsoleSettingsAction,
) -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(90, 34)) as pilot:
        popover = _popover(live_committer=commit)
        await app.push_screen(popover, callback=app.capture)
        if expected_action is not ConsoleSettingsAction.APPLY_TO_CHAT:
            await pilot.click("#console-popover-defaults")
        button = popover.query_one(button_id, Button)
        button.focus()
        await pilot.press("enter")
        await pilot.pause()

    assert len(submissions) == 1
    assert submissions[0].action is expected_action
    assert app.capture_count == 1
    assert isinstance(app.result, ConsoleSettingsCommittedSubmission)


@pytest.mark.parametrize("activation", ("mouse", "keyboard"))
@pytest.mark.asyncio
async def test_popover_apply_mouse_and_keyboard_share_one_submission(
    activation: str,
) -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            _popover(live_committer=commit),
            callback=app.capture,
        )
        apply_button = app.screen.query_one("#console-popover-apply", Button)
        if activation == "mouse":
            await pilot.click(apply_button)
        else:
            apply_button.focus()
            await pilot.press("enter")
        await pilot.pause()

    assert len(submissions) == 1
    assert app.capture_count == 1
    assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
    assert app.result.submission is submissions[0]
    assert submissions[0].action is ConsoleSettingsAction.APPLY_TO_CHAT
    assert submissions[0].default_field_mask == frozenset()


@pytest.mark.parametrize(
    ("button_id", "expected_action"),
    (
        (
            "#console-popover-save-model-default",
            ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
        ),
        (
            "#console-popover-make-new-chat-default",
            ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
        ),
    ),
)
@pytest.mark.asyncio
async def test_popover_default_actions_submit_the_quick_model_field_mask(
    button_id: str,
    expected_action: ConsoleSettingsAction,
) -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            _popover(live_committer=commit),
            callback=app.capture,
        )
        await pilot.click("#console-popover-defaults")
        await pilot.click(button_id)
        await pilot.pause()

    assert len(submissions) == 1
    assert app.capture_count == 1
    assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
    assert submissions[0].action is expected_action
    assert submissions[0].default_field_mask == frozenset(
        {"temperature", "streaming"}
    )


@pytest.mark.asyncio
async def test_popover_origin_rejection_returns_none_and_reports_exact_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _ContextHarness()
    attempted: list[ConsoleSettingsSubmission] = []
    notifications: list[tuple[str, str]] = []

    def reject(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        attempted.append(submission)
        raise ValueError("Chat closed; nothing applied.")

    popover = _popover(live_committer=reject)
    monkeypatch.setattr(
        popover,
        "notify",
        lambda message, *, severity="information", **_kwargs: notifications.append(
            (str(message), severity)
        ),
    )
    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(popover, callback=app.capture)
        await pilot.click("#console-popover-defaults")
        await pilot.click("#console-popover-make-new-chat-default")
        await pilot.pause()

    assert len(attempted) == 1
    assert attempted[0].action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
    assert app.capture_count == 1
    assert app.result is None
    assert notifications == [("Chat closed; nothing applied", "warning")]


@pytest.mark.parametrize(
    "temperature",
    ("not-a-number", "nan", "inf", "-inf", "2.01", "-0.01"),
)
@pytest.mark.asyncio
async def test_popover_invalid_temperature_stays_open_with_error(
    temperature: str,
) -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(90, 34)) as pilot:
        popover = _popover(live_committer=commit)
        await app.push_screen(popover, callback=app.capture)
        temperature_input = app.screen.query_one(
            "#console-popover-temperature", Input
        )
        temperature_input.value = temperature
        await pilot.click("#console-popover-apply")
        await pilot.pause()

        assert app.screen is popover
        error = app.screen.query_one("#console-popover-error", Static)
        assert error.display
        assert "Temperature" in str(error.renderable)
        assert app.focused is temperature_input

    assert submissions == []
    assert app.capture_count == 0


@pytest.mark.asyncio
async def test_popover_full_settings_transfers_draft_without_committing() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            _popover(live_committer=commit),
            callback=app.capture,
        )
        app.screen.query_one("#console-popover-temperature", Input).value = "0.31"
        await pilot.click("#console-popover-full-settings")
        await pilot.pause()

    assert submissions == []
    assert isinstance(app.result, ConsoleSettingsTransfer)
    assert app.result.origin.session_id == "session-a"
    assert app.result.draft.settings.temperature == pytest.approx(0.31)
    temperature = next(
        field
        for field in app.result.draft.field_drafts
        if field.name == "temperature"
    )
    assert temperature.dirty


@pytest.mark.asyncio
async def test_popover_provider_model_rebase_restores_keyed_drafts_and_provenance() -> (
    None
):
    app = _ContextHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9100",
        temperature=0.2,
        streaming=True,
    )
    config = {
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "model-a",
            "temperature": 0.7,
            "streaming": True,
        },
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9100",
                "model_defaults": {
                    "model-a": {"temperature": 0.2, "streaming": True}
                },
            },
            "vllm": {
                "api_url": "http://127.0.0.1:9200",
                "model_defaults": {
                    "model-b": {"temperature": 0.8, "streaming": False}
                },
            },
        },
    }
    async with app.run_test(size=(100, 38)) as pilot:
        popover = _popover(
            settings=settings,
            providers_models={
                "llama_cpp": ["model-a"],
                "vllm": ["model-b"],
            },
            app_config=config,
        )
        await app.push_screen(popover)
        temperature = app.screen.query_one("#console-popover-temperature", Input)
        assert "Inherited" == str(
            app.screen.query_one(
                "#console-popover-streaming-provenance", Static
            ).renderable
        )
        temperature.value = "0.33"
        await pilot.pause()

        app.screen.query_one(
            "#console-popover-provider", Select
        ).value = "vllm"
        await pilot.pause()
        assert "Inherited" == str(
            app.screen.query_one(
                "#console-popover-streaming-provenance", Static
            ).renderable
        )
        picker = app.screen.query_one("#console-popover-model-search")
        picker.set_model_value("model-b")
        picker.post_message(picker.ModelSelected("model-b"))
        await pilot.pause()

        assert popover._draft.settings.base_url == "http://127.0.0.1:9200"
        assert temperature.value == "0.33"
        assert "Edited — carried from llama_cpp/model-a" == str(
            app.screen.query_one(
                "#console-popover-temperature-provenance", Static
            ).renderable
        )
        assert "Inherited" == str(
            app.screen.query_one(
                "#console-popover-streaming-provenance", Static
            ).renderable
        )

        temperature.value = "0.44"
        await pilot.pause()
        app.screen.query_one(
            "#console-popover-provider", Select
        ).value = "llama_cpp"
        await pilot.pause()
        picker.set_model_value("model-a")
        picker.post_message(picker.ModelSelected("model-a"))
        await pilot.pause()
        assert temperature.value == "0.33"

        app.screen.query_one(
            "#console-popover-provider", Select
        ).value = "vllm"
        await pilot.pause()
        picker.set_model_value("model-b")
        picker.post_message(picker.ModelSelected("model-b"))
        await pilot.pause()
        assert temperature.value == "0.44"


@pytest.mark.asyncio
async def test_full_modal_has_stable_views_and_saves_conversation_policy() -> None:
    app = _ContextHarness()
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=_state(memory=_memory()),
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        assert not app.screen.query_one(
            "#console-settings-provider-model-section"
        ).display
        assert app.screen.query_one("#console-settings-context-view").display
        assert "local-first deployment" in str(
            app.screen.query_one("#console-settings-memory-review", Static).renderable
        )
        save_defaults = app.screen.query_one(
            "#console-settings-save-default",
            Button,
        )
        assert str(save_defaults.label) == "Save model defaults"
        assert save_defaults.display is False
        scope = str(
            app.screen.query_one("#console-settings-scope", Static).renderable
        )
        assert "this conversation" in scope
        assert "F9 Settings > Console behavior" in scope
        context_labels = {
            "#console-context-custom-budget": "Conversation max tokens",
            "#console-context-trigger-percent": "Compact at (%)",
            "#console-context-target-percent": "Reduce conversation to (%)",
            "#console-context-summary-max": "Summary response max",
            "#console-context-failure-behavior": "If compaction fails",
            "#console-context-carry-forward": "Keep after compaction",
            "#console-context-compaction-representation": "Representation",
        }
        for selector, expected in context_labels.items():
            control = app.screen.query_one(selector)
            label = control.parent.query_one(".console-settings-modal-label", Static)
            assert str(label.renderable) == expected
        representation = app.screen.query_one(
            "#console-context-compaction-representation", Select
        )
        representation_options = representation.query_one(OptionList)
        assert representation_options.get_option_at_index(1).disabled
        assert representation_options.get_option_at_index(2).disabled
        assert "vision-capable" in str(
            app.screen.query_one(
                "#console-context-representation-status", Static
            ).renderable
        )

        app.screen.query_one(
            "#console-context-budget-mode", Select
        ).value = ContextBudgetMode.CUSTOM.value
        app.screen.query_one("#console-context-custom-budget").value = "70000"
        app.screen.query_one(
            "#console-context-compaction-mode", Select
        ).value = ContextCompactionMode.AUTOMATIC.value
        await pilot.click("#console-settings-save")
        await pilot.pause()

    assert isinstance(app.result, ConsoleSettingsResult)
    assert (
        app.result.context_policy_overrides.compaction_mode
        is ContextCompactionMode.AUTOMATIC
    )
    assert app.result.context_policy_overrides.custom_budget_tokens == 70_000


@pytest.mark.asyncio
async def test_visual_representation_choices_enable_for_vision_model() -> None:
    app = _ContextHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="gpt-4o",
        max_tokens=4_000,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["gpt-4o"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=build_console_context_control_state(
                    settings=settings,
                    estimate=ConsoleSettingsContextEstimate(
                        42_000, 100_000, "42,000 / 100,000 tokens"
                    ),
                ),
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        representation = app.screen.query_one(
            "#console-context-compaction-representation", Select
        )
        options = representation.query_one(OptionList)
        assert not options.get_option_at_index(1).disabled
        assert not options.get_option_at_index(2).disabled
        representation.value = ContextCompactionRepresentation.HYBRID.value
        await pilot.click("#console-settings-save")
        await pilot.pause()

    assert isinstance(app.result, ConsoleSettingsResult)
    assert (
        app.result.context_policy_overrides.compaction_representation
        is ContextCompactionRepresentation.HYBRID
    )


@pytest.mark.asyncio
async def test_provider_defaults_write_excludes_memory_and_prompt_ownership(
    monkeypatch,
) -> None:
    from tldw_chatbook.Widgets.Console import console_settings_modal as modal_module

    writes: list[dict[str, dict[str, object]]] = []
    monkeypatch.setattr(
        modal_module,
        "save_settings_to_cli_config",
        lambda sections: writes.append(sections) or True,
    )
    app = _ContextHarness()
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=_state(),
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        app.screen.query_one(
            "#console-context-compaction-mode", Select
        ).value = ContextCompactionMode.AUTOMATIC.value
        await pilot.click("#console-settings-view-model")
        await pilot.pause()
        await pilot.click("#console-settings-save-default")
        await pilot.pause()

    assert len(writes) == 1
    assert set(writes[0]) <= {
        "api_settings.llama_cpp",
        "console.provider_defaults.llama_cpp",
        "chat_defaults",
    }
    serialized_keys = " ".join(
        f"{section} {' '.join(values)}" for section, values in writes[0].items()
    ).lower()
    assert "memory" not in serialized_keys
    assert "prompt" not in serialized_keys
    assert app.result.context_policy_overrides.compaction_mode is (
        ContextCompactionMode.AUTOMATIC
    )


@pytest.mark.asyncio
async def test_branch_reset_is_undoable_and_reset_all_is_separately_confirmed() -> None:
    app = _ContextHarness()
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=_state(memory=_memory()),
                can_save=True,
                focus_context=True,
                reset_current_memory=app.reset_current,
                undo_current_memory_reset=app.undo_current,
                reset_all_memories=app.reset_all,
            )
        )
        reset_current = app.screen.query_one("#console-context-reset-current", Button)
        reset_current.press()
        await pilot.pause()
        assert app.reset_calls == 1
        assert app.screen.query_one("#console-context-undo-reset", Button).display
        app.screen.query_one("#console-context-undo-reset", Button).press()
        await pilot.pause()
        assert app.undo_calls == [("memory-1", 2)]

        reset_all = app.screen.query_one("#console-context-reset-all", Button)
        reset_all.press()
        await pilot.pause()
        assert app.reset_all_calls == 0
        status = str(
            app.screen.query_one("#console-context-action-status", Static).renderable
        )
        assert "every branch" in status
        assert "Transcript messages will not change" in status
        app.screen.query_one("#console-context-confirm-reset-all", Button).press()
        await pilot.pause()
        assert app.reset_all_calls == 1
        assert not app.screen.query_one("#console-context-undo-reset", Button).display


def test_context_controls_add_no_forbidden_keybindings() -> None:
    forbidden = {
        "ctrl+c",
        "ctrl+v",
        "ctrl+x",
        "ctrl+s",
        "ctrl+d",
        "ctrl+z",
        "ctrl+a",
        "ctrl+r",
        "ctrl+w",
        "ctrl+p",
        "ctrl+q",
        "f1",
        "f6",
    }
    keys = {
        key
        for binding in (
            *ConsoleModelPopover.BINDINGS,
            *ConsoleSettingsModal.BINDINGS,
        )
        for key in str(
            binding.key if hasattr(binding, "key") else binding[0]
        ).split(",")
    }
    assert keys.isdisjoint(forbidden)


@pytest.mark.asyncio
async def test_context_view_fits_narrow_terminal_and_keeps_focusable_controls() -> None:
    app = _ContextHarness()
    async with app.run_test(size=(72, 24)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=ConsoleSettingsContextEstimate(
                    42_000, 100_000, "42,000 / 100,000 tokens"
                ),
                context_state=_state(),
                can_save=True,
                focus_context=True,
            )
        )
        await pilot.pause()
        modal = app.screen.query_one("#console-settings-modal")
        assert modal.region.x >= 0
        assert modal.region.right <= 72
        assert modal.region.y >= 0
        assert modal.region.bottom <= 24
        budget = app.screen.query_one("#console-context-budget-mode", Select)
        await pilot.pause()
        assert app.focused is budget
        body = app.screen.query_one("#console-settings-body")
        hint = app.screen.query_one("#console-settings-fold-hint", Static)
        assert hint.display, (
            body.virtual_size,
            body.container_size,
            body.max_scroll_y,
        )
        actions = app.screen.query_one("#console-settings-actions")
        assert actions.region.bottom <= 24


@pytest.mark.asyncio
async def test_quick_popover_keeps_actions_visible_and_marks_the_narrow_fold() -> None:
    """Keep the context route discoverable before a new user starts scrolling."""
    app = _ContextHarness()
    async with app.run_test(size=(72, 24)) as pilot:
        await app.push_screen(
            _popover()
        )
        await pilot.pause()
        await pilot.pause()

        hint = app.screen.query_one("#console-popover-fold-hint", Static)
        actions = app.screen.query_one("#console-popover-main-actions")
        context_button = app.screen.query_one(
            "#console-popover-full-settings",
            Button,
        )
        assert hint.display
        assert actions.region.bottom <= 24
        assert context_button.region.bottom <= 24
        focus_order: list[str] = []
        for _ in range(14):
            focused = app.focused
            focus_order.append(getattr(focused, "id", "") or "")
            if focus_order[-1] == "console-popover-apply":
                break
            await pilot.press("tab")
            await pilot.pause()
        assert focus_order.index("console-popover-temperature") < focus_order.index(
            "console-popover-streaming"
        )
        assert focus_order.index("console-popover-streaming") < focus_order.index(
            "console-popover-compaction-mode"
        )
        assert focus_order[-1] == "console-popover-apply"


@pytest.mark.asyncio
async def test_unverified_model_capacity_is_labeled_as_estimated() -> None:
    """Never present the 8,001-token fallback as model-verified capacity."""
    estimate = ConsoleSettingsContextEstimate(
        10,
        8001,
        "10 / 8,001 tokens (estimated; model unverified)",
        token_limit_verified=False,
        token_limit_source="provider fallback",
    )
    state = build_console_context_control_state(
        settings=_settings(),
        estimate=estimate,
    )
    app = _ContextHarness()
    async with app.run_test(size=(100, 34)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=_settings(),
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["model-a"]},
                context_estimate=estimate,
                context_state=state,
                can_save=True,
                focus_context=True,
            )
        )
        await pilot.pause()

        window = str(
            app.screen.query_one("#console-context-model-window", Static).renderable
        )
        status = str(
            app.screen.query_one("#console-context-capacity-status", Static).renderable
        )
        assert "Model window (est.)" in window
        assert "model capacity is unverified" in status
        assert "Providers & Models" in status


@pytest.mark.asyncio
async def test_quick_popover_mounts_with_no_model_selected() -> None:
    """A session with no model opens the popover on the blank model row.

    TASK-16502: on Textual 8.x ``Select.BLANK`` silently resolves to
    ``Widget.BLANK`` (``False``), which is not a legal Select value, so the
    popover crashed at mount with InvalidSelectValueError for any session
    whose settings carry no model.
    """
    app = _ContextHarness()
    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            _popover(
                settings=ConsoleSessionSettings(
                    provider="llama_cpp",
                    model=None,
                    max_tokens=4_000,
                ),
                providers_models={"llama_cpp": ["model-a"]},
            ),
            callback=app.capture,
        )
        await pilot.pause()

        model_select = app.screen.query_one("#console-popover-model", Select)
        assert model_select.value is Select.NULL
