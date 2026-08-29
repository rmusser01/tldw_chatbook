"""Mounted contracts for current-conversation context controls."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static

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
    ConsoleSettingsReadiness,
    build_console_settings_readiness,
    build_target_default_console_session_settings,
)
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleEndpointDraft,
    ConsoleModelDraft,
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsTransfer,
)
from tldw_chatbook.Chat.console_settings_defaults import (
    ConsoleDefaultDurabilityState,
    ConsoleDefaultMutationIntent,
    ConsoleDefaultRecoveryAction,
    ConsoleDefaultRecoveryRequest,
    ConsoleDefaultSavePhase,
)
from tldw_chatbook.Widgets.Console.console_context_controls import (
    build_console_context_control_state,
)
from tldw_chatbook.Widgets.Console.console_model_popover import (
    ConsoleModelPopover,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS,
    ConsoleSettingsModal,
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
    default_readiness_resolver=None,
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

    def resolve_default_readiness(
        provider: str,
        model: str | None,
    ) -> ConsoleSettingsReadiness:
        target = build_target_default_console_session_settings(
            config,
            provider,
            model,
        )
        return build_console_settings_readiness(target, app_config=config)

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
        default_readiness_resolver=(
            default_readiness_resolver
            if default_readiness_resolver is not None
            else resolve_default_readiness
        ),
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
async def test_popover_make_default_uses_config_owned_target_readiness() -> None:
    app = _ContextHarness()
    config = {
        "chat_defaults": {"provider": "ollama", "model": "llama3"},
        "api_settings": {
            "ollama": {"api_url": "http://127.0.0.1:11434"},
        },
    }
    settings = ConsoleSessionSettings(
        provider="ollama",
        model="llama3",
        base_url="http://127.0.0.1:22468",
    )
    resolved_targets: list[ConsoleSessionSettings] = []

    def resolve_config_owned_target(
        provider: str,
        model: str | None,
    ) -> ConsoleSettingsReadiness:
        target = build_target_default_console_session_settings(
            config,
            provider,
            model,
        )
        resolved_targets.append(target)
        return build_console_settings_readiness(target, app_config=config)

    async with app.run_test(size=(90, 34)) as pilot:
        await app.push_screen(
            _popover(
                settings=settings,
                providers_models={"ollama": ["llama3"]},
                app_config=config,
                default_readiness_resolver=resolve_config_owned_target,
            )
        )
        await pilot.click("#console-popover-defaults")
        await pilot.pause()

        assert resolved_targets
        assert resolved_targets[-1].base_url == "http://127.0.0.1:11434"
        assert resolved_targets[-1].base_url != settings.base_url
        assert not app.screen.query_one(
            "#console-popover-make-new-chat-default", Button
        ).disabled


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
async def test_popover_custom_model_typing_preserves_mode_and_each_character() -> (
    None
):
    app = _ContextHarness()
    settings = ConsoleSessionSettings(
        provider="ollama",
        model=None,
        base_url="http://127.0.0.1:11434",
    )
    async with app.run_test(size=(100, 38)) as pilot:
        popover = _popover(
            settings=settings,
            providers_models={"ollama": []},
            app_config={
                "api_settings": {
                    "ollama": {"api_url": "http://127.0.0.1:11434"}
                }
            },
        )
        await app.push_screen(popover)
        await pilot.pause()
        await pilot.click("#model-search-picker-custom")
        picker = app.screen.query_one("#console-popover-model-search")
        custom_input = picker.query_one("#model-search-picker-input", Input)

        expected = ""
        for character in "private-model-v1":
            await pilot.press(character)
            await pilot.pause()
            expected += character
            assert picker.custom_mode
            assert custom_input.value == expected
            assert popover._draft.settings.model == expected


def _draft_with_endpoint_intent(
    settings: ConsoleSessionSettings,
) -> tuple[ConsoleSettingsDraftState, ConsoleEndpointDraft]:
    endpoint = ConsoleEndpointDraft(
        value=str(settings.base_url),
        bound_provider_config_key=settings.provider,
        dirty=True,
        checked=True,
    )
    draft = _draft(settings)
    return (
        replace(
            draft,
            model_drafts=(
                ConsoleModelDraft(
                    provider=settings.provider,
                    model=settings.model,
                    settings=settings,
                    field_drafts=draft.field_drafts,
                    endpoint_draft=endpoint,
                ),
            ),
            endpoint_draft=endpoint,
        ),
        endpoint,
    )


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
async def test_popover_quick_commits_strip_initial_endpoint_intent(
    button_id: str,
    expected_action: ConsoleSettingsAction,
) -> None:
    app = _ContextHarness()
    settings = ConsoleSessionSettings(
        provider="ollama",
        model="llama3",
        base_url="http://127.0.0.1:11434",
    )
    draft, _endpoint = _draft_with_endpoint_intent(settings)
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(100, 38)) as pilot:
        popover = _popover(
            settings=settings,
            initial_draft=draft,
            providers_models={"ollama": ["llama3"]},
            app_config={
                "api_settings": {
                    "ollama": {"api_url": "http://127.0.0.1:11434"}
                }
            },
            live_committer=commit,
        )
        await app.push_screen(popover, callback=app.capture)
        if expected_action is not ConsoleSettingsAction.APPLY_TO_CHAT:
            await pilot.click("#console-popover-defaults")
        await pilot.click(button_id)
        await pilot.pause()

    assert len(submissions) == 1
    assert submissions[0].action is expected_action
    assert submissions[0].draft.endpoint_draft is None
    assert submissions[0].draft.settings.base_url == settings.base_url
    assert all(
        remembered.endpoint_draft is None
        for remembered in submissions[0].draft.model_drafts
    )
    assert all(
        remembered.settings.base_url == settings.base_url
        for remembered in submissions[0].draft.model_drafts
    )


@pytest.mark.asyncio
async def test_popover_full_settings_preserves_initial_endpoint_intent() -> None:
    app = _ContextHarness()
    settings = ConsoleSessionSettings(
        provider="ollama",
        model="llama3",
        base_url="http://127.0.0.1:11434",
    )
    draft, endpoint = _draft_with_endpoint_intent(settings)

    async with app.run_test(size=(100, 38)) as pilot:
        await app.push_screen(
            _popover(
                settings=settings,
                initial_draft=draft,
                providers_models={"ollama": ["llama3"]},
                app_config={
                    "api_settings": {
                        "ollama": {"api_url": "http://127.0.0.1:11434"}
                    }
                },
            ),
            callback=app.capture,
        )
        await pilot.click("#console-popover-full-settings")
        await pilot.pause()

    assert isinstance(app.result, ConsoleSettingsTransfer)
    assert app.result.draft.endpoint_draft == endpoint
    assert app.result.draft.endpoint_draft.checked
    assert app.result.draft.settings.base_url == settings.base_url
    assert all(
        remembered.settings.base_url == settings.base_url
        for remembered in app.result.draft.model_drafts
    )


@pytest.mark.parametrize("transfer_to_full_settings", (False, True))
@pytest.mark.asyncio
async def test_popover_rebase_created_endpoint_is_stripped_only_from_quick_commit(
    transfer_to_full_settings: bool,
) -> None:
    app = _ContextHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9100",
    )
    config = {
        "api_settings": {
            "llama_cpp": {"api_url": "http://127.0.0.1:9100"},
            "vllm": {"api_url": "http://127.0.0.1:9200"},
        }
    }
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(100, 38)) as pilot:
        popover = _popover(
            settings=settings,
            providers_models={
                "llama_cpp": ["model-a"],
                "vllm": ["model-b"],
            },
            app_config=config,
            live_committer=commit,
        )
        await app.push_screen(popover, callback=app.capture)
        app.screen.query_one("#console-popover-provider", Select).value = "vllm"
        await pilot.pause()
        picker = app.screen.query_one("#console-popover-model-search")
        picker.set_model_value("model-b")
        picker.post_message(picker.ModelSelected("model-b"))
        await pilot.pause()
        rebased_endpoint = popover._draft.endpoint_draft
        assert rebased_endpoint is not None
        assert rebased_endpoint.value == "http://127.0.0.1:9200"

        await pilot.click(
            "#console-popover-full-settings"
            if transfer_to_full_settings
            else "#console-popover-apply"
        )
        await pilot.pause()

    if transfer_to_full_settings:
        assert submissions == []
        assert isinstance(app.result, ConsoleSettingsTransfer)
        assert app.result.draft.endpoint_draft == rebased_endpoint
        result_draft = app.result.draft
    else:
        assert len(submissions) == 1
        assert submissions[0].draft.endpoint_draft is None
        assert all(
            remembered.endpoint_draft is None
            for remembered in submissions[0].draft.model_drafts
        )
        result_draft = submissions[0].draft
    assert result_draft.settings.base_url == "http://127.0.0.1:9200"
    target_draft = next(
        remembered
        for remembered in result_draft.model_drafts
        if remembered.provider == "vllm" and remembered.model == "model-b"
    )
    assert target_draft.settings.base_url == "http://127.0.0.1:9200"


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
        assert str(save_defaults.label) == "Save as model default"
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

    assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
    assert (
        app.result.live_commit.context_policy_overrides.compaction_mode
        is ContextCompactionMode.AUTOMATIC
    )
    assert app.result.live_commit.context_policy_overrides.custom_budget_tokens == 70_000


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

    assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
    assert (
        app.result.live_commit.context_policy_overrides.compaction_representation
        is ContextCompactionRepresentation.HYBRID
    )


@pytest.mark.asyncio
async def test_provider_default_submission_excludes_context_from_default_mask() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

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
                live_committer=commit,
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

    [submission] = submissions
    assert submission.default_field_mask == FULL_MODEL_DEFAULT_FIELDS
    assert "memory" not in " ".join(submission.default_field_mask).lower()
    assert "prompt" not in " ".join(submission.default_field_mask).lower()
    assert submission.draft.context_policy_overrides.compaction_mode is (
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


def _full_draft(
    settings: ConsoleSessionSettings,
    *,
    context_state=None,
    endpoint_draft: ConsoleEndpointDraft | None = None,
    streaming_override: bool | None = None,
) -> ConsoleSettingsDraftState:
    controls = context_state or _state()
    field_drafts = tuple(
        ConsoleSettingsFieldDraft(
            name=name,
            effective_value=getattr(settings, name),
            profile_override=(
                streaming_override if name == "streaming" else getattr(settings, name)
            ),
            provenance=ConsoleSettingsFieldProvenance.INHERITED,
            dirty=False,
        )
        for name in sorted(FULL_MODEL_DEFAULT_FIELDS)
    )
    return ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=controls.overrides,
        field_drafts=field_drafts,
        model_drafts=(),
        endpoint_draft=endpoint_draft,
    )


def _full_modal(
    *,
    settings: ConsoleSessionSettings | None = None,
    initial_draft: ConsoleSettingsDraftState | None = None,
    transfer: ConsoleSettingsTransfer | None = None,
    app_config=None,
    providers_models=None,
    live_committer=_accept_live_submission,
    draft_rebaser=_rebase_quick_draft,
    default_readiness_resolver=None,
    default_durability_state: ConsoleDefaultDurabilityState | None = None,
    default_recovery_handler=None,
    context_state=None,
) -> ConsoleSettingsModal:
    session_settings = settings or (transfer.draft.settings if transfer else _settings())
    config = app_config or {
        "chat_defaults": {
            "provider": session_settings.provider,
            "model": session_settings.model,
        },
        "api_settings": {session_settings.provider: {}},
    }

    def resolve_default_readiness(
        provider: str,
        model: str | None,
    ) -> ConsoleSettingsReadiness:
        target = build_target_default_console_session_settings(
            config,
            provider,
            model,
        )
        return build_console_settings_readiness(target, app_config=config)

    draft = initial_draft or (transfer.draft if transfer else _full_draft(session_settings))
    return ConsoleSettingsModal(
        settings=session_settings,
        origin=(
            transfer.origin
            if transfer is not None
            else ConsoleSettingsOrigin("session-a", None, 0)
        ),
        initial_draft=draft,
        transfer=transfer,
        app_config=config,
        providers_models=providers_models
        or {session_settings.provider: [session_settings.model]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        context_state=context_state,
        can_save=True,
        draft_rebaser=draft_rebaser,
        live_committer=live_committer,
        default_readiness_resolver=(
            default_readiness_resolver
            if default_readiness_resolver is not None
            else resolve_default_readiness
        ),
        default_durability_state=(
            default_durability_state
            if default_durability_state is not None
            else ConsoleDefaultDurabilityState()
        ),
        default_recovery_handler=default_recovery_handler,
    )


@pytest.mark.asyncio
async def test_full_settings_apply_returns_typed_exact_origin_submission() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(_full_modal(live_committer=commit), callback=app.capture)
        await pilot.pause()
        app.screen.query_one("#console-context-compaction-mode", Select).value = (
            ContextCompactionMode.OFF.value
        )
        await pilot.click("#console-settings-save")
        await pilot.pause()

    assert len(submissions) == 1
    submission = submissions[0]
    assert submission.action is ConsoleSettingsAction.APPLY_TO_CHAT
    assert submission.default_field_mask == frozenset()
    assert submission.origin == ConsoleSettingsOrigin("session-a", None, 0)
    assert (
        submission.draft.context_policy_overrides.compaction_mode
        is ContextCompactionMode.OFF
    )
    assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
    assert app.result.submission is submission


@pytest.mark.parametrize(
    ("button_id", "action", "expects_endpoint"),
    (
        (
            "#console-settings-save-default",
            ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
            True,
        ),
        (
            "#console-settings-make-default",
            ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            True,
        ),
    ),
)
@pytest.mark.asyncio
async def test_full_settings_default_actions_use_full_mask_and_safe_endpoint_opt_in(
    button_id: str,
    action: ConsoleSettingsAction,
    expects_endpoint: bool,
) -> None:
    app = _ContextHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        base_url="http://127.0.0.1:9099",
    )
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            settings=settings,
            app_config={
                "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
                "api_settings": {
                    "llama_cpp": {"api_url": "http://127.0.0.1:9099"}
                },
            },
            live_committer=commit,
        )
        await app.push_screen(modal, callback=app.capture)
        await pilot.pause()
        checkbox = modal.query_one("#console-settings-save-endpoint", Checkbox)
        assert checkbox.disabled

        modal.query_one("#console-settings-base-url", Input).value = (
            "https://secret.example.test:8443/v1?token=hidden#fragment"
        )
        await pilot.pause()
        assert not checkbox.disabled
        assert str(checkbox.label) == (
            "Also save connection: secret.example.test:8443 · Remote/unknown"
        )
        visible = " ".join(
            str(widget.renderable)
            for widget in modal.query(Static)
            if widget.display
        )
        assert "/v1" not in visible
        assert "token=hidden" not in visible
        checkbox.value = True
        await pilot.pause()
        await pilot.click(button_id)
        await pilot.pause()

    [submission] = submissions
    assert submission.action is action
    assert submission.default_field_mask == FULL_MODEL_DEFAULT_FIELDS
    assert (submission.draft.endpoint_draft is not None) is expects_endpoint


@pytest.mark.asyncio
async def test_full_settings_save_model_default_retains_live_endpoint_outside_mask() -> (
    None
):
    app = _ContextHarness()
    settings = replace(
        _settings(),
        base_url="http://127.0.0.1:9099/private",
    )
    endpoint = ConsoleEndpointDraft(
        value="http://127.0.0.1:9099/private",
        bound_provider_config_key="llama_cpp",
        dirty=True,
        checked=True,
    )
    initial = _full_draft(settings, endpoint_draft=endpoint)
    initial = replace(
        initial,
        model_drafts=(
            ConsoleModelDraft(
                provider="llama_cpp",
                model="model-a",
                settings=settings,
                field_drafts=initial.field_drafts,
                endpoint_draft=endpoint,
            ),
        ),
    )
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(initial_draft=initial, live_committer=commit)
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-settings-save-default")
        await pilot.pause()

    [submission] = submissions
    assert submission.draft.endpoint_draft == endpoint
    assert submission.default_field_mask == FULL_MODEL_DEFAULT_FIELDS
    assert "base_url" not in submission.default_field_mask


@pytest.mark.parametrize(
    ("button_id", "action", "expected_mask"),
    (
        (
            "#console-settings-save",
            ConsoleSettingsAction.APPLY_TO_CHAT,
            frozenset(),
        ),
        (
            "#console-settings-save-default",
            ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
            FULL_MODEL_DEFAULT_FIELDS,
        ),
    ),
)
@pytest.mark.asyncio
async def test_full_settings_live_actions_retain_unchecked_deliberate_endpoint(
    button_id: str,
    action: ConsoleSettingsAction,
    expected_mask: frozenset[str],
) -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(live_committer=commit)
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-base-url", Input).value = (
            "http://127.0.0.1:9191/v1"
        )
        await pilot.pause()
        checkbox = modal.query_one("#console-settings-save-endpoint", Checkbox)
        assert not checkbox.disabled
        assert not checkbox.value
        await pilot.click(button_id)
        await pilot.pause()

    [submission] = submissions
    assert submission.action is action
    assert submission.default_field_mask == expected_mask
    assert "base_url" not in submission.default_field_mask
    assert submission.draft.endpoint_draft == ConsoleEndpointDraft(
        value="http://127.0.0.1:9191/v1",
        bound_provider_config_key="llama_cpp",
        dirty=True,
        checked=False,
    )
    assert submission.draft.settings.base_url == "http://127.0.0.1:9191/v1"


@pytest.mark.asyncio
async def test_full_settings_save_default_preserves_unedited_inherited_profile() -> None:
    app = _ContextHarness()
    settings = replace(_settings(), temperature=0.61)
    initial = _full_draft(settings)
    initial = replace(
        initial,
        field_drafts=tuple(
            replace(field, profile_override=None)
            if field.name == "temperature"
            else field
            for field in initial.field_drafts
        ),
    )
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(initial_draft=initial, live_committer=commit)
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-settings-save-default")
        await pilot.pause()

    [submission] = submissions
    temperature = next(
        field for field in submission.draft.field_drafts if field.name == "temperature"
    )
    assert temperature.effective_value == 0.61
    assert temperature.profile_override is None
    assert temperature.provenance is ConsoleSettingsFieldProvenance.INHERITED
    assert not temperature.dirty


@pytest.mark.asyncio
async def test_full_settings_save_default_marks_edited_profile_explicit() -> None:
    app = _ContextHarness()
    settings = replace(_settings(), temperature=0.61)
    initial = _full_draft(settings)
    initial = replace(
        initial,
        field_drafts=tuple(
            replace(field, profile_override=None)
            if field.name == "temperature"
            else field
            for field in initial.field_drafts
        ),
    )
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(initial_draft=initial, live_committer=commit)
        await app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#console-settings-temperature", Input).value = "0.72"
        await pilot.pause()
        await pilot.click("#console-settings-save-default")
        await pilot.pause()

    [submission] = submissions
    temperature = next(
        field for field in submission.draft.field_drafts if field.name == "temperature"
    )
    assert temperature.effective_value == 0.72
    assert temperature.profile_override == 0.72
    assert temperature.provenance is ConsoleSettingsFieldProvenance.EXPLICIT
    assert temperature.dirty


@pytest.mark.asyncio
async def test_full_settings_streaming_cycles_inherit_on_off_and_submits_inherit() -> None:
    app = _ContextHarness()
    settings = replace(_settings(), streaming=False)
    draft = _full_draft(settings, streaming_override=None)
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(120, 48)) as pilot:
        modal = _full_modal(
            settings=settings,
            initial_draft=draft,
            live_committer=commit,
        )
        await app.push_screen(modal, callback=app.capture)
        await pilot.pause()
        toggle = modal.query_one("#console-settings-streaming", Button)
        assert str(toggle.label) == "Inherit"
        toggle.press()
        await pilot.pause()
        assert str(toggle.label) == "On"
        toggle.press()
        await pilot.pause()
        assert str(toggle.label) == "Off"
        toggle.press()
        await pilot.pause()
        assert str(toggle.label) == "Inherit"
        await pilot.click("#console-settings-save-default")
        await pilot.pause()

    [submission] = submissions
    streaming = next(
        field for field in submission.draft.field_drafts if field.name == "streaming"
    )
    assert streaming.profile_override is None
    assert submission.draft.settings.streaming is False


@pytest.mark.asyncio
async def test_full_settings_transfer_and_provider_change_use_shared_rebase_seam() -> None:
    app = _ContextHarness()
    source = replace(
        _settings(),
        temperature=0.31,
        base_url="http://127.0.0.1:9099",
    )
    endpoint = ConsoleEndpointDraft(
        value="http://127.0.0.1:9099",
        bound_provider_config_key="llama_cpp",
        dirty=True,
        checked=True,
    )
    draft = replace(
        _full_draft(source, endpoint_draft=endpoint),
        context_policy_overrides=ConsoleContextPolicyOverrides(
            compaction_mode=ContextCompactionMode.OFF
        ),
    )
    transfer = ConsoleSettingsTransfer(
        ConsoleSettingsOrigin("origin-session", "conversation-1", 7),
        draft,
    )
    calls: list[tuple[str, str | None, frozenset[str]]] = []

    def rebase(state, *, provider, model, app_config, exposed_fields):
        calls.append((provider, model, exposed_fields))
        return replace(
            state,
            settings=replace(
                state.settings,
                provider=provider,
                model=model or "model-b",
                base_url="http://127.0.0.1:9200",
            ),
            endpoint_draft=ConsoleEndpointDraft(
                value="http://127.0.0.1:9200",
                bound_provider_config_key=provider,
                dirty=False,
                checked=False,
            ),
        )

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            transfer=transfer,
            providers_models={
                "llama_cpp": ["model-a"],
                "vllm": ["model-b"],
            },
            app_config={
                "api_settings": {
                    "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
                    "vllm": {"api_url": "http://127.0.0.1:9200"},
                }
            },
            draft_rebaser=rebase,
        )
        await app.push_screen(modal)
        await pilot.pause()
        assert modal._origin == transfer.origin
        assert modal._draft.context_policy_overrides.compaction_mode is ContextCompactionMode.OFF
        assert modal.query_one("#console-settings-temperature", Input).value == "0.31"
        assert modal.query_one("#console-context-compaction-mode", Select).value == (
            ContextCompactionMode.OFF.value
        )

        modal.query_one("#console-settings-provider", Select).value = "vllm"
        await pilot.pause()

        assert calls == [("vllm", None, FULL_MODEL_DEFAULT_FIELDS)]
        assert modal._draft.settings.provider == "vllm"
        assert modal._endpoint_draft.bound_provider_config_key == "vllm"
        assert not modal.query_one("#console-settings-save-endpoint", Checkbox).value


@pytest.mark.asyncio
async def test_full_settings_transfer_context_overrides_win_over_stale_snapshot() -> None:
    app = _ContextHarness()
    stale_context = _state(memory=_memory())
    transferred_draft = replace(
        _full_draft(_settings()),
        context_policy_overrides=ConsoleContextPolicyOverrides(
            compaction_mode=ContextCompactionMode.OFF
        ),
    )
    transfer = ConsoleSettingsTransfer(
        ConsoleSettingsOrigin("origin-session", "conversation-1", 7),
        transferred_draft,
    )

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            transfer=transfer,
            context_state=stale_context,
        )
        await app.push_screen(modal)
        await pilot.pause()

        assert modal.query_one("#console-context-compaction-mode", Select).value == (
            ContextCompactionMode.OFF.value
        )
        assert modal._context_state.active_memory == stale_context.active_memory


@pytest.mark.asyncio
async def test_full_settings_endpoint_draft_restores_across_a_b_a_rebase() -> None:
    app = _ContextHarness()
    settings = replace(
        _settings(),
        base_url="http://127.0.0.1:9099/private",
    )
    endpoint = ConsoleEndpointDraft(
        value="http://127.0.0.1:9099/private",
        bound_provider_config_key="llama_cpp",
        dirty=True,
        checked=True,
    )
    config = {
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "model-a",
            },
            "vllm": {
                "api_url": "http://127.0.0.1:9200",
                "model": "model-b",
            },
        }
    }

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            initial_draft=_full_draft(settings, endpoint_draft=endpoint),
            app_config=config,
            providers_models={
                "llama_cpp": ["model-a"],
                "vllm": ["model-b"],
            },
        )
        await app.push_screen(modal)
        await pilot.pause()
        assert modal.query_one(
            "#console-settings-save-endpoint", Checkbox
        ).value

        modal.query_one("#console-settings-provider", Select).value = "vllm"
        await pilot.pause()
        assert modal._endpoint_draft.bound_provider_config_key == "vllm"
        assert not modal._endpoint_draft.dirty
        assert not modal._endpoint_draft.checked

        modal.query_one("#console-settings-provider", Select).value = "llama_cpp"
        await pilot.pause()
        assert modal._endpoint_draft == endpoint
        assert modal.query_one(
            "#console-settings-save-endpoint", Checkbox
        ).value


@pytest.mark.asyncio
async def test_full_settings_model_change_restores_drafts_across_a_b_a_rebase() -> None:
    app = _ContextHarness()
    config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
        "api_settings": {"llama_cpp": {"model": "model-a"}},
    }

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            app_config=config,
            providers_models={"llama_cpp": ["model-a", "model-b"]},
        )
        await app.push_screen(modal)
        await pilot.pause()
        temperature = modal.query_one("#console-settings-temperature", Input)
        model = modal.query_one("#console-settings-model-select", Select)

        temperature.value = "0.33"
        model.value = "model-b"
        await pilot.pause()
        assert temperature.value == "0.33"

        temperature.value = "0.44"
        model.value = "model-a"
        await pilot.pause()
        assert temperature.value == "0.33"

        model.value = "model-b"
        await pilot.pause()
        assert temperature.value == "0.44"


@pytest.mark.asyncio
async def test_full_settings_custom_model_change_rebases_once_after_edit_settles() -> None:
    app = _ContextHarness()
    calls: list[tuple[str, str | None]] = []

    def rebase(state, *, provider, model, app_config, exposed_fields):
        calls.append((provider, model))
        return _rebase_quick_draft(
            state,
            provider=provider,
            model=model,
            app_config=app_config,
            exposed_fields=exposed_fields,
        )

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(draft_rebaser=rebase)
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-settings-model-custom")
        picker = modal.query_one("#console-settings-model-picker")
        assert picker.custom_mode
        picker.query_one("#model-search-picker-input", Input).value = "model-custom"
        await pilot.pause()
        assert calls == []

        await pilot.pause(CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS + 0.1)
        assert calls == [("llama_cpp", "model-custom")]
        assert picker.custom_mode
        assert picker.value == "model-custom"


@pytest.mark.asyncio
async def test_full_settings_submission_aborts_when_rebase_returns_wrong_target() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def wrong_target(state, **_kwargs):
        return state

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            draft_rebaser=wrong_target,
            live_committer=commit,
        )
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-settings-model-custom")
        picker = modal.query_one("#console-settings-model-picker")
        picker.query_one("#model-search-picker-input", Input).value = "model-b"
        await pilot.pause()

        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert submissions == []
        assert isinstance(app.screen, ConsoleSettingsModal)
        assert "requested provider/model" in str(
            modal.query_one("#console-settings-error", Static).renderable
        )


@pytest.mark.asyncio
async def test_full_settings_debounced_rebase_rejects_wrong_target_before_apply() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []
    calls: list[tuple[str, str | None]] = []

    def wrong_target(state, *, provider, model, **_kwargs):
        calls.append((provider, model))
        return state

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            draft_rebaser=wrong_target,
            live_committer=commit,
        )
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-settings-model-custom")
        picker = modal.query_one("#console-settings-model-picker")
        picker.query_one("#model-search-picker-input", Input).value = "model-b"

        await pilot.pause(CONSOLE_SETTINGS_READINESS_DEBOUNCE_SECONDS + 0.1)
        model_after_debounce = picker.value
        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert submissions == []
        assert calls == [
            ("llama_cpp", "model-b"),
            ("llama_cpp", "model-b"),
        ]
        assert model_after_debounce == "model-b"
        assert picker.value == "model-b"
        assert modal._draft.settings.model == "model-a"
        assert isinstance(app.screen, ConsoleSettingsModal)


@pytest.mark.asyncio
async def test_full_settings_submission_contains_rebase_seam_exception() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def failing_rebase(_state, **_kwargs):
        raise RuntimeError("controller unavailable")

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            draft_rebaser=failing_rebase,
            live_committer=commit,
        )
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-settings-model-custom")
        picker = modal.query_one("#console-settings-model-picker")
        picker.query_one("#model-search-picker-input", Input).value = "model-b"
        await pilot.pause()

        await pilot.click("#console-settings-save")
        await pilot.pause()

        assert submissions == []
        assert isinstance(app.screen, ConsoleSettingsModal)
        assert "could not be rebased" in str(
            modal.query_one("#console-settings-error", Static).renderable
        )


@pytest.mark.asyncio
async def test_full_settings_mismatched_endpoint_binding_cannot_be_saved() -> None:
    app = _ContextHarness()
    settings = replace(_settings(), base_url="http://127.0.0.1:9099")
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    endpoint = ConsoleEndpointDraft(
        value="http://127.0.0.1:9200/secret",
        bound_provider_config_key="vllm",
        dirty=True,
        checked=True,
    )
    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            initial_draft=_full_draft(settings, endpoint_draft=endpoint),
            live_committer=commit,
        )
        await app.push_screen(modal)
        await pilot.pause()
        checkbox = modal.query_one("#console-settings-save-endpoint", Checkbox)
        assert checkbox.disabled
        assert not checkbox.value
        await pilot.click("#console-settings-make-default")
        await pilot.pause()

    [submission] = submissions
    assert submission.draft.endpoint_draft is None


def test_full_settings_has_no_direct_config_writer() -> None:
    source = inspect.getsource(ConsoleSettingsModal)

    assert "save_settings_to_cli_config" not in source


@pytest.mark.parametrize("cancel_action", ("button", "escape"))
@pytest.mark.asyncio
async def test_full_settings_cancel_does_not_apply(cancel_action: str) -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(
            _full_modal(live_committer=commit),
            callback=app.capture,
        )
        await pilot.pause()
        if cancel_action == "button":
            await pilot.click("#console-settings-cancel")
        else:
            await pilot.press("escape")
        await pilot.pause()

    assert submissions == []
    assert app.result is None


@pytest.mark.asyncio
async def test_full_settings_exact_origin_rejection_returns_no_committed_result() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def reject(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        raise ValueError("Chat closed; nothing applied")

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(
            _full_modal(live_committer=reject),
            callback=app.capture,
        )
        await pilot.pause()
        await pilot.click("#console-settings-make-default")
        await pilot.pause()

    assert len(submissions) == 1
    assert app.result is None


@pytest.mark.parametrize(
    ("phase", "button_id", "action", "status_copy"),
    (
        (
            ConsoleDefaultSavePhase.BEFORE_REPLACE,
            "#console-settings-default-retry",
            ConsoleDefaultRecoveryAction.RETRY_SAVE,
            "Not written to disk",
        ),
        (
            ConsoleDefaultSavePhase.BEFORE_REPLACE,
            "#console-settings-default-discard",
            ConsoleDefaultRecoveryAction.DISCARD_RETRY,
            "Not written to disk",
        ),
        (
            ConsoleDefaultSavePhase.CACHE_PUBLICATION,
            "#console-settings-default-refresh",
            ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP,
            "Saved on disk; running app refresh failed",
        ),
        (
            ConsoleDefaultSavePhase.CACHE_PUBLICATION,
            "#console-settings-default-dismiss",
            ConsoleDefaultRecoveryAction.DISMISS_REFRESH,
            "Saved on disk; running app refresh failed",
        ),
    ),
)
@pytest.mark.asyncio
async def test_full_settings_recovery_emits_generation_bound_request_and_refreshes(
    phase: ConsoleDefaultSavePhase,
    button_id: str,
    action: ConsoleDefaultRecoveryAction,
    status_copy: str,
) -> None:
    app = _ContextHarness()
    intent = ConsoleDefaultMutationIntent(
        generation=9,
        action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
        provider_config_key="llama_cpp",
        literal_model_id="model-a",
        field_mask=FULL_MODEL_DEFAULT_FIELDS,
        values={name: None for name in FULL_MODEL_DEFAULT_FIELDS},
        endpoint_patch=None,
    )
    failed = ConsoleDefaultDurabilityState(
        newest_intent_generation=9,
        recovery_intent=intent,
        failure_phase=phase,
    )
    requests: list[ConsoleDefaultRecoveryRequest] = []

    async def recover(request: ConsoleDefaultRecoveryRequest):
        requests.append(request)
        return ConsoleDefaultDurabilityState(newest_intent_generation=9)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            default_durability_state=failed,
            default_recovery_handler=recover,
        )
        await app.push_screen(modal)
        await pilot.pause()
        recovery = modal.query_one("#console-settings-default-recovery")
        summary = str(
            modal.query_one(
                "#console-settings-default-recovery-summary", Static
            ).renderable
        )
        assert recovery.display
        assert status_copy in summary
        assert "Make default for new chats: llama_cpp/model-a" in summary
        await pilot.click(button_id)
        await pilot.pause()
        assert not recovery.display

    assert requests == [
        ConsoleDefaultRecoveryRequest(action, 9)
    ]


@pytest.mark.asyncio
async def test_full_settings_recovery_summary_reports_quick_intent_field_scope() -> None:
    app = _ContextHarness()
    intent = ConsoleDefaultMutationIntent(
        generation=10,
        action=ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
        provider_config_key="llama_cpp",
        literal_model_id="model-a",
        field_mask=QUICK_MODEL_DEFAULT_FIELDS,
        values={"temperature": 0.4, "streaming": True},
        endpoint_patch=None,
    )

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            default_durability_state=ConsoleDefaultDurabilityState(
                newest_intent_generation=10,
                recovery_intent=intent,
                failure_phase=ConsoleDefaultSavePhase.BEFORE_REPLACE,
            ),
        )
        await app.push_screen(modal)
        await pilot.pause()
        summary = str(
            modal.query_one(
                "#console-settings-default-recovery-summary", Static
            ).renderable
        )

    assert "fields: streaming, temperature" in summary
    assert "All supported generation fields shown here" not in summary


@pytest.mark.asyncio
async def test_full_settings_blocked_default_leaves_apply_available() -> None:
    app = _ContextHarness()

    def blocked(_provider: str, _model: str | None) -> ConsoleSettingsReadiness:
        return ConsoleSettingsReadiness(
            label="Blocked",
            detail="API key is missing.",
            native_send_supported=False,
        )

    async with app.run_test(size=(120, 48)) as pilot:
        modal = _full_modal(default_readiness_resolver=blocked)
        await app.push_screen(modal)
        await pilot.pause()
        assert modal.query_one("#console-settings-make-default", Button).disabled
        assert not modal.query_one("#console-settings-save", Button).disabled
        assert "API key is missing" in str(
            modal.query_one(
                "#console-settings-new-chat-default-block", Static
            ).renderable
        )


@pytest.mark.asyncio
async def test_full_settings_model_less_target_disables_both_default_actions() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []
    settings = replace(_settings(), model=None)

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(120, 48)) as pilot:
        modal = _full_modal(
            settings=settings,
            providers_models={"llama_cpp": []},
            live_committer=commit,
        )
        await app.push_screen(modal)
        await pilot.pause()

        assert modal.query_one("#console-settings-save-default", Button).disabled
        assert modal.query_one("#console-settings-make-default", Button).disabled
        assert not modal.query_one("#console-settings-save", Button).disabled
        block = modal.query_one(
            "#console-settings-new-chat-default-block", Static
        )
        assert block.display
        assert "choose a model first" in str(block.renderable)

        await pilot.click("#console-settings-save")
        await pilot.pause()

    [submission] = submissions
    assert submission.action is ConsoleSettingsAction.APPLY_TO_CHAT
    assert submission.draft.settings.model is None


@pytest.mark.asyncio
async def test_full_settings_save_default_rechecks_model_after_validation() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(live_committer=commit)
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-settings-model-custom")
        picker = modal.query_one("#console-settings-model-picker")
        picker.query_one("#model-search-picker-input", Input).value = ""
        save_default = modal.query_one(
            "#console-settings-save-default", Button
        )
        assert modal._current_model_value() is None
        assert not save_default.disabled

        modal._submit(ConsoleSettingsAction.SAVE_MODEL_DEFAULT)
        await pilot.pause()

        assert submissions == []
        assert isinstance(app.screen, ConsoleSettingsModal)
        assert modal._current_model_value() is None
        assert save_default.disabled
        assert "choose a model first" in str(
            modal.query_one(
                "#console-settings-new-chat-default-block", Static
            ).renderable
        )


@pytest.mark.asyncio
async def test_full_settings_make_default_rechecks_custom_model_after_rebase() -> None:
    app = _ContextHarness()
    submissions: list[ConsoleSettingsSubmission] = []

    def readiness(_provider: str, model: str | None) -> ConsoleSettingsReadiness:
        blocked = model == "blocked-custom"
        return ConsoleSettingsReadiness(
            label="Blocked" if blocked else "Ready",
            detail="Custom target is not configured." if blocked else "Ready.",
            native_send_supported=not blocked,
        )

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        submissions.append(submission)
        return _accept_live_submission(submission)

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(
            default_readiness_resolver=readiness,
            live_committer=commit,
        )
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-settings-model-custom")
        picker = modal.query_one("#console-settings-model-picker")
        picker.query_one("#model-search-picker-input", Input).value = "blocked-custom"
        await pilot.pause()
        make_default = modal.query_one("#console-settings-make-default", Button)
        assert not make_default.disabled

        await pilot.click("#console-settings-make-default")
        await pilot.pause()

        assert submissions == []
        assert isinstance(app.screen, ConsoleSettingsModal)
        assert make_default.disabled
        assert "Custom target is not configured" in str(
            modal.query_one(
                "#console-settings-new-chat-default-block", Static
            ).renderable
        )


@pytest.mark.asyncio
async def test_full_settings_context_view_hides_both_default_actions() -> None:
    app = _ContextHarness()

    def blocked(_provider: str, _model: str | None) -> ConsoleSettingsReadiness:
        return ConsoleSettingsReadiness(
            label="Blocked",
            detail="API key is missing.",
            native_send_supported=False,
        )

    async with app.run_test(size=(130, 52)) as pilot:
        modal = _full_modal(default_readiness_resolver=blocked)
        await app.push_screen(modal)
        await pilot.pause()
        save_default = modal.query_one("#console-settings-save-default", Button)
        make_default = modal.query_one("#console-settings-make-default", Button)
        apply = modal.query_one("#console-settings-save", Button)
        readiness = modal.query_one(
            "#console-settings-new-chat-default-block", Static
        )
        assert save_default.display
        assert make_default.display
        assert readiness.display

        await pilot.click("#console-settings-view-context")
        await pilot.pause()
        assert not save_default.display
        assert not make_default.display
        assert not readiness.display
        assert apply.display

        await pilot.click("#console-settings-view-model")
        await pilot.pause()
        assert save_default.display
        assert make_default.display
        assert readiness.display
