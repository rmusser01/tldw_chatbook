"""Production-hierarchy coverage for Console provider Apply and defaults.

These tests keep Textual's real ``ChatScreen`` composition, the consolidated
production CSS, and the real Console store/controller orchestration.  Provider
execution is never entered; the observable boundary is the detached turn
configuration captured immediately before a send would reach the gateway.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import Button, Input, Select, Static

import tldw_chatbook.Chat.console_settings_defaults as defaults_module
from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_context_policy import (
    CompactionFailureBehavior,
    ConsoleContextPolicyOverrides,
    ContextCarryForwardMode,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    hydrate_console_generation_settings,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleSettingsComponent,
    ConsoleSettingsPolicyFailureLabel,
)
from tldw_chatbook.Chat.console_settings_defaults import (
    ConsoleDefaultDurabilityState,
    ConsoleDefaultMutationIntent,
    ConsoleDefaultSavePhase,
    ConsoleEndpointPatch,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    blank_console_session_settings,
    build_target_default_console_session_settings,
)
from tldw_chatbook.Chat.console_settings_apply import (
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Console_Modules.left_rail import (
    CONSOLE_DISCARD_DEFAULT_RETRY_ID,
    CONSOLE_DISMISS_DEFAULT_REFRESH_ID,
    CONSOLE_REFRESH_RUNNING_APP_ID,
    CONSOLE_RETRY_DEFAULT_SAVE_ID,
    ConsoleLeftRail,
)
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.config import load_settings


@pytest.fixture(autouse=True)
def _reset_default_intent_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep process-global default intent fencing independent per journey."""

    monkeypatch.setattr(defaults_module, "_LATEST_INTENT_GENERATION", None)
    monkeypatch.setattr(defaults_module, "_LATEST_INTENT_FINGERPRINT", None)
    monkeypatch.setattr(defaults_module, "_LATEST_INTENT_LIFECYCLE", None)
    monkeypatch.setattr(defaults_module, "_ACTIVE_INTENT_CALLS", set())
    monkeypatch.setattr(defaults_module, "_PENDING_RETRY_STATE", None)


class _ConsoleFlowHarness(ConsolidatedCSSApp):
    """Mount the shipping Console screen beneath the production stylesheet."""

    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _console_app():
    app = _build_test_app(
        config_overrides={
            "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
            "api_settings": {
                "llama_cpp": {
                    "api_url": "http://127.0.0.1:9099",
                    "model": "model-a",
                },
                "vllm": {
                    "api_url": "http://127.0.0.1:9098",
                    "model": "model-b",
                },
            },
        }
    )
    attach_chachanotes_db(app)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.providers_models = {
        "llama_cpp": ["model-a"],
        "vllm": ["model-b"],
    }
    return app


def _persisted_console_app():
    """Build from the sandbox config file the default writer will mutate."""

    adapter = SettingsConfigAdapter()
    assert adapter.save_sections(
        {
            "chat_defaults": {
                "provider": "llama_cpp",
                "model": "model-a",
            },
            "api_settings.llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "model-a",
            },
            "api_settings.vllm": {
                "api_url": "http://127.0.0.1:9098",
                "model": "vendor/model:b",
                "streaming": True,
            },
        }
    )
    app = _build_test_app()
    attach_chachanotes_db(app)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.providers_models = {
        "llama_cpp": ["model-a"],
        "vllm": ["vendor/model:b"],
    }
    return app


async def _open_provider_popover(
    console: ChatScreen,
    harness: _ConsoleFlowHarness,
    pilot,
) -> ConsoleModelPopover:
    await console.action_open_console_model_popover()
    await pilot.pause()
    modal = harness.screen
    assert isinstance(modal, ConsoleModelPopover)
    return modal


async def _select_vllm_model(
    modal: ConsoleModelPopover,
    pilot,
    *,
    model: str = "model-b",
    temperature: str = "0.31",
) -> None:
    modal.query_one("#console-popover-provider", Select).value = "vllm"
    await pilot.pause()
    picker = modal.query_one("#console-popover-model-search")
    picker.set_model_value(model)
    picker.post_message(picker.ModelSelected(model))
    await pilot.pause()
    modal.query_one("#console-popover-temperature", Input).value = temperature
    modal.query_one("#console-popover-compaction-mode", Select).value = (
        ContextCompactionMode.AUTOMATIC.value
    )
    await pilot.pause()


async def _drain_settings_tasks(app) -> None:
    """Wait until the application-lifetime settings owner is quiescent."""

    owner = app.console_settings_durability_owner
    while owner.tasks:
        await asyncio.gather(*tuple(owner.tasks))


def _apply_submission(
    store: ConsoleChatStore,
    session_id: str,
    *,
    settings: ConsoleSessionSettings,
    context_policy: ConsoleContextPolicyOverrides,
    submission_id: str,
) -> ConsoleSettingsSubmission:
    return ConsoleSettingsSubmission(
        submission_id=submission_id,
        action=ConsoleSettingsAction.APPLY_TO_CHAT,
        surface=ConsoleSettingsSurface.QUICK_POPOVER,
        origin=store.capture_console_settings_origin(session_id),
        draft=ConsoleSettingsDraftState(
            settings=settings,
            context_policy_overrides=context_policy,
            field_drafts=tuple(
                ConsoleSettingsFieldDraft(
                    name=name,
                    effective_value=getattr(settings, name),
                    profile_override=getattr(settings, name),
                    provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
                    dirty=True,
                )
                for name in sorted(QUICK_MODEL_DEFAULT_FIELDS)
            ),
            model_drafts=(),
            endpoint_draft=None,
        ),
        user_display_name_override=None,
        default_field_mask=frozenset(),
    )


@pytest.mark.parametrize("activation", ("mouse", "keyboard"))
@pytest.mark.asyncio
async def test_apply_closes_and_changes_only_later_send_context(
    activation: str,
) -> None:
    """Removing the live commit or modal callback makes this journey fail."""

    app = _console_app()
    notifications: list[str] = []
    app.notify = lambda message, **_kwargs: notifications.append(str(message))
    harness = _ConsoleFlowHarness(app)

    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        console._provider_readiness_app_config = lambda: app.app_config
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session = store.switch_session(store.active_session_id)
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(
                provider="llama_cpp",
                model="model-a",
                base_url="http://127.0.0.1:9099",
                temperature=0.8,
            ),
        )
        policy_before = ConsoleContextPolicyOverrides(
            compaction_mode=ContextCompactionMode.ASK,
            trigger_ratio=0.9,
            target_ratio=0.6,
            summary_max_tokens=2_048,
            failure_behavior=CompactionFailureBehavior.OMIT_OLDER_CONTEXT,
            carry_forward_mode=ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE,
        )
        store.set_session_context_policy_overrides(session.id, policy_before)
        captured_before = console._session._build_console_turn_execution_context(
            session.id
        )

        modal = await _open_provider_popover(console, harness, pilot)
        await _select_vllm_model(modal, pilot)
        apply_button = modal.query_one("#console-popover-apply", Button)
        if activation == "mouse":
            await pilot.click(apply_button)
        else:
            apply_button.focus()
            await pilot.press("enter")
        await pilot.pause()

        assert harness.screen is console
        applied = store.session_settings(session.id)
        assert applied is not None
        assert (applied.provider, applied.model, applied.temperature) == (
            "vllm",
            "model-b",
            pytest.approx(0.31),
        )
        assert store.session_context_policy_overrides(session.id) == replace(
            policy_before,
            compaction_mode=ContextCompactionMode.AUTOMATIC,
        )
        captured_after = console._session._build_console_turn_execution_context(
            session.id
        )

        assert (
            captured_before.provider_selection.provider,
            captured_before.provider_selection.explicit_model,
            captured_before.provider_payload_settings["temperature"],
        ) == ("llama_cpp", "model-a", 0.8)
        assert (
            captured_after.provider_selection.provider,
            captured_after.provider_selection.explicit_model,
            captured_after.provider_payload_settings["temperature"],
        ) == ("vllm", "model-b", pytest.approx(0.31))
        assert notifications.count("This chat updated") == 1


@pytest.mark.asyncio
async def test_apply_lifecycle_stages_persists_resumes_and_promotes() -> None:
    """Dropping either durable owner breaks an observable lifecycle assertion."""

    app = _console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        console._provider_readiness_app_config = lambda: app.app_config
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None

        modal = await _open_provider_popover(console, harness, pilot)
        await _select_vllm_model(modal, pilot)
        await pilot.click("#console-popover-apply")
        await pilot.pause()
        await _drain_settings_tasks(app)

        staged = store.switch_session(session_id)
        assert staged.persisted_conversation_id is None
        assert staged.generation_settings_revision > 0
        assert staged.context_policy_revision > 0
        assert staged.generation_durable_snapshot is None
        assert staged.context_policy_durable_revision is None

        conversation_id = store.persist_session_if_needed(session_id)
        assert conversation_id is not None
        persistence = store.persistence
        assert persistence is not None
        generation = persistence.get_conversation_generation_settings(
            conversation_id
        ).snapshot
        policy = persistence.get_conversation_context_policy(conversation_id)
        assert generation is not None
        assert (generation.provider, generation.model, generation.temperature) == (
            "vllm",
            "model-b",
            pytest.approx(0.31),
        )
        assert policy.overrides.compaction_mode is ContextCompactionMode.AUTOMATIC

        conversation = app.chachanotes_db.get_conversation_by_id(conversation_id)
        assert conversation is not None
        hydration = hydrate_console_generation_settings(app.app_config, conversation)
        resumed_store = ConsoleChatStore(persistence=persistence)
        resumed = resumed_store.restore_persisted_session(
            title="Resumed",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=[],
            settings=hydration.settings,
            generation_durable_snapshot=hydration.durable_snapshot,
            generation_metadata_status=hydration.metadata_status,
        )
        assert resumed.settings is not None
        assert (resumed.settings.provider, resumed.settings.model) == (
            "vllm",
            "model-b",
        )
        assert (
            resumed.context_policy_overrides.compaction_mode
            is ContextCompactionMode.AUTOMATIC
        )

        temporary = store.create_session(
            settings=ConsoleSessionSettings(
                provider="llama_cpp",
                model="model-a",
                base_url="http://127.0.0.1:9099",
            ),
            ephemeral=True,
        )
        temporary_policy = ConsoleContextPolicyOverrides(
            compaction_mode=ContextCompactionMode.OFF,
            trigger_ratio=0.88,
            target_ratio=0.55,
        )
        submission = _apply_submission(
            store,
            temporary.id,
            settings=ConsoleSessionSettings(
                provider="vllm",
                model="model-b",
                base_url="http://127.0.0.1:9098",
                temperature=0.27,
                streaming=False,
            ),
            context_policy=temporary_policy,
            submission_id="temporary-apply",
        )
        live_commit = console._commit_console_settings_submission_live(submission)
        console._dispatch_console_settings_submission(
            ConsoleSettingsCommittedSubmission(submission, live_commit)
        )
        await _drain_settings_tasks(app)
        assert temporary.persisted_conversation_id is None
        assert temporary.generation_durable_snapshot is None
        assert temporary.context_policy_durable_revision is None

        promoted_id = store.promote_ephemeral_session(temporary.id)
        assert promoted_id is not None
        promoted_generation = persistence.get_conversation_generation_settings(
            promoted_id
        ).snapshot
        promoted_policy = persistence.get_conversation_context_policy(promoted_id)
        assert promoted_generation is not None
        assert (
            promoted_generation.provider,
            promoted_generation.model,
            promoted_generation.temperature,
            promoted_generation.streaming,
        ) == ("vllm", "model-b", pytest.approx(0.27), False)
        assert promoted_policy.overrides == temporary_policy


@pytest.mark.asyncio
async def test_normal_sync_projects_delayed_first_persist_failure_for_switched_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first-persist failure follows the session when it becomes current."""

    app = _console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-left-rail")
        store = console._ensure_console_chat_store()
        failed_session = store.switch_session(store.active_session_id)
        store.set_session_context_policy_overrides(
            failed_session.id,
            ConsoleContextPolicyOverrides(
                compaction_mode=ContextCompactionMode.AUTOMATIC,
            ),
        )
        persistence = store.persistence
        assert persistence is not None

        def fail_first_persist_context_write(**_kwargs):
            raise RuntimeError("delayed first-persist context failure")

        monkeypatch.setattr(
            persistence,
            "update_conversation_context_policy",
            fail_first_persist_context_write,
        )
        assert store.persist_session_if_needed(failed_session.id) is not None
        failure = failed_session.settings_persistence_failures[
            ConsoleSettingsComponent.CONTEXT_POLICY
        ]

        clean_session = store.create_session(title="Clean session")
        await console._sync_native_console_chat_ui()
        context_row = console.query_one("#console-context-recovery-row")
        assert context_row.display is False

        await console._session._activate_native_console_session(failed_session.id)
        context_retry = console.query_one(
            "#console-retry-context-settings", Button
        )
        assert context_row.display is True
        assert context_retry.console_settings_session_id == failed_session.id
        assert context_retry.console_settings_revision == failure.revision

        await console._session._activate_native_console_session(clean_session.id)
        assert context_row.display is False
        assert context_retry.disabled is True


@pytest.mark.asyncio
async def test_normal_sync_projects_delayed_promotion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promotion ledger failures appear without reopening the settings UI."""

    app = _console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-left-rail")
        store = console._ensure_console_chat_store()
        temporary = store.create_session(title="Temporary", ephemeral=True)
        store.set_session_context_policy_overrides(
            temporary.id,
            ConsoleContextPolicyOverrides(
                compaction_mode=ContextCompactionMode.AUTOMATIC,
            ),
        )
        persistence = store.persistence
        assert persistence is not None

        def fail_promotion_context_write(**_kwargs):
            raise RuntimeError("delayed promotion context failure")

        monkeypatch.setattr(
            persistence,
            "update_conversation_context_policy",
            fail_promotion_context_write,
        )

        # Run the store's atomic promotion synchronously here. The production
        # screen offloads this same call to a thread, but the test database is
        # SQLite ``:memory:`` and therefore connection/thread-local.
        assert store.promote_ephemeral_session(temporary.id) is not None
        await console._sync_native_console_chat_ui()

        failure = temporary.settings_persistence_failures[
            ConsoleSettingsComponent.CONTEXT_POLICY
        ]
        context_row = console.query_one("#console-context-recovery-row")
        context_retry = console.query_one(
            "#console-retry-context-settings", Button
        )
        assert temporary.ephemeral is False
        assert context_row.display is True
        assert context_retry.console_settings_session_id == temporary.id
        assert context_retry.console_settings_revision == failure.revision


@pytest.mark.asyncio
async def test_stale_compaction_retry_cannot_replace_newer_full_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed quick write loses ownership when a later policy is applied."""

    app = _console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        console._provider_readiness_app_config = lambda: app.app_config
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        original_policy = ConsoleContextPolicyOverrides(
            compaction_mode=ContextCompactionMode.ASK,
            trigger_ratio=0.91,
            target_ratio=0.62,
            summary_max_tokens=3_100,
            failure_behavior=CompactionFailureBehavior.OMIT_OLDER_CONTEXT,
            carry_forward_mode=ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE,
        )
        store.set_session_context_policy_overrides(session_id, original_policy)
        assert store.persist_session_if_needed(session_id) is not None
        persistence = store.persistence
        assert persistence is not None
        original_writer = persistence.update_conversation_context_policy

        def fail_context_write(**_kwargs):
            raise RuntimeError("PRIVATE_CONTEXT_WRITE_FAILURE")

        monkeypatch.setattr(
            persistence,
            "update_conversation_context_policy",
            fail_context_write,
        )
        modal = await _open_provider_popover(console, harness, pilot)
        await _select_vllm_model(modal, pilot)
        await pilot.click("#console-popover-apply")
        await pilot.pause()
        await _drain_settings_tasks(app)

        session = store.switch_session(session_id)
        failure = session.settings_persistence_failures[
            ConsoleSettingsComponent.CONTEXT_POLICY
        ]
        assert (
            failure.policy_failure_label
            is ConsoleSettingsPolicyFailureLabel.COMPACTION
        )
        assert failure.context_policy_overrides == replace(
            original_policy,
            compaction_mode=ContextCompactionMode.AUTOMATIC,
        )

        monkeypatch.setattr(
            persistence,
            "update_conversation_context_policy",
            original_writer,
        )
        newer_full_policy = ConsoleContextPolicyOverrides(
            compaction_mode=ContextCompactionMode.OFF,
            trigger_ratio=0.73,
            target_ratio=0.44,
            summary_max_tokens=1_337,
            failure_behavior=CompactionFailureBehavior.STOP_AND_ASK,
            carry_forward_mode=ContextCarryForwardMode.MEMORY_WITH_RECENT_TURNS,
        )
        _session, persisted = store.set_session_context_policy_overrides(
            session_id,
            newer_full_policy,
        )
        assert persisted is True
        assert await store.retry_console_settings_persistence(
            session_id=session_id,
            component=ConsoleSettingsComponent.CONTEXT_POLICY,
            revision=failure.revision,
        ) is False
        assert store.session_context_policy_overrides(session_id) == newer_full_policy


@pytest.mark.asyncio
async def test_default_actions_persist_exact_scope_and_publish_blank_chat_defaults(
) -> None:
    """Profile/global scope drift or missing runtime publication fails here."""

    literal_model = "vendor/model:b"
    app = _persisted_console_app()
    notifications: list[str] = []
    app.notify = lambda message, **_kwargs: notifications.append(str(message))
    harness = _ConsoleFlowHarness(app)

    async with harness.run_test(size=(160, 48)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        origin_id = store.active_session_id
        assert origin_id is not None
        preexisting = store.create_session(
            settings=ConsoleSessionSettings(
                provider="llama_cpp",
                model="preexisting-model",
                base_url="http://127.0.0.1:9099",
                temperature=0.77,
            )
        )
        store.switch_session(origin_id)

        modal = await _open_provider_popover(console, harness, pilot)
        await _select_vllm_model(
            modal,
            pilot,
            model=literal_model,
            temperature="0.42",
        )
        streaming = modal.query_one("#console-popover-streaming", Button)
        assert str(streaming.label) == "Streaming: on"
        streaming.scroll_visible(animate=False, force=True)
        await pilot.pause()
        assert await pilot.click(streaming) is True
        await pilot.pause()
        assert str(streaming.label) == "Streaming: off"
        await pilot.click("#console-popover-defaults")
        await pilot.pause()
        await pilot.click("#console-popover-save-model-default")
        await pilot.pause()
        assert harness.screen is console
        await _drain_settings_tasks(app)

        profile_only_reload = load_settings(force_reload=True)
        profile = profile_only_reload["api_settings"]["vllm"]["model_defaults"][
            literal_model
        ]
        assert profile["temperature"] == pytest.approx(0.42)
        assert profile["streaming"] is False
        assert profile_only_reload["chat_defaults"]["provider"] == "llama_cpp"
        assert profile_only_reload["chat_defaults"]["model"] == "model-a"
        restarted_target = build_target_default_console_session_settings(
            profile_only_reload,
            "vllm",
            literal_model,
        )
        assert restarted_target.temperature == pytest.approx(0.42)
        assert restarted_target.streaming is False

        modal = await _open_provider_popover(console, harness, pilot)
        modal.query_one("#console-popover-temperature", Input).value = "0.23"
        streaming = modal.query_one("#console-popover-streaming", Button)
        assert str(streaming.label) == "Streaming: off"
        streaming.scroll_visible(animate=False, force=True)
        await pilot.pause()
        assert await pilot.click(streaming) is True
        await pilot.pause()
        assert str(streaming.label) == "Streaming: on"
        await pilot.click("#console-popover-defaults")
        await pilot.pause()
        await pilot.click("#console-popover-make-new-chat-default")
        await pilot.pause()
        assert harness.screen is console
        await _drain_settings_tasks(app)

        assert app.console_default_durability_state.failure_phase is None
        assert app.console_new_chat_default_generation == 1, notifications
        assert app.app_config["chat_defaults"]["provider"] == "vllm"
        assert app.app_config["chat_defaults"]["model"] == literal_model
        assert store.session_settings(preexisting.id) == ConsoleSessionSettings(
            provider="llama_cpp",
            model="preexisting-model",
            base_url="http://127.0.0.1:9099",
            temperature=0.77,
        )

        await console._session._create_native_console_session_from_active_context()
        ordinary_id = store.active_session_id
        assert ordinary_id is not None
        ordinary = store.session_settings(ordinary_id)
        assert ordinary is not None
        assert (
            ordinary.provider,
            ordinary.model,
            ordinary.temperature,
            ordinary.streaming,
        ) == ("vllm", literal_model, pytest.approx(0.23), True)

        await console._session._create_native_console_session_from_active_context(
            ephemeral=True
        )
        temporary_id = store.active_session_id
        assert temporary_id is not None
        temporary = store.switch_session(temporary_id)
        assert temporary.ephemeral is True
        assert temporary.settings is not None
        assert (
            temporary.settings.provider,
            temporary.settings.model,
            temporary.settings.temperature,
        ) == ("vllm", literal_model, pytest.approx(0.23))

        conversations_toggle = console.query_one(
            "#console-rail-section-toggle-conversations", Button
        )
        conversations_toggle.scroll_visible(animate=False, force=True)
        await pilot.pause()
        assert await pilot.click(conversations_toggle) is True
        await pilot.pause()
        await _wait_for_selector(
            console,
            pilot,
            "#console-new-workspace-conversation",
        )
        workspace_entry = console.query_one(
            "#console-new-workspace-conversation", Button
        )
        workspace_entry.scroll_visible(animate=False, force=True)
        await pilot.pause()
        before_workspace_entry_id = store.active_session_id
        workspace_entry.focus()
        await pilot.press("enter")
        await pilot.pause()
        workspace_id = store.active_session_id
        assert workspace_id is not None
        assert workspace_id != before_workspace_entry_id
        workspace = store.switch_session(workspace_id)
        assert workspace.workspace_id == "workspace-default"
        assert workspace.settings is not None
        assert (workspace.settings.provider, workspace.settings.model) == (
            "vllm",
            literal_model,
        )

        explicit_source = ConsoleSessionSettings(
            provider="llama_cpp",
            model="source-owned-model",
            base_url="http://127.0.0.1:9099",
            temperature=0.66,
            source="user",
        )
        source_session = console._ensure_console_chat_controller().new_session(
            settings=explicit_source,
            canonical_settings_baseline=explicit_source,
        )
        assert store.session_settings(source_session.id) == explicit_source

    rebooted = load_settings(force_reload=True)
    rebooted_blank = blank_console_session_settings(rebooted)
    assert (
        rebooted_blank.provider,
        rebooted_blank.model,
        rebooted_blank.temperature,
        rebooted_blank.streaming,
    ) == ("vllm", literal_model, pytest.approx(0.23), True)
    assert notifications.count("This chat updated") == 2
    assert f"Model profile default saved: vllm/{literal_model}" in notifications
    assert f"Eligible new-chat default saved: vllm/{literal_model}" in notifications


@pytest.mark.asyncio
async def test_default_failures_render_exact_sanitized_recovery_actions() -> None:
    """App-owned failure phase selects the only valid recovery controls."""

    app = _console_app()
    harness = _ConsoleFlowHarness(app)
    intent = ConsoleDefaultMutationIntent(
        generation=7,
        action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
        provider_config_key="vllm",
        literal_model_id="vendor/private:model",
        field_mask=QUICK_MODEL_DEFAULT_FIELDS,
        values={"temperature": 0.22, "streaming": True},
        endpoint_patch=ConsoleEndpointPatch(
            value="http://192.168.1.9:8000/v1?api_key=never-render",
            bound_provider_config_key="vllm",
            dirty=True,
            checked=True,
        ),
    )

    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        rail = console.query_one(ConsoleLeftRail)

        app.console_default_durability_state = ConsoleDefaultDurabilityState(
            newest_intent_generation=7,
            recovery_intent=intent,
            failure_phase=ConsoleDefaultSavePhase.BEFORE_REPLACE,
        )
        console._sync_console_settings_recovery_surfaces()
        await pilot.pause()

        copy = str(
            rail.query_one("#console-default-recovery-copy", Static).renderable
        )
        assert copy == (
            "Not written to disk · Make default for new chats · "
            "vllm/vendor/private:model · fields: streaming, temperature · "
            "192.168.1.9:8000 · LAN"
        )
        assert "api_key" not in copy and "never-render" not in copy
        retry = rail.query_one(f"#{CONSOLE_RETRY_DEFAULT_SAVE_ID}", Button)
        discard = rail.query_one(f"#{CONSOLE_DISCARD_DEFAULT_RETRY_ID}", Button)
        refresh = rail.query_one(f"#{CONSOLE_REFRESH_RUNNING_APP_ID}", Button)
        dismiss = rail.query_one(f"#{CONSOLE_DISMISS_DEFAULT_REFRESH_ID}", Button)
        assert retry.display and discard.display
        assert not refresh.display and not dismiss.display
        assert retry.console_default_intent_generation == 7
        assert discard.console_default_intent_generation == 7

        app.console_default_durability_state = ConsoleDefaultDurabilityState(
            newest_intent_generation=7,
            recovery_intent=intent,
            failure_phase=ConsoleDefaultSavePhase.CACHE_PUBLICATION,
        )
        console._sync_console_settings_recovery_surfaces()
        await pilot.pause()

        copy = str(
            rail.query_one("#console-default-recovery-copy", Static).renderable
        )
        assert copy.startswith("Saved on disk; running app refresh failed · ")
        assert refresh.display and dismiss.display
        assert not retry.display and not discard.display
        assert refresh.console_default_intent_generation == 7
        assert dismiss.console_default_intent_generation == 7


@pytest.mark.asyncio
async def test_custom_and_unconfigured_selection_stays_literal_and_blocks_default() -> (
    None
):
    """Catalog/config gaps stay visible instead of selecting a fallback."""

    app = _console_app()
    app.app_config.setdefault("api_settings", {})["anthropic"] = {}
    app.providers_models["anthropic"] = []
    harness = _ConsoleFlowHarness(app)

    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        literal_model = "vendor/missing:custom-model"
        store.replace_session_settings(
            session_id,
            ConsoleSessionSettings(
                provider="anthropic",
                model=literal_model,
                temperature=0.41,
            ),
        )

        modal = await _open_provider_popover(console, harness, pilot)
        provider = modal.query_one("#console-popover-provider", Select)
        picker = modal.query_one("#console-popover-model-search")
        assert provider.value == "anthropic"
        assert picker.value == literal_model

        await pilot.click("#console-popover-defaults")
        await pilot.pause()
        make_default = modal.query_one(
            "#console-popover-make-new-chat-default", Button
        )
        block_copy = str(
            modal.query_one(
                "#console-popover-new-chat-default-block", Static
            ).renderable
        )
        assert make_default.disabled is True
        assert "unavailable" in block_copy.lower()
        assert store.session_settings(session_id).model == literal_model
