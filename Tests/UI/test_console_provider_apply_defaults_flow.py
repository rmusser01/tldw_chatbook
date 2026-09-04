"""Production-hierarchy coverage for Console provider Apply and defaults.

These tests keep Textual's real ``ChatScreen`` composition, the consolidated
production CSS, and the real Console store/controller orchestration.  Provider
execution is never entered; the observable boundary is the detached turn
configuration captured immediately before a send would reach the gateway.
"""

from __future__ import annotations

import asyncio
import threading
import tomllib
from dataclasses import replace

import pytest
from textual.widgets import Button, Input, Select, Static

import tldw_chatbook.Chat.console_settings_defaults as defaults_module
from Tests.console_provider_doubles import provider_resolution
from Tests.UI.background_signals import (
    await_background_task,
    wait_for_background_signal,
    wait_for_signal,
)
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
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatStore,
    ConsoleSettingsComponent,
    ConsoleSettingsPolicyFailureLabel,
)
from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicySnapshot
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
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Console_Modules.left_rail import (
    CONSOLE_DISCARD_DEFAULT_RETRY_ID,
    CONSOLE_DISMISS_DEFAULT_REFRESH_ID,
    CONSOLE_REFRESH_RUNNING_APP_ID,
    CONSOLE_RETRY_DEFAULT_SAVE_ID,
    ConsoleLeftRail,
)
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_settings_summary import (
    ConsoleSettingsSummary,
)
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


class _ControlledSettingsPersistence:
    """Thread-safe settings persistence with one blocked promotion flush."""

    db = None

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.promotion_context_started = asyncio.Event()
        self.release_promotion_context = threading.Event()
        self.promotion_conversation_id: str | None = None

    @staticmethod
    def _committed_policy(candidate) -> ConsoleLibraryPolicySnapshot:
        return ConsoleLibraryPolicySnapshot(
            auto_retrieve=candidate.auto_retrieve,
            assistant_access=candidate.assistant_access,
            policy_revision=1,
            source="durable",
        )

    def create_conversation(self, **_kwargs):
        raise AssertionError("atomic first persistence must be used")

    def persist_console_conversation_with_policy(self, **kwargs):
        return self._committed_policy(kwargs["policy_candidate"])

    def promote_console_conversation_bundle(self, **kwargs):
        self.promotion_conversation_id = kwargs["conversation_id"]
        return self._committed_policy(kwargs["policy_candidate"])

    def update_conversation_context_policy(self, **kwargs):
        if kwargs["conversation_id"] == self.promotion_conversation_id:
            self._loop.call_soon_threadsafe(self.promotion_context_started.set)
            if not self.release_promotion_context.wait(timeout=10):
                raise AssertionError("promotion context flush was not released")
        raise RuntimeError("controlled context persistence failure")


class _RecordingProviderGateway:
    """Stub only provider I/O while retaining production action routing."""

    def __init__(self) -> None:
        self.selections = []

    async def resolve_for_send(self, selection):
        self.selections.append(selection)
        return provider_resolution(base_url="http://127.0.0.1:9099")

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        yield "completed"


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
async def test_existing_chat_action_routes_ignore_later_new_chat_default() -> None:
    """Retry, Continue, and both branch routes keep the chat-owned selection.

    Console has no separate Duplicate-chat or Branch-chat session command in the
    shipping action catalog.  The production duplicate-response route is Retry,
    while Regenerate and Edit-and-resend are the two production sibling-branch
    routes.  Exercising those controller entry points closes the same ownership
    boundary without inventing a test-only session constructor.
    """

    app = _console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        source_settings = ConsoleSessionSettings(
            provider="llama_cpp",
            model="source-owned-model",
            base_url="http://127.0.0.1:9099",
            temperature=0.66,
            source="user",
        )
        controller = console._ensure_console_chat_controller()
        source_session = controller.new_session(settings=source_settings)
        gateway = _RecordingProviderGateway()
        controller.provider_gateway = gateway

        user = store.append_message(
            source_session.id,
            role=ConsoleMessageRole.USER,
            content="Keep this chat's provider.",
        )
        failed = store.append_message(
            source_session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
        )
        store.mark_message_failed(failed.id)

        app.app_config["chat_defaults"] = {
            "provider": "vllm",
            "model": "model-b",
        }
        app.app_config["api_settings"]["vllm"]["model_defaults"] = {
            "model-b": {"temperature": 0.23, "streaming": False}
        }
        app.console_new_chat_default_generation += 1

        retry = await controller.retry_message(failed.id)
        continued = await controller.continue_from_message(failed.id)
        continuation_id = store.active_leaf(source_session.id)
        assert continuation_id is not None
        regenerated = await controller.regenerate_message(continuation_id)
        edited = await controller.edit_and_resend_message(
            user.id,
            "Keep this chat's provider after editing.",
        )

        assert all(
            result.accepted
            for result in (retry, continued, regenerated, edited)
        ), [
            (result.accepted, result.visible_copy)
            for result in (retry, continued, regenerated, edited)
        ]
        assert [
            (selection.provider, selection.explicit_model)
            for selection in gateway.selections
        ] == [("llama_cpp", "source-owned-model")] * 4
        assert store.session_settings(source_session.id) is source_settings
        assert source_session.canonical_settings_baseline is None


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
async def test_promotion_completion_projects_current_session_recovery_after_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed promotion completion projects only the then-current ledger."""

    app = _console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-left-rail")
        store = console._ensure_console_chat_store()
        clean_session = store.switch_session(store.active_session_id)
        persistence = _ControlledSettingsPersistence(asyncio.get_running_loop())
        store.persistence = persistence
        temporary = store.create_session(title="Temporary", ephemeral=True)
        store.set_session_context_policy_overrides(
            temporary.id,
            ConsoleContextPolicyOverrides(
                compaction_mode=ContextCompactionMode.AUTOMATIC,
            ),
        )

        promotion_task = asyncio.create_task(
            console._session._promote_console_temporary_session()
        )
        try:
            await wait_for_background_signal(
                persistence.promotion_context_started,
                promotion_task,
                what="the delayed temporary-chat promotion context write",
            )

            await console._session._activate_native_console_session(clean_session.id)
            store.set_session_context_policy_overrides(
                clean_session.id,
                ConsoleContextPolicyOverrides(
                    compaction_mode=ContextCompactionMode.AUTOMATIC,
                ),
            )
            assert store.persist_session_if_needed(clean_session.id) is not None
            clean_failure = clean_session.settings_persistence_failures[
                ConsoleSettingsComponent.CONTEXT_POLICY
            ]

            context_row = console.query_one("#console-context-recovery-row")
            assert context_row.display is False

            rail = console.query_one("#console-left-rail", ConsoleLeftRail)
            projected = asyncio.Event()
            projection_sessions: list[str | None] = []
            sync_model_recovery = rail.sync_model_recovery

            def observe_recovery_projection(*, session_id, failures, default_state):
                projection_sessions.append(session_id)
                sync_model_recovery(
                    session_id=session_id,
                    failures=failures,
                    default_state=default_state,
                )
                if (
                    session_id == clean_session.id
                    and ConsoleSettingsComponent.CONTEXT_POLICY in failures
                ):
                    projected.set()

            monkeypatch.setattr(
                rail,
                "sync_model_recovery",
                observe_recovery_projection,
            )
        finally:
            persistence.release_promotion_context.set()

        await await_background_task(
            promotion_task,
            what="the delayed temporary-chat promotion",
        )
        await wait_for_signal(
            projected,
            what="the promotion completion recovery projection",
        )

        promotion_failure = temporary.settings_persistence_failures[
            ConsoleSettingsComponent.CONTEXT_POLICY
        ]
        context_row = console.query_one("#console-context-recovery-row")
        context_retry = console.query_one(
            "#console-retry-context-settings", Button
        )
        assert temporary.ephemeral is False
        assert promotion_failure.persisted_conversation_id != (
            clean_failure.persisted_conversation_id
        )
        assert store.active_session_id == clean_session.id
        assert context_row.display is True
        assert context_retry.console_settings_session_id == clean_session.id
        assert context_retry.console_settings_revision == clean_failure.revision
        assert projection_sessions
        assert set(projection_sessions) == {clean_session.id}


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
        if console._current_console_rail_state().conversations_open:
            console._toggle_console_rail_section("conversations", next_open=False)
            await pilot.pause()
        assert await pilot.click(conversations_toggle) is True
        await pilot.pause()
        await console._sync_console_legacy_workspace_context_aliases()
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


def _install_ready_vllm_target(app, *, generation: int = 7):
    from tldw_chatbook.UI.LLM_Management.vllm_connection import (
        VllmActivityEvent,
        VllmConnectionOwner,
        VllmProbeResult,
    )
    from tldw_chatbook.UI.LLM_Management.vllm_setup import (
        VllmConnectionTarget,
        VllmLaunchDraft,
        VllmMode,
        VllmModelSource,
        VllmReadinessState,
    )

    owner = VllmConnectionOwner()
    draft = VllmLaunchDraft(
        mode=VllmMode.EXISTING,
        python_environment="python",
        model_source=VllmModelSource.HUGGING_FACE,
        model_value="",
        existing_server_url="http://127.0.0.1:8000/v1",
    )
    token = None
    for _ in range(generation):
        token = owner.begin(draft, runtime_owner="external")
    assert token is not None
    target = VllmConnectionTarget(
        provider_key="vllm",
        api_url="http://127.0.0.1:8000/v1/chat/completions",
        model_id="chatbook-vllm",
        runtime_owner="external",
        generation=token.generation,
        credential_source="none",
    )
    assert owner.settle(
        token,
        VllmProbeResult(
            token=token,
            state=VllmReadinessState.READY,
            target=target,
            issue=None,
            activity=(VllmActivityEvent("ready", "under_1s"),),
        ),
    )
    app._vllm_connection_owner = owner
    return owner, target


@pytest.mark.asyncio
async def test_vllm_console_handoff_replaces_only_active_session_without_config_write(
    monkeypatch,
) -> None:
    """Calling any durable writer or rebasing from saved vLLM loses this contract."""

    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent
    import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module

    app = _console_app()
    _, target = _install_ready_vllm_target(app)
    app.pending_handoffs.stage(
        HandoffChannel.VLLM_CONSOLE,
        VllmConsoleIntent.from_target(target),
    )

    def forbidden_write(*_args, **_kwargs):
        raise AssertionError("session adoption must not write configuration")

    monkeypatch.setattr(
        chat_screen_module,
        "save_settings_to_cli_config",
        forbidden_write,
    )
    harness = _ConsoleFlowHarness(app)

    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        await pilot.pause(0.3)
        settings = console._session._active_console_session_settings()
        assert settings is not None
        assert (
            settings.provider,
            settings.model,
            settings.base_url,
            settings.source,
        ) == (
            "vllm",
            "chatbook-vllm",
            "http://127.0.0.1:8000/v1/chat/completions",
            "user",
        )
        assert (
            app.app_config["api_settings"]["vllm"]["api_url"]
            == "http://127.0.0.1:9098"
        )
        summary = console._build_console_settings_summary_state()
        assert summary.provider_row == "Provider: vllm"
        assert summary.model_row == "Model: chatbook-vllm (Endpoint not saved)"
        assert "127.0.0.1:8000" in summary.endpoint_row
        assert summary.readiness_label == "Endpoint not saved"
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)


@pytest.mark.asyncio
async def test_vllm_console_handoff_rolls_back_after_post_mutation_sync_failure(
    monkeypatch,
) -> None:
    """A sync exception after store mutation must restore every active projection."""

    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent

    app = _console_app()
    _, target = _install_ready_vllm_target(app)
    harness = _ConsoleFlowHarness(app)

    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        session_store = console._ensure_console_chat_store()
        before = console._session._active_console_session_settings()
        session_id = session_store.active_session_id
        assert before is not None and session_id is not None
        controller = console._ensure_console_chat_controller()
        controller_before = (
            controller.provider,
            controller.model,
            controller.base_url,
        )
        summary_widget = console.query_one(
            "#console-settings-summary",
            ConsoleSettingsSummary,
        )
        original_sync = console._sync_console_settings_summary
        calls = 0

        def sync_then_raise_once() -> None:
            nonlocal calls
            calls += 1
            original_sync()
            if calls == 1:
                raise RuntimeError("controlled post-mutation sync failure")

        monkeypatch.setattr(console, "_sync_console_settings_summary", sync_then_raise_once)
        app.pending_handoffs.stage(
            HandoffChannel.VLLM_CONSOLE,
            VllmConsoleIntent.from_target(target),
        )

        assert console.consume_pending_vllm_console_intent() is False
        assert session_store.active_session_id == session_id
        assert session_store.session_settings(session_id) == before
        assert (
            controller.provider,
            controller.model,
            controller.base_url,
        ) == controller_before
        assert summary_widget.state == console._build_console_settings_summary_state()
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)

        monkeypatch.setattr(console, "_sync_console_settings_summary", original_sync)
        assert console.consume_pending_vllm_console_intent() is True
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)


@pytest.mark.asyncio
async def test_vllm_console_handoff_releases_stale_and_failed_claims_for_replay() -> (
    None
):
    """Dropping the claim on a stale owner or failed replace loses user intent."""

    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent

    app = _console_app()
    owner, target = _install_ready_vllm_target(app)
    harness = _ConsoleFlowHarness(app)

    async with harness.run_test(size=(120, 42)) as pilot:
        console = harness.screen_stack[-1]
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        before = console._session._active_console_session_settings()
        assert before is not None

        intent = VllmConsoleIntent.from_target(target)
        app.pending_handoffs.stage(HandoffChannel.VLLM_CONSOLE, intent)
        owner.invalidate("target_changed")
        assert console.consume_pending_vllm_console_intent() is False
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        assert console._session._active_console_session_settings() == before

        _, fresh_target = _install_ready_vllm_target(app)
        app.pending_handoffs.stage(
            HandoffChannel.VLLM_CONSOLE,
            VllmConsoleIntent.from_target(fresh_target),
        )
        original_replace = console._session._replace_active_console_session_settings
        console._session._replace_active_console_session_settings = (
            lambda _settings: (_ for _ in ()).throw(RuntimeError("replace failed"))
        )
        assert console.consume_pending_vllm_console_intent() is False
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        console._session._replace_active_console_session_settings = original_replace
        assert console.consume_pending_vllm_console_intent() is True
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)

        rollback_settings = replace(
            console._session._active_console_session_settings(),
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
        )
        original_replace(rollback_settings)
        session_store = console._ensure_console_chat_store()
        origin_id = session_store.active_session_id
        assert origin_id is not None
        app.pending_handoffs.stage(
            HandoffChannel.VLLM_CONSOLE,
            VllmConsoleIntent.from_target(fresh_target),
        )

        def replace_then_switch(settings):
            original_replace(settings)
            session_store.create_session(settings=rollback_settings)

        console._session._replace_active_console_session_settings = replace_then_switch
        assert console.consume_pending_vllm_console_intent() is False
        assert session_store.session_settings(origin_id) == rollback_settings
        assert session_store.active_session_id != origin_id
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        console._session._replace_active_console_session_settings = original_replace
        app.pending_handoffs.clear_pending(HandoffChannel.VLLM_CONSOLE)

    app.pending_handoffs.stage(
        HandoffChannel.VLLM_CONSOLE,
        VllmConsoleIntent.from_target(fresh_target),
    )
    assert console.consume_pending_vllm_console_intent() is False
    assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)


class _SettingsFlowHarness(ConsolidatedCSSApp):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        screen = SettingsScreen(self.app_instance)
        screen.apply_navigation_context(
            {"category": SettingsCategoryId.PROVIDERS_MODELS.value}
        )
        await self.push_screen(screen)


@pytest.mark.asyncio
async def test_vllm_default_handoff_stages_settings_and_revert_is_byte_identical() -> (
    None
):
    """Prefill must remain a draft until ordinary Settings Save is chosen."""

    from tldw_chatbook.config import get_cli_config_path
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmDefaultIntent

    app = _console_app()
    _, target = _install_ready_vllm_target(app)
    config_path = get_cli_config_path()
    before = config_path.read_bytes()
    app.pending_handoffs.stage(
        HandoffChannel.VLLM_DEFAULT,
        VllmDefaultIntent.from_target(target),
    )
    harness = _SettingsFlowHarness(app)

    async with harness.run_test(size=(180, 50)) as pilot:
        screen = harness.screen_stack[-1]
        assert isinstance(screen, SettingsScreen)
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        await pilot.pause(0.2)
        draft = screen._provider_draft()
        assert draft is not None
        assert draft.values["provider"] == "vllm"
        assert draft.values["model"] == "chatbook-vllm"
        assert (
            draft.values["endpoint"]
            == "http://127.0.0.1:8000/v1/chat/completions"
        )
        assert screen.query_one("#settings-model-value", Input).value == (
            "chatbook-vllm"
        )
        assert screen.query_one("#settings-provider-endpoint-value", Input).value == (
            "http://127.0.0.1:8000/v1/chat/completions"
        )
        review_copy = str(
            screen.query_one("#settings-provider-save-result", Static).renderable
        )
        assert "Saved endpoint:" in review_copy
        assert "Selected endpoint:" in review_copy
        assert config_path.read_bytes() == before
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_DEFAULT)

        screen._revert_category(SettingsCategoryId.PROVIDERS_MODELS)
        await pilot.pause()
        assert screen._provider_draft() is None
        assert config_path.read_bytes() == before

    assert config_path.read_bytes() == before


@pytest.mark.asyncio
async def test_vllm_default_handoff_persists_only_through_ordinary_settings_save() -> (
    None
):
    """The handoff prefill itself stays unsaved; the existing Save owns durability."""

    from tldw_chatbook.config import get_cli_config_path
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmDefaultIntent

    app = _console_app()
    _, target = _install_ready_vllm_target(app)
    config_path = get_cli_config_path()
    before = config_path.read_bytes()
    app.pending_handoffs.stage(
        HandoffChannel.VLLM_DEFAULT,
        VllmDefaultIntent.from_target(target),
    )
    harness = _SettingsFlowHarness(app)

    async with harness.run_test(size=(180, 50)) as pilot:
        screen = harness.screen_stack[-1]
        assert isinstance(screen, SettingsScreen)
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        await pilot.pause(0.2)
        assert config_path.read_bytes() == before
        assert screen.query_one("#settings-save-category", Button).disabled is False

        await pilot.click("#settings-save-category")
        await pilot.pause()

    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert config_path.read_bytes() != before
    assert saved["chat_defaults"]["provider"] == "vllm"
    assert saved["chat_defaults"]["model"] == "chatbook-vllm"
    assert (
        saved["api_settings"]["vllm"]["api_url"]
        == "http://127.0.0.1:8000/v1/chat/completions"
    )


@pytest.mark.asyncio
async def test_vllm_default_handoff_stale_target_rolls_back_and_replays() -> None:
    """A stale owner must leave both the draft and exact handoff retryable."""

    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmDefaultIntent

    app = _console_app()
    owner, target = _install_ready_vllm_target(app)
    app.pending_handoffs.stage(
        HandoffChannel.VLLM_DEFAULT,
        VllmDefaultIntent.from_target(target),
    )
    owner.invalidate("target_changed")
    harness = _SettingsFlowHarness(app)

    async with harness.run_test(size=(180, 50)) as pilot:
        screen = harness.screen_stack[-1]
        assert isinstance(screen, SettingsScreen)
        await _wait_for_selector(screen, pilot, "#settings-provider-save-result")
        await pilot.pause(0.2)
        assert screen._provider_draft() is None
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_DEFAULT)
