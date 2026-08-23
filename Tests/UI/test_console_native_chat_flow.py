import asyncio
import time
from copy import deepcopy
import inspect
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual import on

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.content import Content
from textual.css.query import NoMatches
from textual.pilot import OutOfBounds
from textual.widgets import Button, Checkbox, Input, Static, TextArea

from Tests.fixtures.required_doubles import exploding_double
from Tests.UI.background_signals import wait_for_background_signal, wait_for_signal
from Tests.UI.console_controller_stubs import stub_image_controller
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)

from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces import ConsoleConversationBrowserInputRow
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleCitationNoticeCode,
    ConsoleCitationPhase,
    ConsoleCitationPresentation,
    ConsoleMessageRole,
    ConsoleRunStatus,
    GenerationVariantMeta,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_image_view import IMAGE_CACHE_MAX_ENTRIES
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionResult,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_ACTIVE_RUN_STATUSES,
    CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
    ChatScreen,
)
import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
import tldw_chatbook.UI.Console_Modules.message as message_module
import tldw_chatbook.UI.Console_Modules.session as session_module
from tldw_chatbook.UI.Console_Modules.character import ConsoleCharacterController
from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.Widgets.Console import (
    ConsoleComposerBar,
    ConsolePromptsModal,
    ConsoleSetupModal,
    ConsoleTranscript,
    ConsoleWorkspaceContextTray,
    ConsoleWorkspaceTree,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor
from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


DUMMY_OPENAI_API_KEY = "DUMMY_OPENAI_API_KEY"
_ASYNC_SETTLE_TIMEOUT = 10.0


def test_checking_citations_is_an_active_console_run_status():
    assert ConsoleRunStatus.CHECKING_CITATIONS in CONSOLE_ACTIVE_RUN_STATUSES


def _configure_openai_missing_api_key(app) -> None:
    """Keep setup-state tests on the API-key recovery path."""
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4o"}
    app.app_config["api_settings"] = {"openai": {"api_key": ""}}


def _configure_native_ready_console(app, model: str = "local-model") -> None:
    """Configure a send-ready Console so the first-run setup modal stays hidden.

    Workbench-interaction tests (rail/tab/composer clicks and focus) need the
    blocking setup modal dismissed; a ready llama.cpp provider satisfies the
    readiness single source that drives it.
    """
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": model}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": model}
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = model


def test_console_store_uses_app_citation_repository_for_matching_database():
    app = _build_test_app()
    db = Mock(name="chachanotes-db")
    repository = SimpleNamespace(db=db)
    app.chachanotes_db = db
    app.citation_trace_repository = repository
    screen = ChatScreen(app)

    store = screen._ensure_console_chat_store()

    assert store.persistence is not None
    assert store.persistence.citation_repository is repository
    assert store.persistence.db is repository.db


def test_native_paste_entry_point_advances_edit_serial_for_stale_apply_guard():
    composer = ConsoleComposerBar()
    before = composer.edit_serial

    composer.insert_pasted_text("ordinary paste")

    assert composer.edit_serial == before + 1
    assert composer.capture_draft_snapshot().segments[0].origin == "paste"


def test_console_store_rejects_mismatched_citation_repository():
    app = _build_test_app()
    db = Mock(name="chachanotes-db")
    repository = SimpleNamespace(db=Mock(name="other-chachanotes-db"))
    app.chachanotes_db = db
    app.citation_trace_repository = repository
    screen = ChatScreen(app)

    store = screen._ensure_console_chat_store()

    assert store.persistence is not None
    assert store.persistence.db is db
    assert store.persistence.citation_repository is None


def test_console_store_uses_app_citation_repository_only_with_database():
    app = _build_test_app()
    app.chachanotes_db = None
    app.citation_trace_repository = SimpleNamespace(db=Mock(name="unused-db"))
    screen = ChatScreen(app)

    store = screen._ensure_console_chat_store()

    assert store.persistence is None


def test_console_workspace_conversation_titles_wrap_instead_of_truncating():
    """Long workspace conversation titles wrap at the rail budget."""
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        wrap_console_conversation_title,
    )

    assert wrap_console_conversation_title("Console UAT Workspace Chat", 20) == (
        "Console UAT",
        "Workspace Chat",
    )


def test_console_workspace_conversation_title_preserves_duplicate_suffix():
    """Duplicate-title disambiguators should remain visible in rail labels."""
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        wrap_console_conversation_title,
    )

    title = "Chat [deadbeef]"
    assert ConsoleWorkspaceContextTray._conversation_title(title) == title
    assert wrap_console_conversation_title(title, 20) == (title,)


def test_console_workspace_status_row_empty_value_uses_unavailable():
    """Status labels ending in a colon should not repeat the label as the value."""
    assert ConsoleWorkspaceDetailsTray._split_status_row(
        "Authority: ", "Authority"
    ) == (
        "Authority",
        "unavailable",
    )


def test_console_workspace_conversation_search_worker_uses_dedicated_group():
    # TASK-15454 moved the `run_worker` call out of the `Input.Changed`
    # handler and into the debounced callback the handler now arms -- the
    # DB work in front of the timer was running once per keystroke. The
    # contract this test pins (the search runs in its own exclusive worker
    # group, so a newer search cancels an in-flight one) is unchanged; only
    # the function that expresses it moved.
    source = inspect.getsource(
        ConsoleWorkspaceController._start_console_conversation_browser_search
    )

    assert 'group="console-workspace-conversation-search"' in source
    assert "exclusive=True" in source

    transition = inspect.getsource(ConsoleWorkspaceController.transition_browser_search)
    assert "_start_console_conversation_browser_search" in transition
    assert "_schedule_console_browser_timer(" in transition


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_clear_button_stops_pending_timer():
    app = _build_test_app()
    console = ChatScreen(app)
    timer = Mock()
    sync_calls: list[None] = []
    focus_calls: list[None] = []
    console._sync_console_workspace_context = lambda: sync_calls.append(None)
    console._focus_console_workspace_conversation_search = lambda: focus_calls.append(
        None
    )
    console.call_after_refresh = lambda callback: callback()
    console._console_conversation_browser_query = "alpha"
    console._console_conversation_browser_search_token = 6
    console._console_conversation_browser_search_timer = timer
    console._console_conversation_browser_rows = (
        ConsoleConversationBrowserInputRow(
            row_key="conv-alpha",
            conversation_id="conv-alpha",
            native_session_id=None,
            title="Alpha",
            scope_type="workspace",
            workspace_id=DEFAULT_WORKSPACE_ID,
            workspace_label="Chats",
            status="saved",
            selected=False,
            source_kind="persisted",
        ),
    )
    console._console_conversation_browser_total = 3
    console._console_conversation_browser_error = "old"
    event = Button.Pressed(Button(id="console-workspace-conversation-search-clear"))

    await console.on_button_pressed(event)

    timer.stop.assert_called_once_with()
    assert console._console_conversation_browser_search_timer is None
    assert console._console_conversation_browser_search_token == 7
    assert console._console_conversation_browser_query == ""
    assert console._console_conversation_browser_rows == ()
    assert console._console_conversation_browser_total is None
    assert console._console_conversation_browser_error == ""
    assert sync_calls == [None]
    assert focus_calls == [None]


def test_console_workspace_conversation_search_selection_refresh_invalidates_token():
    source = inspect.getsource(
        ConsoleWorkspaceController._refresh_console_conversation_browser_after_selection
    )
    active_query_branch = source.split("if not query.strip():", 1)[1]
    before_refresh = active_query_branch.split(
        "await self._refresh_console_conversation_browser_search",
        1,
    )[0]

    assert "_console_conversation_browser_search_token += 1" in before_refresh


class _ReadyResolutionGateway:
    async def resolve_for_send(self, selection):
        return SimpleNamespace(
            provider=selection.provider,
            base_url=selection.base_url or "",
            model=selection.explicit_model
            or selection.configured_model
            or "test-model",
            ready=True,
            visible_copy="",
        )


class _PromptImprovementGateway:
    """Return one strict rewrite using the request's protected projection."""

    def __init__(self) -> None:
        self.auxiliary_calls = 0
        self.stream_calls = 0

    async def resolve_for_send(self, selection):
        return ConsoleProviderResolution(
            provider="llama_cpp",
            base_url=selection.base_url or "http://127.0.0.1:9099",
            model=selection.explicit_model
            or selection.configured_model
            or "local-model",
            ready=True,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
        )

    async def complete_auxiliary(self, request):
        self.auxiliary_calls += 1
        payload = json.loads(str(request.messages[-1]["content"]))
        rewritten = str(payload["source_prompt"]).replace("Draft", "Improved", 1)
        return AuxiliaryCompletionResult(
            provider=request.resolution.provider,
            model=str(request.resolution.model),
            text=json.dumps({"kind": "prompt_rewrite", "rewritten_prompt": rewritten}),
        )

    async def stream_chat(self, *_args, **_kwargs):
        self.stream_calls += 1
        raise AssertionError("Prompt improvement must not use normal Console send")


class _HoldingPromptImprovementGateway(_PromptImprovementGateway):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete_auxiliary(self, request):
        self.started.set()
        await self.release.wait()
        return await super().complete_auxiliary(request)


class _MutableResolutionPromptGateway(_PromptImprovementGateway):
    """Expose config-only endpoint drift without changing the selection."""

    def __init__(self) -> None:
        super().__init__()
        self.endpoint = "http://127.0.0.1:9099"
        self.resolve_calls: list[ConsoleProviderResolution] = []

    async def resolve_for_send(self, selection):
        resolution = ConsoleProviderResolution(
            provider="llama_cpp",
            base_url=self.endpoint,
            model=(
                selection.explicit_model or selection.configured_model or "local-model"
            ),
            ready=True,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
        )
        self.resolve_calls.append(resolution)
        return resolution


def _native_prompt_record(
    *,
    artifact_type: str,
    identifier: str,
    version: int = 4,
    source_id: str | None = None,
) -> dict[str, object]:
    kind = "block_recipe" if artifact_type == "recipe" else "block_prompt"
    record: dict[str, object] = {
        "id": identifier,
        "name": "Saved structured artifact",
        "artifact_type": artifact_type,
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": {
            "kind": kind,
            "schema_version": 2,
            "lanes": [
                {
                    "id": "system",
                    "blocks": [
                        {
                            "id": "role",
                            "title": "Role",
                            "syntax": "markdown",
                            "content": "Be exact.",
                        }
                    ],
                },
                {
                    "id": "user",
                    "blocks": [
                        {
                            "id": "goal",
                            "title": "Goal",
                            "syntax": "freeform",
                            "content": "Answer the question.",
                        }
                    ],
                },
            ],
        },
        "system_prompt": "# Role\n\nBe exact.",
        "user_prompt": "Answer the question.",
        "version": version,
        "backend": "local",
    }
    if source_id is not None:
        record["source_id"] = source_id
    return record


class _NativePromptScopeService:
    def __init__(self, record: dict[str, object]) -> None:
        self.record = record
        self.detail_calls: list[tuple[str, str]] = []
        self.usage_calls: list[tuple[str, str]] = []
        self.usage_error: Exception | None = None

    async def get_capabilities(self, *, mode: str):
        return SimpleNamespace(
            structured_kinds=frozenset({(2, "block_prompt"), (2, "block_recipe")}),
            artifact_types=frozenset({"prompt", "recipe"}),
            conditional_update=True,
        )

    async def list_prompts(self, *, mode: str, page: int, per_page: int):
        return {
            "items": [deepcopy(self.record)],
            "page": page,
            "per_page": per_page,
            "total_items": 1,
            "total_pages": 1,
        }

    async def search_prompts(self, *, mode: str, query: str, limit: int):
        return [deepcopy(self.record)]

    async def get_prompt(self, *, mode: str, prompt_identifier: str):
        self.detail_calls.append((mode, prompt_identifier))
        return deepcopy(self.record)

    async def save_prompt(self, *, mode: str, **payload):
        return payload

    async def record_prompt_usage(self, *, mode: str, prompt_identifier: str):
        self.usage_calls.append((mode, prompt_identifier))
        if self.usage_error is not None:
            raise self.usage_error
        return deepcopy(self.record)


class _CollidingRecipeSourceService:
    """Expose one raw Recipe ID from both sources with a held Local detail."""

    def __init__(self) -> None:
        source_id = "shared-recipe"
        local = _native_prompt_record(
            artifact_type="recipe",
            identifier=f"local:prompt:{source_id}",
            source_id=source_id,
            version=4,
        )
        server = _native_prompt_record(
            artifact_type="recipe",
            identifier=f"server:prompt:{source_id}",
            source_id=source_id,
            version=9,
        )
        server["backend"] = "server"
        server["user_prompt"] = "Answer from the server Recipe."
        definition = server["prompt_definition"]
        assert isinstance(definition, dict)
        lanes = definition["lanes"]
        assert isinstance(lanes, list)
        lanes[1]["blocks"][0]["content"] = "Answer from the server Recipe."
        self.records = {"local": local, "server": server}
        self.local_detail_started = asyncio.Event()
        self.release_late_local_detail = asyncio.Event()
        self.detail_calls: list[tuple[str, str]] = []

    async def get_capabilities(self, *, mode: str):
        return SimpleNamespace(
            structured_kinds=frozenset({(2, "block_prompt"), (2, "block_recipe")}),
            artifact_types=frozenset({"prompt", "recipe"}),
            conditional_update=True,
        )

    async def list_prompts(self, *, mode: str, page: int, per_page: int):
        return {
            "items": [deepcopy(self.records[mode])],
            "page": page,
            "per_page": per_page,
            "total_items": 1,
            "total_pages": 1,
        }

    async def search_prompts(self, *, mode: str, query: str, limit: int):
        return [deepcopy(self.records[mode])]

    async def get_prompt(self, *, mode: str, prompt_identifier: str):
        self.detail_calls.append((mode, prompt_identifier))
        if mode == "local" and self.detail_calls.count((mode, prompt_identifier)) == 1:
            self.local_detail_started.set()
            await self.release_late_local_detail.wait()
        return deepcopy(self.records[mode])

    async def save_prompt(self, *, mode: str, **payload):
        return payload


@pytest.mark.asyncio
async def test_prompt_auto_improvement_applies_once_and_menu_undo_restores_exact_draft():
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _PromptImprovementGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft ")
        composer.insert_file_segment("PRIVATE INLINE BYTES", "secret.txt · 20 B")
        composer.insert_text(" answer")
        before = composer.capture_draft_snapshot()
        store = console._ensure_console_chat_store()
        attachment = PendingAttachment(
            file_path="/tmp/photo.png",
            display_name="photo.png",
            file_type="image",
            insert_mode="attachment",
            data=b"\x89PNG-staged",
            mime_type="image/png",
            original_size=11,
            processed_size=11,
        )
        attachment_state = vars(attachment).copy()
        store.set_pending_attachment(store.active_session_id, attachment)
        composer.set_pending_attachment_label(attachment.label)
        transcript_before = tuple(store.messages_for_session(store.active_session_id))

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsolePromptsModal)
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-auto-improve", Button).press()
        for _ in range(12):
            await pilot.pause()
            if host.screen_stack[-1] is console:
                break

        assert composer.draft_text() == "Improved PRIVATE INLINE BYTES answer"
        assert composer.improvement_undo_available
        assert gateway.auxiliary_calls == 1
        assert gateway.stream_calls == 0
        assert (
            tuple(store.messages_for_session(store.active_session_id))
            == transcript_before
        )
        assert store.pending_attachment(store.active_session_id) is attachment
        assert vars(attachment) == attachment_state
        assert composer._pending_attachment_label == attachment.label

        console._handle_console_composer_menu_choice("undo-prompt-improvement")
        assert composer.capture_draft_snapshot().segments == before.segments
        assert composer.draft_text() == "Draft PRIVATE INLINE BYTES answer"
        assert not composer.improvement_undo_available
        assert store.pending_attachment(store.active_session_id) is attachment
        assert vars(attachment) == attachment_state
        assert composer._pending_attachment_label == attachment.label


@pytest.mark.asyncio
async def test_prompt_library_projection_collision_keeps_manual_recipe_available() -> (
    None
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _MutableResolutionPromptGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft [[TLDW_PROTECTED:literal-collision]] ")
        composer.insert_file_segment("PRIVATE INLINE BYTES", "secret.txt · 20 B")

        console._open_console_prompts_modal()
        await pilot.pause()
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsolePromptsModal)
        assert modal.state.mode == "browse"
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()

        assert modal.query_one("#console-prompts-auto-improve", Button).disabled
        assert modal.query_one("#console-prompts-review-improve", Button).disabled
        assert not modal.query_one(
            "#console-prompts-structured-recipe", Button
        ).disabled
        recovery = str(
            modal.query_one("#console-prompts-improvement-status", Static).renderable
        ) or str(modal.query_one("#console-prompts-auto-improve", Button).tooltip)
        assert "protected" in recovery.lower()
        modal_text = _visible_text(modal)
        assert "PRIVATE INLINE BYTES" not in modal_text
        assert "secret.txt" not in modal_text

        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-blank", Button).press()
        await pilot.pause()
        assert modal.query_one(PromptBlockEditor)
        assert gateway.auxiliary_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["selection", "config_endpoint"])
async def test_improvement_disclosure_is_pinned_and_drift_before_click_is_blocked(
    drift: str,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _MutableResolutionPromptGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft answer")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        summary = str(
            modal.query_one("#console-prompts-provider-summary", Static).renderable
        )
        assert "llama_cpp" in summary
        assert "local-model" in summary
        assert "http://127.0.0.1:9099" in summary
        assert len(gateway.resolve_calls) == 1

        if drift == "selection":
            settings = store.switch_session(session_id).settings
            assert settings is not None
            store.replace_session_settings(
                session_id,
                replace(settings, model="changed-model"),
            )
        else:
            gateway.endpoint = "http://127.0.0.1:9191"

        modal.query_one("#console-prompts-auto-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert gateway.auxiliary_calls == 0
        assert host.screen_stack[-1] is modal
        assert (
            "changed"
            in str(
                modal.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            ).lower()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact_type", ["prompt", "recipe"])
async def test_manual_apply_compares_captured_and_fresh_effective_resolution(
    artifact_type: str,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _MutableResolutionPromptGateway()
    app.console_provider_gateway_factory = lambda: gateway
    identifier = f"{artifact_type}-1"
    service = _NativePromptScopeService(
        _native_prompt_record(artifact_type=artifact_type, identifier=identifier)
    )
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft answer")
        before = composer.capture_draft_snapshot()

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        if artifact_type == "recipe":
            modal.query_one("#console-prompts-improve", Button).press()
            await pilot.pause()
            modal.query_one("#console-prompts-structured-recipe", Button).press()
            await pilot.pause()
            modal.query_one("#console-prompts-recipe-saved", Button).press()
            await pilot.pause()
        modal.query_one(f"#console-prompts-result-{identifier}", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert len(gateway.resolve_calls) == 1

        gateway.endpoint = "http://127.0.0.1:9191"
        modal.query_one("#prompt-editor-apply", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert host.screen_stack[-1] is modal
        assert composer.capture_draft_snapshot() == before
        assert gateway.auxiliary_calls == 0
        assert len(gateway.resolve_calls) == 2
        assert (
            "endpoint changed"
            in str(
                modal.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            ).lower()
        )
        assert service.usage_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["session", "draft", "system", "provider"])
async def test_auto_improvement_live_drift_is_reviewable_and_never_partially_applies(
    drift: str,
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _HoldingPromptImprovementGateway()
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft answer")
        store = console._ensure_console_chat_store()
        original_session_id = store.active_session_id
        transcript_before = tuple(store.messages_for_session(original_session_id))
        original_settings = store.switch_session(original_session_id).settings

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-auto-improve", Button).press()
        await wait_for_signal(
            gateway.started,
            what="the auto-improvement gateway call starting",
        )

        if drift == "session":
            store.create_session(settings=original_settings)
        elif drift == "draft":
            composer.insert_text(" manual edit")
        elif drift == "system":
            store.set_session_system_prompt(original_session_id, "Changed system")
        else:
            assert original_settings is not None
            store.replace_session_settings(
                original_session_id,
                replace(
                    original_settings,
                    model="changed-model",
                    base_url="http://127.0.0.1:9191",
                ),
            )
        expected_draft = composer.capture_draft_snapshot()
        expected_system = store.switch_session(
            original_session_id
        ).settings.system_prompt
        if drift == "session":
            store.create_session(settings=original_settings)
        gateway.release.set()
        for _ in range(12):
            await pilot.pause()
            if modal.query("#console-prompts-review-user"):
                break

        assert host.screen_stack[-1] is modal
        candidate = modal.query_one("#console-prompts-review-user", TextArea)
        assert "Improved answer" in candidate.text
        assert composer.capture_draft_snapshot() == expected_draft
        assert (
            store.switch_session(original_session_id).settings.system_prompt
            == expected_system
        )
        assert (
            tuple(store.messages_for_session(original_session_id)) == transcript_before
        )
        assert gateway.stream_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["id", "version", "fingerprint"])
async def test_saved_recipe_identity_drift_never_applies_or_records_usage(drift: str):
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _PromptImprovementGateway()
    app.console_provider_gateway_factory = lambda: gateway
    service = _NativePromptScopeService(
        _native_prompt_record(artifact_type="recipe", identifier="recipe-1")
    )
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft answer")
        before = composer.capture_draft_snapshot()
        store = console._ensure_console_chat_store()
        transcript_before = tuple(store.messages_for_session(store.active_session_id))

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-saved", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-result-recipe-1", Button).press()
        await pilot.pause()

        if drift == "id":
            service.record["id"] = "recipe-replaced"
        elif drift == "version":
            service.record["version"] = 5
        else:
            definition = service.record["prompt_definition"]
            assert isinstance(definition, dict)
            lanes = definition["lanes"]
            assert isinstance(lanes, list)
            lanes[1]["blocks"][0]["content"] = "Changed recipe content."

        modal.query_one("#prompt-editor-apply", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert host.screen_stack[-1] is modal
        assert composer.capture_draft_snapshot() == before
        assert (
            "changed"
            in str(
                modal.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            ).lower()
        )
        assert service.usage_calls == []
        assert (
            tuple(store.messages_for_session(store.active_session_id))
            == transcript_before
        )
        assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_saved_recipe_apply_creates_unsaved_prompt_copy_with_zero_recipe_usage():
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _PromptImprovementGateway()
    app.console_provider_gateway_factory = lambda: gateway
    service = _NativePromptScopeService(
        _native_prompt_record(artifact_type="recipe", identifier="recipe-1")
    )
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft answer")
        store = console._ensure_console_chat_store()
        original_system = store.switch_session(
            store.active_session_id
        ).settings.system_prompt
        transcript_before = tuple(store.messages_for_session(store.active_session_id))

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-saved", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-result-recipe-1", Button).press()
        await pilot.pause()
        modal.query_one("#prompt-editor-apply", Button).press()
        for _ in range(8):
            await pilot.pause()
            if host.screen_stack[-1] is console:
                break

        assert composer.draft_text() == "Answer the question."
        assert (
            store.switch_session(store.active_session_id).settings.system_prompt
            == original_system
        )
        assert service.usage_calls == []
        assert (
            tuple(store.messages_for_session(store.active_session_id))
            == transcript_before
        )
        assert gateway.stream_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("usage_fails", [False, True])
async def test_saved_prompt_apply_records_usage_without_rolling_back_on_usage_failure(
    usage_fails: bool,
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _PromptImprovementGateway()
    app.console_provider_gateway_factory = lambda: gateway
    service = _NativePromptScopeService(
        _native_prompt_record(artifact_type="prompt", identifier="prompt-1")
    )
    if usage_fails:
        service.usage_error = RuntimeError("usage endpoint unavailable")
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft answer")
        store = console._ensure_console_chat_store()
        transcript_before = tuple(store.messages_for_session(store.active_session_id))

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        modal.query_one("#console-prompts-result-prompt-1", Button).press()
        await pilot.pause()
        apply_button = modal.query_one("#prompt-editor-apply", Button)
        assert apply_button.disabled is False
        apply_button.press()
        for _ in range(8):
            await pilot.pause()
            if host.screen_stack[-1] is console:
                break

        assert composer.draft_text() == "Answer the question."
        assert service.usage_calls == [("local", "prompt-1")]
        assert (
            tuple(store.messages_for_session(store.active_session_id))
            == transcript_before
        )
        assert gateway.stream_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact_type", ["prompt", "recipe"])
async def test_normalized_saved_artifact_validation_and_usage_use_source_id(
    artifact_type: str,
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _PromptImprovementGateway()
    app.console_provider_gateway_factory = lambda: gateway
    source_id = f"{artifact_type}-source"
    service = _NativePromptScopeService(
        _native_prompt_record(
            artifact_type=artifact_type,
            identifier=f"local:prompt:{source_id}",
            source_id=source_id,
        )
    )
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft answer")

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        if artifact_type == "recipe":
            modal.query_one("#console-prompts-improve", Button).press()
            await pilot.pause()
            modal.query_one("#console-prompts-structured-recipe", Button).press()
            await pilot.pause()
            modal.query_one("#console-prompts-recipe-saved", Button).press()
            await pilot.pause()
        modal.query_one(".console-prompts-result", Button).press()
        await pilot.pause()
        await pilot.pause()

        modal.query_one("#prompt-editor-apply", Button).press()
        for _ in range(8):
            await pilot.pause()
            if host.screen_stack[-1] is console:
                break

        assert host.screen_stack[-1] is console
        assert composer.draft_text() == "Answer the question."
        assert service.detail_calls == [
            ("local", source_id),
            ("local", source_id),
        ]
        assert service.usage_calls == (
            [("local", source_id)] if artifact_type == "prompt" else []
        )
        assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_saved_recipe_source_is_not_redirected_by_late_colliding_detail() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _PromptImprovementGateway()
    app.console_provider_gateway_factory = lambda: gateway
    service = _CollidingRecipeSourceService()
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft answer")

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsolePromptsModal)
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-saved", Button).press()
        await pilot.pause()

        late_local = asyncio.create_task(modal.open_artifact("shared-recipe"))
        await wait_for_background_signal(
            service.local_detail_started,
            late_local,
            what="the late local detail open",
        )
        await modal.switch_source("server")
        await modal.open_artifact("shared-recipe")
        service.release_late_local_detail.set()
        await late_local
        await pilot.pause()

        modal.query_one("#prompt-editor-apply", Button).press()
        for _ in range(8):
            await pilot.pause()
            if host.screen_stack[-1] is console:
                break

        assert host.screen_stack[-1] is console
        assert composer.draft_text() == "Answer from the server Recipe."
        assert service.detail_calls == [
            ("local", "shared-recipe"),
            ("server", "shared-recipe"),
            ("server", "shared-recipe"),
        ]
        assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_edited_saved_prompt_applies_as_unsaved_copy_with_zero_usage():
    app = _build_test_app()
    _configure_native_ready_console(app)
    gateway = _PromptImprovementGateway()
    app.console_provider_gateway_factory = lambda: gateway
    service = _NativePromptScopeService(
        _native_prompt_record(artifact_type="prompt", identifier="prompt-1")
    )
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Draft answer")

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        modal.query_one("#console-prompts-result-prompt-1", Button).press()
        await pilot.pause()
        editor = modal.query_one(PromptBlockEditor)
        await editor._change_field("goal", "content", "Edited unsaved answer.")
        await pilot.pause()
        modal.query_one("#prompt-editor-apply", Button).press()
        for _ in range(8):
            await pilot.pause()
            if host.screen_stack[-1] is console:
                break

        assert composer.draft_text() == "Edited unsaved answer."
        assert service.usage_calls == []
        assert gateway.stream_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(140, 40), (100, 30), (80, 24)])
async def test_prompt_improvement_settled_native_layout_keeps_shell_and_footer_visible(
    size: tuple[int, int],
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = _PromptImprovementGateway
    host = ConsoleHarness(app)

    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        store.set_session_system_prompt(store.active_session_id, "Be accurate.")
        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        await pilot.pause()

        shell = modal.query_one("#console-prompts-modal")
        footer = modal.query_one("#console-prompts-footer")
        assert 0 <= shell.region.x
        assert 0 <= shell.region.y
        assert shell.region.x + shell.region.width <= size[0]
        assert shell.region.y + shell.region.height <= size[1]
        assert shell.region.contains_region(footer.region)
        assert footer.region.height > 0
        assert modal.query_one("#console-prompts-current-system", TextArea).read_only
        assert modal.query_one("#console-prompts-current-user", TextArea).read_only
        include_system = modal.query_one("#console-prompts-include-system", Checkbox)
        assert str(include_system.label) == "Include system prompt as analysis context"
        assert include_system.value is True
        assert [
            str(modal.query_one(selector, Button).label)
            for selector in (
                "#console-prompts-auto-improve",
                "#console-prompts-review-improve",
                "#console-prompts-structured-recipe",
            )
        ] == [
            "Analyze and auto-improve",
            "Analyze and user review",
            "Create or follow a structured recipe",
        ]
        assert not console.query("#console-control-prompts")


@pytest.mark.asyncio
async def test_recipe_system_persistence_failure_keeps_modal_and_retry_only_saves():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = _PromptImprovementGateway
    host = ConsoleHarness(app)

    class _FailOncePersistence:
        def __init__(self) -> None:
            self.calls = 0

        def update_conversation_system_prompt(self, **_kwargs) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated persistence failure")

    persistence = _FailOncePersistence()

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("Keep this draft")
        before = composer.capture_draft_snapshot()
        store = console._ensure_console_chat_store()
        session = store.switch_session(store.active_session_id)
        session.persisted_conversation_id = "conversation-1"
        store.persistence = persistence

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        modal.query_one("#console-prompts-improve", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-structured-recipe", Button).press()
        await pilot.pause()
        modal.query_one("#console-prompts-recipe-outcome-first", Button).press()
        await pilot.pause()
        editor = modal.query_one(PromptBlockEditor)
        await editor._change_field("role", "content", "Be precise.")
        await pilot.pause()
        editor.query_one("#prompt-editor-apply-system", Checkbox).value = True
        await pilot.pause()
        assert not editor.query_one("#prompt-editor-apply", Button).disabled
        editor.query_one("#prompt-editor-apply", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert host.screen_stack[-1] is modal
        assert (
            str(
                modal.query_one(
                    "#console-prompts-improvement-status", Static
                ).renderable
            )
            == "Applied to this session, but could not save to the conversation."
        )
        assert composer.capture_draft_snapshot() == before
        applied_system = store.switch_session(
            store.active_session_id
        ).settings.system_prompt
        assert "Be precise." in str(applied_system)
        assert persistence.calls == 1

        modal.query_one("#console-prompts-persistence-retry", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert persistence.calls == 2
        assert (
            store.switch_session(store.active_session_id).settings.system_prompt
            == applied_system
        )
        assert composer.capture_draft_snapshot() == before


class SelectionCapturingGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.selections = []
        self.sent_messages = []

    async def resolve_for_send(self, selection):
        self.selections.append(selection)
        return await super().resolve_for_send(selection)

    async def stream_chat(self, resolution, messages, **kwargs):
        self.sent_messages.append(list(messages))
        yield "accepted"


class WaitingGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat(self, resolution, messages, **kwargs):
        yield "partial"
        self.started.set()
        await self.release.wait()
        yield " done"


class DelayedWaitingGateway(WaitingGateway):
    def __init__(self) -> None:
        super().__init__()
        self.validation_started = asyncio.Event()
        self.validation_release = asyncio.Event()

    async def resolve_for_send(self, selection):
        self.validation_started.set()
        await self.validation_release.wait()
        return await super().resolve_for_send(selection)


class ConsoleNavigationHarness(ConsoleHarness):
    def __init__(self, app_instance: object) -> None:
        super().__init__(app_instance)
        self.navigation_messages = []

    @on(NavigateToScreen)
    def capture_navigation(self, message: NavigateToScreen) -> None:
        self.navigation_messages.append(message)
        message.stop()


class RestoredConsoleHarness(ConsolidatedCSSApp):
    """Mount a Console ChatScreen from a previously saved state.

    Args:
        app_instance: Test application object injected into the screen.
        restored_state: Serialized screen state passed to ``ChatScreen.restore_state``.
    """

    def __init__(self, app_instance: object, restored_state: dict) -> None:
        """Initialize the restore harness with the target app and state payload.

        Args:
            app_instance: Test application object injected into the screen.
            restored_state: Serialized screen state used during mount.
        """
        super().__init__()
        self.app_instance = app_instance
        self.restored_state = restored_state

    async def on_mount(self) -> None:
        """Restore and mount a Console ChatScreen for lifecycle regression tests."""
        screen = ChatScreen(self.app_instance)
        screen.restore_state(self.restored_state)
        await self.push_screen(screen)


class BlockedGateway:
    async def resolve_for_send(self, selection):
        return SimpleNamespace(
            provider="llama_cpp",
            base_url=selection.base_url or "",
            model="test-model",
            ready=False,
            visible_copy="Provider blocked: llama.cpp unavailable.",
        )

    async def stream_chat(self, resolution, messages, **kwargs):
        raise AssertionError("Blocked gateway should not stream")


class CapturingGateway(_ReadyResolutionGateway):
    def __init__(self, chunks=("accepted",)) -> None:
        self.chunks = chunks
        self.sent_messages = []

    async def stream_chat(self, resolution, messages, **kwargs):
        self.sent_messages.append(list(messages))
        for chunk in self.chunks:
            yield chunk


class WorkspaceLinkingPersistence:
    def __init__(self, registry_service) -> None:
        self.registry_service = registry_service
        self.conversation_count = 0
        self.message_count = 0

    def create_conversation(self, **kwargs):
        self.conversation_count += 1
        conversation_id = f"persisted-conversation-{self.conversation_count}"
        workspace_id = kwargs.get("workspace_id")
        if kwargs.get("scope_type") == "workspace" and workspace_id:
            self.registry_service.link_membership(
                workspace_id,
                item_type="conversation",
                item_id=conversation_id,
                role="workspace-thread",
                title=kwargs.get("conversation_title") or "Chat 1",
            )
        return conversation_id

    def create_message(self, **kwargs):
        self.message_count += 1
        return f"persisted-message-{self.message_count}"

    def update_message_content(self, **kwargs):
        return True


class StaticConversationTreeService:
    """Return deterministic persisted trees for regression tests.

    This is a CI service double only. CDP/UAT approval evidence must use the
    running app with real persistence and live provider/API responses.
    """

    def __init__(self, trees):
        self.trees = dict(trees)
        self.calls = []

    async def get_conversation_tree(self, conversation_id: str, **kwargs):
        self.calls.append({"conversation_id": conversation_id, **kwargs})
        return self.trees.get(
            conversation_id,
            {
                "conversation": None,
                "root_threads": [],
                "pagination": {"total_root_threads": 0},
            },
        )


class SearchableConversationService(StaticConversationTreeService):
    def __init__(self, conversations: dict[str, dict]) -> None:
        super().__init__(conversations)
        self.list_calls: list[dict[str, object]] = []

    async def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        query = str(kwargs.get("query") or "").strip().lower()
        scope_type = str(kwargs.get("scope_type") or "").strip()
        workspace_id = str(kwargs.get("workspace_id") or "").strip()
        limit = int(kwargs.get("limit") or 50)
        items = []
        for conversation_id, tree in self.trees.items():
            conversation = tree.get("conversation", {})
            title = str(conversation.get("title") or "")
            conversation_workspace_id = str(
                conversation.get("workspace_id") or ""
            ).strip()
            conversation_scope = str(conversation.get("scope_type") or "").strip()
            if scope_type == "global":
                if conversation_scope != "global" and conversation_workspace_id:
                    continue
            elif scope_type == "workspace":
                if conversation_workspace_id != workspace_id:
                    continue
            elif workspace_id and conversation_workspace_id != workspace_id:
                continue
            if query and query not in title.lower():
                continue
            items.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "workspace_id": conversation.get("workspace_id"),
                    "scope_type": conversation.get("scope_type"),
                    "state": conversation.get("state", "active"),
                }
            )
        return {
            "items": items[:limit],
            "pagination": {
                "total": len(items),
                "limit": limit,
                "offset": 0,
            },
        }


class FailingSearchConversationService(StaticConversationTreeService):
    def __init__(self) -> None:
        super().__init__({})
        self.list_calls: list[dict[str, object]] = []

    async def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        raise RuntimeError("search failed")


class SlowSearchConversationService(StaticConversationTreeService):
    def __init__(self) -> None:
        super().__init__({})
        self.list_calls: list[dict[str, object]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        self.started.set()
        await self.release.wait()
        return {
            "items": [],
            "pagination": {
                "total": 0,
                "limit": int(kwargs.get("limit") or 50),
                "offset": int(kwargs.get("offset") or 0),
            },
        }


class SlowFirstSearchableConversationService(SearchableConversationService):
    def __init__(self, conversations: dict[str, dict]) -> None:
        super().__init__(conversations)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        if len(self.list_calls) == 1:
            self.started.set()
            await self.release.wait()
        query = str(kwargs.get("query") or "").strip().lower()
        scope_type = str(kwargs.get("scope_type") or "").strip()
        workspace_id = str(kwargs.get("workspace_id") or "").strip()
        limit = int(kwargs.get("limit") or 50)
        items = []
        for conversation_id, tree in self.trees.items():
            conversation = tree.get("conversation", {})
            title = str(conversation.get("title") or "")
            conversation_workspace_id = str(
                conversation.get("workspace_id") or ""
            ).strip()
            conversation_scope = str(conversation.get("scope_type") or "").strip()
            if scope_type == "global":
                if conversation_scope != "global" and conversation_workspace_id:
                    continue
            elif scope_type == "workspace":
                if conversation_workspace_id != workspace_id:
                    continue
            elif workspace_id and conversation_workspace_id != workspace_id:
                continue
            if query and query not in title.lower():
                continue
            items.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "workspace_id": conversation.get("workspace_id"),
                    "scope_type": conversation.get("scope_type"),
                    "state": conversation.get("state", "active"),
                }
            )
        return {
            "items": items[:limit],
            "pagination": {
                "total": len(items),
                "limit": limit,
                "offset": 0,
            },
        }


class SyncSearchableConversationService(SearchableConversationService):
    def list_conversations(self, *, mode: str = "local", **kwargs):
        self.list_calls.append({"mode": mode, **kwargs})
        query = str(kwargs.get("query") or "").strip().lower()
        scope_type = str(kwargs.get("scope_type") or "").strip()
        workspace_id = str(kwargs.get("workspace_id") or "").strip()
        limit = int(kwargs.get("limit") or 50)
        items = []
        for conversation_id, tree in self.trees.items():
            conversation = tree.get("conversation", {})
            title = str(conversation.get("title") or "")
            conversation_workspace_id = str(
                conversation.get("workspace_id") or ""
            ).strip()
            conversation_scope = str(conversation.get("scope_type") or "").strip()
            if scope_type == "global":
                if conversation_scope != "global" and conversation_workspace_id:
                    continue
            elif scope_type == "workspace":
                if conversation_workspace_id != workspace_id:
                    continue
            elif workspace_id and conversation_workspace_id != workspace_id:
                continue
            if query and query not in title.lower():
                continue
            items.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "workspace_id": conversation.get("workspace_id"),
                    "scope_type": conversation.get("scope_type"),
                    "state": conversation.get("state", "active"),
                }
            )
        return {
            "items": items[:limit],
            "pagination": {
                "total": len(items),
                "limit": limit,
                "offset": 0,
            },
        }


class NoModeSyncSearchableConversationService(SearchableConversationService):
    def list_conversations(
        self,
        *,
        query: str = "",
        scope_type: str = "",
        workspace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        self.list_calls.append(
            {
                "query": query,
                "scope_type": scope_type,
                "workspace_id": workspace_id,
                "limit": limit,
                "offset": offset,
            }
        )
        normalized_query = str(query or "").strip().lower()
        normalized_scope_type = str(scope_type or "").strip()
        normalized_workspace_id = str(workspace_id or "").strip()
        items = []
        for conversation_id, tree in self.trees.items():
            conversation = tree.get("conversation", {})
            title = str(conversation.get("title") or "")
            conversation_workspace_id = str(
                conversation.get("workspace_id") or ""
            ).strip()
            conversation_scope = str(conversation.get("scope_type") or "").strip()
            if normalized_scope_type == "global":
                if conversation_scope != "global" and conversation_workspace_id:
                    continue
            elif normalized_scope_type == "workspace":
                if conversation_workspace_id != normalized_workspace_id:
                    continue
            elif (
                normalized_workspace_id
                and conversation_workspace_id != normalized_workspace_id
            ):
                continue
            if normalized_query and normalized_query not in title.lower():
                continue
            items.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "workspace_id": conversation.get("workspace_id"),
                    "scope_type": conversation.get("scope_type"),
                    "state": conversation.get("state", "active"),
                }
            )
        return {
            "items": items[:limit],
            "pagination": {
                "total": len(items),
                "limit": limit,
                "offset": offset,
            },
        }


class FakeConversationLocalMarksService:
    def __init__(self, starred: tuple[str, ...] = ()) -> None:
        self.starred = set(starred)

    def star_conversation(self, conversation_id: str) -> None:
        self.starred.add(conversation_id)

    def unstar_conversation(self, conversation_id: str) -> None:
        self.starred.discard(conversation_id)

    def is_starred(self, conversation_id: str) -> bool:
        return conversation_id in self.starred

    def list_marked_conversation_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.starred))


class FailThenRecoverGateway(_ReadyResolutionGateway):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, resolution, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield "partial"
            raise RuntimeError("llama.cpp stream failed")
        yield "recovered"


async def _wait_for_text(screen, pilot, expected: str, *, attempts: int = 80) -> None:
    for _ in range(attempts):
        if expected in _visible_text(screen):
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"Text not found: {expected!r}. Visible text: {_visible_text(screen)!r}"
    )


async def _wait_for_focus(app, pilot, widget, *, attempts: int = 40) -> None:
    for _ in range(attempts):
        if getattr(app, "focused", None) is widget:
            return
        await pilot.pause(0.05)
    focused = getattr(app, "focused", None)
    raise AssertionError(
        f"Focus did not reach {getattr(widget, 'id', widget)!r}; "
        f"focused={getattr(focused, 'id', focused)!r}"
    )


async def _wait_for_active_session_change(
    store: ConsoleChatStore,
    pilot,
    previous_session_id: str | None,
    *,
    attempts: int = 40,
) -> str:
    """Wait for the Console store to activate a different session."""
    for _ in range(attempts):
        active_session_id = store.active_session_id
        if active_session_id is not None and active_session_id != previous_session_id:
            return active_session_id
        await pilot.pause(0.05)
    raise AssertionError(
        "Console active session did not change. "
        f"previous={previous_session_id!r}; active={store.active_session_id!r}"
    )


async def _wait_for_active_session(
    store: ConsoleChatStore,
    pilot,
    expected_session_id: str,
    *,
    attempts: int = 40,
) -> None:
    """Wait for the Console store to activate the expected session."""
    for _ in range(attempts):
        if store.active_session_id == expected_session_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        "Console active session did not match expected session. "
        f"expected={expected_session_id!r}; active={store.active_session_id!r}"
    )


async def _open_console_inspector_rail(console: ChatScreen, pilot) -> None:
    """Open the right rail before asserting inspector-visible content."""
    rail_state = replace(
        console._current_console_rail_state(),
        right_open=True,
    )
    console._sync_console_rail_visibility(rail_state)
    await _wait_for_selector(console, pilot, "#console-run-inspector-state")
    for _ in range(40):
        inspector = console.query_one("#console-run-inspector-state")
        if (
            inspector.display
            and inspector.region.width > 0
            and inspector.region.height > 0
        ):
            return
        await pilot.pause(0.05)
    inspector = console.query_one("#console-run-inspector-state")
    raise AssertionError(
        "Console run inspector is not visible/actionable: "
        f"display={inspector.display!r} region={inspector.region!r}"
    )


async def _open_console_context_rail(console: ChatScreen, pilot) -> None:
    """Open the left rail before asserting context-visible content."""
    rail_state = replace(
        console._current_console_rail_state(),
        left_open=True,
    )
    console._sync_console_rail_visibility(rail_state)
    # Storage/Sync/handoff status rows now live in the collapsible Details
    # section; expand it so its rows lay out with a real screen region.
    if not console._current_console_rail_state().details_open:
        console._toggle_console_rail_section("details")
    await _wait_for_selector(console, pilot, "#console-workspace-authority-label")
    for _ in range(40):
        label = console.query_one("#console-workspace-authority-label")
        if label.display and label.region.width > 0 and label.region.height > 0:
            return
        await pilot.pause(0.05)
    label = console.query_one("#console-workspace-authority-label")
    raise AssertionError(
        "Console workspace authority row is not visible/actionable: "
        f"display={label.display!r} region={label.region!r}"
    )


def _static_plain_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _message_row_plain_text(console, message_id: str) -> str:
    """Renderer-agnostic plain text of one transcript message row (TASK-1990).

    Grouped Assistant messages carry their sole role label on the outer turn
    surface. Standalone rows retain their pre-grouping message-row shape.
    """
    from textual.widgets import Markdown

    from tldw_chatbook.Widgets.Console.console_transcript import (
        ConsoleMarkdownMessage,
    )

    try:
        surface = console.query_one(f"#console-assistant-turn-{message_id}")
    except NoMatches:
        surface = console.query_one(f"#console-message-{message_id}")
    row = console.query_one(f"#console-message-{message_id}")
    if isinstance(row, ConsoleMarkdownMessage):
        parts = [_static_plain_text(static) for static in surface.query(Static)]
        parts.append(row.query_one(Markdown).source)
        return "\n".join(parts)
    if surface is row:
        return _static_plain_text(row)
    return "\n".join(_static_plain_text(static) for static in surface.query(Static))


def _widget_text(widget) -> str:
    if hasattr(widget, "renderable"):
        renderable = widget.renderable
        return getattr(renderable, "plain", str(renderable))
    label = getattr(widget, "label", "")
    return getattr(label, "plain", str(label))


def _console_workspace_conversation_texts(console) -> list[str]:
    rows = [
        _widget_text(row)
        for row in console.query(".console-workspace-conversation-row")
    ]
    try:
        tree = console.query_one("#console-workspace-tree", ConsoleWorkspaceTree)
    except NoMatches:
        return rows
    rows.extend(node.label.plain for node in tree.conversation_nodes.values())
    return rows


def _normalized_row_text(row) -> str:
    """Row label with wrapped name lines rejoined for containment checks.

    Rail row names wrap at the rail budget, so a title like "Needle in
    Workspace B" renders across two label lines; joining on whitespace lets
    tests keep matching the full title.
    """
    return " ".join(_widget_text(row).split())


def _row_is_selected(row) -> bool:
    """Whether a rail row is marked active.

    The flush-left row rework removed the textual active marker ("▸ ");
    selection is expressed solely via the selected row class.
    """
    return row.has_class("console-workspace-conversation-row-selected")


def _selected_workspace_conversation_texts(console) -> list[str]:
    """Whitespace-normalized labels of the rows marked active."""
    return [
        _normalized_row_text(row)
        for row in console.query(".console-workspace-conversation-row")
        if _row_is_selected(row)
    ]


def _console_conversation_browser_rows(console):
    """Flatten every row out of the live grouped conversation-browser state.

    ``ConsoleConversationBrowserRow.updated_label`` is only populated by
    ``_normalize_input_row`` inside ``build_console_conversation_browser_state``
    (see ``tldw_chatbook/Workspaces/conversation_browser_state.py``); the raw
    ``_native_console_browser_rows``/``_membership_console_browser_rows``
    accessors return un-normalized input rows whose ``updated_label`` is
    always the dataclass default (``""``). Rebuilding the full context state
    is the seam that actually feeds the rendered row label.
    """
    browser = (
        console._workspace._build_console_workspace_context_state().conversation_browser
    )
    rows = []
    for section in browser.sections:
        rows.extend(section.rows)
        for group in section.groups:
            rows.extend(group.rows)
    return rows


def _workspace_conversation_row_by_id(console, conversation_id: str):
    for row in console.query(".console-workspace-conversation-row"):
        if getattr(row, "conversation_id", None) == conversation_id:
            return row
    return None


def _workspace_conversation_row_by_key(console, row_key: str):
    for row in console.query(".console-workspace-conversation-row"):
        if getattr(row, "row_key", None) == row_key:
            return row
    return None


def _console_workspace_conversation_row_id_for_session(console, session_id: str) -> str:
    target_conversation_id = f"native:{session_id}"
    for row in console.query(".console-workspace-conversation-row"):
        if getattr(row, "conversation_id", None) == target_conversation_id:
            return str(row.id)
    rows = [
        (
            getattr(row, "id", ""),
            getattr(row, "conversation_id", None),
            _widget_text(row),
        )
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation row for {target_conversation_id!r} not found. "
        f"Rows: {rows!r}"
    )


async def _click_console_workspace_conversation_for_session(
    console,
    pilot,
    store,
    session_id: str,
    *,
    attempts: int = 20,
) -> None:
    """Click a workspace conversation row once Textual hit-testing is ready."""
    row_id = _console_workspace_conversation_row_id_for_session(console, session_id)
    for _ in range(attempts):
        # task-2902 investigation: `pilot.click(selector)` computes its click
        # coordinates from the target's `.region`, but the workspace rows are
        # rebuilt by every Console sync pass and a rebuilt row's region can be
        # zero (pre-layout) or stale relative to the rendered cell map for
        # whole retry windows — synthetic clicks then land where the row
        # isn't (minimal repro in task-2902's notes; real mouse input is
        # unaffected because the driver resolves against the rendered cell
        # map). `press()` drives the identical Pressed->handler chain the
        # click would, without deriving coordinates from a racing layout.
        try:
            row = console.query_one(f"#{row_id}", Button)
        except Exception:
            await pilot.pause(0.05)
            continue
        if not row.disabled:
            row.press()
            for _ in range(10):
                if store.active_session_id == session_id:
                    return
                await pilot.pause(0.05)
        await pilot.pause(0.05)
    rows = [
        (
            getattr(row, "id", ""),
            getattr(row, "conversation_id", None),
            getattr(row, "region", None),
            _widget_text(row),
        )
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation click did not activate {session_id!r}. "
        f"active={store.active_session_id!r}; rows={rows!r}"
    )


async def _scroll_console_rail_row_into_view(pilot, row) -> None:
    """Scroll one left-rail row into the visible screen region before clicking.

    task-14920: task-14810 split the Console left rail into peer disclosure
    sections (Sessions, Workspaces, Conversations, Model, ...) inside the
    ``#console-left-rail-body`` ``VerticalScroll``. All of them open by
    default, so at the 160x48 harness size the rail's virtual height is ~99
    rows against a 29-row viewport and the Conversations section starts ~20
    rows below the fold. ``pilot.click`` addresses SCREEN coordinates and
    raises ``OutOfBounds`` for anything outside the visible region, so the
    click has to be preceded by the same scroll a real user performs.

    Scrolling (rather than switching to ``Button.press()``) is deliberate: it
    keeps these tests asserting that a real hit-tested click on the row
    activates it, which is the claim they were written to pin. Measured on
    ``b4c5105ed``: the row moves from y=70 to y=37 and ``pilot.click``
    returns True.
    """
    row.scroll_visible(animate=False, force=True)
    await pilot.pause()


async def _click_after_scrolling_into_view(
    console,
    pilot,
    selector: str,
    *,
    attempts: int = 40,
) -> None:
    """Scroll a widget into view and click it, re-trying while layout settles.

    task-14920: the same screen-coordinate problem as
    ``_scroll_console_rail_row_into_view``, but for the transcript. A single
    ``scroll_visible()`` followed by a fixed ``pilot.pause`` is not enough when
    a later recompose (image prep landing, the guidance setup card) moves the
    target again: the click then raises ``OutOfBounds``. Re-scroll and re-click
    until the click is delivered, so the assertion that follows is still about
    a REAL hit-tested click.

    The click's return value is deliberately NOT required to be ``True``: a
    container whose top-left cell belongs to a child (an inline image row here)
    reports ``False`` while still delivering the event, which is the semantics
    the caller had before this helper existed. Only the coordinate failure is
    retried.
    """
    for _ in range(attempts):
        try:
            widget = console.query_one(selector)
        except NoMatches:
            await pilot.pause(0.05)
            continue
        widget.scroll_visible(animate=False, force=True)
        await pilot.pause()
        try:
            await pilot.click(selector)
            return
        except (OutOfBounds, NoMatches):
            pass
        await pilot.pause(0.05)
    raise AssertionError(
        f"Click on {selector!r} never landed inside the visible screen region."
    )


async def _click_console_workspace_conversation_for_id(
    console,
    pilot,
    conversation_id: str,
    *,
    attempts: int = 40,
) -> str:
    """Click a workspace conversation row by persisted conversation id.

    Retries until the click actually registers. The Console conversation rail is
    still settling its layout for a beat after the rows first become queryable
    (the empty transcript recomposes into the multi-line setup card on the first
    guidance sync), so a single ``pilot.click`` fired the instant the row appears
    can land on a mid-reflow offset and miss without ever invoking the row
    handler. Mirror the click-until-effect pattern used by
    ``_click_console_workspace_conversation_for_session`` and re-click until the
    press is delivered (``pilot.click`` returns ``True``).
    """
    out_of_bounds: OutOfBounds | None = None
    for _ in range(attempts):
        row_widget = None
        for row in console.query(".console-workspace-conversation-row"):
            if getattr(row, "conversation_id", None) == conversation_id:
                row_widget = row
                break
        if row_widget is not None:
            await _scroll_console_rail_row_into_view(pilot, row_widget)
            row_id = str(row_widget.id)
            try:
                if await pilot.click(f"#{row_id}"):
                    return row_id
            except OutOfBounds as error:
                # A rebuild mid-retry can move the row again; retry rather
                # than abort, but keep the error for the failure message.
                out_of_bounds = error
            except NoMatches:
                # The rail rebuilt between the scroll and the click; the next
                # attempt re-queries. The loop still has to land a real click.
                pass
        try:
            tree = console.query_one("#console-workspace-tree", ConsoleWorkspaceTree)
        except NoMatches:
            tree = None
        if tree is not None:
            node = tree.conversation_nodes.get(conversation_id)
            if node is not None:
                tree.move_cursor(node)
                tree.focus()
                await pilot.press("enter")
                await pilot.pause()
                return str(tree.id)
        await pilot.pause(0.05)
    rows = [
        (
            getattr(row, "id", ""),
            getattr(row, "conversation_id", None),
            getattr(row, "region", None),
            _widget_text(row),
        )
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation row for {conversation_id!r} not clicked "
        f"(last out-of-bounds: {out_of_bounds!r}). Rows: {rows!r}"
    )


async def _click_console_workspace_conversation_for_row_key(
    console,
    pilot,
    row_key: str,
    *,
    attempts: int = 40,
) -> str:
    """Click a workspace conversation row by grouped browser row key."""
    for _ in range(attempts):
        for row in console.query(".console-workspace-conversation-row"):
            if getattr(row, "row_key", None) == row_key:
                await _scroll_console_rail_row_into_view(pilot, row)
                row_id = str(row.id)
                await pilot.click(f"#{row_id}")
                return row_id
        await pilot.pause(0.05)
    rows = [
        (
            getattr(row, "id", ""),
            getattr(row, "row_key", None),
            getattr(row, "conversation_id", None),
            _widget_text(row),
        )
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(
        f"Workspace conversation row key {row_key!r} not found. Rows: {rows!r}"
    )


async def _wait_for_workspace_conversation_text(
    console,
    pilot,
    expected: str,
    *,
    selected: bool | None = None,
    attempts: int = 40,
) -> list[str]:
    """Wait for a rail row whose (wrap-normalized) label contains ``expected``.

    Returns the normalized label texts of every rail row. ``selected`` gates
    the match on the row's active state (the selected row class -- rows no
    longer carry a textual marker prefix).
    """
    for _ in range(attempts):
        rows = list(console.query(".console-workspace-conversation-row"))
        row_texts = [_normalized_row_text(row) for row in rows]
        for row, text in zip(rows, row_texts):
            if expected not in text:
                continue
            if selected is None or _row_is_selected(row) == selected:
                return row_texts
        try:
            tree = console.query_one("#console-workspace-tree", ConsoleWorkspaceTree)
        except NoMatches:
            tree = None
        if tree is not None:
            tree_texts = [
                " ".join(node.label.plain.split())
                for node in tree.conversation_nodes.values()
            ]
            for node, text in zip(tree.conversation_nodes.values(), tree_texts):
                if expected not in text:
                    continue
                data = node.data
                if selected is None or bool(data and data.selected) == selected:
                    return [*row_texts, *tree_texts]
        await pilot.pause(0.05)
    raise AssertionError(
        f"Workspace conversation {expected!r} not found. "
        f"Rows: {_console_workspace_conversation_texts(console)!r}"
    )


async def _wait_for_console_rename_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if (
            host.screen_stack
            and host.screen_stack[-1].query("#console-rename-session-modal")
            and host.screen_stack[-1].query("#console-rename-session-title")
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console rename modal did not open")


async def _wait_for_console_screen(host: ConsoleHarness, console, pilot) -> None:
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1] is console:
            await pilot.pause()
            return
        await pilot.pause(0.05)
    raise AssertionError("Console modal did not dismiss")


async def _wait_for_workspace_switcher_modal(host: ConsoleHarness, pilot):
    for _ in range(40):
        if host.screen_stack and host.screen_stack[-1].query(
            "#console-workspace-switcher-modal"
        ):
            await pilot.pause()
            return host.screen_stack[-1]
        await pilot.pause(0.05)
    raise AssertionError("Console workspace switcher modal did not open")


def _select_llamacpp_console(console: ChatScreen) -> None:
    """Select the native llama.cpp path after mounted controls initialize.

    ``_console_control_provider``/``_console_control_model`` mirror the
    legacy compact-provider bar into Console's *display* labels only (see
    ``on_console_compact_provider_changed``'s docstring) -- they do not
    reach the active session's settings, which were already snapshotted
    when the screen mounted its first session (before this helper ever
    runs). Without also pushing the selection through
    ``_replace_active_console_session_settings`` (the same call the real
    Console Settings modal apply path and ``_apply_detected_local_server``
    use), the already-existing session stays on its mount-time default
    provider and every send this helper is meant to unblock stays gated
    behind the first-run setup modal.

    Args:
        console: The ChatScreen instance to configure with llama.cpp provider
            settings. Session settings are updated to route sends through the
            configured llama.cpp endpoint.
    """
    app_config = console.app_instance.app_config
    api_settings = app_config.setdefault("api_settings", {})
    llama_settings = api_settings.setdefault("llama_cpp", {})
    llama_settings.setdefault("api_url", "http://127.0.0.1:9099/v1")
    llama_settings.setdefault("model", "test-model")
    console._console_control_provider = "llama_cpp"
    console._console_control_model = "test-model"
    settings = console._session._ensure_active_console_session_settings()
    console._session._replace_active_console_session_settings(
        replace(
            settings,
            provider="llama_cpp",
            model="test-model",
            base_url=None,
            source="user",
        )
    )
    console._sync_console_control_bar()


@pytest.mark.asyncio
async def test_console_native_generic_provider_send_renders_completed_message(
    monkeypatch,
):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {}
    captured_kwargs = []

    def fake_chat_api_call(**_kwargs):
        captured_kwargs.append(_kwargs)
        return "generic provider response"

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        gateway = console._ensure_console_provider_gateway()
        app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}
        # TASK-2154.6: mounted setup-blocked (no key) -> Send genuinely
        # disabled; re-sync after the config fix so the block lifts, exactly
        # as the Settings save path syncs after a real key fix.
        await console._sync_native_console_chat_ui()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "generic provider response")

        assert isinstance(gateway, ConsoleProviderGateway)
        assert captured_kwargs
        assert captured_kwargs[-1]["api_endpoint"] == "openai"
        assert captured_kwargs[-1]["api_key"] == DUMMY_OPENAI_API_KEY
        assert (
            console._ensure_console_chat_controller().run_state.status
            is ConsoleRunStatus.COMPLETED
        )
        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assistant_messages = [
            message
            for message in messages
            if message.role is ConsoleMessageRole.ASSISTANT
        ]
        assert assistant_messages[-1].status == "complete"


@pytest.mark.asyncio
async def test_console_native_send_button_click_dispatches_message(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {}
    captured_kwargs = []

    def fake_chat_api_call(**_kwargs):
        captured_kwargs.append(_kwargs)
        return "click provider response"

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}
        # TASK-2154.6: the console mounted setup-blocked (no key), so Send is
        # genuinely disabled; a raw post-mount config mutation only lifts the
        # block once the UI re-syncs -- the same umbrella sync the Settings
        # save path runs after a real key fix.
        await console._sync_native_console_chat_ui()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("click send")
        # Settle the layout: typing hid the disabled-reason strip, which
        # shifts Send right by the strip's cells -- a click issued before
        # the reflow lands in the (newly widened) draft instead.
        await pilot.pause(0.1)

        await pilot.click("#console-send-message")
        await _wait_for_text(console, pilot, "click provider response")

        assert captured_kwargs
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_successful_send_does_not_leave_empty_send_tooltip(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}

    def fake_chat_api_call(**_kwargs):
        return "sent response"

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("send once")

        await pilot.click("#console-send-message")
        await _wait_for_text(console, pilot, "sent response")

        send_button = console.query_one("#console-send-message", Button)
        assert composer.draft_text() == ""
        assert send_button.tooltip != "Type a message before sending."


@pytest.mark.asyncio
async def test_console_native_missing_key_blocks_before_clearing_generic_draft():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "MISSING_OPENAI_KEY"}
    }
    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("preserve this")

        console.query_one("#console-send-message", Button).press()
        # task-16474: this used to wait for "missing API key", a string that
        # only existed inside the COLLAPSED inspector rail's row Statics --
        # rows the rail-collapse cascade stamps display=False on, so their
        # scrape-visibility depended on whether an ambient post-mount sync
        # happened to restore them (the compact-bar mount burst used to
        # provide that extra cycle). The composer's own blocked copy is the
        # deterministic, user-facing statement of the same missing-key
        # block, so that is what the contract pins now.
        await _wait_for_text(console, pilot, "add an API key to continue")

        assert composer.draft_text() == "preserve this"


@pytest.mark.asyncio
async def test_console_native_enter_while_setup_blocked_is_inert_behind_modal():
    # Formerly asserted an Enter-triggered recovery notification; with the
    # blocking modal, Enter/typing never reach the covered composer, so no send
    # is attempted and no recovery notification fires (Phase 2 spec, section 2
    # revised). The modal's own action button owns recovery now.
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "MISSING_OPENAI_KEY"}
    }
    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.display is True
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        assert composer.can_focus is False

        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.pause(0.05)

        assert composer.draft_text() == ""
        assert notifications == []


@pytest.mark.asyncio
async def test_console_setup_blocked_send_is_unreachable_behind_modal():
    """The blocking setup modal makes the Enter send path unreachable.

    Formerly this asserted a durable SYSTEM recovery message; with setup blocked
    behind the modal the composer is inert, so Enter never triggers a send and
    no transcript message is appended (Phase 2 spec, section 2 revised).
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key_env_var": "MISSING_OPENAI_KEY"}
    }
    app.console_provider_gateway_factory = lambda: ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.display is True
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        await pilot.press("b", "l", "o", "c", "k", "e", "d")
        await pilot.press("enter")
        await pilot.pause(0.1)

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert composer.draft_text() == ""
        assert messages == []

    assert notifications == []


@pytest.mark.asyncio
async def test_console_native_blocked_send_preserves_composer_text_and_shows_recovery():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = BlockedGateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("blocked draft")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "Provider blocked")

        assert composer.draft_text() == "blocked draft"


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    (
        ("http://127.0.0.1:9099/v1/chat/completions", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/v1/models", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/v1", "http://127.0.0.1:9099"),
        ("127.0.0.1:9099", "http://127.0.0.1:9099"),
        ("127.0.0.1:9099/v1", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/completion", "http://127.0.0.1:9099"),
        ("http://127.0.0.1:9099/", "http://127.0.0.1:9099"),
        (None, "http://127.0.0.1:9099"),
    ),
)
def test_console_llamacpp_base_url_normalizes_openai_compatible_endpoints(
    raw_url, expected
):
    screen = ChatScreen(_build_test_app())

    assert screen._normalize_llamacpp_base_url(raw_url) == expected


def test_console_transcript_sync_timer_polls_at_coarse_interval(monkeypatch):
    screen = ChatScreen(_build_test_app())
    captured = {}

    def fake_set_interval(interval, callback):
        captured["interval"] = interval
        captured["callback"] = callback
        return SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr(screen, "set_interval", fake_set_interval)

    screen._start_console_transcript_sync_timer()

    assert captured["interval"] >= 0.15


def test_console_transcript_fingerprint_tolerates_empty_variant_container():
    screen = ChatScreen(_build_test_app())
    message = SimpleNamespace(
        id="m1",
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        status="complete",
        turn_id="turn-1",
        persisted_message_id=None,
        variants=SimpleNamespace(selected_index=0, variants=None),
    )

    fingerprint = screen._native_console_transcript_fingerprint([message])

    assert fingerprint[1][0][-2] == (0, ())
    assert fingerprint[1][0][-1] is None


def test_console_provider_selection_reads_local_llamacpp_configured_model():
    app = _build_test_app()
    # Provider/model resolution reads `app_config["chat_defaults"]` (see
    # `_effective_console_provider_model`) -- `chat_api_provider_value` /
    # `chat_api_model_value` are legacy root-chat attributes with no reader
    # left in the native Console path (removed under TASK-650) and setting
    # them here was a no-op, so the unconfigured provider fell through to
    # the `"llama_cpp"` hardcoded fallback instead of this test's intended
    # `"local_llamacpp"` selection.
    app.app_config["chat_defaults"] = {
        "provider": "local_llamacpp",
        "model": "runtime-model",
    }
    app.app_config["api_settings"] = {
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099/v1/chat/completions",
            "model": "configured-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.provider == "local_llamacpp"
    assert selection.base_url == "http://127.0.0.1:9099"
    assert selection.explicit_model == "runtime-model"
    assert selection.configured_model == "configured-model"
    assert selection.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID


def test_console_provider_selection_carries_active_session_system_prompt():
    """The selection built for the controller carries the session's system prompt."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    screen = ChatScreen(app)
    settings = screen._session._ensure_active_console_session_settings()
    screen._session._replace_active_console_session_settings(
        replace(settings, system_prompt="Answer only in French.")
    )

    selection = screen._build_console_provider_selection()

    assert selection.system_prompt == "Answer only in French."

    controller = screen._ensure_console_chat_controller()
    assert controller.system_prompt == "Answer only in French."


def test_console_provider_selection_floors_missing_active_workspace_read_only():
    """No active workspace: the context floors to Default WITHOUT writing.

    TASK-21118 relocated `ensure_default_workspace`'s restore/repair side-
    effects off the per-keystroke context read (which runs this builder):
    the read now floors a missing active workspace to the same
    `DEFAULT_WORKSPACE_ID` the ensure returned -- preserving the
    capability-less Default identity every policy consumer compares
    against -- but leaves the registry untouched. The registry is restored
    at the session-start/workspace-switch seams instead (app wiring's
    ensure at boot, `set_active_workspace`'s switch-to-Default repair,
    the Console session-switch and browser seams), which the final ensure
    below stands in for.
    """
    app = _build_test_app()
    service = app.workspace_registry_service
    with service.db.transaction() as conn:
        conn.execute("UPDATE workspace_records SET active = 0")
    assert service.get_active_workspace() is None
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID
    # Read-only: the passive read must NOT have healed the registry ...
    assert service.get_active_workspace() is None
    # ... that is the ensure seams' job, and it still works.
    service.ensure_default_workspace()
    assert service.get_active_workspace().workspace_id == DEFAULT_WORKSPACE_ID
    selection_after = screen._build_console_provider_selection()
    assert (
        selection_after.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID
    )


def test_console_configured_llamacpp_override_wins_over_provider_api_url():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "configured-model"
    # The builder reads the SESSION's provider (derived from
    # `[chat_defaults] provider`), never `chat_api_provider_value`. These
    # three used to reach the llama.cpp branch only via
    # `provider_config_key(...) or "llama_cpp"` -- the fallback an empty test
    # config left them on. The shipped template sets provider = "OpenAI"
    # (task-15270), so select llama.cpp the way a user does.
    app.app_config.setdefault("chat_defaults", {})["provider"] = "llama_cpp"
    app.app_config["console"] = {
        "llama_cpp_base_url_override": "http://127.0.0.1:9099/v1",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"


def test_console_llamacpp_api_base_url_wins_over_merged_provider_api_url(monkeypatch):
    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "configured-model"
    # The builder reads the SESSION's provider (derived from
    # `[chat_defaults] provider`), never `chat_api_provider_value`. These
    # three used to reach the llama.cpp branch only via
    # `provider_config_key(...) or "llama_cpp"` -- the fallback an empty test
    # config left them on. The shipped template sets provider = "OpenAI"
    # (task-15270), so select llama.cpp the way a user does.
    app.app_config.setdefault("chat_defaults", {})["provider"] = "llama_cpp"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "api_base_url": "http://127.0.0.1:9099/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"


def test_console_llamacpp_env_url_wins_over_provider_api_url(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:9099/v1")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "configured-model"
    # The builder reads the SESSION's provider (derived from
    # `[chat_defaults] provider`), never `chat_api_provider_value`. These
    # three used to reach the llama.cpp branch only via
    # `provider_config_key(...) or "llama_cpp"` -- the fallback an empty test
    # config left them on. The shipped template sets provider = "OpenAI"
    # (task-15270), so select llama.cpp the way a user does.
    app.app_config.setdefault("chat_defaults", {})["provider"] = "llama_cpp"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"


def test_console_session_settings_blank_base_url_keeps_llamacpp_fallback(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:9099/v1")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "runtime-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="settings-model",
            base_url=None,
        )
    )
    store.switch_session(session.id)
    screen._console_chat_store = store

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9099"
    assert selection.explicit_model == "settings-model"


def test_console_session_settings_base_url_wins_over_llamacpp_fallback(monkeypatch):
    monkeypatch.setenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:9099/v1")
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "runtime-model"
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://localhost:8080/v1",
            "model": "fallback-model",
        }
    }
    screen = ChatScreen(app)
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="settings-model",
            base_url="http://127.0.0.1:9999/v1",
        )
    )
    store.switch_session(session.id)
    screen._console_chat_store = store

    selection = screen._build_console_provider_selection()

    assert selection.base_url == "http://127.0.0.1:9999"
    assert selection.explicit_model == "settings-model"


@pytest.mark.asyncio
async def test_console_stop_interrupts_stream_and_keeps_partial_message_visible():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await _wait_for_text(console, pilot, "partial")
        assert "streaming" in _visible_text(console).lower()

        console.query_one("#console-stop-generation", Button).press()
        await _wait_for_text(console, pilot, "stopped")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        # TASK-337: an explicit stopped-by-user record follows the partial.
        assert messages[-1].content == "Response stopped by user."
        assert messages[-2].content == "partial"
        assert messages[-2].status == "stopped"


@pytest.mark.asyncio
async def test_console_collapsed_stop_interrupts_real_run_without_expanding():
    gateway = WaitingGateway()
    app = _build_test_app()
    _configure_native_ready_console(app, model="test-model")
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        composer.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await _wait_for_text(console, pilot, "partial")
        console._set_console_composer_collapsed(True)
        await pilot.pause()

        collapsed_stop = composer.query_one(
            "#console-collapsed-stop-generation", Button
        )
        expanded_stop = composer.query_one("#console-stop-generation", Button)
        assert composer.region.height == 1
        assert collapsed_stop.display is True
        assert composer.query_one("#console-composer-expanded").display is False

        collapsed_stop.press()
        await _wait_for_text(console, pilot, "stopped")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert messages[-1].content == "Response stopped by user."
        assert messages[-2].content == "partial"
        assert messages[-2].status == "stopped"
        assert composer.collapsed is True
        assert composer.query_one("#console-composer-expanded").display is False
        assert expanded_stop.display is False


@pytest.mark.asyncio
async def test_console_collapsed_stop_stale_action_warns_without_expanding():
    gateway = WaitingGateway()
    app = _build_test_app()
    _configure_native_ready_console(app, model="test-model")
    app.console_provider_gateway_factory = lambda: gateway
    notifications: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        composer.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await _wait_for_text(console, pilot, "partial")
        console._set_console_composer_collapsed(True)
        await pilot.pause()
        assert composer.query_one("#console-collapsed-stop-generation", Button).display

        gateway.release.set()
        await _wait_for_text(console, pilot, "partial done")
        await console._stop_console_generation_from_visible_action()

        assert (
            "No active Console run to stop.",
            {"severity": "warning"},
        ) in notifications
        assert composer.collapsed is True


@pytest.mark.asyncio
async def test_console_composer_stop_is_subdued_when_idle():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        stop_button = composer.query_one("#console-stop-generation", Button)

        assert stop_button.disabled is False
        assert stop_button.has_class("console-action-disabled")
        assert stop_button.has_class("console-stop-idle")
        assert not stop_button.has_class("console-stop-active")

        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await _wait_for_text(console, pilot, "partial")

        # TASK-2154.6: a blocked Send is now genuinely disabled, not just
        # class-subdued.
        # TASK-15121 (contract change, not a rename): since TASK-14808 /
        # ADR-046 an ACCEPTED live turn no longer blocks Send -- it re-labels
        # it to "Queue" and admits the draft as a FIFO follow-up turn. So
        # `console-send-blocked` is no longer reachable here; it now covers
        # only the states that genuinely refuse a draft (Preparing before
        # acceptance, Queue full, setup/attachment blocks -- see
        # `derive_prompt_queue_presentation`). The original claim of this
        # assertion block is kept where it is still true: with the draft
        # consumed by the send, Send is genuinely disabled and non-primary --
        # it just fails the empty-draft gate now, not the run gate.
        assert send_button.disabled is True
        assert send_button.has_class("console-action-disabled")
        assert send_button.has_class("console-send-inactive")
        assert not send_button.has_class("console-send-blocked")
        assert not send_button.has_class("console-action-primary")
        assert send_button.label.plain == "Queue"
        assert stop_button.disabled is False
        assert stop_button.has_class("console-stop-active")
        assert not stop_button.has_class("console-action-disabled")
        assert not stop_button.has_class("console-stop-idle")

        stop_button.press()
        await _wait_for_text(console, pilot, "stopped")


@pytest.mark.asyncio
async def test_console_duplicate_send_during_stream_does_not_break_stop_control():
    gateway = WaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await _wait_for_text(console, pilot, "partial")

        composer.load_draft("second send")
        send_button = console.query_one("#console-send-message", Button)
        # TASK-2154.6: genuinely disabled while the run blocks sends; the
        # direct handler dispatch below (not `press()`, which no-ops on a
        # disabled control) is exactly how the Enter hotkey still reaches it.
        # TASK-15121: superseded by TASK-14808 / ADR-046 -- "once accepted,
        # the normal Send action becomes Queue; Enter and the button both
        # enqueue the exact canonical text draft". A second draft mid-stream
        # is therefore no longer REFUSED, so the disabled/`console-send-blocked`
        # pin below is gone. What this test is named for is unchanged and is
        # asserted harder below: the duplicate send must not start a second
        # run, must not be silently eaten, and must not break Stop.
        assert send_button.disabled is False
        assert not send_button.has_class("console-send-blocked")
        assert send_button.label.plain == "Queue"
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        assert controller.prompt_queue_registry.snapshot(session_id).total_count == 0

        await console.handle_console_send_message(Button.Pressed(send_button))
        await pilot.pause(0.1)
        # The live turn is untouched -- one run, still streaming, never a
        # second concurrent generation.
        assert (
            console._ensure_console_chat_controller().run_state.status.value
            == "streaming"
        )
        # ...and the duplicate landed in the bounded queue rather than being
        # dropped: admission (not an attempted enqueue) is what clears the
        # draft, so an empty composer here is evidence the text was accepted.
        assert controller.prompt_queue_registry.snapshot(session_id).total_count == 1
        assert composer.draft_text() == ""

        console.query_one("#console-stop-generation", Button).press()
        await _wait_for_text(console, pilot, "stopped")


@pytest.mark.asyncio
async def test_console_streaming_chunks_render_after_slow_provider_validation():
    gateway = DelayedWaitingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(
            gateway.validation_started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT
        )
        assert (
            console._ensure_console_chat_controller().run_state.status
            is ConsoleRunStatus.VALIDATING
        )
        console._sync_console_control_bar()
        send_button = console.query_one("#console-send-message", Button)
        stop_button = console.query_one("#console-stop-generation", Button)

        # TASK-2154.6: VALIDATING blocks sends, so Send is now genuinely
        # disabled here, not merely class-subdued (classes unchanged below).
        assert send_button.disabled is True
        assert send_button.has_class("console-action-disabled")
        assert send_button.has_class("console-send-blocked")
        assert not send_button.has_class("console-action-primary")
        assert stop_button.disabled is False
        assert stop_button.has_class("console-stop-idle")

        gateway.validation_release.set()
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await _wait_for_text(console, pilot, "partial")
        gateway.release.set()
        await _wait_for_text(console, pilot, "partial done")


class _HoldingStreamGateway(_ReadyResolutionGateway):
    """Resolves ready immediately, then holds the stream open emitting NOTHING
    until released — so the only thing that can surface the user's own message
    in the transcript is an acceptance-time echo, never streamed content."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat(self, resolution, messages, **kwargs):
        self.started.set()
        await self.release.wait()
        if False:  # pragma: no cover - async generator that yields no chunks
            yield ""


@pytest.mark.asyncio
async def test_console_send_echoes_user_message_before_transcript_poll(monkeypatch):
    """A sent message must appear the instant the submit is accepted, not only on
    the next coarse 0.2s transcript poll.

    Regression guard for task-351(a): the composer clears at ~acceptance while
    the transcript still read "No messages yet" for ~600ms, reading as
    "not sent". The poll is disabled here so the echo has to stand on its own.
    """
    gateway = _HoldingStreamGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        # Neutralise the 0.2s transcript poll: with no timer to eventually
        # surface the message, only an acceptance-time echo can.
        monkeypatch.setattr(
            console, "_start_console_transcript_sync_timer", lambda: None
        )
        console._stop_console_transcript_sync_timer()

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("ECHONOW")
        console.query_one("#console-send-message", Button).press()

        # Stream reached => the USER row was appended and the submit accepted.
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        try:
            # With the poll disabled and no stream content, the sent message can
            # only reach the transcript via the acceptance-time echo. Use the
            # file's stable wait helper (bounded pilot.pause loop) rather than a
            # hand-rolled fixed budget so this stays deterministic under CI load.
            await _wait_for_text(console, pilot, "ECHONOW")
        finally:
            gateway.release.set()


@pytest.mark.asyncio
async def test_console_collapsed_paste_sends_full_payload_not_visible_token():
    long_text = "x" * 80
    gateway = CapturingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_pasted_text(long_text)

        assert "Pasted text | 80 characters | Expand" in _visible_text(console)
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

    assert gateway.sent_messages[-1][-1]["content"] == long_text
    assert (
        "Pasted text | 80 characters | Expand"
        not in gateway.sent_messages[-1][-1]["content"]
    )


@pytest.mark.asyncio
async def test_console_native_send_preserves_expanded_payload_whitespace():
    gateway = CapturingGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("  padded payload  ")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

    assert gateway.sent_messages[-1][-1]["content"] == "  padded payload  "


@pytest.mark.asyncio
async def test_console_configured_model_reaches_gateway_when_ui_model_is_unset():
    gateway = SelectionCapturingGateway()
    app = _build_test_app()
    # `chat_defaults.provider` (read by `_effective_console_provider_model`
    # at session-creation time), not `_console_control_provider` set after
    # mount: the active session's settings -- and so its provider -- are
    # snapshotted once when the session is first created, and a later
    # `_console_control_provider` assignment only relabels the legacy
    # compact-bar display; it never re-derives the already-created session's
    # settings. Setting it post-mount left the session on the hardcoded
    # `"llama_cpp"` fallback instead of this test's intended
    # `"local_llamacpp"`, so `api_settings.llama_cpp` (never configured
    # here) produced no model and the send stayed blocked.
    app.app_config["chat_defaults"] = {"provider": "local_llamacpp"}
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config["api_settings"] = {
        "local_llamacpp": {
            "api_url": "http://127.0.0.1:9099/v1/chat/completions",
            "model": "configured-model",
        }
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

    assert gateway.selections[-1].explicit_model is None
    assert gateway.selections[-1].configured_model == "configured-model"


@pytest.mark.asyncio
async def test_console_native_send_clears_composer_after_acceptance_and_updates_store():
    """Verify accepted sends clear the composer and render compact transcript text."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("hel", "lo")
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        # Renderer-agnostic reply wait: the plain row renders "Assistant  hello"
        # (two-space separator); the markdown row renders label and body as
        # separate lines — collapse whitespace before matching (TASK-1990).
        for _ in range(80):
            if "Assistant hello" in " ".join(_visible_text(console).split()):
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(
                f"Assistant reply never rendered. Visible: {_visible_text(console)!r}"
            )

        assert composer.draft_text() == ""
        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        assert messages[-2].content == "hello"
        assert messages[-1].content == "hello"


@pytest.mark.asyncio
async def test_console_chat_lifecycle_state_survives_screen_recreation_return():
    """Verify Console chat tabs, transcript, and draft restore after recreation."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("assistant return",)
    )
    saved_state: dict | None = None
    first_session_id: str | None = None
    second_session_id: str | None = None

    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("typed text")
        composer.insert_pasted_text(" and pasted text")
        assert "typed text and pasted text" in _visible_text(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "assistant return")

        store = console._ensure_console_chat_store()
        first_session_id = store.active_session_id
        assert first_session_id is not None
        await pilot.click("#console-new-chat-tab")
        second_session_id = await _wait_for_active_session_change(
            store,
            pilot,
            first_session_id,
        )
        await _wait_for_selector(
            console, pilot, f"#console-session-tab-{second_session_id}"
        )
        composer.load_draft("draft before return")
        await console._sync_native_console_chat_ui()

        saved_state = console.save_state()

    assert saved_state is not None
    assert first_session_id is not None
    assert second_session_id is not None

    restored_host = RestoredConsoleHarness(app, saved_state)
    async with restored_host.run_test(size=(160, 48)) as pilot:
        console = restored_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(
            console, pilot, f"#console-session-tab-{second_session_id}"
        )
        await _wait_for_text(console, pilot, "draft before return")

        store = console._ensure_console_chat_store()
        assert store.active_session_id == second_session_id

        await pilot.click(f"#console-session-tab-{first_session_id}")
        await _wait_for_active_session(store, pilot, first_session_id)
        await _wait_for_text(console, pilot, "typed text and pasted text")
        await _wait_for_text(console, pilot, "assistant return")


@pytest.mark.asyncio
async def test_console_send_refreshes_workspace_conversation_rail_after_persistence():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("accepted",)
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        row_texts = _console_workspace_conversation_texts(console)
        assert any("Chat 1" in text for text in row_texts)
        assert len(console.query("#console-workspace-empty-conversations")) == 0
        store = console._ensure_console_chat_store()
        store.persistence = WorkspaceLinkingPersistence(app.workspace_registry_service)
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")
        await _wait_for_selector(console, pilot, "#console-workspace-conversation-0")

        row = console.query_one("#console-workspace-conversation-0")
        row_text = _widget_text(row)
        assert _row_is_selected(row)
        # Once the first message is accepted, the default "Chat 1" title is
        # replaced by an auto-title derived from the message (see
        # _maybe_auto_title_session in console_chat_controller.py).
        assert "hello" in row_text
        assert "Chat 1" not in row_text
        assert "\n" in row_text
        # TASK-374 removes the redundant workspace/group label from grouped
        # conversation rows while retaining a non-default state differentiator.
        assert "active session" in row_text
        assert "workspace-thread" not in row_text
        assert not re.search(r"\[[0-9a-f]{8}\]", row_text)
        # The row metadata also carries a relative age label appended after
        # persistence (e.g. "now", "2m", "1h"...). The rendered metadata line
        # cell-truncates at the rail budget, so assert the age at the
        # normalized browser-state seam that feeds the row label instead.
        assert any(
            "now" in str(getattr(state_row, "updated_label", ""))
            for state_row in _console_conversation_browser_rows(console)
        )
        assert len(console.query("#console-workspace-empty-conversations")) == 0


@pytest.mark.asyncio
async def test_console_send_after_workspace_switch_persists_to_selected_workspace():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("accepted",)
    )
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-change-workspace")
        store = console._ensure_console_chat_store()
        store.persistence = WorkspaceLinkingPersistence(service)
        _select_llamacpp_console(console)
        first_session = store.ensure_session()
        store.replace_session_settings(
            first_session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
        )
        assert first_session.workspace_id == "ws-a"

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)
        assert service.get_active_workspace().workspace_id == "ws-b"

        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == "ws-b"
        assert active_session.title == "Workspace B Chat"
        assert active_session.settings.provider == "llama_cpp"
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello from b")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        workspace_a_conversations = service.list_workspace_conversations("ws-a")
        workspace_b_conversations = service.list_workspace_conversations("ws-b")
        assert workspace_a_conversations == ()
        assert [row.title for row in workspace_b_conversations] == [
            active_session.title
        ]


@pytest.mark.asyncio
async def test_console_workspace_switch_refreshes_visible_session_tabs():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-change-workspace")
        store = console._ensure_console_chat_store()
        first_session = store.ensure_session()
        assert first_session.workspace_id == "ws-a"

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)

        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == "ws-b"
        await _wait_for_selector(
            console, pilot, f"#console-session-tab-{active_session.id}"
        )
        assert "Workspace B Chat" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_workspace_switch_refresh_is_not_dropped_during_inflight_sync():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-change-workspace")
        store = console._ensure_console_chat_store()
        first_session = store.ensure_session()
        assert first_session.workspace_id == "ws-a"

        first_sync_blocked = asyncio.Event()
        release_first_sync = asyncio.Event()
        original_sync_tabs = console._sync_console_native_session_tabs
        blocked_once = False

        async def blocking_sync_tabs():
            nonlocal blocked_once
            await original_sync_tabs()
            if blocked_once:
                return
            blocked_once = True
            first_sync_blocked.set()
            await release_first_sync.wait()

        console._sync_console_native_session_tabs = blocking_sync_tabs
        first_sync_task = asyncio.create_task(console._sync_native_console_chat_ui())
        await wait_for_background_signal(
            first_sync_blocked,
            first_sync_task,
            what="the first native-console UI sync",
        )

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)

        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == "ws-b"
        release_first_sync.set()
        await first_sync_task

        await _wait_for_selector(
            console, pilot, f"#console-session-tab-{active_session.id}"
        )
        assert "Workspace B Chat" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_mount_uses_active_workspace_title_for_initial_session():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_text(console, pilot, "Workspace A Chat")
        assert "Workspace A" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_tab_switch_aligns_active_workspace_context():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Workspace A Chat", workspace_id="ws-a")
        second = store.create_session(title="Workspace B Chat", workspace_id="ws-b")
        service.set_active_workspace("ws-b")
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-session-tab-{first.id}")
        assert store.active_session_id == second.id
        assert service.get_active_workspace().workspace_id == "ws-b"

        await pilot.click(f"#console-session-tab-{first.id}")

        assert store.active_session_id == first.id
        assert service.get_active_workspace().workspace_id == "ws-a"
        await _wait_for_text(console, pilot, "Workspace A")


@pytest.mark.asyncio
async def test_console_unsupported_provider_block_renders_one_normalized_system_message():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="wip_provider", model="test-model"),
        )
        await console._sync_native_console_chat_ui()

        await console._submit_console_native_draft("hello")
        await _wait_for_text(console, pilot, "Provider blocked")

        messages = store.messages_for_session(store.active_session_id)
        system_messages = [
            message.content
            for message in messages
            if message.role is ConsoleMessageRole.SYSTEM
        ]
        assert system_messages == [
            "Provider blocked: 'wip_provider' is not available in Console yet. "
            "Choose a supported provider."
        ]
        assert (
            console._ensure_console_chat_controller().run_state.visible_copy
            == system_messages[0]
        )


@pytest.mark.asyncio
async def test_console_add_api_key_recovery_targets_provider_settings_category():
    app = _build_test_app()
    app.app_config["api_settings"] = {"huggingface": {}}
    host = ConsoleNavigationHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(
                provider="huggingface", model="meta-llama/test-model"
            ),
        )
        await console._sync_native_console_chat_ui()
        # The shared Workbench recovery banner stays hidden now — the setup
        # card's action button carries this recovery instead (Phase 2 spec,
        # section 2).
        await _wait_for_selector(console, pilot, "#console-setup-modal-action")

        await pilot.click("#console-setup-modal-action")

        assert len(host.navigation_messages) == 1
        message = host.navigation_messages[0]
        assert message.screen_name == "settings"
        assert message.screen_context == {
            "category": SettingsCategoryId.PROVIDERS_MODELS.value,
            "provider": "huggingface",
            "model": "meta-llama/test-model",
            "field": "api_key",
        }


@pytest.mark.asyncio
async def test_console_add_api_key_recovery_tolerates_missing_session_settings():
    app = _build_test_app()
    host = ConsoleNavigationHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(
                provider="huggingface", model="meta-llama/test-model"
            ),
        )
        await console._sync_native_console_chat_ui()
        # The shared Workbench recovery banner stays hidden now — the setup
        # card's action button carries this recovery instead (Phase 2 spec,
        # section 2).
        await _wait_for_selector(console, pilot, "#console-setup-modal-action")
        console._active_console_provider_model_display = lambda: (
            "huggingface",
            "meta-llama/test-model",
            None,
        )

        await pilot.click("#console-setup-modal-action")

        assert len(host.navigation_messages) == 1
        message = host.navigation_messages[0]
        assert message.screen_context == {
            "category": SettingsCategoryId.PROVIDERS_MODELS.value,
            "provider": "huggingface",
            "model": "meta-llama/test-model",
            "field": "api_key",
        }


@pytest.mark.asyncio
async def test_console_assistant_message_click_exposes_selected_actions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        await pilot.click(f"#console-message-{message.id}")
        await _wait_for_selector(
            console, pilot, f"#console-message-action-regenerate-{message.id}"
        )

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.selected_message_id == message.id


@pytest.mark.asyncio
async def test_console_transcript_wraps_long_message_content_without_horizontal_overflow():
    app = _build_test_app()
    host = ConsoleHarness(app)

    long_answer = " ".join(["wrapped assistant response segment"] * 180)

    async with host.run_test(size=(92, 32)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=long_answer,
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        row = console.query_one(f"#console-message-{message.id}")

        assert row.region.width <= transcript.region.width
        assert transcript.virtual_size.width <= transcript.region.width
        assert row.region.height > 2


@pytest.mark.asyncio
async def test_console_selected_message_copy_action_uses_app_clipboard():
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-copy-{message.id}"
        )

        await pilot.click(f"#console-message-action-copy-{message.id}")
        await pilot.pause()

    app.copy_to_clipboard.assert_called_once_with("answer")
    assert console._last_console_action.action_id == "copy"


def test_console_original_attempt_action_parser_prefers_full_prefix():
    assert ChatScreen._parse_console_message_action_button_id(
        "console-message-action-view-original-attempt-message-1"
    ) == ("view-original-attempt", "message-1")


@pytest.mark.asyncio
async def test_console_original_attempt_preview_toggles_without_changing_selected_content():
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    save_note = Mock(return_value="saved-note")
    app.notes_scope_service = SimpleNamespace(save_note=save_note)
    app.post_message = Mock()
    host = ConsoleHarness(app)
    original = "Original unselected attempt"
    repaired = "Selected repaired answer [S1]"

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Question",
        )
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=repaired,
        )
        store.set_citation_presentation(
            message.id,
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.REPAIRED,
            ),
        )
        controller._remember_original_attempt(message.id, original)
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        view_selector = f"#console-message-action-view-original-attempt-{message.id}"
        await _wait_for_selector(console, pilot, view_selector)

        async def assert_repaired_consumers() -> None:
            selected = store.get_message(message.id)
            app.copy_to_clipboard.reset_mock()
            await console.handle_console_message_action(
                SimpleNamespace(
                    button=SimpleNamespace(
                        id=f"console-message-action-copy-{message.id}"
                    ),
                    stop=Mock(),
                )
            )
            copied_text = app.copy_to_clipboard.call_args.args[0]
            plain_export = transcript.to_plain_text()
            save_note.reset_mock()
            await console._save_console_message_as_note(message.id)
            save_payload = save_note.call_args.kwargs["content"]
            provider_messages = controller._provider_messages_for_session(session.id)
            provider_contents = [row.get("content") for row in provider_messages]

            app.post_message.reset_mock()
            console._console_speaking_message_id = None
            await console.handle_console_message_action(
                SimpleNamespace(
                    button=SimpleNamespace(
                        id=f"console-message-action-speak-{message.id}"
                    ),
                    stop=Mock(),
                )
            )
            spoken_event = next(
                call.args[0]
                for call in app.post_message.call_args_list
                if call.args[0].__class__.__name__ == "TTSMessageSpeechRequestEvent"
            )

            assert selected.content == repaired
            assert copied_text == repaired
            assert repaired in plain_export
            assert save_payload == repaired
            assert repaired in provider_contents
            assert spoken_event.snapshot.raw_content == repaired
            for output in (
                selected.content,
                copied_text,
                plain_export,
                save_payload,
                *provider_contents,
                spoken_event.snapshot.raw_content,
            ):
                assert original not in str(output)

        await assert_repaired_consumers()
        await pilot.click(view_selector)
        await _wait_for_selector(
            console,
            pilot,
            f"#console-original-attempt-{message.id}",
        )
        assert console._console_original_attempt_previews == {message.id: original}
        await assert_repaired_consumers()

        view_button = console.query_one(view_selector, Button)
        view_button.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert console._console_original_attempt_previews == {}
        assert len(console.query(f"#console-original-attempt-{message.id}")) == 0
        await assert_repaired_consumers()

        controller.clear_original_attempt(message.id)
        console._console_original_attempt_previews[message.id] = original
        await console._sync_native_console_chat_ui()
        assert console._console_original_attempt_previews == {}
        assert store.get_message(message.id).content == repaired


@pytest.mark.asyncio
async def test_transcript_role_label_renders_dim_body_full_contrast():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        from tldw_chatbook.Widgets.Console.console_transcript import (
            ConsoleMarkdownMessage,
        )

        row = console.query_one(f"#console-message-{message.id}")
        if isinstance(row, ConsoleMarkdownMessage):
            # The one dim role label belongs to the outer Assistant turn; the
            # headerless Markdown answer remains a separate full-contrast body.
            turn = console.query_one(f"#console-assistant-turn-{message.id}")
            labels = list(turn.query(".console-transcript-speaker-label"))
            assert len(labels) == 1
            rendered = turn.query_one(
                ".console-transcript-speaker-label", Static
            ).renderable
            body_text = _message_row_plain_text(console, message.id)
        else:
            rendered = row.renderable
            body_text = rendered.plain

    assert isinstance(rendered, Content)
    # Content with spans: the role prefix span carries "dim"; the body stays
    # unstyled (full contrast) even though the combined plain text is unchanged.
    assert rendered.plain.startswith("Assistant")
    assert "answer" in body_text
    assert any("dim" in str(span.style) for span in rendered.spans), rendered.spans
    if "answer" in rendered.plain:
        body_start = rendered.plain.index("answer")
        body_styles = [
            str(span.style)
            for span in rendered.spans
            if span.start <= body_start < span.end
        ]
        assert not any("dim" in style for style in body_styles), body_styles


@pytest.mark.asyncio
async def test_console_clicking_rendered_message_shows_action_row():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{message.id}")

        await pilot.click(f"#console-message-{message.id}")
        await _wait_for_selector(
            console, pilot, f"#console-message-action-copy-{message.id}"
        )


@pytest.mark.asyncio
async def test_console_selected_message_copy_action_works_from_keyboard():
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()
        await _wait_for_focus(console.app, pilot, transcript)
        await pilot.press("down")
        await _wait_for_selector(
            console, pilot, f"#console-message-action-copy-{message.id}"
        )

        copy_selector = f"console-message-action-copy-{message.id}"
        for _ in range(16):
            focused = getattr(console.app, "focused", None)
            if getattr(focused, "id", None) == copy_selector:
                break
            await pilot.press("tab")
        else:
            raise AssertionError(
                "Keyboard focus did not reach the selected-message Copy action"
            )

        await pilot.press("enter")
        await pilot.pause()

    app.copy_to_clipboard.assert_called_once_with("answer")
    assert console._last_console_action.action_id == "copy"


@pytest.mark.asyncio
async def test_transcript_c_key_copies_selected_message():
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()
        await pilot.press("down")
        await pilot.pause(0.1)
        assert transcript.selected_message_id is not None
        await pilot.press("c")
        await pilot.pause(0.3)

    app.copy_to_clipboard.assert_called_once_with("answer")
    assert console._last_console_action.action_id == "copy"


@pytest.mark.asyncio
async def test_transcript_rapid_select_then_action_retries_after_deferred_mount():
    # The action row mounts via call_later(refresh_messages) after
    # select_message; firing the selection and the action key back-to-back
    # with no settling between them (mirroring the switcher's rapid-refresh
    # test, which posts two Input.Changed values with no await between them)
    # reproduces a fast Down->c race where the button isn't mounted yet.
    app = _build_test_app()
    app.copy_to_clipboard = Mock()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()
        transcript.action_select_next()
        transcript.action_invoke_selected_action("copy")
        await pilot.pause()

    app.copy_to_clipboard.assert_called_once_with("answer")
    assert console._last_console_action.action_id == "copy"


@pytest.mark.asyncio
async def test_console_message_action_keyboard_focus_stays_inside_action_row():
    app = _build_test_app()
    host = ConsoleHarness(app)

    # ConsoleHarness is a bare App (not TldwCli), so it
    # never loads the app's built CSS bundle -- only widget DEFAULT_CSS (see
    # the note at chat_screen.py:726-728) -- so the bundle's
    # `.console-transcript-action-button { min-width: 5 }` override never
    # applies in this harness and every button falls back to Textual's
    # built-in Button min-width of 16. That no longer fits inside the
    # previous 160-col reference width without the trailing buttons (delete
    # included) landing off the right edge of the terminal, so this widened
    # just enough to keep every button in-bounds and genuinely clickable
    # here. This is a test-harness gap, not a CSS bug: the real app loads
    # the bundle and renders the row far narrower.
    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-delete-{message.id}"
        )

        transcript.focus_action(message.id, "delete")
        delete_button = console.query_one(
            f"#console-message-action-delete-{message.id}", Button
        )
        await _wait_for_focus(console.app, pilot, delete_button)

        await pilot.press("tab")
        copy_button = console.query_one(
            f"#console-message-action-copy-{message.id}", Button
        )
        await _wait_for_focus(console.app, pilot, copy_button)

        # task-17656: idle Speak moved from the message header into the
        # action row between Copy and Edit (8ae87242a) — the walk gains a
        # stop, exactly as the row's on-screen guide lists it.
        await pilot.press("tab")
        speak_button = console.query_one(
            f"#console-message-action-speak-{message.id}", Button
        )
        await _wait_for_focus(console.app, pilot, speak_button)

        await pilot.press("tab")
        edit_button = console.query_one(
            f"#console-message-action-edit-{message.id}", Button
        )
        await _wait_for_focus(console.app, pilot, edit_button)

        await pilot.press("tab")
        save_as_button = console.query_one(
            f"#console-message-action-save-as-{message.id}", Button
        )
        await _wait_for_focus(console.app, pilot, save_as_button)

        # task-17656: walk the rest of the row full circle back to Delete —
        # every stop of a completed assistant reply is Tab-reachable in
        # visual order, and focus never escapes the row.
        for action_id in (
            "regenerate",
            "continue",
            "feedback-up",
            "feedback-down",
            "delete",
        ):
            await pilot.press("tab")
            stop = console.query_one(
                f"#console-message-action-{action_id}-{message.id}", Button
            )
            await _wait_for_focus(console.app, pilot, stop)

        transcript.focus_action(message.id, "save-as")
        await _wait_for_focus(console.app, pilot, save_as_button)
        await pilot.press("enter")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-save-as-modal")

    assert console._last_console_action.action_id == "save-as"


@pytest.mark.asyncio
async def test_console_inspector_hides_selected_message_group_without_selection():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-inspector-state")

        inspector = console.query_one("#console-run-inspector-state")
        assert "Selected Message" not in _visible_text(inspector)
        assert len(console.query("#console-inspector-selected-message-heading")) == 0


@pytest.mark.asyncio
async def test_console_setup_required_state_groups_recovery_and_action_copy():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")

        # The shared Workbench recovery banner stays hidden — the blocking setup
        # modal groups the setup-blocked copy and the recovery action together
        # instead (Phase 2 spec, section 2 revised).
        recovery = console.query_one("#workbench-recovery-callout")
        assert recovery.display is False
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.display is True
        assert "Connect a provider (API key or local server)" in _visible_text(console)
        assert (
            str(console.query_one("#console-setup-modal-action", Button).label)
            == CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL
        )


@pytest.mark.asyncio
async def test_console_empty_transcript_teaches_setup_and_start_paths():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")

        # Setup guidance is on the blocking modal; Attach/Run-RAG start paths
        # stay reachable on the control bar (never on the modal).
        modal = console.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.display is True
        console_text = _visible_text(console)
        assert "Get started" in console_text
        assert "Connect a provider (API key or local server)" in console_text
        assert "Send your first message" in console_text
        assert "Attach context" in console_text
        assert "Search Library" in console_text


def _assert_selector_hidden_or_absent(console, selector: str) -> None:
    """Assert a selector is either absent or mounted but not displayed."""
    for widget in console.query(selector):
        assert not widget.display, f"{selector} unexpectedly displayed"


@pytest.mark.asyncio
async def test_console_blocked_empty_transcript_shows_setup_card_steps():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-transcript-empty-state")
        text = _visible_text(console)
        assert "Get started" in text
        assert "Connect a provider (API key or local server)" in text
        assert "Pick a model" in text
        assert "Send your first message" in text
        # The legacy provider recovery strip must not compete with the setup card.
        _assert_selector_hidden_or_absent(console, "#console-provider-recovery-strip")


@pytest.mark.asyncio
async def test_console_first_send_flag_switches_empty_state_to_quiet():
    app = _build_test_app()
    app.app_config.setdefault("console", {})["onboarding"] = {
        "first_send_completed": True
    }
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-transcript-empty-state")
        text = _visible_text(console)
        assert "No messages yet." in text
        assert "Get started" not in text
        assert "Connect a provider" not in text


@pytest.mark.asyncio
async def test_console_accepted_send_records_first_send_flag():
    # Reuse the ready-provider send harness from
    # test_console_send_refreshes_workspace_conversation_rail_after_persistence:
    # same fixtures/gateway stub, then assert the persisted global flag.
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("accepted",)
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        store.persistence = WorkspaceLinkingPersistence(app.workspace_registry_service)
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        onboarding = app.app_config.get("console", {}).get("onboarding", {})
        assert onboarding.get("first_send_completed") is True


@pytest.mark.asyncio
async def test_console_failed_send_does_not_record_first_send_flag():
    """A FAILED first send must not set the one-time onboarding flag (task-182d)."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: FailThenRecoverGateway()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")

        onboarding = app.app_config.get("console", {})
        if isinstance(onboarding, dict):
            onboarding = onboarding.get("onboarding", {})
        assert not (
            isinstance(onboarding, dict) and onboarding.get("first_send_completed")
        )
        assert console._console_first_send_completed() is False

        # The provider error must not be stored as assistant message content.
        store = console._ensure_console_chat_store()
        assistant_contents = [
            message.content
            for message in store.messages_for_session(store.active_session_id)
            if message.role is ConsoleMessageRole.ASSISTANT
        ]
        assert all(
            "Provider stream failed" not in content for content in assistant_contents
        )


@pytest.mark.asyncio
async def test_console_accepted_send_clears_composer_before_run_end():
    """The composer clears when the submit is accepted, not only at run end."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("accepted",)
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        controller = console._ensure_console_chat_controller()
        assert controller.on_submission_accepted is not None

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("clear me on accept")
        # Invoke the acceptance hook exactly as the controller does the moment
        # the user message is persisted; the composer must clear immediately.
        controller.on_submission_accepted()
        await pilot.pause()
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_record_first_send_repairs_corrupt_config_value():
    # Regression: a corrupt (non-dict) "console" value used to crash
    # _record_console_first_send via unguarded .setdefault() chaining -- the
    # write path must replace the corrupt value with a fresh dict and still
    # persist the flag rather than silently skipping the write.
    app = _build_test_app()
    app.app_config["console"] = "corrupt"
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        console._record_console_first_send()

        console_cfg = app.app_config.get("console")
        assert isinstance(console_cfg, dict)
        onboarding_cfg = console_cfg.get("onboarding")
        assert isinstance(onboarding_cfg, dict)
        assert onboarding_cfg.get("first_send_completed") is True


@pytest.mark.asyncio
async def test_console_workspace_authority_rows_are_structured_for_scanning():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_context_rail(console, pilot)

        assert (
            _static_plain_text(
                console.query_one("#console-workspace-authority-label", Static)
            )
            == "Storage"
        )
        assert "local" in _static_plain_text(
            console.query_one("#console-workspace-authority-value", Static)
        )
        assert (
            _static_plain_text(
                console.query_one("#console-workspace-runtime-label", Static)
            )
            == "Local files"
        )
        assert "Private scratch" in _static_plain_text(
            console.query_one("#console-workspace-runtime-value", Static)
        )
        # TASK-715: factory-default sync/server/ACP rows collapse into one line.
        assert (
            "not configured"
            in _static_plain_text(
                console.query_one(
                    "#console-workspace-server-features-collapsed", Static
                )
            ).lower()
        )


class _CharacterHandoffStore(ConsoleChatStore):
    """Capture character-session identity at greeting persistence time.

    task-14920: this used to be a hand-rolled stub that implemented only
    ``create_session`` and ``append_message``. When TASK a6cc05d8b ("seed
    dynamic character chat templates") moved the greeting seam from
    ``store.append_message(...)`` to ``store.seed_character_roleplay(...)``,
    the stub silently lost the method -- and the handoff wraps the seed call
    in ``except Exception``, so the resulting ``AttributeError`` was swallowed
    and four tests observed "no greeting was ever appended" instead of a
    broken double. Subclassing the real ``ConsoleChatStore`` (persistence
    ``None`` = in-memory only) keeps the greeting rendering and the
    identity-at-append capture pinned to production behaviour rather than to
    a hand-copied imitation of it, so the next seam move fails loudly here.
    """

    def __init__(self) -> None:
        super().__init__()
        self.create_kwargs: dict | None = None
        self.session: ConsoleChatSession | None = None
        self.messages: list[dict] = []
        self.identity_at_append: dict | None = None

    def create_session(self, **kwargs):
        self.create_kwargs = dict(kwargs)
        self.session = super().create_session(**kwargs)
        return self.session

    def append_message(self, session_id, *, role, content, persist=False, **kwargs):
        assert self.session is not None
        self.identity_at_append = {
            "runtime_backend": self.session.runtime_backend,
            "assistant_kind": self.session.assistant_kind,
            "assistant_id": self.session.assistant_id,
            "assistant_authority_id": self.session.assistant_authority_id,
            "character_id": self.session.character_id,
            "character_ref": self.session.character_ref(),
        }
        self.messages.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "persist": persist,
            }
        )
        return super().append_message(
            session_id,
            role=role,
            content=content,
            persist=persist,
            **kwargs,
        )


def _character_start_handoff(
    *,
    runtime_backend: str = "local",
    selected_kind: str = "character",
    source: str = "personas",
    active_server_profile_id: str | None = None,
    character_id: object = "7",
) -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source=source,
        item_type=f"{selected_kind}-card",
        title="Elara",
        body="Character summary",
        runtime_backend=runtime_backend,
        source_owner=runtime_backend,
        source_selector_state=runtime_backend,
        active_server_profile_id=active_server_profile_id,
        metadata={
            "intent": "start_chat",
            "selected_kind": selected_kind,
            "selected_record_id": character_id,
            "selected_name": "Elara",
            "selected_target_id": (f"{runtime_backend}:{selected_kind}:{character_id}"),
            "backend": runtime_backend,
        },
    )


def _character_card() -> dict:
    return {
        "id": 7,
        "name": "Elara",
        "first_message": "Hello {{user}}, I am {{char}}.",
        "system_prompt": "Stay curious.",
    }


_DEFAULT_HANDOFF_VALUE = object()


def _character_handoff_runtime(
    *,
    active_server_id: str | None = None,
    local_authority: str | None = "local-authority",
    card=_DEFAULT_HANDOFF_VALUE,
    server_authority: object = "server-user-v1:" + ("a" * 64),
) -> SimpleNamespace:
    """Build a source-aware handoff runtime with inspectable boundaries."""
    scoped_card = _character_card() if card is _DEFAULT_HANDOFF_VALUE else card
    db = SimpleNamespace(
        get_local_authority_id=Mock(return_value=local_authority),
        # A usable legacy card makes forbidden fallback observable: if the
        # scoped lookup misses and production consults this seam, the test
        # would incorrectly create a session instead of failing closed.
        get_character_card_by_id=Mock(return_value=_character_card()),
    )
    scope_service = SimpleNamespace(
        get_character=AsyncMock(return_value=scoped_card),
    )
    resolver = AsyncMock(return_value=server_authority)
    initial_capture = SimpleNamespace(account="A")
    authority_context_state = {"current": initial_capture}
    capture_context = Mock(
        side_effect=lambda *, expected_server_id: authority_context_state["current"]
    )
    capture_is_current = Mock(
        side_effect=lambda capture: capture is authority_context_state["current"]
    )
    resolve_captures: list[object | None] = []

    async def resolve_character_authority_id(
        *,
        expected_server_id: str,
        context_capture: object | None = None,
    ):
        resolve_captures.append(context_capture)
        return await resolver(expected_server_id=expected_server_id)

    app = SimpleNamespace(
        app_config={},
        active_server_id=active_server_id,
        chachanotes_db=db,
        character_persona_scope_service=scope_service,
        server_context_provider=SimpleNamespace(
            capture_character_authority_context=capture_context,
            is_character_authority_context_current=capture_is_current,
            resolve_character_authority_id=resolve_character_authority_id,
        ),
    )
    return SimpleNamespace(
        app=app,
        db=db,
        scope_service=scope_service,
        resolver=resolver,
        initial_capture=initial_capture,
        authority_context_state=authority_context_state,
        capture_context=capture_context,
        capture_is_current=capture_is_current,
        resolve_captures=resolve_captures,
    )


def _handoff_chat_screen(monkeypatch, app, store: _CharacterHandoffStore) -> ChatScreen:
    screen = ChatScreen(app)
    monkeypatch.setattr(
        ChatScreen,
        "_ensure_console_chat_store",
        lambda self: store,
    )
    monkeypatch.setattr(
        ConsoleSessionController,
        "_default_console_session_settings",
        lambda self: ConsoleSessionSettings(
            provider="anthropic",
            model="claude-3-haiku",
        ),
    )
    monkeypatch.setattr(
        ChatScreen,
        "_sync_native_console_chat_ui",
        AsyncMock(),
    )
    monkeypatch.setattr(
        ChatScreen,
        "_focus_console_composer_if_needed",
        lambda self, **kwargs: None,
    )
    return screen


async def _run_character_handoff(monkeypatch, runtime, payload):
    store = _CharacterHandoffStore()
    screen = _handoff_chat_screen(monkeypatch, runtime.app, store)
    started = await screen._session._start_character_console_session(payload)
    return started, store


@pytest.mark.parametrize(
    ("field", "value"),
    (
        pytest.param("source", "library", id="wrong-origin"),
        pytest.param("item_type", "persona-card", id="wrong-item-type"),
        pytest.param("runtime_backend", "LOCAL", id="non-exact-runtime-source"),
        pytest.param("source_owner", "server", id="contradictory-owner"),
        pytest.param(
            "source_selector_state",
            "server",
            id="contradictory-selector",
        ),
    ),
)
def test_character_start_handoff_requires_exact_coherent_envelope(
    field,
    value,
):
    payload = _character_start_handoff()
    setattr(payload, field, value)

    assert session_module._character_session_identity_from_handoff(payload) is None


@pytest.mark.parametrize(
    ("metadata_key", "value"),
    (
        pytest.param("intent", " start_chat", id="non-exact-intent"),
        pytest.param("selected_kind", " character", id="non-exact-kind"),
        pytest.param("backend", "server", id="contradictory-backend"),
        pytest.param("backend", None, id="missing-backend"),
    ),
)
def test_character_start_handoff_requires_exact_coherent_metadata(
    metadata_key,
    value,
):
    payload = _character_start_handoff()
    payload.metadata[metadata_key] = value

    assert session_module._character_session_identity_from_handoff(payload) is None


@pytest.mark.parametrize(
    "character_id",
    (
        pytest.param("01", id="leading-zero"),
        pytest.param(" 7", id="leading-whitespace"),
        pytest.param("7 ", id="trailing-whitespace"),
        pytest.param("７", id="unicode-digit"),
        pytest.param("0", id="zero"),
        pytest.param("-1", id="negative"),
        pytest.param(7.0, id="float"),
        pytest.param(True, id="bool"),
        pytest.param(7, id="integer"),
        pytest.param("9" * 5000, id="overlong"),
        pytest.param(str(1 << 63), id="signed-64-overflow"),
    ),
)
def test_character_start_handoff_requires_canonical_positive_numeric_wire_id(
    character_id,
):
    payload = _character_start_handoff(character_id=character_id)

    assert session_module._character_session_identity_from_handoff(payload) is None


@pytest.mark.parametrize(
    ("record_id", "target_id"),
    (
        pytest.param(None, "local:character:7", id="missing-record"),
        pytest.param("7", None, id="missing-target"),
        pytest.param("7", "local:character:8", id="conflicting-ids"),
    ),
)
def test_character_start_handoff_requires_matching_record_and_target_ids(
    record_id,
    target_id,
):
    payload = _character_start_handoff()
    payload.metadata["selected_record_id"] = record_id
    payload.metadata["selected_target_id"] = target_id

    assert session_module._character_session_identity_from_handoff(payload) is None


@pytest.mark.asyncio
async def test_local_character_handoff_uses_db_authority_and_scoped_card_service(
    monkeypatch,
):
    authority_id = "local-authority"
    card_model = SimpleNamespace(model_dump=Mock(return_value=_character_card()))
    runtime = _character_handoff_runtime(
        local_authority=authority_id,
        card=card_model,
    )
    runtime.resolver.side_effect = AssertionError(
        "local handoff must not resolve server authority"
    )
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(runtime_backend="local"),
    )

    assert started is True
    runtime.db.get_local_authority_id.assert_called_once_with()
    runtime.db.get_character_card_by_id.assert_not_called()
    runtime.scope_service.get_character.assert_awaited_once_with(7, mode="local")
    card_model.model_dump.assert_called_once_with(mode="json")
    runtime.resolver.assert_not_awaited()
    assert store.create_kwargs is not None
    assert store.create_kwargs["runtime_backend"] == "local"
    assert store.create_kwargs["assistant_kind"] == "character"
    assert store.create_kwargs["assistant_id"] == "7"
    assert store.create_kwargs["assistant_authority_id"] == authority_id
    assert store.create_kwargs["character_id"] == 7
    assert store.session is not None
    assert store.session.local_character_id() == 7
    assert store.session.character_ref() is not None
    assert store.session.character_ref().authority_id == authority_id
    assert store.identity_at_append is not None
    assert store.identity_at_append["character_ref"] == store.session.character_ref()
    assert store.messages == [
        {
            "session_id": store.session.id,
            "role": ConsoleMessageRole.ASSISTANT,
            "content": "Hello User, I am Elara.",
            "persist": True,
        }
    ]


@pytest.mark.asyncio
async def test_local_character_handoff_without_db_authority_falls_back(
    monkeypatch,
):
    runtime = _character_handoff_runtime()
    runtime.db.get_local_authority_id.side_effect = RuntimeError("unavailable")
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(),
    )

    assert started is False
    runtime.scope_service.get_character.assert_not_awaited()
    assert store.session is None


@pytest.mark.asyncio
async def test_local_character_handoff_without_scoped_card_falls_back(monkeypatch):
    runtime = _character_handoff_runtime(card=None)
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(),
    )

    assert started is False
    runtime.scope_service.get_character.assert_awaited_once_with(7, mode="local")
    runtime.db.get_character_card_by_id.assert_not_called()
    assert store.session is None


@pytest.mark.parametrize("runtime_backend", ("local", "server"))
@pytest.mark.parametrize(
    "returned_character_id",
    (
        pytest.param(_DEFAULT_HANDOFF_VALUE, id="missing-id"),
        pytest.param("07", id="noncanonical-id"),
        pytest.param(8, id="mismatched-id"),
    ),
)
@pytest.mark.asyncio
async def test_character_handoff_requires_exact_returned_card_identity(
    monkeypatch,
    runtime_backend,
    returned_character_id,
):
    card = _character_card()
    if returned_character_id is _DEFAULT_HANDOFF_VALUE:
        card.pop("id")
    else:
        card["id"] = returned_character_id
    active_server_id = "configured-target-7" if runtime_backend == "server" else None
    runtime = _character_handoff_runtime(
        active_server_id=active_server_id,
        card=card,
    )

    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(
            runtime_backend=runtime_backend,
            active_server_profile_id=active_server_id,
        ),
    )

    assert started is False
    runtime.scope_service.get_character.assert_awaited_once_with(
        7,
        mode=runtime_backend,
    )
    assert store.session is None
    assert store.messages == []


@pytest.mark.asyncio
async def test_cancel_after_character_session_commit_consumes_without_replay(
    monkeypatch,
):
    runtime = _character_handoff_runtime()
    runtime.app.pending_handoffs = PendingHandoffStore()
    runtime.app.pending_handoffs.stage(
        HandoffChannel.CHAT,
        _character_start_handoff(),
    )
    store = _CharacterHandoffStore()
    screen = _handoff_chat_screen(monkeypatch, runtime.app, store)
    sync_started = asyncio.Event()
    hold_sync = asyncio.Event()

    async def _block_post_commit_sync():
        sync_started.set()
        await hold_sync.wait()

    monkeypatch.setattr(
        screen,
        "_sync_native_console_chat_ui",
        _block_post_commit_sync,
    )

    consume_task = asyncio.create_task(screen._consume_pending_chat_handoff())
    await asyncio.wait_for(sync_started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
    consume_task.cancel()
    await consume_task

    assert store.session is not None
    assert len(store.messages) == 1
    assert not runtime.app.pending_handoffs.has_pending(HandoffChannel.CHAT)

    await screen._consume_pending_chat_handoff()

    assert len(store.messages) == 1
    assert not runtime.app.pending_handoffs.has_pending(HandoffChannel.CHAT)


@pytest.mark.asyncio
async def test_server_character_handoff_scopes_exact_target_without_local_projection(
    monkeypatch,
):
    expected_server_id = "configured-target-7"
    authority_id = "server-user-v1:" + ("a" * 64)
    runtime = _character_handoff_runtime(
        active_server_id=expected_server_id,
        server_authority=authority_id,
    )
    runtime.db.get_local_authority_id.side_effect = AssertionError(
        "server handoff must not use local authority"
    )
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(
            runtime_backend="server",
            active_server_profile_id=expected_server_id,
        ),
    )

    assert started is True
    runtime.resolver.assert_awaited_once_with(expected_server_id=expected_server_id)
    runtime.scope_service.get_character.assert_awaited_once_with(7, mode="server")
    runtime.db.get_local_authority_id.assert_not_called()
    runtime.db.get_character_card_by_id.assert_not_called()
    assert store.create_kwargs is not None
    assert store.create_kwargs["runtime_backend"] == "server"
    assert store.create_kwargs["assistant_kind"] == "character"
    assert store.create_kwargs["assistant_id"] == "7"
    assert store.create_kwargs["assistant_authority_id"] == authority_id
    assert store.create_kwargs["character_id"] is None
    assert store.session is not None
    assert store.session.local_character_id() is None
    assert store.session.character_ref() is not None
    assert store.session.character_ref().source == "server"
    assert store.session.character_ref().authority_id == authority_id
    assert store.session.character_ref().character_id == "7"
    assert store.identity_at_append is not None
    assert store.identity_at_append["character_ref"] == store.session.character_ref()


@pytest.mark.asyncio
async def test_server_identity_failure_still_seeds_unscoped_character_session(
    monkeypatch,
):
    expected_server_id = "configured-target-7"
    runtime = _character_handoff_runtime(
        active_server_id=expected_server_id,
    )
    runtime.resolver.side_effect = RuntimeError("identity unavailable")
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(
            runtime_backend="server",
            active_server_profile_id=expected_server_id,
        ),
    )

    assert started is True
    runtime.resolver.assert_awaited_once_with(expected_server_id=expected_server_id)
    runtime.scope_service.get_character.assert_awaited_once_with(7, mode="server")
    assert store.session is not None
    assert store.session.runtime_backend == "server"
    assert store.session.assistant_kind == "character"
    assert store.session.assistant_authority_id is None
    assert store.session.character_id is None
    assert store.session.character_ref() is None
    assert [message["content"] for message in store.messages] == [
        "Hello User, I am Elara."
    ]


@pytest.mark.parametrize(
    "resolved_authority_id",
    (
        pytest.param(
            "server-user-v2:" + ("a" * 64),
            id="wrong-prefix",
        ),
        pytest.param(
            "server-user-v1:" + ("a" * 63),
            id="short-digest",
        ),
        pytest.param(
            "server-user-v1:" + ("a" * 65),
            id="long-digest",
        ),
        pytest.param(
            "server-user-v1:" + ("A" * 64),
            id="uppercase-digest",
        ),
        pytest.param(7, id="non-string"),
        pytest.param(
            " server-user-v1:" + ("a" * 64),
            id="leading-whitespace",
        ),
        pytest.param(
            "server-user-v1:" + ("a" * 64) + " ",
            id="trailing-whitespace",
        ),
    ),
)
@pytest.mark.asyncio
async def test_server_character_handoff_rejects_malformed_resolver_authority(
    monkeypatch,
    resolved_authority_id,
):
    expected_server_id = "configured-target-7"
    runtime = _character_handoff_runtime(
        active_server_id=expected_server_id,
        server_authority=resolved_authority_id,
    )

    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(
            runtime_backend="server",
            active_server_profile_id=expected_server_id,
        ),
    )

    assert started is True
    assert store.session is not None
    assert store.session.assistant_authority_id is None
    assert store.session.character_ref() is None


@pytest.mark.asyncio
async def test_server_target_mismatch_before_resolver_fetches_nothing(monkeypatch):
    runtime = _character_handoff_runtime(
        active_server_id="different-target",
    )
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(
            runtime_backend="server",
            active_server_profile_id="configured-target-7",
        ),
    )

    assert started is False
    runtime.resolver.assert_not_awaited()
    runtime.scope_service.get_character.assert_not_awaited()
    assert store.session is None


@pytest.mark.asyncio
async def test_server_target_switch_during_resolver_discards_authority(monkeypatch):
    expected_server_id = "configured-target-7"
    runtime = _character_handoff_runtime(
        active_server_id=expected_server_id,
    )

    async def _switch_target(**kwargs):
        assert kwargs == {"expected_server_id": expected_server_id}
        runtime.app.active_server_id = "different-target"
        return "server-user-v1:" + ("b" * 64)

    runtime.resolver.side_effect = _switch_target
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(
            runtime_backend="server",
            active_server_profile_id=expected_server_id,
        ),
    )

    assert started is False
    runtime.resolver.assert_awaited_once_with(expected_server_id=expected_server_id)
    runtime.scope_service.get_character.assert_not_awaited()
    assert store.session is None


@pytest.mark.asyncio
async def test_server_target_switch_during_card_fetch_discards_card(monkeypatch):
    expected_server_id = "configured-target-7"
    runtime = _character_handoff_runtime(
        active_server_id=expected_server_id,
    )

    async def _fetch_then_switch(character_id, *, mode):
        assert (character_id, mode) == (7, "server")
        runtime.app.active_server_id = "different-target"
        return _character_card()

    runtime.scope_service.get_character.side_effect = _fetch_then_switch
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(
            runtime_backend="server",
            active_server_profile_id=expected_server_id,
        ),
    )

    assert started is False
    runtime.resolver.assert_awaited_once_with(expected_server_id=expected_server_id)
    runtime.scope_service.get_character.assert_awaited_once_with(7, mode="server")
    assert store.session is None


@pytest.mark.parametrize(
    "authenticated_transition",
    ("a_to_b", "a_to_b_to_a"),
)
@pytest.mark.asyncio
async def test_server_authenticated_context_change_during_card_fetch_aborts_session(
    monkeypatch,
    authenticated_transition,
):
    expected_server_id = "configured-target-7"
    runtime = _character_handoff_runtime(
        active_server_id=expected_server_id,
    )
    card_fetch_started = asyncio.Event()
    release_card_fetch = asyncio.Event()

    async def _blocked_card_fetch(character_id, *, mode):
        assert (character_id, mode) == (7, "server")
        card_fetch_started.set()
        await release_card_fetch.wait()
        return _character_card()

    runtime.scope_service.get_character.side_effect = _blocked_card_fetch
    handoff = asyncio.create_task(
        _run_character_handoff(
            monkeypatch,
            runtime,
            _character_start_handoff(
                runtime_backend="server",
                active_server_profile_id=expected_server_id,
            ),
        )
    )
    await wait_for_background_signal(
        card_fetch_started,
        handoff,
        what="the character-card fetch",
    )

    runtime.authority_context_state["current"] = SimpleNamespace(account="B")
    if authenticated_transition == "a_to_b_to_a":
        runtime.authority_context_state["current"] = SimpleNamespace(account="A")
    assert runtime.app.active_server_id == expected_server_id

    release_card_fetch.set()
    started, store = await handoff

    assert started is False
    assert runtime.resolve_captures == [runtime.initial_capture]
    runtime.scope_service.get_character.assert_awaited_once_with(7, mode="server")
    assert store.session is None
    assert store.messages == []


@pytest.mark.asyncio
async def test_server_authenticated_context_change_during_resolver_fetches_no_card(
    monkeypatch,
):
    expected_server_id = "configured-target-7"
    runtime = _character_handoff_runtime(
        active_server_id=expected_server_id,
    )

    async def _resolve_then_change(**kwargs):
        assert kwargs == {"expected_server_id": expected_server_id}
        runtime.authority_context_state["current"] = SimpleNamespace(account="B")
        return "server-user-v1:" + ("b" * 64)

    runtime.resolver.side_effect = _resolve_then_change
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(
            runtime_backend="server",
            active_server_profile_id=expected_server_id,
        ),
    )

    assert started is False
    assert runtime.resolve_captures == [runtime.initial_capture]
    runtime.scope_service.get_character.assert_not_awaited()
    assert store.session is None


@pytest.mark.asyncio
async def test_server_authenticated_context_change_immediately_before_commit_aborts(
    monkeypatch,
):
    expected_server_id = "configured-target-7"
    runtime = _character_handoff_runtime(
        active_server_id=expected_server_id,
    )
    store = _CharacterHandoffStore()
    screen = _handoff_chat_screen(monkeypatch, runtime.app, store)

    def _settings_then_change_context():
        runtime.authority_context_state["current"] = SimpleNamespace(account="B")
        return ConsoleSessionSettings(
            provider="anthropic",
            model="claude-3-haiku",
        )

    monkeypatch.setattr(
        screen._session,
        "_default_console_session_settings",
        _settings_then_change_context,
    )
    started = await screen._session._start_character_console_session(
        _character_start_handoff(
            runtime_backend="server",
            active_server_profile_id=expected_server_id,
        )
    )

    assert started is False
    assert runtime.resolve_captures == [runtime.initial_capture]
    assert store.session is None
    assert store.messages == []


@pytest.mark.asyncio
async def test_persona_start_chat_does_not_create_character_session(monkeypatch):
    runtime = _character_handoff_runtime(
        active_server_id="configured-target-7",
    )
    started, store = await _run_character_handoff(
        monkeypatch,
        runtime,
        _character_start_handoff(
            runtime_backend="server",
            selected_kind="persona",
            active_server_profile_id="configured-target-7",
            character_id="p-7",
        ),
    )

    assert started is False
    runtime.resolver.assert_not_awaited()
    runtime.scope_service.get_character.assert_not_awaited()
    assert store.session is None


@pytest.mark.asyncio
async def test_console_inspector_setup_state_explains_blocked_send_without_selection():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _open_console_inspector_rail(console, pilot)

        inspector_text = _visible_text(
            console.query_one("#console-run-inspector-state")
        )
        assert "Setup" in inspector_text
        assert "Blocked impact" in inspector_text
        assert CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL in inspector_text
        assert "Selected Message" not in inspector_text


@pytest.mark.asyncio
async def test_console_composer_setup_blocker_keeps_recovery_outside_input():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 54)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer_text = _visible_text(composer)
        assert ConsoleComposerBar.DRAFT_PLACEHOLDER in composer_text
        assert "Setup required" not in composer_text
        # The composer's own blocked-reason (the Send tooltip) carries the
        # "blocked" impact guidance now instead of the shared Workbench
        # recovery banner (Phase 2 spec, section 2).
        assert not console.query_one("#workbench-recovery-callout").display
        assert console.query_one("#console-send-message", Button).tooltip == (
            "Add API key in Settings > Providers & Models before sending."
        )


@pytest.mark.asyncio
async def test_console_selected_message_updates_inspector_action_guidance():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="first assistant variant",
        )
        store.add_variant(message.id, "second assistant variant")
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, "#console-inspector-selected-message")

        inspector_text = _visible_text(
            console.query_one("#console-run-inspector-state")
        )
        assert "Selected message: Assistant message" in inspector_text
        assert (
            "Message actions: Copy, Edit, Save as..., Regenerate, Continue, Feedback, Delete"
            in inspector_text
        )
        assert (
            "Keyboard: Tab/Shift+Tab cycle actions; Enter activates; Esc clears selection"
            in inspector_text
        )
        assert "Variants: 2 variants, showing 2/2" in inspector_text
        assert "Excerpt: second assistant variant" in inspector_text


@pytest.mark.asyncio
async def test_console_display_only_activity_selection_updates_and_clears_inspector():
    """Inspector resolves only the selected activity in the active projection."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Activity owner")
        owner = store.append_message(
            first.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="The workspace contains two files.",
        )
        marker = store.append_message(
            first.id,
            role=ConsoleMessageRole.TOOL,
            content="a.txt\nb.txt",
            tool_output_full="a.txt\nb.txt",
            activity_presentation=ConsoleActivityPresentation(
                "tool", "fs_list", "success"
            ),
        )
        second = store.create_session(title="Other session", activate=False)
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.display_message(marker.id) is not None
        with pytest.raises(KeyError):
            store.get_message(marker.id)

        transcript.select_message(marker.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-actions-{marker.id}"
        )

        inspector_text = _visible_text(
            console.query_one("#console-run-inspector-state")
        )
        assert "Selected message: Tool message" in inspector_text
        assert "Message actions:" in inspector_text
        assert "Excerpt: a.txt b.txt" in inspector_text
        action_ids = {
            button.id
            for button in console.query(f"#console-message-actions-{marker.id} Button")
        }
        assert action_ids
        assert all(action_id.endswith(marker.id) for action_id in action_ids)

        transcript.select_message(owner.id)
        await console._sync_native_console_chat_ui()
        inspector_text = _visible_text(
            console.query_one("#console-run-inspector-state")
        )
        assert "Selected message: Assistant message" in inspector_text
        assert "Excerpt: The workspace contains two files." in inspector_text

        transcript.select_message(marker.id)
        store.switch_session(second.id)
        await console._sync_native_console_chat_ui()
        assert transcript.selected_message_id is None
        assert "Selected Message" not in _visible_text(
            console.query_one("#console-run-inspector-state")
        )

        store.switch_session(first.id)
        await console._sync_native_console_chat_ui()
        transcript.select_message(marker.id)
        await console._sync_native_console_chat_ui()
        store.delete_message(owner.id)
        await console._sync_native_console_chat_ui()
        assert transcript.selected_message_id is None
        assert transcript.display_message(marker.id) is None
        assert "Selected Message" not in _visible_text(
            console.query_one("#console-run-inspector-state")
        )


@pytest.mark.asyncio
async def test_console_selected_message_feedback_action_records_rating():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-feedback-up-{message.id}"
        )

        await pilot.click(f"#console-message-action-feedback-up-{message.id}")
        await pilot.pause()

    updated = store.get_message(message.id)
    assert updated.feedback == "up"
    assert console._last_console_action.action_id == "feedback-up"
    assert console._last_console_action.visible_copy == "Marked message feedback: up."


@pytest.mark.asyncio
async def test_console_selected_message_delete_action_removes_message_from_transcript():
    app = _build_test_app()
    host = ConsoleHarness(app)

    # TASK-1: widened per the comment on test_console_message_action_
    # keyboard_focus_stays_inside_action_row -- ConsoleHarness's missing CSS
    # bundle (not a CSS bug) forces the row's now-9 buttons to Textual's
    # default min-width of 16, no longer fitting inside 160 cols, which put
    # the coordinate-clicked delete button off the right edge of the
    # terminal.
    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-delete-{message.id}"
        )

        await pilot.click(f"#console-message-action-delete-{message.id}")
        await pilot.pause()

        assert store.messages_for_session(session.id) == [message]
        assert console._last_console_action.action_id == "delete"
        assert (
            console._last_console_action.visible_copy
            == "Press Delete again to remove this message."
        )

        delete_button = console.query_one(
            f"#console-message-action-delete-{message.id}", Button
        )
        delete_button.press()
        await pilot.pause()

    assert store.messages_for_session(session.id) == []
    assert console._last_console_action.action_id == "delete"
    assert (
        console._last_console_action.visible_copy == "Deleted message from transcript."
    )


@pytest.mark.asyncio
async def test_console_original_attempt_delete_clears_parent_and_descendant_previews():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session()
        parent = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Parent repaired [S1]",
        )
        descendant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Descendant repaired [S1]",
        )
        for message, original in (
            (parent, "Parent original"),
            (descendant, "Descendant original"),
        ):
            store.set_citation_presentation(
                message.id,
                ConsoleCitationPresentation(
                    phase=ConsoleCitationPhase.SELECTED,
                    notice_code=ConsoleCitationNoticeCode.REPAIRED,
                ),
            )
            controller._remember_original_attempt(message.id, original)
            console._console_original_attempt_previews[message.id] = original
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(parent.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console,
            pilot,
            f"#console-message-action-delete-{parent.id}",
        )
        delete_button = console.query_one(
            f"#console-message-action-delete-{parent.id}",
            Button,
        )
        delete_button.press()
        await pilot.pause()
        delete_button.press()
        await pilot.pause()

        assert controller._original_attempts == {}
        assert console._console_original_attempt_previews == {}
        with pytest.raises(KeyError):
            store.get_message(parent.id)
        with pytest.raises(KeyError):
            store.get_message(descendant.id)


@pytest.mark.asyncio
async def test_console_delete_confirmation_resets_when_selection_changes():
    app = _build_test_app()
    host = ConsoleHarness(app)

    # TASK-1: widened for the same reason as the other two tests above --
    # ConsoleHarness's missing CSS bundle (not a CSS bug) forces the row's
    # now-9 buttons to Textual's default min-width of 16, no longer fitting
    # inside 160 cols.
    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        first_message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="first answer",
        )
        second_message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="second answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(first_message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-delete-{first_message.id}"
        )

        await pilot.click(f"#console-message-action-delete-{first_message.id}")
        await pilot.pause()
        assert (
            console._last_console_action.visible_copy
            == "Press Delete again to remove this message."
        )

        transcript.select_message(second_message.id)
        await pilot.pause()
        transcript.select_message(first_message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-delete-{first_message.id}"
        )

        delete_button = console.query_one(
            f"#console-message-action-delete-{first_message.id}", Button
        )
        delete_button.press()
        await pilot.pause()

    assert [message.id for message in store.messages_for_session(session.id)] == [
        first_message.id,
        second_message.id,
    ]
    assert console._last_console_action.action_id == "delete"
    assert (
        console._last_console_action.visible_copy
        == "Press Delete again to remove this message."
    )


@pytest.mark.asyncio
async def test_console_selected_message_edit_action_opens_modal_and_saves_content():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        controller = console._ensure_console_chat_controller()
        store.set_citation_presentation(
            message.id,
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.REPAIRED,
            ),
        )
        controller._remember_original_attempt(message.id, "original answer")
        console._console_original_attempt_previews[message.id] = "original answer"
        assert controller.original_attempt_for_message(message.id) == "original answer"
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-edit-{message.id}"
        )

        await pilot.click(f"#console-message-action-edit-{message.id}")
        await _wait_for_selector(
            host.screen_stack[-1], pilot, "#console-edit-message-modal"
        )
        edit_modal = host.screen_stack[-1]
        assert "Editing existing transcript message" in _static_plain_text(
            edit_modal.query_one("#console-edit-message-context", Static)
        )

        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        assert editor.text == "answer"
        editor.text = "edited answer"
        await pilot.click("#console-edit-message-save")
        await pilot.pause()

    assert store.get_message(message.id).content == "edited answer"
    assert controller.original_attempt_for_message(message.id) is None
    assert message.id not in console._console_original_attempt_previews
    assert console._last_console_action.action_id == "edit"
    assert console._last_console_action.visible_copy == "Edited message."


@pytest.mark.asyncio
async def test_console_edit_resend_clears_replaced_descendant_original_attempt():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("new", " reply")
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session()
        user_message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="original question",
        )
        replaced_assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="repaired answer [S1]",
        )
        store.set_citation_presentation(
            replaced_assistant.id,
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.REPAIRED,
            ),
        )
        controller._remember_original_attempt(
            replaced_assistant.id,
            "original answer",
        )
        console._console_original_attempt_previews[replaced_assistant.id] = (
            "original answer"
        )
        assert (
            controller.original_attempt_for_message(replaced_assistant.id)
            == "original answer"
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(user_message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console,
            pilot,
            f"#console-message-action-edit-{user_message.id}",
        )
        await pilot.click(f"#console-message-action-edit-{user_message.id}")
        await _wait_for_selector(
            host.screen_stack[-1],
            pilot,
            "#console-edit-message-modal",
        )
        edit_modal = host.screen_stack[-1]
        edit_modal.query_one(
            "#console-edit-message-body", TextArea
        ).text = "edited question"
        await pilot.click("#console-edit-message-resend")
        await _wait_for_text(console, pilot, "new reply")

        assert controller.original_attempt_for_message(replaced_assistant.id) is None
        assert replaced_assistant.id not in console._console_original_attempt_previews
        assert replaced_assistant.id not in store.active_path_message_ids(session.id)


@pytest.mark.asyncio
async def test_console_selected_message_edit_action_cancel_preserves_content():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-edit-{message.id}"
        )

        await pilot.click(f"#console-message-action-edit-{message.id}")
        await _wait_for_selector(
            host.screen_stack[-1], pilot, "#console-edit-message-modal"
        )
        edit_modal = host.screen_stack[-1]
        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "discard this"
        await pilot.click("#console-edit-message-cancel")
        await pilot.pause()

    assert store.get_message(message.id).content == "answer"


@pytest.mark.asyncio
async def test_console_selected_message_edit_action_blank_save_stays_open_with_error():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-edit-{message.id}"
        )

        await pilot.click(f"#console-message-action-edit-{message.id}")
        await _wait_for_selector(
            host.screen_stack[-1], pilot, "#console-edit-message-modal"
        )
        edit_modal = host.screen_stack[-1]
        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "   "
        await pilot.click("#console-edit-message-save")
        await _wait_for_selector(edit_modal, pilot, "#console-edit-message-error")

        assert (
            "cannot be blank"
            in _static_plain_text(
                edit_modal.query_one("#console-edit-message-error", Static)
            ).lower()
        )
        assert store.get_message(message.id).content == "answer"


@pytest.mark.asyncio
async def test_console_sync_skips_transcript_refresh_when_messages_unchanged(
    monkeypatch,
):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        original_refresh = transcript.refresh_messages
        refresh_calls = 0
        await pilot.pause()

        async def counted_refresh():
            nonlocal refresh_calls
            refresh_calls += 1
            await original_refresh()

        monkeypatch.setattr(transcript, "refresh_messages", counted_refresh)

        await console._sync_native_console_chat_ui()
        baseline_refresh_calls = refresh_calls
        assert baseline_refresh_calls >= 1

        await console._sync_native_console_chat_ui()
        assert refresh_calls == baseline_refresh_calls

        store.add_variant(message.id, "updated answer")
        await console._sync_native_console_chat_ui()
        assert refresh_calls == baseline_refresh_calls + 1


@pytest.mark.asyncio
async def test_console_selected_message_save_as_action_opens_modal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-save-as-{message.id}"
        )

        await pilot.click(f"#console-message-action-save-as-{message.id}")
        await _wait_for_selector(host.screen_stack[-1], pilot, "#console-save-as-modal")

    assert console._last_console_action.action_id == "save-as"


def _install_console_save_service_fakes(app) -> None:
    """Give the test app callable handles for every Save-as destination."""
    app.notes_scope_service = SimpleNamespace(
        save_note=AsyncMock(return_value={"id": "note-1"})
    )
    app.media_db = SimpleNamespace(
        add_media_with_keywords=Mock(return_value=(7, "media-uuid-7", "Media added."))
    )
    app.prompts_db = SimpleNamespace(
        add_prompt=Mock(return_value=(5, "prompt-uuid-5", "Prompt added."))
    )
    app.local_chatbook_service = SimpleNamespace(
        create_chatbook=AsyncMock(return_value={"id": "1", "chatbook_id": 1})
    )


async def _open_save_as_modal_for_message(host, pilot, console, role, content):
    """Append one message, select it, and open its Save as modal."""
    store = console._ensure_console_chat_store()
    session = store.ensure_session()
    message = store.append_message(session.id, role=role, content=content)
    await console._sync_native_console_chat_ui()

    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    transcript.select_message(message.id)
    await console._sync_native_console_chat_ui()
    await _wait_for_selector(
        console, pilot, f"#console-message-action-save-as-{message.id}"
    )

    await pilot.click(f"#console-message-action-save-as-{message.id}")
    await _wait_for_selector(host.screen_stack[-1], pilot, "#console-save-as-modal")
    return message, host.screen_stack[-1]


@pytest.mark.asyncio
async def test_console_save_as_modal_offers_all_wired_destinations():
    app = _build_test_app()
    _install_console_save_service_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _message, save_as_modal = await _open_save_as_modal_for_message(
            host, pilot, console, ConsoleMessageRole.ASSISTANT, "answer"
        )

        assert "Saving selected Assistant message" in _static_plain_text(
            save_as_modal.query_one("#console-save-as-context", Static)
        )
        assert "answer" in _static_plain_text(
            save_as_modal.query_one("#console-save-as-excerpt", Static)
        )
        for destination in ("chatbook", "note", "media", "prompt"):
            assert save_as_modal.query(f"#console-save-as-destination-{destination}")
        modal_text = _visible_text(save_as_modal)
        assert "WIP" not in modal_text
        assert "unavailable" not in modal_text
        assert "No Save as destinations are wired" not in modal_text


@pytest.mark.asyncio
async def test_console_save_as_modal_gates_chatbook_for_user_messages_with_honest_reason():
    app = _build_test_app()
    _install_console_save_service_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _message, save_as_modal = await _open_save_as_modal_for_message(
            host, pilot, console, ConsoleMessageRole.USER, "question"
        )

        for destination in ("note", "media", "prompt"):
            assert save_as_modal.query(f"#console-save-as-destination-{destination}")
        assert not save_as_modal.query("#console-save-as-destination-chatbook")
        gated_copy = _static_plain_text(
            save_as_modal.query_one("#console-save-as-unavailable-chatbook", Static)
        )
        assert (
            "Only assistant responses can be saved as Chatbook artifacts." in gated_copy
        )
        assert "WIP" not in gated_copy
        assert "No Save as destinations are wired" not in _visible_text(save_as_modal)


def test_console_save_as_destinations_gate_on_runtime_services_and_role():
    app = _build_test_app()
    screen = ChatScreen(app)
    _install_console_save_service_fakes(app)
    assistant = SimpleNamespace(role=ConsoleMessageRole.ASSISTANT, content="answer")
    user = SimpleNamespace(role=ConsoleMessageRole.USER, content="question")

    wired = screen._console_save_as_destinations(assistant)
    assert [d.label for d in wired] == ["Chatbook", "Note", "Media", "Prompt"]
    assert all(d.available for d in wired)

    gated = screen._console_save_as_destinations(user)
    chatbook = next(d for d in gated if d.label == "Chatbook")
    assert chatbook.available is False
    assert (
        chatbook.reason
        == "Only assistant responses can be saved as Chatbook artifacts."
    )
    assert [d.label for d in gated if d.available] == ["Note", "Media", "Prompt"]

    app.notes_scope_service = None
    app.media_db = None
    app.prompts_db = None
    app.local_chatbook_service = None
    dark = screen._console_save_as_destinations(assistant)
    assert all(d.available is False for d in dark)
    reasons = {d.label: d.reason for d in dark}
    assert reasons["Note"] == "Notes service is not ready in this session."
    assert reasons["Media"] == "Media library is not ready in this session."
    assert reasons["Prompt"] == "Prompts service is not ready in this session."
    assert (
        reasons["Chatbook"]
        == "Chatbook artifacts service is not ready in this session."
    )
    assert all("WIP" not in reason for reason in reasons.values())


def test_console_save_as_destinations_are_blocked_in_a_temporary_chat():
    """A temporary chat blocks every Save as... destination, regardless of
    service readiness -- the write itself is the problem, not the wiring.

    The control: the same screen with the same services wired, but a
    non-ephemeral session, returns the pre-existing availability.
    """
    from tldw_chatbook.Chat.console_ephemeral import blocked_reason

    app = _build_test_app()
    screen = ChatScreen(app)
    _install_console_save_service_fakes(app)
    assistant = SimpleNamespace(role=ConsoleMessageRole.ASSISTANT, content="answer")

    screen._console_active_session_is_ephemeral = lambda: True
    blocked = screen._console_save_as_destinations(assistant)
    assert all(d.available is False for d in blocked)
    reasons = {d.label: d.reason for d in blocked}
    assert reasons["Note"] == blocked_reason("save-as-note", ephemeral=True)
    assert reasons["Media"] == blocked_reason("save-as-media", ephemeral=True)
    assert reasons["Prompt"] == blocked_reason("save-as-prompt", ephemeral=True)
    assert reasons["Chatbook"] == blocked_reason("save-as-chatbook", ephemeral=True)

    screen._console_active_session_is_ephemeral = lambda: False
    wired = screen._console_save_as_destinations(assistant)
    assert [d.label for d in wired] == ["Chatbook", "Note", "Media", "Prompt"]
    assert all(d.available for d in wired)


def test_console_save_as_destinations_never_show_a_literal_none_reason():
    """F4 (task-9 review): `blocked_reason` returns `str | None`, but the
    reasons dict is typed `dict[str, str]` -- a future registry-key drift
    (a "save-as-<label>" id renamed/removed) must fall back to a real
    sentence, not silently surface the literal string "None"."""
    app = _build_test_app()
    screen = ChatScreen(app)
    _install_console_save_service_fakes(app)
    assistant = SimpleNamespace(role=ConsoleMessageRole.ASSISTANT, content="answer")
    screen._console_active_session_is_ephemeral = lambda: True

    with patch.object(message_module, "blocked_reason", lambda *a, **k: None):
        destinations = screen._console_save_as_destinations(assistant)

    reasons = {d.label: d.reason for d in destinations}
    assert all(
        reason == "Not available in a temporary chat." for reason in reasons.values()
    )
    assert all("None" not in reason for reason in reasons.values())


def test_console_save_as_labels_are_all_registered_in_the_ephemeral_gate():
    """F3 (final-review): a genuinely enumerative check on the registry.

    The three tests in ``Tests/Chat/test_console_ephemeral.py`` that touch
    ``EPHEMERAL_BLOCKED_ACTIONS`` all iterate the registry's OWN keys, so
    none of them can detect an artifact-producing action missing FROM the
    registry -- exactly the failure mode the module docstring claims they
    guard against. This test derives its expected action ids from the REAL
    runtime call site instead: ``_console_save_as_destinations`` builds
    each save-as action id as ``f"save-as-{label.lower()}"`` from its own
    ``("Chatbook", "Note", "Media", "Prompt")`` label list (the one
    dynamically-constructed id family in the whole registry, and the one
    the F4/task-9 review comment right above that call site already flags
    as the most likely to drift). A spy on ``blocked_reason`` records every
    action id that call path actually asks the registry about; if a future
    label is added there without a matching registry row, this fails on
    that id specifically -- unlike the own-keys tests, which would stay
    green.
    """
    from tldw_chatbook.Chat.console_ephemeral import (
        EPHEMERAL_BLOCKED_ACTIONS,
        blocked_reason as real_blocked_reason,
    )

    app = _build_test_app()
    screen = ChatScreen(app)
    _install_console_save_service_fakes(app)
    assistant = SimpleNamespace(role=ConsoleMessageRole.ASSISTANT, content="answer")
    screen._console_active_session_is_ephemeral = lambda: True

    requested_action_ids: list[str] = []

    def spy(action_id, *, ephemeral):
        requested_action_ids.append(action_id)
        return real_blocked_reason(action_id, ephemeral=ephemeral)

    with patch.object(message_module, "blocked_reason", spy):
        destinations = screen._console_save_as_destinations(assistant)

    assert requested_action_ids, (
        "the ephemeral Save-as path never asked the registry anything -- "
        "this test would not catch a missing entry"
    )
    for action_id in requested_action_ids:
        assert action_id in EPHEMERAL_BLOCKED_ACTIONS, (
            f"{action_id!r} is exercised by the real Save-as destinations "
            "path but has no entry in EPHEMERAL_BLOCKED_ACTIONS -- it would "
            "silently fall back to the generic 'Not available in a "
            "temporary chat.' sentence instead of naming the artifact it "
            "writes"
        )
    # The real per-artifact sentences came back, never the generic
    # fallback that masks exactly this kind of registry gap (see
    # test_console_save_as_destinations_never_show_a_literal_none_reason
    # above for the fallback's own contract).
    reasons = {d.label: d.reason for d in destinations}
    assert "Not available in a temporary chat." not in reasons.values()


@pytest.mark.asyncio
async def test_console_selected_message_save_as_note_creates_note_from_message():
    app = _build_test_app()
    app.notes_scope_service = SimpleNamespace(
        save_note=AsyncMock(
            return_value={
                "id": "note-1",
                "title": "Console message",
                "content": "answer",
            }
        )
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        message, _modal = await _open_save_as_modal_for_message(
            host, pilot, console, ConsoleMessageRole.ASSISTANT, "answer"
        )
        await _wait_for_selector(
            host.screen_stack[-1], pilot, "#console-save-as-destination-note"
        )
        await pilot.click("#console-save-as-destination-note")
        await pilot.pause()

    app.notes_scope_service.save_note.assert_awaited_once()
    kwargs = app.notes_scope_service.save_note.await_args.kwargs
    # Title carries the conversation title plus a short UTC date, e.g.
    # "Console message — Chat 1 (2026-07-11)" (UAT: no more generic titles).
    assert kwargs["title"].startswith("Console message — Chat 1 (")
    assert kwargs["title"].endswith(")")
    assert len(kwargs["title"]) <= 80
    assert kwargs["scope"] == "local_note"
    assert kwargs["content"] == "answer"
    assert kwargs["note_id"] is None
    assert kwargs["version"] is None
    assert kwargs["user_id"] == "default_user"
    assert kwargs["workspace_id"] is None
    assert kwargs["keywords"] == ["console"]
    assert console._last_console_action.action_id == "save-as-note"
    assert console._last_console_action.visible_copy == "Saved message as Note."


@pytest.mark.asyncio
async def test_console_selected_message_save_as_media_adds_plaintext_media():
    app = _build_test_app()
    _install_console_save_service_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        message, _modal = await _open_save_as_modal_for_message(
            host, pilot, console, ConsoleMessageRole.ASSISTANT, "answer"
        )
        await _wait_for_selector(
            host.screen_stack[-1], pilot, "#console-save-as-destination-media"
        )
        await pilot.click("#console-save-as-destination-media")
        await pilot.pause()

    add_media = app.media_db.add_media_with_keywords
    add_media.assert_called_once()
    kwargs = add_media.call_args.kwargs
    assert kwargs["media_type"] == "plaintext"
    assert kwargs["content"] == "answer"
    assert kwargs["keywords"] == ["console"]
    assert kwargs["title"].startswith("Console assistant message — Chat 1 (")
    assert len(kwargs["title"]) <= 80
    assert console._last_console_action.action_id == "save-as-media"
    assert console._last_console_action.visible_copy == "Saved message as Media."


@pytest.mark.asyncio
async def test_console_selected_message_save_as_prompt_persists_prompt():
    app = _build_test_app()
    _install_console_save_service_fakes(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        message, _modal = await _open_save_as_modal_for_message(
            host, pilot, console, ConsoleMessageRole.ASSISTANT, "answer"
        )
        await _wait_for_selector(
            host.screen_stack[-1], pilot, "#console-save-as-destination-prompt"
        )
        await pilot.click("#console-save-as-destination-prompt")
        await pilot.pause()

    add_prompt = app.prompts_db.add_prompt
    add_prompt.assert_called_once()
    kwargs = add_prompt.call_args.kwargs
    assert kwargs["name"].startswith("Console message — Chat 1 (")
    assert kwargs["system_prompt"] == "answer"
    assert kwargs["author"] == "Console"
    assert kwargs["keywords"] == ["console"]
    assert kwargs["overwrite"] is False
    assert "Chat 1" in kwargs["details"]
    assert console._last_console_action.action_id == "save-as-prompt"
    assert console._last_console_action.visible_copy == "Saved message as Prompt."


@pytest.mark.asyncio
async def test_console_save_as_prompt_retries_with_suffix_on_name_conflict():
    from tldw_chatbook.DB.Prompts_DB import ConflictError as PromptsConflictError

    app = _build_test_app()
    _install_console_save_service_fakes(app)
    calls = []

    def add_prompt(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise PromptsConflictError("Prompt already exists.")
        return (9, "prompt-uuid-9", "added")

    app.prompts_db = SimpleNamespace(add_prompt=add_prompt)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _message, _modal = await _open_save_as_modal_for_message(
            host, pilot, console, ConsoleMessageRole.ASSISTANT, "answer"
        )
        await _wait_for_selector(
            host.screen_stack[-1], pilot, "#console-save-as-destination-prompt"
        )
        await pilot.click("#console-save-as-destination-prompt")
        await pilot.pause()

    assert len(calls) == 2
    assert calls[1]["name"] == f"{calls[0]['name']} (2)"
    assert console._last_console_action.action_id == "save-as-prompt"


@pytest.mark.asyncio
async def test_console_selected_message_save_as_chatbook_registers_console_artifact():
    app = _build_test_app()
    _install_console_save_service_fakes(app)
    owner_request = object()
    app.citation_artifact_ownership_coordinator = SimpleNamespace(
        reconcile_pending=Mock()
    )
    host = ConsoleHarness(app)

    with patch(
        "tldw_chatbook.UI.Console_Modules.message.resolve_console_artifact_owner_request",
        return_value=owner_request,
    ):
        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")
            message, _modal = await _open_save_as_modal_for_message(
                host, pilot, console, ConsoleMessageRole.ASSISTANT, "answer"
            )
            await _wait_for_selector(
                host.screen_stack[-1], pilot, "#console-save-as-destination-chatbook"
            )
            await pilot.click("#console-save-as-destination-chatbook")
            await pilot.pause()

    create_chatbook = app.local_chatbook_service.create_chatbook
    create_chatbook.assert_awaited_once()
    kwargs = create_chatbook.await_args.kwargs
    assert kwargs["name"].startswith("Console message — Chat 1 (")
    assert kwargs["tags"] == ["console", "artifact"]
    metadata = kwargs["metadata"]
    assert metadata["artifact_source"] == "console"
    assert metadata["artifact_kind"] == "assistant-response"
    assert metadata["content"] == "answer"
    assert metadata["message_id"] == message.id
    assert kwargs["provenance_owner_request"] is owner_request
    app.citation_artifact_ownership_coordinator.reconcile_pending.assert_called_once_with(
        limit=1
    )
    assert console._last_console_action.action_id == "save-as-chatbook"
    assert (
        console._last_console_action.visible_copy
        == "Saved message as Chatbook artifact."
    )


@pytest.mark.asyncio
async def test_console_save_as_media_failure_notifies_without_crashing():
    app = _build_test_app()
    _install_console_save_service_fakes(app)
    app.media_db = SimpleNamespace(
        add_media_with_keywords=exploding_double(
            RuntimeError("disk full"),
            reason="the failing media write must actually be attempted",
            awaitable=False,
        )
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        notifications = []
        app.notify = lambda *args, **kwargs: notifications.append((args, kwargs))
        message, _modal = await _open_save_as_modal_for_message(
            host, pilot, console, ConsoleMessageRole.ASSISTANT, "answer"
        )
        await _wait_for_selector(
            host.screen_stack[-1], pilot, "#console-save-as-destination-media"
        )
        await pilot.click("#console-save-as-destination-media")
        await pilot.pause()

        # Screen stays alive and responsive after the failed save.
        assert console.query("#console-native-transcript")

    failure_messages = [
        args[0]
        for args, kwargs in notifications
        if args and "Save as Media failed" in str(args[0])
    ]
    assert failure_messages
    assert "disk full" in failure_messages[0]
    assert console._last_console_action.action_id == "save-as"


@pytest.mark.asyncio
async def test_console_failed_stream_renders_inline_retry_and_recovers():
    gateway = FailThenRecoverGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")

        store = console._ensure_console_chat_store()
        # The failure copy now lands in a trailing system row; retry targets
        # the failed assistant message itself.
        failed = next(
            message
            for message in reversed(store.messages_for_session(store.active_session_id))
            if message.role is ConsoleMessageRole.ASSISTANT
            and message.status == "failed"
        )
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(failed.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-retry-{failed.id}"
        )
        retry_button = console.query_one(
            f"#console-message-action-retry-{failed.id}", Button
        )
        assert str(retry_button.label) == "Retry"
        assert retry_button.tooltip == "Retry the failed response."

        await pilot.click(f"#console-message-action-retry-{failed.id}")
        await _wait_for_text(console, pilot, "recovered")

    assert store.get_message(failed.id).status == "complete"


@pytest.mark.asyncio
async def test_console_continue_action_streams_new_message_from_selected_turn():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("hel", "lo")
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="prompt",
        )
        source = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="seed",
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(source.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-continue-{source.id}"
        )

        await pilot.click(f"#console-message-action-continue-{source.id}")
        await _wait_for_text(console, pilot, "hello")

        messages = store.messages_for_session(session.id)
        assert messages[-1].role is ConsoleMessageRole.ASSISTANT
        assert messages[-1].content == "hello"
        assert messages[-1].id != source.id
        assert transcript.selected_message_id is None
        assert not list(console.query(f"#console-message-actions-{source.id}"))


@pytest.mark.asyncio
async def test_console_regenerate_action_streams_selected_variant():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("hel", "lo")
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="prompt",
        )
        source = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="seed",
        )
        controller = console._ensure_console_chat_controller()
        store.set_citation_presentation(
            source.id,
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.REPAIRED,
            ),
        )
        controller._remember_original_attempt(source.id, "original seed")
        console._console_original_attempt_previews[source.id] = "original seed"
        assert controller.original_attempt_for_message(source.id) == "original seed"
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(source.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-regenerate-{source.id}"
        )

        await pilot.click(f"#console-message-action-regenerate-{source.id}")
        await _wait_for_text(console, pilot, "hello")

        # TASK-6: regenerate forks a persisted SIBLING node and streams into
        # it, rather than replacing the anchor's content in place as an
        # in-message variant. The anchor is untouched and drops off the
        # active path; the new sibling is the active leaf and carries the
        # freshly streamed text.
        unchanged_source = store.get_message(source.id)
        assert unchanged_source.content == "seed"
        assert unchanged_source.variants is None
        assert source.id not in store.active_path_message_ids(session.id)

        new_leaf_id = store.active_leaf(session.id)
        assert new_leaf_id != source.id
        new_sibling = store.get_message(new_leaf_id)
        assert new_sibling.content == "hello"
        assert new_sibling.variants is None
        assert controller.original_attempt_for_message(source.id) is None
        assert source.id not in console._console_original_attempt_previews


@pytest.mark.asyncio
async def test_console_rejected_regenerate_preserves_original_attempt_preview():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = BlockedGateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _select_llamacpp_console(console)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session(title="Chat 1")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="prompt",
        )
        source = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="repaired answer [S1]",
        )
        store.set_citation_presentation(
            source.id,
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.REPAIRED,
            ),
        )
        controller._remember_original_attempt(source.id, "original answer")
        console._console_original_attempt_previews[source.id] = "original answer"
        assert controller.original_attempt_for_message(source.id) == "original answer"
        seeded = store.get_message(source.id)
        assert seeded.citation_presentation is not None
        assert seeded.citation_presentation.original_attempt_available is True
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(source.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console,
            pilot,
            f"#console-message-action-regenerate-{source.id}",
        )
        await pilot.click(f"#console-message-action-regenerate-{source.id}")
        await _wait_for_text(console, pilot, "Provider blocked")

        retained = store.get_message(source.id)
        assert controller.original_attempt_for_message(source.id) == "original answer"
        assert retained.citation_presentation is not None
        assert retained.citation_presentation.original_attempt_available is True
        assert console._console_original_attempt_previews == {
            source.id: "original answer"
        }
        assert source.id in store.active_path_message_ids(session.id)
        assert [
            message.id
            for message in store.messages_for_session(session.id)
            if message.role is ConsoleMessageRole.ASSISTANT
        ] == [source.id]


@pytest.mark.asyncio
async def test_console_sibling_swipe_buttons_navigate_between_regenerated_siblings():
    """TASK-7: `<`/`>` navigate persisted SIBLING nodes (Task 6's regenerate
    fork), not the old in-memory ``ConsoleVariantSet`` cycling this test used
    to exercise via ``store.add_variant`` -- superseded now that the gate is
    ``sibling_count``-based. Also pins the `(n/m)` counter this task adds."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        a1 = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="seed",
        )
        a2 = store.create_sibling(
            a1.id, role=ConsoleMessageRole.ASSISTANT, content="updated answer"
        )
        await console._sync_native_console_chat_ui()

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(a2.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-variant-previous-{a2.id}"
        )
        await _wait_for_selector(
            console, pilot, f"#console-message-action-variant-next-{a2.id}"
        )

        previous_button = console.query_one(
            f"#console-message-action-variant-previous-{a2.id}", Button
        )
        next_button = console.query_one(
            f"#console-message-action-variant-next-{a2.id}", Button
        )
        assert previous_button.disabled is False
        assert next_button.disabled is True
        row_text = _message_row_plain_text(console, a2.id)
        assert "updated answer" in row_text
        assert "(2/2)" in row_text

        await pilot.click(f"#console-message-action-variant-previous-{a2.id}")
        await _wait_for_text(console, pilot, "seed")
        row_text = _message_row_plain_text(console, a1.id)
        assert "(1/2)" in row_text
        # task-501: the selection FOLLOWS the swipe onto the landed sibling
        # (a2 dropped off the active path, which would have cleared the old
        # selection), so the action row stays available and repeated `<`/`>`
        # presses work without re-clicking the row. Other selection-clearing
        # actions ("continue" etc.) keep their existing clear-on-swap rule.
        # Re-query the transcript: a recompose may have remounted it since
        # the pre-click capture (the handoff is remount-proof; a stale widget
        # reference in the test would not be).
        await pilot.pause()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.selected_message_id == a1.id
        await _wait_for_selector(
            console, pilot, f"#console-message-action-variant-next-{a1.id}"
        )

        # Repeated swipe with NO re-click: `>` from the still-selected a1 row
        # returns to a2 and the selection follows again.
        await pilot.click(f"#console-message-action-variant-next-{a1.id}")
        await _wait_for_text(console, pilot, "updated answer")
        await pilot.pause()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.selected_message_id == a2.id
        await _wait_for_selector(
            console, pilot, f"#console-message-action-variant-previous-{a2.id}"
        )

        # Swipe back to the FIRST sibling (again without a re-click -- the
        # selection is sitting on a2 from the swipe above) and pin the
        # boundary disabled states on a1's action row.
        await pilot.click(f"#console-message-action-variant-previous-{a2.id}")
        await _wait_for_text(console, pilot, "seed")
        await pilot.pause()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.selected_message_id == a1.id
        await _wait_for_selector(
            console, pilot, f"#console-message-action-variant-next-{a1.id}"
        )
        previous_button = console.query_one(
            f"#console-message-action-variant-previous-{a1.id}", Button
        )
        next_button = console.query_one(
            f"#console-message-action-variant-next-{a1.id}", Button
        )
        assert previous_button.disabled is True
        assert next_button.disabled is False

        await pilot.click(f"#console-message-action-variant-next-{a1.id}")
        await _wait_for_text(console, pilot, "updated answer")
        row_text = _message_row_plain_text(console, a2.id)
        assert "(2/2)" in row_text
        # task-501: the swipe keeps the selection on the landed sibling (the
        # pre-task-501 rule cleared it, forcing a re-click between swipes).
        await pilot.pause()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.selected_message_id == a2.id


@pytest.mark.asyncio
async def test_console_native_tab_strip_creates_and_switches_sessions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")
        assert "Chat 2" in _visible_text(console)

        await pilot.click(f"#console-session-tab-{first.id}")

        assert store.active_session_id == first.id
        assert "Chat 1" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_native_tab_switch_restores_transcript_messages():
    """Verify native tab switching restores the prior session transcript."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        store.append_message(
            first.id,
            role=ConsoleMessageRole.USER,
            content="first tab user prompt",
        )
        store.append_message(
            first.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="first tab assistant reply",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_text(console, pilot, "first tab assistant reply")

        previous = store.active_session_id
        await pilot.click("#console-new-chat-tab")
        second = await _wait_for_active_session_change(store, pilot, previous)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")
        await _wait_for_text(console, pilot, "Get started")
        assert "first tab assistant reply" not in _visible_text(console)

        await pilot.click(f"#console-session-tab-{first.id}")

        await _wait_for_active_session(store, pilot, first.id)
        await _wait_for_text(console, pilot, "first tab user prompt")
        await _wait_for_text(console, pilot, "first tab assistant reply")


@pytest.mark.asyncio
async def test_console_workspace_conversation_switch_restores_transcript_messages():
    """Verify workspace conversation switching restores the prior transcript."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        store.append_message(
            first.id,
            role=ConsoleMessageRole.USER,
            content="workspace row user prompt",
        )
        store.append_message(
            first.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="workspace row assistant reply",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_text(console, pilot, "workspace row assistant reply")

        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        previous = store.active_session_id
        await pilot.click("#console-new-chat-tab")
        second = await _wait_for_active_session_change(store, pilot, previous)
        assert second != first.id
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        # Ready console: the fresh empty tab shows the ready line, not the
        # blocking setup modal (which only appears when setup is incomplete).
        await _wait_for_text(console, pilot, "Ready — type a message to begin.")
        assert "workspace row assistant reply" not in _visible_text(console)

        await _click_console_workspace_conversation_for_session(
            console,
            pilot,
            store,
            first.id,
        )

        await _wait_for_active_session(store, pilot, first.id)
        await _wait_for_text(console, pilot, "workspace row user prompt")
        await _wait_for_text(console, pilot, "workspace row assistant reply")


def _configure_grouped_browser_workspaces(app):
    app.app_config.setdefault("console", {}).setdefault("conversation_browser", {})[
        "collapsed_groups"
    ] = {}
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    return service


def _browser_group_toggle(console, group_id: str) -> Button:
    for button in console.query(Button):
        if getattr(button, "group_id", None) == group_id:
            return button
    groups = [
        (getattr(button, "id", None), getattr(button, "group_id", None))
        for button in console.query(Button)
        if str(getattr(button, "id", "")).startswith("console-conversation-browser")
    ]
    raise AssertionError(
        f"Browser group toggle {group_id!r} not found. Groups: {groups!r}"
    )


def _browser_star_button(console, conversation_id: str) -> Button:
    for button in console.query(".console-conversation-star"):
        if getattr(button, "conversation_id", None) == conversation_id:
            return button
    stars = [
        (getattr(button, "id", None), getattr(button, "conversation_id", None))
        for button in console.query(".console-conversation-star")
    ]
    raise AssertionError(
        f"Star button for {conversation_id!r} not found. Stars: {stars!r}"
    )


#: Wall-clock budget for browser-row renders (TASK-1900). Deliberately
#: generous: this waits on a RENDER, and render latency scales with machine
#: load in a way an iteration count cannot express.
_BROWSER_ROW_RENDER_TIMEOUT = 15.0


async def _wait_for_browser_conversation_row(console, pilot, conversation_id: str):
    """Wait for a browser row to render, on a wall-clock deadline.

    TASK-1900. This used to spin a fixed `for _ in range(80)` over
    `pilot.pause(0.05)` and call it four seconds. It is not four seconds: on
    a busy machine each pause overruns AND the render itself takes longer,
    so the budget shrinks exactly when it needs to grow. That made
    `test_console_conversation_browser_search_ignores_stale_results` fail
    about one run in five, and 3/3 with four CPU burners alongside -- while
    the screen's `_console_conversation_browser_rows` state was already
    correct. The test was right, the clock was wrong.

    Args:
        console: The mounted Console screen.
        pilot: Its `Pilot`, used to let the event loop settle between polls.
        conversation_id: Row to wait for.

    Returns:
        The rendered row widget.

    Raises:
        AssertionError: If the row has not rendered within the deadline.
    """
    deadline = time.monotonic() + _BROWSER_ROW_RENDER_TIMEOUT
    while time.monotonic() < deadline:
        for row in console.query(".console-workspace-conversation-row"):
            if getattr(row, "conversation_id", None) == conversation_id:
                return row
        await pilot.pause(0.05)
    rows = [
        (getattr(row, "conversation_id", None), _widget_text(row))
        for row in console.query(".console-workspace-conversation-row")
    ]
    raise AssertionError(f"Browser row {conversation_id!r} not found. Rows: {rows!r}")


async def _wait_for_browser_render(pilot, predicate, describe) -> None:
    """Wait for a browser-rail render condition on the same wall-clock deadline.

    task-14920: the TASK-1900 diagnosis above ("on a busy machine each pause
    overruns AND the render itself takes longer, so the budget shrinks exactly
    when it needs to grow") applies to every browser-rail assertion made after
    a single fixed ``pilot.pause``, not only to "has the row appeared". Two
    such assertions were observed failing in a whole-file run under load while
    passing 8/8 in isolation; this is the same deadline, reused, so the
    assertion that follows keeps its original claim and only stops guessing at
    a settle time.

    Args:
        pilot: The Console `Pilot`, used to let the event loop settle.
        predicate: Zero-arg callable returning True once the render landed.
        describe: Zero-arg callable returning failure context for the message.

    Raises:
        AssertionError: If the condition never held within the deadline.
    """
    deadline = time.monotonic() + _BROWSER_ROW_RENDER_TIMEOUT
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Browser render never settled: {describe()}")


class _InputChangedEvent:
    def __init__(self, value: str) -> None:
        self.value = value

    def stop(self) -> None:
        return None


async def _set_console_conversation_browser_search(console, pilot, query: str) -> None:
    search = console.query_one("#console-workspace-conversation-search", Input)
    search.value = query
    console.on_console_workspace_conversation_search_changed(_InputChangedEvent(query))
    await pilot.pause(0.3)


@pytest.mark.asyncio
async def test_console_conversation_browser_lists_all_workspace_groups():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="workspace-a-chat",
        role="workspace-thread",
        title="Workspace A saved",
    )
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="workspace-b-chat",
        role="workspace-thread",
        title="Workspace B saved",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        store = console._ensure_console_chat_store()
        default_session = store.create_session(
            title="Global chat",
            workspace_id=DEFAULT_WORKSPACE_ID,
        )
        store.switch_session(default_session.id)
        await console._sync_native_console_chat_ui()

        visible_text = _visible_text(console)
        assert "Starred" in visible_text
        assert "Workspaces" in visible_text
        assert "Workspace A" in visible_text
        assert "Workspace B" in visible_text
        assert "Chats" in visible_text
        assert "Global chat" in visible_text
        assert "Storage" in visible_text
        assert "Server" in visible_text
        assert len(console.query("#console-workspace-conversations-toggle")) == 0
        assert (
            len(
                console.query("#console-conversation-browser-section-toggle-workspaces")
            )
            == 1
        )
        assert any(
            getattr(button, "group_id", None) == "workspace:ws-a"
            for button in console.query(".console-workspace-conversations-toggle")
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_search_filters_all_groups():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="alpha-a",
        role="workspace-thread",
        title="Alpha in Workspace A",
    )
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="needle-b",
        role="workspace-thread",
        title="Needle in Workspace B",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        await pilot.click("#console-workspace-conversation-search")
        search = console.query_one("#console-workspace-conversation-search", Input)
        search.value = "needle"
        console.on_console_workspace_conversation_search_changed(
            _InputChangedEvent("needle")
        )
        for _ in range(40):
            row_texts = _console_workspace_conversation_texts(console)
            if any(
                getattr(row, "conversation_id", None) == "needle-b"
                for row in console.query(".console-workspace-conversation-row")
            ):
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError(f"Needle row not found. Rows: {row_texts!r}")
        row_texts = _console_workspace_conversation_texts(console)

        normalized_texts = [" ".join(text.split()) for text in row_texts]
        assert any("Needle in Workspa" in text for text in normalized_texts)
        assert any("Workspace B" in text for text in normalized_texts)
        assert all("Alpha in Workspace A" not in text for text in normalized_texts)


@pytest.mark.asyncio
async def test_console_browser_selecting_non_default_workspace_native_session_switches_active_workspace():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    app.app_config["console"]["conversation_browser"]["collapsed_groups"][
        "workspace:ws-b"
    ] = False
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Workspace A Chat", workspace_id="ws-a")
        second = store.create_session(title="Workspace B Chat", workspace_id="ws-b")
        store.append_message(
            second.id,
            role=ConsoleMessageRole.USER,
            content="Workspace B prompt",
        )
        store.switch_session(first.id)
        await console._sync_native_console_chat_ui()

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            f"native:{second.id}",
        )
        await _wait_for_active_session(store, pilot, second.id)
        await _wait_for_text(console, pilot, "Workspace B prompt")

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == "ws-b"
        assert store.workspace_context.active_workspace_id == "ws-b"


@pytest.mark.asyncio
async def test_console_browser_selecting_non_default_workspace_persisted_row_switches_active_workspace_before_resume():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    app.app_config["console"]["conversation_browser"]["collapsed_groups"][
        "workspace:ws-b"
    ] = False
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="persisted-ws-b",
        role="workspace-thread",
        title="Workspace B saved",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "persisted-ws-b": {
                "conversation": {
                    "id": "persisted-ws-b",
                    "title": "Workspace B saved",
                },
                "root_threads": [
                    {
                        "id": "message-ws-b",
                        "role": "user",
                        "content": "Workspace B prompt",
                    }
                ],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        store = console._ensure_console_chat_store()

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "persisted-ws-b",
        )
        await _wait_for_text(console, pilot, "Workspace B prompt")
        session = next(
            session
            for session in store.sessions()
            if session.persisted_conversation_id == "persisted-ws-b"
        )

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == "ws-b"
        assert store.workspace_context.active_workspace_id == "ws-b"
        assert session.workspace_id == "ws-b"


@pytest.mark.asyncio
async def test_console_browser_selecting_duplicate_membership_row_ignores_other_workspace_open_session():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    app.app_config["console"]["conversation_browser"]["collapsed_groups"][
        "workspace:ws-b"
    ] = False
    for workspace_id in ("ws-a", "ws-b"):
        service.link_membership(
            workspace_id,
            item_type="conversation",
            item_id="shared-open-chat",
            role="workspace-thread",
            title="Shared open chat",
        )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "shared-open-chat": {
                "conversation": {
                    "id": "shared-open-chat",
                    "title": "Shared open chat",
                },
                "root_threads": [
                    {
                        "id": "shared-open-message",
                        "role": "user",
                        "content": "Workspace B shared prompt",
                    }
                ],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        store = console._ensure_console_chat_store()
        open_ws_a = store.ensure_session(title="Shared open chat", workspace_id="ws-a")
        open_ws_a.persisted_conversation_id = "shared-open-chat"
        store.switch_session(open_ws_a.id)
        await console._sync_native_console_chat_ui()
        # _sync_native_console_chat_ui reentrancy-guards concurrent calls: if one
        # is already running when this call lands, it just flags a follow-up sync
        # and returns immediately, deferring the real rebuild to an unawaited
        # background worker. Give that worker a chance to finish so the browser
        # rows below are the final settled set rather than a transient one whose
        # underlying widget can be unmounted while the click below is in flight.
        await pilot.pause(0.2)
        ws_b_row = _workspace_conversation_row_by_key(
            console,
            "workspace:ws-b:conversation:shared-open-chat",
        )

        assert ws_b_row is not None
        assert not _row_is_selected(ws_b_row)
        assert (
            console._workspace._find_console_browser_row(
                "workspace:missing:conversation:shared-open-chat",
                conversation_id="shared-open-chat",
            )
            is None
        )

        await _click_console_workspace_conversation_for_row_key(
            console,
            pilot,
            "workspace:ws-b:conversation:shared-open-chat",
        )
        await _wait_for_text(console, pilot, "Workspace B shared prompt")
        sessions = [
            session
            for session in store.sessions()
            if session.persisted_conversation_id == "shared-open-chat"
        ]
        active_session = store.switch_session(store.active_session_id)

        assert len(sessions) == 2
        assert active_session.workspace_id == "ws-b"
        selected_shared_rows = [
            row
            for row in console.query(".console-workspace-conversation-row")
            if getattr(row, "conversation_id", None) == "shared-open-chat"
            and _row_is_selected(row)
        ]
        assert len(selected_shared_rows) == 1
        assert (
            getattr(selected_shared_rows[0], "native_session_id", None)
            == active_session.id
        )
        selected_native_rows = [
            row
            for row in console._workspace._native_console_browser_rows(
                "shared-open-chat"
            )
            if row.conversation_id == "shared-open-chat" and row.selected
        ]
        assert len(selected_native_rows) == 1
        assert selected_native_rows[0].native_session_id == active_session.id
        selected_membership_rows = [
            row
            for row in console._workspace._membership_console_browser_rows(
                "shared-open-chat"
            )
            if row.conversation_id == "shared-open-chat" and row.selected
        ]
        assert len(selected_membership_rows) == 1
        assert selected_membership_rows[0].workspace_id == "ws-b"
        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == "ws-b"
        assert store.workspace_context.active_workspace_id == "ws-b"


@pytest.mark.asyncio
async def test_console_browser_selecting_default_native_session_uses_private_scratch():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Workspace A Chat", workspace_id="ws-a")
        second = store.create_session(
            title="Default Chat",
            workspace_id=DEFAULT_WORKSPACE_ID,
        )
        store.append_message(
            second.id,
            role=ConsoleMessageRole.USER,
            content="Default prompt",
        )
        store.switch_session(first.id)
        await console._sync_native_console_chat_ui()

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            f"native:{second.id}",
        )
        await _wait_for_active_session(store, pilot, second.id)
        await _wait_for_text(console, pilot, "Default prompt")

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == DEFAULT_WORKSPACE_ID
        assert store.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID
        assert (
            _static_plain_text(
                console.query_one("#console-workspace-runtime-label", Static)
            )
            == "Local files"
        )
        assert "Private scratch" in _static_plain_text(
            console.query_one("#console-workspace-runtime-value", Static)
        )


@pytest.mark.asyncio
async def test_console_browser_selecting_default_persisted_row_uses_private_scratch():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    service.link_membership(
        DEFAULT_WORKSPACE_ID,
        item_type="conversation",
        item_id="persisted-default",
        role="workspace-thread",
        title="Default saved",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "persisted-default": {
                "conversation": {
                    "id": "persisted-default",
                    "title": "Default saved",
                },
                "root_threads": [
                    {
                        "id": "message-default",
                        "role": "user",
                        "content": "Default persisted prompt",
                    }
                ],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        store = console._ensure_console_chat_store()

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "persisted-default",
        )
        await _wait_for_text(console, pilot, "Default persisted prompt")
        session = next(
            session
            for session in store.sessions()
            if session.persisted_conversation_id == "persisted-default"
        )

        active = service.get_active_workspace()
        assert active is not None
        assert active.workspace_id == DEFAULT_WORKSPACE_ID
        assert store.workspace_context.active_workspace_id == DEFAULT_WORKSPACE_ID
        assert session.workspace_id == DEFAULT_WORKSPACE_ID
        assert (
            _static_plain_text(
                console.query_one("#console-workspace-runtime-label", Static)
            )
            == "Local files"
        )
        assert "Private scratch" in _static_plain_text(
            console.query_one("#console-workspace-runtime-value", Static)
        )


@pytest.mark.asyncio
async def test_console_browser_selecting_global_persisted_row_switches_context_to_global():
    """task-15120 (owner ruling): the workspace context FOLLOWS the conversation.

    A user keeps conversations open across multiple workspaces at once, and
    selecting one switches the context to that conversation's workspace -- for
    a global-scoped conversation, the global scope, on BOTH layers. The
    registry's stable representation of "no explicit workspace" is the
    built-in Default workspace (`ensure_default_workspace` floors every
    context read to it -- capability-less by design), so a global
    conversation lands the registry on Default while the store's context
    reads "global". What can no longer happen is what task-15120 measured:
    the PREVIOUS workspace (ws-a) staying active, its capabilities bleeding
    into a global conversation.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.set_active_workspace("ws-a")
    app.chat_conversation_scope_service = SyncSearchableConversationService(
        {
            "global-persisted": {
                "conversation": {
                    "id": "global-persisted",
                    "title": "Global saved",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [
                    {
                        "id": "message-global",
                        "role": "user",
                        "content": "Global persisted prompt",
                    }
                ],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        store = console._ensure_console_chat_store()
        before = service.get_active_workspace()
        assert before is not None
        assert before.workspace_id == "ws-a"

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "global-persisted",
        )
        await _wait_for_text(console, pilot, "Global persisted prompt")
        session = next(
            session
            for session in store.sessions()
            if session.persisted_conversation_id == "global-persisted"
        )

        after = service.get_active_workspace()
        assert after is not None and after.workspace_id == DEFAULT_WORKSPACE_ID, (
            "opening a global conversation must land the registry on the "
            "capability-less Default workspace, not stay on "
            f"{getattr(after, 'workspace_id', None)!r}"
        )
        assert (
            store.workspace_context.active_workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID
        )
        assert session.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID


@pytest.mark.asyncio
async def test_console_conversation_browser_search_counts_only_matching_local_rows():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = None
    app.local_chat_conversation_service = None
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="needle-local",
        role="workspace-thread",
        title="Needle Local Match",
    )
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="other-local",
        role="workspace-thread",
        title="Other Local Row",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        await _set_console_conversation_browser_search(console, pilot, "needle")

        for _ in range(80):
            status = console.query_one(
                "#console-workspace-conversation-search-status",
                Static,
            )
            if _static_plain_text(status):
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError("Conversation browser search status did not render")

        row_texts = _console_workspace_conversation_texts(console)
        assert _static_plain_text(status) == "1 match"
        assert any("Needle Local" in text for text in row_texts)
        assert all("Other Local Row" not in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_conversation_browser_keeps_multi_workspace_memberships():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    app.app_config["console"]["conversation_browser"]["collapsed_groups"][
        "workspace:ws-b"
    ] = False
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="shared-conversation",
        role="workspace-thread",
        title="Shared Conversation",
    )
    service.link_membership(
        "ws-b",
        item_type="conversation",
        item_id="shared-conversation",
        role="workspace-thread",
        title="Shared Conversation",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        rows = [
            row
            for row in console.query(".console-workspace-conversation-row")
            if getattr(row, "conversation_id", None) == "shared-conversation"
        ]
        stars = [
            button
            for button in console.query(".console-conversation-star")
            if getattr(button, "conversation_id", None) == "shared-conversation"
        ]

        assert len(rows) == 2
        assert {getattr(row, "workspace_id", None) for row in rows} == {"ws-a", "ws-b"}
        assert {getattr(row, "row_key", None) for row in rows} == {
            "workspace:ws-a:conversation:shared-conversation",
            "workspace:ws-b:conversation:shared-conversation",
        }
        assert len(stars) == 2
        assert {getattr(button, "row_key", None) for button in stars} == {
            "workspace:ws-a:conversation:shared-conversation",
            "workspace:ws-b:conversation:shared-conversation",
        }


@pytest.mark.asyncio
async def test_console_conversation_browser_dedupes_membership_and_persisted_same_workspace():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="same-workspace-conversation",
        role="workspace-thread",
        title="Membership Title",
    )
    app.local_chat_conversation_service = SyncSearchableConversationService(
        {
            "same-workspace-conversation": {
                "conversation": {
                    "id": "same-workspace-conversation",
                    "title": "Persisted Title",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        rows = [
            row
            for row in console.query(".console-workspace-conversation-row")
            if (
                getattr(row, "conversation_id", None) == "same-workspace-conversation"
                and getattr(row, "workspace_id", None) == "ws-a"
            )
        ]
        stars = [
            button
            for button in console.query(".console-conversation-star")
            if (
                getattr(button, "conversation_id", None)
                == "same-workspace-conversation"
                and getattr(button, "row_key", None)
                == "workspace:ws-a:conversation:same-workspace-conversation"
            )
        ]

        assert len(rows) == 1
        assert getattr(rows[0], "row_key", None) == (
            "workspace:ws-a:conversation:same-workspace-conversation"
        )
        assert "Membership Title" in _normalized_row_text(rows[0])
        assert len(stars) == 1


@pytest.mark.asyncio
async def test_console_conversation_browser_search_ignores_stale_results():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = SlowFirstSearchableConversationService(
        {
            "stale-alpha": {
                "conversation": {
                    "id": "stale-alpha",
                    "title": "Stale Alpha",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
            "fresh-beta": {
                "conversation": {
                    "id": "fresh-beta",
                    "title": "Fresh Beta",
                    "workspace_id": "ws-b",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        console._console_conversation_browser_query = "alpha"
        console._console_conversation_browser_search_token += 1
        stale_token = console._console_conversation_browser_search_token
        stale_task = asyncio.create_task(
            console._workspace._refresh_console_conversation_browser_search(
                "alpha", stale_token
            )
        )
        for _ in range(40):
            if app.chat_conversation_scope_service.started.is_set():
                break
            await pilot.pause(0.05)
        assert app.chat_conversation_scope_service.started.is_set()

        console._console_conversation_browser_query = "beta"
        console._console_conversation_browser_search_token += 1
        app.chat_conversation_scope_service.release.set()
        await stale_task

        console._console_conversation_browser_query = "beta"
        console._console_conversation_browser_search_token += 1
        fresh_token = console._console_conversation_browser_search_token
        await console._workspace._refresh_console_conversation_browser_search(
            "beta", fresh_token
        )

        # TASK-1900: assert the CLAIM first, on state that is correct
        # synchronously once the refresh returns. The widget check below waits
        # on a render, so when it was the only assertion a slow machine looked
        # exactly like a stale result winning -- the failure said "row not
        # found. Rows: [stale-alpha]" while the state already held only
        # fresh-beta. Now a real regression fails here and names itself, and a
        # render that is merely late fails below saying so.
        assert [
            row.conversation_id for row in console._console_conversation_browser_rows
        ] == ["fresh-beta"], "the stale search result overwrote the fresh one"

        await _wait_for_browser_conversation_row(console, pilot, "fresh-beta")
        row_texts = _console_workspace_conversation_texts(console)
        assert all("Stale Alpha" not in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_conversation_browser_group_collapse_persists_locally():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="collapse-chat",
        role="workspace-thread",
        title="Collapse Target",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        assert "Collapse Target" in _visible_text(console)

        _browser_group_toggle(console, "workspace:ws-a").press()
        await pilot.pause(0.1)

        assert all(
            "Collapse Target" not in " ".join(text.split())
            for text in _console_workspace_conversation_texts(console)
        )
        collapsed_groups = app.app_config["console"]["conversation_browser"][
            "collapsed_groups"
        ]
        assert collapsed_groups["workspace:ws-a"] is True

        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert all(
            "Collapse Target" not in " ".join(text.split())
            for text in _console_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_workspaces_section_collapse_persists_locally():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="section-collapse-chat",
        role="workspace-thread",
        title="Section Collapse Target",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        assert any(
            "Section Collapse" in " ".join(text.split())
            for text in _console_workspace_conversation_texts(console)
        )

        _browser_group_toggle(console, "section:workspaces").press()
        await pilot.pause(0.1)

        assert all(
            "Section Collapse" not in " ".join(text.split())
            for text in _console_workspace_conversation_texts(console)
        )
        collapsed_groups = app.app_config["console"]["conversation_browser"][
            "collapsed_groups"
        ]
        assert collapsed_groups["section:workspaces"] is True

        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert all(
            "Section Collapse" not in " ".join(text.split())
            for text in _console_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_starred_section_updates_from_row_action():
    app = _build_test_app()
    marks = FakeConversationLocalMarksService()
    app.conversation_local_marks_service = marks
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="star-target",
        role="workspace-thread",
        title="Star Target",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        _browser_star_button(console, "star-target").press()

        def _star_target_rows() -> list[str]:
            return [
                _widget_text(row)
                for row in console.query(".console-workspace-conversation-row")
                if getattr(row, "conversation_id", None) == "star-target"
            ]

        # task-14920: `pilot.pause(0.1)` was a guess at how long the rail takes
        # to rebuild into its starred + workspace sections; under whole-file
        # load it is not long enough and the rows list is still empty.
        await _wait_for_browser_render(
            pilot,
            lambda: len(_star_target_rows()) >= 2,
            lambda: f"star-target rows never reached 2: {_star_target_rows()!r}",
        )

        assert marks.is_starred("star-target") is True
        rows = _star_target_rows()
        assert len(rows) >= 2
        assert any("Star Target" in row for row in rows)


@pytest.mark.asyncio
async def test_console_conversation_browser_keeps_starred_row_in_normal_group():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService(
        ("starred-normal",)
    )
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="starred-normal",
        role="workspace-thread",
        title="Starred Normal",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        rows = [
            _widget_text(row)
            for row in console.query(".console-workspace-conversation-row")
            if getattr(row, "conversation_id", None) == "starred-normal"
        ]
        assert len(rows) == 2
        assert all("Starred Normal" in row for row in rows)


@pytest.mark.asyncio
async def test_console_conversation_browser_marks_unavailable_keeps_browsing_enabled():
    app = _build_test_app()
    app.conversation_local_marks_service = None
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="marks-unavailable-chat",
        role="workspace-thread",
        title="Marks Unavailable Chat",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        visible_text = _visible_text(console)
        assert "Starred" in visible_text
        assert "Workspaces" in visible_text
        assert "Chats" in visible_text
        assert "Local stars unavailable" in visible_text
        assert "Marks Unavailable Chat" in visible_text
        star = _browser_star_button(console, "marks-unavailable-chat")
        assert star.disabled is True


@pytest.mark.asyncio
async def test_console_conversation_browser_default_includes_sync_persisted_rows():
    app = _build_test_app()
    app.conversation_local_marks_service = None
    _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = SyncSearchableConversationService(
        {
            "global-default": {
                "conversation": {
                    "id": "global-default",
                    "title": "Global persisted default",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
            "workspace-default": {
                "conversation": {
                    "id": "workspace-default",
                    "title": "Workspace A persisted default",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        global_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "global-default",
        )
        workspace_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "workspace-default",
        )

        visible_text = _visible_text(console)
        assert "Starred" in visible_text
        assert "Workspaces" in visible_text
        assert "Chats" in visible_text
        assert "Global persisted" in _normalized_row_text(global_row)
        assert "Workspace A persi" in _normalized_row_text(workspace_row)
        assert any(
            call.get("scope_type") == "global"
            for call in app.chat_conversation_scope_service.list_calls
        )
        assert any(
            call.get("scope_type") == "workspace" and call.get("workspace_id") == "ws-b"
            for call in app.chat_conversation_scope_service.list_calls
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_default_prefers_sync_local_service():
    app = _build_test_app()
    app.conversation_local_marks_service = None
    _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "async-scope-default": {
                "conversation": {
                    "id": "async-scope-default",
                    "title": "Async scope default should not block local rows",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
        }
    )
    app.local_chat_conversation_service = SyncSearchableConversationService(
        {
            "local-global-default": {
                "conversation": {
                    "id": "local-global-default",
                    "title": "Local global default",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
            "local-workspace-default": {
                "conversation": {
                    "id": "local-workspace-default",
                    "title": "Local Workspace A default",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        global_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-global-default",
        )
        workspace_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-workspace-default",
        )

        assert "Local global default" in _normalized_row_text(global_row)
        assert "Local Workspace" in _normalized_row_text(workspace_row)
        assert any(
            call.get("scope_type") == "global"
            for call in app.local_chat_conversation_service.list_calls
        )
        assert any(
            call.get("scope_type") == "workspace" and call.get("workspace_id") == "ws-a"
            for call in app.local_chat_conversation_service.list_calls
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_default_omits_mode_for_local_service():
    app = _build_test_app()
    app.conversation_local_marks_service = None
    _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "async-scope-default": {
                "conversation": {
                    "id": "async-scope-default",
                    "title": "Async scope fallback",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
        }
    )
    app.local_chat_conversation_service = NoModeSyncSearchableConversationService(
        {
            "local-no-mode-global": {
                "conversation": {
                    "id": "local-no-mode-global",
                    "title": "No mode local global",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
            "local-no-mode-workspace": {
                "conversation": {
                    "id": "local-no-mode-workspace",
                    "title": "No mode Workspace A",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        global_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-no-mode-global",
        )
        workspace_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-no-mode-workspace",
        )

        assert "No mode local global" in _normalized_row_text(global_row)
        assert "No mode Workspace" in _normalized_row_text(workspace_row)
        assert all(
            "mode" not in call
            for call in app.local_chat_conversation_service.list_calls
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_search_omits_mode_for_local_service():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    _configure_grouped_browser_workspaces(app)
    app.chat_conversation_scope_service = None
    app.local_chat_conversation_service = NoModeSyncSearchableConversationService(
        {
            "local-search-global": {
                "conversation": {
                    "id": "local-search-global",
                    "title": "Needle local global",
                    "scope_type": "global",
                    "workspace_id": None,
                },
                "root_threads": [],
            },
            "local-search-workspace": {
                "conversation": {
                    "id": "local-search-workspace",
                    "title": "Needle local Workspace A",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        await _set_console_conversation_browser_search(console, pilot, "needle")

        global_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-search-global",
        )
        workspace_row = await _wait_for_browser_conversation_row(
            console,
            pilot,
            "local-search-workspace",
        )

        assert "Needle local global" in _normalized_row_text(global_row)
        assert "Needle local Wor" in _normalized_row_text(workspace_row)
        for _ in range(80):
            if console._console_conversation_browser_total is not None:
                break
            await pilot.pause(0.05)
        else:
            raise AssertionError("Debounced persisted search did not finish")
        cached_row_ids = {
            row.conversation_id for row in console._console_conversation_browser_rows
        }
        assert console._console_conversation_browser_error == ""
        assert "local-search-global" in cached_row_ids
        assert "local-search-workspace" in cached_row_ids
        assert all(
            "mode" not in call
            for call in app.local_chat_conversation_service.list_calls
        )


@pytest.mark.asyncio
async def test_console_conversation_browser_long_list_keeps_readiness_rows_reachable():
    app = _build_test_app()
    app.conversation_local_marks_service = FakeConversationLocalMarksService()
    service = _configure_grouped_browser_workspaces(app)
    for index in range(30):
        service.link_membership(
            "ws-a",
            item_type="conversation",
            item_id=f"long-chat-{index}",
            role="workspace-thread",
            title=f"Long Chat {index:02d}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        # Storage/Server-handoff readiness rows now live in the collapsible
        # Details section beneath the conversation browser; expand it.
        if not console._current_console_rail_state().details_open:
            console._toggle_console_rail_section("details")
        await pilot.pause()
        conversation_list = console.query_one("#console-workspace-conversations")
        # TASK-715: default server features collapse to one line; it and the
        # Handoff section title are the lower-rail reachability anchors now.
        server_line = console.query_one("#console-workspace-server-features-collapsed")
        handoff_title = console.query_one("#console-workspace-handoff-title")

        assert conversation_list.region.height > 0
        assert server_line.region.y > conversation_list.region.y
        assert handoff_title.region.y >= server_line.region.y
        visible_text = _visible_text(console)
        assert "Storage" in visible_text
        assert "Server features" in visible_text


@pytest.mark.asyncio
async def test_console_new_chat_tab_appears_in_workspace_conversation_rail():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-chat-1",
        role="workspace-thread",
        title="Chat 1",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        first.persisted_conversation_id = "persisted-chat-1"
        await console._sync_native_console_chat_ui()

        await _wait_for_workspace_conversation_text(console, pilot, "Chat 1")

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert any("Chat 1" in text for text in row_texts)


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_includes_all_workspace_persisted_results():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    other_workspace = service.create_workspace(
        workspace_id="ws-other-search",
        name="Other Search",
    )
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "persisted-alpha": {
                "conversation": {
                    "id": "persisted-alpha",
                    "title": "Alpha persisted conversation",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [],
            },
            "other-alpha": {
                "conversation": {
                    "id": "other-alpha",
                    "title": "Alpha other workspace",
                    "workspace_id": other_workspace.workspace_id,
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )

        await _set_console_conversation_browser_search(console, pilot, "alpha")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Alpha persisted",
            selected=False,
        )
        row_texts = _console_workspace_conversation_texts(console)
        assert any("Alpha persisted" in " ".join(text.split()) for text in row_texts)
        assert any(
            getattr(row, "conversation_id", None) == "other-alpha"
            for row in console.query(".console-workspace-conversation-row")
        )
        workspace_calls = [
            call
            for call in app.chat_conversation_scope_service.list_calls
            if call.get("scope_type") == "workspace"
        ]
        assert any(
            call.get("workspace_id") == active_workspace.workspace_id
            for call in workspace_calls
        )
        assert any(
            call.get("workspace_id") == other_workspace.workspace_id
            for call in workspace_calls
        )


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_selection_keeps_query_active():
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "select-alpha": {
                "conversation": {
                    "id": "select-alpha",
                    "title": "Select Alpha",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [
                    {
                        "id": "select-alpha-message",
                        "conversation_id": "select-alpha",
                        "role": "user",
                        "sender": "user",
                        "content": "selected alpha prompt",
                        "children": [],
                    }
                ],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        await _set_console_conversation_browser_search(console, pilot, "alpha")
        await _wait_for_workspace_conversation_text(
            console, pilot, "Select Alpha", selected=False
        )

        await _click_console_workspace_conversation_for_id(
            console, pilot, "select-alpha"
        )

        await _wait_for_text(console, pilot, "selected alpha prompt")
        search = console.query_one("#console-workspace-conversation-search", Input)
        assert search.value == "alpha"
        assert "Select Alpha" in _static_plain_text(
            console.query_one("#console-workspace-selected-conversation", Static)
        )


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_blank_selection_keeps_composer_focus():
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="blank-focus-chat",
        role="workspace-thread",
        title="Blank focus chat",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "blank-focus-chat": {
                "conversation": {
                    "id": "blank-focus-chat",
                    "title": "Blank focus chat",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [
                    {
                        "id": "blank-focus-message",
                        "conversation_id": "blank-focus-chat",
                        "role": "user",
                        "sender": "user",
                        "content": "blank focus prompt",
                        "children": [],
                    }
                ],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Blank focus chat",
            selected=False,
        )

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "blank-focus-chat",
        )

        await _wait_for_text(console, pilot, "blank focus prompt")
        await pilot.pause(0.2)
        search = console.query_one("#console-workspace-conversation-search", Input)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        assert console.app.focused is composer
        assert console.app.focused is not search


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_selection_invalidates_pending_worker():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chat_conversation_scope_service = SlowFirstSearchableConversationService({})
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Slow Alpha")
        first.title = "Slow Alpha"
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")
        second_session = store.switch_session(second)
        first.workspace_id = second_session.workspace_id
        await console._sync_native_console_chat_ui()

        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        await _set_console_conversation_browser_search(console, pilot, "alpha")
        await _wait_for_workspace_conversation_text(
            console, pilot, "Slow Alpha", selected=False
        )
        for _ in range(40):
            if app.chat_conversation_scope_service.started.is_set():
                break
            await pilot.pause(0.05)
        assert app.chat_conversation_scope_service.started.is_set()
        stale_token = console._console_workspace_conversation_search_token

        await _click_console_workspace_conversation_for_session(
            console,
            pilot,
            store,
            first.id,
        )

        assert console._console_workspace_conversation_search_token > stale_token
        app.chat_conversation_scope_service.release.set()
        await pilot.pause(0.5)
        assert "Slow Alpha" in _static_plain_text(
            console.query_one("#console-workspace-selected-conversation", Static)
        )
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Slow Alpha",
            selected=True,
        )
        assert any(
            "Slow Alpha" in text
            for text in _selected_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_workspace_switch_clears_conversation_search_and_restores_collapse_preference():
    app = _build_test_app()
    service = app.workspace_registry_service
    workspace_a = service.get_active_workspace()
    workspace_b = service.create_workspace(
        workspace_id="ws-search-reset", name="Search Reset"
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        await _set_console_conversation_browser_search(console, pilot, "alpha")
        _browser_group_toggle(console, "section:chats").press()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversation-search")) == 1
        stale_token = console._console_conversation_browser_search_token

        service.set_active_workspace(workspace_b.workspace_id)
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        state = console._workspace._build_console_workspace_context_state()
        assert len(console.query("#console-workspace-conversation-search")) == 1
        assert console._console_conversation_browser_search_token == stale_token + 1
        assert console._console_conversation_browser_search_timer is None
        assert console._console_conversation_browser_query == ""
        assert state.conversation_browser is not None
        assert state.conversation_browser.query == ""
        assert state.conversation_section is not None
        assert state.conversation_section.query == ""
        assert (
            console.query_one("#console-workspace-conversation-search", Input).value
            == ""
        )

        service.set_active_workspace(workspace_a.workspace_id)
        console._sync_console_workspace_context()
        await pilot.pause(0.1)
        assert len(console.query("#console-workspace-conversation-search")) == 1


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_shows_cap_and_empty_copy():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    conversations = {
        f"topic-{index}": {
            "conversation": {
                "id": f"topic-{index}",
                "title": f"Topic conversation {index:02d}",
                "workspace_id": active_workspace.workspace_id,
            },
            "root_threads": [],
        }
        for index in range(60)
    }
    app.chat_conversation_scope_service = SearchableConversationService(conversations)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )

        await _set_console_conversation_browser_search(console, pilot, "topic")
        await _wait_for_text(console, pilot, "60 matches")
        await _wait_for_text(console, pilot, "Showing")

        console.query_one("#console-workspace-conversation-search", Input)
        await _set_console_conversation_browser_search(console, pilot, "missing")
        await _wait_for_text(console, pilot, "No workspace conversations.")


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_ignores_stale_workspace_results():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.create_workspace(workspace_id="ws-stale-b", name="Stale B")
    slow_service = SlowFirstSearchableConversationService(
        {
            "stale-a": {
                "conversation": {
                    "id": "stale-a",
                    "title": "Stale Alpha",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )
        await pilot.pause()
        app.chat_conversation_scope_service = slow_service
        await _set_console_conversation_browser_search(console, pilot, "Alpha")
        await asyncio.wait_for(
            slow_service.started.wait(),
            timeout=_ASYNC_SETTLE_TIMEOUT,
        )
        stale_token = console._console_workspace_conversation_search_token

        service.set_active_workspace("ws-stale-b")
        console._sync_console_workspace_context()
        assert console._console_workspace_conversation_search_token > stale_token

        slow_service.release.set()
        await pilot.pause(0.5)
        assert all(
            row.title != "Stale Alpha"
            for row in console._console_conversation_browser_rows
        )


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_blank_query_clears_error_cache():
    app = _build_test_app()
    app.chat_conversation_scope_service = FailingSearchConversationService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )

        await _set_console_conversation_browser_search(console, pilot, "fail")
        await _wait_for_text(
            console,
            pilot,
            "Workspace conversation search is unavailable.",
        )

        search = console.query_one("#console-workspace-conversation-search", Input)
        search.value = ""
        console.on_console_workspace_conversation_search_changed(_InputChangedEvent(""))

        assert console._console_workspace_conversation_search_rows == ()
        assert console._console_workspace_conversation_search_total is None
        assert console._console_workspace_conversation_search_error == ""
        await pilot.pause(0.3)


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_shows_local_rows_before_slow_persisted_search():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="member-alpha",
        role="workspace-thread",
        title="Alpha membership conversation",
    )
    app.chat_conversation_scope_service = SlowSearchConversationService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console,
            pilot,
            "#console-workspace-conversation-search",
        )

        await _set_console_conversation_browser_search(console, pilot, "alpha")
        try:
            for _ in range(40):
                if app.chat_conversation_scope_service.started.is_set():
                    break
                await pilot.pause(0.05)
            assert app.chat_conversation_scope_service.started.is_set()

            # task-14920: the local rows land a render pass after the worker
            # starts, so asserting immediately is a race under whole-file load.
            # The window is still the one under test -- `release` is only set
            # in the `finally` below, so the persisted search cannot have
            # returned while this waits (asserted right after).
            await _wait_for_browser_render(
                pilot,
                lambda: "1 match" in _visible_text(console),
                lambda: (
                    "'1 match' never rendered while the persisted search "
                    f"was still pending: {_visible_text(console)[:400]!r}"
                ),
            )
            assert not app.chat_conversation_scope_service.release.is_set()
            assert "1 match" in _visible_text(console)
            row_texts = _console_workspace_conversation_texts(console)
            assert any(
                "Alpha membership" in " ".join(text.split()) for text in row_texts
            )
        finally:
            app.chat_conversation_scope_service.release.set()
            await pilot.pause(0.2)


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_filters_all_workspace_memberships():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    other_workspace = service.create_workspace(
        workspace_id="ws-other-search", name="Other Search"
    )
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="member-alpha",
        role="workspace-thread",
        title="Alpha membership conversation",
    )
    service.link_membership(
        other_workspace.workspace_id,
        item_type="conversation",
        item_id="member-other-alpha",
        role="workspace-thread",
        title="Alpha other workspace",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )

        await _set_console_conversation_browser_search(console, pilot, "alpha")
        await _wait_for_text(console, pilot, "matches")
        await _wait_for_workspace_conversation_text(
            console, pilot, "Alpha membership", selected=False
        )
        row_texts = _console_workspace_conversation_texts(console)
        assert any("Alpha membership" in " ".join(text.split()) for text in row_texts)
        assert any(
            getattr(row, "conversation_id", None) == "member-other-alpha"
            for row in console.query(".console-workspace-conversation-row")
        )
        assert "matches" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_uses_current_workspace_context():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-search-a", name="Search A")
    service.create_workspace(workspace_id="ws-search-b", name="Search B")
    service.set_active_workspace("ws-search-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        store = console._ensure_console_chat_store()
        assert store.workspace_context.active_workspace_id == "ws-search-a"

        service.set_active_workspace("ws-search-b")

        assert (
            console._workspace._active_console_workspace_id_for_conversation_search()
            == "ws-search-b"
        )


@pytest.mark.asyncio
async def test_console_workspace_tree_restores_and_persists_disclosure_preferences():
    app = _build_test_app()
    app.workspace_registry_service.create_workspace(
        workspace_id="ws-disclosure", name="Disclosure"
    )
    console_config = app.app_config.setdefault("console", {})
    browser_config = console_config.setdefault("conversation_browser", {})
    browser_config["expanded_workspace_ids"] = []
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-tree")
        tree = console.query_one("#console-workspace-tree", ConsoleWorkspaceTree)
        for _ in range(40):
            if "ws-disclosure" in tree.workspace_nodes:
                break
            await pilot.pause(0.05)

        node = tree.workspace_nodes["ws-disclosure"]
        assert node.is_collapsed

        node.expand()
        await pilot.pause()

        assert browser_config["expanded_workspace_ids"] == ["ws-disclosure"]


@pytest.mark.asyncio
async def test_console_workspace_conversation_list_reserves_two_line_rows_with_margin():
    """Verify conversation list height accounts for two-line rows plus margin."""
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    for index in range(3):
        service.link_membership(
            active_workspace.workspace_id,
            item_type="conversation",
            item_id=f"saved-chat-{index}",
            role="workspace-thread",
            title=f"Saved Chat {index}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        conversation_list = console.query_one("#console-workspace-conversations")
        assert conversation_list.styles.height.value >= 9


@pytest.mark.asyncio
async def test_console_new_chat_tab_promotes_active_native_session_in_workspace_rail():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    for index in range(5):
        service.link_membership(
            active_workspace.workspace_id,
            item_type="conversation",
            item_id=f"persisted-chat-{index}",
            role="workspace-thread",
            title=f"Older chat {index + 1}",
        )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert "Chat 2" in row_texts[0]
        first_row = next(iter(console.query(".console-workspace-conversation-row")))
        assert _row_is_selected(first_row)


@pytest.mark.asyncio
async def test_console_workspace_new_conversation_button_is_not_under_composer():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.1)

        button = console.query_one("#console-new-chat-tab", Button)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        hit_x = button.region.x + max(0, button.region.width // 2)
        hit_y = button.region.y + max(0, button.region.height // 2)
        hit_widget, _region = console.get_widget_at(hit_x, hit_y)

        assert button.region.y + button.region.height <= composer.region.y
        assert hit_widget is button


@pytest.mark.asyncio
async def test_console_workspace_new_conversation_button_is_hit_target_in_named_workspace():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        await pilot.pause(0.1)

        button = console.query_one("#console-new-chat-tab", Button)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        hit_x = button.region.x + max(0, button.region.width // 2)
        hit_y = button.region.y + max(0, button.region.height // 2)
        hit_widget, _region = console.get_widget_at(hit_x, hit_y)

        assert button.region.y + button.region.height <= composer.region.y
        assert hit_widget is button


@pytest.mark.asyncio
async def test_console_workspace_rail_new_conversation_creates_default_workspace_session():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    assert active_workspace is not None
    assert active_workspace.workspace_id == DEFAULT_WORKSPACE_ID
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        assert host.screen_stack[-1] is console

        active_session = next(
            session for session in store.sessions() if session.id == second
        )
        assert active_session.workspace_id == DEFAULT_WORKSPACE_ID
        scratch = console._console_runtime().scratch_spaces.snapshot(second)
        assert scratch.root.is_dir()
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert any(
            "Chat 2" in text for text in _selected_workspace_conversation_texts(console)
        )
        assert (
            _static_plain_text(
                console.query_one("#console-workspace-runtime-label", Static)
            )
            == "Local files"
        )
        assert "Private scratch" in _static_plain_text(
            console.query_one("#console-workspace-runtime-value", Static)
        )
        # TASK-715: factory-default sync/server/ACP rows collapse into one line.
        assert (
            "not configured"
            in _static_plain_text(
                console.query_one(
                    "#console-workspace-server-features-collapsed", Static
                )
            ).lower()
        )
        visible_text = _visible_text(console)
        assert (
            "Workspace conversation creation lands in a later slice" not in visible_text
        )


@pytest.mark.asyncio
async def test_console_workspace_rail_new_conversation_stays_scoped_to_active_workspace():
    app = _build_test_app()
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-new-chat-tab")
        store = console._ensure_console_chat_store()
        first_session_id = store.active_session_id

        await pilot.click("#console-new-chat-tab")
        session_id = store.active_session_id
        assert session_id is not None
        assert session_id != first_session_id
        active_session = next(
            session for session in store.sessions() if session.id == session_id
        )
        assert active_session.workspace_id == "ws-a"
        active_title = active_session.title

        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            active_title,
            selected=True,
        )
        assert any(
            active_title in text
            for text in _selected_workspace_conversation_texts(console)
        )

        console.query_one("#console-change-workspace", Button).press()
        modal_screen = await _wait_for_workspace_switcher_modal(host, pilot)
        switch_button = next(
            button
            for button in modal_screen.query(Button)
            if str(button.label) == "Workspace B"
        )
        switch_button.press()
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, "#console-workspace-context")

        assert service.get_active_workspace().workspace_id == "ws-b"
        assert all(
            active_title not in row_text
            for row_text in _console_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_workspace_conversation_row_switches_native_session():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 1",
            selected=False,
        )
        await _click_console_workspace_conversation_for_session(
            console, pilot, store, first.id
        )

        assert store.active_session_id == first.id
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 1",
            selected=True,
        )
        assert any(
            "Chat 1" in text for text in _selected_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_workspace_conversation_row_resumes_persisted_conversation():
    """Resume a saved workspace conversation directly from the Console rail."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-chat-1",
        role="workspace-thread",
        title="Saved research chat",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "persisted-chat-1": {
                "conversation": {
                    "id": "persisted-chat-1",
                    "title": "Saved research chat",
                    "workspace_id": active_workspace.workspace_id,
                },
                "root_threads": [
                    {
                        "id": "persisted-message-1",
                        "conversation_id": "persisted-chat-1",
                        "role": "user",
                        "sender": "user",
                        "content": "resume saved user prompt",
                        "children": [
                            {
                                "id": "persisted-message-2",
                                "conversation_id": "persisted-chat-1",
                                "sender": "Research Bot",
                                "content": "resume saved assistant reply",
                                "children": [],
                            }
                        ],
                    }
                ],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Saved research chat",
            selected=False,
        )

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "persisted-chat-1",
        )

        await _wait_for_text(console, pilot, "resume saved user prompt")
        await _wait_for_text(console, pilot, "resume saved assistant reply")
        store = console._ensure_console_chat_store()
        active_session = store.switch_session(store.active_session_id)
        assert active_session.persisted_conversation_id == "persisted-chat-1"
        assert active_session.title == "Saved research chat"
        assert active_session.workspace_id == active_workspace.workspace_id
        assistant_messages = [
            message
            for message in store.messages_for_session(active_session.id)
            if message.content == "resume saved assistant reply"
        ]
        assert assistant_messages[-1].role is ConsoleMessageRole.ASSISTANT
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Saved research chat",
            selected=True,
        )
        assert any(
            "Saved research chat" in text
            for text in _selected_workspace_conversation_texts(console)
        )
        selected_row = _workspace_conversation_row_by_id(console, "persisted-chat-1")
        assert selected_row is not None
        selected_row_label = str(selected_row.label)
        assert "\n" in selected_row_label
        assert "active session" in selected_row_label
        assert selected_row.has_class("console-workspace-conversation-row-selected")
        console._set_console_rail_preference(right_open=True, notify_on_failure=False)
        await pilot.pause(0.1)
        inspector_text = _visible_text(console.query_one("#console-right-rail"))
        assert "Selected Conversation" in inspector_text
        assert "Selected conversation: Saved research chat" in inspector_text
        assert "Conversation source: saved conversation" in inspector_text
        assert "Resume state: restored from persisted-chat-1" in inspector_text
        assert "Workspace: Default" in inspector_text
        assert app.chat_conversation_scope_service.calls == [
            {
                "conversation_id": "persisted-chat-1",
                "mode": "local",
                # Task 8: resume loads the full tree with raised caps so a long
                # or branchy conversation is not truncated.
                "depth_cap": 10_000,
                "root_limit": 10_000,
            }
        ]


@pytest.mark.asyncio
async def test_console_resume_restores_server_character_identity_without_local_lookup():
    """Server opaque identity comes only from the selected persisted row."""
    scoped_authority = "server-user-v1:" + ("c" * 64)
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "server-scoped": {
                "conversation": {
                    "id": "server-scoped",
                    "title": "Scoped server character",
                    "runtime_backend": "server",
                    "assistant_kind": "character",
                    "assistant_id": "opaque-character",
                    "assistant_authority_id": scoped_authority,
                    "character_id": None,
                },
                "root_threads": [],
            },
            "server-unscoped": {
                "conversation": {
                    "id": "server-unscoped",
                    "title": "Unscoped server character",
                    "runtime_backend": "server",
                    "assistant_kind": "character",
                    "assistant_id": "other-opaque-character",
                    "assistant_authority_id": None,
                    "character_id": None,
                },
                "root_threads": [],
            },
            "malformed-local-scalars": {
                "conversation": {
                    "id": "malformed-local-scalars",
                    "title": "Malformed local character",
                    "runtime_backend": "local",
                    "assistant_kind": "character",
                    "assistant_id": 7,
                    "assistant_authority_id": 123,
                    "character_id": 7,
                },
                "root_threads": [],
            },
            "noncanonical-local-id": {
                "conversation": {
                    "id": "noncanonical-local-id",
                    "title": "Noncanonical local character",
                    "runtime_backend": "local",
                    "assistant_kind": "character",
                    "assistant_id": "007",
                    "assistant_authority_id": "local-authority",
                    "character_id": 7,
                },
                "root_threads": [],
            },
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        active = store.switch_session(store.active_session_id)
        active.runtime_backend = "server"
        active.assistant_kind = "character"
        active.assistant_id = "active-server-character"
        active.assistant_authority_id = "server-user-v1:" + ("d" * 64)
        active.settings = ConsoleSessionSettings(
            provider="llama_cpp",
            character_label="Wrong active character",
        )
        local_lookup = AsyncMock(
            side_effect=AssertionError("server identity must not use local cards")
        )
        console._resolve_resumed_character_name = local_lookup

        assert (
            await console._workspace._resume_console_workspace_conversation(
                "server-scoped"
            )
            is True
        )
        scoped = store.switch_session(store.active_session_id)
        assert scoped.runtime_backend == "server"
        assert scoped.assistant_kind == "character"
        assert scoped.assistant_id == "opaque-character"
        assert scoped.assistant_authority_id == scoped_authority
        assert scoped.character_id is None
        assert scoped.character_ref() is not None
        assert scoped.settings is not None
        assert scoped.settings.character_label == ""

        assert (
            await console._workspace._resume_console_workspace_conversation(
                "server-unscoped"
            )
            is True
        )
        unscoped = store.switch_session(store.active_session_id)
        assert unscoped.runtime_backend == "server"
        assert unscoped.assistant_kind == "character"
        assert unscoped.assistant_id == "other-opaque-character"
        assert unscoped.assistant_authority_id is None
        assert unscoped.character_id is None
        assert unscoped.character_ref() is None
        assert unscoped.settings is not None
        assert unscoped.settings.character_label == ""

        assert (
            await console._workspace._resume_console_workspace_conversation(
                "malformed-local-scalars"
            )
            is True
        )
        malformed = store.switch_session(store.active_session_id)
        assert malformed.runtime_backend == "local"
        assert malformed.assistant_kind == "character"
        assert malformed.assistant_id is None
        assert malformed.assistant_authority_id is None
        assert malformed.character_id is None
        assert malformed.character_ref() is None

        assert (
            await console._workspace._resume_console_workspace_conversation(
                "noncanonical-local-id"
            )
            is True
        )
        noncanonical = store.switch_session(store.active_session_id)
        assert noncanonical.runtime_backend == "local"
        assert noncanonical.assistant_kind == "character"
        assert noncanonical.assistant_id == "007"
        assert noncanonical.assistant_authority_id == "local-authority"
        assert noncanonical.character_id is None
        assert noncanonical.character_ref() is None
        local_lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("include_runtime_backend", "runtime_backend"),
    (
        pytest.param(False, None, id="missing"),
        pytest.param(True, None, id="none"),
        pytest.param(True, 123, id="non-string"),
    ),
)
async def test_console_resume_rejects_character_identity_without_valid_source(
    include_runtime_backend,
    runtime_backend,
):
    """A persisted row cannot infer local provenance from its other fields."""
    conversation = {
        "id": "invalid-source",
        "title": "Invalid source",
        "assistant_kind": "character",
        "assistant_id": "7",
        "assistant_authority_id": "local-authority",
        "character_id": 7,
    }
    if include_runtime_backend:
        conversation["runtime_backend"] = runtime_backend
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "invalid-source": {
                "conversation": conversation,
                "root_threads": [],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        local_lookup = AsyncMock(return_value="Must not resolve")
        console._resolve_resumed_character_name = local_lookup

        assert (
            await console._workspace._resume_console_workspace_conversation(
                "invalid-source"
            )
            is True
        )

        store = console._ensure_console_chat_store()
        session = store.switch_session(store.active_session_id)
        assert session.runtime_backend == ""
        assert session.assistant_kind == "character"
        assert session.assistant_id == "7"
        assert session.assistant_authority_id == "local-authority"
        assert session.character_id is None
        assert session.character_ref() is None
        local_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_console_resume_rehydrates_local_character_name_from_local_projection():
    """Only a local character row drives the local card/name lookup."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "local-character": {
                "conversation": {
                    "id": "local-character",
                    "title": "Local character",
                    "runtime_backend": "local",
                    "assistant_kind": "character",
                    "assistant_id": "7",
                    "assistant_authority_id": "local-authority",
                    "character_id": 7,
                },
                "root_threads": [],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        name_lookup = AsyncMock(return_value="Elara")
        console._resolve_resumed_character_name = name_lookup

        assert (
            await console._workspace._resume_console_workspace_conversation(
                "local-character"
            )
            is True
        )

        store = console._ensure_console_chat_store()
        session = store.switch_session(store.active_session_id)
        assert session.runtime_backend == "local"
        assert session.assistant_kind == "character"
        assert session.assistant_id == "7"
        assert session.assistant_authority_id == "local-authority"
        assert session.character_id == 7
        assert session.character_name == "Elara"
        name_lookup.assert_awaited_once_with(7)


@pytest.mark.asyncio
async def test_console_workspace_conversation_resume_restores_system_prompt():
    """Resuming a saved conversation restores its persisted system prompt.

    Task 0 persistence seam: the resumed session's settings must carry the
    ``system_prompt`` column from the persisted conversation row (read via
    ``get_conversation_by_id``/``get_conversation_tree``), not whatever
    system prompt (if any) the previously active session had.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-chat-system-prompt",
        role="workspace-thread",
        title="System prompt chat",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "persisted-chat-system-prompt": {
                "conversation": {
                    "id": "persisted-chat-system-prompt",
                    "title": "System prompt chat",
                    "workspace_id": active_workspace.workspace_id,
                    "system_prompt": "Answer only in French.",
                },
                "root_threads": [],
                "pagination": {"total_root_threads": 0},
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "System prompt chat",
            selected=False,
        )

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "persisted-chat-system-prompt",
        )

        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "System prompt chat",
            selected=True,
        )
        store = console._ensure_console_chat_store()
        active_session = store.switch_session(store.active_session_id)
        assert (
            active_session.persisted_conversation_id == "persisted-chat-system-prompt"
        )
        assert active_session.settings is not None
        assert active_session.settings.system_prompt == "Answer only in French."


@pytest.mark.asyncio
async def test_console_workspace_conversation_resume_uses_persisted_workspace():
    """Resume into the persisted conversation workspace when it differs from active."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    target_workspace = service.create_workspace(
        workspace_id="ws-resume-target",
        name="Resume Target",
    )
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-cross-workspace-chat",
        role="workspace-thread",
        title="Saved cross workspace",
    )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "persisted-cross-workspace-chat": {
                "conversation": {
                    "id": "persisted-cross-workspace-chat",
                    "title": "Saved cross workspace",
                    "workspace_id": target_workspace.workspace_id,
                },
                "root_threads": [
                    {
                        "id": "persisted-cross-message-1",
                        "conversation_id": "persisted-cross-workspace-chat",
                        "role": "user",
                        "sender": "user",
                        "content": "cross workspace prompt",
                        "children": [],
                    }
                ],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Saved cross works",
            selected=False,
        )

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            "persisted-cross-workspace-chat",
        )

        await _wait_for_text(console, pilot, "cross workspace prompt")
        store = console._ensure_console_chat_store()
        active_session = store.switch_session(store.active_session_id)
        assert active_session.workspace_id == target_workspace.workspace_id
        assert (
            store.workspace_context.active_workspace_id == target_workspace.workspace_id
        )
        assert (
            service.get_active_workspace().workspace_id == target_workspace.workspace_id
        )
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Saved cross works",
            selected=True,
        )
        assert _selected_workspace_conversation_texts(console)


@pytest.mark.asyncio
async def test_console_workspace_conversation_resume_uses_real_local_services(tmp_path):
    """Resume a workspace conversation through real local DB-backed services."""
    workspace_db = WorkspaceDB(tmp_path / "workspaces.db", client_id="test-client")
    chat_db = CharactersRAGDB(tmp_path / "chacha.db", client_id="test-client")
    workspace_service = LocalWorkspaceRegistryService(workspace_db)
    workspace = workspace_service.create_workspace(
        workspace_id="ws-real",
        name="Real Workspace",
    )
    workspace_service.set_active_workspace(workspace.workspace_id)

    chat_service = ChatConversationService(chat_db)
    conversation_id = chat_service.create_conversation(
        id="real-saved-chat-1",
        title="Real saved chat",
        scope_type="workspace",
        workspace_id=workspace.workspace_id,
        state="in-progress",
    )
    user_message_id = chat_db.add_message(
        {
            "id": "real-message-user-1",
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": "real service user prompt",
        }
    )
    chat_db.add_message(
        {
            "id": "real-message-assistant-1",
            "conversation_id": conversation_id,
            "parent_message_id": user_message_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "real service assistant reply",
        }
    )
    workspace_service.link_membership(
        workspace.workspace_id,
        item_type="conversation",
        item_id=conversation_id,
        role="workspace-thread",
        title="Real saved chat",
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    app.workspace_registry_service = workspace_service
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=chat_service,
        server_service=None,
    )
    host = ConsoleHarness(app)
    saved_state = None

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Real saved chat",
            selected=False,
        )

        await _click_console_workspace_conversation_for_id(
            console,
            pilot,
            conversation_id,
        )

        await _wait_for_text(console, pilot, "real service user prompt")
        await _wait_for_text(console, pilot, "real service assistant reply")
        store = console._ensure_console_chat_store()
        active_session = store.switch_session(store.active_session_id)
        assert active_session.persisted_conversation_id == conversation_id
        assert active_session.title == "Real saved chat"
        assert active_session.workspace_id == workspace.workspace_id
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Real saved chat",
            selected=True,
        )
        assert any(
            "Real saved chat" in text
            for text in _selected_workspace_conversation_texts(console)
        )
        left_rail_text = _visible_text(console.query_one("#console-left-rail"))
        console._set_console_rail_preference(right_open=True, notify_on_failure=False)
        await pilot.pause(0.1)
        inspector_text = _visible_text(console.query_one("#console-right-rail"))
        assert "Provider:" not in left_rail_text
        assert "Model:" not in left_rail_text
        assert "Session Settings" in inspector_text
        assert "Provider:" in inspector_text
        assert "Selected conversation: Real saved chat" in inspector_text
        saved_state = console.save_state()

    restored_host = RestoredConsoleHarness(app, saved_state)
    async with restored_host.run_test(size=(160, 48)) as pilot:
        console = restored_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_text(console, pilot, "real service user prompt")
        await _wait_for_text(console, pilot, "real service assistant reply")
        store = console._ensure_console_chat_store()
        restored_session = store.switch_session(store.active_session_id)
        assert restored_session.persisted_conversation_id == conversation_id
        assert restored_session.workspace_id == workspace.workspace_id


def _resume_generation_variant_meta(seed: int) -> GenerationVariantMeta:
    return GenerationVariantMeta(
        prompt="a red dragon",
        negative_prompt="",
        backend="swarmui",
        model=None,
        seed=seed,
        style=None,
        params={},
    )


@pytest.mark.asyncio
async def test_console_workspace_conversation_resume_hydrates_generation_metadata(
    tmp_path,
):
    """Resuming a saved conversation rehydrates a generation card (P2a §2).

    Task 9's report flagged a real gap: ``ChatScreen
    ._resume_console_workspace_conversation`` -- the production path a
    saved-conversation click (and, after an app restart, the only way back
    into a persisted conversation) drives -- restores a generation
    message's tree node and its kept-variant bytes, but never calls the
    store's documented hydration seam (``get_generation_metadata_for_messages``
    + ``hydrate_generation_metadata``), so the resumed message's
    ``generation_metadata`` comes back empty and the transcript renders a
    plain image instead of a generation card (losing the ``< >``/keep/
    regenerate actions and the other variant).

    Persists a 2-variant generation message via a real ``ConsoleChatStore``
    bound to a real ``ChatPersistenceService``, keeps variant position 1
    (promoting it to canonical), then resumes the SAME conversation by
    driving the real production coroutine directly against a FRESH app/
    screen/store -- the same "real production coroutine, not a rebuilt
    double" pattern ``test_console_scope_row.py`` uses for this exact
    method. Asserts the resumed message is card-eligible: non-empty
    ``generation_metadata`` in kept-first order, matching the DB, with no
    manual hydrate call in this test.
    """
    db = CharactersRAGDB(tmp_path / "resume_hydrate.sqlite", "test-client")
    try:
        chat_service = ChatConversationService(db)

        setup_store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        setup_session = setup_store.create_session(title="Generation resume")
        setup_store.active_session_id = setup_session.id
        seed_message = setup_store.append_generation_message(
            setup_session.id,
            content="[image] a red dragon",
            variants=[
                (b"variant-a-bytes", "image/png", _resume_generation_variant_meta(1)),
                (b"variant-b-bytes", "image/png", _resume_generation_variant_meta(2)),
            ],
            persist=True,
        )
        setup_store.keep_generation_variant(
            setup_session.id, seed_message.id, position=1
        )
        conversation_id = setup_session.persisted_conversation_id
        assert conversation_id is not None
        persisted_message_id = seed_message.persisted_message_id
        assert persisted_message_id is not None

        app = _build_test_app()
        _configure_native_ready_console(app)
        app.chachanotes_db = db
        app.chat_conversation_scope_service = ChatConversationScopeService(
            local_service=chat_service,
            server_service=None,
        )
        host = ConsoleHarness(app)
        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")

            resumed = await console._workspace._resume_console_workspace_conversation(
                conversation_id
            )
            await pilot.pause()
            assert resumed is True

            store = console._ensure_console_chat_store()
            active_session = store.switch_session(store.active_session_id)
            assert active_session.persisted_conversation_id == conversation_id

            reloaded = store.messages_for_session(active_session.id)
            reloaded_generation_msg = next(
                m for m in reloaded if m.persisted_message_id == persisted_message_id
            )

            # Card-eligible: non-empty generation_metadata survived resume,
            # kept variant (seed 2) first, matching the DB's post-keep order.
            assert len(reloaded_generation_msg.generation_metadata) == 2
            assert [
                variant.seed for variant in reloaded_generation_msg.generation_metadata
            ] == [2, 1]
            # Position-0/canonical bytes are still the kept variant's.
            assert reloaded_generation_msg.image_data == b"variant-b-bytes"
            assert reloaded_generation_msg.attachments[0].data == b"variant-b-bytes"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_console_workspace_rail_keeps_active_native_session_visible_when_scope_is_global():
    app = _build_test_app()
    service = app.workspace_registry_service
    active_workspace = service.get_active_workspace()
    service.link_membership(
        active_workspace.workspace_id,
        item_type="conversation",
        item_id="persisted-chat-1",
        role="workspace-thread",
        title="Chat 1",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1", workspace_id="global")
        first.persisted_conversation_id = "persisted-chat-1"
        second = store.create_session(title="Chat 2", workspace_id="global")
        await console._sync_native_console_chat_ui()

        assert store.active_session_id == second.id
        row_texts = await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Chat 2",
            selected=True,
        )
        assert any("Chat 1" in text for text in row_texts), row_texts


@pytest.mark.asyncio
async def test_console_workspace_conversation_search_keeps_selected_global_native_session():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(
            title="Global Search Chat",
            workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
        )
        session.title = "Global Search Chat"
        session.workspace_id = CONSOLE_GLOBAL_WORKSPACE_ID
        await console._sync_native_console_chat_ui()

        await _wait_for_selector(
            console, pilot, "#console-workspace-conversation-search"
        )
        console.query_one("#console-workspace-conversation-search", Input).focus()
        await _set_console_conversation_browser_search(console, pilot, "global")
        await _wait_for_text(console, pilot, "1 match")
        await _wait_for_workspace_conversation_text(
            console,
            pilot,
            "Global Search Chat",
            selected=True,
        )

        assert any(
            "Global Search Chat" in text
            for text in _selected_workspace_conversation_texts(console)
        )


@pytest.mark.asyncio
async def test_console_new_chat_focuses_composer_for_immediate_typing():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.press("n")
        await pilot.pause(0.1)

        assert console.app.focused is composer
        assert composer.draft_text() == "n"


@pytest.mark.asyncio
async def test_console_tab_switch_focuses_composer_for_immediate_typing():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()
        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        await pilot.click(f"#console-session-tab-{first.id}")
        assert store.active_session_id == first.id

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.press("s")
        await pilot.pause(0.1)

        assert console.app.focused is composer
        assert composer.draft_text() == "s"


@pytest.mark.asyncio
async def test_console_native_tab_strip_isolates_composer_drafts():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("first tab draft")

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        assert second != first.id
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second}")

        assert composer.draft_text() == ""

        composer.load_draft("second tab draft")
        await pilot.click(f"#console-session-tab-{first.id}")
        assert store.active_session_id == first.id
        assert composer.draft_text() == "first tab draft"

        await pilot.click(f"#console-session-tab-{second}")
        assert store.active_session_id == second
        assert composer.draft_text() == "second tab draft"


@pytest.mark.asyncio
async def test_console_collapsed_layout_follows_cross_workspace_tab_state():
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = app.workspace_registry_service
    service.create_workspace(workspace_id="ws-a", name="Workspace A")
    service.create_workspace(workspace_id="ws-b", name="Workspace B")
    service.set_active_workspace("ws-a")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Workspace A Chat", workspace_id="ws-a")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("workspace A draft")

        second = store.create_session(
            title="Workspace B Chat",
            workspace_id="ws-b",
        )
        store.set_session_draft(second.id, "workspace B draft")
        store.set_pending_attachment(second.id, _staged_image_attachment())
        store.switch_session(first.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-session-tab-{second.id}")

        console._set_console_composer_collapsed(True)
        await pilot.pause()
        await pilot.click(f"#console-session-tab-{second.id}")
        await _wait_for_active_session(store, pilot, second.id)
        expand = composer.query_one("#console-composer-expand", Button)
        await _wait_for_focus(host, pilot, expand)

        status = composer.query_one("#console-composer-collapsed-status", Static)
        assert composer.collapsed is True
        assert composer.draft_text() == "workspace B draft"
        assert "Draft retained" in str(status.renderable)
        assert "Attachment retained" in str(status.renderable)
        assert service.get_active_workspace().workspace_id == "ws-b"

        await pilot.click(f"#console-session-tab-{first.id}")
        await _wait_for_active_session(store, pilot, first.id)
        await _wait_for_focus(host, pilot, expand)

        assert composer.collapsed is True
        assert composer.draft_text() == "workspace A draft"
        assert "Attachment retained" not in str(status.renderable)
        assert service.get_active_workspace().workspace_id == "ws-a"


@pytest.mark.asyncio
async def test_console_native_tab_strip_keeps_compact_close_x():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        close_selector = f"#console-close-session-tab-{first.id}"
        await _wait_for_selector(console, pilot, close_selector)
        close_button = console.query_one(close_selector, Button)

        assert close_button.label.plain == "✕"
        assert 2 <= close_button.region.width <= 4

        await pilot.click("#console-new-chat-tab")
        second = store.active_session_id
        await _wait_for_selector(console, pilot, f"#console-close-session-tab-{second}")
        await pilot.click(f"#console-close-session-tab-{second}")

        assert store.active_session_id == first.id
        assert second not in {session.id for session in store.sessions()}


@pytest.mark.asyncio
async def test_console_native_tab_title_has_stable_visible_label_region():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.rename_session(
            session.id, "Planning session with a long descriptive name"
        )
        await console._sync_native_console_chat_ui()

        tab_selector = f"#console-session-tab-{session.id}"
        await _wait_for_selector(console, pilot, tab_selector)
        tab = console.query_one(tab_selector, Button)

        assert tab.tooltip == (
            "Active Console tab: Planning session with a long descriptive name. "
            "Click again to rename. Middle-click closes the tab."
        )
        # Fleet-UX expert review F7 (task-1234): END-truncated with a
        # single-cell ellipsis, replacing TASK-375's middle-truncation
        # (live UAT: the middle mark landed mid-word and read as garbled).
        assert str(tab.label) == "Planning session w…"
        assert tab.region.width >= 18
        assert "Planning" in _visible_text(console)
        assert "…" in _visible_text(console)


def test_console_tab_label_end_truncates_with_visible_ellipsis():
    """Fleet-UX expert review F7 (task-1234): long tab titles END-truncate
    with a single-cell ellipsis, replacing TASK-375's middle-truncation --
    live UAT found the mark landing mid-word ("What is t…ate an."), judged
    a worse defect than losing TASK-375 AC#2's shared-prefix disambiguation.
    That trade-off is asserted explicitly below (not silently dropped): two
    titles sharing a long common PREFIX can render an identical tab label
    again; the full title is always one hover away in the tab tooltip."""
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        CONSOLE_SESSION_TAB_DISPLAY_CHARS,
        ConsoleSessionSurface,
    )

    display = ConsoleSessionSurface._display_title

    short = display("Chat 1")
    assert short == "Chat 1"  # short titles are untouched

    a = display("Long conversation about embeddings and vector stores in local RAG")
    b = display("Terraform state migration help across every remote backend")
    for label in (a, b):
        assert "…" in label
        assert "..." not in label
        assert len(label) <= CONSOLE_SESSION_TAB_DISPLAY_CHARS
    assert a.startswith("Long conversation")
    assert b.startswith("Terraform state")
    # Titles that diverge early enough stay distinguishable.
    assert a != b

    # Documented trade-off: a long SHARED prefix now collides (TASK-375's
    # AC#2 disambiguation is no longer guaranteed for this case).
    collides_with_a = display(
        "Long conversation about Terraform state migration and remote backends"
    )
    assert collides_with_a == a == "Long conversation…"


@pytest.mark.asyncio
async def test_console_native_active_tab_title_opens_rename_modal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        assert not list(console.query(f"#console-rename-session-tab-{session.id}"))

        await pilot.click(f"#console-session-tab-{session.id}")
        modal_screen = await _wait_for_console_rename_modal(host, pilot)

        rename_input = modal_screen.query_one("#console-rename-session-title", Input)
        assert rename_input.value == "Chat 1"
        assert getattr(console.app.focused, "id", None) == rename_input.id

        await pilot.press(*"Planning")
        modal_screen.query_one("#console-rename-session-save", Button).press()
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{session.id}")

        assert store.sessions()[0].title == "Planning"
        assert "Planning" in _visible_text(console)
        assert not list(console.query(f"#console-session-rename-input-{session.id}"))


@pytest.mark.asyncio
async def test_console_native_rename_modal_buttons_are_not_clipped():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click(f"#console-session-tab-{session.id}")
        modal_screen = await _wait_for_console_rename_modal(host, pilot)

        action_row = modal_screen.query_one("#console-rename-session-actions")
        cancel_button = modal_screen.query_one("#console-rename-session-cancel", Button)
        save_button = modal_screen.query_one("#console-rename-session-save", Button)

        assert action_row.region.height >= 3
        assert cancel_button.region.height >= 3
        assert save_button.region.height >= 3
        assert str(cancel_button.label) == "Cancel"
        assert str(save_button.label) == "Save"


@pytest.mark.asyncio
async def test_console_native_tab_rename_escape_restores_existing_title():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        await console._sync_native_console_chat_ui()

        await pilot.click(f"#console-session-tab-{session.id}")

        modal_screen = await _wait_for_console_rename_modal(host, pilot)
        rename_input = modal_screen.query_one("#console-rename-session-title", Input)
        assert rename_input.value == "Chat 1"
        await pilot.press(*"Discarded")
        await pilot.press("escape")
        await _wait_for_console_screen(host, console, pilot)
        await _wait_for_selector(console, pilot, f"#console-session-tab-{session.id}")

        assert store.sessions()[0].title == "Chat 1"
        assert "Chat 1" in _visible_text(console)
        assert not list(console.query(f"#console-session-rename-input-{session.id}"))


@pytest.mark.asyncio
async def test_console_close_tab_with_messages_shows_confirmation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
        store.create_session(title="Chat 2")
        await console._sync_native_console_chat_ui()

        close_selector = f"#console-close-session-tab-{session.id}"
        await _wait_for_selector(console, pilot, close_selector)
        await pilot.click(close_selector)

        from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

        for _ in range(20):
            await pilot.pause()
            if any(isinstance(s, ConfirmationDialog) for s in host.screen_stack):
                break

        dialog_screens = [
            s for s in host.screen_stack if isinstance(s, ConfirmationDialog)
        ]
        assert len(dialog_screens) == 1, (
            "confirmation dialog should appear for tab with messages"
        )
        assert session.id in {s.id for s in store.sessions()}, "session not closed yet"

        await pilot.click("#confirm-button")
        for _ in range(10):
            await pilot.pause()

        assert session.id not in {s.id for s in store.sessions()}, (
            "session closed after confirm"
        )


def test_native_console_state_serializes_plain_string_message_role():
    """Verify saved Console messages tolerate legacy/plain-string roles."""
    message = SimpleNamespace(
        id="message-a",
        role="assistant",
        content="answer",
        turn_id=None,
        status="complete",
        persisted_message_id=None,
        feedback=None,
        variants=None,
    )

    serialized = ChatScreen._serialize_console_message(message)

    assert serialized["role"] == "assistant"


def _console_snapshot_with_sessions(screen: ChatScreen) -> dict | None:
    """The screen-state payload as it looked BEFORE task-15860 Task 3.

    Console message state stopped travelling through `ScreenStateStore`
    when the app-owned `ConsoleRuntime`'s store became the single source of
    truth for history: `_serialize_native_console_state` no longer emits
    `sessions`/`messages_by_session`/`active_session_id`, and
    `_restore_native_console_state` no longer calls
    `ConsoleChatStore.restore_state`.

    The per-session and per-message (de)serializers themselves are still
    live code with no production caller (retirement tracked as task-16520),
    and the tests below are about THEIR shape -- legacy-payload tolerance,
    character-provenance narrowing, byte-free message projection. So the
    round trip they need is spelled out here, next to them, instead of
    borrowed from a production method that no longer performs it.

    Returns:
        The live view-state payload with the retired session/message keys
        added back, or ``None`` when the screen has no sessions (the same
        condition the production serializer returns ``None`` for).
    """
    payload = screen._serialize_native_console_state()
    if payload is None:
        return None
    store = screen._console_chat_store
    payload["active_session_id"] = store.active_session_id
    payload["sessions"] = [
        screen._session._console_session_to_state(session)
        for session in store.sessions()
    ]
    payload["messages_by_session"] = {
        session.id: [
            ChatScreen._serialize_console_message(message)
            for message in store.messages_for_session(session.id)
        ]
        for session in store.sessions()
    }
    return payload


def _restore_console_snapshot_with_sessions(screen: ChatScreen, payload) -> None:
    """The mirror of `_console_snapshot_with_sessions` -- see its docstring.

    Reproduces the retired half of `_restore_native_console_state` in the
    same order it ran (rehydrate, `store.restore_state`, hydrate generation
    metadata) and then hands the payload to the LIVE method for the view
    state it still owns.
    """
    if not isinstance(payload, dict):
        screen._restore_native_console_state(payload)
        return
    raw_sessions = payload.get("sessions")
    if isinstance(raw_sessions, list) and raw_sessions:
        store = screen._ensure_console_chat_store()
        raw_messages = payload.get("messages_by_session")
        messages_by_session = raw_messages if isinstance(raw_messages, dict) else {}
        restored_sessions = []
        restored_messages: dict[str, list] = {}
        for raw_session in raw_sessions:
            if not isinstance(raw_session, dict):
                continue
            session = screen._session._console_session_from_state(raw_session)
            restored_sessions.append(session)
            restored_messages[session.id] = []
            raw_list = messages_by_session.get(session.id, [])
            if not isinstance(raw_list, list):
                continue
            for raw_message in raw_list:
                message = ChatScreen._restore_console_message(raw_message)
                if message is None:
                    continue
                screen._rehydrate_console_message_image(message)
                restored_messages[session.id].append(message)
        screen._rehydrate_console_message_attachments(
            [message for messages in restored_messages.values() for message in messages]
        )
        active_session_id = payload.get("active_session_id")
        store.restore_state(
            sessions=restored_sessions,
            messages_by_session=restored_messages,
            active_session_id=(
                str(active_session_id) if active_session_id is not None else ""
            ),
        )
        screen._rehydrate_console_message_generation_metadata(store, restored_messages)
    screen._restore_native_console_state(payload)


def _bare_console_screen(store: ConsoleChatStore) -> ChatScreen:
    """Build a native-console screen shell for direct serialize/restore calls.

    Bypasses ``ChatScreen.__init__`` (which requires a mounted Textual app)
    while still resolving the class's inherited serialize/restore helpers
    normally, so ``_serialize_native_console_state`` /
    ``_restore_native_console_state`` can be exercised as plain, fast
    unit-level round trips instead of a full pilot-driven screen.

    Args:
        store: The ConsoleChatStore instance to attach to the screen for
            state serialization and restoration testing.

    Returns:
        ChatScreen: A bare ChatScreen instance with minimal initialization,
            suitable for unit-level serialize/restore round-trip testing.
    """
    screen = ChatScreen.__new__(ChatScreen)
    screen._console_runtime_ref = ConsoleRuntime(None)
    screen._console_chat_store = store
    # A bare, uninitialized `ConsoleSessionController` -- `__init__` was
    # never run, so every OTHER dependency is unset by default. Only the
    # two chat-store accessors are wired (reading back `screen._console_
    # chat_store`, which several tests using this helper reassign later):
    # `_console_session_to_state`/`_console_session_from_state` (this
    # helper's original whole point) tolerate a missing accessor via their
    # own `getattr`/staticmethod shape, but sibling staying methods this
    # file also drives through a bare screen (e.g. `_current_console_rail_
    # character_id` -> `_active_native_console_session`) read `self.
    # _console_chat_store` directly, with no such tolerance.
    screen._session = ConsoleSessionController.__new__(ConsoleSessionController)
    screen._session._chat_store_accessor = lambda: screen._console_chat_store
    screen._session._current_chat_store_accessor = lambda: screen._console_chat_store
    screen._character = ConsoleCharacterController.__new__(ConsoleCharacterController)
    screen._character._active_native_session_accessor = lambda: (
        screen._session._active_native_console_session()
    )
    screen._console_visible_draft_session_id = None
    screen._console_composer_or_none = lambda: None
    screen._task_resume_state = TaskResumeState()
    # `_rehydrate_console_message_image`/`_attachments` read `self.
    # app_instance.chachanotes_db` (best-effort; both tolerate `None`).
    screen.app_instance = SimpleNamespace(
        notify=lambda *a, **k: None, chachanotes_db=None
    )
    stub_image_controller(
        screen,
        context="test_console_native_chat_flow._bare_console_screen",
        ensure_console_image_view=lambda: screen._ensure_console_image_view(),
        recent_console_image_messages=(
            lambda messages: screen._recent_console_image_messages(messages)
        ),
        console_image_default_mode=lambda: screen._console_image_default_mode,
        console_generation_browse=lambda: screen._console_generation_browse(),
        sync_native_console_chat_ui=lambda: screen._sync_native_console_chat_ui(),
        ensure_console_chat_store=lambda: screen._ensure_console_chat_store(),
        build_console_provider_selection=(
            lambda: screen._build_console_provider_selection()
        ),
        ensure_console_provider_gateway=(
            lambda: screen._ensure_console_provider_gateway()
        ),
        console_image_preparing=(
            lambda: getattr(screen, "_console_image_preparing", None)
        ),
        current_console_chat_store=lambda: screen._console_chat_store,
        console_composer_or_none=lambda: screen._console_composer_or_none(),
        console_visible_draft_session_id=(
            lambda: screen._console_visible_draft_session_id
        ),
        append_native_console_system_message=(
            lambda *args, **kwargs: screen._append_native_console_system_message(
                *args, **kwargs
            )
        ),
        request_console_control_bar_sync=(
            lambda: screen._request_console_control_bar_sync()
        ),
        default_console_session_settings=(
            lambda: screen._session._default_console_session_settings()
        ),
        clear_console_composer_draft=(lambda: screen._clear_console_composer_draft()),
    )

    # `_restore_native_console_state`'s message-rehydration calls
    # (`_rehydrate_console_message_image`/`_attachments`/`_generation_
    # metadata`) and several tests using this helper directly
    # (`_save_console_message_image`, `_console_save_as_destinations`,
    # `_save_console_message_as_chatbook`) moved to `ConsoleMessageController`
    # (wave-3 console decomposition, task 1). `screen._message` is built
    # the same bare-`__new__` way `screen._session` already is above, with
    # every constructor callable wired for real -- unlike `_bare_generation_
    # screen` in `Tests/Chat/test_console_generation_actions.py`, this
    # helper's callers span enough of the cluster (serialize/restore,
    # save-as, save-image) that stubbing the "unreached" ones would just
    # move the maintenance burden, not reduce it.
    screen._message = ConsoleMessageController(
        screen,
        app_instance=screen.app_instance,
        chat_store_accessor=lambda: screen._console_chat_store,
        current_chat_store_accessor=lambda: screen._console_chat_store,
        ensure_console_chat_controller=lambda: screen._ensure_console_chat_controller(),
        current_chat_controller_accessor=lambda: getattr(
            screen, "_console_chat_controller", None
        ),
        sync_native_console_chat_ui=lambda: screen._sync_native_console_chat_ui(),
        active_session_is_ephemeral=(
            lambda: screen._session._console_active_session_is_ephemeral()
        ),
        active_native_console_session=(
            lambda: screen._session._active_native_console_session()
        ),
        current_console_conversation_id=(
            lambda: screen._session._current_console_conversation_id()
        ),
        active_console_provider_model_display=(
            lambda: screen._active_console_provider_model_display()
        ),
        console_initial_session_title_for_workspace=lambda workspace_id: "",
        console_change_review_run_id=lambda store, message_id: None,
        open_change_review=lambda run_id: None,
        start_console_transcript_sync_timer=lambda: None,
        clear_native_console_message_selection=lambda: None,
        regenerate_console_generation_variant=(
            lambda message_id: screen._image._regenerate_console_generation_variant(
                message_id
            )
        ),
        select_console_generation_variant=(
            lambda message, direction: screen._image._select_console_generation_variant(
                message, direction=direction
            )
        ),
        keep_console_generation_variant=(
            lambda message: screen._image._keep_console_generation_variant(message)
        ),
        handle_console_toggle_image_view=(
            lambda message_id: screen._image._handle_console_toggle_image_view(
                message_id
            )
        ),
        invalidate_console_persisted_rows_cache=lambda: None,
    )
    return screen


def test_native_console_state_round_trip_preserves_session_updated_at():
    """Verify a restored session keeps its original ``updated_at`` timestamp.

    Without this, every restored session's ``updated_at`` resets to "now" on
    screen recreation, so restored sessions all show age "now" and recent-
    first ordering across restored sessions breaks.
    """
    store = ConsoleChatStore()
    session = ConsoleChatSession(
        id="session-a",
        title="Chat 1",
        updated_at="2020-01-01T00:00:00+00:00",
    )
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    screen = _bare_console_screen(store)

    payload = _console_snapshot_with_sessions(screen)
    assert payload is not None
    assert payload["sessions"][0]["updated_at"] == "2020-01-01T00:00:00+00:00"

    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)
    _restore_console_snapshot_with_sessions(restored_screen, payload)

    restored_session = restored_store.sessions()[0]
    assert restored_session.updated_at == "2020-01-01T00:00:00+00:00"


def test_native_console_state_round_trip_preserves_session_system_prompt():
    """Verify a restored session keeps its per-session system prompt.

    ``ConsoleSessionSettings.system_prompt`` must flow through the generic
    ``__dataclass_fields__``-based whitelist in ``_restore_console_settings``
    the same way ``source`` and every other settings field does, with no
    per-field allowlist entry needed.
    """
    store = ConsoleChatStore()
    session = ConsoleChatSession(
        id="session-a",
        title="Chat 1",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="gpt-4o",
            system_prompt="Be terse and cite sources.",
        ),
    )
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    screen = _bare_console_screen(store)

    payload = _console_snapshot_with_sessions(screen)
    assert payload is not None
    assert (
        payload["sessions"][0]["settings"]["system_prompt"]
        == "Be terse and cite sources."
    )

    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)
    _restore_console_snapshot_with_sessions(restored_screen, payload)

    restored_session = restored_store.sessions()[0]
    assert restored_session.settings is not None
    assert restored_session.settings.system_prompt == "Be terse and cite sources."


def test_native_console_restore_ignores_legacy_identity_without_mutation_or_config_io(
    monkeypatch,
) -> None:
    """Exercise legacy identity filtering through the native state owner."""
    store = ConsoleChatStore()
    session = ConsoleChatSession(
        id="session-a",
        title="Legacy identity",
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
    )
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    payload = _console_snapshot_with_sessions(_bare_console_screen(store))
    assert payload is not None
    settings_payload = payload["sessions"][0]["settings"]
    assert settings_payload is not None
    settings_payload["persona_label"] = "Legacy A"
    settings_payload["user_profile_label"] = "Legacy B"
    payload_before = deepcopy(payload)
    encoded_before = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    config_callbacks = {
        name: Mock(name=name)
        for name in (
            "save_setting_to_cli_config",
            "save_settings_to_cli_config",
            "delete_settings_from_cli_config",
        )
    }
    for name, callback in config_callbacks.items():
        monkeypatch.setattr(chat_screen_module, name, callback)

    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)
    _restore_console_snapshot_with_sessions(restored_screen, payload)
    restored_session = restored_store.sessions()[0]
    serialized = _console_snapshot_with_sessions(restored_screen)

    assert payload == payload_before
    assert (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        == encoded_before
    )
    assert restored_session.settings is not None
    assert not hasattr(restored_session.settings, "user_profile_label")
    assert serialized is not None
    serialized_settings = serialized["sessions"][0]["settings"]
    assert serialized_settings is not None
    assert {
        "persona_label",
        "user_profile_label",
        "assistant_kind",
        "assistant_name",
        "assistant_id",
    }.isdisjoint(serialized_settings)
    for callback in config_callbacks.values():
        callback.assert_not_called()


def test_native_console_state_round_trip_preserves_source_aware_character_identity():
    """Screen state keeps exact assistant provenance and local presentation."""
    store = ConsoleChatStore()
    session = ConsoleChatSession(
        id="session-a",
        title="Chat with Elara",
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
        character_name="Elara",
    )
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    screen = _bare_console_screen(store)

    payload = _console_snapshot_with_sessions(screen)
    assert payload is not None
    assert {
        key: payload["sessions"][0][key]
        for key in (
            "runtime_backend",
            "assistant_kind",
            "assistant_id",
            "assistant_authority_id",
            "character_id",
            "character_name",
        )
    } == {
        "runtime_backend": "local",
        "assistant_kind": "character",
        "assistant_id": "7",
        "assistant_authority_id": "local-authority",
        "character_id": 7,
        "character_name": "Elara",
    }

    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)
    _restore_console_snapshot_with_sessions(restored_screen, payload)

    restored_session = restored_store.sessions()[0]
    assert restored_session.runtime_backend == "local"
    assert restored_session.assistant_kind == "character"
    assert restored_session.assistant_id == "7"
    assert restored_session.assistant_authority_id == "local-authority"
    assert restored_session.character_id == 7
    assert restored_session.character_name == "Elara"
    assert restored_session.character_ref() is not None


def test_live_server_session_never_exposes_local_character_projection():
    """A stray server-side numeric ID cannot drive local rail/card state."""
    session = ConsoleChatSession(
        id="session-a",
        title="Server character",
        runtime_backend="server",
        assistant_kind="character",
        assistant_id="opaque-character",
        assistant_authority_id="server-user-v1:" + ("f" * 64),
        character_id=7,
        character_name="Unrelated local card",
    )
    store = ConsoleChatStore()
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    screen = _bare_console_screen(store)

    assert screen._character._current_console_rail_character_id() is None

    payload = _console_snapshot_with_sessions(screen)
    assert payload is not None
    assert payload["sessions"][0]["character_id"] is None
    assert session.character_id == 7


def test_native_console_state_restore_adapts_legacy_local_character_without_authority():
    """Legacy numeric character state keeps direct-chat kind but stays unproven."""
    payload = {
        "version": "1.0",
        "active_session_id": "session-a",
        "sessions": [
            {
                "id": "session-a",
                "title": "Chat with Elara",
                "workspace_id": CONSOLE_GLOBAL_WORKSPACE_ID,
                "persisted_conversation_id": None,
                "draft": "",
                "settings": None,
                "character_id": 7,
                "character_name": "Elara",
            }
        ],
        "messages_by_session": {"session-a": []},
    }
    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)

    _restore_console_snapshot_with_sessions(restored_screen, payload)

    restored_session = restored_store.sessions()[0]
    assert restored_session.runtime_backend == "local"
    assert restored_session.assistant_kind == "character"
    assert restored_session.assistant_id == "7"
    assert restored_session.assistant_authority_id is None
    assert restored_session.character_id == 7
    assert restored_session.character_ref() is None


@pytest.mark.parametrize(
    ("include_runtime_backend", "runtime_backend"),
    (
        pytest.param(False, None, id="missing"),
        pytest.param(True, None, id="none"),
        pytest.param(True, 123, id="non-string"),
    ),
)
def test_source_aware_native_console_state_rejects_character_without_valid_source(
    include_runtime_backend,
    runtime_backend,
):
    """Partial source-aware state cannot infer local character provenance."""
    raw_session = {
        "id": "session-a",
        "title": "Invalid source",
        "workspace_id": CONSOLE_GLOBAL_WORKSPACE_ID,
        "persisted_conversation_id": None,
        "draft": "",
        "settings": None,
        "assistant_kind": "character",
        "assistant_id": "7",
        "assistant_authority_id": "local-authority",
        "character_id": 7,
    }
    if include_runtime_backend:
        raw_session["runtime_backend"] = runtime_backend
    payload = {
        "version": "1.0",
        "active_session_id": "session-a",
        "sessions": [raw_session],
        "messages_by_session": {"session-a": []},
    }
    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)

    _restore_console_snapshot_with_sessions(restored_screen, payload)

    session = restored_store.sessions()[0]
    assert session.runtime_backend == ""
    assert session.assistant_kind == "character"
    assert session.assistant_id == "7"
    assert session.assistant_authority_id == "local-authority"
    assert session.character_id is None
    assert session.character_ref() is None


def test_native_console_state_restore_does_not_coerce_identity_scalars():
    """Malformed scalar types cannot be promoted into proven provenance."""
    payload = {
        "version": "1.0",
        "active_session_id": "session-a",
        "sessions": [
            {
                "id": "session-a",
                "title": "Malformed identity",
                "workspace_id": CONSOLE_GLOBAL_WORKSPACE_ID,
                "persisted_conversation_id": None,
                "draft": "",
                "settings": None,
                "runtime_backend": "local",
                "assistant_kind": "character",
                "assistant_id": 7,
                "assistant_authority_id": 123,
                "character_id": 7,
            },
            {
                "id": "session-b",
                "title": "Malformed source",
                "workspace_id": CONSOLE_GLOBAL_WORKSPACE_ID,
                "persisted_conversation_id": None,
                "draft": "",
                "settings": None,
                "runtime_backend": 123,
                "assistant_kind": "character",
                "assistant_id": "8",
                "assistant_authority_id": "local-authority",
                "character_id": 8,
            },
        ],
        "messages_by_session": {"session-a": [], "session-b": []},
    }
    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)

    _restore_console_snapshot_with_sessions(restored_screen, payload)

    session = restored_store.sessions()[0]
    assert session.runtime_backend == "local"
    assert session.assistant_kind == "character"
    assert session.assistant_id is None
    assert session.assistant_authority_id is None
    assert session.character_id is None
    assert session.character_ref() is None

    invalid_source = restored_store.sessions()[1]
    assert invalid_source.runtime_backend == ""
    assert invalid_source.assistant_kind == "character"
    assert invalid_source.assistant_id == "8"
    assert invalid_source.assistant_authority_id == "local-authority"
    assert invalid_source.character_id is None
    assert invalid_source.character_ref() is None


def test_native_console_state_restore_drops_server_numeric_local_projection():
    """A server character keeps opaque provenance but no local lookup key."""
    authority_id = "server-user-v1:" + ("e" * 64)
    payload = {
        "version": "1.0",
        "active_session_id": "session-a",
        "sessions": [
            {
                "id": "session-a",
                "title": "Server identity",
                "workspace_id": CONSOLE_GLOBAL_WORKSPACE_ID,
                "persisted_conversation_id": None,
                "draft": "",
                "settings": None,
                "runtime_backend": "server",
                "assistant_kind": "character",
                "assistant_id": "opaque-character",
                "assistant_authority_id": authority_id,
                "character_id": 7,
                "character_name": "Server Card",
            }
        ],
        "messages_by_session": {"session-a": []},
    }
    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)

    _restore_console_snapshot_with_sessions(restored_screen, payload)

    session = restored_store.sessions()[0]
    assert session.runtime_backend == "server"
    assert session.assistant_kind == "character"
    assert session.assistant_id == "opaque-character"
    assert session.assistant_authority_id == authority_id
    assert session.character_id is None
    assert session.character_name == "Server Card"
    assert session.character_ref() is not None


def test_native_console_state_restore_tolerates_legacy_payload_without_updated_at():
    """Verify legacy saved states (no ``updated_at`` key) still restore.

    Older saved screen states were written before ``updated_at`` was
    serialized, so restore must fall back to the ``ConsoleChatSession``
    factory default instead of raising or leaving the field unset.
    """
    payload = {
        "version": "1.0",
        "active_session_id": "session-a",
        "sessions": [
            {
                "id": "session-a",
                "title": "Chat 1",
                "workspace_id": CONSOLE_GLOBAL_WORKSPACE_ID,
                "persisted_conversation_id": None,
                "draft": "",
                "settings": None,
            }
        ],
        "messages_by_session": {"session-a": []},
    }
    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)

    before = datetime.now(timezone.utc)
    _restore_console_snapshot_with_sessions(restored_screen, payload)
    after = datetime.now(timezone.utc)

    restored_session = restored_store.sessions()[0]
    assert restored_session.updated_at
    restored_dt = datetime.fromisoformat(restored_session.updated_at)
    assert before <= restored_dt <= after


@pytest.mark.asyncio
async def test_ctrl_k_opens_session_switcher_and_activates_native_session():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        # Create a second native session so there is something to switch to.
        await pilot.click("#console-new-chat-tab")
        await pilot.pause(0.2)
        store = console._console_chat_store
        first_session = store.sessions()[0]
        assert store.active_session_id != first_session.id

        await pilot.press("ctrl+k")
        await pilot.pause(0.2)
        assert host.screen_stack[-1].__class__.__name__ == "ConsoleSessionSwitcherModal"
        query = host.screen_stack[-1].query_one("#console-switcher-query")
        assert host.focused is query
        # First entry is the ACTIVE session; pick the other one by typing its
        # distinguishing token. Default session titles ("Chat 1", "Chat 2")
        # share their first word, so the trailing number is what disambiguates.
        await pilot.press(*first_session.title.split()[-1].lower())
        await pilot.pause(0.2)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert store.active_session_id == first_session.id


@pytest.mark.asyncio
async def test_ctrl_k_is_inert_while_setup_modal_blocks():
    app = _build_test_app()  # blocked: no provider ready
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        await pilot.press("ctrl+k")
        await pilot.pause(0.2)
        assert host.screen_stack[-1] is console


@pytest.mark.asyncio
async def test_console_setup_modal_blocks_programmatic_composer_toggles():
    app = _build_test_app()
    _configure_openai_missing_api_key(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-setup-modal")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        collapse = composer.query_one("#console-composer-collapse", Button)
        expand = composer.query_one("#console-composer-expand", Button)
        assert console._console_setup_modal_blocking()

        collapse.press()
        expand.press()
        await pilot.pause()

        assert console._console_composer_collapsed is False
        assert composer.collapsed is False
        assert console.check_action("expand_collapsed_console_composer", ()) is False


@pytest.mark.asyncio
async def test_console_setup_modal_retains_collapsed_layout_and_restores_expand_focus():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        console._set_console_composer_collapsed(True)
        await pilot.pause()

        _configure_openai_missing_api_key(app)
        console._session._replace_active_console_session_settings(
            ConsoleSessionSettings(
                provider="openai",
                model="gpt-4o",
                source="user",
            )
        )
        console._sync_console_transcript_guidance()
        await pilot.pause()
        assert console._console_setup_modal_blocking()
        assert console._console_composer_collapsed is True
        assert composer.collapsed is True

        composer.query_one("#console-composer-expand", Button).press()
        await pilot.pause()
        assert composer.collapsed is True
        assert console.check_action("expand_collapsed_console_composer", ()) is False

        _configure_native_ready_console(app)
        console._session._replace_active_console_session_settings(
            ConsoleSessionSettings(
                provider="llama_cpp",
                model="local-model",
                base_url="http://127.0.0.1:9099",
                source="user",
            )
        )
        console._sync_console_transcript_guidance()
        await pilot.pause()
        assert console._console_setup_modal_blocking() is False

        console._restore_console_workbench_focus()
        await pilot.pause()
        expand = composer.query_one("#console-composer-expand", Button)
        assert host.focused is expand
        assert composer.can_focus is False


@pytest.mark.asyncio
async def test_console_navigation_resume_retains_collapsed_composer():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleNavigationHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        console._set_console_composer_collapsed(True)
        await pilot.pause()

        console.post_message(NavigateToScreen("library"))
        await pilot.pause()
        assert [message.screen_name for message in host.navigation_messages] == [
            "library"
        ]

        console.on_screen_resume()
        await pilot.pause()

        assert console._console_composer_collapsed is True
        assert composer.collapsed is True


@pytest.mark.asyncio
async def test_switcher_rename_choice_chains_to_rename_modal():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.press("ctrl+k")
        await pilot.pause(0.2)
        await pilot.press("f2")
        await pilot.pause(0.3)
        assert host.screen_stack[-1].__class__.__name__ == "ConsoleRenameSessionModal"
        await pilot.press("escape")


def test_insert_file_segment_collapses_with_custom_label():
    composer = ConsoleComposerBar()
    composer.insert_file_segment("file body text", "📄 notes.md · 4 KB")

    assert composer.draft_text() == "file body text"
    assert composer._display_draft_text() == "📄 notes.md · 4 KB"


def test_insert_file_segment_appends_after_typed_draft():
    composer = ConsoleComposerBar()
    composer.insert_text("see attached: ")
    composer.insert_file_segment("file body", "📄 a.md · 9 B")

    assert composer.draft_text() == "see attached: file body"
    assert composer._display_draft_text() == "see attached: 📄 a.md · 9 B"


def test_paste_collapse_label_still_defaults_to_character_count():
    composer = ConsoleComposerBar(paste_collapse_threshold=5)
    composer.insert_pasted_text("0123456789")

    assert composer._display_draft_text() == "Pasted text | 10 characters | Expand"


@pytest.mark.asyncio
async def test_attachment_indicator_visibility_follows_label():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        composer.set_pending_attachment_label("photo.png · 240 KB")
        await pilot.pause()
        indicator = console.query_one("#console-attachment-indicator", Static)
        clear_button = console.query_one("#console-clear-attachment", Button)
        assert "photo.png" in str(indicator.renderable)
        assert indicator.styles.display != "none"
        assert clear_button.styles.display != "none"

        composer.set_pending_attachment_label(None)
        await pilot.pause()
        assert indicator.styles.display == "none"
        assert clear_button.styles.display == "none"


@pytest.mark.asyncio
async def test_staged_attachment_count_stays_visible_after_attach_moved_to_the_menu():
    """TASK-380's guarantee, re-homed: the staged count stays legible.

    The original test pinned it on the Attach button's own label/tooltip
    ("Attach +", "2 of 5"). Attach now lives in the composer's ☰ menu, so
    the count reads off the indicator beside the row -- which is where a
    user looks for it anyway -- and off the ✕ control that acts on it.
    TASK-380's actual defect (staging morphing a CONTROL into an "attached
    OK" status glyph) cannot recur for a control that is no longer there."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        assert not console.query("#console-attach-context")
        clear_button = console.query_one("#console-clear-attachment", Button)
        indicator = console.query_one("#console-attachment-indicator", Static)

        composer.set_pending_attachment_label("2 files", count=2, total=5)
        await pilot.pause()
        # The count is on the indicator, and the control that acts on it.
        assert "2 files" in str(indicator.renderable)
        assert clear_button.styles.display == "block"
        assert "2 pending attachments" in str(clear_button.tooltip)

        composer.set_pending_attachment_label("photo.png · 240 B", count=1, total=5)
        await pilot.pause()
        assert "photo.png" in str(indicator.renderable)
        assert str(clear_button.tooltip) == "Remove the pending attachment."


@pytest.mark.asyncio
async def test_console_attachment_worker_stages_image_and_inlines_text(tmp_path):
    from PIL import Image as PILImage

    image_path = tmp_path / "photo.png"
    PILImage.new("RGB", (4, 4), color=(0, 100, 0)).save(image_path, format="PNG")
    text_path = tmp_path / "notes.md"
    text_path.write_text("# heading\nbody")

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        import tldw_chatbook.Chat.attachment_core as attachment_core

        # Test files live in tmp_path, outside $HOME — widen the safety root.
        original = attachment_core.load_processed_file

        async def _rooted(file_path, *, allowed_root=None):
            return await original(file_path, allowed_root=str(tmp_path))

        attachment_core.load_processed_file = _rooted
        try:
            await console._process_console_attachment(str(image_path))
            await pilot.pause()
            store = console._ensure_console_chat_store()
            session_id = store.active_session_id
            pending = store.pending_attachment(session_id)
            assert pending is not None and pending.file_type == "image"
            composer = console.query_one("#console-native-composer", ConsoleComposerBar)
            assert composer._pending_attachment_label is not None

            await console._process_console_attachment(str(text_path))
            await pilot.pause()
            assert "body" in composer.draft_text()
            assert "notes.md" in composer._display_draft_text()
        finally:
            attachment_core.load_processed_file = original


def _staged_image_attachment():
    from tldw_chatbook.Chat.attachment_core import PendingAttachment

    return PendingAttachment(
        file_path="/tmp/photo.png",
        display_name="photo.png",
        file_type="image",
        insert_mode="attachment",
        data=b"\x89PNG-bytes",
        mime_type="image/png",
        original_size=11,
        processed_size=11,
    )


@pytest.mark.asyncio
async def test_pending_image_on_non_vision_model_blocks_send(monkeypatch):
    import tldw_chatbook.Chat.attachment_core as attachment_core

    monkeypatch.setattr(attachment_core, "is_vision_capable", lambda p, m: False)
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.set_pending_attachment(session.id, _staged_image_attachment())

        reason = console._console_send_blocked_reason()
        assert "can't accept images" in reason


@pytest.mark.asyncio
async def test_pending_image_on_vision_model_does_not_block(monkeypatch):
    import tldw_chatbook.Chat.attachment_core as attachment_core

    monkeypatch.setattr(attachment_core, "is_vision_capable", lambda p, m: True)
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.set_pending_attachment(session.id, _staged_image_attachment())

        assert console._console_attachment_blocked_reason() == ""


@pytest.mark.asyncio
async def test_attach_at_cap_blocks_picker_before_selection(monkeypatch):
    """TASK-377: at the attachment cap, pressing Attach must report the limit
    immediately rather than opening the picker and rejecting the pick afterwards.
    """
    from unittest.mock import AsyncMock, MagicMock

    from tldw_chatbook.Chat.console_chat_store import MAX_PENDING_ATTACHMENTS

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        for _ in range(MAX_PENDING_ATTACHMENTS):
            assert store.add_pending_attachment(session.id, _staged_image_attachment())
        assert len(store.pending_attachments(session.id)) == MAX_PENDING_ATTACHMENTS

        pushed: list = []
        monkeypatch.setattr(
            console.app,
            "push_screen",
            AsyncMock(side_effect=lambda *a, **k: pushed.append(a)),
        )
        notes: list = []
        monkeypatch.setattr(
            console.app_instance,
            "notify",
            MagicMock(side_effect=lambda *a, **k: notes.append(a)),
        )

        await console._handle_console_attach_context(MagicMock())

        assert not pushed, "picker opened despite being at the attachment cap"
        assert any("limit reached" in str(a[0]).lower() for a in notes), notes


@pytest.mark.asyncio
async def test_attach_below_cap_still_opens_picker(monkeypatch):
    """TASK-377 guard: below the cap the Attach button still opens the picker."""
    from unittest.mock import AsyncMock, MagicMock

    from tldw_chatbook.Chat.console_chat_store import MAX_PENDING_ATTACHMENTS

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        for _ in range(MAX_PENDING_ATTACHMENTS - 1):
            store.add_pending_attachment(session.id, _staged_image_attachment())

        pushed: list = []
        monkeypatch.setattr(
            console.app,
            "push_screen",
            AsyncMock(side_effect=lambda *a, **k: pushed.append(a)),
        )
        await console._handle_console_attach_context(MagicMock())

        assert pushed, "picker did not open while below the attachment cap"


def test_resume_hydrates_image_messages_including_image_only_rows():
    """Verify resuming a saved conversation keeps image-only rows and their bytes."""
    screen = ChatScreen(_build_test_app())
    tree = {
        "conversation": {"title": "Saved", "workspace_id": None},
        "root_threads": [
            {
                "id": "m-1",
                "sender": "user",
                "content": "",
                "image_data": b"\x89PNG-bytes",
                "image_mime_type": "image/png",
                "children": [
                    {
                        "id": "m-2",
                        "sender": "assistant",
                        "content": "a red square",
                        "children": [],
                    }
                ],
            }
        ],
    }

    messages = screen._console_messages_from_conversation_tree(tree)

    assert len(messages) == 2
    assert messages[0].image_data == b"\x89PNG-bytes"
    assert messages[0].image_mime_type == "image/png"
    assert messages[0].content == ""
    assert messages[1].content == "a red square"


def test_resume_wiring_injects_agent_markers_from_agent_runs_db(tmp_path):
    """Plan-B final-review Medium-1: the ChatScreen-level wiring
    (`_inject_resume_agent_markers`) must re-derive TOOL markers from the
    real sibling `AgentRunsDB` the same way `_ensure_console_agent_bridge`
    locates it (keyed off `chachanotes_db.db_path`), not just the pure
    helper functions in isolation.

    Task 3: the run carries an `assistant_message_id` matching the
    assistant message's `persisted_message_id`, so this exercises the real
    id-anchored placement path -- not the ordinal fallback that would
    coincidentally land the block in the same place for a single-run,
    single-reply conversation."""
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = SimpleNamespace(
        db_path=str(tmp_path / "chacha.db")
    )
    runs_db = AgentRunsDB(tmp_path / "agent_runs.db", client_id="t")
    primary_id = runs_db.create_run(
        conversation_id="conv-x",
        agent_kind="primary",
        assistant_message_id="asst-42",
    )
    runs_db.append_steps(
        primary_id,
        [
            {
                "index": 0,
                "kind": "tool_result",
                "tool_name": "calculator",
                "result": "42",
                "summary": "",
                "args": None,
                "created_at": "",
            },
        ],
    )
    runs_db.set_status(primary_id, "done", result="It is 42.")

    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="what is 6*7"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="It is 42.",
            status="complete",
            persisted_message_id="asst-42",
        ),
    ]

    resumed = screen._agent._inject_resume_agent_markers(messages, "conv-x")

    tool_rows = [m for m in resumed if m.role is ConsoleMessageRole.TOOL]
    assert len(tool_rows) == 1
    assert tool_rows[0].content == "⚙ calculator → 42"
    assert resumed[-1] is tool_rows[0]  # placed right after the assistant answer

    # Idempotent: injecting again onto the already-injected list adds nothing.
    resumed_again = screen._agent._inject_resume_agent_markers(resumed, "conv-x")
    assert len(resumed_again) == len(resumed)


def test_console_message_serialization_carries_image_metadata_not_bytes():
    """Verify screen-state snapshots carry image metadata but never raw bytes."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        attachment_label="photo.png · 11 B",
    )

    payload = ChatScreen._serialize_console_message(message)

    assert payload["image_mime_type"] == "image/png"
    assert payload["attachment_label"] == "photo.png · 11 B"
    assert "image_data" not in payload

    restored = ChatScreen._restore_console_message(payload)

    assert restored is not None
    assert restored.image_mime_type == "image/png"
    assert restored.attachment_label == "photo.png · 11 B"
    assert restored.image_data is None


@pytest.mark.asyncio
async def test_save_console_message_image_writes_file(tmp_path, monkeypatch):
    """Verify the save-image worker writes the message's image bytes to disk."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(
            "tldw_chatbook.UI.Console_Modules.message.get_cli_setting",
            lambda section, key, default=None: (
                str(tmp_path)
                if (section, key) == ("chat.images", "save_location")
                else default
            ),
        )
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="pic",
            image_data=b"\x89PNG-bytes",
            image_mime_type="image/png",
        )

        await console._save_console_message_image(message.id)

        saved = list(tmp_path.glob("console_image_*.png"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == b"\x89PNG-bytes"


@pytest.mark.asyncio
async def test_save_console_message_image_disambiguates_filename_collision(
    tmp_path, monkeypatch
):
    """Verify the save-image worker never silently overwrites a prior save."""
    import datetime as datetime_module

    class _FixedDateTime(datetime_module.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 12, 0, 0)

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(
            "tldw_chatbook.UI.Console_Modules.message.get_cli_setting",
            lambda section, key, default=None: (
                str(tmp_path)
                if (section, key) == ("chat.images", "save_location")
                else default
            ),
        )
        # The save-image worker imports `datetime.datetime` locally on each
        # call, so freezing the clock here forces both saves below to compute
        # the same base filename and deterministically exercise the
        # collision-disambiguation loop.
        monkeypatch.setattr(datetime_module, "datetime", _FixedDateTime)
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="pic",
            image_data=b"\x89PNG-bytes",
            image_mime_type="image/png",
        )

        await console._save_console_message_image(message.id)
        await console._save_console_message_image(message.id)

        saved = sorted(tmp_path.glob("console_image_*.png"))
        assert len(saved) == 2
        assert saved[0].name != saved[1].name
        assert all(path.read_bytes() == b"\x89PNG-bytes" for path in saved)


@pytest.mark.asyncio
async def test_save_image_button_reflects_the_real_screen_ephemeral_accessor():
    """F6 (task-9 review): pins the *name* ``ConsoleTranscript.
    _console_ephemeral_active()`` reads off the owning screen.

    That reader is ``getattr(screen, "_console_active_session_is_ephemeral",
    None)``, falling back to ``False`` -- the UNSAFE direction, since it
    fails toward writing rather than toward blocking -- when the name is
    missing. A pure ``ConsoleMessageActionService`` test can never catch a
    rename in ``ChatScreen`` because it never calls the reader at all; it
    builds the ``ephemeral`` flag by hand. This mounts the REAL
    ``ChatScreen`` and the REAL ``ConsoleTranscript`` together so a future
    rename of ``_console_active_session_is_ephemeral`` breaks this test
    instead of silently un-blocking Save Image.
    """
    from io import BytesIO

    from PIL import Image as PILImage

    from tldw_chatbook.Chat.console_ephemeral import blocked_reason

    async def _save_image_button_disabled_state(*, ephemeral: bool) -> tuple[bool, str]:
        """Mount a fresh Console, add one image message to a session with
        the given ephemeral flag, select it, and return the real (mounted)
        Save Image button's ``(disabled, tooltip)``. A fresh mount per case
        avoids scrolling a second message into view in the same transcript."""
        app = _build_test_app()
        host = ConsoleHarness(app)
        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")
            store = console._ensure_console_chat_store()

            session = store.create_session(title="S", ephemeral=ephemeral)
            store.switch_session(session.id)
            buffer = BytesIO()
            PILImage.new("RGB", (4, 4), (200, 10, 10)).save(buffer, format="PNG")
            message = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="a picture",
                image_data=buffer.getvalue(),
                image_mime_type="image/png",
            )
            await console._sync_native_console_chat_ui()
            # Image prep runs in a worker; wait for the image row before
            # selecting the message, mirroring the other image-bearing
            # tests in this file (e.g. the terminal-image-rendering test
            # above).
            for _ in range(80):
                if console.query(f"#console-image-{message.id}"):
                    break
                await pilot.pause(0.05)
            assert console.query(f"#console-image-{message.id}"), (
                "image row never appeared"
            )
            # task-14920: `scroll_visible()` + a fixed 0.2s pause raced a later
            # transcript recompose and this click raised OutOfBounds in roughly
            # two whole-file runs in five, on dev and unchanged code alike.
            await _click_after_scrolling_into_view(
                console, pilot, f"#console-message-{message.id}"
            )
            save_selector = f"console-message-action-save-image-{message.id}"
            await _wait_for_selector(console, pilot, f"#{save_selector}")

            save_button = console.query_one(f"#{save_selector}", Button)
            return save_button.disabled, save_button.tooltip

    disabled, tooltip = await _save_image_button_disabled_state(ephemeral=True)
    assert disabled is True
    assert tooltip == blocked_reason("save-image", ephemeral=True)

    # Control: a normal (non-ephemeral) session's Save Image stays enabled.
    normal_disabled, _normal_tooltip = await _save_image_button_disabled_state(
        ephemeral=False
    )
    assert normal_disabled is False


def test_rehydrate_console_message_image_refetches_bytes_from_db():
    """Verify restore rehydration refetches bytes screen-state serialization drops.

    Regression test: `_restore_console_message` intentionally restores metadata
    only (no bytes in screen state), but the controller's payload builder only
    attaches an image when `message.image_data is not None`. Without rehydration
    a message that survives a Console navigate-away/navigate-back round trip
    still shows its chip (metadata-only) but the model never sees the image
    again.
    """
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage

    screen = ChatScreen(_build_test_app())
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="",
        image_mime_type="image/png",
        persisted_message_id="msg-123",
    )
    screen.app_instance.chachanotes_db = Mock(
        get_message_by_id=Mock(
            return_value={
                "image_data": b"\x89PNG-bytes",
                "image_mime_type": "image/png",
            }
        )
    )

    screen._rehydrate_console_message_image(message)

    assert message.image_data == b"\x89PNG-bytes"
    assert message.image_mime_type == "image/png"
    screen.app_instance.chachanotes_db.get_message_by_id.assert_called_once_with(
        "msg-123"
    )


def test_rehydrate_console_message_image_degrades_gracefully_on_db_failure():
    """Verify a DB failure during restore rehydration leaves the message metadata-only."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage

    screen = ChatScreen(_build_test_app())
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="",
        image_mime_type="image/png",
        persisted_message_id="msg-123",
    )
    screen.app_instance.chachanotes_db = Mock(
        get_message_by_id=exploding_double(
            Exception("db offline"),
            reason="the DB read must be attempted before it can degrade",
            awaitable=False,
        )
    )

    screen._rehydrate_console_message_image(message)  # must not raise

    assert message.image_data is None
    assert message.image_mime_type == "image/png"


def test_restore_native_console_state_rehydrates_image_bytes_end_to_end():
    """Verify the full restore path rehydrates bytes for a persisted image message."""
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = Mock(
        get_message_by_id=Mock(
            return_value={
                "image_data": b"\x89PNG-bytes",
                "image_mime_type": "image/png",
            }
        )
    )
    payload = {
        "version": "1.0",
        "active_session_id": "session-1",
        "sessions": [
            {
                "id": "session-1",
                "title": "Saved",
                "workspace_id": None,
                "persisted_conversation_id": None,
                "draft": "",
                "settings": None,
                "updated_at": None,
            }
        ],
        "messages_by_session": {
            "session-1": [
                {
                    "role": "user",
                    "content": "",
                    "id": "m-1",
                    "status": "complete",
                    "persisted_message_id": "msg-123",
                    "image_mime_type": "image/png",
                    "attachment_label": "photo.png · 11 B",
                }
            ]
        },
    }

    _restore_console_snapshot_with_sessions(screen, payload)

    store = screen._ensure_console_chat_store()
    restored = store.messages_for_session("session-1")
    assert len(restored) == 1
    assert restored[0].image_data == b"\x89PNG-bytes"
    assert restored[0].image_mime_type == "image/png"


def test_restore_native_console_state_rehydrates_generation_metadata_end_to_end():
    """task-558: the in-memory (tab-switch) restore path must also hydrate
    ``generation_metadata``, the same way it already hydrates attachments --
    today only the DB-resume path (``restore_persisted_session``) does this,
    so a tab-switch-restored generation message loses its card (empty
    ``generation_metadata``) even though the sidecar rows are still in the
    DB. ``_serialize_console_message`` never serializes
    ``generation_metadata`` (no-bytes-in-screen-state policy extends to the
    sidecar row's provenance), so this can only come back via a batched
    ``get_generation_metadata_for_messages`` fetch during restore.
    """
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = Mock(
        get_generation_metadata_for_messages=Mock(
            return_value={
                "msg-123": [
                    {
                        "position": 0,
                        "prompt": "a red dragon",
                        "negative_prompt": "",
                        "backend": "swarmui",
                        "model": None,
                        "seed": 7,
                        "style": None,
                        "params_json": "{}",
                    }
                ]
            }
        ),
        get_message_by_id=Mock(return_value=None),
    )
    payload = {
        "version": "1.0",
        "active_session_id": "session-1",
        "sessions": [
            {
                "id": "session-1",
                "title": "Saved",
                "workspace_id": None,
                "persisted_conversation_id": None,
                "draft": "",
                "settings": None,
                "updated_at": None,
            }
        ],
        "messages_by_session": {
            "session-1": [
                {
                    "role": "assistant",
                    "content": "[image] a red dragon",
                    "id": "m-1",
                    "status": "complete",
                    "persisted_message_id": "msg-123",
                    "image_mime_type": "image/png",
                }
            ]
        },
    }

    _restore_console_snapshot_with_sessions(screen, payload)

    store = screen._ensure_console_chat_store()
    restored = store.messages_for_session("session-1")
    assert len(restored) == 1
    assert len(restored[0].generation_metadata) == 1
    assert restored[0].generation_metadata[0].seed == 7
    assert restored[0].generation_metadata[0].backend == "swarmui"
    screen.app_instance.chachanotes_db.get_generation_metadata_for_messages.assert_called_once_with(
        ["msg-123"]
    )


def test_restore_generation_metadata_survives_stale_session_key_in_payload():
    """A ``messages_by_session`` key with no matching ``sessions`` entry must
    not abort the restore. Pinned as evidence against a review claim that the
    generation-metadata rehydrate loop could hit
    ``hydrate_generation_metadata``'s unknown-session lookup: it structurally
    cannot, because ``_restore_native_console_state`` REBUILDS
    ``restored_messages_by_session`` keyed only by the ids of sessions it
    actually constructs (``messages_by_session.get(session.id)``), so a
    stale/orphaned payload key never reaches the hydrate loop at all.
    """
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = Mock(
        get_generation_metadata_for_messages=Mock(
            return_value={
                "msg-123": [
                    {
                        "position": 0,
                        "prompt": "a red dragon",
                        "negative_prompt": "",
                        "backend": "swarmui",
                        "model": None,
                        "seed": 7,
                        "style": None,
                        "params_json": "{}",
                    }
                ]
            }
        ),
        get_message_by_id=Mock(return_value=None),
    )
    payload = {
        "version": "1.0",
        "active_session_id": "session-1",
        "sessions": [
            {
                "id": "session-1",
                "title": "Saved",
                "workspace_id": None,
                "persisted_conversation_id": None,
                "draft": "",
                "settings": None,
                "updated_at": None,
            }
        ],
        "messages_by_session": {
            "session-1": [
                {
                    "role": "assistant",
                    "content": "[image] a red dragon",
                    "id": "m-1",
                    "status": "complete",
                    "persisted_message_id": "msg-123",
                    "image_mime_type": "image/png",
                }
            ],
            "session-stale": [
                {
                    "role": "assistant",
                    "content": "[image] orphaned",
                    "id": "m-2",
                    "status": "complete",
                    "persisted_message_id": "msg-999",
                    "image_mime_type": "image/png",
                }
            ],
        },
    }

    _restore_console_snapshot_with_sessions(screen, payload)

    store = screen._ensure_console_chat_store()
    restored = store.messages_for_session("session-1")
    assert len(restored) == 1
    assert len(restored[0].generation_metadata) == 1


@pytest.mark.asyncio
async def test_clear_attachment_button_resyncs_composer_blocked_state(monkeypatch):
    """Verify clicking Clear on a staged image resyncs the composer's blocked visuals.

    Regression test: `_process_console_attachment` calls `_sync_console_control_bar()`
    after staging, so the composer immediately reflects the "can't accept images"
    block. `handle_console_clear_attachment` used to skip that sync, leaving the
    composer showing a stale blocked-send state (and tooltip) after the
    attachment was removed via the ✕ button.
    """
    import tldw_chatbook.Chat.attachment_core as attachment_core

    monkeypatch.setattr(attachment_core, "is_vision_capable", lambda p, m: False)
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.set_pending_attachment(session.id, _staged_image_attachment())
        # Mirror the sync `_process_console_attachment` performs right after
        # staging, so the composer starts in the same blocked state a real
        # attach would leave behind.
        console._sync_console_control_bar()
        await pilot.pause()

        send_button = console.query_one("#console-send-message", Button)
        assert composer._send_blocked is True
        assert send_button.tooltip and "can't accept images" in send_button.tooltip

        await pilot.click("#console-clear-attachment")
        await pilot.pause()

        assert store.pending_attachment(session.id) is None
        assert composer._send_blocked is False
        assert (
            not send_button.tooltip or "can't accept images" not in send_button.tooltip
        )


@pytest.mark.asyncio
async def test_image_message_gets_inline_row_after_prep_and_toggle_cycles():
    app = _build_test_app()
    _configure_native_ready_console(app)
    # Pin the session default so the pixels -> graphics -> hidden -> pixels
    # cycle below is deterministic; leaving this on "auto" resolves from the
    # host terminal's TERM/TERM_PROGRAM env vars (see resolve_default_mode),
    # which varies across dev machines and CI.
    #
    # task-15511: the pin must land where production READS it.
    # `_chat_images_config` prefers `COMPREHENSIVE_CONFIG_RAW` whenever the
    # config carries one -- and the real (task-15270) harness config always
    # does -- so writing only `app_config["chat"]` is the dead seam the
    # 15270 triage called out by name: the pin never reached production, the
    # ambient default resolved from the terminal overrides (graphics on this
    # machine), and the FIRST toggle landed on hidden -- the row vanishing
    # one step early was this test's premise being lost, not a render bug.
    images_pin = {"images": {"default_render_mode": "pixels"}}
    app.app_config["chat"] = dict(images_pin)
    raw = app.app_config.get("COMPREHENSIVE_CONFIG_RAW")
    if isinstance(raw, dict):
        raw.setdefault("chat", {})["images"] = dict(images_pin["images"])
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        from io import BytesIO

        from PIL import Image as PILImage

        buffer = BytesIO()
        PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buffer, format="PNG")
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="look at this",
            image_data=buffer.getvalue(),
            image_mime_type="image/png",
        )
        await console._sync_native_console_chat_ui()
        # Prep runs in a worker; wait for the image row to appear.
        for _ in range(80):
            if console.query(f"#console-image-{message.id}"):
                break
            await pilot.pause(0.05)
        assert console.query(f"#console-image-{message.id}"), "image row never appeared"

        # Toggle: pixels -> graphics (widget swaps, still present)
        console._image._handle_console_toggle_image_view(message.id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        assert console.query(f"#console-image-{message.id}")

        # Toggle: graphics -> hidden (row disappears)
        console._image._handle_console_toggle_image_view(message.id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        assert not console.query(f"#console-image-{message.id}")

        # Toggle: hidden -> pixels (row returns)
        console._image._handle_console_toggle_image_view(message.id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        assert console.query(f"#console-image-{message.id}")


def test_image_view_modes_ride_screen_state_allowlist_and_prune_stale():
    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    state, _cache = screen._ensure_console_image_view()
    state.restore({message.id: "hidden", "stale-id": "graphics"})

    payload = _console_snapshot_with_sessions(screen)
    assert payload is not None
    # Live override survives; the stale one is pruned at serialize time.
    assert payload["image_view_modes"] == {message.id: "hidden"}

    fresh = ChatScreen(app)
    _restore_console_snapshot_with_sessions(fresh, payload)
    fresh_state, _ = fresh._ensure_console_image_view()
    assert fresh_state.serialize() == {message.id: "hidden"}


def test_console_image_prep_bounded_to_cache_capacity_avoids_churn():
    """Regression: prep must never chase more images than the cache can hold.

    Before this fix, the sync path computed `cache.pending_ids(messages)`
    over the FULL session while `ConsoleImageRenderCache` is LRU-bounded at
    `IMAGE_CACHE_MAX_ENTRIES`. With more image messages than the cache holds,
    each sync would prep an older message, evict the newest one to make room,
    and the next sync would re-prep the evicted one — an infinite decode +
    refresh churn. `_build_console_image_specs` (and the sync-site pending
    computation) must bound their working set to the most-recent-N
    image-bearing messages so the working set can never exceed cache
    capacity.
    """
    from io import BytesIO

    from PIL import Image as PILImage

    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()

    total_images = IMAGE_CACHE_MAX_ENTRIES + 3
    messages = []
    for index in range(total_images):
        buffer = BytesIO()
        PILImage.new("RGB", (4, 4), (index % 256, 20, 30)).save(buffer, format="PNG")
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=f"pic {index}",
            image_data=buffer.getvalue(),
            image_mime_type="image/png",
        )
        messages.append(message)

    _state, cache = screen._ensure_console_image_view()
    recent = screen._recent_console_image_messages(messages)
    most_recent_ids = [m.id for m in messages[-IMAGE_CACHE_MAX_ENTRIES:]]
    assert len(recent) == IMAGE_CACHE_MAX_ENTRIES
    assert [m.id for m in recent] == most_recent_ids

    # Prepare exactly the bounded (most-recent) subset via the cache
    # directly, mirroring what the fixed sync-site prep kick does.
    for message_id, image_data in cache.pending_ids(recent):
        cache.prepare(message_id, image_data)

    # (a) + (b): specs are bounded to cache capacity and are the most recent
    # image messages — older messages were never prepared, so they can never
    # appear here regardless of how many messages the session holds.
    specs = screen._image._build_console_image_specs(messages)
    assert len(specs) <= IMAGE_CACHE_MAX_ENTRIES
    assert set(specs) == set(most_recent_ids)

    # The older, out-of-window messages were never touched by prep.
    older_ids = [m.id for m in messages[:-IMAGE_CACHE_MAX_ENTRIES]]
    assert older_ids  # sanity: the test actually exceeds cache capacity
    for older_id in older_ids:
        assert cache.get_pil(older_id) is None

    # (c) No churn: recomputing pending over the same bounded subset finds
    # nothing left to prepare — the working set converges instead of
    # flapping between decode and eviction.
    assert cache.pending_ids(screen._recent_console_image_messages(messages)) == []


def test_console_image_prep_kick_skips_ids_already_preparing():
    """Regression: the 0.2s sync tick must not re-kick prep for ids a
    worker is already chewing on.

    Before this fix, every sync tick recomputed `cache.pending_ids(...)`
    over the still-uncached image messages and unconditionally kicked the
    exclusive `console-image-prep` worker for them — cancelling any
    in-flight run and piling duplicate decodes into the executor.
    `_console_image_preparing` tracks in-flight ids so the kick site's
    filtered pending list converges to empty once a batch is staged.
    """
    from io import BytesIO

    from PIL import Image as PILImage

    app = _build_test_app()
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()

    buffer = BytesIO()
    PILImage.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format="PNG")
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=buffer.getvalue(),
        image_mime_type="image/png",
    )

    messages = [message]
    _state, cache = screen._ensure_console_image_view()

    # Same helper chain the sync site uses to compute the raw pending set.
    recent = screen._recent_console_image_messages(messages)
    assert cache.pending_ids(recent) == [(message.id, message.image_data)]

    # Stage the id as already-preparing, exactly as the kick site does right
    # before `run_worker`.
    screen._console_image_preparing.update(mid for mid, _ in cache.pending_ids(recent))
    assert message.id in screen._console_image_preparing

    # The filtered pending list the kick site actually acts on must now be
    # empty — a re-kick for the same id must not fire while it's in flight.
    pending_images = [
        (mid, data)
        for mid, data in cache.pending_ids(recent)
        if mid not in screen._console_image_preparing
    ]
    assert pending_images == []


@pytest.mark.asyncio
async def test_path_paste_routes_to_attach_instead_of_draft(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    from textual.events import Paste

    image_path = tmp_path / "dropped.png"
    PILImage.new("RGB", (8, 8), (9, 9, 9)).save(image_path, format="PNG")

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        # Widen the attach root to tmp_path for both gating and processing.
        # chat_screen imports these helpers BY NAME, so patch the consuming
        # module's bindings, not the source module's.
        import tldw_chatbook.Chat.attachment_core as attachment_core
        import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
        from tldw_chatbook.Chat.console_paste_attach import (
            looks_attachable as original_attachable,
        )

        original_load = attachment_core.load_processed_file

        async def _rooted(file_path, *, allowed_root=None):
            return await original_load(file_path, allowed_root=str(tmp_path))

        monkeypatch.setattr(attachment_core, "load_processed_file", _rooted)
        monkeypatch.setattr(
            chat_screen_module,
            "looks_attachable",
            lambda path, allowed_root=None: original_attachable(
                path, allowed_root=str(tmp_path)
            ),
        )

        console.on_paste(Paste(text=str(image_path)))
        for _ in range(80):
            store = console._ensure_console_chat_store()
            session_id = store.active_session_id
            if session_id and store.pending_attachment(session_id) is not None:
                break
            await pilot.pause(0.05)

        store = console._ensure_console_chat_store()
        pending = store.pending_attachment(store.active_session_id)
        assert pending is not None and pending.file_type == "image"
        assert composer.draft_text() == ""  # path did NOT land as draft text


@pytest.mark.asyncio
async def test_prose_paste_still_lands_in_draft():
    from textual.events import Paste

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        console.on_paste(Paste(text="what does /etc/hosts do?"))
        await pilot.pause()
        assert composer.draft_text() == "what does /etc/hosts do?"


@pytest.mark.asyncio
async def test_alt_v_grabs_clipboard_image_into_pending(monkeypatch):
    from io import BytesIO

    from PIL import Image as PILImage

    import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module

    buffer = BytesIO()
    PILImage.new("RGB", (16, 16), (10, 200, 10)).save(buffer, format="PNG")
    png = buffer.getvalue()

    from tldw_chatbook.Chat.console_paste_attach import ClipboardGrab

    monkeypatch.setattr(
        chat_screen_module,
        "grab_clipboard_image",
        lambda: ClipboardGrab(kind="image", png_bytes=png),
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console.query_one("#console-native-composer", ConsoleComposerBar).focus()
        await pilot.pause()

        await pilot.press("alt+v")
        for _ in range(80):
            store = console._ensure_console_chat_store()
            sid = store.active_session_id
            if sid and store.pending_attachment(sid) is not None:
                break
            await pilot.pause(0.05)

        store = console._ensure_console_chat_store()
        pending = store.pending_attachment(store.active_session_id)
        assert pending is not None
        assert pending.file_type == "image"
        assert pending.display_name.startswith("clipboard-")


@pytest.mark.asyncio
async def test_alt_v_unavailable_platform_toasts(monkeypatch):
    import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module

    from tldw_chatbook.Chat.console_paste_attach import ClipboardGrab

    monkeypatch.setattr(
        chat_screen_module,
        "grab_clipboard_image",
        lambda: ClipboardGrab(kind="unavailable"),
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    notifications: list[str] = []
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(
            console.app_instance,
            "notify",
            lambda message, **kwargs: notifications.append(str(message)),
        )
        await pilot.press("alt+v")
        for _ in range(40):
            if notifications:
                break
            await pilot.pause(0.05)
        assert any("aren't readable on this platform" in n for n in notifications)
        store = console._ensure_console_chat_store()
        sid = store.active_session_id
        assert sid is None or store.pending_attachment(sid) is None


@pytest.mark.asyncio
async def test_alt_v_action_is_inert_while_setup_modal_blocks(monkeypatch):
    import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module

    grab_calls: list[None] = []
    monkeypatch.setattr(
        chat_screen_module,
        "grab_clipboard_image",
        lambda: grab_calls.append(None),
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: True)

        console.action_paste_clipboard_image()
        await pilot.pause(0.2)

        assert grab_calls == []
        store = console._ensure_console_chat_store()
        sid = store.active_session_id
        assert sid is None or store.pending_attachment(sid) is None


@pytest.mark.asyncio
async def test_staging_appends_and_caps_at_five(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    paths = []
    for index in range(6):
        p = tmp_path / f"img{index}.png"
        PILImage.new("RGB", (8, 8), (index * 30, 9, 9)).save(p, format="PNG")
        paths.append(p)

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    notifications: list[str] = []
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        import tldw_chatbook.Chat.attachment_core as attachment_core

        original_load = attachment_core.load_processed_file

        async def _rooted(file_path, *, allowed_root=None):
            return await original_load(file_path, allowed_root=str(tmp_path))

        monkeypatch.setattr(attachment_core, "load_processed_file", _rooted)
        monkeypatch.setattr(
            console.app_instance,
            "notify",
            lambda message, **kwargs: notifications.append(str(message)),
        )

        store = console._ensure_console_chat_store()
        for index in range(5):
            await console._process_console_attachment(str(paths[index]))
            session_id = store.active_session_id
            assert len(store.pending_attachments(session_id)) == index + 1

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        # The composer prepends its own 📎 glyph, so the label PARAMETER is
        # glyph-free while the RENDERED indicator carries exactly one glyph.
        assert composer._pending_attachment_label == "5 files"
        await pilot.pause()
        indicator = console.query_one("#console-attachment-indicator", Static)
        rendered = str(indicator.renderable)
        assert "📎 5 files" in rendered
        assert "📎 📎" not in rendered

        await console._process_console_attachment(str(paths[5]))
        assert len(store.pending_attachments(store.active_session_id)) == 5
        assert any(
            "Attachment limit reached (5 per message)." in n for n in notifications
        )


@pytest.mark.asyncio
async def test_save_image_saves_all_attachments(tmp_path, monkeypatch):
    from tldw_chatbook.Chat.console_chat_models import MessageAttachment

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    notifications: list[str] = []
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(
            "tldw_chatbook.UI.Console_Modules.message.get_cli_setting",
            lambda section, key, default=None: (
                str(tmp_path)
                if (section, key) == ("chat.images", "save_location")
                else default
            ),
        )
        monkeypatch.setattr(
            console.app_instance,
            "notify",
            lambda message, **kwargs: notifications.append(str(message)),
        )
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="three",
            attachments=(
                MessageAttachment(
                    data=b"img-0",
                    mime_type="image/png",
                    display_name="a.png",
                    position=0,
                ),
                MessageAttachment(
                    data=b"img-1",
                    mime_type="image/png",
                    display_name="b.png",
                    position=1,
                ),
                MessageAttachment(
                    data=b"img-2",
                    mime_type="image/jpeg",
                    display_name="c.jpg",
                    position=2,
                ),
            ),
        )

        await console._save_console_message_image(message.id)

        saved = sorted(tmp_path.glob("console_image_*"))
        assert len(saved) == 3
        assert any("Saved 3 images to" in n for n in notifications)


def test_console_message_serialization_round_trips_multi_attachment_labels():
    """Verify multi-attachment messages serialize to labels-only and restore metadata-only.

    Companion to `test_console_message_serialization_carries_image_metadata_not_bytes`:
    that test covers the single-attachment (scalar-only) shape, this one
    covers the `attachment_labels` list a 2+ attachment message carries.
    """
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        MessageAttachment,
    )

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="two files",
        image_data=b"\x89PNG-a",
        image_mime_type="image/png",
        attachment_label="a.png",
        attachments=(
            MessageAttachment(
                data=b"\x89PNG-a",
                mime_type="image/png",
                display_name="a.png",
                position=0,
            ),
            MessageAttachment(
                data=b"\x89PNG-b",
                mime_type="image/png",
                display_name="b.png",
                position=1,
            ),
        ),
    )

    payload = ChatScreen._serialize_console_message(message)

    assert payload["attachment_labels"] == ["a.png", "b.png"]
    assert "image_data" not in payload
    assert not any(isinstance(value, (bytes, bytearray)) for value in payload.values())

    restored = ChatScreen._restore_console_message(payload)

    assert restored is not None
    assert len(restored.attachments) == 2
    assert [a.display_name for a in restored.attachments] == ["a.png", "b.png"]
    assert all(a.data is None for a in restored.attachments)


def test_console_screen_state_round_trips_the_temporary_flag():
    """A temporary chat must not become a persisting one by navigating away.

    `_serialize_native_console_state` writes an explicit field list; a field
    missing from it is silently dropped on restore. For `ephemeral` that
    drop is not cosmetic -- the next send would write the chat to the
    database.
    """
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    screen._session = ConsoleSessionController.__new__(ConsoleSessionController)

    # A REAL round trip: the serializer's own output feeds the restorer.
    # Asserting on a hand-built dict would test neither half.
    temporary = ConsoleChatSession(title="Temporary chat", ephemeral=True)
    payload = screen._session._console_session_to_state(temporary)
    assert payload["ephemeral"] is True
    assert screen._session._console_session_from_state(payload).ephemeral is True

    normal = ConsoleChatSession(title="Normal chat")
    assert (
        screen._session._console_session_from_state(
            screen._session._console_session_to_state(normal)
        ).ephemeral
        is False
    )

    # Legacy payloads predate the key entirely.
    assert (
        screen._session._console_session_from_state(
            {"id": normal.id, "title": normal.title}
        ).ephemeral
        is False
    ), "a payload with no key must default to saved"


def test_console_screen_state_round_trips_only_project_instruction_controls():
    screen = ChatScreen.__new__(ChatScreen)
    screen._session = ConsoleSessionController.__new__(ConsoleSessionController)
    state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-1",
        working_folder_locator_fingerprint="f" * 64,
        project_instruction_notice_key="n" * 64,
    )
    session = ConsoleChatSession(project_instruction_state=state)

    payload = screen._session._console_session_to_state(session)
    restored = screen._session._console_session_from_state(payload)

    assert restored.project_instruction_state == state
    encoded = json.dumps(payload)
    assert "/Users/" not in encoded
    assert "AGENTS" not in encoded
    assert "instruction body" not in encoded


@pytest.mark.parametrize(
    "raw_project_state",
    [
        None,
        {"version": 99},
        {"version": 1, "project_instructions_enabled": "yes"},
        {
            "version": 1,
            "project_instructions_enabled": True,
            "working_folder_binding_id": "binding-1",
            "working_folder_locator_fingerprint": "f" * 64,
            "project_instruction_notice_key": None,
            "raw_path": "/private/repo",
        },
    ],
)
def test_console_screen_state_invalid_project_instruction_state_fails_disabled(
    raw_project_state,
):
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    payload = {"id": "session", "project_instructions": raw_project_state}
    restored = controller._console_session_from_state(payload)
    assert restored.project_instruction_state == (
        ProjectInstructionControlState.legacy_disabled()
    )


def test_temporary_tab_marker_is_presentation_only():
    """The marker must never enter session.title.

    Promotion saves `session.title` verbatim, so a marker written into the
    title would produce a saved conversation literally named after it -- and
    renaming would then fight the marker on every render.
    """
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
    from tldw_chatbook.Chat.console_glyphs import GLYPH_TEMPORARY
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        CONSOLE_SESSION_TAB_DISPLAY_CHARS,
        ConsoleSessionSurface,
        _session_tab_tooltip,
    )

    session = ConsoleChatSession(title="Vector store notes", ephemeral=True)
    label = ConsoleSessionSurface._tab_label(session.title, ephemeral=True)

    assert label.startswith(GLYPH_TEMPORARY)
    assert "Vector store" in label
    assert len(label) <= CONSOLE_SESSION_TAB_DISPLAY_CHARS + 2  # glyph + space
    assert GLYPH_TEMPORARY not in session.title

    plain = ConsoleSessionSurface._tab_label(session.title, ephemeral=False)
    assert GLYPH_TEMPORARY not in plain

    tooltip = _session_tab_tooltip(session, active=False)
    assert "not saved" in tooltip.lower()
    # CN-02 (TASK-2154.13): the tab tooltip's ◌ decode is byte-for-byte the
    # status chip's TEMPORARY_LABEL -- one short name for the concept, no
    # third wording.
    from tldw_chatbook.Chat.console_ephemeral import TEMPORARY_LABEL

    assert tooltip.endswith(f"{TEMPORARY_LABEL}.")
    assert (
        "not saved"
        not in _session_tab_tooltip(
            ConsoleChatSession(title="Normal"), active=False
        ).lower()
    )


# ---------------------------------------------------------------------------
# FB-07 (TASK-2154.17): success confirmations for save/retry/settings actions.
# ---------------------------------------------------------------------------


def _capture_notify_severities(app) -> list[tuple[str, str]]:
    """Replace app.notify with a (message, severity) capture list."""
    notifications: list[tuple[str, str]] = []
    app.notify = lambda message, **kwargs: notifications.append(
        (str(message), str(kwargs.get("severity", "information")))
    )
    return notifications


@pytest.mark.asyncio
async def test_console_save_as_savers_confirm_at_success_severity():
    """All four Save-as destinations toast severity="success" (FB-07)."""
    app = _build_test_app()
    _install_console_save_service_fakes(app)
    notifications = _capture_notify_severities(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session(title="Chat 1")
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
        )
        await console._sync_native_console_chat_ui()

        # task-14920: decomposition wave 3 (391b7bf69) moved the Save-as
        # destinations onto `ConsoleMessageController` and left a ChatScreen
        # delegator for `_as_note` only, so the other three are reached
        # through `_message`. This test was written after that move and had
        # therefore never once run green.
        await console._save_console_message_as_note(message.id)
        await console._message._save_console_message_as_media(message.id)
        await console._message._save_console_message_as_prompt(message.id)
        await console._message._save_console_message_as_chatbook(message.id)

    success_toasts = [m for m, severity in notifications if severity == "success"]
    assert "Saved message as Note." in success_toasts
    assert "Saved message as Media. It appears under Library ▸ Media." in success_toasts
    assert any(m.startswith("Saved message as Prompt '") for m in success_toasts), (
        success_toasts
    )
    assert (
        "Saved message as a Chatbook artifact. It appears under Artifacts."
        in success_toasts
    )


@pytest.mark.asyncio
async def test_console_retry_accepted_fires_success_toast():
    """Retrying a failed response confirms at success severity (FB-07)."""
    gateway = FailThenRecoverGateway()
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    notifications = _capture_notify_severities(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "llama.cpp stream failed")

        store = console._ensure_console_chat_store()
        failed = next(
            message
            for message in reversed(store.messages_for_session(store.active_session_id))
            if message.role is ConsoleMessageRole.ASSISTANT
            and message.status == "failed"
        )
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(failed.id)
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(
            console, pilot, f"#console-message-action-retry-{failed.id}"
        )
        success_before = [m for m, severity in notifications if severity == "success"]
        await pilot.click(f"#console-message-action-retry-{failed.id}")
        await _wait_for_text(console, pilot, "recovered")

    success_after = [m for m, severity in notifications if severity == "success"]
    assert len(success_after) == len(success_before) + 1
    assert "Retrying failed response." in success_after


@pytest.mark.asyncio
async def test_console_settings_save_fires_success_toast():
    """Saving the Console settings modal confirms at success severity (FB-07)."""
    app = _build_test_app()
    notifications = _capture_notify_severities(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        # The palette action is guarded by setup state; the inspector rail's
        # settings button is the established modal route in tests.
        rail_state = replace(
            console._current_console_rail_state(),
            right_open=True,
        )
        console._sync_console_rail_visibility(rail_state)
        await _wait_for_selector(console, pilot, "#console-settings-open")
        console.query_one("#console-settings-open", Button).press()
        modal = None
        for _ in range(50):
            await pilot.pause(0.1)
            if host.screen_stack[-1].query("#console-settings-modal"):
                modal = host.screen_stack[-1]
                break
        assert modal is not None, "ConsoleSettingsModal never opened"
        # task-14920: decomposition wave 2 (4de93c10d) moved this seam onto
        # `ConsoleSessionController` without a ChatScreen delegator; this
        # test was written afterwards and had never run green.
        settings = console._session._ensure_active_console_session_settings()
        modal.dismiss(settings)
        await pilot.pause(0.5)

    assert ("Console settings saved.", "success") in notifications


@pytest.mark.asyncio
async def test_console_save_chatbook_handoff_fires_success_toast():
    """The composer Save Chatbook button confirms the handoff (FB-07)."""
    from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch

    app = _build_test_app()
    notifications = _capture_notify_severities(app)
    app.open_console_live_work_primary_action = Mock(return_value=True)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._pending_console_launch_context = ConsoleLiveWorkLaunch.from_values(
            source="artifacts",
            title="Run artifact",
            payload={"target_id": "run-1:chatbook:7"},
        )
        console._save_console_chatbook_from_visible_action()
        await pilot.pause()

    app.open_console_live_work_primary_action.assert_called_once()
    assert ("Saved — opening the artifact in Artifacts.", "success") in notifications


@pytest.mark.asyncio
async def test_console_routine_send_fires_no_success_toast():
    """AC guard: a plain successful send must NOT gain a success toast."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: CapturingGateway(
        chunks=("hel", "lo")
    )
    notifications = _capture_notify_severities(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "hello")

    success_toasts = [m for m, severity in notifications if severity == "success"]
    assert success_toasts == []
