"""Characterisation + contract tests for the Console prompt cluster.

Written BEFORE the wave-3 task-3 extraction of `ConsolePromptsController`
out of `ChatScreen`, so every assertion below is a statement about
behaviour that must survive the move unchanged:

* the real `_open_console_prompts_modal` path (a live screen, a live chat
  store, a real `ConsolePromptsModal` on the stack) and the SELECTION it
  persists -- `apply_improvement_result` writing the reviewed System
  prompt through `ConsoleChatStore.set_session_system_prompt`, plus its
  stale-guard refusing when the live System prompt moved underneath it;
* `/prompt <name>` and `/system <name>` name resolution, including their
  shared refuse-a-Recipe guard;
* the `/system` editor's save-to-Library outcome copy;
* the Library "Use in Console" staged-insert handoff;
* the lazily-created shared prompt-history store.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsolePromptsModal
from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsResult


class ConsoleHarness(App):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _configure_native_ready_console(app, model: str = "local-model") -> None:
    """A send-ready Console so the blocking first-run setup modal stays hidden."""
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": model}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": model}
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = model


def _prompt_record(
    *,
    name: str = "Summarize",
    artifact_type: str = "prompt",
    system_prompt: str = "You are terse.",
    user_prompt: str = "Summarize the following.",
) -> dict[str, object]:
    return {
        "id": "local:prompt:11",
        "local_id": 11,
        "source_id": "11",
        "name": name,
        "artifact_type": artifact_type,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "version": 1,
    }


class _PromptScopeService:
    """The `app_instance.prompt_scope_service` seam the cluster reads."""

    def __init__(self, record: dict[str, object] | None = None) -> None:
        self.record = record if record is not None else _prompt_record()
        self.search_calls: list[dict[str, object]] = []
        self.saved: list[dict[str, object]] = []
        self.existing: dict[str, object] | None = None
        self.save_result: object = {"local_id": 42}

    async def get_capabilities(self, *, mode: str):
        return SimpleNamespace(
            structured_kinds=frozenset(),
            artifact_types=frozenset({"prompt", "recipe"}),
            conditional_update=True,
        )

    async def list_prompts(self, *, mode: str, page: int, per_page: int):
        return {
            "items": [dict(self.record)],
            "page": page,
            "per_page": per_page,
            "total_items": 1,
            "total_pages": 1,
        }

    async def search_prompts(self, *, mode: str, query: str, limit: int, **kwargs):
        self.search_calls.append({"mode": mode, "query": query, "limit": limit, **kwargs})
        return [dict(self.record)]

    async def get_prompt(self, *, mode: str, prompt_identifier: str, **kwargs):
        return self.existing

    async def save_prompt(self, *, mode: str, **payload):
        self.saved.append({"mode": mode, **payload})
        return self.save_result


# ---------------------------------------------------------------------------
# The real modal-open path, and the selection it persists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_modal_open_persists_the_reviewed_system_selection() -> None:
    """Drive `_open_console_prompts_modal` for real, then apply through the
    modal's own host coordinator and assert the store PERSISTED it."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("draft body")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsolePromptsModal)

        result = ConsolePromptsResult(
            kind="apply",
            composer_snapshot=composer.capture_draft_snapshot(),
            user_text=None,
            system_text="You answer in one sentence.",
            apply_user=False,
            apply_system=True,
            captured_system_fingerprint=None,
        )
        outcome = await modal._apply_improvement_result(result, None)

        assert outcome.kind == "applied"
        settings = console._session._ensure_active_console_session_settings()
        assert settings.system_prompt == "You answer in one sentence."
        assert (
            store.session_settings(session_id).system_prompt
            == "You answer in one sentence."
        )


@pytest.mark.asyncio
async def test_prompts_modal_apply_refuses_when_the_system_prompt_moved() -> None:
    """The opening fingerprint is the guard: a System prompt changed behind
    the modal must make the apply stale rather than clobber it."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]

        # Somebody else moved the live System prompt after the modal opened.
        store.set_session_system_prompt(session_id, "Changed elsewhere.")

        result = ConsolePromptsResult(
            kind="apply",
            composer_snapshot=composer.capture_draft_snapshot(),
            user_text=None,
            system_text="Never lands.",
            apply_user=False,
            apply_system=True,
            captured_system_fingerprint=None,
        )
        outcome = await modal._apply_improvement_result(result, None)

        assert outcome.kind == "stale"
        assert (
            store.session_settings(session_id).system_prompt == "Changed elsewhere."
        )


@pytest.mark.asyncio
async def test_prompts_modal_reads_provider_recovery_off_the_screen_at_open() -> None:
    """The Configure-provider seam is resolved when the modal is BUILT, so a
    screen-level replacement made beforehand must reach the modal."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        recovery = AsyncMock()
        console._open_console_provider_recovery = recovery
        console._console_provider_blocker_copy = lambda: "No provider configured."

        console._open_console_prompts_modal()
        await pilot.pause()
        modal = host.screen_stack[-1]

        assert modal._configure_provider is recovery
        assert modal._improve_unavailable_reason == "No provider configured."


# ---------------------------------------------------------------------------
# `/prompt` and `/system` name resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_command_replaces_the_draft_with_the_resolved_body() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("/prompt Summarize")

        await console._console_command_insert_prompt(
            SimpleNamespace(args="Summarize")
        )
        await pilot.pause()

        assert composer.draft_text() == "Summarize the following."


@pytest.mark.asyncio
async def test_system_command_applies_and_persists_the_resolved_system_part() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("/system Summarize")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id

        await console._console_command_apply_system(SimpleNamespace(args="Summarize"))
        await pilot.pause()

        assert store.session_settings(session_id).system_prompt == "You are terse."
        # A handled command never leaves its own invocation text behind.
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_prompt_search_filters_recipes_and_uses_a_prefix_fts_query() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = _PromptScopeService(_prompt_record(artifact_type="recipe"))
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        records = await console._console_prompt_search("Summ")

        assert records == []
        assert service.search_calls[-1]["fts_match_query"] == '"Summ"*'
        assert await console._resolve_console_prompt_by_name("Summ") is None


def test_recipe_records_are_never_executable() -> None:
    assert ChatScreen._is_recipe_prompt_record({"artifact_type": "Recipe"}) is True
    assert ChatScreen._is_recipe_prompt_record({"artifact_type": "prompt"}) is False
    assert ChatScreen._is_recipe_prompt_record({}) is False


def test_prompt_prefix_fts_query_escapes_embedded_quotes() -> None:
    assert ChatScreen._console_prompt_prefix_fts_query('a"b') == '"a""b"*'


# ---------------------------------------------------------------------------
# Save-to-Library, handoff insert, prompt history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_prompt_save_to_library_reports_create_and_collision() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    service = _PromptScopeService()
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        assert (
            await console._save_console_system_prompt_to_library("", "body")
            == "Enter a name to save this system prompt to Library."
        )
        assert (
            await console._save_console_system_prompt_to_library("Name", "  ")
            == "Enter a system prompt to save."
        )
        assert (
            await console._save_console_system_prompt_to_library("Fresh", "Be terse.")
            == "Saved."
        )
        assert service.saved[-1]["name"] == "Fresh"
        assert service.saved[-1]["system_prompt"] == "Be terse."

        service.existing = {"name": "Fresh", "id": 1}
        outcome = await console._save_console_system_prompt_to_library(
            "Fresh", "Be terse."
        )
        assert "already in use" in outcome


@pytest.mark.asyncio
async def test_prompt_history_store_is_lazy_shared_and_factory_seamed() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    sentinel = object()
    app.console_prompt_history_factory = lambda: sentinel
    screen = ChatScreen(app)

    first = screen._ensure_console_prompt_history()

    assert first is sentinel
    assert screen._ensure_console_prompt_history() is sentinel


@pytest.mark.asyncio
async def test_library_prompt_insert_handoff_appends_onto_the_live_draft() -> None:
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.clear_draft()
        composer.insert_text("existing")
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT, "staged body"
        )

        await console._consume_pending_console_prompt_insert()
        await pilot.pause()

        assert composer.draft_text() == "existing\nstaged body"
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )


@pytest.mark.asyncio
async def test_library_prompt_insert_handoff_is_blocked_before_setup_completes() -> None:
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = _PromptScopeService()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.clear_draft()
        console._console_setup_blocked_reason = lambda: "blocked"
        notify = Mock()
        app.notify = notify
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT, "staged body"
        )

        await console._consume_pending_console_prompt_insert()
        await pilot.pause()

        assert composer.draft_text() == ""
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )
        assert notify.called
