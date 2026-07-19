"""Tests for the Console `/system` command, system prompt editor modal, and
Model-rail System preview line (Task 14).

Mirrors ``test_console_command_composer.py``'s harness/helpers for `/prompt`
(Task 12) since `/system` resolution shares the same exact-name -> unique-
prefix -> picker contract, just applying a prompt's ``system_prompt`` instead
of inserting its ``user_prompt``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_console_native_chat_flow import (
    _configure_native_ready_console,
    _static_plain_text,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService as ScopeLocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_prompt_picker_modal import (
    FILTER_INPUT_ID,
    ROW_ID_PREFIX,
)
from tldw_chatbook.Widgets.Console.console_system_prompt_modal import (
    APPLY_BUTTON_ID,
    CANCEL_BUTTON_ID,
    CLEAR_BUTTON_ID,
    NAME_INPUT_ID,
    SAVE_LIBRARY_BUTTON_ID,
    SAVE_STATUS_ID,
    TEXT_AREA_ID,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_TERMINAL = REPO_ROOT / "tldw_chatbook" / "css" / "components" / "_agentic_terminal.tcss"
BUNDLED_STYLESHEET = REPO_ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"

NO_SYSTEM_PART_COPY = 'Prompt "NoSystem" has no system part.'
NAME_IN_USE_COPY = "Name already in use — pick another or open the existing prompt."


def _real_prompt_scope_service(tmp_path):
    """Build a real ``PromptsDatabase`` + ``PromptScopeService`` (mirrors
    ``test_console_command_composer.py``'s helper of the same name)."""
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    service = PromptScopeService(local_service=ScopeLocalPromptService(db), server_service=None)
    return db, service


def _rail_system_line_text(console) -> str:
    return _static_plain_text(console.query_one("#console-rail-system-line", Static))


def _rail_system_line_is_dim(console) -> bool:
    return "console-rail-system-line-dim" in console.query_one(
        "#console-rail-system-line", Static
    ).classes


class FakeConsolePersistence:
    """Minimal persistence double recording conversation/system-prompt writes."""

    def __init__(self) -> None:
        self.updated_system_prompts: list[dict] = []

    def create_conversation(self, **_kwargs) -> str:
        return "conv-1"

    def create_message(self, **_kwargs) -> str:
        return "msg-1"

    def update_message_content(self, **_kwargs) -> bool:
        return True

    def update_conversation_system_prompt(self, *, conversation_id, system_prompt) -> bool:
        self.updated_system_prompts.append(
            {"conversation_id": conversation_id, "system_prompt": system_prompt}
        )
        return True


# ---------------------------------------------------------------------------
# Rail default state.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_rail_system_line_defaults_to_dim_none_state():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-system-line")

        assert _rail_system_line_text(console) == "System: none"
        assert _rail_system_line_is_dim(console)


@pytest.mark.asyncio
async def test_console_rail_system_line_click_opens_editor_modal():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-rail-system-line")

        await pilot.click("#console-rail-system-line")
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1
        modal = host.screen_stack[-1]
        assert modal.query_one(f"#{TEXT_AREA_ID}", TextArea).text == ""


# ---------------------------------------------------------------------------
# Bare `/system` opens the editor modal.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_system_bare_command_opens_editor_with_current_text():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._apply_console_session_system_prompt("Answer tersely.")
        await pilot.pause(0.1)

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/system")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1
        modal = host.screen_stack[-1]
        assert modal.query_one(f"#{TEXT_AREA_ID}", TextArea).text == "Answer tersely."
        assert composer.draft_text() == "/system", "bare /system must not touch the draft"


# ---------------------------------------------------------------------------
# Apply / Clear via the modal.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_system_prompt_modal_apply_updates_settings_and_rail_preview():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        assert _rail_system_line_text(console) == "System: none"

        console.run_worker(console._open_console_system_prompt_editor(), exclusive=False)
        await pilot.pause(0.2)
        modal = host.screen_stack[-1]
        modal.query_one(f"#{TEXT_AREA_ID}", TextArea).text = "Be concise."
        modal.query_one(f"#{APPLY_BUTTON_ID}", Button).press()
        await pilot.pause(0.2)

        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt == "Be concise."
        assert _rail_system_line_text(console) == "System: Be concise."
        assert not _rail_system_line_is_dim(console)


@pytest.mark.asyncio
async def test_console_system_prompt_modal_clear_resets_settings_and_rail_to_none():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._apply_console_session_system_prompt("Be terse.")
        await pilot.pause(0.1)
        assert _rail_system_line_text(console) == "System: Be terse."

        console.run_worker(console._open_console_system_prompt_editor(), exclusive=False)
        await pilot.pause(0.2)
        modal = host.screen_stack[-1]
        assert modal.query_one(f"#{TEXT_AREA_ID}", TextArea).text == "Be terse."
        modal.query_one(f"#{CLEAR_BUTTON_ID}", Button).press()
        await pilot.pause(0.2)

        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt is None
        assert _rail_system_line_text(console) == "System: none"
        assert _rail_system_line_is_dim(console)


@pytest.mark.asyncio
async def test_console_system_prompt_modal_cancel_leaves_settings_untouched():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._apply_console_session_system_prompt("Keep me.")
        await pilot.pause(0.1)

        console.run_worker(console._open_console_system_prompt_editor(), exclusive=False)
        await pilot.pause(0.2)
        modal = host.screen_stack[-1]
        modal.query_one(f"#{TEXT_AREA_ID}", TextArea).text = "Discard this."
        modal.query_one(f"#{CANCEL_BUTTON_ID}", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth
        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt == "Keep me."


@pytest.mark.asyncio
async def test_apply_console_session_system_prompt_preserves_formatting_verbatim():
    """Finding 4: applying a system prompt with leading whitespace and
    internal blank-line formatting must store it verbatim -- `strip()` is
    only used to decide whether the prompt is blank, never to reshape a
    non-blank prompt."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    formatted_prompt = "  line1\n\n  line2  "

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        console._apply_console_session_system_prompt(formatted_prompt)
        await pilot.pause(0.1)

        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt == formatted_prompt


# ---------------------------------------------------------------------------
# `/system <name>` resolution.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_system_command_unique_name_applies_and_updates_rail_preview(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Terse",
        author="",
        details="",
        system_prompt="Answer tersely.",
        user_prompt="",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/system Terse")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth, "a unique match must not open the picker"
        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt == "Answer tersely."
        assert _rail_system_line_text(console) == "System: Answer tersely."
        # Finding 2 (final review): a successful named `/system` apply must
        # clear its own invocation text from the composer -- mirrors
        # `/prompt`'s successful-insert draft-replace behavior -- instead of
        # leaving "/system Terse" sitting in the composer after Enter.
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_system_command_unique_prefix_match_resolves(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Terse", author="", details="", system_prompt="Be terse.", user_prompt="", keywords=[]
    )
    db.add_prompt(
        name="Formal", author="", details="", system_prompt="Be formal.", user_prompt="", keywords=[]
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/system Ter")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth
        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt == "Be terse."


@pytest.mark.asyncio
async def test_console_system_command_empty_system_part_shows_exact_inline_error_and_leaves_settings_unchanged(
    tmp_path,
):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="NoSystem",
        author="",
        details="",
        system_prompt="",
        user_prompt="Body only.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/system NoSystem")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, NO_SYSTEM_PART_COPY)

        assert len(host.screen_stack) == baseline_depth, "no picker/modal should open on this error"
        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt is None
        assert _rail_system_line_text(console) == "System: none"
        # Finding 2 (final review): the empty-system-part ERROR path must
        # NOT clear the draft -- the session is unchanged, and leaving the
        # text lets the user correct it (only the direct-named SUCCESS path
        # clears).
        assert composer.draft_text() == "/system NoSystem"


@pytest.mark.asyncio
async def test_console_system_command_ambiguous_exact_match_opens_apply_system_picker(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Foo", author="", details="", system_prompt="Sys A.", user_prompt="", keywords=[])
    db.add_prompt(name="foo", author="", details="", system_prompt="Sys B.", user_prompt="", keywords=[])
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/system Foo")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1, "the picker must have opened"
        picker = host.screen_stack[-1]
        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.value == "Foo"
        assert composer.draft_text() == "/system Foo"


@pytest.mark.asyncio
async def test_console_system_command_picker_selection_applies_system_prompt(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Sunny", author="", details="", system_prompt="Sunny system.", user_prompt="", keywords=[]
    )
    db.add_prompt(
        name="Sundial", author="", details="", system_prompt="", user_prompt="", keywords=[]
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        # Ambiguous prefix: both "Sunny" and "Sundial" start with "Su".
        composer.load_draft("/system Su")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)
        picker = host.screen_stack[-1]
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        row = picker.query_one(f"#{ROW_ID_PREFIX}{prompt_id}", Button)
        row.press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth, "the picker must have dismissed"
        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt == "Sunny system."


@pytest.mark.asyncio
async def test_console_system_command_picker_escape_leaves_settings_untouched(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Foo", author="", details="", system_prompt="Sys A.", user_prompt="", keywords=[])
    db.add_prompt(name="foo", author="", details="", system_prompt="Sys B.", user_prompt="", keywords=[])
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/system Foo")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)
        picker = host.screen_stack[-1]
        await pilot.press("escape")
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth
        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt is None


# ---------------------------------------------------------------------------
# Save to Library.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_system_prompt_modal_save_to_library_creates_new_prompt(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console.run_worker(console._open_console_system_prompt_editor(), exclusive=False)
        await pilot.pause(0.2)
        modal = host.screen_stack[-1]
        modal.query_one(f"#{TEXT_AREA_ID}", TextArea).text = "Be formal."
        modal.query_one(f"#{NAME_INPUT_ID}", Input).value = "Formal Tone"
        modal.query_one(f"#{SAVE_LIBRARY_BUTTON_ID}", Button).press()
        await pilot.pause(0.1)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        status = modal.query_one(f"#{SAVE_STATUS_ID}", Static)
        assert _static_plain_text(status) == "Saved."

    saved = db.fetch_prompt_details("Formal Tone")
    assert saved is not None
    assert saved["system_prompt"] == "Be formal."


@pytest.mark.asyncio
async def test_console_system_prompt_modal_save_to_library_duplicate_name_shows_task4_copy(
    tmp_path,
):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Existing", author="", details="", system_prompt="Old.", user_prompt="", keywords=[]
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console.run_worker(console._open_console_system_prompt_editor(), exclusive=False)
        await pilot.pause(0.2)
        modal = host.screen_stack[-1]
        modal.query_one(f"#{TEXT_AREA_ID}", TextArea).text = "New system text."
        modal.query_one(f"#{NAME_INPUT_ID}", Input).value = "Existing"
        modal.query_one(f"#{SAVE_LIBRARY_BUTTON_ID}", Button).press()
        await pilot.pause(0.1)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        status = modal.query_one(f"#{SAVE_STATUS_ID}", Static)
        assert _static_plain_text(status) == NAME_IN_USE_COPY


@pytest.mark.asyncio
async def test_console_system_prompt_modal_save_to_library_missing_name_shows_inline_copy():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console.run_worker(console._open_console_system_prompt_editor(), exclusive=False)
        await pilot.pause(0.2)
        modal = host.screen_stack[-1]
        modal.query_one(f"#{TEXT_AREA_ID}", TextArea).text = "Some system text."
        modal.query_one(f"#{SAVE_LIBRARY_BUTTON_ID}", Button).press()
        await pilot.pause(0.1)

        status = modal.query_one(f"#{SAVE_STATUS_ID}", Static)
        assert _static_plain_text(status) == "Enter a name to save this system prompt to Library."


# ---------------------------------------------------------------------------
# Persistence: applying to an already-saved conversation persists it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_system_prompt_apply_persists_when_conversation_already_saved():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        console._ensure_active_console_session_settings()
        session_id = store.active_session_id
        fake_persistence = FakeConsolePersistence()
        store.persistence = fake_persistence
        store.persist_session_if_needed(session_id)

        console._apply_console_session_system_prompt("Persisted prompt.")
        await pilot.pause(0.1)

        assert fake_persistence.updated_system_prompts == [
            {"conversation_id": "conv-1", "system_prompt": "Persisted prompt."}
        ]


@pytest.mark.asyncio
async def test_console_system_prompt_apply_notifies_on_persistence_failure():
    """Finding 3: a persistence failure while applying a system prompt must
    not crash the apply flow. The in-memory session keeps the applied
    value (this store's existing convention -- mutations are not rolled
    back when the durable write that follows them fails) and the user
    gets an honest warning toast rather than silence or a crash.
    """

    class RaisingPersistence(FakeConsolePersistence):
        def update_conversation_system_prompt(self, *, conversation_id, system_prompt):
            raise RuntimeError("conversation vanished")

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        console._ensure_active_console_session_settings()
        session_id = store.active_session_id
        store.persistence = FakeConsolePersistence()
        store.persist_session_if_needed(session_id)
        store.persistence = RaisingPersistence()

        notifications: list[tuple[str, str]] = []
        app.notify = lambda message, severity="information", **kwargs: notifications.append(
            (message, severity)
        )

        console._apply_console_session_system_prompt("New prompt")
        await pilot.pause(0.1)

        settings = console._ensure_active_console_session_settings()
        assert settings.system_prompt == "New prompt"
        assert any(severity == "warning" for _message, severity in notifications)


# ---------------------------------------------------------------------------
# Context estimate counts the applied system prompt.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_context_estimate_counts_system_prompt_after_apply():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        baseline = console._active_console_settings_context_estimate()
        assert baseline.used_tokens is not None

        console._apply_console_session_system_prompt(
            "Answer using formal English at all times, citing sources where possible."
        )
        await pilot.pause(0.1)

        after = console._active_console_settings_context_estimate()
        assert after.used_tokens is not None
        assert after.used_tokens > baseline.used_tokens


# ---------------------------------------------------------------------------
# Command palette entries.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_command_provider_lists_insert_prompt_and_edit_system_prompt():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        provider = ConsoleCommandProvider(screen=console, match_style=None)

        insert_hits = [hit async for hit in provider.search("insert prompt")]
        matching_insert = [hit for hit in insert_hits if "Insert prompt" in str(hit.text)]
        assert matching_insert, "expected an 'Insert prompt…' palette hit"
        assert matching_insert[0].command == console.action_open_console_prompt_insert

        system_hits = [hit async for hit in provider.search("edit system prompt")]
        matching_system = [hit for hit in system_hits if "Edit system prompt" in str(hit.text)]
        assert matching_system, "expected an 'Edit system prompt' palette hit"
        assert matching_system[0].command == console.action_open_console_system_prompt_editor


@pytest.mark.asyncio
async def test_action_open_console_prompt_insert_opens_picker_with_empty_query():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")

        console.action_open_console_prompt_insert()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1
        picker = host.screen_stack[-1]
        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.value == ""


@pytest.mark.asyncio
async def test_action_open_console_system_prompt_editor_opens_modal():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")

        console.action_open_console_system_prompt_editor()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1
        modal = host.screen_stack[-1]
        assert modal.query_one(f"#{TEXT_AREA_ID}", TextArea).text == ""


# ---------------------------------------------------------------------------
# CSS parity/pin discipline.
# ---------------------------------------------------------------------------


def _css_block(text: str, selector: str) -> str:
    """Return a CSS rule body starting at ``selector`` (mirrors the helper of
    the same name in ``test_console_prompt_picker.py``)."""
    start = text.index(selector)
    block_start = text.index("{", start)
    block_end = text.index("}", block_start)
    return text[block_start:block_end]


def test_system_prompt_modal_and_rail_line_css_pinned_in_source_and_bundle():
    """The modal/rail-line ids/classes must be styled in BOTH the module
    source (``_agentic_terminal.tcss``) and the generated bundle
    (``tldw_cli_modular.tcss``) -- proves ``build_css.py`` was re-run after
    the source edit, mirroring ``test_console_prompt_picker.py``'s dual-file
    CSS-parity discipline for this feature branch."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        for selector in (
            "ConsoleSystemPromptModal {",
            "#console-system-prompt-modal {",
            f"#{TEXT_AREA_ID} {{",
            f"#{TEXT_AREA_ID}:focus {{",
            f"#{NAME_INPUT_ID} {{",
            f"#{SAVE_STATUS_ID} {{",
            ".console-system-prompt-actions {",
            "#console-rail-system-line {",
            "#console-rail-system-line.console-rail-system-line-dim {",
        ):
            assert selector in text, f"missing CSS for {selector!r}"

        dim_block = _css_block(text, "#console-rail-system-line.console-rail-system-line-dim {")
        assert "color: $ds-text-muted;" in dim_block
