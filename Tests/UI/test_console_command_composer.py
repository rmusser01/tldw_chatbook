"""Composer command interception + unknown-command Enter-again (Task 10);
`/prompt` resolution + insertion + Library-insert consumption (Task 12)."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from textual.color import Color
from textual.events import Key
from textual.widgets import Button, Input, Static

from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _build_console_send_test_app,
    _configure_native_ready_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_command_grammar import default_console_registry
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService as ScopeLocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.Prompt_Management.prompt_variables import (
    PromptVariableApplication,
    fingerprint_system_text,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Console_Modules.prompts import ConsolePromptsController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleCommandPopup, ConsoleComposerBar
from tldw_chatbook.Widgets.Console import console_composer_bar as composer_module
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash
from tldw_chatbook.Widgets.Console.console_prompt_picker_modal import (
    FILTER_INPUT_ID,
    ROW_ID_PREFIX,
    SEARCH_DEBOUNCE_SECONDS,
)
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    ACTION_UNDO_PROMPT_IMPROVEMENT,
)
from tldw_chatbook.Widgets.Console.prompt_variables_dialog import (
    APPLY_BUTTON_ID as VARIABLES_APPLY_BUTTON_ID,
    CANCEL_BUTTON_ID as VARIABLES_CANCEL_BUTTON_ID,
    ORIGINAL_BUTTON_ID as VARIABLES_ORIGINAL_BUTTON_ID,
    SYSTEM_CHECKBOX_ID,
    VARIABLE_INPUT_CLASS,
    PromptVariablesDialog,
)


def _real_prompt_scope_service(tmp_path):
    """Build a real ``PromptsDatabase`` + ``PromptScopeService`` (mirrors
    ``Tests/UI/test_library_prompts_canvas.py``'s helper of the same name)."""
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    service = PromptScopeService(
        local_service=ScopeLocalPromptService(db), server_service=None
    )
    return db, service


async def _wait_for_picker_search(pilot) -> None:
    """Advance past the picker's debounce timer and let its search settle."""
    await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def _unknown_command_hint(name: str) -> str:
    available = ", ".join(
        f"/{command_name}"
        for command_name in default_console_registry().available_names()
    )
    return (
        f"Unknown command /{name} — available: {available}. "
        "Press Enter again to send as text."
    )


UNKNOWN_NOPE_HINT = _unknown_command_hint("nope")
UNKNOWN_NADA_HINT = _unknown_command_hint("nada")


def _library_prompt_application(
    console,
    user_text: str | None,
    *,
    system_text: str | None = None,
    apply_system: bool = False,
    target_session_id: str | None = None,
    system_fingerprint: str | None = None,
    created_monotonic: float | None = None,
) -> PromptVariableApplication:
    store = console._ensure_console_chat_store()
    session_id = target_session_id or store.active_session_id
    assert session_id is not None
    settings = store.session_settings(store.active_session_id)
    current_system = "" if settings is None else str(settings.system_prompt or "")
    return PromptVariableApplication(
        system_text=system_text if apply_system else None,
        user_text=user_text,
        apply_system=apply_system,
        apply_user=user_text is not None,
        destination="append_active",
        target_session_id=session_id,
        composer_fingerprint=None,
        system_fingerprint=(
            system_fingerprint
            if apply_system and system_fingerprint is not None
            else (fingerprint_system_text(current_system) if apply_system else None)
        ),
        created_monotonic=(
            time.monotonic() if created_monotonic is None else created_monotonic
        ),
    )


class _StagedPromptConsoleHarness(ConsolidatedCSSApp):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        console = ChatScreen(self.app_instance)
        console._session._ensure_active_console_session_settings()
        self.app_instance.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(console, "staged on mount"),
        )
        await self.push_screen(console)


class _RawCliComposerHarness(ConsolidatedCSSApp):
    CSS_PATH = [str(BUNDLED_STYLESHEET)]

    def compose(self):
        yield ConsoleComposerBar(id="console-native-composer")


def _type_raw_cli_prefix(composer: ConsoleComposerBar) -> None:
    assert composer.handle_console_key(Key("exclamation_mark", "!")) is True
    assert composer.handle_console_key(Key("space", " ")) is True


def _raw_draft(kind: str, text: str):
    return composer_module.ConsoleRawDraft(kind, text)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "trusted", "expected"),
    [
        ("", False, ("chat", "")),
        ("hello", False, ("chat", "hello")),
        ("! ", True, ("raw", "")),
        ("! pwd", True, ("raw", "pwd")),
        ("! pwd", False, ("chat", "! pwd")),
        (r"\! pwd", True, ("escaped_chat", "! pwd")),
    ],
)
def test_raw_cli_classifier_is_pure_and_escape_takes_precedence(
    text: str,
    trusted: bool,
    expected: tuple[str, str],
) -> None:
    stash = ConsoleDraftStash(
        segments=[],
        text=text,
        has_paste=False,
        raw_cli_prefix_typed=trusted,
    )

    assert composer_module.classify_console_raw_draft(stash) == _raw_draft(*expected)


@pytest.mark.unit
def test_raw_cli_physical_prefix_at_offsets_zero_and_one_creates_trusted_stash() -> (
    None
):
    composer = ConsoleComposerBar()

    _type_raw_cli_prefix(composer)
    stash = composer.stash_draft_for_send()

    assert stash is not None
    assert stash.raw_cli_prefix_typed is True
    assert composer_module.classify_console_raw_draft(stash) == _raw_draft("raw", "")
    assert composer.draft_text() == ""
    assert composer._raw_cli_prefix_typed is False


@pytest.mark.unit
@pytest.mark.parametrize("modifier", ["alt", "ctrl", "super", "meta", "cmd"])
def test_raw_cli_modified_exclamation_never_advances_prefix_provenance(
    modifier: str,
) -> None:
    composer = ConsoleComposerBar()

    handled = composer.handle_console_key(Key(f"{modifier}+exclamation_mark", "!"))

    assert handled is False
    assert composer.draft_text() == ""
    assert composer._raw_cli_prefix_stage_one is False
    assert composer._raw_cli_prefix_typed is False


@pytest.mark.unit
@pytest.mark.parametrize("modifier", ["alt", "ctrl", "super", "meta", "cmd"])
def test_raw_cli_modified_space_interrupts_prefix_provenance(
    modifier: str,
) -> None:
    composer = ConsoleComposerBar()
    assert composer.handle_console_key(Key("exclamation_mark", "!")) is True
    assert composer._raw_cli_prefix_stage_one is True

    handled = composer.handle_console_key(Key(f"{modifier}+space", " "))

    assert handled is False
    assert composer.draft_text() == "!"
    assert composer._raw_cli_prefix_stage_one is False
    assert composer.handle_console_key(Key("space", " ")) is True
    stash = composer.stash_draft_for_send()
    assert stash is not None
    assert stash.raw_cli_prefix_typed is False
    assert composer_module.classify_console_raw_draft(stash).kind == "chat"


@pytest.mark.unit
@pytest.mark.parametrize("source", ["paste", "programmatic"])
def test_raw_cli_pasted_or_programmatic_prefix_never_creates_trust(source: str) -> None:
    composer = ConsoleComposerBar()
    if source == "paste":
        composer.insert_pasted_text("! pwd")
    else:
        composer.load_draft("! pwd")

    stash = composer.stash_draft_for_send()

    assert stash is not None
    assert stash.raw_cli_prefix_typed is False
    assert composer_module.classify_console_raw_draft(stash) == _raw_draft(
        "chat", "! pwd"
    )


@pytest.mark.unit
def test_raw_cli_typed_prefix_then_pasted_body_remains_trusted_despite_paste() -> None:
    composer = ConsoleComposerBar()
    _type_raw_cli_prefix(composer)

    composer.insert_pasted_text("printf hello")
    stash = composer.stash_draft_for_send()

    assert stash is not None
    assert stash.has_paste is True
    assert stash.raw_cli_prefix_typed is True
    assert composer_module.classify_console_raw_draft(stash) == _raw_draft(
        "raw", "printf hello"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation",
    [
        lambda composer: composer.delete_left(),
        lambda composer: (composer.move_cursor_left(), composer.delete_left()),
        lambda composer: (composer.move_cursor_home(), composer.delete_right()),
        lambda composer: (composer.move_cursor_home(), composer.insert_text("x")),
        lambda composer: (
            composer.select_all_draft(),
            composer.insert_text("! replacement"),
        ),
        lambda composer: composer.load_draft("! replacement"),
    ],
)
def test_raw_cli_edit_delete_move_or_replace_of_prefix_clears_trust(mutation) -> None:
    composer = ConsoleComposerBar()
    _type_raw_cli_prefix(composer)

    mutation(composer)
    stash = composer.stash_draft_for_send()

    assert stash is not None
    assert stash.raw_cli_prefix_typed is False
    assert composer_module.classify_console_raw_draft(stash).kind != "raw"


def _move_raw_cli_cursor(composer: ConsoleComposerBar, offset: int) -> None:
    composer.move_cursor_home()
    for _ in range(offset):
        assert composer.move_cursor_right() is True


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["text", "paste", "file"])
@pytest.mark.parametrize(("offset", "trusted"), [(1, False), (2, True)])
def test_raw_cli_insertions_preserve_trust_only_at_command_body_boundary(
    mutation: str,
    offset: int,
    trusted: bool,
) -> None:
    composer = ConsoleComposerBar()
    _type_raw_cli_prefix(composer)
    _move_raw_cli_cursor(composer, offset)

    if mutation == "text":
        composer.insert_text(" ")
    elif mutation == "paste":
        composer.insert_pasted_text(" ")
    else:
        composer.insert_file_segment(" ", "one-space.txt")
    stash = composer.stash_draft_for_send()

    assert stash is not None
    assert stash.text == "!  "
    assert stash.raw_cli_prefix_typed is trusted
    expected_kind = "raw" if trusted else "chat"
    assert composer_module.classify_console_raw_draft(stash).kind == expected_kind


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["left", "right", "word_left"])
@pytest.mark.parametrize(("delete_body", "trusted"), [(False, False), (True, True)])
def test_raw_cli_deletions_preserve_trust_only_when_deleted_range_starts_at_two(
    mutation: str,
    delete_body: bool,
    trusted: bool,
) -> None:
    composer = ConsoleComposerBar()
    _type_raw_cli_prefix(composer)
    composer.insert_text("x")
    if mutation == "left":
        _move_raw_cli_cursor(composer, 3 if delete_body else 2)
        composer.delete_left()
    elif mutation == "right":
        _move_raw_cli_cursor(composer, 2 if delete_body else 1)
        composer.delete_right()
    else:
        _move_raw_cli_cursor(composer, 3 if delete_body else 2)
        composer.delete_word_left()
    stash = composer.stash_draft_for_send()

    assert stash is not None
    assert stash.raw_cli_prefix_typed is trusted
    expected_kind = "raw" if trusted else "chat"
    assert composer_module.classify_console_raw_draft(stash).kind == expected_kind


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["left", "right", "word_left"])
def test_raw_cli_noop_deletions_do_not_clear_trust(mutation: str) -> None:
    composer = ConsoleComposerBar()
    _type_raw_cli_prefix(composer)
    composer.insert_text("x")
    if mutation in {"left", "word_left"}:
        composer.move_cursor_home()

    if mutation == "left":
        composer.delete_left()
    elif mutation == "right":
        composer.delete_right()
    else:
        assert composer.delete_word_left() is False
    stash = composer.stash_draft_for_send()

    assert stash is not None
    assert stash.text == "! x"
    assert stash.raw_cli_prefix_typed is True
    assert composer_module.classify_console_raw_draft(stash).kind == "raw"


@pytest.mark.unit
def test_raw_cli_refused_stash_restore_preserves_exact_trusted_payload() -> None:
    composer = ConsoleComposerBar()
    _type_raw_cli_prefix(composer)
    composer.insert_pasted_text("pwd")
    refused = composer.stash_draft_for_send()
    assert refused is not None

    composer.insert_text(" typed after refusal")
    composer.restore_stashed_draft(refused)
    restored = composer.stash_draft_for_send()

    assert restored is not None
    assert restored.raw_cli_prefix_typed is True
    assert composer_module.classify_console_raw_draft(restored) == _raw_draft(
        "raw", "pwd typed after refusal"
    )


@pytest.mark.unit
def test_raw_cli_undo_redo_and_history_shaped_replacements_never_mint_trust() -> None:
    composer = ConsoleComposerBar()
    composer.insert_text("! ")
    assert composer._raw_cli_prefix_typed is False

    composer.clear_draft(record_history=True)
    assert composer.undo() is True
    assert composer.draft_text() == "! "
    assert composer._raw_cli_prefix_typed is False
    assert composer.redo() is True
    assert composer._raw_cli_prefix_typed is False

    trusted = ConsoleComposerBar()
    _type_raw_cli_prefix(trusted)
    trusted.insert_text("pwd")
    assert trusted.undo() is True
    assert trusted._raw_cli_prefix_typed is False
    assert trusted.redo() is True
    assert trusted.draft_text().startswith("! ")
    assert trusted._raw_cli_prefix_typed is False


@pytest.mark.asyncio
async def test_raw_cli_visible_send_consumes_same_trusted_stash_as_enter() -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        _type_raw_cli_prefix(composer)
        composer.insert_pasted_text("pwd")
        dispatch = AsyncMock(return_value=True)
        console._dispatch_console_draft_send = dispatch

        console.query_one("#console-send-message", Button).press()
        await pilot.pause()

        stash = dispatch.await_args.kwargs["stash"]
        assert stash is not None
        assert stash.raw_cli_prefix_typed is True
        assert composer_module.classify_console_raw_draft(stash) == _raw_draft(
            "raw", "pwd"
        )
        assert composer.draft_text() == ""
        assert composer._raw_cli_prefix_typed is False


@pytest.mark.asyncio
async def test_raw_cli_trusted_state_is_text_labeled_danger_without_geometry_shift() -> (
    None
):
    host = _RawCliComposerHarness()

    async with host.run_test(size=(120, 20)) as pilot:
        composer = host.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        original_region = composer.region
        original_focus = host.focused

        _type_raw_cli_prefix(composer)
        await pilot.pause()

        status = composer.query_one("#console-raw-cli-status", Static)
        assert str(status.renderable) == "RAW CLI · HOST ACCESS"
        assert composer.has_class("console-raw-cli-danger")
        assert status.has_class("console-raw-cli-danger")
        assert status.has_class("console-voice-status-error")
        assert status.region.height == 1
        assert composer.region.width == original_region.width
        assert composer.region.height == original_region.height
        assert host.focused is original_focus


@pytest.mark.asyncio
async def test_raw_cli_partial_prefix_is_invalidated_by_focus_detour() -> None:
    host = _RawCliComposerHarness()

    async with host.run_test(size=(120, 20)) as pilot:
        composer = host.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        assert composer.handle_console_key(Key("exclamation_mark", "!")) is True
        assert composer._raw_cli_prefix_stage_one is True

        send = composer.query_one("#console-send-message", Button)
        send.focus()
        await pilot.pause()
        assert host.focused is send
        composer.focus()
        await pilot.pause()
        assert host.focused is composer

        assert composer.handle_console_key(Key("space", " ")) is True
        stash = composer.stash_draft_for_send()
        assert stash is not None
        assert stash.text == "! "
        assert stash.raw_cli_prefix_typed is False
        assert composer_module.classify_console_raw_draft(stash).kind == "chat"


@pytest.mark.asyncio
async def test_raw_cli_completed_prefix_survives_focus_detour() -> None:
    host = _RawCliComposerHarness()

    async with host.run_test(size=(120, 20)) as pilot:
        composer = host.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        _type_raw_cli_prefix(composer)

        send = composer.query_one("#console-send-message", Button)
        send.focus()
        await pilot.pause()
        assert host.focused is send
        composer.focus()
        await pilot.pause()
        assert host.focused is composer

        stash = composer.stash_draft_for_send()
        assert stash is not None
        assert stash.raw_cli_prefix_typed is True
        assert composer_module.classify_console_raw_draft(stash).kind == "raw"


@pytest.mark.asyncio
async def test_raw_cli_collapsed_state_retains_danger_label_and_one_row_geometry() -> (
    None
):
    host = _RawCliComposerHarness()

    async with host.run_test(size=(120, 20)) as pilot:
        composer = host.query_one("#console-native-composer", ConsoleComposerBar)
        composer.set_collapsed(True)
        await pilot.pause()

        collapsed = composer.query_one("#console-composer-collapsed")
        status = composer.query_one("#console-composer-collapsed-status", Static)
        ordinary_presentation = (
            status.styles.color,
            status.styles.background,
            status.styles.text_style.bold,
        )

        composer.set_collapsed(False)
        composer.focus()
        await pilot.pause()
        _type_raw_cli_prefix(composer)
        await pilot.pause()
        semantic_error_color = composer.query_one(
            "#console-raw-cli-status", Static
        ).styles.color
        # Production $ds-status-error-readable resolves to this AA-safe foreground.
        readable_error_color = Color.parse("#ff8fa3")
        assert semantic_error_color == readable_error_color
        assert semantic_error_color != ordinary_presentation[0]

        composer.set_collapsed(True)
        await pilot.pause()

        expand = composer.query_one("#console-composer-expand", Button)
        expand.focus()
        await pilot.pause()
        retained_focus = host.focused
        active_widths = (composer.region.width, collapsed.region.width)

        assert str(status.renderable) == "RAW CLI · HOST ACCESS"
        assert collapsed.has_class("console-raw-cli-danger")
        assert status.has_class("console-raw-cli-danger")
        assert status.has_class("console-voice-status-error")
        assert status.styles.color == semantic_error_color
        assert status.styles.color == readable_error_color
        assert status.styles.color != ordinary_presentation[0]
        assert status.styles.background == ordinary_presentation[1]
        assert status.styles.text_style.bold
        assert composer.region.height == 1
        assert collapsed.region.height == 1
        assert status.region.height == 1
        assert host.focused is expand

        composer.load_draft("ordinary")
        await pilot.pause()

        assert str(status.renderable) == "Composer hidden · Draft retained"
        assert not collapsed.has_class("console-raw-cli-danger")
        assert not status.has_class("console-raw-cli-danger")
        assert not status.has_class("console-voice-status-error")
        assert (
            status.styles.color,
            status.styles.background,
            status.styles.text_style.bold,
        ) == ordinary_presentation
        assert composer.region.height == 1
        assert collapsed.region.height == 1
        assert status.region.height == 1
        assert (composer.region.width, collapsed.region.width) == active_widths
        assert host.focused is retained_focus


def _recipe_record() -> dict[str, object]:
    return {
        "id": "local:prompt:77",
        "local_id": 77,
        "name": "Outcome first",
        "artifact_type": "recipe",
        "system_prompt": "Never apply this directly.",
        "user_prompt": "Never insert this directly.",
    }


def _system_message_contents(console) -> list[str]:
    store = console._ensure_console_chat_store()
    if store.active_session_id is None:
        return []
    messages = store.messages_for_session(store.active_session_id)
    return [
        message.content
        for message in messages
        if message.role is ConsoleMessageRole.SYSTEM
    ]


async def _spy_submit_draft(console) -> AsyncMock:
    """Wrap the active controller's ``submit_draft`` so real sends still work."""
    controller = console._ensure_console_chat_controller()
    spy = AsyncMock(wraps=controller.submit_draft)
    controller.submit_draft = spy
    return spy


@pytest.mark.asyncio
async def test_console_prompt_name_resolution_refuses_recipe_candidates() -> None:
    # `_resolve_console_prompt_by_name` moved to `ConsolePromptsController`
    # (wave-3 console decomposition, task 3); still exercised unbound against
    # a hand-built `self`, now the controller's rather than the screen's.
    controller = SimpleNamespace(
        _console_prompt_search=AsyncMock(return_value=[_recipe_record()]),
        _is_recipe_prompt_record=ConsolePromptsController._is_recipe_prompt_record,
    )

    resolved = await ConsolePromptsController._resolve_console_prompt_by_name(
        controller, "Outcome first"
    )

    assert resolved is None


@pytest.mark.asyncio
async def test_prompt_command_rejects_recipe_before_composer_mutation() -> None:
    target = object()
    controller = SimpleNamespace(
        _capture_prompt_replace_target=Mock(return_value=target),
        _resolve_console_prompt_by_name=AsyncMock(return_value=_recipe_record()),
        _launch_prompt_application=Mock(),
        _open_console_prompt_picker_for_insert=AsyncMock(),
        _append_native_console_system_message=AsyncMock(),
        _is_recipe_prompt_record=ConsolePromptsController._is_recipe_prompt_record,
        _RECIPE_EXECUTION_BLOCKED_COPY=(
            ConsolePromptsController._RECIPE_EXECUTION_BLOCKED_COPY
        ),
    )

    await ConsolePromptsController._console_command_insert_prompt(
        controller, SimpleNamespace(args="Outcome first")
    )

    controller._launch_prompt_application.assert_not_called()
    controller._open_console_prompt_picker_for_insert.assert_not_awaited()
    controller._append_native_console_system_message.assert_awaited_once()
    assert (
        "recipe"
        in controller._append_native_console_system_message.await_args.args[0].lower()
    )


@pytest.mark.asyncio
async def test_system_command_rejects_recipe_before_session_or_draft_mutation() -> None:
    # The `self._session._apply_console_session_system_prompt(...)` reach in
    # the pre-move body is now the controller's own same-named session-seam
    # property (wave-3 console decomposition, task 3), so the fake `self`
    # carries it directly instead of behind a `_session` namespace.
    controller = SimpleNamespace(
        _resolve_console_prompt_by_name=AsyncMock(return_value=_recipe_record()),
        _open_console_system_prompt_editor=AsyncMock(),
        _open_console_prompt_picker_for_apply_system=AsyncMock(),
        _append_native_console_system_message=AsyncMock(),
        _apply_console_session_system_prompt=Mock(),
        _clear_console_composer_draft=Mock(),
        _is_recipe_prompt_record=ConsolePromptsController._is_recipe_prompt_record,
        _RECIPE_EXECUTION_BLOCKED_COPY=(
            ConsolePromptsController._RECIPE_EXECUTION_BLOCKED_COPY
        ),
    )

    await ConsolePromptsController._console_command_apply_system(
        controller, SimpleNamespace(args="Outcome first")
    )

    controller._apply_console_session_system_prompt.assert_not_called()
    controller._clear_console_composer_draft.assert_not_called()
    controller._open_console_prompt_picker_for_apply_system.assert_not_awaited()
    controller._append_native_console_system_message.assert_awaited_once()
    assert (
        "recipe"
        in controller._append_native_console_system_message.await_args.args[0].lower()
    )


@pytest.mark.asyncio
async def test_console_unknown_command_first_enter_renders_hint_and_does_not_send():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)

        assert composer.draft_text() == "/nope x"
        submit_spy.assert_not_called()
        assert console._console_unknown_send_armed == "/nope x"


@pytest.mark.asyncio
async def test_console_unknown_command_second_unmodified_enter_sends_as_text():
    gateway = CapturingGateway()
    app = _build_console_send_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        send_button = console.query_one("#console-send-message", Button)

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        submit_spy.assert_not_called()

        send_button.press()
        await _wait_for_text(console, pilot, "accepted")

        submit_spy.assert_awaited_once_with(
            "/nope x",
            session_id=console._ensure_console_chat_store().active_session_id,
        )
        assert gateway.sent_messages[-1][-1]["content"] == "/nope x"
        assert console._console_unknown_send_armed is None


@pytest.mark.asyncio
async def test_console_unknown_command_edit_between_enters_re_hints_and_does_not_send():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        send_button = console.query_one("#console-send-message", Button)

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        assert console._console_unknown_send_armed == "/nope x"

        # Edit the draft to a different unknown command between Enters.
        composer.load_draft("/nada y")
        await pilot.pause()
        assert console._console_unknown_send_armed is None

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NADA_HINT)

        submit_spy.assert_not_called()
        assert composer.draft_text() == "/nada y"
        assert console._console_unknown_send_armed == "/nada y"
        contents = _system_message_contents(console)
        assert contents.count(UNKNOWN_NOPE_HINT) == 1
        assert contents.count(UNKNOWN_NADA_HINT) == 1


@pytest.mark.asyncio
async def test_console_unknown_command_roundtrip_edit_back_to_armed_text_requires_fresh_arm():
    """Editing away and back to the armed text still disarms (Task 10 hardening).

    Comparing the armed snapshot to the current draft text alone would let a
    user edit away from an armed unknown draft and back to the exact same
    text, then have an unrelated second Enter silently send it. The composer
    change subscription must disarm on *any* edit, not just a text mismatch.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        send_button = console.query_one("#console-send-message", Button)

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        assert console._console_unknown_send_armed == "/nope x"

        composer.load_draft("/nope xy")
        composer.load_draft("/nope x")
        await pilot.pause()
        assert console._console_unknown_send_armed is None

        send_button.press()
        await pilot.pause(0.1)

        submit_spy.assert_not_called()
        assert composer.draft_text() == "/nope x"
        assert console._console_unknown_send_armed == "/nope x"


@pytest.mark.asyncio
async def test_console_collapse_disarms_unknown_command_literal_send():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        assert console._console_unknown_send_armed == "/nope x"

        console._set_console_composer_collapsed(True)
        console._set_console_composer_collapsed(False)
        await pilot.pause()
        assert console._console_unknown_send_armed is None

        console.query_one("#console-send-message", Button).press()
        await pilot.pause()
        submit_spy.assert_not_called()
        assert console._console_unknown_send_armed == "/nope x"


@pytest.mark.asyncio
async def test_console_collapsed_paste_starting_with_slash_sends_normally():
    gateway = CapturingGateway()
    app = _build_console_send_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    pasted_text = "/nope " + ("x" * 80)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_pasted_text(pasted_text)
        assert composer.has_paste_segments()
        submit_spy = await _spy_submit_draft(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        submit_spy.assert_awaited_once_with(
            pasted_text,
            session_id=console._ensure_console_chat_store().active_session_id,
        )
        assert gateway.sent_messages[-1][-1]["content"] == pasted_text
        assert console._console_unknown_send_armed is None
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_prompt_command_dispatches_insert_prompt_stub():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt")
        submit_spy = await _spy_submit_draft(console)
        insert_prompt_spy = AsyncMock()
        console._console_command_insert_prompt = insert_prompt_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.1)

        insert_prompt_spy.assert_called_once()
        called_parse = insert_prompt_spy.call_args.args[0]
        assert called_parse.name == "prompt"
        submit_spy.assert_not_called()
        assert composer.draft_text() == "/prompt"


@pytest.mark.asyncio
async def test_console_system_command_dispatches_apply_system_stub():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/system helpful")
        submit_spy = await _spy_submit_draft(console)
        apply_system_spy = AsyncMock()
        console._console_command_apply_system = apply_system_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.1)

        apply_system_spy.assert_called_once()
        called_parse = apply_system_spy.call_args.args[0]
        assert called_parse.name == "system"
        assert called_parse.args == "helpful"
        submit_spy.assert_not_called()
        assert composer.draft_text() == "/system helpful"


# ---------------------------------------------------------------------------
# Task 12: `/prompt` resolution + insertion + Library-insert consumption.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_prompt_command_unique_exact_name_replaces_draft(tmp_path):
    """A unique exact (case-insensitive) name match REPLACES the draft with
    the resolved prompt's ``user_prompt``, via paste semantics (a short body
    inserts inline, unchanged)."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Summarize",
        author="Alice",
        details="",
        system_prompt="",
        user_prompt="Please summarize the following text.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        # Case-insensitive: typed lowercase, stored mixed-case.
        composer.load_draft("/prompt summarize")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert composer.draft_text() == "Please summarize the following text."
        assert len(host.screen_stack) == baseline_depth, (
            "no picker should have opened for a unique match"
        )


@pytest.mark.asyncio
async def test_console_prompt_replacement_resyncs_stale_command_popup(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Summarize",
        author="",
        details="",
        system_prompt="",
        user_prompt="Body.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        popup = console.query_one("#console-command-popup", ConsoleCommandPopup)
        composer.load_draft("/prompt")
        console._sync_console_command_popup()
        assert popup.is_open is True
        composer.load_draft("/prompt Summarize")
        assert popup.is_open is True

        await console._console_command_insert_prompt(SimpleNamespace(args="Summarize"))

        assert composer.draft_text() == "Body."
        assert popup.is_open is False


@pytest.mark.asyncio
async def test_console_prompt_command_captures_before_resolution_and_refuses_stale_draft():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Slow")
        notify = Mock()
        app.notify = notify

        async def resolve_after_edit(_query: str):
            composer.insert_text(" changed")
            return {
                "name": "Slow",
                "artifact_type": "prompt",
                "system_prompt": "",
                "user_prompt": "resolved body",
            }

        console._prompts._resolve_console_prompt_by_name = resolve_after_edit

        await console._console_command_insert_prompt(SimpleNamespace(args="Slow"))
        await pilot.pause()

        assert composer.draft_text() == "/prompt Slow changed"
        notify.assert_called_once()
        assert notify.call_args.kwargs["severity"] == "warning"
        assert "resolved body" not in str(notify.call_args)


@pytest.mark.asyncio
async def test_console_prompt_slash_fallback_threads_dispatch_snapshot_to_picker():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt ambiguous")
        captured = composer.capture_draft_snapshot()

        async def resolve_after_edit(_query: str):
            composer.insert_text(" changed")
            return None

        picker = AsyncMock()
        console._prompts._resolve_console_prompt_by_name = resolve_after_edit
        console._prompts._open_console_prompt_picker_for_insert = picker

        await console._console_command_insert_prompt(SimpleNamespace(args="ambiguous"))

        picker.assert_awaited_once()
        assert picker.await_args.args == ("ambiguous",)
        assert picker.await_args.kwargs["target"].composer_snapshot == captured
        assert composer.draft_text() == "/prompt ambiguous changed"


@pytest.mark.asyncio
async def test_console_prompt_command_uses_shared_dialog_and_shared_value(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Greet",
        author="",
        details="",
        system_prompt="System for {customer}",
        user_prompt="Hello {customer}",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        original_system = store.session_settings(session_id).system_prompt
        composer.load_draft("/prompt Greet")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert isinstance(host.screen_stack[-1], PromptVariablesDialog)
        value_input = host.screen_stack[-1].query_one(f".{VARIABLE_INPUT_CLASS}", Input)
        value_input.value = "Acme"
        await pilot.click(f"#{VARIABLES_APPLY_BUTTON_ID}")
        await pilot.pause()

        assert composer.draft_text() == "Hello Acme"
        assert store.session_settings(session_id).system_prompt == original_system


@pytest.mark.asyncio
async def test_console_prompt_command_use_original_keeps_placeholders(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Literal",
        author="",
        details="",
        system_prompt="",
        user_prompt="Hello {customer}",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Literal")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)
        assert isinstance(host.screen_stack[-1], PromptVariablesDialog)
        await pilot.click(f"#{VARIABLES_ORIGINAL_BUTTON_ID}")
        await pilot.pause()

        assert composer.draft_text() == "Hello {customer}"


@pytest.mark.asyncio
async def test_prompt_replacement_exposes_generic_undo_and_restores_exact_snapshot(
    tmp_path,
):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Undo",
        author="",
        details="",
        system_prompt="",
        user_prompt="replacement",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Undo ")
        composer.insert_pasted_text("pasted")
        composer.insert_file_segment("inline secret", "notes.txt · 13 B")
        composer.select_all_draft()
        before = composer.capture_draft_snapshot()

        await console._console_command_insert_prompt(SimpleNamespace(args="Undo"))
        await pilot.pause()
        assert composer.draft_text() == "replacement"
        assert composer.improvement_undo_available is True

        await console._open_console_composer_menu()
        await pilot.pause()
        undo = host.screen_stack[-1].query_one(
            f"#console-composer-menu-{ACTION_UNDO_PROMPT_IMPROVEMENT}", Button
        )
        assert str(undo.label) == "Undo Prompt change"
        undo.press()
        await pilot.pause()

        restored = composer.capture_draft_snapshot()
        assert restored.segments == before.segments
        assert restored.cursor_index == before.cursor_index
        assert restored.selection == before.selection
        store = console._ensure_console_chat_store()
        assert store.session_draft(store.active_session_id) == composer.draft_text()


@pytest.mark.asyncio
async def test_console_prompt_dialog_applies_shared_value_to_authorized_system(
    tmp_path,
):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Both",
        author="",
        details="",
        system_prompt="System {customer}",
        user_prompt="User {customer}",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Both")

        await console._console_command_insert_prompt(SimpleNamespace(args="Both"))
        await pilot.pause()
        dialog = host.screen_stack[-1]
        assert isinstance(dialog, PromptVariablesDialog)
        dialog.query_one(f".{VARIABLE_INPUT_CLASS}", Input).value = "Acme"
        await pilot.click(f"#{SYSTEM_CHECKBOX_ID}")
        await pilot.pause()
        await pilot.click(f"#{VARIABLES_APPLY_BUTTON_ID}")
        await pilot.pause()

        store = console._ensure_console_chat_store()
        assert composer.draft_text() == "User Acme"
        assert store.session_settings(store.active_session_id).system_prompt == (
            "System Acme"
        )


@pytest.mark.asyncio
async def test_console_prompt_system_only_requires_opt_in_and_clears_snapshot(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="SystemOnly",
        author="",
        details="",
        system_prompt="System only",
        user_prompt="   ",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt SystemOnly")

        await console._console_command_insert_prompt(SimpleNamespace(args="SystemOnly"))
        await pilot.pause()
        dialog = host.screen_stack[-1]
        assert isinstance(dialog, PromptVariablesDialog)
        assert (
            dialog.query_one(f"#{VARIABLES_APPLY_BUTTON_ID}", Button).disabled is True
        )

        await pilot.click(f"#{SYSTEM_CHECKBOX_ID}")
        await pilot.pause()
        assert (
            dialog.query_one(f"#{VARIABLES_APPLY_BUTTON_ID}", Button).disabled is False
        )
        await pilot.click(f"#{VARIABLES_APPLY_BUTTON_ID}")
        await pilot.pause()

        store = console._ensure_console_chat_store()
        assert composer.draft_text() == ""
        assert store.session_settings(store.active_session_id).system_prompt == (
            "System only"
        )


@pytest.mark.asyncio
async def test_console_prompt_dialog_cancel_preserves_exact_snapshot_and_refocuses(
    tmp_path,
):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Cancel",
        author="",
        details="",
        system_prompt="",
        user_prompt="Hello {customer}",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Cancel")
        before = composer.capture_draft_snapshot()

        await console._console_command_insert_prompt(SimpleNamespace(args="Cancel"))
        await pilot.pause()
        assert isinstance(host.screen_stack[-1], PromptVariablesDialog)
        await pilot.click(f"#{VARIABLES_CANCEL_BUTTON_ID}")
        await pilot.pause()

        assert composer.capture_draft_snapshot() == before
        assert composer.has_focus_within is True


@pytest.mark.asyncio
async def test_console_prompt_command_large_user_prompt_collapses_to_paste_token(
    tmp_path,
):
    """An oversized resolved body collapses to a paste token for DISPLAY,
    while the canonical (sent) draft text keeps the full body -- exactly
    like a real paste."""
    large_body = "y" * 200
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Big",
        author="",
        details="",
        system_prompt="",
        user_prompt=large_body,
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Big")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert composer.draft_text() == large_body
        assert "Pasted text |" in composer._display_draft_text()
        assert large_body not in composer._display_draft_text()


@pytest.mark.asyncio
async def test_console_prompt_command_unique_prefix_match_resolves(tmp_path):
    """No exact match, but a unique case-insensitive name PREFIX match,
    still resolves without opening the picker."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Summarize",
        author="",
        details="",
        system_prompt="",
        user_prompt="Summarize body.",
        keywords=[],
    )
    db.add_prompt(
        name="Translate",
        author="",
        details="",
        system_prompt="",
        user_prompt="Translate body.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Summ")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert composer.draft_text() == "Summarize body."
        assert len(host.screen_stack) == baseline_depth


@pytest.mark.asyncio
async def test_console_prompt_command_exact_match_wins_over_ambiguous_prefix(tmp_path):
    """Resolution order: an exact name match resolves immediately even when
    the query is ALSO an ambiguous prefix of another prompt's name."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Sum",
        author="",
        details="",
        system_prompt="",
        user_prompt="Exact body.",
        keywords=[],
    )
    db.add_prompt(
        name="Summarize",
        author="",
        details="",
        system_prompt="",
        user_prompt="Prefix body.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Sum")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert composer.draft_text() == "Exact body."
        assert len(host.screen_stack) == baseline_depth


@pytest.mark.asyncio
async def test_console_prompt_command_ambiguous_exact_match_opens_picker(tmp_path):
    """Two prompts differing only by name CASE (the DB's UNIQUE constraint on
    ``name`` is case-sensitive, so both can exist) means the ci exact-match
    stage itself is ambiguous -- must fall through to the picker rather than
    guessing, with the typed args prefilled into the picker's filter."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Foo",
        author="",
        details="",
        system_prompt="",
        user_prompt="Upper body.",
        keywords=[],
    )
    db.add_prompt(
        name="foo",
        author="",
        details="",
        system_prompt="",
        user_prompt="Lower body.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Foo")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1, (
            "the picker must have opened"
        )
        picker = host.screen_stack[-1]
        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.value == "Foo"
        # The draft is left exactly as typed -- the picker is a detour, not
        # a replacement, until the user actually picks something.
        assert composer.draft_text() == "/prompt Foo"


@pytest.mark.asyncio
async def test_console_prompt_command_no_args_opens_picker_with_empty_query(tmp_path):
    """`/prompt` with no args at all skips resolution entirely and opens the
    picker to browse, rather than attempting a meaningless empty-name match."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Summarize",
        author="",
        details="",
        system_prompt="",
        user_prompt="Body.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1
        picker = host.screen_stack[-1]
        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.value == ""


@pytest.mark.asyncio
async def test_console_prompt_command_picker_uses_real_bound_search_and_selection_replaces_draft(
    tmp_path,
):
    """The picker's ``prompt_search`` is really bound to the scope service
    (fresh reads, not a boot-time snapshot): typing further into the filter
    narrows results from the live DB, and picking a row replaces the draft
    with paste semantics -- same as a directly-resolved `/prompt <name>`."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="",
        details="",
        system_prompt="",
        user_prompt="Summarize body.",
        keywords=[],
    )
    db.add_prompt(
        name="Sundial",
        author="",
        details="",
        system_prompt="",
        user_prompt="Sundial body.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        # Ambiguous prefix: both "Summarize" and "Sundial" start with "Su".
        composer.load_draft("/prompt Su")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)
        assert len(host.screen_stack) == baseline_depth + 1
        picker = host.screen_stack[-1]

        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        filter_input.value = "Summariz"
        await _wait_for_picker_search(pilot)

        row = picker.query_one(f"#{ROW_ID_PREFIX}{prompt_id}", Button)
        row.press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth, (
            "the picker must have dismissed"
        )
        assert composer.draft_text() == "Summarize body."


@pytest.mark.asyncio
async def test_console_prompt_command_picker_escape_leaves_draft_untouched(tmp_path):
    """Escaping the picker dismisses with ``None`` and never touches the
    composer draft."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Foo",
        author="",
        details="",
        system_prompt="",
        user_prompt="Upper body.",
        keywords=[],
    )
    db.add_prompt(
        name="foo",
        author="",
        details="",
        system_prompt="",
        user_prompt="Lower body.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Foo")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)
        assert len(host.screen_stack) == baseline_depth + 1

        await pilot.press("escape")
        await pilot.pause(0.1)

        assert len(host.screen_stack) == baseline_depth
        assert composer.draft_text() == "/prompt Foo"


# -- Library "Use in Console" consumption (ChatScreen-side gating/insertion) --


@pytest.mark.asyncio
async def test_console_pending_prompt_insert_is_consumed_automatically_on_mount():
    """The staged handoff is consumed by the real ``on_mount`` wiring itself
    (not just the private method called directly) -- proves the Library
    hand-off actually lands without any test-only shortcut."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = _StagedPromptConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        for _ in range(40):
            if composer.draft_text() == "staged on mount":
                break
            await pilot.pause(0.05)

        assert composer.draft_text() == "staged on mount"
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )


@pytest.mark.asyncio
async def test_console_pending_prompt_insert_is_consumed_automatically_on_resume():
    """Same as the ``on_mount`` variant above, but exercises the real
    ``on_screen_resume`` timer path -- the finding this regression guards
    against is specific to resume, where (unlike ``on_mount``) nothing
    schedules an equivalent ``_sync_native_console_chat_ui`` pass ahead of
    the 0.15s consumption timer."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(console, "staged on resume"),
        )
        console.on_screen_resume()

        for _ in range(40):
            if composer.draft_text() == "staged on resume":
                break
            await pilot.pause(0.05)

        assert composer.draft_text() == "staged on resume"
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )


@pytest.mark.asyncio
async def test_console_resume_triggered_prompt_insert_survives_stale_session_switch():
    """Regression for the resume wipe-race: if a session switch races ahead
    of a resume-triggered insert, ``_console_visible_draft_session_id`` can
    be stale relative to the store's active session when the insert's own
    0.15s timer fires. Without the fix, a *later* call to
    ``_sync_console_session_draft`` (as several real call sites make, e.g.
    the periodic transcript poller or any other action routed through
    ``_sync_native_console_chat_ui``) would then unconditionally reload the
    composer from the newly-active session's stale stored draft, silently
    discarding the insert -- with no retry, since the handoff is already
    acknowledged once the insert lands. This must not happen: the insert
    consumption itself has to settle the draft tracker before inserting, so a
    later sync pass is a no-op instead of a clobber."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        store = console._ensure_console_chat_store()
        first_session = store.ensure_session()
        # Let the mount-time sync pass settle the visible-draft tracker onto
        # the first session before simulating a session switch that races
        # ahead of the not-yet-fired resume consumption below.
        await pilot.pause(0.1)
        assert console._console_visible_draft_session_id == first_session.id

        second_session = store.create_session(
            title="Second",
            settings=first_session.settings,
        )
        store.set_session_draft(second_session.id, "stale leftover draft")
        assert store.active_session_id == second_session.id
        # The tracker has NOT caught up with the switch yet -- this is the
        # exact staleness the finding describes.
        assert console._console_visible_draft_session_id == first_session.id

        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(console, "resume-triggered insert"),
        )
        console.on_screen_resume()
        await pilot.pause(0.25)  # past the 0.15s consumption timer

        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )
        assert "resume-triggered insert" in composer.draft_text()

        # Simulate a later, unrelated sync pass -- any of several real call
        # sites (periodic transcript polling, another send/stop cycle, a
        # settings-modal callback) route through this same method. It must
        # not retroactively wipe the insert by reloading the stale draft.
        console._session._sync_console_session_draft()
        assert "resume-triggered insert" in composer.draft_text()
        assert console._console_visible_draft_session_id == second_session.id


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_empty_draft_is_clean_insert():
    """An empty composer draft gets a clean insert -- no separator noise."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        assert composer.draft_text() == ""

        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(console, "inserted body"),
        )
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "inserted body"
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_appends_to_existing_draft():
    """Library's insert-in-console NEVER clobbers an in-progress draft --
    it appends onto it instead."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("abc")

        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(console, "inserted body"),
        )
        await console._consume_pending_console_prompt_insert()

        draft = composer.draft_text()
        assert draft.startswith("abc")
        assert "inserted body" in draft
        assert draft == "abc\ninserted body"
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_large_body_appends_as_collapsed_token():
    """An oversized appended body still collapses to a display token, exactly
    like a real paste onto an existing draft."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    large_body = "z" * 200

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("abc")

        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(console, large_body),
        )
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == f"abc\n{large_body}"
        assert "Pasted text |" in composer._display_draft_text()
        assert large_body not in composer._display_draft_text()
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_blocked_shows_exact_toast():
    """First-run setup blocked (no provider/model configured): the insert
    shows the exact toast copy, leaves the draft completely untouched, and
    still acknowledges the handoff (no stale re-fire on a later mount)."""
    app = _build_test_app()  # deliberately NOT _configure_native_ready_console
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("abc")
        notify_spy = Mock()
        app.notify = notify_spy

        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(console, "inserted body"),
        )
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "abc"
        notify_spy.assert_called_once_with(
            "Finish provider setup to insert prompts.", severity="warning"
        )
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_noop_when_nothing_pending():
    """Nothing pending: no-op, no notify, draft untouched."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("abc")
        notify_spy = Mock()
        app.notify = notify_spy

        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "abc"
        notify_spy.assert_not_called()
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )


@pytest.mark.asyncio
async def test_console_rejects_empty_user_only_append_as_noop():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("settled draft")
        application = _library_prompt_application(console, "")
        app.pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, application)
        notify_spy = Mock()
        app.notify = notify_spy

        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "settled draft"
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )
        notify_spy.assert_called_once_with(
            "This Prompt has no text to append.", severity="warning"
        )


@pytest.mark.asyncio
async def test_console_library_append_targets_draft_at_consumption_and_preserves_attachment():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        application = _library_prompt_application(console, "settled append")
        app.pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, application)

        composer.load_draft("draft changed after Library authorization")
        composer.set_pending_attachment_label("evidence.txt", count=1, total=5)
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == (
            "draft changed after Library authorization\nsettled append"
        )
        assert composer._pending_attachment_label == "evidence.txt"


@pytest.mark.asyncio
async def test_console_library_append_expired_claim_warns_once_and_discards():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep")
        app.notify = Mock()
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(
                console,
                "expired secret",
                created_monotonic=time.monotonic() - 120.0,
            ),
        )

        await console._consume_pending_console_prompt_insert()
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "keep"
        app.notify.assert_called_once_with(
            "This Prompt insertion expired. Open the Prompt and retry.",
            severity="warning",
        )
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None


@pytest.mark.asyncio
async def test_console_library_append_latest_wins_and_wrong_session_is_discarded():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(console, "superseded"),
        )
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(
                console,
                "wrong target",
                target_session_id="wrong-session",
            ),
        )
        app.notify = Mock()

        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == ""
        app.notify.assert_called_once_with(
            "The Console session or System prompt changed. Open the Prompt and retry.",
            severity="warning",
        )
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None


@pytest.mark.asyncio
async def test_console_library_append_stale_system_discards_before_draft_mutation():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep")
        app.notify = Mock()
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(
                console,
                "do not append",
                system_text="new system",
                apply_system=True,
                system_fingerprint=fingerprint_system_text("stale system"),
            ),
        )

        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "keep"
        app.notify.assert_called_once_with(
            "The Console session or System prompt changed. Open the Prompt and retry.",
            severity="warning",
        )


@pytest.mark.asyncio
async def test_console_library_append_missing_composer_releases_for_retry():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        application = _library_prompt_application(console, "retry me")
        app.pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, application)
        real_accessor = console._prompts._composer_accessor
        console._prompts._composer_accessor = lambda: None

        await console._consume_pending_console_prompt_insert()

        assert app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_PROMPT_INSERT)
        console._prompts._composer_accessor = real_accessor
        await console._consume_pending_console_prompt_insert()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        assert composer.draft_text() == "retry me"


@pytest.mark.parametrize("transient", ["sync", "composer"])
@pytest.mark.asyncio
async def test_console_library_append_expiry_during_transient_release_warns_once(
    transient,
):
    now = [129.9]
    app = _build_test_app()
    app.pending_handoffs = PendingHandoffStore(monotonic_clock=lambda: now[0])
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(
                console,
                "expires during retry",
                created_monotonic=10.0,
            ),
        )

        if transient == "sync":

            def expire_during_sync():
                now[0] = 130.0
                raise RuntimeError("private transient")

            console._prompts._sync_console_session_draft_fn = expire_during_sync
        else:

            def expire_before_missing_composer():
                now[0] = 130.0
                return None

            console._prompts._composer_accessor = expire_before_missing_composer

        app.notify = Mock()
        await console._consume_pending_console_prompt_insert()
        await console._consume_pending_console_prompt_insert()

        app.notify.assert_called_once_with(
            "This Prompt insertion expired. Open the Prompt and retry.",
            severity="warning",
        )
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None


@pytest.mark.asyncio
async def test_console_library_append_expiry_during_cancelled_sync_warns_once():
    now = [129.9]
    app = _build_test_app()
    app.pending_handoffs = PendingHandoffStore(monotonic_clock=lambda: now[0])
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(
                console,
                "expires during cancellation",
                created_monotonic=10.0,
            ),
        )

        def cancel_after_expiry():
            now[0] = 130.0
            raise asyncio.CancelledError

        console._prompts._sync_console_session_draft_fn = cancel_after_expiry
        app.notify = Mock()

        with pytest.raises(asyncio.CancelledError):
            await console._consume_pending_console_prompt_insert()

        app.notify.assert_called_once_with(
            "This Prompt insertion expired. Open the Prompt and retry.",
            severity="warning",
        )
        assert app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None


@pytest.mark.asyncio
async def test_console_library_system_only_leaves_draft_and_warns_on_persistence_failure(
    monkeypatch,
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("unchanged")
        store = console._ensure_console_chat_store()
        real_set_system = store.set_session_system_prompt

        def set_system_but_report_unsaved(session_id, system_text):
            session, _persisted = real_set_system(session_id, system_text)
            return session, False

        monkeypatch.setattr(
            store, "set_session_system_prompt", set_system_but_report_unsaved
        )
        app.notify = Mock()
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(
                console,
                None,
                system_text="authorized system",
                apply_system=True,
            ),
        )

        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "unchanged"
        assert store.session_settings(store.active_session_id).system_prompt == (
            "authorized system"
        )
        app.notify.assert_called_once_with(
            "System prompt applied for this session, but the change could not be "
            "saved -- it may not survive a reload.",
            severity="warning",
        )


@pytest.mark.asyncio
async def test_console_library_append_rolls_back_draft_when_system_mutation_raises(
    monkeypatch,
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("undo origin")
        composer.insert_file_segment("file body", "file.txt")
        prompt_undo = composer.capture_draft_snapshot()
        composer.replace_snapshot_as_paste(prompt_undo, "active draft")

        history_source = ConsoleComposerBar()
        history_source.insert_text("first")
        history_source.insert_text_as_paste(" second")
        assert history_source.undo() is True
        composer.restore_undo_history(history_source.export_undo_history())
        composer.select_all_draft()
        before = composer.capture_draft_snapshot()
        history_before = composer.export_undo_history()
        store = console._ensure_console_chat_store()
        real_set_system = store.set_session_system_prompt

        def fail_system_mutation(session_id, system_text):
            real_set_system(session_id, system_text)
            raise RuntimeError("private failure")

        monkeypatch.setattr(store, "set_session_system_prompt", fail_system_mutation)
        app.notify = Mock()
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(
                console,
                "append",
                system_text="new system",
                apply_system=True,
            ),
        )

        await console._consume_pending_console_prompt_insert()

        after = composer.capture_draft_snapshot()
        assert after.segments == before.segments
        assert after.cursor_index == before.cursor_index
        assert after.selection == before.selection
        assert after.edit_serial == before.edit_serial
        assert composer.export_undo_history() == history_before
        assert composer.improvement_undo_available is True
        assert store.session_draft(store.active_session_id) == composer.draft_text()
        assert store.session_settings(store.active_session_id).system_prompt is None
        app.notify.assert_called_once_with(
            "The Prompt could not be applied. The Console draft was restored.",
            severity="warning",
        )
        assert composer.undo_improvement() is True
        restored = composer.capture_draft_snapshot()
        assert restored.segments == prompt_undo.segments
        assert restored.cursor_index == prompt_undo.cursor_index
        assert restored.selection == prompt_undo.selection


@pytest.mark.asyncio
async def test_console_library_append_rolls_back_when_paste_mutates_then_raises(
    monkeypatch,
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_text("before")
        before = composer.capture_draft_snapshot()
        real_insert = composer.insert_text_as_paste

        def mutate_then_raise(text):
            real_insert(text)
            raise RuntimeError("private failure")

        monkeypatch.setattr(composer, "insert_text_as_paste", mutate_then_raise)
        app.notify = Mock()
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(console, "append"),
        )

        await console._consume_pending_console_prompt_insert()

        after = composer.capture_draft_snapshot()
        assert after.segments == before.segments
        assert after.cursor_index == before.cursor_index
        store = console._ensure_console_chat_store()
        assert store.session_draft(store.active_session_id) == "before"


@pytest.mark.asyncio
async def test_console_library_system_only_replaces_stale_prompt_undo(
    monkeypatch,
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        captured = composer.capture_draft_snapshot()
        composer.replace_snapshot_as_paste(captured, "prior Prompt change")
        assert composer.improvement_undo_available is True

        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(
                console,
                None,
                system_text="authorized system",
                apply_system=True,
            ),
        )
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "prior Prompt change"
        assert composer.improvement_undo_available is False

        second_capture = composer.capture_draft_snapshot()
        composer.replace_snapshot_as_paste(second_capture, "newer Prompt change")
        assert composer.improvement_undo_available is True

        def fail_system_mutation(_session_id, _system_text):
            raise RuntimeError("private failure")

        monkeypatch.setattr(
            console._ensure_console_chat_store(),
            "set_session_system_prompt",
            fail_system_mutation,
        )
        app.pending_handoffs.stage(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            _library_prompt_application(
                console,
                None,
                system_text="failed system",
                apply_system=True,
            ),
        )
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "newer Prompt change"
        assert composer.improvement_undo_available is True


@pytest.mark.asyncio
async def test_console_prefill_command_arms_one_shot_and_confirms():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prefill Sure thing:")

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "Prefill armed for next send")

        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        assert store.session_one_shot_prefill(session_id) == "Sure thing:"
        assert composer.draft_text() == ""  # handled command clears its draft


@pytest.mark.asyncio
async def test_console_prefill_pin_and_clear_round_trip():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = console.query_one("#console-send-message", Button)
        store = console._ensure_console_chat_store()

        composer.load_draft("/prefill pin Voice:")
        send_button.press()
        await _wait_for_text(console, pilot, "Prefill pinned")
        session_id = store.active_session_id
        assert store.session_settings(session_id).pinned_prefill == "Voice:"

        composer.load_draft("/prefill clear")
        send_button.press()
        await _wait_for_text(console, pilot, "Prefill cleared")
        assert store.session_settings(session_id).pinned_prefill is None
        assert store.session_one_shot_prefill(session_id) is None
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_prefill_pin_seeds_settings_on_settings_less_session():
    """PR #729 Qodo finding 3: a session created without settings (e.g. by a
    bare system-message append before any send) must not make `/prefill pin`
    a silent no-op — the handler seeds default settings first."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session = store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id
        )
        store.replace_session_settings(session.id, None)
        assert store.session_settings(session.id) is None

        composer.load_draft("/prefill pin Voice:")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "Prefill pinned")

        settings = store.session_settings(session.id)
        assert settings is not None
        assert settings.pinned_prefill == "Voice:"
