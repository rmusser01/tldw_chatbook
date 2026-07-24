"""Tests for ``ConsoleStylePickerModal`` and its Console palette wiring
(image-gen P2b, Task 4).

Modal-level tests mirror ``Tests/UI/test_console_skill_picker.py``'s
``ModalHarness`` (bare ``App[None]`` subclass + ``push_screen(...,
callback=...)``), since this picker is a deliberate structural copy of
``ConsoleSkillPickerModal`` -- filter ``Input``, synthetic Up/Down
highlight, Enter-selects, Escape-cancels -- minus the injected async
search callable (the searched set is the small, static, in-memory
``BUILTIN_TEMPLATES`` table, so filtering is synchronous with no debounce).

Screen-level tests mirror ``Tests/UI/test_console_system_prompt.py``'s
palette-launch tests (``test_console_command_provider_lists_insert_prompt_
and_edit_system_prompt`` / ``test_action_open_console_prompt_insert_opens_
picker_with_empty_query``) using the same full ``ConsoleHarness`` app, since
the palette action under test (``action_open_console_style_insert``) is a
near-verbatim mirror of ``action_open_console_prompt_insert``.
"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Input

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_command_grammar import (
    KIND_COMMAND,
    default_console_registry,
)
from tldw_chatbook.Chat.console_generate_image import parse_generate_image_args
from tldw_chatbook.Media_Creation.generation_templates import BUILTIN_TEMPLATES
from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_style_picker_modal import (
    EMPTY_STORE_COPY,
    FILTER_INPUT_ID,
    ROW_ID_PREFIX,
    ConsoleStylePickerModal,
    search_style_templates,
)

# ---------------------------------------------------------------------------
# Modal-level: filtering (pure helper).
# ---------------------------------------------------------------------------


def test_search_style_templates_empty_query_returns_all_thirteen():
    results = search_style_templates("")
    assert len(results) == 13
    assert len(BUILTIN_TEMPLATES) == 13


def test_search_style_templates_filters_by_id_prefix():
    results = search_style_templates("style_")
    assert {t.id for t in results} == {"style_anime", "style_watercolor", "style_cyberpunk"}


def test_search_style_templates_filters_by_name_substring():
    results = search_style_templates("water")
    assert [t.id for t in results] == ["style_watercolor"]


def test_search_style_templates_filters_by_category_case_insensitive():
    results = search_style_templates("CHAT")
    assert {t.id for t in results} == {"chat_character_visual", "chat_scene_visual"}


def test_search_style_templates_unknown_query_returns_empty():
    assert search_style_templates("zzz-no-such-style") == []


# ---------------------------------------------------------------------------
# Modal-level: widget behavior (bare App harness).
# ---------------------------------------------------------------------------


class ModalHarness(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.dismissed_with = "not-called"

    def capture(self, value) -> None:
        self.dismissed_with = value


@pytest.mark.asyncio
async def test_modal_renders_all_thirteen_templates_unfiltered():
    app = ModalHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(ConsoleStylePickerModal(), callback=app.capture)
        await pilot.pause()

        rows = app.screen.query(f".{'console-style-picker-row'}")
        assert len(rows) == 13


@pytest.mark.asyncio
async def test_modal_row_shows_name_category_and_id():
    """Uses `chat_scene_visual` deliberately: its category ("Chat") is NOT a
    substring of its name ("Scene Visualization"), unlike `style_anime`
    (name "Anime Style" already contains "Style", the same string as its
    own category) -- that overlap let a prior version of this assertion
    pass even if the category were silently missing from the label."""
    app = ModalHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(ConsoleStylePickerModal(), callback=app.capture)
        await pilot.pause()

        row = app.screen.query_one(f"#{ROW_ID_PREFIX}chat_scene_visual", Button)
        label = row.label.plain
        assert "Scene Visualization" in label
        assert "Chat" in label
        assert "chat_scene_visual" in label


@pytest.mark.asyncio
async def test_modal_typing_filters_rows_by_id_prefix():
    app = ModalHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(ConsoleStylePickerModal(), callback=app.capture)
        await pilot.pause()

        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        filter_input.value = "style_"
        await pilot.pause(0.1)

        rows = app.screen.query(".console-style-picker-row")
        assert len(rows) == 3
        assert app.screen.query_one(f"#{ROW_ID_PREFIX}style_anime", Button)
        assert app.screen.query_one(f"#{ROW_ID_PREFIX}style_watercolor", Button)
        assert app.screen.query_one(f"#{ROW_ID_PREFIX}style_cyberpunk", Button)


@pytest.mark.asyncio
async def test_modal_unmatched_filter_shows_empty_copy():
    app = ModalHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(ConsoleStylePickerModal(), callback=app.capture)
        await pilot.pause()

        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        filter_input.value = "zzz-no-such-style"
        await pilot.pause(0.1)

        from textual.widgets import Static

        empty = app.screen.query_one("#console-style-picker-empty", Static)
        assert str(empty.renderable) == EMPTY_STORE_COPY


@pytest.mark.asyncio
async def test_modal_row_click_dismisses_with_id_and_name():
    app = ModalHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(ConsoleStylePickerModal(), callback=app.capture)
        await pilot.pause()

        await pilot.click(f"#{ROW_ID_PREFIX}style_anime")
        await pilot.pause()

    assert app.dismissed_with == {"id": "style_anime", "name": "Anime Style"}


@pytest.mark.asyncio
async def test_modal_enter_on_highlighted_row_dismisses_with_that_record():
    app = ModalHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(
            ConsoleStylePickerModal(initial_query="style_"), callback=app.capture
        )
        await pilot.pause()

        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    # "style_" narrows to anime/watercolor/cyberpunk in BUILTIN_TEMPLATES'
    # declared order; one Down from row 0 lands on the second: watercolor.
    assert app.dismissed_with == {"id": "style_watercolor", "name": "Watercolor Style"}


@pytest.mark.asyncio
async def test_modal_escape_dismisses_none():
    app = ModalHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(ConsoleStylePickerModal(), callback=app.capture)
        await pilot.pause()
        await pilot.press("escape")

    assert app.dismissed_with is None


@pytest.mark.asyncio
async def test_modal_filter_has_focus_on_open():
    app = ModalHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(ConsoleStylePickerModal(), callback=app.capture)
        await pilot.pause()

        filter_input = app.screen.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.has_focus


def test_modal_css_blocks_pinned_in_source_and_bundle():
    """`ConsoleStylePickerModal`'s ids/classes must be styled in BOTH the
    module source (`_agentic_terminal.tcss`) and the generated bundle
    (`tldw_cli_modular.tcss`) -- proves `build_css.py` was re-run after the
    source edit, mirroring the skill picker's own dual-file CSS-parity test."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    agentic_terminal = (
        repo_root / "tldw_chatbook" / "css" / "components" / "_agentic_terminal.tcss"
    ).read_text(encoding="utf-8")
    bundled_stylesheet = (
        repo_root / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"
    ).read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        for selector in (
            "ConsoleStylePickerModal {",
            "#console-style-picker-modal {",
            f"#{FILTER_INPUT_ID} {{",
            f"#{FILTER_INPUT_ID}:focus {{",
            ".console-style-picker-row {",
            ".console-style-picker-row-highlighted {",
            "#console-style-picker-empty {",
        ):
            assert selector in text, f"missing CSS for {selector!r}"


# ---------------------------------------------------------------------------
# Screen-level: command palette + composer insert.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_command_provider_lists_insert_image_style():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        provider = ConsoleCommandProvider(screen=console, match_style=None)

        hits = [hit async for hit in provider.search("insert image style")]
        matching = [hit for hit in hits if "Insert image style" in str(hit.text)]
        assert matching, "expected an 'Insert image style…' palette hit"
        assert matching[0].command == console.action_open_console_style_insert


@pytest.mark.asyncio
async def test_action_open_console_style_insert_opens_picker():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")

        console.action_open_console_style_insert()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1
        picker = host.screen_stack[-1]
        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.value == ""


@pytest.mark.asyncio
async def test_style_picker_selection_inserts_style_token_into_draft():
    """An empty draft is prefixed with the command word AND the style token
    (Major review fix): the old behavior of inserting a bare `@style_anime `
    with no command word ahead of it was never a valid `/generate-image`
    invocation -- it would ship to the LLM as plain chat text."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        console.action_open_console_style_insert()
        await pilot.pause(0.2)
        assert len(host.screen_stack) == baseline_depth + 1

        await pilot.click(f"#{ROW_ID_PREFIX}style_anime")
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth, "the picker must have dismissed"
        assert composer.draft_text() == "/generate-image @style_anime "


@pytest.mark.asyncio
async def test_style_picker_insert_composes_valid_command_after_command_word():
    """The insert must land the `@style` token right AFTER the command word,
    never before it: `ConsoleCommandRegistry.parse` only recognizes drafts
    that START with `/`, so prepending the token ahead of an already-typed
    `/generate-image ...` draft would silently stop it from parsing as a
    command at all (Major review fix -- this test replaces a prior version
    that pinned exactly that broken composition)."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/generate-image a red dragon")

        console.action_open_console_style_insert()
        await pilot.pause(0.2)

        await pilot.click(f"#{ROW_ID_PREFIX}style_anime")
        await pilot.pause(0.2)

        assert composer.draft_text() == "/generate-image @style_anime a red dragon"


@pytest.mark.asyncio
async def test_style_picker_insert_replaces_existing_leading_style_token():
    """A draft that already carries a leading `@style` token gets that token
    REPLACED, not stacked -- closes the undisclosed last-wins double-style
    edge the reviewer flagged (both tokens surviving into the parsed args
    would silently pick whichever `parse_generate_image_args` sees last)."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/generate-image :swarmui @old a red dragon")

        console.action_open_console_style_insert()
        await pilot.pause(0.2)

        await pilot.click(f"#{ROW_ID_PREFIX}style_anime")
        await pilot.pause(0.2)

        assert (
            composer.draft_text()
            == "/generate-image :swarmui @style_anime a red dragon"
        )


@pytest.mark.asyncio
async def test_style_picker_insert_result_parses_as_generate_image_command():
    """Closes the insert-to-send gap at the actual choke point: the
    post-insert draft must round-trip through `ConsoleCommandRegistry.parse`
    (exactly what dispatch runs on Send) as a `generate-image` command, and
    `parse_generate_image_args` must recover the chosen style AND the
    original prompt text -- not just "look right" as a string."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/generate-image a red dragon")

        console.action_open_console_style_insert()
        await pilot.pause(0.2)

        await pilot.click(f"#{ROW_ID_PREFIX}style_anime")
        await pilot.pause(0.2)

        draft = composer.draft_text()
        registry = default_console_registry()
        parse = registry.parse(draft)
        assert parse.kind == KIND_COMMAND
        assert parse.name == "generate-image"

        args = parse_generate_image_args(parse.args)
        assert args.style == "style_anime"
        assert args.prompt == "a red dragon"


@pytest.mark.asyncio
async def test_style_picker_escape_leaves_draft_untouched():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/generate-image a red dragon")

        console.action_open_console_style_insert()
        await pilot.pause(0.2)
        assert len(host.screen_stack) == baseline_depth + 1

        await pilot.press("escape")
        await pilot.pause(0.1)

        assert len(host.screen_stack) == baseline_depth
        assert composer.draft_text() == "/generate-image a red dragon"
