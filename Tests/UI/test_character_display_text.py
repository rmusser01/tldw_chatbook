"""Display-boundary tests for character-card text."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Select, Static, TextArea
from textual.widgets._select import SelectOverlay

from tldw_chatbook.UI.CCP_Modules import (
    ccp_character_handler as character_handler_module,
)
from tldw_chatbook.UI.CCP_Modules.ccp_character_handler import CCPCharacterHandler
from tldw_chatbook.UI.character_display_text import (
    sanitize_character_display_items,
    sanitize_character_display_label,
    sanitize_character_display_text,
)
from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterOption,
    ConsoleCharacterPickerModal,
    filter_character_options,
)


def test_character_display_sanitizer_replaces_invalid_terminal_sequences() -> None:
    raw = "A\ufffdB\x00C\ud800D\u200bE"

    assert sanitize_character_display_text(raw, max_characters=20) == "A?B?C?D?E"


def test_character_display_sanitizer_preserves_valid_unicode_newline_and_tab() -> None:
    raw = "Cafe \u2615\n\t\u6771\u4eac"

    assert sanitize_character_display_text(raw, max_characters=100) == raw


def test_character_display_sanitizer_bounds_before_projecting() -> None:
    assert sanitize_character_display_text("ab\x00cd", max_characters=3) == "ab?"
    assert sanitize_character_display_text("text", max_characters=0) == ""


@pytest.mark.parametrize("max_characters", [-1, True, 1.5, "4"])
def test_character_display_sanitizer_rejects_invalid_maximum(max_characters: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        sanitize_character_display_text("text", max_characters=max_characters)  # type: ignore[arg-type]


def test_character_display_sanitizer_is_deterministic_for_weird_objects() -> None:
    class BrokenDisplay:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    assert sanitize_character_display_text(42, max_characters=20) == "42"
    assert sanitize_character_display_text(object(), max_characters=20) == "<object>"
    assert (
        sanitize_character_display_text(BrokenDisplay(), max_characters=20)
        == "<BrokenDisplay>"
    )


def test_character_display_sanitizer_never_calls_custom_string_conversion() -> None:
    calls: list[str] = []

    class SideEffectingDisplay:
        def __str__(self) -> str:
            calls.append("str")
            return "x" * 1_000_000

        def __repr__(self) -> str:
            calls.append("repr")
            return "unsafe"

    assert (
        sanitize_character_display_text(
            SideEffectingDisplay(),
            max_characters=100,
        )
        == "<SideEffectingDisplay>"
    )
    assert calls == []


def test_character_display_label_collapses_multiline_whitespace_without_merging_words() -> None:
    raw = "  Captain\n\tRowan\u2003of  the Guard  "

    assert sanitize_character_display_label(raw, max_characters=100) == (
        "Captain Rowan of the Guard"
    )


def test_character_display_items_are_bounded_and_do_not_consume_custom_iterables() -> None:
    calls: list[str] = []

    class DangerousIterable:
        def __iter__(self):
            calls.append("iter")
            yield from ("one", "two")

    assert sanitize_character_display_items(
        "solo",
        max_items=3,
        max_item_characters=20,
        max_total_characters=40,
    ) == ("solo",)
    assert sanitize_character_display_items(
        {"forged": "value"},
        max_items=3,
        max_item_characters=20,
        max_total_characters=40,
    ) == ("<dict>",)
    assert sanitize_character_display_items(
        DangerousIterable(),
        max_items=3,
        max_item_characters=20,
        max_total_characters=40,
    ) == ("<DangerousIterable>",)
    assert calls == []


def test_character_display_sanitizer_does_not_mutate_card() -> None:
    card = {"name": "Name\ufffd", "description": "Original\x00value"}

    shown = sanitize_character_display_text(card["description"], max_characters=200)

    assert shown == "Original?value"
    assert card == {"name": "Name\ufffd", "description": "Original\x00value"}


@pytest.mark.parametrize(
    ("character", "expected_width"),
    (("\x1b", -1), ("\u0301", 0), ("\u754c", 2), ("A", 1)),
)
def test_character_display_sanitizer_matches_public_wcwidth_classes(
    character: str,
    expected_width: int,
) -> None:
    wcwidth_module = importlib.import_module("wcwidth")

    assert wcwidth_module.wcwidth(character) == expected_width
    expected = "?" if expected_width < 0 else character
    assert sanitize_character_display_text(character, max_characters=1) == expected


def test_character_display_sanitizer_calls_public_wcwidth(monkeypatch) -> None:
    wcwidth_module = importlib.import_module("wcwidth")
    calls: list[str] = []

    def negative_width(character: str) -> int:
        calls.append(character)
        return -1

    monkeypatch.setattr(wcwidth_module, "wcwidth", negative_width)

    assert sanitize_character_display_text("A", max_characters=1) == "?"
    assert calls == ["A"]


def test_wcwidth_is_declared_in_both_runtime_dependency_lists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = {
        line.strip()
        for line in (repo_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    requirement = "wcwidth>=0.2.14,<1"
    assert requirement in project["project"]["dependencies"]
    assert requirement in requirements


class _CCPDisplayApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("", id="ccp-card-name-display", markup=False)
        yield TextArea("", id="ccp-card-description-display", read_only=True)


class _CCPSelectApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Select(
            [("Initial", "initial")],
            id="conv-char-character-select",
            allow_blank=False,
        )


class _CCPWindow:
    def __init__(self, app: App[None]) -> None:
        self.app_instance = app
        self.app = app

    def query_one(self, *args, **kwargs):
        return self.app.query_one(*args, **kwargs)


@pytest.mark.asyncio
async def test_ccp_read_only_display_sanitizes_without_changing_source_data() -> None:
    source = {"name": "N\ufffdme", "description": "Line\x00one\nLine two"}
    app = _CCPDisplayApp()
    async with app.run_test() as pilot:
        handler = CCPCharacterHandler(_CCPWindow(app))
        handler.current_character_data = source

        handler._update_field("#ccp-card-name-display", source["name"])
        handler._update_textarea(
            "#ccp-card-description-display", source["description"]
        )
        await pilot.pause()

        assert str(app.query_one("#ccp-card-name-display", Static).renderable) == "N?me"
        assert app.query_one("#ccp-card-description-display", TextArea).text == (
            "Line?one\nLine two"
        )

    assert source == {"name": "N\ufffdme", "description": "Line\x00one\nLine two"}


@pytest.mark.asyncio
async def test_ccp_character_select_renders_saved_names_as_literal_text(
    monkeypatch,
) -> None:
    records = [
        {"id": 7, "name": "[/x]\n\tforged row"},
        {"id": 8, "name": "[bold]Literal[/bold]"},
    ]
    monkeypatch.setattr(
        character_handler_module,
        "fetch_all_characters",
        lambda: records,
    )
    app = _CCPSelectApp()

    async with app.run_test() as pilot:
        handler = CCPCharacterHandler(_CCPWindow(app))
        await handler.refresh_character_list()
        select = app.query_one("#conv-char-character-select", Select)

        select.focus()
        await pilot.press("enter")
        await pilot.pause()

        overlay = select.query_one(SelectOverlay)
        prompts = [option.prompt for option in overlay.options]
        assert select.expanded is True
        assert all(isinstance(prompt, Text) for prompt in prompts)
        assert [prompt.plain for prompt in prompts] == [
            "[/x] forged row",
            "[bold]Literal[/bold]",
        ]
        assert select.value == "7"

    assert records == [
        {"id": 7, "name": "[/x]\n\tforged row"},
        {"id": 8, "name": "[bold]Literal[/bold]"},
    ]
    assert handler.character_list == records


@pytest.mark.asyncio
async def test_console_picker_sanitizes_only_rendered_character_text() -> None:
    malformed_name = "A\ufffdda\n\tAdmin\x00[/red]"
    malformed_description = "semantic\u200b description"
    option = ConsoleCharacterOption(7, malformed_name, malformed_description)
    app = App[None]()

    async with app.run_test() as pilot:
        modal = ConsoleCharacterPickerModal(options=(option,))
        await app.push_screen(modal)
        await pilot.pause()

        row = modal.query_one("#console-character-picker-row-7", Static)
        assert "A?da Admin?[/red]" in str(row.renderable)
        assert "\n" not in str(row.renderable)
        assert "\t" not in str(row.renderable)
        assert malformed_name not in str(row.renderable)

        modal._select(option)
        await pilot.pause()
        hint = modal.query_one("#console-character-picker-hint", Static)
        assert "A?da Admin?[/red]" in str(hint.renderable)

        captured = []
        modal.dismiss = captured.append  # type: ignore[method-assign]
        modal._finish("new")

    assert filter_character_options((option,), "semantic") == (option,)
    assert captured[0].name == malformed_name
    assert option.description == malformed_description


class _CCPFallbackDisplayApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("", id="ccp-card-name-display", markup=False)
        for suffix in (
            "description",
            "personality",
            "scenario",
            "first-message",
            "creator-notes",
            "system-prompt",
            "post-history-instructions",
            "alternate-greetings",
        ):
            yield TextArea("", id=f"ccp-card-{suffix}-display", read_only=True)
        for suffix in ("tags", "creator", "version", "keywords"):
            yield Static("", id=f"ccp-card-{suffix}-display", markup=False)


@pytest.mark.asyncio
async def test_ccp_fallback_display_handles_malformed_collection_shapes_without_mutation() -> None:
    calls: list[str] = []

    class DangerousIterable:
        def __iter__(self):
            calls.append("iter")
            yield "unsafe"

    source = {
        "name": "Nyx",
        "tags": "solo",
        "keywords": 7,
        "alternate_greetings": {"forged": "greeting"},
    }
    original = dict(source)
    source_with_iterable = dict(source, tags=DangerousIterable())
    app = _CCPFallbackDisplayApp()

    async with app.run_test() as pilot:
        handler = CCPCharacterHandler(_CCPWindow(app))
        handler.current_character_data = source
        handler._display_character_card()
        await pilot.pause()

        assert str(app.query_one("#ccp-card-tags-display", Static).renderable) == "solo"
        assert str(app.query_one("#ccp-card-keywords-display", Static).renderable) == "7"
        assert app.query_one(
            "#ccp-card-alternate-greetings-display", TextArea
        ).text == "<dict>"

        handler.current_character_data = source_with_iterable
        handler._display_character_card()
        await pilot.pause()
        assert str(app.query_one("#ccp-card-tags-display", Static).renderable) == (
            "<DangerousIterable>"
        )

    assert source == original
    assert calls == []
