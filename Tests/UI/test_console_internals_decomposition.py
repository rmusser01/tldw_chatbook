import time
from pathlib import Path

import pytest
from rich.text import Text
from textual.events import Paste
from textual.widgets import Button, Footer, Input, Select, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE15_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/"
    "2026-05-07-gate-1-5-console-internals-decomposition.md"
)
ROADMAP = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
TASK_10_6 = Path(
    "backlog/tasks/task-10.6 - "
    "Product-Maturity-Phase-3.6-Gate-1.5-Console-Internals-Decomposition.md"
)


def _repo_text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class StaticConsoleLibraryRagSearchService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def search(self, query, scope, mode, **kwargs):
        self.calls.append(
            {
                "query": query,
                "scope": scope,
                "mode": mode,
                **kwargs,
            }
        )
        return self.result


class _PressedEvent:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


async def _wait_for_console_library_rag_button_state(
    console,
    pilot,
    *,
    disabled: bool,
    tooltip_contains: str = "",
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        buttons = list(console.query("#console-run-library-rag"))
        if buttons:
            button = buttons[0]
            tooltip = str(button.tooltip or "")
            if button.disabled is disabled and tooltip_contains in tooltip:
                await pilot.pause()
                return
        await pilot.pause(0.01)
    raise AssertionError(
        "Timed out waiting for Console Library RAG run button "
        f"disabled={disabled!r} tooltip={tooltip_contains!r}"
    )


def _assert_single_style_span(renderable: Text, *, style: str, expected_text: str) -> None:
    matching_spans = [span for span in renderable.spans if span.style == style]
    assert len(matching_spans) == 1
    span = matching_spans[0]
    assert renderable.plain[span.start : span.end] == expected_text


def test_gate15_console_internals_evidence_is_tracked():
    evidence = _repo_text(GATE15_EVIDENCE)
    roadmap = _repo_text(ROADMAP)
    readme = _repo_text(PHASE_3_README)
    task = _repo_text(TASK_10_6)

    for heading in (
        "## Scope",
        "## Walkthrough",
        "## Functional Result",
        "## Verification",
        "## Defects",
        "## Residual Risk",
        "## Exit Decision",
    ):
        assert heading in evidence
    for selector in (
        "#console-control-bar",
        "#console-session-surface",
        "#console-native-composer",
        "#console-run-inspector",
    ):
        assert selector in evidence
    assert "/Users/macbook-dev/" not in evidence
    assert GATE15_EVIDENCE.name in readme
    assert GATE15_EVIDENCE.name in roadmap
    assert "Gate 1.5" in roadmap
    assert "TASK-10.6" in roadmap
    assert "status: Done" in task
    assert "## Implementation Notes" in task


@pytest.mark.asyncio
async def test_console_gate15_does_not_mount_full_legacy_chat_window_chrome():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        assert console.query_one("#console-control-bar")
        assert console.query_one("#console-session-surface")
        assert console.query_one("#console-native-composer")

        assert len(console.query("#chat-enhanced-sidebar")) == 0
        assert len(console.query("#toggle-chat-left-sidebar")) == 0
        assert len(console.query("#chat-main-content")) == 0


@pytest.mark.asyncio
async def test_console_gate15_keeps_existing_chat_send_control_reachable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        text = _visible_text(console)
        assert "Send" in text
        assert "Stop" in text
        assert "Attach" in text
        assert "Save Chatbook" in text
        send_controls = [
            button
            for button in console.query(Button)
            if (button.id or "").startswith("send-stop-chat")
            or button.has_class("console-send-button")
        ]
        assert send_controls
        assert console.query_one("#console-stop-generation", Button)
        assert console.query_one("#console-attach-context", Button)
        assert console.query_one("#console-save-chatbook", Button)


@pytest.mark.asyncio
async def test_console_native_composer_spans_below_workbench_with_single_input_surface():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        workbench = console.query_one("#console-workspace-grid")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        command_input = composer.query_one("#console-command-input", Input)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        send_button = composer.query_one("#console-send-message", Button)
        stop_button = composer.query_one("#console-stop-generation", Button)
        attach_button = composer.query_one("#console-attach-context", Button)
        save_button = composer.query_one("#console-save-chatbook", Button)
        legacy_inputs = [
            widget
            for widget in console.query(".chat-input-area")
            if widget.region.height > 0 and widget.region.width > 0
        ]

        assert composer.region.x == workbench.region.x
        assert composer.region.width == workbench.region.width
        assert composer.region.y > workbench.region.y
        assert composer.region.y + composer.region.height <= console.size.height
        assert command_input.display is False
        assert visible_draft.region.width > 20
        assert 4 <= composer.region.height <= 6
        assert visible_draft.region.height == 1
        composer.load_draft("visible composer text")
        await pilot.pause(0.1)
        assert composer.draft_text() == "visible composer text"
        assert "visible composer text" in visible_draft.renderable.plain
        assert "visible composer text" in _visible_text(composer)
        for action_button in (send_button, stop_button, attach_button, save_button):
            assert action_button.compact is True
            assert action_button.region.height == 1
            assert action_button.region.y + action_button.region.height <= composer.region.y + composer.region.height
        assert str(send_button.label) == "Send"
        assert str(stop_button.label) == "Stop"
        assert str(attach_button.label) == "Attach"
        assert str(save_button.label) == "Save Chatbook"
        assert save_button.region.width >= 20
        assert legacy_inputs == []


@pytest.mark.asyncio
async def test_console_native_composer_receives_typing_on_open():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        command_input = composer.query_one("#console-command-input", Input)
        visible_draft = composer.query_one("#console-command-visible-text", Static)

        assert command_input.display is False
        await pilot.press("v", "i", "s", "i", "b", "l", "e")
        await pilot.pause(0.1)

        assert composer.draft_text() == "visible"
        assert "visible" in visible_draft.renderable.plain
        assert "visible" in _visible_text(composer)


@pytest.mark.asyncio
async def test_console_native_composer_auto_expands_for_long_drafts():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        long_draft = "long composer qa " * 20

        composer.load_draft(long_draft)
        await pilot.pause(0.2)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == long_draft
        assert composer.region.height > 5
        assert composer.region.height <= 10
        assert visible_draft.region.height > 1
        assert visible_draft.region.height <= 4
        assert "\n" in visible_plain
        assert "Pasted Text:" not in visible_plain
        assert "long composer qa" in visible_plain


@pytest.mark.asyncio
async def test_console_large_paste_collapses_visible_token_but_preserves_payload():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        command_input = composer.query_one("#console-command-input", Input)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "pasted composer qa " * 80
        expected_token = f"Pasted Text: {len(pasted_text)} Characters"

        console.on_paste(Paste(pasted_text))
        await pilot.pause(0.2)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == pasted_text
        assert command_input.value == pasted_text
        assert composer.region.height <= 10
        assert visible_draft.region.height <= 4
        assert expected_token in visible_plain
        assert pasted_text not in visible_plain
        assert len(visible_plain) < len(pasted_text)
        assert isinstance(visible_draft.renderable, Text)
        _assert_single_style_span(
            visible_draft.renderable,
            style=ConsoleComposerBar.PASTE_TOKEN_STYLE,
            expected_text=expected_token,
        )


def test_console_paste_token_style_span_survives_literal_ellipsis_prefix():
    pasted_text = "x" * 51
    expected_token = f"Pasted Text: {len(pasted_text)} Characters"
    display_text = f"... {expected_token}"

    renderable = ConsoleComposerBar._draft_renderable(
        display_text,
        width=200,
        style_ranges=[(4, len(display_text), ConsoleComposerBar.PASTE_TOKEN_STYLE)],
    )

    assert renderable.plain == display_text
    _assert_single_style_span(
        renderable,
        style=ConsoleComposerBar.PASTE_TOKEN_STYLE,
        expected_text=expected_token,
    )


def test_console_paste_token_style_span_survives_crlf_before_token():
    pasted_text = "x" * 51
    expected_token = f"Pasted Text: {len(pasted_text)} Characters"
    display_text = f"before\r\n{expected_token}"
    token_start = display_text.index(expected_token)

    renderable = ConsoleComposerBar._draft_renderable(
        display_text,
        width=200,
        style_ranges=[(token_start, len(display_text), ConsoleComposerBar.PASTE_TOKEN_STYLE)],
    )

    assert renderable.plain == f"before\n{expected_token}"
    _assert_single_style_span(
        renderable,
        style=ConsoleComposerBar.PASTE_TOKEN_STYLE,
        expected_text=expected_token,
    )


def test_console_literal_segments_merge_during_typing_and_small_pastes():
    composer = ConsoleComposerBar()

    composer.insert_text("a")
    composer.insert_text("b")
    composer.insert_pasted_text("small paste")

    assert composer.draft_text() == "absmall paste"
    assert len(composer._segments) == 1
    assert composer._segments[0].collapse_state == "literal"


def test_console_composer_empty_placeholder_is_task_oriented():
    renderable = ConsoleComposerBar._draft_renderable("")

    assert renderable.plain == "Ask, command, or paste task..."


def test_console_provider_blocker_reason_lowercases_without_breaking_acronyms():
    assert ChatScreen._lower_first_char("Missing API key") == "missing API key"
    assert ChatScreen._lower_first_char("API key missing") == "API key missing"


@pytest.mark.asyncio
async def test_console_paste_under_threshold_remains_literal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "x" * ConsoleComposerBar.PASTE_COLLAPSE_THRESHOLD

        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == pasted_text
        assert visible_plain == pasted_text
        assert "Pasted Text:" not in visible_plain


@pytest.mark.parametrize("collapse_setting", [False, "false"])
@pytest.mark.asyncio
async def test_console_large_paste_collapse_can_be_disabled_from_config(collapse_setting):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": collapse_setting}
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "literal paste chunk " * 10
        assert len(pasted_text) > ConsoleComposerBar.PASTE_COLLAPSE_THRESHOLD

        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == pasted_text
        assert "Pasted Text:" not in visible_plain
        assert visible_plain.replace("\n", "") == pasted_text


@pytest.mark.asyncio
async def test_console_clear_draft_keeps_canonical_payload_empty():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "clear me " * 20

        composer.insert_pasted_text(pasted_text)
        composer.clear_draft()
        await pilot.pause(0.1)

        assert composer.draft_text() == ""
        assert visible_draft.renderable.plain == ConsoleComposerBar.DRAFT_PLACEHOLDER
        assert pasted_text not in visible_draft.renderable.plain


@pytest.mark.asyncio
async def test_console_collapsed_paste_backspace_deletes_whole_chunk():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        prefix = "literal prefix "
        pasted_text = "delete after paste " * 10

        composer.insert_text(prefix)
        composer.insert_pasted_text(pasted_text)
        composer.delete_left()
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert composer.draft_text() == prefix
        assert visible_plain == prefix
        assert pasted_text not in composer.draft_text()
        assert "Pasted Text:" not in visible_plain


@pytest.mark.asyncio
async def test_console_collapsed_paste_real_click_enters_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "click to confirm paste " * 10

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.pause(0.1)

        assert visible_draft.renderable.plain == "Unfurl?"
        assert composer.draft_text() == pasted_text
        assert isinstance(visible_draft.renderable, Text)
        _assert_single_style_span(
            visible_draft.renderable,
            style=ConsoleComposerBar.PASTE_CONFIRM_STYLE,
            expected_text="Unfurl?",
        )


@pytest.mark.asyncio
async def test_console_collapsed_paste_second_click_unfurls_literal_text():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "literal unfurled paste " * 10

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.click("#console-command-visible-text")
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert "literal unfurled paste" in visible_plain
        assert "Pasted Text:" not in visible_plain
        assert "Unfurl?" not in visible_plain
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_collapsed_paste_click_targets_token_after_literal_newline():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "newline preceded paste " * 10

        composer.load_draft("prefix\n")
        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        await pilot.click("#console-command-visible-text", offset=(0, 1))
        await pilot.pause(0.1)

        assert visible_draft.renderable.plain == "prefix\nUnfurl?"
        assert composer.draft_text() == f"prefix\n{pasted_text}"


@pytest.mark.asyncio
async def test_console_collapsed_paste_click_targets_second_chunk_independently():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        first_paste = "first large paste " * 10
        second_paste = "second large paste " * 10
        first_token = f"Pasted Text: {len(first_paste)} Characters"
        second_token = f"Pasted Text: {len(second_paste)} Characters"

        composer.insert_pasted_text(first_paste)
        composer.insert_pasted_text(second_paste)
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == f"{first_token}{second_token}"

        await pilot.click(
            "#console-command-visible-text",
            offset=(len(first_token) + 2, 0),
        )
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == f"{first_token}Unfurl?"

        await pilot.click(
            "#console-command-visible-text",
            offset=(len(first_token) + 2, 0),
        )
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert first_token in visible_plain
        assert "second large paste" in visible_plain
        assert "first large paste" not in visible_plain
        assert "Unfurl?" not in visible_plain
        assert composer.draft_text() == f"{first_paste}{second_paste}"


@pytest.mark.asyncio
async def test_console_collapsed_paste_typing_resets_pending_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "typing resets pending paste " * 10
        expected_token = f"Pasted Text: {len(pasted_text)} Characters"

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text", offset=(1, 0))
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == "Unfurl?"

        await pilot.press("x")
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == f"{expected_token}x"
        assert composer.draft_text() == f"{pasted_text}x"

        await pilot.click("#console-command-visible-text", offset=(1, 0))
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == "Unfurl?x"
        assert composer.draft_text() == f"{pasted_text}x"


@pytest.mark.asyncio
async def test_console_collapsed_paste_click_targets_token_after_visible_clipping():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        prefix = "preceding wrapped composer text " * 40
        pasted_text = "visible clipped paste " * 10
        expected_token = f"Pasted Text: {len(pasted_text)} Characters"

        composer.insert_text(prefix)
        composer.insert_pasted_text(pasted_text)
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        visible_lines = visible_plain.splitlines()
        assert len(composer._wrap_draft_lines(composer._display_draft_text(), composer._draft_render_width())) > (
            ConsoleComposerBar.MAX_DRAFT_ROWS
        )
        assert visible_lines[0].startswith("...")
        token_row = next(index for index, line in enumerate(visible_lines) if expected_token in line)
        token_column = visible_lines[token_row].index(expected_token)

        await pilot.click(
            "#console-command-visible-text",
            offset=(token_column + 1, token_row),
        )
        await pilot.pause(0.1)

        assert "Unfurl?" in visible_draft.renderable.plain
        assert expected_token not in visible_draft.renderable.plain
        assert composer.draft_text() == f"{prefix}{pasted_text}"


@pytest.mark.asyncio
async def test_console_collapsed_paste_click_elsewhere_resets_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "reset pending unfurl " * 10
        expected_token = f"Pasted Text: {len(pasted_text)} Characters"

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == "Unfurl?"

        await pilot.click("#console-workspace-grid")
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert expected_token in visible_plain
        assert "Unfurl?" not in visible_plain
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_console_normal_typing_remains_literal_over_paste_threshold():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        typed_text = "normaltypedcomposertext" * 4

        await pilot.press(*typed_text)
        await pilot.pause(0.1)

        visible_plain = visible_draft.renderable.plain
        assert len(typed_text) > ConsoleComposerBar.PASTE_COLLAPSE_THRESHOLD
        assert composer.draft_text() == typed_text
        assert "Pasted Text:" not in visible_plain
        assert "normaltypedcomposertext" in visible_plain
        assert isinstance(visible_draft.renderable, Text)
        assert not visible_draft.renderable.spans


@pytest.mark.asyncio
async def test_console_native_composer_captures_printable_typing_from_non_text_focus():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        command_input = composer.query_one("#console-command-input", Input)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        save_button = composer.query_one("#console-save-chatbook", Button)

        save_button.focus()
        await pilot.pause(0.1)
        await pilot.press("k")
        await pilot.pause(0.1)

        assert command_input.display is False
        assert composer.draft_text() == "k"
        assert "k" in visible_draft.renderable.plain
        assert "k" in _visible_text(composer)


@pytest.mark.asyncio
async def test_console_native_composer_does_not_capture_typing_from_select_focus(monkeypatch):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        compact_bar = console.query_one("#console-compact-model-bar")
        provider_select = compact_bar.query_one("#compact-api-provider", Select)

        monkeypatch.setattr(type(console.app), "focused", property(lambda _app: provider_select))
        await pilot.press("x")
        await pilot.pause(0.1)

        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_native_composer_does_not_capture_paste_from_select_focus(monkeypatch):
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        compact_bar = console.query_one("#console-compact-model-bar")
        provider_select = compact_bar.query_one("#compact-api-provider", Select)

        monkeypatch.setattr(type(console.app), "focused", property(lambda _app: provider_select))
        console.on_paste(Paste("paste should stay with focused control"))
        await pilot.pause(0.1)

        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_send_without_ready_runtime_shows_native_blocked_event(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config.setdefault("api_settings", {}).setdefault("openai", {})["api_key"] = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello console")
        send_button = console.query_one("#console-send-message", Button)
        await console.handle_console_send_message(Button.Pressed(send_button))
        await pilot.pause(0.2)

        text = _visible_text(console)
        assert "Console send blocked" in text
        assert "Missing API key" in text
        assert "Internal Error" not in text
        assert "Missing UI elements" not in text
        assert composer.draft_text() == "hello console"


@pytest.mark.asyncio
async def test_console_enter_sends_native_composer_draft(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config.setdefault("api_settings", {}).setdefault("openai", {})["api_key"] = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("enter")
        await pilot.pause(0.2)

        text = _visible_text(console)
        assert "Console send blocked" in text
        assert "Missing API key" in text
        assert "Internal Error" not in text
        assert "Missing UI elements" not in text
        assert composer.draft_text() == "hello"


@pytest.mark.asyncio
async def test_console_empty_transcript_promotes_start_here_and_provider_recovery():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-start-here")
        await _wait_for_selector(console, pilot, "#console-provider-blocker")

        start_here = console.query_one("#console-start-here", Static)
        provider_strip = console.query_one("#console-provider-recovery-strip")
        assert provider_strip.region.y < start_here.region.y

        text = _visible_text(console)
        for expected in (
            "Empty transcript",
            "Start here",
            "Ask a question",
            "Attach sources",
            "Run command",
            "Provider setup needed",
            "OpenAI missing API key",
            "Settings",
            "No messages yet. Send a prompt or attach context.",
            "Provider setup required before sending.",
            "Enter send",
            "Ctrl+P commands",
            "Attach context",
            "Ask, command, or paste task...",
        ):
            assert expected in text
        assert "Provider: OpenAI is not ready" not in text
        assert "Provider setup is shown in the recovery strip above." not in text
        assert text.lower().count("missing api key") == 1


@pytest.mark.asyncio
async def test_console_provider_blocker_exposes_open_settings_action(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-provider-recovery-strip")
        await _wait_for_selector(console, pilot, "#console-provider-blocker")
        await _wait_for_selector(console, pilot, "#console-open-provider-settings")

        strip = console.query_one("#console-provider-recovery-strip")
        blocker = console.query_one("#console-provider-blocker", Static)
        button = console.query_one("#console-open-provider-settings", Button)
        assert button.display is True
        assert button.disabled is False
        assert button.region.height == 1
        assert button.region.width >= len("Open Settings")
        assert str(button.label) == "Open Settings"
        assert blocker.region.y == button.region.y
        assert str(strip.styles.height) == "auto"
        assert str(blocker.styles.height) == "auto"
        assert button.region.x > blocker.region.x
        assert blocker.region.x >= strip.region.x
        assert button.region.x + button.region.width <= strip.region.x + strip.region.width
        blocker_text = getattr(blocker.render(), "plain", str(blocker.render()))
        assert blocker_text == "Provider setup needed: OpenAI missing API key"
        text = _visible_text(console)
        assert "Open Settings" in text
        assert text.lower().count("missing api key") == 1


@pytest.mark.asyncio
async def test_console_provider_settings_action_posts_navigation_message(monkeypatch):
    app = _build_test_app()
    console = ChatScreen(app)
    event = _PressedEvent()
    posted_messages: list[object] = []
    monkeypatch.setattr(console, "post_message", posted_messages.append)

    await console.handle_console_open_provider_settings(event)  # type: ignore[arg-type]

    assert event.stopped is True
    assert [
        message.screen_name
        for message in posted_messages
        if isinstance(message, NavigateToScreen)
    ] == ["settings"]


@pytest.mark.asyncio
async def test_console_provider_settings_action_hidden_when_provider_ready(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {"api_key": "DUMMY_TEST_KEY"}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.1)

        buttons = [
            button
            for button in console.query("#console-open-provider-settings")
            if button.display
        ]
        assert buttons == []


@pytest.mark.asyncio
async def test_console_provider_blocker_updates_without_transcript_recompose(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-provider-blocker")
        blocker = console.query_one("#console-provider-blocker", Static)

        assert blocker.styles.display != "none"
        assert "OpenAI missing API key" in str(blocker.renderable)

        app.app_config["api_settings"]["openai"]["api_key"] = "sk-test"
        console._sync_console_control_bar()
        await pilot.pause()

        assert blocker.styles.display == "none"
        assert str(blocker.renderable) == ""

        app.app_config["api_settings"]["openai"]["api_key"] = ""
        console._sync_console_control_bar()
        await pilot.pause()

        assert blocker.styles.display != "none"
        assert "OpenAI missing API key" in str(blocker.renderable)


@pytest.mark.asyncio
async def test_console_start_guidance_hides_after_transcript_has_messages():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-start-here")
        await _wait_for_selector(console, pilot, "#console-action-hints")
        start_here = console.query_one("#console-start-here", Static)
        action_hints = console.query_one("#console-action-hints", Static)

        assert start_here.styles.display != "none"
        assert action_hints.styles.display != "none"

        session = console._get_active_chat_session()
        assert session is not None
        session.session_data.message_count = 1
        console._sync_console_control_bar()
        await pilot.pause()

        assert start_here.styles.display == "none"
        assert action_hints.styles.display == "none"


@pytest.mark.asyncio
async def test_console_native_transcript_is_visible_transcript_surface():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        transcript = console.query_one("#console-native-transcript")
        legacy_tabs = console.query_one("#console-chat-tabs")

        assert transcript.region.width > 0
        assert transcript.region.height > 0
        assert transcript.styles.display != "none"
        assert legacy_tabs.styles.display == "none"
        assert "Empty transcript" in _visible_text(console)


@pytest.mark.asyncio
async def test_console_legacy_chat_tab_controls_are_hidden_by_native_transcript():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        chat_tab_buttons = [
            button
            for button in console.query(Button)
            if button.id and button.id.startswith("chat-tab-")
        ]
        close_buttons = [
            button
            for button in console.query(Button)
            if button.id and button.id.startswith("close-tab-")
        ]

        assert len(chat_tab_buttons) == 1
        assert len(close_buttons) == 1

        close_button = close_buttons[0]

        assert str(close_button.label) == "x"
        assert console.query_one("#console-chat-tabs").styles.display == "none"
        assert all(button.region.width == 0 for button in chat_tab_buttons)
        assert all(button.region.width == 0 for button in close_buttons)


@pytest.mark.asyncio
async def test_console_app_footer_status_bar_remains_visible_below_console():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        footer = console.query_one(Footer)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        assert footer.region.height == 1
        assert footer.region.y == console.size.height - 1
        assert composer.region.y + composer.region.height <= footer.region.y


@pytest.mark.asyncio
async def test_console_inspector_live_work_sources_stay_near_top():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(220, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-live-work-source-readiness")

        inspector = console.query_one("#console-run-inspector")
        inspector_state = console.query_one("#console-run-inspector-state")
        source_readiness = console.query_one("#console-live-work-source-readiness")

        assert inspector_state.region.height <= 14
        assert source_readiness.region.y <= inspector.region.y + inspector_state.region.height + 2
        assert source_readiness.region.height <= 18


@pytest.mark.asyncio
async def test_console_inspector_source_readiness_rows_fit_without_tooltip_overlay():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-live-work-source-readiness")

        run_rag = console.query_one("#console-run-library-rag", Button)
        scope = console.query_one("#console-library-rag-scope", Static)
        rows = list(console.query(".console-live-work-source-row"))

        assert run_rag.disabled is True
        assert str(run_rag.tooltip or "") == ""
        scope_plain = getattr(scope.render(), "plain", str(scope.render()))
        assert len(scope_plain) <= scope.region.width
        assert rows
        for row in rows:
            rendered = row.render()
            plain = getattr(rendered, "plain", str(rendered))
            assert len(plain) <= row.region.width


@pytest.mark.asyncio
async def test_console_empty_inspector_hides_disabled_actions_until_actionable():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-inspector-state")

        for selector in (
            "#console-inspector-review-approval",
            "#console-inspector-review-tool-call",
            "#console-inspector-save-chatbook",
        ):
            button = console.query_one(selector, Button)
            assert button.disabled is True
            assert button.region.height == 0
            assert str(button.tooltip or "") == ""


@pytest.mark.asyncio
async def test_console_run_inspector_groups_state_approvals_and_source_readiness():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-run-state-heading")
        await _wait_for_selector(console, pilot, "#console-inspector-approvals-heading")
        await _wait_for_selector(console, pilot, "#console-inspector-source-readiness-heading")

        text = _visible_text(console)
        assert "Run State" in text
        assert "Approvals" in text
        assert "Source Readiness" in text


@pytest.mark.asyncio
async def test_console_workbench_weights_transcript_as_primary_region():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")

        staged = console.query_one("#console-left-rail")
        main = console.query_one("#console-main-column")
        inspector = console.query_one("#console-run-inspector")
        transcript = console.query_one("#console-transcript-region")

        assert main.region.width > staged.region.width
        assert main.region.width > inspector.region.width
        assert staged.region.width >= 40
        assert inspector.region.width >= 40
        assert transcript.region.width == main.region.width


@pytest.mark.asyncio
async def test_console_workbench_panes_have_visible_terminal_frames():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")

        for selector in (
            "#console-workspace-grid",
            "#console-staged-context-tray",
            "#console-workspace-context",
            "#console-transcript-region",
            "#console-run-inspector",
            "#console-native-composer",
        ):
            border = console.query_one(selector).styles.border
            assert border.top[0] == "solid", f"{selector} missing top frame"
            assert border.right[0] == "solid", f"{selector} missing right frame"
            assert border.bottom[0] == "solid", f"{selector} missing bottom frame"
            assert border.left[0] == "solid", f"{selector} missing left frame"


@pytest.mark.asyncio
async def test_console_empty_staged_context_recovery_fits_tray():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-staged-context-recovery")

        recovery = console.query_one("#console-staged-context-recovery")
        rendered = recovery.render()
        plain = getattr(rendered, "plain", str(rendered))

        assert all(len(line) <= recovery.region.width for line in plain.splitlines())


@pytest.mark.asyncio
async def test_console_control_bar_renders_readable_summary_line():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-control-bar")

        summary = console.query_one("#console-control-status-line", Static)
        plain = getattr(summary.render(), "plain", str(summary.render()))

        assert "Provider:" in plain
        assert " | Model:" in plain
        assert " | Assistant:" in plain
        assert " | Sources:" in plain


@pytest.mark.asyncio
async def test_console_composer_status_renders_session_metadata_as_plain_text():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.sync_session_data(
            ChatSessionData(
                tab_id="metadata",
                title="[red]Injected[/red]",
                runtime_backend="[blue]server[/blue]",
                assistant_id="[green]persona[/green]",
                scope_type="workspace",
                workspace_id="[yellow]workspace[/yellow]",
            )
        )
        await pilot.pause()

        status = console.query_one("#console-composer-status", Static)
        rendered = status.render()
        plain = getattr(rendered, "plain", str(rendered))

        assert "[red]Injected[/red]" in plain
        assert "[blue]server[/blue]" in plain
        assert "[green]persona[/green]" in plain
        assert "[yellow]workspace[/yellow]" in plain


@pytest.mark.asyncio
async def test_console_native_control_bar_and_staged_context_reflect_pending_handoff():
    app = _build_test_app()
    app.pending_console_launch = {
        "source": "Library Search/RAG",
        "title": "Transformer notes",
        "status": "ready",
        "recovery": "Review citations before sending.",
        "payload": {"source_id": "note-1", "citation_count": 2},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-control-bar")

        text = _visible_text(console)
        assert "Provider:" in text
        assert "Model:" in text
        assert "Assistant: General" in text
        assert "RAG:" in text
        assert "Sources: 1 staged" in text
        assert "Transformer notes" in text
        assert "citation_count: 2" in text
        assert "Review citations before sending." in text


@pytest.mark.asyncio
async def test_console_native_control_bar_uses_existing_compact_model_sync_seam():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        compact_bar = console.query_one("#console-compact-model-bar")
        console._sync_compact_shell_controls(
            model="console-test-model",
            temperature="0.2",
        )

        assert (
            compact_bar.query_one("#compact-api-model", Select).value
            == "console-test-model"
        )
        assert compact_bar.query_one("#compact-temperature", Input).value == "0.2"


@pytest.mark.asyncio
async def test_console_mounts_only_one_compact_model_bar():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        assert len(console.query(CompactModelBar)) == 1
        assert len(console.query("#compact-model-bar")) == 0


@pytest.mark.asyncio
async def test_console_control_labels_refresh_after_compact_control_sync():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")

        compact_bar = console.query_one("#console-compact-model-bar")
        provider_select = compact_bar.query_one("#compact-api-provider", Select)
        provider = next(
            value
            for _, value in provider_select._options
            if isinstance(value, str)
        )

        console._sync_compact_shell_controls(
            provider=provider,
            model="console-test-model",
        )
        await pilot.pause()

        assert provider in str(console.query_one("#console-provider-label").renderable)
        assert "console-test-model" in str(
            console.query_one("#console-model-label").renderable
        )


def test_console_control_state_tolerates_missing_config_and_precise_rag_source():
    app = _build_test_app()
    app.app_config = None
    screen = ChatScreen(app)

    assert screen._chat_default_value("provider") is None

    non_rag_state = screen._build_console_control_state(
        ConsoleLiveWorkLaunch(source="storage", title="Storage queue"),
    )
    rag_state = screen._build_console_control_state(
        ConsoleLiveWorkLaunch(source="Library Search/RAG", title="RAG result"),
    )

    assert non_rag_state.rag_label == "RAG: off"
    assert rag_state.rag_label == "RAG: on"


def test_console_control_state_tolerates_missing_launch_source():
    app = _build_test_app()
    screen = ChatScreen(app)

    state = screen._build_console_control_state(
        ConsoleLiveWorkLaunch(source=None, title="Unknown source"),
    )

    assert state.rag_label == "RAG: off"


def test_console_control_and_inspector_share_effective_provider_model_sources():
    app = _build_test_app()
    app.app_config = {"chat_defaults": {"model": "reactive-model"}}
    app.chat_api_provider_value = "ReactiveOpenAI"
    screen = ChatScreen(app)

    control_state = screen._build_console_control_state(None)
    inspector_state = screen._build_console_inspector_state(None)
    rows_by_label = {row.label: row for row in inspector_state.rows}

    assert control_state.provider_label == "Provider: ReactiveOpenAI"
    assert control_state.model_label == "Model: reactive-model"
    assert rows_by_label["Provider"].text == "Provider: ready"


@pytest.mark.asyncio
async def test_console_run_inspector_shows_blocked_provider_and_missing_rag_source():
    app = _build_test_app()
    app.app_config = {"chat_defaults": {}}
    app.console_provider_ready = False
    app.pending_console_launch = {
        "source": "Library Search/RAG",
        "title": "Grounded answer",
        "status": "ready",
        "recovery": "Attach a source before asking the model.",
        "payload": {},
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-provider")

        assert "Provider: blocked" in str(
            console.query_one("#console-inspector-provider", Static).renderable
        )
        assert "Select a provider and model before sending." in str(
            console.query_one("#console-inspector-provider", Static).renderable
        )
        assert "RAG/source: missing source" in str(
            console.query_one("#console-inspector-rag-source", Static).renderable
        )
        assert console.query_one("#console-inspector-review-tool-call", Button).disabled is True
        assert "No tool calls are ready for review." in str(
            console.query_one("#console-inspector-review-tool-call-reason", Static).renderable
        )


@pytest.mark.asyncio
async def test_console_run_inspector_exposes_pending_approval_and_chatbook_artifact_actions():
    app = _build_test_app()
    app.console_pending_approval_count = 1
    app.console_tool_count = 1
    app.pending_console_launch = {
        "source": "artifacts",
        "title": "Grounded Answer Chatbook",
        "status": "ready",
        "recovery": "Review this Chatbook artifact in Console or return to Artifacts.",
        "payload": {"target_id": "local:chatbook:77", "chatbook_id": 77},
        "action_label": "Open Chatbook artifact",
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-review-approval")

        assert "Approvals: 1 pending" in str(
            console.query_one("#console-inspector-approvals", Static).renderable
        )
        assert "Artifacts: Chatbook artifact available" in str(
            console.query_one("#console-inspector-artifacts", Static).renderable
        )
        assert console.query_one("#console-inspector-review-approval", Button).disabled is False
        assert console.query_one("#console-inspector-review-tool-call", Button).disabled is False
        assert console.query_one("#console-inspector-save-chatbook", Button).disabled is False
        assert (
            console.query_one("#console-inspector-tools").region.y
            < console.query_one("#console-inspector-review-tool-call").region.y
            < console.query_one("#console-inspector-approvals-heading").region.y
        )
        assert (
            console.query_one("#console-inspector-approvals-heading").region.y
            < console.query_one("#console-inspector-review-approval").region.y
            < console.query_one("#console-inspector-source-readiness-heading").region.y
        )
        assert (
            console.query_one("#console-inspector-artifacts").region.y
            < console.query_one("#console-inspector-save-chatbook").region.y
        )
        assert console.query_one("#console-live-work-primary-action", Button).disabled is False


@pytest.mark.asyncio
async def test_console_rag_action_requests_library_retrieval_and_stages_result():
    app = _build_test_app()
    service = StaticConsoleLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "score": 0.93,
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                    "runtime_backend": "local-fts",
                    "citations": [{"label": "Incident Review p.2"}],
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    app.library_rag_search_service = service
    host = ConsoleHarness(app)
    query = "Why did the incident happen?"

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-library-rag")

        assert "Scope: notes, media, conversations" in _visible_text(console)
        query_input = console.query_one("#console-library-rag-query-input", Input)
        query_input.value = query
        await pilot.pause(0.1)

        run_button = console.query_one("#console-run-library-rag", Button)
        assert run_button.disabled is False
        run_button.press()
        await _wait_for_selector(console, pilot, "#console-live-work-payload-source-id")

        assert service.calls == [
            {
                "query": query,
                "scope": ("notes", "media", "conversations"),
                "mode": "rag",
                "top_k": 5,
                "include_citations": True,
            }
        ]
        text = _visible_text(console)
        assert "RAG/source: staged from Library Search/RAG" in text
        assert "Title: Incident Review" in text
        assert "source_id: note-42" in text
        assert "chunk_id: chunk-7" in text
        assert "Review citations before sending." in text


@pytest.mark.asyncio
async def test_console_rag_query_validation_blocks_unsafe_markup():
    app = _build_test_app()
    service = StaticConsoleLibraryRagSearchService({"results": []})
    app.library_rag_search_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-library-rag")

        console.query_one("#console-library-rag-query-input", Input).value = (
            "<script>alert('bad')</script>"
        )
        await _wait_for_console_library_rag_button_state(
            console,
            pilot,
            disabled=True,
        )
        assert str(console.query_one("#console-run-library-rag", Button).tooltip or "") == ""

        assert console._console_library_rag_query == ""
        assert service.calls == []


@pytest.mark.asyncio
async def test_console_rag_action_without_service_stages_recoverable_blocker():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-run-library-rag")

        console.query_one("#console-library-rag-query-input", Input).value = "What changed?"
        await pilot.pause(0.1)
        console.query_one("#console-run-library-rag", Button).press()
        await _wait_for_selector(console, pilot, "#console-live-work-status")

        text = _visible_text(console)
        assert "Status: blocked" in text
        assert "RAG/source: unavailable" in text
        assert "Unavailable: Library Search/RAG retrieval." in text
        assert "Owner: Library retrieval service." in text
