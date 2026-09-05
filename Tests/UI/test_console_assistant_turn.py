"""Focused tests for Console Assistant-turn presentation primitives."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_roleplay_identity import ConsolePresentationContext
from tldw_chatbook.css.Themes.themes import ALL_THEMES
from tldw_chatbook.Widgets.Console.console_assistant_turn import (
    ConsoleActivityActivated,
    ConsoleActivityDisclosure,
    ConsoleActivityHeader,
    ConsoleAssistantTurnWidget,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


_CSS_DIR = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"


def _painted_background(app: App, widget) -> object:
    """Return the compositor-painted background near a widget's right edge."""
    strips = app.screen._compositor.render_strips()
    y = widget.region.y
    x = widget.region.x + max(0, widget.region.width - 2)
    cursor = 0
    for segment in strips[y]:
        next_cursor = cursor + segment.cell_length
        if cursor <= x < next_cursor:
            return None if segment.style is None else segment.style.bgcolor
        cursor = next_cursor
    raise AssertionError(f"no painted segment at ({x}, {y})")


def _relative_luminance(color) -> float:
    """Return WCAG relative luminance for one compositor-painted color."""
    triplet = color.get_truecolor()

    def _channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * _channel(triplet.red)
        + 0.7152 * _channel(triplet.green)
        + 0.0722 * _channel(triplet.blue)
    )


def _contrast(first, second) -> float:
    """Return WCAG contrast between two compositor-painted colors."""
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_foreground_and_background(app: App, widget) -> tuple[object, object]:
    """Return the first visible glyph's compositor foreground/background."""
    strips = app.screen._compositor.render_strips()
    for y in range(widget.region.y, widget.region.bottom):
        cursor = 0
        for segment in strips[y]:
            next_cursor = cursor + segment.cell_length
            overlaps = cursor < widget.region.right and next_cursor > widget.region.x
            if overlaps:
                overlap_start = max(0, widget.region.x - cursor)
                overlap_end = min(segment.cell_length, widget.region.right - cursor)
                _, overlap = segment.split_cells(overlap_start)
                overlap, _ = overlap.split_cells(overlap_end - overlap_start)
            if overlaps and overlap.text.strip() and overlap.style is not None:
                foreground = overlap.style.color
                background = overlap.style.bgcolor
                if foreground is not None and background is not None:
                    return foreground, background
            cursor = next_cursor
    raise AssertionError(f"no painted glyph colors inside {widget.region!r}")


class ActivityHarness(App[None]):
    """Mount two independent disclosures and apply their emitted state."""

    def __init__(self, *disclosures: ConsoleActivityDisclosure) -> None:
        super().__init__()
        self.disclosures = disclosures
        self.activations: list[ConsoleActivityActivated] = []

    def compose(self) -> ComposeResult:
        yield from self.disclosures

    def on_console_activity_activated(self, event: ConsoleActivityActivated) -> None:
        self.activations.append(event)
        disclosure = self.query_one(
            f"#console-activity-disclosure-{event.message_id}",
            ConsoleActivityDisclosure,
        )
        disclosure.sync_state(
            expanded=(not disclosure.expanded)
            if event.toggle_requested
            else disclosure.expanded,
            selected=True,
        )


class StyledActivityHarness(ActivityHarness):
    """Activity harness loading the exact production stylesheet stack."""

    CSS_PATH = [
        str(_CSS_DIR / "screen_css_scoped.tcss"),
        str(_CSS_DIR / "tldw_cli_modular.tcss"),
        str(_CSS_DIR / "screen_agentic_console.tcss"),
        str(_CSS_DIR / "screen_css_self.tcss"),
    ]


def _disclosure(
    activity_message_id: str,
    *,
    label: str = "Thinking",
    status: str = "done",
    selected: bool = False,
    actions: tuple[Static, ...] = (),
    details: tuple[Static, ...] = (),
) -> ConsoleActivityDisclosure:
    return ConsoleActivityDisclosure(
        activity_message_id,
        label,
        status,
        selected=selected,
        action_widgets=actions,
        detail_widgets=details,
    )


@pytest.mark.asyncio
async def test_disclosures_are_independently_collapsed_by_default() -> None:
    app = ActivityHarness(
        _disclosure("activity-one", details=(Static("one detail"),)),
        _disclosure("activity-two", details=(Static("two detail"),)),
    )

    async with app.run_test():
        first = app.query_one(
            "#console-activity-disclosure-activity-one", ConsoleActivityDisclosure
        )
        second = app.query_one(
            "#console-activity-disclosure-activity-two", ConsoleActivityDisclosure
        )

        assert not first.expanded
        assert not second.expanded
        assert not first.query_one(".console-activity-detail-stack").display
        assert not second.query_one(".console-activity-detail-stack").display


@pytest.mark.asyncio
async def test_expandable_header_click_enter_and_space_emit_original_id() -> None:
    disclosure = _disclosure(
        "activity-expandable",
        label="fs_[literal]",
        status="success",
        details=(Static("detail"),),
    )
    app = ActivityHarness(disclosure)

    async with app.run_test() as pilot:
        header = app.query_one(ConsoleActivityHeader)
        assert header.can_focus
        assert header.renderable.plain == "▸ fs_[literal] · success"

        await pilot.click("#console-activity-header-activity-expandable")
        await pilot.pause()
        assert [
            (event.message_id, event.toggle_requested) for event in app.activations
        ] == [("activity-expandable", True)]
        assert header.renderable.plain == "▾ fs_[literal] · success"

        header.focus()
        await pilot.press("enter")
        await pilot.press("space")
        await pilot.press("x")
        await pilot.pause()

    assert [
        (event.message_id, event.toggle_requested) for event in app.activations
    ] == [
        ("activity-expandable", True),
        ("activity-expandable", True),
        ("activity-expandable", True),
    ]


@pytest.mark.asyncio
async def test_activity_header_child_regions_preserve_literal_copy_and_activation() -> (
    None
):
    """Split label/status children stay literal and bubble activation to the header."""
    disclosure = _disclosure(
        "child-activation",
        label="fs_[literal]",
        status="success",
        details=(Static("detail"),),
    )
    app = StyledActivityHarness(disclosure)

    async with app.run_test(size=(42, 12)) as pilot:
        label = app.query_one("#console-activity-label-child-activation", Static)
        status = app.query_one("#console-activity-status-child-activation", Static)
        assert label.renderable.plain == "▸ fs_[literal]"
        assert status.renderable.plain == "· success"

        await pilot.click(label)
        await pilot.click(status)
        disclosure.header.focus()
        await pilot.press("enter")
        await pilot.press("space")
        await pilot.pause()

    assert [
        (event.message_id, event.toggle_requested) for event in app.activations
    ] == [
        ("child-activation", True),
        ("child-activation", True),
        ("child-activation", True),
        ("child-activation", True),
    ]


@pytest.mark.asyncio
async def test_no_detail_activity_is_focusable_and_selects_without_toggle() -> None:
    app = ActivityHarness(_disclosure("thinking-empty"))

    async with app.run_test() as pilot:
        header = app.query_one(ConsoleActivityHeader)
        assert header.can_focus
        assert header.renderable.plain == "Thinking · done"
        assert "▸" not in header.renderable.plain
        assert "▾" not in header.renderable.plain

        header.focus()
        await pilot.press("enter")
        await pilot.pause()

        disclosure = app.query_one(ConsoleActivityDisclosure)
        assert disclosure.selected
        assert not disclosure.expanded

    assert [
        (event.message_id, event.toggle_requested) for event in app.activations
    ] == [("thinking-empty", False)]


@pytest.mark.asyncio
async def test_selected_actions_precede_hidden_detail_while_collapsed() -> None:
    disclosure = _disclosure(
        "activity-selected",
        selected=True,
        actions=(Static("Copy", id="activity-action"),),
        details=(Static("private detail", id="activity-detail"),),
    )
    app = ActivityHarness(disclosure)

    async with app.run_test():
        children = list(disclosure.children)
        assert [child.id for child in children] == [
            "console-activity-header-activity-selected",
            "console-activity-actions-activity-selected",
            "console-activity-detail-activity-selected",
        ]
        assert children[1].display
        assert not children[2].display
        assert disclosure.query_one("#activity-action").display


@pytest.mark.asyncio
async def test_parent_applied_expansion_leaves_sibling_collapsed() -> None:
    app = ActivityHarness(
        _disclosure("first", details=(Static("first detail"),)),
        _disclosure("second", details=(Static("second detail"),)),
    )

    async with app.run_test() as pilot:
        await pilot.click("#console-activity-header-first")
        await pilot.pause()

        first = app.query_one(
            "#console-activity-disclosure-first", ConsoleActivityDisclosure
        )
        second = app.query_one(
            "#console-activity-disclosure-second", ConsoleActivityDisclosure
        )
        assert first.expanded
        assert first.query_one(".console-activity-detail-stack").display
        assert not second.expanded
        assert not second.query_one(".console-activity-detail-stack").display


@pytest.mark.asyncio
async def test_activity_sync_preserves_child_identity_and_replaces_status_class() -> (
    None
):
    disclosure = _disclosure(
        "sync-status",
        label="fs_list",
        status="success",
        details=(Static("detail"),),
    )
    app = ActivityHarness(disclosure)

    async with app.run_test():
        label = disclosure.header.label_widget
        status = disclosure.header.status_widget

        disclosure.sync_activity(
            "fs_list retry",
            "failed",
            expanded=False,
            selected=True,
        )

        assert disclosure.header.label_widget is label
        assert disclosure.header.status_widget is status
        assert label.renderable.plain == "▸ fs_list retry"
        assert status.renderable.plain == "· failed"
        assert status.has_class("console-activity-status-failed")
        assert not status.has_class("console-activity-status-success")


class AssistantTurnHarness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.header = Static("Assistant", id="assistant-header")
        self.answer = Static("Final answer", id="assistant-answer")
        self.turn = ConsoleAssistantTurnWidget(
            "assistant-one",
            self.header,
            (Static("Thinking", id="activity-old"),),
            self.answer,
            (Static("Sources", id="assistant-adjunct"),),
        )

    def compose(self) -> ComposeResult:
        yield self.turn


@pytest.mark.asyncio
async def test_assistant_turn_dom_order_is_header_activity_answer_adjunct() -> None:
    app = AssistantTurnHarness()

    async with app.run_test():
        assert [child.id for child in app.turn.children] == [
            "assistant-header",
            "console-assistant-activities-assistant-one",
            "assistant-answer",
            "console-assistant-adjuncts-assistant-one",
        ]
        assert app.turn.assistant_message_id == "assistant-one"


@pytest.mark.asyncio
async def test_replacing_activities_preserves_turn_header_and_answer_identity() -> None:
    app = AssistantTurnHarness()

    async with app.run_test():
        turn = app.turn
        original_turn = turn
        original_header = app.header
        original_answer = app.answer
        replacement = Static("Tool", id="activity-old")

        await turn.replace_activity_widgets(
            (replacement, Static("Thinking again", id="activity-new"))
        )

        assert turn is original_turn
        assert turn.query_one("#assistant-header") is original_header
        assert turn.query_one("#assistant-answer") is original_answer
        assert turn.query_one("#activity-old") is replacement
        assert len(turn.query("#activity-old")) == 1
        assert len(turn.query("#activity-new")) == 1
        assert [child.id for child in turn.activity_stack.children] == [
            "activity-old",
            "activity-new",
        ]


class StyledTranscriptHarness(App[None]):
    """Production-shaped transcript host loading the exact app CSS stack."""

    CSS_PATH = [
        str(_CSS_DIR / "screen_css_scoped.tcss"),
        str(_CSS_DIR / "tldw_cli_modular.tcss"),
        str(_CSS_DIR / "screen_agentic_console.tcss"),
        str(_CSS_DIR / "screen_css_self.tcss"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.transcript = ConsoleTranscript(id="console-native-transcript")

    def compose(self) -> ComposeResult:
        yield self.transcript


class AllThemesStyledTranscriptHarness(StyledTranscriptHarness):
    """Production transcript harness with every shipped theme registered."""

    def __init__(self) -> None:
        super().__init__()
        for theme in ALL_THEMES:
            self.register_theme(theme)


def _styled_turn_messages() -> tuple[ConsoleChatMessage, ConsoleChatMessage]:
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="The requested files are listed below, with their current status.",
        id="styled-assistant",
    )
    activity = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="short preview",
        id="styled-tool",
        activity_presentation=ConsoleActivityPresentation(
            "tool",
            "fs_list · a deliberately long literal workspace label [not markup]",
            "success",
        ),
        tool_output_full="short preview\nfull detail line that remains visibly nested",
    )
    return assistant, activity


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_size", [(120, 32), (42, 24)])
async def test_assistant_turn_geometry_under_production_bundle(
    terminal_size: tuple[int, int],
) -> None:
    """Wide and narrow Console sizes keep the complete Assistant unit contained."""
    app = StyledTranscriptHarness()

    async with app.run_test(size=terminal_size) as pilot:
        assistant, activity = _styled_turn_messages()
        transcript = app.transcript
        transcript.set_messages([assistant, activity])
        await transcript.refresh_messages()
        transcript.toggle_tool_output(activity.id)
        await pilot.pause(0.2)

        turn = transcript.query_one("#console-assistant-turn-styled-assistant")
        header = transcript.query_one("#console-activity-header-styled-tool")
        label = transcript.query_one("#console-activity-label-styled-tool", Static)
        status = transcript.query_one("#console-activity-status-styled-tool", Static)
        detail = transcript.query_one("#console-activity-detail-styled-tool")
        answer = transcript.query_one("#console-message-styled-assistant")

        for name, widget in {
            "turn": turn,
            "activity header": header,
            "activity label": label,
            "activity status": status,
            "expanded detail": detail,
            "Assistant answer": answer,
        }.items():
            assert widget.region.width > 0, f"{name} collapsed to zero width"
            assert widget.region.height > 0, f"{name} collapsed to zero height"

        content = turn.content_region
        for name, widget in {
            "activity header": header,
            "expanded detail": detail,
            "Assistant answer": answer,
        }.items():
            assert widget.region.x >= content.x, f"{name} overflows left"
            assert widget.region.right <= content.right, f"{name} overflows right"

        assert label.region.right <= status.region.x
        assert status.region.width == 9
        assert status.region.right <= header.content_region.right
        assert detail.region.width >= header.region.width - 4
        assert "success" in app.export_screenshot(), "status text is compositor-clipped"


@pytest.mark.asyncio
async def test_narrow_activity_label_ellipsizes_before_fixed_status() -> None:
    """The bounded status never yields its columns to a worst-case literal label."""
    app = StyledTranscriptHarness()

    async with app.run_test(size=(42, 24)) as pilot:
        assistant, activity = _styled_turn_messages()
        app.transcript.set_messages([assistant, activity])
        await app.transcript.refresh_messages()
        await pilot.pause(0.2)

        header = app.query_one("#console-activity-header-styled-tool")
        label = app.query_one("#console-activity-label-styled-tool", Static)
        status = app.query_one("#console-activity-status-styled-tool", Static)
        screenshot = app.export_screenshot()

        assert label.region.width < len(activity.activity_presentation.label)
        assert label.region.right <= status.region.x
        assert status.region.width == 9
        assert status.region.right <= header.content_region.right
        assert "…" in screenshot
        assert "success" in screenshot


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_composite_retains_roleplay_and_failed_answer_backgrounds(
    theme: str,
) -> None:
    """The shared surface must not erase answer-level semantic fills."""
    app = StyledTranscriptHarness()
    app.app_config = {"chat_defaults": {"assistant_markdown": False}}

    async with app.run_test(size=(90, 30)) as pilot:
        app.theme = theme
        transcript = app.transcript
        transcript.set_presentation_context(
            ConsolePresentationContext(
                assistant_kind="character",
                character_name="Alraune",
            )
        )
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="character answer",
                    id="roleplay-answer",
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.TOOL,
                    content="roleplay tool",
                    id="roleplay-tool",
                    activity_presentation=ConsoleActivityPresentation(
                        "tool", "fs_list", "success"
                    ),
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="failed answer",
                    id="failed-answer",
                    status="failed",
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.TOOL,
                    content="failed tool",
                    id="failed-tool",
                    activity_presentation=ConsoleActivityPresentation(
                        "tool", "fs_write", "failed"
                    ),
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.SYSTEM,
                    content="neutral",
                    id="neutral-system",
                ),
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause(0.2)

        roleplay = transcript.query_one("#console-message-roleplay-answer")
        failed = transcript.query_one("#console-message-failed-answer")
        neutral = transcript.query_one("#console-message-neutral-system")
        roleplay_background = _painted_background(app, roleplay)
        failed_background = _painted_background(app, failed)
        neutral_background = _painted_background(app, neutral)

        assert roleplay_background != neutral_background
        assert failed_background != neutral_background
        assert failed_background != roleplay_background


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("status", ["success", "failed", "blocked", "done"])
@pytest.mark.parametrize("state", ["rest", "focus", "selected"])
async def test_activity_status_compositor_contrast(
    theme: str,
    status: str,
    state: str,
) -> None:
    """Every terminal status remains ordinary-text readable in every state."""
    disclosure = _disclosure(
        f"contrast-{theme}-{status}-{state}",
        label="fs_list",
        status=status,
        selected=state == "selected",
        details=(Static("detail"),),
    )
    app = StyledActivityHarness(disclosure)
    app.theme = theme

    async with app.run_test(size=(48, 12)) as pilot:
        if state == "focus":
            disclosure.header.focus()
        else:
            app.set_focus(None)
        await pilot.pause(0.2)

        status_widget = disclosure.header.status_widget
        foreground, background = _painted_foreground_and_background(app, status_widget)
        ratio = _contrast(foreground, background)
        assert ratio >= 4.5, (
            f"{status} status contrast is {ratio:.2f}:1 under {theme}/{state}; "
            f"foreground={foreground}, background={background}"
        )


@pytest.mark.asyncio
async def test_success_status_rest_contrast_in_every_shipped_theme() -> None:
    """Success remains readable against every shipped theme's resolved tokens."""
    app = AllThemesStyledTranscriptHarness()

    async with app.run_test(size=(48, 12)) as pilot:
        assistant, activity = _styled_turn_messages()
        app.transcript.set_messages([assistant, activity])
        await app.transcript.refresh_messages()
        await pilot.pause(0.2)
        status_widget = app.query_one("#console-activity-status-styled-tool", Static)
        results: dict[str, float] = {}
        for theme in ALL_THEMES:
            app.theme = theme.name
            await pilot.pause()
            foreground, background = _painted_foreground_and_background(
                app, status_widget
            )
            results[theme.name] = _contrast(foreground, background)

        theme_name, ratio = min(results.items(), key=lambda item: item[1])
        assert len(results) == len(ALL_THEMES)
        assert ratio >= 4.5, (
            f"success status contrast is {ratio:.3f}:1 under {theme_name}; "
            f"all results={results}"
        )
