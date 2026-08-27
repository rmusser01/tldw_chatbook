from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Markdown, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleCitationNoticeCode,
    ConsoleCitationPhase,
    ConsoleCitationPresentation,
    ConsoleMessageRole,
    ConsoleVariantSet,
    GenerationVariantMeta,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_message_actions import (
    ConsoleMessageActionService,
    ConsoleSaveDestination,
)
from tldw_chatbook.Chat.console_chat_fork import ConsoleForkEligibility
from tldw_chatbook.Chat.console_onboarding_state import ConsoleSetupCardState
from tldw_chatbook.Chat.console_roleplay_identity import (
    ConsolePresentationContext,
    ConsoleTranscriptStyle,
    resolve_console_message_presentation,
)
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Widgets.Console.console_save_as_modal import ConsoleSaveAsModal
from tldw_chatbook.Widgets.Console.console_assistant_turn import (
    ConsoleActivityDisclosure,
    ConsoleAssistantTurnWidget,
)
from tldw_chatbook.Widgets.Console.console_generation_card import (
    ConsoleGenerationCardSpec,
)
from tldw_chatbook.Widgets.Console.console_video_card import ConsoleVideoCardSpec
import tldw_chatbook.Widgets.Console.console_transcript as transcript_module
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleMessageHeader,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)


def test_unit_span_grouping_is_built_once_per_message_ingest(monkeypatch) -> None:
    real_group = transcript_module.group_console_transcript_messages
    group_calls = 0

    def counted_group(messages):
        nonlocal group_calls
        group_calls += 1
        return real_group(messages)

    monkeypatch.setattr(
        transcript_module,
        "group_console_transcript_messages",
        counted_group,
    )
    transcript = ConsoleTranscript()
    transcript.set_messages(
        [
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER,
                content="Question",
                id="user-turn",
            ),
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="Answer",
                id="assistant-turn",
            ),
            ConsoleChatMessage(
                role=ConsoleMessageRole.TOOL,
                content="Result",
                id="tool-turn",
            ),
        ]
    )

    assert transcript._unit_span_at(transcript._messages, 1)[:2] == (1, 3)
    assert transcript._unit_span_at(transcript._messages, 2)[:2] == (1, 3)
    assert group_calls == 1


def _message_row_text(transcript: ConsoleTranscript, message_id: str) -> str:
    """Renderer-agnostic visible text of one message row (TASK-1990).

    Plain rows expose a single Content renderable; markdown rows expose
    header/footer Statics plus the Markdown source.
    """
    row = transcript.query_one(f"#console-message-{message_id}")
    if isinstance(row, ConsoleMarkdownMessage):
        parts = [str(static.renderable) for static in row.query(Static)]
        turns = list(transcript.query(f"#console-assistant-turn-{message_id}"))
        if turns:
            parts[:0] = [
                str(static.renderable)
                for static in turns[0].header_widget.query(Static)
            ]
        parts.append(row.query_one(Markdown).source)
        return "\n".join(parts)
    statics = list(row.query(Static))
    if statics:
        return "\n".join(str(static.renderable) for static in statics)
    return str(row.renderable)


def _spy_move_child(transcript: ConsoleTranscript) -> list[tuple[tuple, dict]]:
    """Wrap ``transcript.move_child`` with a call-recording spy (task-15453).

    Returns the list the spy appends ``(args, kwargs)`` to; still delegates
    to the real ``Widget.move_child`` so reconciliation behaves normally.
    """
    calls: list[tuple[tuple, dict]] = []
    original = transcript.move_child

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    transcript.move_child = _spy  # type: ignore[method-assign]
    return calls


def _rendered_message_ids(transcript: ConsoleTranscript) -> list[str]:
    """Message ids in actual mounted DOM order (not store order)."""
    return [
        widget.message_id for widget in transcript.query(".console-transcript-message")
    ]


def _planned_message_widgets(transcript: ConsoleTranscript) -> list[object]:
    """Flatten message bodies from top-level rows and Assistant turn shells."""
    planned: list[object] = []
    for widget in transcript._message_widgets():
        if isinstance(widget, ConsoleAssistantTurnWidget):
            planned.append(widget.answer_widget)
        else:
            planned.append(widget)
    return planned


def _speaker_label_for(transcript: ConsoleTranscript, message_id: str) -> Static:
    """Resolve the label from a standalone row or an Assistant turn header."""
    turns = list(transcript.query(f"#console-assistant-turn-{message_id}"))
    owner = (
        turns[0].header_widget
        if turns
        else transcript.query_one(f"#console-message-{message_id}")
    )
    return owner.query_one(".console-transcript-speaker-label", Static)


_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


def _roleplay_context(
    *,
    user_name: str = "Captain [Rowan]",
    character_name: str = "Alraune",
    revision: int = 1,
) -> ConsolePresentationContext:
    return ConsolePresentationContext(
        user_name=user_name,
        assistant_kind="character",
        character_name=character_name,
        revision=revision,
    )


def _painted_background(app: App, widget) -> object:
    """Return the compositor-painted background at the row's right padding."""
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
    """WCAG relative luminance of a compositor-painted Rich color."""
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
    """WCAG contrast ratio between two compositor-painted colors."""
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_foreground_and_background(app: App, widget) -> tuple[object, object]:
    """Return the first visible glyph's compositor-painted foreground/background."""
    strips = app.screen._compositor.render_strips()
    for y in range(widget.region.y, widget.region.bottom):
        cursor = 0
        for segment in strips[y]:
            next_cursor = cursor + segment.cell_length
            overlaps = cursor < widget.region.right and next_cursor > widget.region.x
            if overlaps and segment.text.strip() and segment.style is not None:
                foreground = segment.style.color
                background = segment.style.bgcolor
                if foreground is not None and background is not None:
                    return foreground, background
            cursor = next_cursor
    raise AssertionError(f"no painted glyph colors inside {widget.region!r}")


# Speaker labels are ordinary-sized text, so their compositor-painted colors
# must meet WCAG AA in every supported theme. The literal bold speaker name
# and explicit "Failed" status also keep failure understandable without color.
MIN_SPEAKER_CONTRAST = 4.5


class TranscriptHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="hello", id="m1"
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT, content="answer", id="m2"
                ),
            ]
        )
        yield transcript


class EmptyTranscriptHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


class MutableTranscriptHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _assistant_turn_messages() -> list[ConsoleChatMessage]:
    """Production-shaped Assistant turn with two owned activity markers."""
    return [
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER, content="inspect", id="u-turn"
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="The workspace contains two files.",
            id="a-turn",
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.TOOL,
            content="safe planning preamble",
            id="thinking-turn",
            activity_presentation=ConsoleActivityPresentation(
                "thinking", "Thinking", "done"
            ),
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_list → a.txt, b.txt…",
            id="tool-turn",
            tool_output_full="a.txt\nb.txt\nfull tail",
            activity_presentation=ConsoleActivityPresentation(
                "tool", "fs_list", "success"
            ),
        ),
    ]


def test_assistant_turn_visual_order_navigation_keeps_nested_message_ids() -> None:
    """j/k walks the painted turn order while Inspector ids stay causal."""
    transcript = ConsoleTranscript()
    messages = [
        *_assistant_turn_messages(),
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER, content="continue", id="u-next"
        ),
    ]
    transcript.set_messages(messages)

    visited: list[str | None] = []
    for _ in messages:
        transcript.action_select_next()
        visited.append(transcript.selected_message_id)

    assert visited == [
        "u-turn",
        "thinking-turn",
        "tool-turn",
        "a-turn",
        "u-next",
    ]
    assert transcript.display_message("thinking-turn") is messages[2]
    assert transcript.display_message("tool-turn") is messages[3]


def test_assistant_turn_plain_export_is_ordered_bounded_and_expansion_independent() -> (
    None
):
    """Plain export uses structured activity previews, never disclosure detail."""
    transcript = ConsoleTranscript()
    user, assistant, thinking, tool = _assistant_turn_messages()
    tool.tool_diff = ("secret.txt", "DIFF-BEFORE-SECRET", "DIFF-AFTER-SECRET")
    transcript.set_messages([user, assistant, thinking, tool])
    transcript.select_message(assistant.id)

    collapsed = transcript.to_plain_text(width=48)
    transcript.toggle_tool_output(tool.id)
    expanded = transcript.to_plain_text(width=48)

    assert expanded == collapsed
    assert collapsed.count("Assistant") == 1
    thinking_header = "Thinking · done"
    tool_header = "fs_list · success"
    assert collapsed.index("User") < collapsed.index("Assistant")
    assert collapsed.index("Assistant") < collapsed.index(thinking_header)
    assert collapsed.index(thinking_header) < collapsed.index("safe planning preamble")
    assert collapsed.index("safe planning preamble") < collapsed.index(tool_header)
    assert collapsed.index(tool_header) < collapsed.index("a.txt, b.txt")
    assert collapsed.index("a.txt, b.txt") < collapsed.index(assistant.content)
    assert "full tail" not in collapsed
    assert "DIFF-BEFORE-SECRET" not in collapsed
    assert "DIFF-AFTER-SECRET" not in collapsed
    assert "Copy" in collapsed


@pytest.mark.asyncio
async def test_assistant_turn_nests_owned_activities_before_headerless_answer() -> None:
    """The mounted DOM, not a planner helper, proves the causal hierarchy."""
    app = MutableTranscriptHarness()
    orphan = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="legacy orphan",
        id="orphan-tool",
    )

    async with app.run_test(size=(100, 34)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([orphan, *_assistant_turn_messages()])
        await transcript.refresh_messages()
        await pilot.pause()

        turn = transcript.query_one(
            "#console-assistant-turn-a-turn", ConsoleAssistantTurnWidget
        )
        disclosures = list(turn.query(ConsoleActivityDisclosure))
        answer = turn.query_one("#console-message-a-turn")

        assert [row.activity_message_id for row in disclosures] == [
            "thinking-turn",
            "tool-turn",
        ]
        assert all(row.parent is turn.activity_stack for row in disclosures)
        assert turn.children.index(turn.activity_stack) < turn.children.index(answer)
        assert turn in transcript.children
        for activity_id in ("thinking-turn", "tool-turn"):
            nested_body = transcript.query_one(f"#console-message-{activity_id}")
            assert isinstance(nested_body.parent.parent, ConsoleActivityDisclosure)
        assert transcript.query_one("#console-message-orphan-tool")


@pytest.mark.asyncio
async def test_owned_activity_selection_keeps_details_collapsed_and_actions_visible() -> (
    None
):
    """Selecting an owned marker targets its header without selecting the answer."""
    app = MutableTranscriptHarness()

    async with app.run_test(size=(100, 34)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_assistant_turn_messages())
        await transcript.refresh_messages()
        await pilot.pause()

        transcript.select_message("tool-turn")
        await transcript.refresh_messages()
        await pilot.pause()
        header = transcript.query_one("#console-activity-header-tool-turn")
        disclosure = transcript.query_one(
            "#console-activity-disclosure-tool-turn", ConsoleActivityDisclosure
        )
        answer = transcript.query_one("#console-message-a-turn")

        assert transcript.selected_message_id == "tool-turn"
        assert header.has_class("console-activity-header-selected")
        assert not answer.has_class("console-transcript-message-selected")
        assert disclosure.action_stack.display
        assert not disclosure.detail_stack.display


@pytest.mark.asyncio
async def test_assistant_turn_reconcile_preserves_container_and_answer_identity() -> (
    None
):
    """Streaming and later activity arrival only reconcile mutable nested state."""
    app = MutableTranscriptHarness()
    user, assistant, thinking, tool = _assistant_turn_messages()
    unrelated = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="unrelated", id="u-unrelated"
    )

    async with app.run_test(size=(100, 34)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([user, assistant, unrelated])
        await transcript.refresh_messages()
        await pilot.pause()
        turn = transcript.query_one(
            "#console-assistant-turn-a-turn", ConsoleAssistantTurnWidget
        )
        answer = turn.query_one("#console-message-a-turn")
        activity_stack = turn.activity_stack
        unrelated_row = transcript.query_one("#console-message-u-unrelated")

        assistant.content = "The workspace contains"
        assistant.status = "streaming"
        transcript.set_messages([user, assistant, unrelated])
        await transcript.refresh_messages()
        await pilot.pause()
        assert transcript.query_one("#console-assistant-turn-a-turn") is turn
        assert turn.query_one("#console-message-a-turn") is answer

        assistant.content = "The workspace contains two files."
        assistant.status = "complete"
        transcript.set_messages([user, assistant, thinking, tool, unrelated])
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.query_one("#console-assistant-turn-a-turn") is turn
        assert turn.query_one("#console-message-a-turn") is answer
        assert turn.activity_stack is activity_stack
        assert transcript.query_one("#console-message-u-unrelated") is unrelated_row
        assert [
            row.activity_message_id for row in turn.query(ConsoleActivityDisclosure)
        ] == ["thinking-turn", "tool-turn"]


def test_assistant_turn_signature_ignores_unrelated_selection() -> None:
    """A selection in another unit is not render state for this Assistant turn."""
    transcript = ConsoleTranscript()
    messages = [
        *_assistant_turn_messages(),
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER, content="follow-up", id="other-user"
        ),
    ]
    transcript.set_messages(messages)
    before = next(
        row.signature
        for row in transcript._transcript_rows()
        if row.key == "assistant-turn:a-turn"
    )

    transcript.selected_message_id = "other-user"
    after = next(
        row.signature
        for row in transcript._transcript_rows()
        if row.key == "assistant-turn:a-turn"
    )

    assert after == before


@pytest.mark.asyncio
async def test_disjoint_session_switch_clears_owned_activity_expansion() -> None:
    """Disclosure expansion is session-local view state."""
    app = MutableTranscriptHarness()

    async with app.run_test(size=(100, 34)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_assistant_turn_messages())
        await transcript.refresh_messages()
        transcript.toggle_tool_output("tool-turn")
        await pilot.pause()
        assert "tool-turn" in transcript._expanded_tool_output_ids

        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="new", id="new-u"
                )
            ]
        )
        await transcript.refresh_messages()

        assert transcript._expanded_tool_output_ids == set()


@pytest.mark.asyncio
async def test_activity_header_click_and_o_share_expansion_state() -> None:
    """Disclosure activation selects the original id and uses the existing seam."""
    app = MutableTranscriptHarness()

    async with app.run_test(size=(100, 34)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_assistant_turn_messages())
        await transcript.refresh_messages()

        header = transcript.query_one("#console-activity-header-tool-turn")
        header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        assert transcript.selected_message_id == "tool-turn"
        assert "tool-turn" in transcript._expanded_tool_output_ids
        assert transcript.query_one(
            "#console-activity-disclosure-tool-turn", ConsoleActivityDisclosure
        ).detail_stack.display

        transcript.focus()
        await pilot.press("o")
        await pilot.pause()
        assert transcript.selected_message_id == "tool-turn"
        assert "tool-turn" not in transcript._expanded_tool_output_ids


@pytest.mark.asyncio
async def test_unknown_empty_activity_uses_neutral_nonexpandable_header() -> None:
    """Legacy metadata is not guessed from content and empty detail has no toggle."""
    app = MutableTranscriptHarness()
    messages = [
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
            id="neutral-answer",
        ),
        ConsoleChatMessage(role=ConsoleMessageRole.TOOL, content="", id="neutral-tool"),
    ]

    async with app.run_test(size=(100, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        header = transcript.query_one("#console-activity-header-neutral-tool")

        assert str(header.renderable) == "Activity · done"
        assert not header.has_class("console-activity-header-expandable")
        header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        assert transcript.selected_message_id == "neutral-tool"
        assert "neutral-tool" not in transcript._expanded_tool_output_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("detail_kind", ["full-output", "diff"])
async def test_empty_preview_activity_advertises_detail_and_all_toggles_agree(
    detail_kind: str,
) -> None:
    """Hidden full output and diffs remain discoverable with an empty preview."""
    app = MutableTranscriptHarness()
    tool_kwargs = (
        {"tool_output_full": "FULL-ONLY-DETAIL"}
        if detail_kind == "full-output"
        else {"tool_diff": ("file.txt", "before", "after")}
    )
    messages = [
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
            id=f"answer-{detail_kind}",
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.TOOL,
            content="",
            id=f"activity-{detail_kind}",
            activity_presentation=ConsoleActivityPresentation(
                "tool", f"literal [{detail_kind}]", "success"
            ),
            **tool_kwargs,
        ),
    ]
    activity_id = f"activity-{detail_kind}"

    async with app.run_test(size=(100, 28)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        header = transcript.query_one(f"#console-activity-header-{activity_id}")
        disclosure = transcript.query_one(
            f"#console-activity-disclosure-{activity_id}",
            ConsoleActivityDisclosure,
        )

        assert header.has_class("console-activity-header-expandable")
        assert header.renderable.plain.startswith("▸ literal [")
        assert not disclosure.detail_stack.display
        assert len(transcript.query(f"#console-message-{activity_id}")) == 0
        assert len(transcript.query(f"#console-tool-diff-{activity_id}")) == 0

        header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        assert activity_id in transcript._expanded_tool_output_ids
        expanded = transcript.query_one(
            f"#console-activity-disclosure-{activity_id}",
            ConsoleActivityDisclosure,
        )
        assert expanded.detail_stack.display
        if detail_kind == "full-output":
            assert "FULL-ONLY-DETAIL" in _message_row_text(transcript, activity_id)
        else:
            assert len(transcript.query(f"#console-tool-diff-{activity_id}")) == 1

        header = transcript.query_one(f"#console-activity-header-{activity_id}")
        header.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert activity_id not in transcript._expanded_tool_output_ids

        transcript.focus()
        await pilot.press("o")
        await pilot.pause()
        assert activity_id in transcript._expanded_tool_output_ids


@pytest.mark.asyncio
async def test_session_switch_cancels_finished_selection_in_nested_answer() -> None:
    """A detached Assistant answer cannot remain the selection-manager domain."""
    app = MutableTranscriptHarness()

    async with app.run_test(size=(100, 24)):
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_assistant_turn_messages())
        await transcript.refresh_messages()
        answer = transcript.query_one("#console-message-a-turn")
        transcript.selection_manager.begin_drag(answer.id, 0)
        transcript.selection_manager.extend_drag(answer.id, 3)
        assert transcript.selection_manager.finish_drag() is not None

        transcript.set_messages(
            [ConsoleChatMessage(role=ConsoleMessageRole.USER, content="new", id="new")]
        )
        await transcript.refresh_messages()

        assert not answer.is_attached
        assert transcript.selection_manager.state.selection is None


@pytest.mark.asyncio
async def test_activity_stack_insertion_preserves_finished_nested_selection() -> None:
    """Adding earlier thinking keeps the expanded Tool detail and selection."""
    app = MutableTranscriptHarness()
    user, assistant, thinking, tool = _assistant_turn_messages()

    async with app.run_test(size=(100, 28)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([user, assistant, tool])
        await transcript.refresh_messages()
        transcript.toggle_tool_output(tool.id)
        await pilot.pause()
        detail = transcript.query_one(f"#console-message-{tool.id}")
        turn = transcript.query_one(
            f"#console-assistant-turn-{assistant.id}", ConsoleAssistantTurnWidget
        )
        answer = turn.answer_widget
        transcript.selection_manager.begin_drag(detail.id, 0)
        transcript.selection_manager.extend_drag(detail.id, 4)
        assert transcript.selection_manager.finish_drag() is not None

        transcript.set_messages([user, assistant, thinking, tool])
        await transcript.refresh_messages()

        assert detail.is_attached
        assert transcript.query_one(f"#console-message-{tool.id}") is detail
        assert turn.answer_widget is answer
        assert transcript.selection_manager.state.selection is not None
        assert transcript.selection_manager.state.selection.row_key == detail.id


@pytest.mark.asyncio
@pytest.mark.parametrize("adjunct_kind", ["image", "citations"])
async def test_adjunct_only_activity_is_expandable_through_click_enter_and_o(
    adjunct_kind: str,
) -> None:
    """Derived activity adjuncts use the same disclosure truth as every toggle."""
    app = MutableTranscriptHarness()
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        id=f"adjunct-answer-{adjunct_kind}",
    )
    activity = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="",
        id=f"adjunct-activity-{adjunct_kind}",
        activity_presentation=ConsoleActivityPresentation(
            "activity", f"{adjunct_kind} adjunct", "done"
        ),
    )

    async with app.run_test(size=(100, 28)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([assistant, activity])
        if adjunct_kind == "image":
            transcript.set_image_specs(
                {activity.id: _image_row_spec(activity.id, "pixels")}
            )
        else:
            transcript.set_citation_counts({activity.id: 2})
        await transcript.refresh_messages()
        header = transcript.query_one(f"#console-activity-header-{activity.id}")
        disclosure = transcript.query_one(
            f"#console-activity-disclosure-{activity.id}",
            ConsoleActivityDisclosure,
        )

        assert header.has_class("console-activity-header-expandable")
        assert not disclosure.detail_stack.display

        header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        assert activity.id in transcript._expanded_tool_output_ids
        detail_selector = (
            f"#console-image-{activity.id}"
            if adjunct_kind == "image"
            else f"#console-citation-sources-{activity.id}"
        )
        assert len(transcript.query(detail_selector)) == 1

        header = transcript.query_one(f"#console-activity-header-{activity.id}")
        header.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert activity.id not in transcript._expanded_tool_output_ids

        transcript.focus()
        await pilot.press("o")
        await pilot.pause()
        assert activity.id in transcript._expanded_tool_output_ids


@pytest.mark.asyncio
async def test_activity_keyboard_toggle_preserves_header_identity_and_focus() -> None:
    """Enter/Space reconcile same-id disclosure state without detaching focus."""
    app = MutableTranscriptHarness()
    user, assistant, _thinking, tool = _assistant_turn_messages()

    async with app.run_test(size=(100, 28)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([user, assistant, tool])
        await transcript.refresh_messages()
        disclosure = transcript.query_one(
            f"#console-activity-disclosure-{tool.id}", ConsoleActivityDisclosure
        )
        header = transcript.query_one(f"#console-activity-header-{tool.id}")
        header.focus()
        await pilot.pause()
        assert header.has_focus

        await pilot.press("enter")
        await pilot.pause()
        assert transcript.query_one(f"#console-activity-header-{tool.id}") is header
        assert (
            transcript.query_one(f"#console-activity-disclosure-{tool.id}")
            is disclosure
        )
        assert header.has_focus
        assert tool.id in transcript._expanded_tool_output_ids

        await pilot.press("space")
        await pilot.pause()
        assert transcript.query_one(f"#console-activity-header-{tool.id}") is header
        assert (
            transcript.query_one(f"#console-activity-disclosure-{tool.id}")
            is disclosure
        )
        assert header.has_focus
        assert tool.id not in transcript._expanded_tool_output_ids


def test_session_identity_change_clears_same_id_activity_expansion() -> None:
    """A real session boundary clears disclosure state even for recycled ids."""
    transcript = ConsoleTranscript()
    _user, assistant, _thinking, activity = _assistant_turn_messages()
    messages = [assistant, activity]

    transcript.set_messages(messages, session_id="session-a")
    transcript.toggle_tool_output(activity.id)
    assert activity.id in transcript._expanded_tool_output_ids

    transcript.set_messages(messages, session_id="session-b")

    assert activity.id not in transcript._expanded_tool_output_ids


def test_same_session_update_preserves_activity_expansion() -> None:
    """Ordinary refreshes in one session keep the user's disclosure state."""
    transcript = ConsoleTranscript()
    _user, assistant, _thinking, activity = _assistant_turn_messages()
    messages = [assistant, activity]

    transcript.set_messages(messages, session_id="session-a")
    transcript.toggle_tool_output(activity.id)
    transcript.set_messages(messages, session_id="session-a")

    assert activity.id in transcript._expanded_tool_output_ids


def test_legacy_set_messages_preserves_same_id_activity_expansion() -> None:
    """Callers that omit session identity retain the existing id-based behavior."""
    transcript = ConsoleTranscript()
    _user, assistant, _thinking, activity = _assistant_turn_messages()
    messages = [assistant, activity]

    transcript.set_messages(messages)
    transcript.toggle_tool_output(activity.id)
    transcript.set_messages(messages)

    assert activity.id in transcript._expanded_tool_output_ids


class StyledRoleplayTranscriptHarness(ConsolidatedCSSApp):
    CSS_PATH = str(_BUNDLE)

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


class HeaderlessMessageHarness(App[None]):
    """Mount plain and Markdown rows in their Assistant-turn answer shape."""

    def compose(self) -> ComposeResult:
        yield ConsoleTranscriptMessage(
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="plain before",
                id="plain-headerless",
            ),
            show_header=False,
        )
        yield ConsoleMarkdownMessage(
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="markdown before",
                id="markdown-headerless",
                status="streaming",
            ),
            show_header=False,
        )


def test_standalone_message_widgets_keep_headers_by_default() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="default-header"
    )

    assert isinstance(
        next(iter(ConsoleTranscriptMessage(message).compose())), ConsoleMessageHeader
    )
    assert isinstance(
        next(iter(ConsoleMarkdownMessage(message).compose())), ConsoleMessageHeader
    )


@pytest.mark.asyncio
async def test_show_header_false_omits_plain_and_markdown_headers() -> None:
    app = HeaderlessMessageHarness()

    async with app.run_test():
        plain = app.query_one("#console-message-plain-headerless")
        markdown = app.query_one("#console-message-markdown-headerless")

        assert len(plain.query(ConsoleMessageHeader)) == 0
        assert len(markdown.query(ConsoleMessageHeader)) == 0


@pytest.mark.asyncio
async def test_headerless_plain_sync_still_updates_body() -> None:
    app = HeaderlessMessageHarness()

    async with app.run_test() as pilot:
        plain = app.query_one(
            "#console-message-plain-headerless", ConsoleTranscriptMessage
        )
        updated = ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="plain after",
            id="plain-headerless",
        )
        plain.sync_message(
            updated,
            resolve_console_message_presentation(updated, ConsolePresentationContext()),
        )
        await pilot.pause()

        body = plain.query_one(".console-transcript-message-body", Static)
        assert body.renderable.plain == "plain after"


@pytest.mark.asyncio
async def test_headerless_markdown_sync_updates_stream_and_footer() -> None:
    app = HeaderlessMessageHarness()

    async with app.run_test() as pilot:
        markdown_row = app.query_one(
            "#console-message-markdown-headerless", ConsoleMarkdownMessage
        )
        updated = ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="markdown before and after",
            id="markdown-headerless",
            status="streaming",
        )
        updated.attachments = (
            MessageAttachment(
                data=b"report",
                mime_type="text/plain",
                display_name="report.txt",
                position=0,
            ),
        )

        markdown_row.sync_message(
            updated,
            resolve_console_message_presentation(updated, ConsolePresentationContext()),
        )
        await pilot.pause()

        assert markdown_row.query_one(Markdown).source == "markdown before and after"
        footer = markdown_row.query_one(".console-markdown-footer", Static)
        assert footer.display
        assert "report.txt" in footer.renderable.plain


def test_roleplay_plain_text_uses_literal_names_and_generic_rows_gain_accents():
    transcript = ConsoleTranscript()
    transcript.set_presentation_context(_roleplay_context())
    transcript.set_messages(
        [
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="Hi", id="u1"),
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT, content="Hello", id="a1"
            ),
        ]
    )

    plain = transcript.to_plain_text(width=80)
    assert "Captain [Rowan]" in plain
    assert "Alraune" in plain
    assert "Assistant" not in plain

    transcript.set_presentation_context(
        ConsolePresentationContext(user_name="Builder", revision=2)
    )
    generic_rows = {
        row.id: row
        for row in _planned_message_widgets(transcript)
        if row.id is not None
    }
    assert "Builder" in transcript.to_plain_text(width=80)
    assert "Assistant" in transcript.to_plain_text(width=80)
    assert (
        "console-transcript-message-role-user"
        in generic_rows["console-message-u1"].classes
    )
    assert (
        "console-transcript-message-role-assistant"
        in generic_rows["console-message-a1"].classes
    )

    transcript.set_presentation_context(
        ConsolePresentationContext(
            user_name="Builder",
            revision=3,
            transcript_style=ConsoleTranscriptStyle.NEUTRAL,
        )
    )
    neutral_rows = [
        row for row in _planned_message_widgets(transcript) if row.id is not None
    ]
    assert all(
        "console-transcript-message-role-user" not in row.classes
        and "console-transcript-message-role-assistant" not in row.classes
        for row in neutral_rows
    )


def test_chat_screen_transcript_fingerprint_tracks_presentation_revision():
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    holder = {"context": _roleplay_context(revision=1)}
    screen = SimpleNamespace(
        _ensure_console_chat_store=lambda: SimpleNamespace(
            active_session_id="session-1"
        ),
        _console_presentation_context=lambda: holder["context"],
    )
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="same body", id="a1"
    )

    before = ChatScreen._native_console_transcript_fingerprint(screen, [message])
    holder["context"] = _roleplay_context(revision=2)
    after = ChatScreen._native_console_transcript_fingerprint(screen, [message])

    assert before != after
    assert before[1] == after[1]


@pytest.mark.asyncio
async def test_plain_roleplay_identity_revision_updates_rows_in_place():
    app = MutableTranscriptHarness()
    app.app_config = {"chat_defaults": {"assistant_markdown": False}}
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_presentation_context(_roleplay_context())
        transcript.set_messages(
            [
                ConsoleChatMessage(role=ConsoleMessageRole.USER, content="Hi", id="u1"),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT, content="Hello", id="a1"
                ),
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()

        original_user = transcript.query_one("#console-message-u1")
        original_assistant = transcript.query_one("#console-message-a1")
        user_label = _speaker_label_for(transcript, "u1")
        assistant_label = _speaker_label_for(transcript, "a1")
        assert user_label.renderable.plain == "Captain [Rowan]"
        assert assistant_label.renderable.plain == "Alraune"
        assert "console-transcript-roleplay-user-label" in user_label.classes
        assert "console-transcript-roleplay-character-label" in assistant_label.classes
        assert "console-transcript-message-roleplay-user" in original_user.classes
        assert (
            "console-transcript-message-roleplay-character"
            in original_assistant.classes
        )

        transcript.select_message("a1")
        await transcript.refresh_messages()
        await pilot.pause()
        original_follow_state = transcript.is_anchored
        transcript.set_presentation_context(
            _roleplay_context(
                user_name="Captain [bold red]", character_name="Cecelia", revision=2
            )
        )
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.query_one("#console-message-u1") is original_user
        assert transcript.query_one("#console-message-a1") is original_assistant
        assert transcript.selected_message_id == "a1"
        assert transcript.is_anchored is original_follow_state
        assert (
            _speaker_label_for(transcript, "u1").renderable.plain
            == "Captain [bold red]"
        )
        assert _speaker_label_for(transcript, "a1").renderable.plain == "Cecelia"


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_roleplay_tints_and_selected_precedence_are_compositor_painted(theme):
    app = StyledRoleplayTranscriptHarness()
    async with app.run_test(size=(90, 28)) as pilot:
        app.theme = theme
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_presentation_context(_roleplay_context())
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="user body", id="u1"
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="character body",
                    id="a1",
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.SYSTEM, content="neutral body", id="s1"
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="failed character body",
                    status="failed",
                    id="f1",
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.SYSTEM,
                    content="failed neutral body",
                    status="failed",
                    id="fs1",
                ),
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()

        user_row = transcript.query_one("#console-message-u1")
        character_row = transcript.query_one("#console-message-a1")
        system_row = transcript.query_one("#console-message-s1")
        failed_roleplay_row = transcript.query_one("#console-message-f1")
        failed_neutral_row = transcript.query_one("#console-message-fs1")
        neutral_background = _painted_background(app, system_row)
        assert _painted_background(app, user_row) != neutral_background
        assert _painted_background(app, character_row) != neutral_background
        assert _painted_background(app, failed_roleplay_row) == _painted_background(
            app, failed_neutral_row
        )
        assert _painted_background(app, failed_roleplay_row) != _painted_background(
            app, character_row
        )
        for speaker_kind, row in (
            ("user", user_row),
            ("character", character_row),
        ):
            label = _speaker_label_for(transcript, row.message_id)
            foreground, background = _painted_foreground_and_background(app, label)
            ratio = _contrast(foreground, background)
            assert ratio >= MIN_SPEAKER_CONTRAST, (
                f"{speaker_kind} speaker label contrast is {ratio:.2f}:1 under "
                f"{theme}; expected at least {MIN_SPEAKER_CONTRAST}:1 "
                f"(foreground={foreground}, background={background})"
            )
        failed_label = _speaker_label_for(transcript, "f1")
        foreground, background = _painted_foreground_and_background(app, failed_label)
        ratio = _contrast(foreground, background)
        assert ratio >= MIN_SPEAKER_CONTRAST, (
            f"failed speaker label contrast is {ratio:.2f}:1 under {theme}; "
            f"expected at least {MIN_SPEAKER_CONTRAST}:1 "
            f"(foreground={foreground}, background={background})"
        )

        transcript.select_message("f1")
        await transcript.refresh_messages()
        await pilot.pause()
        assert failed_roleplay_row.has_class("console-transcript-message-selected")
        assert not transcript.query_one(
            "#console-assistant-turn-f1", ConsoleAssistantTurnWidget
        ).header_widget.has_class("console-transcript-message-selected")

        transcript.select_message("u1")
        await transcript.refresh_messages()
        await pilot.pause()
        selected_roleplay_background = _painted_background(app, user_row)

        transcript.select_message("s1")
        await transcript.refresh_messages()
        await pilot.pause()
        assert selected_roleplay_background == _painted_background(app, system_row)
        painted_text = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in app.screen._compositor.render_strips()
        )
        assert "Captain [Rowan]" in painted_text
        assert "Alraune" in painted_text


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize(
    "transcript_style",
    [ConsoleTranscriptStyle.ROLE_ACCENTS, ConsoleTranscriptStyle.IMMERSIVE_RP],
)
async def test_generic_role_accents_and_immersive_prose_are_accessibly_painted(
    theme, transcript_style
):
    app = StyledRoleplayTranscriptHarness()
    app.app_config = {"chat_defaults": {"assistant_markdown": False}}
    async with app.run_test(size=(90, 28)) as pilot:
        app.theme = theme
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_presentation_context(
            ConsolePresentationContext(
                user_name="Builder",
                assistant_kind="generic",
                transcript_style=transcript_style,
            )
        )
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="user prose", id="gu1"
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="assistant prose",
                    id="ga1",
                ),
            ]
        )
        await transcript.refresh_messages()
        await pilot.pause()

        user_row = transcript.query_one("#console-message-gu1")
        assistant_row = transcript.query_one("#console-message-ga1")
        assert "console-transcript-message-role-user" in user_row.classes
        assert "console-transcript-message-role-assistant" in assistant_row.classes
        assert _painted_background(app, user_row) != _painted_background(
            app, assistant_row
        )

        painted_labels = []
        for row in (user_row, assistant_row):
            label = _speaker_label_for(transcript, row.message_id)
            foreground, background = _painted_foreground_and_background(app, label)
            assert _contrast(foreground, background) >= MIN_SPEAKER_CONTRAST
            painted_labels.append(foreground)
        assert painted_labels[0] != painted_labels[1]

        immersive_class = "console-transcript-message-immersive-"
        if transcript_style is ConsoleTranscriptStyle.IMMERSIVE_RP:
            assert any(name.startswith(immersive_class) for name in user_row.classes)
            assert any(
                name.startswith(immersive_class) for name in assistant_row.classes
            )
            for row, label_foreground in zip(
                (user_row, assistant_row), painted_labels, strict=True
            ):
                body = row.query_one(".console-transcript-message-body", Static)
                foreground, background = _painted_foreground_and_background(app, body)
                assert _contrast(foreground, background) >= MIN_SPEAKER_CONTRAST
                assert foreground == label_foreground
        else:
            assert not any(
                name.startswith(immersive_class)
                for row in (user_row, assistant_row)
                for name in row.classes
            )
            for row, label_foreground in zip(
                (user_row, assistant_row), painted_labels, strict=True
            ):
                body = row.query_one(".console-transcript-message-body", Static)
                foreground, _background = _painted_foreground_and_background(app, body)
                assert foreground != label_foreground


def _generation_message(*, variant_count: int, message_id: str = "gen-1"):
    """Build a message shaped like ``ConsoleChatStore.append_generation_message``'s output."""
    attachments = tuple(
        MessageAttachment(
            data=f"img{index}".encode(),
            mime_type="image/png",
            display_name="",
            position=index,
        )
        for index in range(variant_count)
    )
    meta = GenerationVariantMeta(
        prompt="a red dragon",
        negative_prompt="blurry",
        backend="swarmui",
        model=None,
        seed=42,
        style=None,
        params={},
    )
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[image] a red dragon",
        id=message_id,
    )
    message.attachments = attachments
    message.generation_metadata = tuple(meta for _ in range(variant_count))
    message.image_data = attachments[0].data
    message.image_mime_type = attachments[0].mime_type
    return message


class GenerationActionRowHarness(ConsolidatedCSSApp):
    """Mount one selected generation message, optionally pre-browsed.

    ``on_mount`` stamps ``_generation_browse`` directly onto ``self.screen``
    (the App's own default screen here, not a real ``ChatScreen`` -- Task 8's
    ``ConsoleTranscript._generation_browsed_index`` only ever reads the
    attribute via ``getattr``, so any screen-like object works) BEFORE
    selecting the message, so the very first action-row build already sees
    the browsed index.
    """

    def __init__(self, message: ConsoleChatMessage, *, browsed_index: int = 0) -> None:
        super().__init__()
        self._message = message
        self._browsed_index = browsed_index

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")

    def on_mount(self) -> None:
        transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([self._message])
        if self._browsed_index:
            self.screen._generation_browse = {self._message.id: self._browsed_index}
        transcript.set_generation_card_specs(
            {
                self._message.id: ConsoleGenerationCardSpec(
                    message_id=self._message.id,
                    browsed_index=self._browsed_index,
                    variant_count=len(self._message.generation_metadata),
                    meta=self._message.generation_metadata[self._browsed_index],
                    mode="pixels",
                )
            }
        )
        transcript.select_message(self._message.id)


class SpeakActionRowHarness(ConsolidatedCSSApp):
    """Mount one selected message, optionally marked as the Console TTS
    "speaking" message (task-559 unit 2).

    ``on_mount`` stamps ``_console_speaking_message_id`` directly onto
    ``self.screen`` (mirrors ``GenerationActionRowHarness``'s
    ``_generation_browse`` stamping above -- ``ConsoleTranscript`` only ever
    reads the attribute via ``getattr``, so any screen-like object works)
    BEFORE selecting the message, so the very first action-row build already
    reflects it.
    """

    def __init__(
        self, message: ConsoleChatMessage, *, speaking_message_id: str | None = None
    ) -> None:
        super().__init__()
        self._message = message
        self._speaking_message_id = speaking_message_id

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")

    def on_mount(self) -> None:
        transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([self._message])
        self.screen._console_speaking_message_id = self._speaking_message_id
        transcript.select_message(self._message.id)


class SaveAsModalHarness(ConsolidatedCSSApp):
    def __init__(
        self, destinations: list[ConsoleSaveDestination] | None = None
    ) -> None:
        super().__init__()
        self.destinations = (
            save_as_modal_destinations() if destinations is None else destinations
        )
        self.selected_destination: str | None = None

    def on_mount(self) -> None:
        self.push_screen(
            ConsoleSaveAsModal(destinations=self.destinations),
            self._capture_destination,
        )

    def _capture_destination(self, destination: str | None) -> None:
        self.selected_destination = destination


def save_as_modal_destinations() -> list[ConsoleSaveDestination]:
    return [
        ConsoleSaveDestination(label="Chatbook", available=True, reason=""),
        ConsoleSaveDestination(
            label="Note",
            available=False,
            reason="Notes service is not ready in this session.",
        ),
    ]


def test_console_transcript_enter_binding_describes_selection_toggle():
    """Keep the visible Enter hint aligned with its select-or-clear behavior."""
    enter_binding = next(
        binding for binding in ConsoleTranscript.BINDINGS if binding[0] == "enter"
    )

    assert enter_binding[2] == "Toggle message selection"


def test_console_transcript_renderable_uses_full_width_rules():
    transcript = ConsoleTranscript()
    transcript.set_messages(
        [
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello"),
            ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="world"),
        ]
    )

    plain = transcript.to_plain_text(width=40)

    assert "─" * 40 in plain
    assert "User" in plain
    assert "Assistant" in plain
    assert "hello" in plain
    assert "world" in plain


def test_console_transcript_widget_rules_carry_no_dash_text():
    """task-17658: separators paint via the stylesheet hatch, not text.

    The old fixed 200-dash renderable stopped ~4/5 across very wide
    terminals; the hatch spans any width (painted full-width contract in
    test_console_transcript_rule_spans_full_width_on_wide_terminals), so
    the widget itself must stay empty — dash text over the hatch would
    reintroduce a width-dependent seam in a different color layer.
    """
    transcript = ConsoleTranscript()
    transcript.set_messages(
        [
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello"),
        ]
    )

    first_rule = transcript._message_widgets()[0]
    renderable = getattr(first_rule, "renderable", "")

    assert str(renderable) == ""


@pytest.mark.asyncio
async def test_console_transcript_append_preserves_existing_message_rows():
    app = MutableTranscriptHarness()
    messages = [
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER, content=f"message {index}", id=f"m{index}"
        )
        for index in range(12)
    ]

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        before_counts = transcript.row_build_counts()

        transcript.set_messages(
            messages
            + [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT, content="new answer", id="m-new"
                )
            ]
        )
        await transcript.refresh_messages()
        after_counts = transcript.row_build_counts()

    for message in messages:
        assert (
            after_counts[f"message:{message.id}"]
            == before_counts[f"message:{message.id}"]
        )
    assert after_counts["assistant-turn:m-new"] == 1


@pytest.mark.asyncio
async def test_console_transcript_streaming_update_preserves_unrelated_message_rows():
    app = MutableTranscriptHarness()
    user = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="prompt", id="m-user"
    )
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial",
        id="m-assistant",
        status="streaming",
    )
    trailing = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="next", id="m-next"
    )

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([user, assistant, trailing])
        await transcript.refresh_messages()
        before_counts = transcript.row_build_counts()

        assistant.content = "partial response"
        transcript.set_messages([user, assistant, trailing])
        await transcript.refresh_messages()
        after_counts = transcript.row_build_counts()
        rendered_text = _message_row_text(transcript, "m-assistant")

    assert after_counts["message:m-user"] == before_counts["message:m-user"]
    assert after_counts["message:m-next"] == before_counts["message:m-next"]
    assert (
        after_counts["assistant-turn:m-assistant"]
        == before_counts["assistant-turn:m-assistant"]
    )
    assert "partial response" in rendered_text


@pytest.mark.asyncio
async def test_console_transcript_selection_update_preserves_message_rows():
    app = MutableTranscriptHarness()
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="prompt", id="m-user"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="answer", id="m-assistant"
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER, content="followup", id="m-followup"
        ),
    ]

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()

        transcript.selected_message_id = "m-user"
        await transcript.refresh_messages()
        before_counts = transcript.row_build_counts()

        transcript.selected_message_id = "m-assistant"
        await transcript.refresh_messages()
        after_counts = transcript.row_build_counts()
        assistant_actions_mounted = len(
            transcript.query("#console-message-actions-m-assistant")
        )

    for message in messages:
        key = (
            f"assistant-turn:{message.id}"
            if message.role is ConsoleMessageRole.ASSISTANT
            else f"message:{message.id}"
        )
        assert after_counts[key] == before_counts[key]
    assert "actions:m-user" not in transcript.row_render_signatures()
    assert assistant_actions_mounted == 1


@pytest.mark.asyncio
async def test_console_transcript_selection_onto_turn_file_card_does_not_rebuild():
    """Turn-file-card final-review fix: a card row's signature folds in
    ``selected`` exactly like every other "message" kind row (shared
    ``_message_row_signature``), so moving keyboard/click selection onto or
    off the card row DOES reach ``_update_row_widget``. Before the fix that
    method had no ``ConsoleTurnFileCard`` branch, so it fell through to a
    full rebuild -- collapsing any expanded diff and dropping the row's
    diff cache purely from a selection change. This pins: the SAME widget
    instance survives a selection round-trip, an expanded diff stays
    expanded, the ``-selected`` class toggles, and a header click selects
    the row (parity with ``ConsoleTranscriptMessage.on_click``).
    """
    from Tests.UI.test_console_turn_file_card import _FakeProvider
    from tldw_chatbook.Widgets.Console.console_turn_file_card import (
        ConsoleTurnFileCard,
    )

    card_message = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="✎ Edited 2 files  +8 −3 — review with `v`",
        id="m-card",
        change_review_run_id="run-1",
    )
    other_message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="hello", id="m-other"
    )

    app = MutableTranscriptHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_change_review_provider_factory(lambda: _FakeProvider())
        transcript.set_messages([card_message, other_message])
        await transcript.refresh_messages()

        card = transcript.query_one(
            f"#console-turn-file-card-{card_message.id}", ConsoleTurnFileCard
        )
        for _ in range(60):
            if card.query(".console-turn-file-row"):
                break
            await pilot.pause(0.02)
        assert card.query(".console-turn-file-row"), "card rows never loaded"
        assert not card.has_class("console-turn-file-card-selected")

        # Header click selects the row -- checked first, while layout is
        # still simple (a later selection round-trip mounts an action row
        # for the other message and can scroll the card off-screen, which
        # would make a click target here flaky for reasons unrelated to
        # what this assertion pins). Tail-follow scrolls to the newest
        # message on mount, which can carry the card above the fold; force
        # the scroll position back to the top before targeting it.
        transcript.scroll_home(animate=False)
        await pilot.pause()
        await pilot.click(".console-turn-file-header")
        await pilot.pause()
        assert transcript.selected_message_id == card_message.id
        transcript.selected_message_id = None
        await transcript.refresh_messages()

        # Expand the first row's diff -- real state that a rebuild would
        # lose. Poking the widget's own post-expand state directly (rather
        # than driving a click/keypress through the row Button) keeps this
        # test independent of Button focus/binding resolution inside a real
        # ConsoleTranscript, which is unrelated to what this test pins; the
        # expand mechanism itself is already covered by
        # Tests/UI/test_console_turn_file_card.py.
        from tldw_chatbook.Chat.console_display_state import DiffHunk

        body = card.query(".console-turn-file-diff").first()
        body.display = True
        sentinel_hunks = [
            DiffHunk(header="", body_lines=("SENTINEL_DIFF_TEXT",), file_prelude="")
        ]
        card._hunk_cache[0] = sentinel_hunks

        # Move selection ONTO the card row (same direct-state + refresh
        # pattern this file's other rebuild-avoidance tests use above).
        transcript.selected_message_id = card_message.id
        await transcript.refresh_messages()
        card_after_select = transcript.query_one(
            f"#console-turn-file-card-{card_message.id}", ConsoleTurnFileCard
        )
        assert card_after_select is card, "selecting the card row rebuilt the widget"
        assert card.has_class("console-turn-file-card-selected")
        assert card.query(".console-turn-file-diff").first().display, (
            "expanded diff collapsed on selection"
        )
        assert card._hunk_cache.get(0) == sentinel_hunks, (
            "diff cache dropped on selection"
        )

        # Move selection OFF the card row.
        transcript.selected_message_id = other_message.id
        await transcript.refresh_messages()
        card_after_deselect = transcript.query_one(
            f"#console-turn-file-card-{card_message.id}", ConsoleTurnFileCard
        )
        assert card_after_deselect is card, (
            "deselecting the card row rebuilt the widget"
        )
        assert not card.has_class("console-turn-file-card-selected")
        assert card.query(".console-turn-file-diff").first().display, (
            "expanded diff lost on deselection"
        )
        assert card._hunk_cache.get(0) == sentinel_hunks, (
            "diff cache dropped on deselection"
        )


@pytest.mark.asyncio
async def test_console_transcript_removes_build_counts_for_stale_rows():
    app = MutableTranscriptHarness()
    removed = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="remove me", id="m-removed"
    )
    kept = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="keep me", id="m-kept"
    )

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([removed, kept])
        await transcript.refresh_messages()
        assert "message:m-removed" in transcript.row_build_counts()

        transcript.set_messages([kept])
        await transcript.refresh_messages()
        build_counts = transcript.row_build_counts()

    assert "rule:m-removed" not in build_counts
    assert "message:m-removed" not in build_counts
    assert "assistant-turn:m-kept" in build_counts


def test_console_transcript_compose_resets_build_count_bookkeeping():
    transcript = ConsoleTranscript()
    transcript._row_build_counts["message:stale"] = 3
    transcript.set_messages(
        [
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER,
                content="current",
                id="m-current",
            )
        ]
    )

    list(transcript.compose())

    build_counts = transcript.row_build_counts()
    assert "message:stale" not in build_counts
    assert build_counts["message:m-current"] == 1


@pytest.mark.asyncio
async def test_console_transcript_empty_state_accepts_setup_copy():
    app = EmptyTranscriptHarness()

    async with app.run_test() as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)

        transcript.sync_empty_state(
            ConsoleSetupCardState(
                mode="ready_line",
                body_copy="Choose a model in Console Settings to start chatting.",
            )
        )
        await pilot.pause()

        empty_state = transcript.query_one(".console-transcript-empty-state", Static)
        empty_text = getattr(
            empty_state.renderable, "plain", str(empty_state.renderable)
        )
        assert empty_text == "Choose a model in Console Settings to start chatting."


def test_console_transcript_selected_message_shows_action_row():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    plain = transcript.to_plain_text(width=80)

    assert "Copy 🔊 Edit Fork ♻ ---> More…" in plain
    assert "|" not in plain


def test_console_user_message_regenerate_action_is_disabled_and_blocks_dispatch():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="prompt", id="m1"
    )
    service = ConsoleMessageActionService()

    regenerate = next(
        action
        for action in service.available_actions(message)
        if action.action_id == "regenerate"
    )
    result = service.dispatch("regenerate", message)

    assert not regenerate.enabled
    assert regenerate.disabled_reason == "Only assistant messages can be regenerated."
    assert result.status == "blocked"
    assert result.visible_copy == "Only assistant messages can be regenerated."


def test_console_transcript_selected_message_does_not_apply_inline_border_geometry():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )
    widget = ConsoleTranscriptMessage(message, selected=True)

    assert widget.has_class("console-transcript-message-selected")
    assert "solid" not in repr(widget.styles.border)

    widget.sync_message(message, selected=False)

    assert not widget.has_class("console-transcript-message-selected")
    assert "solid" not in repr(widget.styles.border)

    widget.sync_message(message, selected=True)

    assert widget.has_class("console-transcript-message-selected")
    assert "solid" not in repr(widget.styles.border)


def test_console_transcript_action_row_stays_within_terminal_width_budget():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    action_row = next(
        line
        for line in transcript.to_plain_text(width=48).splitlines()
        if line.startswith("Copy")
    )

    assert action_row == "Copy 🔊 Edit Fork ♻ ---> More…"
    assert len(action_row) <= 48


def test_console_transcript_selected_message_explains_icon_actions():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    rendered = transcript.to_plain_text(width=80)

    assert "Copy 🔊 Edit Fork ♻ ---> More…" in rendered
    # TASK-362 AC#2: the guide names the single-key shortcuts, not just icons.
    assert "Guide:" in rendered
    assert (
        "c Copy" in rendered and "e Edit" in rendered and "r ♻ Regenerate" in rendered
    )
    assert "j/k select" in rendered
    # task-2154.14 (DS-01): the guide also names each glyph-only button in
    # words, derived from the row's own actions, so the meaning is on screen
    # instead of behind a tooltip.
    assert "🔊 Speak" in rendered
    assert "---> Continue" in rendered
    assert "More…" in rendered


def test_console_transcript_variant_navigation_changes_displayed_content():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
        id="m1",
        # TASK-7: the `<`/`>` action row is now gated on sibling_count (a
        # forked-branch read model), not on `.variants` -- this message
        # still carries a `ConsoleVariantSet` below purely to exercise the
        # (retired, test-only) in-message content cycling the assertions
        # check; the action row's own visibility needs sibling_count > 1.
        sibling_count=2,
    )
    message.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    assert "first" in transcript.to_plain_text(width=80)

    transcript.select_next_variant("m1")

    rendered = transcript.to_plain_text(width=80)
    assert "second" in rendered
    assert "first" not in rendered
    assert "More…" in rendered


def test_console_transcript_variant_action_row_stays_within_terminal_width_budget():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
        id="m1",
        sibling_count=2,
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    action_row = next(
        line
        for line in transcript.to_plain_text(width=48).splitlines()
        if line.startswith("Copy")
    )

    assert action_row == "Copy 🔊 Edit < > Fork ♻ ---> More…"
    assert len(action_row) <= 48


def test_console_transcript_failed_action_row_includes_retry_without_exceeding_budget():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="failed",
        id="m1",
        status="failed",
    )
    transcript = ConsoleTranscript()
    transcript.set_messages([message])
    transcript.select_message("m1")

    action_row = next(
        line
        for line in transcript.to_plain_text(width=48).splitlines()
        if line.startswith("Copy")
    )

    assert action_row == "Copy Edit Fork Retry ---> More…"
    assert len(action_row) <= 48


@pytest.mark.asyncio
async def test_console_transcript_enter_selects_first_message_when_none_selected():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()

        await pilot.press("enter")
        await pilot.pause()

        assert transcript.selected_message_id == "m1"
        assert "More…" in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_enter_clears_keyboard_selected_message():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()

        await pilot.press("down")
        await pilot.pause()
        assert transcript.selected_message_id == "m1"
        assert "More…" in _visible_text(app)

        await pilot.press("enter")
        await pilot.pause()

        assert transcript.selected_message_id is None
        assert "More…" not in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_boundary_navigation_keeps_last_message_selected():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()

        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()

        assert transcript.selected_message_id == "m2"
        assert "More…" in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_enter_on_action_button_preserves_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        await pilot.click("#console-message-m2")
        transcript.focus_action("m2", "copy")
        await pilot.pause()

        button = app.query_one("#console-message-action-copy-m2", Button)
        assert button.has_focus

        await pilot.press("enter")
        await pilot.pause()

        assert transcript.selected_message_id == "m2"
        assert "More…" in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_click_selects_message_and_shows_actions():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        text = _visible_text(app)

    assert "Copy" in text
    assert "Fork" in text
    assert "♻" in text
    assert "More…" in text
    assert "Guide:" in text and "r ♻ Regenerate" in text
    # task-2154.14 (DS-01): the legend names the glyph-only buttons in words.
    assert "🔊 Speak" in text and "f Fork" in text
    assert "|" not in text


@pytest.mark.asyncio
async def test_console_transcript_more_menu_captures_message_and_closes_before_choice():
    app = TranscriptHarness(css_path=str(_BUNDLE))

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        await pilot.click("#console-message-m2")
        await pilot.click("#console-message-action-more-m2")
        await pilot.pause()

        menu = app.query_one("#console-message-more-menu")
        assert menu.message_id == "m2"
        assert [button.label.plain for button in menu.query(Button)] == [
            "Save as…",
            "Helpful",
            "Not helpful",
            "Delete",
        ]

        await pilot.press("down", "enter")
        await pilot.pause()

        assert not app.query("#console-message-more-menu")
        assert transcript.selected_message_id == "m2"


@pytest.mark.parametrize(
    ("dismissal", "expected_selection"),
    (
        ("selection-change", "m1"),
        ("row-recompose", "m2"),
        ("row-removal", None),
        ("refresh", "m2"),
        ("escape", "m2"),
    ),
)
@pytest.mark.asyncio
async def test_console_more_menu_lifecycle_dismisses_without_dispatch(
    monkeypatch, dismissal, expected_selection
):
    app = TranscriptHarness(css_path=str(_BUNDLE))

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await pilot.click("#console-message-m2")
        await pilot.click("#console-message-action-more-m2")
        await _wait_for_selector(app, pilot, "#console-message-more-menu")
        dispatched = []
        monkeypatch.setattr(
            transcript,
            "dispatch_captured_message_action",
            lambda *args, **kwargs: dispatched.append((args, kwargs)),
        )

        if dismissal == "selection-change":
            transcript.select_message("m1")
            await pilot.pause()
        elif dismissal == "row-recompose":
            transcript._messages[1].content = "recomposed answer"
            transcript.set_messages(transcript._messages)
            await transcript.refresh_messages()
        elif dismissal == "row-removal":
            transcript.set_messages([transcript._messages[0]])
            await transcript.refresh_messages()
        elif dismissal == "refresh":
            await transcript.refresh_messages()
        else:
            await pilot.press("escape")
            await pilot.pause()

        assert not app.query("#console-message-more-menu")
        assert transcript.selected_message_id == expected_selection
        assert dispatched == []
        if dismissal == "escape":
            assert app.query_one("#console-message-action-more-m2").has_focus
        elif dismissal == "selection-change":
            assert transcript.has_focus
            assert transcript.selected_message_id == "m1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("click_target", "expected_selection"),
    (
        ("#console-message-action-copy-m2", "m2"),
        ("#console-message-m2", None),
    ),
)
async def test_console_more_menu_closes_on_in_transcript_click_without_dispatch(
    monkeypatch, click_target, expected_selection
):
    app = TranscriptHarness(css_path=str(_BUNDLE))

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await pilot.click("#console-message-m2")
        await pilot.click("#console-message-action-more-m2")
        await _wait_for_selector(app, pilot, "#console-message-more-menu")
        dispatched = []
        monkeypatch.setattr(
            transcript,
            "dispatch_captured_message_action",
            lambda *args, **kwargs: dispatched.append((args, kwargs)),
        )

        await pilot.click(click_target)
        await pilot.pause()

        assert not app.query("#console-message-more-menu")
        assert transcript.selected_message_id == expected_selection
        assert dispatched == []


@pytest.mark.asyncio
async def test_console_real_recompose_closes_more_and_restores_opener(monkeypatch):
    app = TranscriptHarness(css_path=str(_BUNDLE))

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await pilot.click("#console-message-m2")
        await pilot.click("#console-message-action-more-m2")
        await _wait_for_selector(app, pilot, "#console-message-more-menu")
        dispatched = []
        monkeypatch.setattr(
            transcript,
            "dispatch_captured_message_action",
            lambda *args, **kwargs: dispatched.append((args, kwargs)),
        )

        transcript.refresh(recompose=True)
        await pilot.pause()

        assert not app.query("#console-message-more-menu")
        assert transcript.selected_message_id == "m2"
        assert app.query_one("#console-message-action-more-m2").has_focus
        assert dispatched == []


@pytest.mark.asyncio
async def test_console_more_choice_keeps_captured_target_after_selection_race(
    monkeypatch,
):
    app = TranscriptHarness(css_path=str(_BUNDLE))

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await pilot.click("#console-message-m2")
        await pilot.click("#console-message-action-more-m2")
        await _wait_for_selector(app, pilot, "#console-message-more-feedback-up")
        captured = []

        def dispatch(message_id, action_id, *, opener_button_id):
            captured.append(
                (
                    message_id,
                    action_id,
                    opener_button_id,
                    bool(app.query("#console-message-more-menu")),
                    transcript.selected_message_id,
                )
            )

        monkeypatch.setattr(transcript, "dispatch_captured_message_action", dispatch)
        choice = app.query_one("#console-message-more-feedback-up", Button)
        choice.press()
        transcript.selected_message_id = "m1"
        await pilot.pause()

        assert captured == [
            ("m2", "feedback-up", "console-message-action-more-m2", False, "m1")
        ]


@pytest.mark.asyncio
async def test_console_more_keyboard_traversal_skips_disabled_and_restores_opener():
    app = TranscriptHarness(css_path=str(_BUNDLE))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.click("#console-message-m2")
        await pilot.click("#console-message-action-more-m2")
        await _wait_for_selector(app, pilot, "#console-message-more-menu")

        save = app.query_one("#console-message-more-save-as", Button)
        helpful = app.query_one("#console-message-more-feedback-up", Button)
        not_helpful = app.query_one("#console-message-more-feedback-down", Button)
        save.disabled = True
        helpful.focus(scroll_visible=False)
        await pilot.pause()
        assert save.disabled
        assert helpful.has_focus
        await pilot.press("down")
        assert not_helpful.has_focus
        await pilot.press("up")
        assert helpful.has_focus
        await pilot.press("escape")
        await pilot.pause()

        assert not app.query("#console-message-more-menu")
        assert app.query_one("#console-message-action-more-m2").has_focus


@pytest.mark.asyncio
async def test_console_more_focus_falls_back_to_composer_after_row_removal():
    class ComposerFallbackHarness(ConsolidatedCSSApp):
        CSS_PATH = str(_BUNDLE)

        def compose(self) -> ComposeResult:
            transcript = ConsoleTranscript(id="console-native-transcript")
            transcript.set_messages(
                [
                    ConsoleChatMessage(
                        role=ConsoleMessageRole.ASSISTANT,
                        content="answer",
                        id="removed-message",
                    )
                ]
            )
            yield transcript
            yield Button("Composer", id="console-native-composer")

    app = ComposerFallbackHarness()
    async with app.run_test(size=(100, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.select_message("removed-message")
        await transcript.refresh_messages()
        opener = app.query_one("#console-message-action-more-removed-message", Button)
        await transcript._open_message_more_menu("removed-message", opener)
        await _wait_for_selector(app, pilot, "#console-message-more-menu")

        transcript.set_messages([])
        await transcript.refresh_messages()
        await pilot.pause()

        assert not app.query("#console-message-more-menu")
        assert transcript.selected_message_id is None
        assert app.query_one("#console-native-composer").has_focus


@pytest.mark.parametrize("size", ((120, 35), (100, 30), (80, 24)))
@pytest.mark.parametrize(
    ("message", "expected_actions"),
    (
        (
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER, content="question", id="row-user"
            ),
            ("copy", "edit", "fork", "regenerate", "continue", "more"),
        ),
        (
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="answer",
                id="row-assistant",
            ),
            ("copy", "speak", "edit", "fork", "regenerate", "continue", "more"),
        ),
        (
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="stopped",
                status="stopped",
                id="row-stopped",
            ),
            ("copy", "edit", "fork", "regenerate", "continue", "more"),
        ),
        (
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="failed",
                status="failed",
                id="row-failed",
            ),
            ("copy", "edit", "fork", "retry", "continue", "more"),
        ),
        (
            ConsoleChatMessage(
                role=ConsoleMessageRole.TOOL,
                content="tool preview",
                tool_output_full="full output",
                id="row-tool",
            ),
            ("tool-output",),
        ),
    ),
)
@pytest.mark.asyncio
async def test_console_selected_action_rows_fit_reference_terminals(
    size, message, expected_actions
):
    app = MutableTranscriptHarness(css_path=str(_BUNDLE))

    async with app.run_test(size=size) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([message])
        transcript.select_message(message.id)
        await transcript.refresh_messages()
        await pilot.pause()
        buttons = list(transcript.query(".console-transcript-action-button"))

        assert tuple(button.console_action_id for button in buttons) == expected_actions
        assert transcript.max_scroll_x == 0
        assert all(
            0 <= button.region.x < button.region.right <= size[0] for button in buttons
        )


@pytest.mark.parametrize("size", ((120, 35), (100, 30), (80, 24)))
@pytest.mark.parametrize("media_kind", ("generated-image", "video"))
@pytest.mark.asyncio
async def test_console_media_card_actions_fit_reference_terminals(size, media_kind):
    app = MutableTranscriptHarness(css_path=str(_BUNDLE))

    async with app.run_test(size=size) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        if media_kind == "generated-image":
            message = _generation_message(variant_count=3, message_id="fit-image")
            app.screen._generation_browse = {message.id: 1}
            transcript.set_generation_card_specs(
                {
                    message.id: ConsoleGenerationCardSpec(
                        message_id=message.id,
                        browsed_index=1,
                        variant_count=3,
                        meta=message.generation_metadata[1],
                        mode="pixels",
                    )
                }
            )
            card_selector = f"#console-generation-card-{message.id}"
            expected_actions = (
                "variant-previous",
                "variant-next",
                "keep",
                "toggle-image-view",
                "save-image",
            )
        else:
            meta = VideoGenerationMetadata(
                name="fit-video",
                prompt="a red dragon",
                backend="minimax",
            )
            message = ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="[video] a red dragon",
                video_metadata=meta,
                id="fit-video",
            )
            transcript.set_video_card_specs(
                {
                    message.id: ConsoleVideoCardSpec(
                        message_id=message.id,
                        meta=meta,
                        status="ready",
                        file_path="/tmp/fit-video.mp4",
                    )
                }
            )
            card_selector = f"#console-video-card-{message.id}"
            expected_actions = ("video-play", "video-save-copy")

        transcript.set_messages([message])
        transcript.select_message(message.id)
        await transcript.refresh_messages()
        await pilot.pause()

        card = app.query_one(card_selector)
        action_row = card.query_one(".console-media-card-actions")
        buttons = list(card.query(".console-media-card-action"))
        assert tuple(button.console_action_id for button in buttons) == expected_actions
        if media_kind == "generated-image":
            assert tuple(button.label.plain for button in buttons) == (
                "<",
                ">",
                "Keep",
                "View",
                "Save",
            )
        assert transcript.max_scroll_x == 0
        assert action_row.region.height == 1
        assert all(button.region.height == 1 for button in buttons)
        assert all(
            card.region.x <= button.region.x < button.region.right <= card.region.right
            for button in buttons
        )
        assert all(
            button.content_region.width >= len(button.label.plain) for button in buttons
        )


@pytest.mark.asyncio
async def test_console_ineligible_fork_reason_is_visible_and_repeated_by_f(
    monkeypatch,
):
    reason = "Save this chat before forking from the selected message."
    app = TranscriptHarness(css_path=str(_BUNDLE))

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.select_message("m2")
        transcript.set_fork_eligibilities({"m2": ConsoleForkEligibility(False, reason)})
        await transcript.refresh_messages()
        await pilot.pause()
        fork = app.query_one("#console-message-action-fork-m2", Button)
        notices = []
        monkeypatch.setattr(
            transcript,
            "notify",
            lambda message, **kwargs: notices.append((message, kwargs)),
        )
        transcript.focus()
        await pilot.press("f")
        await pilot.pause()

        assert fork.disabled
        assert str(fork.tooltip) == reason
        assert f"Fork unavailable — {reason}" in _visible_text(app)
        assert notices == [(reason, {"severity": "warning"})]


@pytest.mark.asyncio
async def test_console_transcript_click_selected_message_clears_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)

        await pilot.click("#console-message-m2")
        await pilot.pause()
        assert transcript.selected_message_id == "m2"
        assert "More…" in _visible_text(app)

        await pilot.click("#console-message-m2")
        await pilot.pause()

        assert transcript.selected_message_id is None
        assert "More…" not in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_click_background_clears_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        assert "More…" in _visible_text(app)

        # Click empty space below the rendered messages.
        await pilot.click("#console-native-transcript", offset=(5, 20))
        await pilot.pause()

        assert "More…" not in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_click_action_button_preserves_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        assert "More…" in _visible_text(app)

        await pilot.click("#console-message-action-copy-m2")
        await pilot.pause()

        assert "More…" in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_click_rule_separator_preserves_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        assert "More…" in _visible_text(app)

        await pilot.click(".console-transcript-rule")
        await pilot.pause()

        assert "More…" in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_rule_spans_full_width_on_wide_terminals():
    """task-17658: message separators reach edge to edge at any width.

    The rule used to be a fixed 200-dash string, stopping ~4/5 of the way
    across very wide terminals; the hatch fill spans whatever width the
    transcript actually has.
    """

    class _BundledTranscriptHarness(TranscriptHarness):
        CSS_PATH = str(_BUNDLE)

    app = _BundledTranscriptHarness()
    async with app.run_test(size=(250, 20)) as pilot:
        await pilot.pause()
        rule = app.query(".console-transcript-rule").first()
        strips = app.screen._compositor.render_strips()
        row = "".join(seg.text for seg in strips[rule.region.y])
        painted = row[rule.region.x : rule.region.x + rule.region.width]
        assert set(painted) == {"─"}, repr(painted[:24] + "…" + painted[-12:])


@pytest.mark.asyncio
async def test_console_transcript_click_scrollbar_does_not_clear_selection():
    app = MutableTranscriptHarness()
    messages = [
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER, content=f"message {index}", id=f"m{index}"
        )
        for index in range(30)
    ]

    async with app.run_test(size=(40, 12)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()

        transcript.select_message("m15")
        await pilot.pause()
        assert transcript.selected_message_id == "m15"

        scrollbar = transcript.vertical_scrollbar
        await pilot.click(scrollbar)
        await pilot.pause()

        assert transcript.selected_message_id == "m15"


@pytest.mark.asyncio
async def test_console_transcript_click_action_row_background_preserves_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await pilot.pause()
        assert "More…" in _visible_text(app)

        # Route a container-background click without stale screen coordinates.
        action_row = app.query_one("#console-message-actions-m2")
        transcript = app.query_one(ConsoleTranscript)
        await transcript.on_click(
            SimpleNamespace(control=action_row, stop=lambda: None)
        )
        await pilot.pause()

        assert "More…" in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_click_action_help_preserves_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await pilot.pause()
        assert "More…" in _visible_text(app)

        await pilot.click(".console-transcript-action-guide")
        await pilot.pause()

        assert "More…" in _visible_text(app)


@pytest.mark.asyncio
async def test_console_transcript_click_empty_state_panel_preserves_selection():
    app = EmptyTranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.selected_message_id is None

        await pilot.click(".console-transcript-empty-panel")
        await pilot.pause()

        assert transcript.selected_message_id is None


@pytest.mark.asyncio
async def test_console_selected_message_uses_class_without_inline_frame():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await _wait_for_selector(app, pilot, "#console-message-m2")
        selected = app.query_one("#console-message-m2")

    assert "console-transcript-message-selected" in selected.classes
    assert "solid" not in repr(selected.styles.border)


@pytest.mark.asyncio
async def test_console_transcript_action_buttons_have_stable_ids():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-copy-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-fork-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-more-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-regenerate-m2")
        text = _visible_text(app)

    assert "Copy" in text
    assert "Fork" in text
    assert "♻" in text
    assert "More…" in text
    assert "|" not in text


@pytest.mark.asyncio
async def test_console_transcript_generation_message_action_row_hides_keep_at_browsed_zero():
    """A generation message's mounted action row: `<`/`>` visible+gated by
    the GENERATION browsed index, "keep" absent while browsed at 0 --
    proves the transcript's real `self.screen`-sourced wiring, not just the
    pure action-service gating already covered elsewhere."""
    message = _generation_message(variant_count=3)
    app = GenerationActionRowHarness(message, browsed_index=0)

    async with app.run_test(size=(100, 32)) as pilot:
        await _wait_for_selector(
            app, pilot, f"#console-message-action-variant-previous-{message.id}"
        )
        previous_button = app.query_one(
            f"#console-message-action-variant-previous-{message.id}"
        )
        next_button = app.query_one(
            f"#console-message-action-variant-next-{message.id}"
        )
        keep_buttons = app.query(f"#console-message-action-keep-{message.id}")

    assert previous_button.disabled is True  # browsed index 0 -- no "previous"
    assert next_button.disabled is False
    assert len(keep_buttons) == 0


@pytest.mark.asyncio
async def test_console_transcript_generation_message_action_row_shows_keep_when_browsed():
    """Browsed away from the canonical (position 0) variant: "keep" appears,
    and `<`/`>` enable state reflects the NEW browsed index."""
    message = _generation_message(variant_count=3)
    app = GenerationActionRowHarness(message, browsed_index=2)

    async with app.run_test(size=(100, 32)) as pilot:
        await _wait_for_selector(
            app, pilot, f"#console-message-action-keep-{message.id}"
        )
        previous_button = app.query_one(
            f"#console-message-action-variant-previous-{message.id}"
        )
        next_button = app.query_one(
            f"#console-message-action-variant-next-{message.id}"
        )

    assert previous_button.disabled is False
    assert next_button.disabled is True  # browsed index 2 == last of 3


@pytest.mark.asyncio
async def test_console_transcript_single_variant_generation_message_hides_nav_and_keep():
    message = _generation_message(variant_count=1)
    app = GenerationActionRowHarness(message, browsed_index=0)

    async with app.run_test(size=(100, 32)) as pilot:
        await _wait_for_selector(
            app, pilot, f"#console-message-action-regenerate-{message.id}"
        )
        nav_buttons = app.query(
            f"#console-message-action-variant-previous-{message.id}"
        )
        keep_buttons = app.query(f"#console-message-action-keep-{message.id}")

    assert len(nav_buttons) == 0
    assert len(keep_buttons) == 0


# --- task-559 unit 2: Console TTS stop toggle in the mounted action row ---


@pytest.mark.asyncio
async def test_console_transcript_action_row_shows_speak_when_not_speaking():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )
    app = SpeakActionRowHarness(message, speaking_message_id=None)

    async with app.run_test(size=(100, 32)) as pilot:
        await _wait_for_selector(app, pilot, "#console-message-action-speak-m1")
        stop_buttons = app.query("#console-message-action-speak-stop-m1")

    assert len(stop_buttons) == 0


@pytest.mark.asyncio
async def test_console_transcript_action_row_swaps_to_stop_for_speaking_message():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )
    app = SpeakActionRowHarness(message, speaking_message_id="m1")

    async with app.run_test(size=(100, 32)) as pilot:
        await _wait_for_selector(app, pilot, "#console-message-action-speak-stop-m1")
        speak_buttons = app.query("#console-message-action-speak-m1")
        stop_button = app.query_one("#console-message-action-speak-stop-m1")

    assert len(speak_buttons) == 0
    assert str(stop_button.label) == "⏹"
    assert stop_button.disabled is False


@pytest.mark.asyncio
async def test_console_transcript_action_row_unaffected_by_other_message_speaking():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )
    app = SpeakActionRowHarness(message, speaking_message_id="some-other-message")

    async with app.run_test(size=(100, 32)) as pilot:
        await _wait_for_selector(app, pilot, "#console-message-action-speak-m1")
        stop_buttons = app.query("#console-message-action-speak-stop-m1")

    assert len(stop_buttons) == 0


@pytest.mark.asyncio
async def test_console_transcript_action_tooltips_explain_compact_labels():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        await _wait_for_selector(app, pilot, "#console-message-action-continue-m2")
        continue_action = app.query_one("#console-message-action-continue-m2")
        more_action = app.query_one("#console-message-action-more-m2")

    assert "extend" in str(continue_action.tooltip).lower()
    assert "more message" in str(more_action.tooltip).lower()


@pytest.mark.asyncio
async def test_console_transcript_escape_collapses_selected_action_row():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.click("#console-message-m2")
        assert "More…" in _visible_text(app)

        await pilot.press("escape")

        assert "More…" not in _visible_text(app)


@pytest.mark.asyncio
async def test_save_as_modal_lists_available_and_unavailable_destinations():
    app = SaveAsModalHarness()

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.1)
        await _wait_for_selector(app.screen, pilot, "#console-save-as-modal")
        text = _visible_text(app.screen)

    assert "Chatbook" in text
    assert "Note (unavailable)" in text
    assert "Notes service is not ready in this session." in text
    assert "WIP" not in text


@pytest.mark.asyncio
async def test_save_as_modal_available_destination_is_clickable_control():
    app = SaveAsModalHarness(
        destinations=[
            ConsoleSaveDestination(label="Chatbook", available=True, reason=""),
            ConsoleSaveDestination(
                label="Note",
                available=False,
                reason="Notes service is not ready in this session.",
            ),
        ]
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_for_selector(
            app.screen, pilot, "#console-save-as-destination-chatbook"
        )
        destination_button = app.screen.query_one(
            "#console-save-as-destination-chatbook", Button
        )
        text = _visible_text(app.screen)

        assert destination_button.disabled is False
        assert "Note (unavailable)" in text
        assert not app.screen.query("#console-save-as-destination-note")

        await pilot.click("#console-save-as-destination-chatbook")
        await pilot.pause(0.1)

    assert app.selected_destination == "Chatbook"


@pytest.mark.asyncio
async def test_save_as_modal_harness_preserves_empty_destination_list():
    app = SaveAsModalHarness(destinations=[])

    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-save-as-modal")
        text = _visible_text(app.screen)

    assert "No Save as destinations are wired for selected messages yet." in text
    assert "Chatbook" not in text
    assert "Note" not in text


@pytest.mark.asyncio
async def test_save_as_modal_empty_state_names_the_temporary_chat_when_ephemeral():
    """F3 (task-9 review): in a temporary chat every destination is
    unavailable, so the generic "not wired" copy always fires -- it reads
    as a bug/unfinished-feature message rather than the actual rule. Must
    say WHY (the chat is temporary) -- and the generic copy must still be
    the one shown otherwise (the control)."""

    class _EphemeralSaveAsModalHarness(ConsolidatedCSSApp):
        def on_mount(self) -> None:
            self.push_screen(
                ConsoleSaveAsModal(
                    destinations=[
                        ConsoleSaveDestination(
                            label="Note", available=False, reason="blocked"
                        )
                    ],
                    ephemeral=True,
                )
            )

    app = _EphemeralSaveAsModalHarness()
    async with app.run_test(size=(100, 30)) as pilot:
        await _wait_for_selector(app.screen, pilot, "#console-save-as-modal")
        text = _visible_text(app.screen)

    assert "No Save as destinations are wired for selected messages yet." not in text
    assert "temporary" in text.lower()

    # Control: the generic copy still shows for a non-ephemeral empty list.
    normal_app = SaveAsModalHarness(destinations=[])
    async with normal_app.run_test(size=(100, 30)) as pilot:
        await _wait_for_selector(normal_app.screen, pilot, "#console-save-as-modal")
        normal_text = _visible_text(normal_app.screen)

    assert "No Save as destinations are wired for selected messages yet." in normal_text


@pytest.mark.asyncio
async def test_console_mounts_native_transcript_region():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        assert console.query_one("#console-native-transcript", ConsoleTranscript)


@pytest.mark.asyncio
async def test_mounted_console_repaints_when_transcript_style_changes():
    app = _build_test_app()
    app.app_config["appearance"] = {"console_transcript_style": "neutral"}
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="A quiet opening.",
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="An amber reply.",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for_selector(console, pilot, f"#console-message-{user.id}")

        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        user_row = transcript.query_one(f"#console-message-{user.id}")
        assistant_row = transcript.query_one(f"#console-message-{assistant.id}")
        assert "console-transcript-message-role-user" not in user_row.classes
        assert "console-transcript-message-role-assistant" not in assistant_row.classes

        app.app_config["appearance"]["console_transcript_style"] = "role_accents"
        assert (
            console._console_transcript_style() is ConsoleTranscriptStyle.ROLE_ACCENTS
        )
        for _ in range(400):
            if not console._console_sync_in_progress:
                break
            await pilot.pause(0.01)
        assert console.request_console_appearance_refresh(1) is True
        for _ in range(400):
            user_row = transcript.query_one(f"#console-message-{user.id}")
            assistant_row = transcript.query_one(f"#console-message-{assistant.id}")
            if (
                "console-transcript-message-role-user" in user_row.classes
                and "console-transcript-message-role-assistant" in assistant_row.classes
            ):
                break
            await pilot.pause(0.01)

        assert "console-transcript-message-role-user" in user_row.classes
        assert "console-transcript-message-role-assistant" in assistant_row.classes
        assert console.request_console_appearance_refresh(1) is False


@pytest.mark.asyncio
async def test_console_tab_reaches_major_console_screen_regions():
    """Keyboard traversal reaches the major Console regions in the post-onboarding state.

    First-run focus is intentionally owned by the blocking setup modal (see
    ``ConsoleSetupModal.is_blocking``), which traps Tab until setup completes.
    That is by design and covered separately. This test marks onboarding
    complete (the same ``first_send_completed`` flag the app persists after a
    real first send) so the modal renders in its non-blocking "quiet" mode
    and the workbench regions are reachable during normal use.

    TASK-2154.11 (AC-02): Tab/Shift+Tab cycle WITHIN the focused Console
    region; F6/Shift+F6 move between panes (context rail -> transcript ->
    Inspector -> composer). Both rails are opened explicitly so the
    Inspector pane joins the F6 cycle.
    """
    app = _build_test_app()
    app.app_config.setdefault("console", {})["onboarding"] = {
        "first_send_completed": True
    }
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-staged-context-tray")
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _wait_for_selector(console, pilot, "#console-inspector-rail-open")

        console._set_console_rail_preference(
            left_open=True, right_open=True, notify_on_failure=False
        )
        await pilot.pause(0.3)
        console.query_one("#console-native-composer").focus()
        await pilot.pause(0.2)

        seen_focus_ids = set()

        def _record() -> None:
            focused = getattr(console.app, "focused", None)
            if focused is not None and getattr(focused, "id", None):
                seen_focus_ids.add(focused.id)

        # F6 cycles the four panes; a few Tabs inside each pane stay local.
        for _ in range(8):
            _record()
            await pilot.press("f6")
            await pilot.pause(0.05)
            for _ in range(6):
                _record()
                await pilot.press("tab")
                await pilot.pause(0.05)

    assert "console-native-transcript" in seen_focus_ids
    assert "console-native-composer" in seen_focus_ids
    assert "console-inspector-rail-collapse" in seen_focus_ids


def test_console_streaming_assistant_row_shows_generating_placeholder_until_first_token():
    """Between send-accepted and first token the assistant row must not be empty."""
    from tldw_chatbook.Widgets.Console.console_transcript import (
        CONSOLE_FAILED_EMPTY_PLACEHOLDER,
        CONSOLE_GENERATING_PLACEHOLDER,
        _message_body,
        _message_render_text,
    )

    pending = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        id="m-generating",
        status="streaming",
    )
    assert _message_body(pending) == CONSOLE_GENERATING_PLACEHOLDER
    rendered = _message_render_text(pending, selected=False)
    assert CONSOLE_GENERATING_PLACEHOLDER in rendered.plain

    # First streamed token replaces the placeholder immediately. FB-01
    # (task-2154.16): the streaming state is a dim status line under the
    # body, not a "[streaming]" token appended to the content.
    started = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Once",
        id="m-generating",
        status="streaming",
    )
    assert _message_body(started) == "Once"
    started_rendered = _message_render_text(started, selected=False)
    assert "[streaming]" not in started_rendered.plain
    assert "Streaming…" in started_rendered.plain

    # FB-01 (task-2154.16): an empty failed row renders a meaningful
    # placeholder plus a dim "Failed" status line -- never a bare "[failed]".
    failed = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        id="m-failed",
        status="failed",
    )
    assert _message_body(failed) == CONSOLE_FAILED_EMPTY_PLACEHOLDER
    failed_rendered = _message_render_text(failed, selected=False)
    assert "[failed]" not in failed_rendered.plain
    assert CONSOLE_FAILED_EMPTY_PLACEHOLDER in failed_rendered.plain
    assert "Failed" in failed_rendered.plain

    # A failed row WITH partial content keeps that content; the state is the
    # status line, not a token glued onto the text.
    failed_partial = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="half an answer",
        id="m-failed-partial",
        status="failed",
    )
    partial_rendered = _message_render_text(failed_partial, selected=False)
    assert _message_body(failed_partial) == "half an answer"
    assert "half an answer" in partial_rendered.plain
    assert "[failed]" not in partial_rendered.plain
    assert "Failed" in partial_rendered.plain

    # TASK-457(a): a USER row only carries "failed" via the send-blocked echo;
    # the SYSTEM block-row explains it, so the user's text stays clean (no
    # "[failed]" suffix, which is an assistant-response state).
    blocked_user = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="hello",
        id="m-blocked",
        status="failed",
    )
    assert _message_body(blocked_user) == "hello"


def test_image_message_row_renders_chip_line():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="what is this?",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
        attachment_label="photo.png · 11 B",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼 photo.png · 11 B" in rendered.plain


def test_image_only_message_row_renders_chip_without_body():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼" in rendered.plain


def test_sibling_counter_rendered_for_message_with_siblings():
    """TASK-7: a message with persisted siblings shows an `(n/m)` counter."""
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="second answer",
        id="m1",
        sibling_index=1,
        sibling_count=2,
    )

    rendered = _message_render_text(message, selected=False)

    assert "(2/2)" in rendered.plain


def test_sibling_counter_rendered_for_first_of_several_siblings():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="first answer",
        id="m1",
        sibling_index=0,
        sibling_count=3,
    )

    rendered = _message_render_text(message, selected=False)

    assert "(1/3)" in rendered.plain


def test_no_sibling_counter_for_single_child_message():
    """TASK-7: a linear (unforked) message renders no `(n/m)` counter."""
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="only answer", id="m1"
    )

    rendered = _message_render_text(message, selected=False)

    assert "(1/1)" not in rendered.plain
    assert "(" not in rendered.plain


@pytest.mark.parametrize(
    ("presentation", "expected"),
    (
        (
            ConsoleCitationPresentation(phase=ConsoleCitationPhase.CHECKING),
            "Checking citations…",
        ),
        (
            ConsoleCitationPresentation(phase=ConsoleCitationPhase.REPAIRING),
            "Checking citations…",
        ),
        (
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.REPAIRED,
                original_attempt_available=True,
            ),
            "Citations repaired · View original attempt",
        ),
        (
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.REPAIRED,
                original_attempt_available=False,
            ),
            "Citations repaired",
        ),
        (
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
            ),
            "Citation repair unavailable · Original response kept",
        ),
        (
            ConsoleCitationPresentation(
                phase=ConsoleCitationPhase.SELECTED,
                notice_code=ConsoleCitationNoticeCode.CANCELED,
            ),
            "Citation repair canceled",
        ),
    ),
)
def test_citation_notice_is_exact_and_never_claims_support(presentation, expected):
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Selected answer [S1]",
        citation_presentation=presentation,
    )

    rendered = _message_render_text(message, selected=False).plain

    assert expected in rendered
    lowered_notice = rendered.splitlines()[-1].lower()
    for forbidden in ("grounded", "verified", "supported", "canonical"):
        assert forbidden not in lowered_notice


@pytest.mark.asyncio
async def test_original_attempt_preview_is_literal_distinct_row_after_owner():
    app = MutableTranscriptHarness()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Selected **repaired** answer [S1]",
        id="repaired-message",
        citation_presentation=ConsoleCitationPresentation(
            phase=ConsoleCitationPhase.SELECTED,
            notice_code=ConsoleCitationNoticeCode.REPAIRED,
            original_attempt_available=True,
        ),
    )
    original = "Original **literal** attempt"

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([message])
        transcript.set_original_attempt_previews({message.id: original})
        await transcript.refresh_messages()
        turn = transcript.query_one(
            f"#console-assistant-turn-{message.id}", ConsoleAssistantTurnWidget
        )
        preview = app.query_one(
            f"#console-original-attempt-{message.id}",
            Static,
        )
        answer_precedes_adjuncts = turn.children.index(
            turn.answer_widget
        ) < turn.children.index(turn.adjunct_stack)
        preview_is_nested = preview.parent is turn.adjunct_stack

    assert answer_precedes_adjuncts
    assert preview_is_nested
    assert "Original attempt (not selected)" in str(preview.renderable)
    assert original in str(preview.renderable)
    assert message.content == "Selected **repaired** answer [S1]"
    assert message.citation_presentation.original_attempt_available is True


@pytest.mark.asyncio
async def test_original_attempt_availability_updates_action_and_message_signatures():
    app = MutableTranscriptHarness()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Selected answer [S1]",
        id="repaired-message",
        citation_presentation=ConsoleCitationPresentation(
            phase=ConsoleCitationPhase.SELECTED,
            notice_code=ConsoleCitationNoticeCode.REPAIRED,
            original_attempt_available=True,
        ),
    )

    async with app.run_test(size=(160, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([message])
        transcript.select_message(message.id)
        await transcript.refresh_messages()
        await pilot.pause()
        await pilot.click(f"#console-message-action-more-{message.id}")
        await _wait_for_selector(
            app,
            pilot,
            "#console-message-more-view-original-attempt",
        )
        await transcript.dismiss_message_more_menu()
        before = transcript.row_render_signatures()
        before_actions = next(
            row.signature
            for row in transcript._flat_transcript_rows()
            if row.key == f"actions:{message.id}"
        )

        message.citation_presentation = ConsoleCitationPresentation(
            phase=ConsoleCitationPhase.SELECTED,
            notice_code=ConsoleCitationNoticeCode.REPAIRED,
            original_attempt_available=False,
        )
        transcript.set_messages([message])
        await transcript.refresh_messages()
        after = transcript.row_render_signatures()
        after_actions = next(
            row.signature
            for row in transcript._flat_transcript_rows()
            if row.key == f"actions:{message.id}"
        )

    assert len(app.query("#console-message-more-view-original-attempt")) == 0
    assert (
        before[f"assistant-turn:{message.id}"] != after[f"assistant-turn:{message.id}"]
    )
    assert before_actions == after_actions


def test_checking_citations_uses_active_jump_pill_copy():
    from tldw_chatbook.Widgets.Console.console_transcript import _JUMP_PILL_TEXT

    assert _JUMP_PILL_TEXT["checking_citations"] == (
        "▼ checking citations below — jump to latest"
    )


def test_transcript_message_widget_shows_sibling_counter_via_row_construction():
    """Counter reaches the actual mounted row (``ConsoleTranscriptMessage``),
    not just the pure ``_message_render_text`` helper."""
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="second answer",
        id="m1",
        sibling_index=1,
        sibling_count=2,
    )
    widget = ConsoleTranscriptMessage(message)

    assert "(2/2)" in widget.renderable.plain


def test_save_image_action_only_offered_for_image_messages():
    service = ConsoleMessageActionService()
    plain = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="text")
    with_image = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    plain_ids = [action.action_id for action in service.available_actions(plain)]
    image_ids = [action.action_id for action in service.available_actions(with_image)]
    assert "save-image" not in plain_ids
    assert "save-image" in image_ids

    result = service.dispatch("save-image", with_image)
    assert result.status == "completed"
    assert result.visible_copy == "Saving image to disk."


def test_image_chip_falls_back_to_mime_and_size_without_label():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="",
        image_data=b"x" * 2048,
        image_mime_type="image/png",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼 image/png · 2 KB" in rendered.plain


def test_image_chip_metadata_only_keeps_bare_mime():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_mime_type="image/png",
    )
    rendered = _message_render_text(message, selected=False)
    assert "🖼 image/png" in rendered.plain


def test_multi_attachment_message_renders_chip_per_attachment():
    from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="three pics",
    )
    message.attachments = (
        MessageAttachment(
            data=b"1", mime_type="image/png", display_name="a.png", position=0
        ),
        MessageAttachment(
            data=b"22", mime_type="image/jpeg", display_name="b.jpg", position=1
        ),
        MessageAttachment(
            data=None, mime_type="image/png", display_name="", position=2
        ),
    )
    message.image_data = b"1"
    message.image_mime_type = "image/png"
    message.attachment_label = "a.png"

    rendered = _message_render_text(message, selected=False)
    plain = rendered.plain
    assert "🖼 a.png" in plain
    assert "🖼 b.jpg" in plain
    assert plain.count("🖼") == 3  # dataless third falls back to mime label


def _image_row_spec(message_id: str, mode: str = "pixels"):
    from PIL import Image as PILImage
    from rich_pixels import Pixels

    from tldw_chatbook.Chat.console_image_view import ConsoleImageRowSpec

    pil = PILImage.new("RGB", (16, 16), (10, 120, 40))
    return ConsoleImageRowSpec(
        message_id=message_id,
        mode=mode,
        pixels=Pixels.from_image(pil) if mode == "pixels" else None,
        pil=pil if mode == "graphics" else None,
    )


def test_transcript_emits_image_row_when_spec_present():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])
    transcript.set_image_specs({message.id: _image_row_spec(message.id)})

    rows = transcript._transcript_rows()
    kinds = [row.kind for row in rows]
    assert "image" in kinds
    image_row = next(row for row in rows if row.kind == "image")
    assert image_row.key == f"image:{message.id}"
    assert image_row.signature == ("image", message.id, "pixels")
    # Order: message row immediately precedes its image row.
    message_index = kinds.index("message")
    assert kinds[message_index + 1] == "image"


def test_transcript_omits_image_row_without_spec_or_when_hidden():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])
    # No specs set at all -> no image rows (unmounted-test posture).
    assert all(row.kind != "image" for row in transcript._transcript_rows())
    # Hidden mode is expressed by the screen simply omitting the spec.
    transcript.set_image_specs({})
    assert all(row.kind != "image" for row in transcript._transcript_rows())


def test_image_row_signature_stable_across_streaming_ticks():
    transcript = ConsoleTranscript()
    user = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="", status="streaming"
    )
    transcript.set_messages([user, assistant])
    transcript.set_image_specs({user.id: _image_row_spec(user.id)})

    first = next(r for r in transcript._transcript_rows() if r.kind == "image")
    assistant.content = "more streamed text"
    transcript.set_messages([user, assistant])
    second = next(r for r in transcript._transcript_rows() if r.kind == "image")
    assert first.signature == second.signature


def test_image_row_widget_builds_for_both_modes():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])

    transcript.set_image_specs({message.id: _image_row_spec(message.id, "pixels")})
    pixels_row = next(r for r in transcript._transcript_rows() if r.kind == "image")
    pixels_widget = transcript._build_row_widget(pixels_row, track=False)
    assert pixels_widget.id == f"console-image-{message.id}"

    transcript.set_image_specs({message.id: _image_row_spec(message.id, "graphics")})
    graphics_row = next(r for r in transcript._transcript_rows() if r.kind == "image")
    graphics_widget = transcript._build_row_widget(graphics_row, track=False)
    assert graphics_widget.id == f"console-image-{message.id}"
    # Graphics images now carry an EXPLICIT fitted cell size (not just
    # max-width/max-height): textual_image's "auto" sizing could resolve to a
    # transient 0-region mid-mount and crash PIL.resize(). A 16x16 square fits
    # the 80x40 box exactly at (80, 40).
    assert graphics_widget.styles.width.value == 80
    assert graphics_widget.styles.height.value == 40


def test_image_row_rebuild_tracked_on_mode_change():
    transcript = ConsoleTranscript()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    transcript.set_messages([message])
    transcript.set_image_specs({message.id: _image_row_spec(message.id, "pixels")})
    rows = transcript._transcript_rows()
    image_row = next(r for r in rows if r.kind == "image")
    widget = transcript._build_row_widget(image_row, track=True)
    assert transcript.row_build_counts()[f"image:{message.id}"] == 1

    transcript.set_image_specs({message.id: _image_row_spec(message.id, "graphics")})
    new_row = next(r for r in transcript._transcript_rows() if r.kind == "image")
    assert new_row.signature != image_row.signature
    updated = transcript._update_row_widget(widget, new_row)
    assert updated is not widget
    assert transcript.row_build_counts()[f"image:{message.id}"] == 2


def test_toggle_image_view_action_offered_and_dispatched_for_image_messages():
    service = ConsoleMessageActionService()
    plain = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="text")
    with_image = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="pic",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )
    plain_ids = [action.action_id for action in service.available_actions(plain)]
    image_ids = [action.action_id for action in service.available_actions(with_image)]
    assert "toggle-image-view" not in plain_ids
    assert "toggle-image-view" in image_ids
    assert image_ids.index("toggle-image-view") < image_ids.index("save-image")

    result = service.dispatch("toggle-image-view", with_image)
    assert result.status == "completed"
    assert result.visible_copy == "Toggled image view."
    assert result.target_message_id == with_image.id


# ---------------------------------------------------------------------------
# TASK-259: per-message row-signature cache. Derivation of the expensive row
# render payload must be O(changed messages) per changed tick, with
# correctness preserved for delete, reorder, and variant-switch.
# ---------------------------------------------------------------------------


def _cache_test_messages(count: int) -> list[ConsoleChatMessage]:
    return [
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER
            if index % 2 == 0
            else ConsoleMessageRole.ASSISTANT,
            content=f"message body {index}",
            id=f"m{index}",
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_transcript_signature_derivation_is_o_changed_messages():
    """A changed tick re-derives only the changed message's render signature."""
    app = MutableTranscriptHarness()
    messages = _cache_test_messages(10)

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        baseline = transcript.message_signature_compute_counts()
        assert all(count == 1 for count in baseline.values())

        # Unchanged tick (same objects re-set, as the 0.2s sync does).
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        assert transcript.message_signature_compute_counts() == baseline

        # Changed tick: exactly one message's content changed.
        messages[4].content = "message body 4 (edited)"
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        after = transcript.message_signature_compute_counts()
        assert "message body 4 (edited)" in _message_row_text(transcript, "m4")

    assert after["m4"] == baseline["m4"] + 1
    for message in messages:
        if message.id == "m4":
            continue
        assert after[message.id] == baseline[message.id]


@pytest.mark.asyncio
async def test_transcript_signature_cache_miss_on_equal_length_edit():
    """The cache must key on content, never on content length alone."""
    app = MutableTranscriptHarness()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="aaaa", id="m-edit"
    )

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([message])
        await transcript.refresh_messages()

        message.content = "bbbb"  # same length, same status
        transcript.set_messages([message])
        await transcript.refresh_messages()
        assert "bbbb" in _message_row_text(transcript, "m-edit")
        assert transcript.message_signature_compute_counts()["m-edit"] == 2


@pytest.mark.asyncio
async def test_transcript_signature_cache_survives_delete():
    """Deleting a message prunes its cache entry and its mounted row."""
    app = MutableTranscriptHarness()
    messages = _cache_test_messages(5)

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        assert "m2" in transcript.message_signature_cache_ids()

        survivors = [message for message in messages if message.id != "m2"]
        transcript.set_messages(survivors)
        await transcript.refresh_messages()

        assert "m2" not in transcript.message_signature_cache_ids()
        assert len(transcript.query("#console-message-m2")) == 0
        # Survivors were not re-derived by the delete.
        counts = transcript.message_signature_compute_counts()
        assert all(counts[message.id] == 1 for message in survivors)

        # Re-adding the same id with different content renders the new text.
        replacement = ConsoleChatMessage(
            role=ConsoleMessageRole.USER, content="replacement body", id="m2"
        )
        transcript.set_messages(survivors + [replacement])
        await transcript.refresh_messages()
        assert "replacement body" in _message_row_text(transcript, "m2")


@pytest.mark.asyncio
async def test_transcript_signature_cache_survives_reorder():
    """Reordering messages re-derives nothing and renders the new order."""
    app = MutableTranscriptHarness()
    messages = _cache_test_messages(4)

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()
        baseline = transcript.message_signature_compute_counts()

        reordered = [messages[2], messages[0], messages[3], messages[1]]
        transcript.set_messages(reordered)
        await transcript.refresh_messages()

        assert transcript.message_signature_compute_counts() == baseline
        rendered_ids = [
            widget.message_id
            for widget in transcript.query(".console-transcript-message")
        ]

    assert rendered_ids == ["m2", "m0", "m3", "m1"]


# ---------------------------------------------------------------------------
# TASK-15453: `_reconcile_rows` must not `move_child` a row that is already
# in the right position. A steady-state pass (no order change) must issue
# zero `move_child` calls; a genuine reorder must still issue >0 calls and
# still land the correct final child order. See
# Docs/Design/2026-08-11-input-latency-audit.md (Console section).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_rows_steady_state_issues_zero_move_child_calls():
    """A no-op refresh (same messages re-set, like the 0.2s stream tick) moves nothing."""
    app = MutableTranscriptHarness()
    messages = _cache_test_messages(8)

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()  # first mount: nothing to move yet
        baseline_order = _rendered_message_ids(transcript)

        calls = _spy_move_child(transcript)
        # Same objects, unchanged content/order -- exactly what the 0.2s
        # streaming tick and a transcript click (full reconcile) do when
        # nothing actually changed.
        transcript.set_messages(list(messages))
        await transcript.refresh_messages()

        assert calls == [], (
            f"steady-state reconcile issued {len(calls)} move_child call(s); "
            "rows already in position must not be moved"
        )
        assert _rendered_message_ids(transcript) == baseline_order


@pytest.mark.asyncio
async def test_reconcile_rows_reorder_moves_widgets_and_lands_correct_order():
    """A genuine reorder still issues move_child calls and produces correct order."""
    app = MutableTranscriptHarness()
    messages = _cache_test_messages(4)

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(messages)
        await transcript.refresh_messages()

        calls = _spy_move_child(transcript)
        reordered = [messages[2], messages[0], messages[3], messages[1]]
        transcript.set_messages(reordered)
        await transcript.refresh_messages()

        assert len(calls) > 0, "a real reorder must still move at least one row"
        assert _rendered_message_ids(transcript) == ["m2", "m0", "m3", "m1"]


@pytest.mark.asyncio
async def test_reconcile_rows_session_switch_lands_correct_order():
    """Switching to an entirely different message set (session switch) orders correctly."""
    app = MutableTranscriptHarness()
    first_session = _cache_test_messages(3)
    second_session = [
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER
            if index % 2 == 0
            else ConsoleMessageRole.ASSISTANT,
            content=f"other session body {index}",
            id=f"s{index}",
        )
        for index in range(3)
    ]

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(first_session)
        await transcript.refresh_messages()
        assert _rendered_message_ids(transcript) == ["m0", "m1", "m2"]

        transcript.set_messages(second_session)
        await transcript.refresh_messages()

        assert _rendered_message_ids(transcript) == ["s0", "s1", "s2"]


@pytest.mark.asyncio
async def test_reconcile_rows_branch_navigation_replaces_suffix_in_order():
    """Branch navigation (shared prefix, swapped-in sibling suffix) orders correctly."""
    app = MutableTranscriptHarness()
    shared_prefix = _cache_test_messages(2)  # m0, m1
    original_tail = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="q2", id="m2"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="a2", id="m3"),
    ]
    sibling_tail = [
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER, content="q2 (sibling)", id="m2b"
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="a2 (sibling)", id="m3b"
        ),
    ]

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(shared_prefix + original_tail)
        await transcript.refresh_messages()
        assert _rendered_message_ids(transcript) == ["m0", "m1", "m2", "m3"]

        # `set_active_leaf` swipe to a sibling branch: shared prefix stays,
        # the tail past the branch point is replaced by the sibling's nodes.
        transcript.set_messages(shared_prefix + sibling_tail)
        await transcript.refresh_messages()

        assert _rendered_message_ids(transcript) == ["m0", "m1", "m2b", "m3b"]


@pytest.mark.asyncio
async def test_transcript_signature_cache_survives_variant_switch():
    """Switching variants re-derives only that message and shows the variant."""
    app = MutableTranscriptHarness()
    plain = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="prompt", id="m-plain"
    )
    varied = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="first answer", id="m-varied"
    )
    varied.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first answer", "second answer"],
    )

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([plain, varied])
        await transcript.refresh_messages()
        baseline = transcript.message_signature_compute_counts()
        assert "first answer" in _message_row_text(transcript, "m-varied")

        varied.variants.selected_index = 1
        transcript.set_messages([plain, varied])
        await transcript.refresh_messages()
        after = transcript.message_signature_compute_counts()

        assert "second answer" in _message_row_text(transcript, "m-varied")
        assert after["m-varied"] == baseline["m-varied"] + 1
        assert after["m-plain"] == baseline["m-plain"]
        # TASK-15453: a variant switch must not disturb row position.
        assert _rendered_message_ids(transcript) == ["m-plain", "m-varied"]


@pytest.mark.asyncio
async def test_transcript_signature_cache_miss_on_status_change():
    """A status-only change (streaming -> complete) re-derives the row."""
    app = MutableTranscriptHarness()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial",
        id="m-stream",
        status="streaming",
    )

    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages([message])
        await transcript.refresh_messages()
        assert "Streaming…" in _message_row_text(transcript, "m-stream")

        message.status = "complete"
        transcript.set_messages([message])
        await transcript.refresh_messages()

        assert "Streaming…" not in _message_row_text(transcript, "m-stream")
        assert transcript.message_signature_compute_counts()["m-stream"] == 2


# ---------------------------------------------------------------------------
# SP2 /rewind: render-derived "summarize up to here" boundary banner
# ---------------------------------------------------------------------------


def _boundary_messages():
    return [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="q1", id="m1"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="a1", id="m2"),
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="q3", id="m3"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="a3", id="m4"),
    ]


def test_console_transcript_summary_banner_renders_above_boundary():
    from tldw_chatbook.Widgets.Console.console_transcript import (
        CONSOLE_SUMMARY_BANNER_COPY,
    )

    transcript = ConsoleTranscript()
    transcript.set_messages(_boundary_messages())
    transcript.set_summary_boundary("m3")

    plain = transcript.to_plain_text(width=80)
    lines = plain.splitlines()

    assert CONSOLE_SUMMARY_BANNER_COPY in plain
    banner_index = next(
        i for i, line in enumerate(lines) if CONSOLE_SUMMARY_BANNER_COPY in line
    )
    # The banner sits ABOVE the boundary turn (q3) and BELOW the prior turn.
    q3_index = next(i for i, line in enumerate(lines) if "q3" in line)
    a1_index = next(i for i, line in enumerate(lines) if "a1" in line)
    assert a1_index < banner_index < q3_index


def test_console_transcript_summary_banner_absent_when_boundary_not_rendered():
    from tldw_chatbook.Widgets.Console.console_transcript import (
        CONSOLE_SUMMARY_BANNER_COPY,
    )

    transcript = ConsoleTranscript()
    transcript.set_messages(_boundary_messages())
    # A dangling boundary (not in the rendered messages) shows no banner.
    transcript.set_summary_boundary("ghost")

    assert CONSOLE_SUMMARY_BANNER_COPY not in transcript.to_plain_text(width=80)


@pytest.mark.asyncio
async def test_console_transcript_summary_banner_mounts_and_clears():
    from tldw_chatbook.Widgets.Console.console_transcript import (
        CONSOLE_SUMMARY_BANNER_COPY,
    )

    app = MutableTranscriptHarness()
    async with app.run_test(size=(100, 32)):
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_messages(_boundary_messages())
        transcript.set_summary_boundary("m3")
        await transcript.refresh_messages()
        banners = transcript.query(".console-transcript-summary-banner")
        assert len(banners) == 1
        assert CONSOLE_SUMMARY_BANNER_COPY in str(list(banners)[0].renderable)

        # Restore to before the boundary -> banner disappears.
        transcript.set_summary_boundary(None)
        await transcript.refresh_messages()
        assert len(transcript.query(".console-transcript-summary-banner")) == 0


def test_console_transcript_empty_state_is_centered_in_stylesheets():
    """LY-12 (TASK-2154.24): both stylesheets center the fresh-session empty
    state so it reads as intentionally empty rather than stranded at an edge.

    Source-level pin: bare-widget harnesses do not load the app stylesheets,
    so geometry cannot be asserted here; the pattern mirrors the css-string
    pins in test_master_shell_navigation.py.
    """
    from pathlib import Path

    for css_path in (
        Path("tldw_chatbook/css/components/_agentic_terminal.tcss"),
        Path("tldw_chatbook/css/tldw_cli_modular.tcss"),
    ):
        css = css_path.read_text(encoding="utf-8")
        panel_rule = css.split(".console-transcript-empty-panel {", 1)[1].split("}", 1)[
            0
        ]
        assert "align: center middle" in panel_rule, css_path
        body_rule = css.split(".console-transcript-empty-body {", 1)[1].split("}", 1)[0]
        assert "text-align: center" in body_rule, css_path
