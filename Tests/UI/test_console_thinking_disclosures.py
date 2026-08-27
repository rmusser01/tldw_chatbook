"""Mounted contracts for Console model-thinking disclosures."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    PROPRIETARY_THINKING_NOTICE,
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_turn_grouping import project_thinking_activities
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
)
from tldw_chatbook.Widgets.Console.console_assistant_turn import (
    ConsoleActivityDisclosure,
    ConsoleAssistantTurnWidget,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


class ThinkingTranscriptHarness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.copied: list[str] = []

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")

    def copy_to_clipboard(self, text: str) -> None:
        self.copied.append(text)


def _displayable(
    text: str,
    *,
    block_id: str = "thinking-block-0",
    round_ordinal: int = 0,
    status: str = "complete",
) -> DisplayableThinkingBlock:
    return DisplayableThinkingBlock(
        block_id=block_id,
        round_ordinal=round_ordinal,
        provider="local_llamacpp",
        model="model.gguf",
        protocol="openai_chat",
        source_format="start_anchored_think",
        status=status,  # type: ignore[arg-type]
        text=text,
    )


def _proprietary() -> ProprietaryThinkingBlock:
    return ProprietaryThinkingBlock(
        block_id="private-block-0",
        round_ordinal=0,
        provider="moonshot",
        model="kimi-k2",
        protocol="openai_chat",
        source_format="reasoning_content",
        status="complete",
    )


def _assistant(
    *,
    content: str = "",
    status: str = "streaming",
    blocks: tuple[DisplayableThinkingBlock | ProprietaryThinkingBlock, ...] = (),
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
        status=status,  # type: ignore[arg-type]
        id="assistant-thinking",
        thinking=ThinkingEnvelope(blocks) if blocks else None,
    )


def _activity_id(assistant: ConsoleChatMessage) -> str:
    return project_thinking_activities(assistant=assistant)[0].activity_id


def _disclosure(
    transcript: ConsoleTranscript, assistant: ConsoleChatMessage
) -> ConsoleActivityDisclosure:
    return transcript.query_one(
        f"#console-activity-disclosure-{_activity_id(assistant)}",
        ConsoleActivityDisclosure,
    )


@pytest.mark.asyncio
async def test_first_live_evidence_expands_and_delta_updates_same_widgets() -> None:
    app = ThinkingTranscriptHarness()
    initial = _assistant(blocks=(_displayable("first"),))

    async with app.run_test(size=(100, 28)):
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([initial], session_id="session-a")
        await transcript.refresh_messages()
        disclosure = _disclosure(transcript, initial)
        turn = transcript.query_one(ConsoleAssistantTurnWidget)
        answer = turn.answer_widget

        assert disclosure.expanded
        assert disclosure.detail_stack.children
        assert (
            transcript.thinking_detail_text(disclosure.activity_message_id) == "first"
        )

        updated = replace(
            initial,
            thinking=ThinkingEnvelope((_displayable("first second"),)),
        )
        transcript.set_messages([updated], session_id="session-a")
        await transcript.refresh_messages()

        assert _disclosure(transcript, updated) is disclosure
        assert transcript.query_one(ConsoleAssistantTurnWidget) is turn
        assert turn.answer_widget is answer
        assert transcript.thinking_detail_text(disclosure.activity_message_id) == (
            "first second"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["answer", "tool", "terminal"])
async def test_first_answer_tool_or_terminal_boundary_auto_collapses_once(
    boundary: str,
) -> None:
    app = ThinkingTranscriptHarness()
    live = _assistant(blocks=(_displayable("private chain"),))

    async with app.run_test(size=(100, 28)):
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([live], session_id="session-a")
        await transcript.refresh_messages()
        disclosure = _disclosure(transcript, live)
        assert disclosure.expanded

        if boundary == "answer":
            messages = [replace(live, content="Answer")]
        elif boundary == "terminal":
            messages = [replace(live, status="complete")]
        else:
            messages = [
                live,
                ConsoleChatMessage(
                    role=ConsoleMessageRole.TOOL,
                    content="tool result",
                    id="tool-boundary",
                    activity_round_ordinal=0,
                    activity_presentation=ConsoleActivityPresentation(
                        "tool", "fs_read", "success"
                    ),
                ),
            ]
        transcript.set_messages(messages, session_id="session-a")
        await transcript.refresh_messages()

        assert not disclosure.expanded
        assert not disclosure.detail_stack.children


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["mouse", "enter", "space", "o"])
async def test_manual_toggle_wins_over_pending_auto_collapse(control: str) -> None:
    app = ThinkingTranscriptHarness()
    live = _assistant(blocks=(_displayable("private chain"),))

    async with app.run_test(size=(100, 28)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([live], session_id="session-a")
        await transcript.refresh_messages()
        disclosure = _disclosure(transcript, live)
        header = disclosure.header
        if control == "mouse":
            header.on_click(SimpleNamespace(stop=lambda: None))
        elif control in {"enter", "space"}:
            header.focus()
            await pilot.press(control)
        else:
            transcript.selected_message_id = disclosure.activity_message_id
            transcript.focus()
            await pilot.press("o")
        await pilot.pause()
        assert not disclosure.expanded

        transcript.set_messages(
            [replace(live, content="Answer")], session_id="session-a"
        )
        await transcript.refresh_messages()

        assert not disclosure.expanded
        assert disclosure.activity_message_id in transcript._manual_thinking_disclosures


@pytest.mark.asyncio
async def test_manual_expand_also_wins_over_pending_auto_collapse() -> None:
    app = ThinkingTranscriptHarness()
    live = _assistant(blocks=(_displayable("private chain"),))

    async with app.run_test(size=(100, 28)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([live], session_id="session-a")
        await transcript.refresh_messages()
        disclosure = _disclosure(transcript, live)
        disclosure.header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        disclosure.header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        assert disclosure.expanded

        transcript.set_messages(
            [replace(live, content="Answer")], session_id="session-a"
        )
        await transcript.refresh_messages()
        assert disclosure.expanded


@pytest.mark.asyncio
async def test_historical_detail_is_collapsed_lazy_and_resolved_on_demand() -> None:
    app = ThinkingTranscriptHarness()
    historical = _assistant(
        content="Answer",
        status="complete",
        blocks=(_displayable("full historical thinking"),),
    )

    async with app.run_test(size=(100, 28)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([historical], session_id="session-a")
        await transcript.refresh_messages()
        disclosure = _disclosure(transcript, historical)

        assert not disclosure.expanded
        assert disclosure.detail_available
        assert not disclosure.detail_stack.children
        assert transcript.thinking_detail_text(disclosure.activity_message_id) == (
            "full historical thinking"
        )

        disclosure.header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        assert disclosure.expanded
        assert disclosure.detail_stack.children

        disclosure.header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        assert not disclosure.detail_stack.children


@pytest.mark.asyncio
async def test_late_terminal_proprietary_evidence_starts_collapsed() -> None:
    app = ThinkingTranscriptHarness()
    terminal = _assistant(status="complete", blocks=(_proprietary(),))

    async with app.run_test(size=(100, 28)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([terminal], session_id="session-a")
        await transcript.refresh_messages()
        disclosure = _disclosure(transcript, terminal)

        assert not disclosure.expanded
        assert not disclosure.detail_stack.children
        assert transcript.thinking_detail_text(disclosure.activity_message_id) == (
            PROPRIETARY_THINKING_NOTICE
        )

        disclosure.header.on_click(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        assert disclosure.detail_stack.children


@pytest.mark.asyncio
async def test_no_actual_evidence_mounts_no_thinking_disclosure() -> None:
    app = ThinkingTranscriptHarness()

    async with app.run_test(size=(100, 20)):
        transcript = app.query_one(ConsoleTranscript)
        assistant = _assistant(content="Answer", status="complete")
        assistant.opaque_thinking_json = '{"version":99,"secret":"opaque"}'
        transcript.set_messages([assistant], session_id="session-a")
        await transcript.refresh_messages()

        assert not list(transcript.query(ConsoleActivityDisclosure))


def test_collapsed_inspector_resolves_full_thinking_without_changing_answer() -> None:
    transcript = ConsoleTranscript()
    assistant = _assistant(
        content="Public answer",
        status="complete",
        blocks=(_displayable("full private thinking"),),
    )
    transcript.set_messages([assistant], session_id="session-a")
    activity_id = _activity_id(assistant)

    assert transcript.thinking_detail_text(activity_id) == "full private thinking"
    assert transcript.display_message(activity_id).content == "full private thinking"
    assert transcript.display_message(assistant.id).content == "Public answer"


@pytest.mark.asyncio
async def test_collapsed_thinking_copy_uses_full_body() -> None:
    app = ThinkingTranscriptHarness()
    assistant = _assistant(
        content="Public answer",
        status="complete",
        blocks=(_displayable("full private thinking"),),
    )

    async with app.run_test(size=(100, 28)):
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([assistant], session_id="session-a")
        await transcript.refresh_messages()
        disclosure = _disclosure(transcript, assistant)
        transcript.select_message(disclosure.activity_message_id)
        assert transcript.selected_message_id == disclosure.activity_message_id
        assert transcript.thinking_detail_text(disclosure.activity_message_id) == (
            "full private thinking"
        )
        assert transcript.app is app
        transcript.action_invoke_selected_action("copy")

        assert not disclosure.expanded
        assert app.copied == ["full private thinking"]


def test_session_switch_prunes_thinking_owner_state_and_tool_expansion_survives_updates() -> (
    None
):
    transcript = ConsoleTranscript()
    assistant = _assistant(blocks=(_displayable("live"),))
    thinking_id = _activity_id(assistant)
    tool = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="preview",
        tool_output_full="full",
        id="tool-keep-expanded",
        activity_presentation=ConsoleActivityPresentation("tool", "fs_read", "success"),
    )
    transcript.set_messages([assistant, tool], session_id="session-a")
    transcript.toggle_tool_output(tool.id)
    transcript.set_messages([assistant, tool], session_id="session-a")
    assert tool.id in transcript._expanded_tool_output_ids

    replacement = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Other conversation",
        id="assistant-other-session",
    )
    transcript.set_messages([replacement], session_id="session-b")
    assert thinking_id not in transcript._thinking_activity_refs
    assert not transcript._pending_thinking_auto_collapse
    assert not transcript._manual_thinking_disclosures
