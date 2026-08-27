"""Mounted contracts for Console model-thinking disclosures."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from tldw_chatbook.Chat.console_chat_models import (
    PROPRIETARY_THINKING_NOTICE,
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_turn_grouping import (
    group_console_transcript_messages,
    project_thinking_activities,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
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


class StyledThinkingTranscriptHarness(ThinkingTranscriptHarness):
    """Thinking harness using the same bundled stylesheet as production."""

    CSS_PATH = str(BUNDLED_STYLESHEET)


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
async def test_unowned_tool_arrival_collapses_only_current_live_thinking() -> None:
    """Arrival, not a guessed round, is the boundary for ordinal-less tools."""
    app = ThinkingTranscriptHarness()
    first_block = _displayable("first round", block_id="round-zero")
    first = _assistant(blocks=(first_block,))
    old_tool = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="tool result",
        id="ordinal-less-tool",
        activity_round_ordinal=None,
        activity_presentation=ConsoleActivityPresentation("tool", "fs_read", "success"),
    )

    async with app.run_test(size=(100, 28)):
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([first], session_id="session-a")
        await transcript.refresh_messages()
        first_disclosure = _disclosure(transcript, first)
        assert first_disclosure.expanded

        transcript.set_messages([first, old_tool], session_id="session-a")
        await transcript.refresh_messages()
        assert not first_disclosure.expanded
        assert first_disclosure.status == "done"

        second = replace(
            first,
            thinking=ThinkingEnvelope(
                (
                    first_block,
                    _displayable(
                        "second round",
                        block_id="round-one",
                        round_ordinal=1,
                    ),
                )
            ),
        )
        transcript.set_messages([second, old_tool], session_id="session-a")
        await transcript.refresh_messages()
        second_ref = project_thinking_activities(assistant=second)[1]
        second_disclosure = transcript.query_one(
            f"#console-activity-disclosure-{second_ref.activity_id}",
            ConsoleActivityDisclosure,
        )
        assert second_disclosure.expanded
        assert second_disclosure.status == "live"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["answer", "tool", "terminal"])
async def test_live_proprietary_notice_expands_then_collapses_at_real_boundary(
    boundary: str,
) -> None:
    app = ThinkingTranscriptHarness()
    live = _assistant(blocks=(_proprietary(),))

    async with app.run_test(size=(100, 28)):
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([live], session_id="session-a")
        await transcript.refresh_messages()
        disclosure = _disclosure(transcript, live)

        assert disclosure.expanded
        assert disclosure.status == "unavailable"
        assert transcript.thinking_detail_text(disclosure.activity_message_id) == (
            PROPRIETARY_THINKING_NOTICE
        )

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
                    id="proprietary-tool-boundary",
                    activity_round_ordinal=None,
                    activity_presentation=ConsoleActivityPresentation(
                        "tool", "fs_read", "success"
                    ),
                ),
            ]
        transcript.set_messages(messages, session_id="session-a")
        await transcript.refresh_messages()

        assert not disclosure.expanded
        assert disclosure.status == "unavailable"
        assert not disclosure.detail_stack.children


@pytest.mark.asyncio
async def test_proprietary_status_is_fully_painted_at_narrow_width() -> None:
    """The literal unavailable state must not be clipped to a color-only cue."""
    app = StyledThinkingTranscriptHarness()
    live = _assistant(blocks=(_proprietary(),))

    async with app.run_test(size=(60, 18)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([live], session_id="session-a")
        await transcript.refresh_messages()
        await pilot.pause(0.2)
        disclosure = _disclosure(transcript, live)

        assert disclosure.header.status_widget.region.width >= len("· unavailable")
        assert disclosure.header.region.height == 1


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


@pytest.mark.asyncio
async def test_visibility_gate_hides_only_thinking_and_restores_it_collapsed() -> None:
    app = ThinkingTranscriptHarness()
    assistant = _assistant(
        content="Public answer",
        status="complete",
        blocks=(_displayable("captured thinking"),),
    )
    tool = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="preview",
        tool_output_full="full tool result",
        id="tool-keeps-state",
        activity_presentation=ConsoleActivityPresentation("tool", "fs_read", "success"),
    )

    async with app.run_test(size=(100, 28)):
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([assistant, tool], session_id="session-a")
        await transcript.refresh_messages()
        disclosure = _disclosure(transcript, assistant)
        turn = transcript.query_one(ConsoleAssistantTurnWidget)
        answer = turn.answer_widget
        transcript.toggle_tool_output(tool.id)
        transcript.select_message(disclosure.activity_message_id)

        transcript.set_model_thinking_visible(False)
        await transcript.refresh_messages()

        selector = f"#console-activity-disclosure-{_activity_id(assistant)}"
        assert not list(transcript.query(selector))
        assert transcript.query_one(ConsoleAssistantTurnWidget) is turn
        assert turn.answer_widget is answer
        assert transcript.selected_message_id is None
        assert tool.id in transcript._expanded_tool_output_ids
        assert transcript.thinking_detail_text(_activity_id(assistant)) == (
            "captured thinking"
        )

        transcript.set_model_thinking_visible(True)
        await transcript.refresh_messages()

        restored = _disclosure(transcript, assistant)
        assert not restored.expanded
        assert transcript.query_one(ConsoleAssistantTurnWidget) is turn
        assert turn.answer_widget is answer
        assert tool.id in transcript._expanded_tool_output_ids


@pytest.mark.asyncio
async def test_hidden_live_thinking_resumes_its_pending_expanded_lifecycle() -> None:
    app = ThinkingTranscriptHarness()
    live = _assistant(blocks=(_displayable("live private chain"),))

    async with app.run_test(size=(100, 28)):
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_model_thinking_visible(False)
        transcript.set_messages([live], session_id="session-a")
        await transcript.refresh_messages()
        assert not list(
            transcript.query(f"#console-activity-disclosure-{_activity_id(live)}")
        )

        transcript.set_model_thinking_visible(True)
        await transcript.refresh_messages()

        assert _disclosure(transcript, live).expanded


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


def test_plain_transcript_real_grouping_keeps_activities_but_omits_thinking() -> None:
    raw_continuation = "RAW-CONTINUATION-TRANSCRIPT-CANARY"
    transcript = ConsoleTranscript()
    assistant = _assistant(
        content="VISIBLE-ANSWER-CANARY",
        status="complete",
        blocks=(
            _displayable("DISPLAYABLE-THINKING-CANARY"),
            replace(_proprietary(), round_ordinal=1),
        ),
    )
    assistant = replace(
        assistant,
        provider_continuation=ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider="moonshot",
            protocol="chat_completions",
            model="kimi-k2.6",
            api_base_url="https://api.moonshot.ai/v1",
            state="complete",
            rounds=(
                ContinuationRound(
                    assistant_content="VISIBLE-ANSWER-CANARY",
                    reasoning_blocks=(raw_continuation,),
                    calls=(),
                ),
            ),
        ),
    )
    planning = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="VISIBLE-PLANNING-CANARY",
        id="planning-activity",
        activity_presentation=ConsoleActivityPresentation(
            "planning", "Planning", "done"
        ),
    )
    tool = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="VISIBLE-TOOL-CANARY",
        id="tool-activity",
        activity_presentation=ConsoleActivityPresentation("tool", "fs_read", "success"),
    )
    messages = [assistant, planning, tool]
    transcript.set_messages(messages, session_id="session-a")

    units = group_console_transcript_messages(messages)
    assert len(units) == 1
    turn = units[0].assistant_turn
    assert turn is not None
    assert turn.activities == (planning, tool)
    assert all(isinstance(activity, ConsoleChatMessage) for activity in turn.activities)

    plain = transcript.to_plain_text()

    assert assistant.provider_continuation is not None
    assert "VISIBLE-ANSWER-CANARY" in plain
    assert "VISIBLE-PLANNING-CANARY" in plain
    assert "VISIBLE-TOOL-CANARY" in plain
    assert "DISPLAYABLE-THINKING-CANARY" not in plain
    assert raw_continuation not in plain
    assert PROPRIETARY_THINKING_NOTICE not in plain


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
