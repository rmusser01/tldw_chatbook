"""Pure presentation tests for Console model-thinking activities."""

import re

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    PROPRIETARY_THINKING_NOTICE,
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_turn_grouping import (
    project_thinking_activities,
    thinking_activity_id,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
)


def _displayable(
    *,
    block_id: str = "block-0",
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
        text="Exact model thinking",
    )


def _proprietary(*, status: str = "complete") -> ProprietaryThinkingBlock:
    return ProprietaryThinkingBlock(
        block_id="private-0",
        round_ordinal=0,
        provider="moonshot",
        model="kimi-k2",
        protocol="openai_chat",
        source_format="reasoning_content",
        status=status,  # type: ignore[arg-type]
    )


def _assistant(
    message_id: str,
    *blocks: DisplayableThinkingBlock | ProprietaryThinkingBlock,
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Answer",
        id=message_id,
        thinking=ThinkingEnvelope(tuple(blocks)) if blocks else None,
    )


@pytest.mark.parametrize(
    ("block", "expected_status"),
    [
        (_displayable(status="complete"), "done"),
        (_displayable(status="stopped"), "stopped"),
        (_displayable(status="failed"), "failed"),
    ],
)
def test_displayable_thinking_projects_supported_terminal_statuses(
    block: DisplayableThinkingBlock,
    expected_status: str,
) -> None:
    refs = project_thinking_activities(
        session_id="session-1",
        assistant=_assistant("assistant-1", block),
    )

    assert [(ref.label, ref.status, ref.block_id) for ref in refs] == [
        ("Thinking", expected_status, block.block_id)
    ]


def test_current_capture_block_projects_live_status() -> None:
    block = _displayable()

    refs = project_thinking_activities(
        session_id="session-1",
        assistant=_assistant("assistant-1", block),
        live_block_id=block.block_id,
    )

    assert refs[0].status == "live"


def test_proprietary_evidence_projects_unavailable_without_stored_body() -> None:
    ref = project_thinking_activities(
        session_id="session-1",
        assistant=_assistant("assistant-1", _proprietary(status="failed")),
    )[0]

    assert ref.label == "Thinking"
    assert ref.status == "unavailable"
    assert PROPRIETARY_THINKING_NOTICE == (
        "Proprietary thinking obfuscated - not available"
    )
    assert not hasattr(ref, "body")


def test_no_envelope_or_only_opaque_data_projects_no_activity() -> None:
    assistant = _assistant("assistant-1")
    assistant.opaque_thinking_json = '{"version":99,"secret":"not evidence"}'

    assert (
        project_thinking_activities(session_id="session-1", assistant=assistant) == ()
    )


def test_activity_ids_are_deterministic_trusted_and_hide_hostile_block_ids() -> None:
    hostile = "raw'] #widget { color: red; }\nprivate"
    assistant = _assistant("assistant-1", _displayable(block_id=hostile))

    first = project_thinking_activities(session_id="session-1", assistant=assistant)[0]
    second = project_thinking_activities(session_id="session-1", assistant=assistant)[0]

    assert first.activity_id == second.activity_id
    assert re.fullmatch(r"thinking-[0-9a-f]{32}", first.activity_id)
    assert hostile not in first.activity_id


def test_duplicate_block_ids_are_namespaced_by_session_and_assistant_owner() -> None:
    first_owner = _assistant("assistant-1", _displayable(block_id="duplicate"))
    second_owner = _assistant("assistant-2", _displayable(block_id="duplicate"))

    ids = {
        project_thinking_activities(session_id="session-1", assistant=first_owner)[
            0
        ].activity_id,
        project_thinking_activities(session_id="session-2", assistant=first_owner)[
            0
        ].activity_id,
        project_thinking_activities(session_id="session-1", assistant=second_owner)[
            0
        ].activity_id,
    }

    assert len(ids) == 3


def test_activity_id_components_cannot_collide_across_hostile_boundaries() -> None:
    assert thinking_activity_id(
        session_id="session",
        assistant_message_id="assistant\0block",
        block_id="tail",
    ) != thinking_activity_id(
        session_id="session",
        assistant_message_id="assistant",
        block_id="block\0tail",
    )


@pytest.mark.parametrize(
    "status",
    [
        pytest.param("live", id="active"),
        pytest.param("stopped", id="stopped"),
        pytest.param("unavailable", id="unavailable"),
    ],
)
def test_activity_presentation_accepts_expanded_thinking_statuses(status: str) -> None:
    assert (
        ConsoleActivityPresentation(
            "thinking",
            "Thinking",
            status,  # type: ignore[arg-type]
        ).status
        == status
    )
