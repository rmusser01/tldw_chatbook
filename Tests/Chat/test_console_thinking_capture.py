"""Round-owned capture of explicit provider thinking evidence."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_provider_gateway import (
    ProviderProprietaryThinkingEvidence,
    ProviderThinkingCaptureError,
    ProviderThinkingDelta,
    ProviderToolCalls,
)
from tldw_chatbook.Chat.console_thinking_capture import ThinkingCapture
from tldw_chatbook.Chat.thinking_blocks import (
    MAX_THINKING_BLOCK_ID_CHARS,
    MAX_THINKING_TEXT_BYTES,
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
)


def _delta(text: str = "reason") -> ProviderThinkingDelta:
    return ProviderThinkingDelta(
        text=text,
        provider="llama_cpp",
        model="reasoner",
        protocol="chat_completions",
        source_format="start_anchored_think",
    )


def _proprietary() -> ProviderProprietaryThinkingEvidence:
    return ProviderProprietaryThinkingEvidence(
        provider="moonshot",
        model="kimi",
        protocol="chat_completions",
        source_format="reasoning_content",
    )


def test_multiple_deltas_form_one_round_owned_block() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")

    first = capture.observe(_delta("first "))
    second = capture.observe(_delta("second"))
    settled = capture.settle("complete")

    assert first.changed_block_id == second.changed_block_id
    assert settled.envelope is not None
    assert len(settled.envelope.blocks) == 1
    block = settled.envelope.blocks[0]
    assert isinstance(block, DisplayableThinkingBlock)
    assert block.text == "first second"
    assert block.round_ordinal == 0
    assert block.status == "complete"


def test_tool_boundary_advances_round_and_preserves_order() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")

    capture.observe(_delta("plan"))
    boundary = capture.observe(ProviderToolCalls(()))
    capture.observe(_delta("result"))
    settled = capture.settle("complete")

    assert boundary.collapse_boundary_reached is True
    assert settled.envelope is not None
    assert [block.round_ordinal for block in settled.envelope.blocks] == [0, 1]
    assert [block.text for block in settled.envelope.blocks] == ["plan", "result"]
    assert settled.envelope.blocks[0].block_id != settled.envelope.blocks[1].block_id


def test_proprietary_occurrence_is_deduplicated_per_round_without_text() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")

    first = capture.observe(_proprietary())
    second = capture.observe(_proprietary())
    settled = capture.settle("complete")

    assert first.changed_block_id == second.changed_block_id
    assert settled.envelope is not None
    assert len(settled.envelope.blocks) == 1
    assert isinstance(settled.envelope.blocks[0], ProprietaryThinkingBlock)
    assert not hasattr(settled.envelope.blocks[0], "text")


def test_answer_first_records_collapse_boundary_without_fabricating_evidence() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")

    boundary = capture.observe("answer")
    settled = capture.settle("complete")

    assert boundary.collapse_boundary_reached is True
    assert boundary.changed_block_id is None
    assert settled.envelope is None


def test_answer_boundary_is_reported_only_once_per_round() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")
    capture.observe(_delta())

    first = capture.observe("first answer chunk")
    second = capture.observe("second answer chunk")
    tool = capture.observe(ProviderToolCalls(()))

    assert first.collapse_boundary_reached is True
    assert second.collapse_boundary_reached is False
    assert tool.collapse_boundary_reached is False


def test_tool_first_records_boundary_and_next_evidence_belongs_to_next_round() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")

    boundary = capture.observe(ProviderToolCalls(()))
    capture.observe(_delta())
    settled = capture.settle("complete")

    assert boundary.envelope is None
    assert boundary.collapse_boundary_reached is True
    assert settled.envelope is not None
    assert settled.envelope.blocks[0].round_ordinal == 1


def test_terminal_only_proprietary_event_is_actual_evidence() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")

    capture.observe(_proprietary())
    settled = capture.settle("complete")

    assert settled.envelope is not None
    assert settled.envelope.blocks[0].status == "complete"
    assert settled.collapse_boundary_reached is True


def test_evidence_after_answer_boundary_is_immediately_collapsed_once() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")
    capture.observe("answer")

    first = capture.observe(_proprietary())
    duplicate = capture.observe(_proprietary())
    settled = capture.settle("complete")

    assert first.collapse_boundary_reached is True
    assert duplicate.collapse_boundary_reached is False
    assert settled.collapse_boundary_reached is False


@pytest.mark.parametrize("outcome", ["stopped", "failed"])
def test_noncomplete_terminal_marks_only_open_round(outcome: str) -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")
    capture.observe(_delta("closed round"))
    capture.observe(ProviderToolCalls(()))
    capture.observe(_delta("open round"))

    settled = capture.settle(outcome)

    assert settled.envelope is not None
    assert [block.status for block in settled.envelope.blocks] == [
        "complete",
        outcome,
    ]


def test_capture_rejects_cumulative_text_overflow_without_exposing_content() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")
    capture.observe(_delta("a" * MAX_THINKING_TEXT_BYTES))

    with pytest.raises(
        ProviderThinkingCaptureError, match="Provider thinking capture failed"
    ) as error:
        capture.observe(_delta("b"))

    assert "aaaa" not in str(error.value)


def test_capture_rejects_block_overflow() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")
    for _ in range(32):
        capture.observe(_proprietary())
        capture.observe(ProviderToolCalls(()))

    with pytest.raises(
        ProviderThinkingCaptureError, match="Provider thinking capture failed"
    ):
        capture.observe(_proprietary())


def test_block_ids_are_capture_unique_bounded_and_provider_text_free() -> None:
    first = ThinkingCapture(assistant_owner_id="assistant-1")
    second = ThinkingCapture(assistant_owner_id="assistant-1")
    other = ThinkingCapture(assistant_owner_id="assistant-2")

    first_id = first.observe(_delta()).changed_block_id
    second_id = second.observe(_delta()).changed_block_id
    other_id = other.observe(_delta()).changed_block_id

    assert len({first_id, second_id, other_id}) == 3
    assert first_id is not None
    assert first_id.isascii()
    assert len(first_id) <= MAX_THINKING_BLOCK_ID_CHARS
    assert "llama_cpp" not in str(first_id)


def test_no_event_means_no_recorded_evidence() -> None:
    capture = ThinkingCapture(assistant_owner_id="assistant-1")
    capture.observe("answer")

    assert capture.settle("complete").envelope is None
