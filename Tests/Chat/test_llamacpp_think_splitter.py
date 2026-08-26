import pytest

from tldw_chatbook.Chat.llamacpp_think_filter import (
    StartAnchoredThinkSplitter,
    ThinkSplitChunk,
    split_start_anchored_thinking,
)
from tldw_chatbook.Chat.thinking_blocks import (
    MAX_THINKING_TEXT_BYTES,
    ThinkingEnvelopeValidationError,
)


_BOUNDARY_CASES = [
    pytest.param(raw, cut, id=f"{opening[1:-1]}-{cut}")
    for opening, closing in (
        ("<think>", "</think>"),
        ("<thinking>", "</thinking>"),
    )
    for raw in (f"{opening}reason{closing}answer",)
    for cut in range(len(raw) + 1)
]


def _split_chunks(*chunks: str) -> ThinkSplitChunk:
    splitter = StartAnchoredThinkSplitter()
    updates = [splitter.feed(chunk) for chunk in chunks]
    updates.append(splitter.flush())
    return ThinkSplitChunk(
        thinking="".join(update.thinking for update in updates),
        content="".join(update.content for update in updates),
        status=updates[-1].status,
    )


@pytest.mark.parametrize(("raw", "cut"), _BOUNDARY_CASES)
def test_splitter_is_chunk_boundary_invariant(raw: str, cut: int) -> None:
    assert _split_chunks(raw[:cut], raw[cut:]) == ThinkSplitChunk(
        thinking="reason",
        content="answer",
        status="complete",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            " \n\n<think>reason</think>answer",
            ThinkSplitChunk(thinking="reason", content="answer", status="complete"),
        ),
        (
            "<thinking></thinking>answer",
            ThinkSplitChunk(content="answer", status="complete"),
        ),
        (
            "answer with <think>literal</think> markup",
            ThinkSplitChunk(
                content="answer with <think>literal</think> markup",
                status="complete",
            ),
        ),
        (
            "<thin",
            ThinkSplitChunk(status="failed"),
        ),
        (
            "<think>reason</thinking>answer",
            ThinkSplitChunk(
                thinking="reason</thinking>answer",
                status="failed",
            ),
        ),
        (
            "<thinking>reason</think>answer",
            ThinkSplitChunk(
                thinking="reason</think>answer",
                status="failed",
            ),
        ),
        (
            "<think>reason",
            ThinkSplitChunk(thinking="reason", status="failed"),
        ),
    ],
)
def test_splitter_classifies_edge_cases(raw: str, expected: ThinkSplitChunk) -> None:
    assert _split_chunks(raw) == expected


def test_answer_in_later_chunk_and_post_close_newlines_are_partition_invariant() -> (
    None
):
    expected = ThinkSplitChunk(
        thinking="reason",
        content="answer",
        status="complete",
    )
    assert _split_chunks("<think>reason</think>", "\n", "\nanswer") == expected
    assert _split_chunks("<think>reason</think>\n\nanswer") == expected


def test_visible_answer_makes_all_later_tags_literal() -> None:
    assert _split_chunks("answer", "<think>literal</think>") == ThinkSplitChunk(
        content="answer<think>literal</think>",
        status="complete",
    )


@pytest.mark.parametrize(
    "raw",
    [
        "<think>reason</think>answer",
        "<thinking>reason</thinking>answer",
        " \n<think>reason</think>\nanswer",
        "answer with <think>literal</think>",
        "<think>unfinished",
        "<thin",
    ],
)
def test_non_streaming_helper_matches_every_two_chunk_partition(raw: str) -> None:
    expected = split_start_anchored_thinking(raw)
    for cut in range(len(raw) + 1):
        assert _split_chunks(raw[:cut], raw[cut:]) == expected


def test_splitter_rejects_non_string_chunks() -> None:
    with pytest.raises(TypeError, match="chunks must be strings"):
        StartAnchoredThinkSplitter().feed(1)  # type: ignore[arg-type]


def test_splitter_accepts_exact_thinking_byte_limit() -> None:
    raw = f"<think>{'x' * MAX_THINKING_TEXT_BYTES}</think>answer"
    assert split_start_anchored_thinking(raw).status == "complete"


@pytest.mark.parametrize(
    "thinking",
    [
        "x" * (MAX_THINKING_TEXT_BYTES + 1),
        "é" * (MAX_THINKING_TEXT_BYTES // 2 + 1),
    ],
)
def test_splitter_rejects_thinking_byte_overflow_without_echoing_text(
    thinking: str,
) -> None:
    with pytest.raises(ThinkingEnvelopeValidationError) as raised:
        split_start_anchored_thinking(f"<think>{thinking}</think>answer")
    assert thinking[:100] not in str(raised.value)


def test_split_chunk_repr_hides_captured_values() -> None:
    chunk = ThinkSplitChunk(thinking="private canary", content="visible canary")
    assert "private canary" not in repr(chunk)
    assert "visible canary" not in repr(chunk)
