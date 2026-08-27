import pytest

from tldw_chatbook.Chat.llamacpp_think_filter import (
    StartAnchoredThinkFilter,
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

_UNCONFIRMED_PREFIX_CASES = [
    pytest.param(raw, cut, id=f"{opening[1:-1]}-{length}-{leading!r}-{cut}")
    for opening in ("<think>", "<thinking>")
    for length in range(1, len(opening))
    for leading in ("", " \n\t")
    for raw in (leading + opening[:length],)
    for cut in range(len(raw) + 1)
]

_SURROGATE_CASES = [
    pytest.param(surrogate, raw, cut, id=f"{kind}-{position}-{cut}")
    for kind, surrogate in (("high", "\ud800"), ("low", "\udfff"))
    for position, raw in (
        ("start", f"<think>{surrogate}tail</think>answer"),
        ("middle", f"<think>head{surrogate}tail</think>answer"),
        ("close-prefix", f"<think>head</thi{surrogate}tail"),
    )
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
            ThinkSplitChunk(status="complete"),
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
    ("opening", "closing"),
    [("<think>", "</think>"), ("<thinking>", "</thinking>")],
)
@pytest.mark.parametrize(
    "whitespace",
    [" " * 21, " \n\t" * 10_000],
    ids=("above-old-cap", "well-beyond-old-cap"),
)
def test_leading_whitespace_never_exposes_later_anchored_thinking(
    opening: str, closing: str, whitespace: str
) -> None:
    raw = f"{whitespace}{opening}secret{closing}answer"
    partitionings = (
        (raw,),
        (raw[:20], raw[20:]),
        tuple(raw[index : index + 7] for index in range(0, len(raw), 7)),
    )

    for chunks in partitionings:
        assert _split_chunks(*chunks) == ThinkSplitChunk(
            thinking="secret",
            content="answer",
            status="complete",
        )


@pytest.mark.parametrize(("raw", "cut"), _UNCONFIRMED_PREFIX_CASES)
def test_unconfirmed_opener_prefix_at_eof_is_complete_no_evidence(
    raw: str, cut: int
) -> None:
    assert _split_chunks(raw[:cut], raw[cut:]) == ThinkSplitChunk(status="complete")


@pytest.mark.parametrize(("raw", "cut"), _UNCONFIRMED_PREFIX_CASES)
def test_compatibility_filter_drops_unconfirmed_opener_prefix(
    raw: str, cut: int
) -> None:
    stream_filter = StartAnchoredThinkFilter()
    content = stream_filter.feed(raw[:cut]) + stream_filter.feed(raw[cut:])
    assert content + stream_filter.flush() == ""


def test_long_leading_whitespace_is_discarded_without_growing_probe_state() -> None:
    whitespace = " \n\t" * 100_000
    splitter = StartAnchoredThinkSplitter()

    update = splitter.feed(whitespace)

    assert update == ThinkSplitChunk()
    assert max(
        len(value) for value in vars(splitter).values() if isinstance(value, str)
    ) <= len("</thinking>")
    assert splitter.flush() == ThinkSplitChunk(status="complete")


def test_long_leading_whitespace_keeps_probe_open_for_later_thinking() -> None:
    whitespace = " " * 100
    splitter = StartAnchoredThinkSplitter()

    assert splitter.feed(whitespace) == ThinkSplitChunk()
    assert splitter.feed("<think>reason</think>answer") == ThinkSplitChunk(
        thinking="reason",
        content="answer",
    )
    assert splitter.flush().status == "complete"


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
    ids=("ascii", "utf8"),
)
def test_splitter_rejects_thinking_byte_overflow_without_echoing_text(
    thinking: str,
) -> None:
    with pytest.raises(ThinkingEnvelopeValidationError) as raised:
        split_start_anchored_thinking(f"<think>{thinking}</think>answer")
    assert thinking[:100] not in str(raised.value)


@pytest.mark.parametrize(
    "thinking",
    [
        "PRIVATE_ASCII_CANARY" + "x" * MAX_THINKING_TEXT_BYTES,
        "PRIVATE_UTF8_CANARY" + "é" * (MAX_THINKING_TEXT_BYTES // 2),
    ],
    ids=("ascii", "utf8"),
)
def test_one_chunk_overflow_clears_private_state_and_closes_splitter(
    thinking: str,
) -> None:
    splitter = StartAnchoredThinkSplitter()

    with pytest.raises(ThinkingEnvelopeValidationError) as raised:
        splitter.feed(f"<think>{thinking}</think>visible")

    assert thinking[:20] not in str(raised.value)
    assert all(
        thinking[:20] not in value
        for value in vars(splitter).values()
        if isinstance(value, str)
    )
    # The first frame is this caller and necessarily owns the test input.
    # Parser frames below it must not retain another reference after failure.
    traceback = raised.value.__traceback__.tb_next
    while traceback is not None:
        assert all(
            thinking[:20] not in value
            for value in traceback.tb_frame.f_locals.values()
            if isinstance(value, str)
        )
        traceback = traceback.tb_next
    assert max(
        len(value) for value in vars(splitter).values() if isinstance(value, str)
    ) <= len("</thinking>")
    assert splitter.flush() == ThinkSplitChunk(status="failed")
    assert splitter.flush() == ThinkSplitChunk(status="failed")
    with pytest.raises(RuntimeError, match="already closed"):
        splitter.feed("later")


@pytest.mark.parametrize(
    ("exact_limit", "overflow"),
    [
        ("x" * MAX_THINKING_TEXT_BYTES, "A"),
        ("é" * (MAX_THINKING_TEXT_BYTES // 2), "é"),
    ],
    ids=("ascii", "utf8"),
)
def test_split_overflow_is_cumulative_and_terminal(
    exact_limit: str, overflow: str
) -> None:
    splitter = StartAnchoredThinkSplitter()
    assert splitter.feed("<think>" + exact_limit).thinking == exact_limit

    with pytest.raises(ThinkingEnvelopeValidationError):
        splitter.feed(overflow)

    assert all(
        overflow not in value
        for value in vars(splitter).values()
        if isinstance(value, str)
    )
    assert splitter.flush() == ThinkSplitChunk(status="failed")
    with pytest.raises(RuntimeError, match="already closed"):
        splitter.feed("answer")


@pytest.mark.parametrize(
    "exact_limit",
    [
        "PRIVATE_ASCII_CANARY" + "x" * (MAX_THINKING_TEXT_BYTES - 20),
        "PRIVATE_UTF8_CANARY_" + "é" * ((MAX_THINKING_TEXT_BYTES - 20) // 2),
    ],
    ids=("ascii", "utf8"),
)
def test_helper_clears_private_locals_when_held_suffix_overflows_at_flush(
    exact_limit: str,
) -> None:
    raw = f"<think>{exact_limit}<"

    with pytest.raises(ThinkingEnvelopeValidationError) as raised:
        split_start_anchored_thinking(raw)

    canary = exact_limit[:20]
    # The first frame is this caller and necessarily owns the test input.
    traceback = raised.value.__traceback__.tb_next
    while traceback is not None:
        for value in traceback.tb_frame.f_locals.values():
            if isinstance(value, str):
                assert canary not in value
            assert not (isinstance(value, ThinkSplitChunk) and canary in value.thinking)
        traceback = traceback.tb_next


@pytest.mark.parametrize(("surrogate", "raw", "cut"), _SURROGATE_CASES)
def test_surrogates_fail_content_free_at_every_position_and_partition(
    surrogate: str, raw: str, cut: int
) -> None:
    splitter = StartAnchoredThinkSplitter()

    with pytest.raises(ThinkingEnvelopeValidationError) as raised:
        splitter.feed(raw[:cut])
        splitter.feed(raw[cut:])

    assert surrogate not in str(raised.value)
    assert surrogate not in repr(raised.value)
    assert all(
        surrogate not in value
        for value in vars(splitter).values()
        if isinstance(value, str)
    )
    assert splitter.flush() == ThinkSplitChunk(status="failed")
    assert splitter.flush() == ThinkSplitChunk(status="failed")
    with pytest.raises(RuntimeError, match="already closed"):
        splitter.feed("later")


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"], ids=("high", "low"))
def test_surrogate_after_held_close_prefix_clears_buffer(surrogate: str) -> None:
    splitter = StartAnchoredThinkSplitter()
    assert splitter.feed("<think>reason</thi").thinking == "reason"
    assert splitter._buffer == "</thi"

    with pytest.raises(ThinkingEnvelopeValidationError):
        splitter.feed(surrogate)

    assert splitter._buffer == ""
    assert splitter.flush() == ThinkSplitChunk(status="failed")


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"], ids=("high", "low"))
def test_surrogate_failure_traceback_retains_no_private_text(surrogate: str) -> None:
    raw = f"<think>PRIVATE_CANARY{surrogate}tail</think>"

    with pytest.raises(ThinkingEnvelopeValidationError) as raised:
        split_start_anchored_thinking(raw)

    # The first frame is this caller and necessarily owns the test input.
    traceback = raised.value.__traceback__.tb_next
    while traceback is not None:
        for value in traceback.tb_frame.f_locals.values():
            if isinstance(value, str):
                assert surrogate not in value
                assert "PRIVATE_CANARY" not in value
            elif isinstance(value, ThinkSplitChunk):
                assert surrogate not in value.thinking
                assert "PRIVATE_CANARY" not in value.thinking
            elif isinstance(value, StartAnchoredThinkSplitter):
                assert all(
                    surrogate not in item and "PRIVATE_CANARY" not in item
                    for item in vars(value).values()
                    if isinstance(item, str)
                )
        traceback = traceback.tb_next


def test_valid_astral_text_preserves_exact_utf8_byte_cap() -> None:
    thinking = chr(0x1F600) * (MAX_THINKING_TEXT_BYTES // 4)
    result = split_start_anchored_thinking(f"<think>{thinking}</think>answer")

    assert result == ThinkSplitChunk(
        thinking=thinking,
        content="answer",
        status="complete",
    )


def test_split_chunk_repr_hides_captured_values() -> None:
    chunk = ThinkSplitChunk(thinking="private canary", content="visible canary")
    assert "private canary" not in repr(chunk)
    assert "visible canary" not in repr(chunk)
