"""Behavioral tests for bounded persistent-terminal input/output actors."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import os
from statistics import quantiles
from threading import Barrier
from types import SimpleNamespace

import pytest

from tldw_chatbook.Terminal import io_actors as io_actors_module
from tldw_chatbook.Terminal.contracts import (
    MAX_COLUMNS,
    MAX_IO_CHUNK_BYTES,
    MAX_PARSER_TURN_BYTES,
    MAX_PARSER_TURN_SECONDS,
    MAX_PASTE_BYTES,
    MAX_PENDING_INPUT_BYTES,
    MAX_PENDING_OUTPUT_BYTES,
    MAX_ROWS,
    MIN_COLUMNS,
    MIN_ROWS,
)
from tldw_chatbook.Terminal.io_actors import (
    InputEventKind,
    InputRefusalReason,
    OutputRefusalReason,
    PasteRefusalReason,
    TerminalInputActor,
    TerminalOutputActor,
    TerminalPriorityControl,
)
from tldw_chatbook.Terminal.screen_model import TerminalScreenModel


BRACKETED_PASTE_START = b"\x1b[200~"
BRACKETED_PASTE_END = b"\x1b[201~"


class ManualClock:
    """Small injected monotonic clock for rate and parser-budget behavior."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _drain_input(actor: TerminalInputActor) -> list[tuple[InputEventKind, bytes]]:
    drained: list[tuple[InputEventKind, bytes]] = []
    while (event := actor.take_nowait()) is not None:
        assert event.encoded_size == len(event.data)
        drained.append((event.kind, event.data))
    return drained


@pytest.mark.asyncio
async def test_actors_consume_shared_validation_model_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def validate_key(data: object) -> SimpleNamespace:
        calls.append(("key", data))
        return SimpleNamespace(data=b"validated-key")

    def validate_paste(text: object, bracketed: object) -> SimpleNamespace:
        calls.append(("paste", (text, bracketed)))
        return SimpleNamespace(
            bracketed=True,
            classify=lambda: (None, b"validated-paste"),
        )

    def validate_reply(data: object) -> SimpleNamespace:
        calls.append(("reply", data))
        return SimpleNamespace(data=b"validated-reply")

    def validate_resize(columns: object, rows: object) -> SimpleNamespace:
        calls.append(("resize", (columns, rows)))
        return SimpleNamespace(columns=80, rows=24)

    def validate_output(data: object) -> SimpleNamespace:
        calls.append(("output", data))
        return SimpleNamespace(data=b"validated-output")

    monkeypatch.setattr(
        io_actors_module, "validate_terminal_key_input", validate_key, raising=False
    )
    monkeypatch.setattr(
        io_actors_module,
        "validate_terminal_paste_input",
        validate_paste,
        raising=False,
    )
    monkeypatch.setattr(
        io_actors_module,
        "validate_terminal_reply_input",
        validate_reply,
        raising=False,
    )
    monkeypatch.setattr(
        io_actors_module,
        "validate_terminal_resize_input",
        validate_resize,
        raising=False,
    )
    monkeypatch.setattr(
        io_actors_module,
        "validate_terminal_output_input",
        validate_output,
        raising=False,
    )

    input_actor = TerminalInputActor()
    output_actor = TerminalOutputActor()
    assert input_actor.offer_key(b"raw-key").accepted is True
    assert input_actor.offer_paste("raw-paste", bracketed=False).accepted is True
    assert input_actor.offer_reply(b"raw-reply").accepted is True
    input_actor.offer_resize(columns=1, rows=1)
    assert output_actor.offer_output(b"raw-output").accepted is True

    assert _drain_input(input_actor) == [
        (InputEventKind.KEY, b"validated-key"),
        (
            InputEventKind.PASTE,
            BRACKETED_PASTE_START + b"validated-paste" + BRACKETED_PASTE_END,
        ),
        (InputEventKind.REPLY, b"validated-reply"),
    ]
    resize = await input_actor.take_resize_debounced()
    assert resize is not None
    assert (resize.columns, resize.rows) == (80, 24)
    consumed: list[bytes] = []
    output_actor.process_parser_turn(consumed.append, visible=False)
    assert consumed == [b"validated-output"]
    assert calls == [
        ("key", b"raw-key"),
        ("paste", ("raw-paste", False)),
        ("reply", b"raw-reply"),
        ("resize", (1, 1)),
        ("output", b"raw-output"),
    ]


def test_key_paste_and_reply_share_one_ordered_byte_counted_queue() -> None:
    actor = TerminalInputActor()

    assert actor.offer_key(b"a").accepted is True
    assert actor.offer_paste("b\n", bracketed=True).accepted is True
    assert actor.offer_reply(b"reply").accepted is True

    assert actor.pending_events == 3
    assert actor.pending_bytes == 1 + 2 + 12 + 5
    assert _drain_input(actor) == [
        (InputEventKind.KEY, b"a"),
        (
            InputEventKind.PASTE,
            BRACKETED_PASTE_START + b"b\n" + BRACKETED_PASTE_END,
        ),
        (InputEventKind.REPLY, b"reply"),
    ]
    assert actor.pending_bytes == 0


def test_input_envelopes_are_frozen_and_carry_precomputed_size() -> None:
    actor = TerminalInputActor()
    assert actor.offer_key(b"key").accepted is True

    event = actor.take_nowait()
    assert event is not None
    assert event.encoded_size == 3
    assert "__dict__" not in event.__slots__
    with pytest.raises(FrozenInstanceError):
        event.encoded_size = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "control",
    [
        "\x00",
        "\x01",
        "\x08",
        "\x0b",
        "\x0c",
        "\x1b",
        "\x1f",
        "\x7f",
        "\x80",
        "\x9f",
    ],
)
def test_prohibited_paste_is_refused_before_any_bytes_are_enqueued(
    control: str,
) -> None:
    private_payload = f"safe{control}PRIVATE-PASTE-CONTENT"
    actor = TerminalInputActor(capacity_bytes=MAX_PENDING_INPUT_BYTES)

    result = actor.offer_paste(private_payload, bracketed=True)

    assert result.accepted is False
    assert result.reason is PasteRefusalReason.PROHIBITED_CONTROL
    assert actor.pending_bytes == 0
    assert actor.take_nowait() is None
    assert "PRIVATE-PASTE-CONTENT" not in result.safe_message
    assert "PRIVATE-PASTE-CONTENT" not in repr(result)


def test_tab_cr_and_lf_remain_allowed_in_one_atomic_paste() -> None:
    actor = TerminalInputActor()

    result = actor.offer_paste("a\tb\rc\nd", bracketed=False)

    assert result.accepted is True
    event = actor.take_nowait()
    assert event is not None
    assert event.kind is InputEventKind.PASTE
    assert event.data == b"a\tb\rc\nd"


def test_bracketed_markers_are_added_only_after_validation_and_credit() -> None:
    payload = "safe"
    encoded_size = (
        len(payload.encode()) + len(BRACKETED_PASTE_START) + len(BRACKETED_PASTE_END)
    )
    exact = TerminalInputActor(capacity_bytes=encoded_size)
    short = TerminalInputActor(capacity_bytes=encoded_size - 1)

    assert exact.offer_paste(payload, bracketed=True).accepted is True
    refused = short.offer_paste(payload, bracketed=True)

    assert refused.reason is PasteRefusalReason.BACKPRESSURE
    assert short.pending_bytes == 0
    assert short.take_nowait() is None


def test_paste_limit_counts_replacement_encoded_payload_bytes_atomically() -> None:
    actor = TerminalInputActor()

    accepted = actor.offer_paste("a" * MAX_PASTE_BYTES, bracketed=False)
    actor.take_nowait()
    refused = actor.offer_paste("a" * (MAX_PASTE_BYTES + 1), bracketed=False)

    assert accepted.accepted is True
    assert refused.reason is PasteRefusalReason.TOO_LARGE
    assert actor.pending_bytes == 0


def test_invalid_unicode_is_replaced_before_transport_byte_accounting() -> None:
    actor = TerminalInputActor(capacity_bytes=1)

    refused = actor.offer_paste("\ud800\ud800", bracketed=False)

    assert refused.reason is PasteRefusalReason.BACKPRESSURE
    assert actor.pending_bytes == 0


def test_key_backpressure_never_truncates_or_claims_delivery() -> None:
    actor = TerminalInputActor(capacity_bytes=3)

    assert actor.offer_key(b"abc").accepted is True
    refused = actor.offer_key(b"d")

    assert refused.reason is InputRefusalReason.BACKPRESSURE
    assert actor.pending_bytes == 3
    assert _drain_input(actor) == [(InputEventKind.KEY, b"abc")]


def test_empty_input_offers_are_noops_instead_of_unbounded_envelopes() -> None:
    actor = TerminalInputActor(capacity_bytes=1)

    for _ in range(1_000):
        assert actor.offer_key(b"").accepted is True
        assert actor.offer_paste("", bracketed=False).accepted is True
        assert actor.offer_reply(b"").accepted is True

    assert actor.pending_bytes == 0
    assert actor.pending_events == 0


def test_fixed_replies_are_individually_bounded_and_rate_limited() -> None:
    clock = ManualClock()
    actor = TerminalInputActor(clock=clock)

    oversized = actor.offer_reply(b"x" * 257)
    assert oversized.reason is InputRefusalReason.REPLY_TOO_LARGE

    for _ in range(16):
        assert actor.offer_reply(b"x" * 256).accepted is True
    limited = actor.offer_reply(b"y")
    assert limited.reason is InputRefusalReason.REPLY_RATE_LIMIT
    assert actor.pending_bytes == 4 * 1024

    clock.advance(1.0)
    assert actor.offer_reply(b"y").accepted is True


@pytest.mark.asyncio
async def test_resize_is_latest_only_and_debounced_at_least_one_loop_turn() -> None:
    actor = TerminalInputActor()
    actor.offer_resize(columns=80, rows=24)
    actor.offer_resize(columns=120, rows=40)
    loop_advanced = False

    async def mark_next_turn() -> None:
        nonlocal loop_advanced
        await asyncio.sleep(0)
        loop_advanced = True

    marker = asyncio.create_task(mark_next_turn())
    resize = await actor.take_resize_debounced()
    await marker

    assert loop_advanced is True
    assert resize is not None
    assert (resize.columns, resize.rows) == (120, 40)
    assert await actor.take_resize_debounced() is None
    assert actor.pending_bytes == 0


@pytest.mark.parametrize(
    ("columns", "rows"),
    [
        (MIN_COLUMNS - 1, MIN_ROWS),
        (MAX_COLUMNS + 1, MIN_ROWS),
        (MIN_COLUMNS, MIN_ROWS - 1),
        (MIN_COLUMNS, MAX_ROWS + 1),
    ],
)
def test_resize_refuses_dimensions_outside_the_terminal_contract(
    columns: int, rows: int
) -> None:
    actor = TerminalInputActor()

    with pytest.raises(ValueError, match="outside contract"):
        actor.offer_resize(columns=columns, rows=rows)


def test_input_credit_race_admits_only_complete_events() -> None:
    workers = 16
    payload = b"x" * 64
    actor = TerminalInputActor(capacity_bytes=len(payload))
    barrier = Barrier(workers)

    def offer() -> bool:
        barrier.wait()
        return actor.offer_key(payload).accepted

    with ThreadPoolExecutor(max_workers=workers) as pool:
        admitted = list(pool.map(lambda _: offer(), range(workers)))

    assert admitted.count(True) == 1
    assert actor.pending_bytes == len(payload)
    assert actor.pending_events == 1


def test_output_chunks_and_total_credit_are_bounded_without_dropping_state() -> None:
    actor = TerminalOutputActor()

    oversized = actor.offer_output(b"x" * (MAX_IO_CHUNK_BYTES + 1))
    assert oversized.reason is OutputRefusalReason.CHUNK_TOO_LARGE
    assert actor.pending_bytes == 0

    chunk = b"x" * MAX_IO_CHUNK_BYTES
    for _ in range(MAX_PENDING_OUTPUT_BYTES // MAX_IO_CHUNK_BYTES):
        assert actor.offer_output(chunk).accepted is True

    assert actor.pending_bytes == MAX_PENDING_OUTPUT_BYTES
    assert actor.read_credit_bytes == 0
    assert actor.next_read_size == 0
    refused = actor.offer_output(b"y")
    assert refused.reason is OutputRefusalReason.BACKPRESSURE
    assert actor.pending_bytes == MAX_PENDING_OUTPUT_BYTES


def test_output_credit_race_never_exceeds_capacity() -> None:
    workers = 16
    payload = b"x" * 64
    actor = TerminalOutputActor(capacity_bytes=len(payload), max_chunk_bytes=64)
    barrier = Barrier(workers)

    def offer() -> bool:
        barrier.wait()
        return actor.offer_output(payload).accepted

    with ThreadPoolExecutor(max_workers=workers) as pool:
        admitted = list(pool.map(lambda _: offer(), range(workers)))

    assert admitted.count(True) == 1
    assert actor.pending_bytes == len(payload)
    assert actor.pending_chunks == 1


def test_parser_turn_splits_a_chunk_at_the_exact_byte_budget() -> None:
    actor = TerminalOutputActor(
        capacity_bytes=64,
        max_chunk_bytes=64,
        max_turn_bytes=5,
    )
    consumed: list[bytes] = []
    assert actor.offer_output(b"abcdefgh").accepted is True

    first = actor.process_parser_turn(consumed.append, visible=True)
    second = actor.process_parser_turn(consumed.append, visible=True)

    assert consumed == [b"abcde", b"fgh"]
    assert first.processed_bytes == 5
    assert first.pending_bytes == 3
    assert first.refresh_requested is True
    assert second.processed_bytes == 3
    assert second.pending_bytes == 0
    assert second.refresh_requested is False


def test_parser_turn_stops_after_injected_eight_millisecond_budget() -> None:
    clock = ManualClock()
    actor = TerminalOutputActor(
        capacity_bytes=32,
        max_chunk_bytes=8,
        max_turn_bytes=32,
        max_turn_seconds=MAX_PARSER_TURN_SECONDS,
        clock=clock,
    )
    for value in (b"aaaaaaaa", b"bbbbbbbb", b"cccccccc"):
        assert actor.offer_output(value).accepted is True
    consumed: list[bytes] = []

    def consume(value: bytes) -> None:
        consumed.append(value)
        clock.advance(MAX_PARSER_TURN_SECONDS)

    result = actor.process_parser_turn(consume, visible=False)

    assert consumed == [b"aaaaaaaa"]
    assert result.processed_bytes == 8
    assert result.pending_bytes == 16
    assert result.refresh_requested is False


def test_parser_turn_observes_time_budget_with_cost_proportional_to_bytes() -> None:
    clock = ManualClock()
    actor = TerminalOutputActor(
        capacity_bytes=8 * 1024,
        max_chunk_bytes=8 * 1024,
        max_turn_bytes=8 * 1024,
        max_turn_seconds=MAX_PARSER_TURN_SECONDS,
        clock=clock,
    )
    consumed: list[bytes] = []
    assert actor.offer_output(b"x" * (8 * 1024)).accepted is True

    def consume(value: bytes) -> None:
        consumed.append(value)
        clock.advance((len(value) / 1024) * 0.004)

    result = actor.process_parser_turn(consume, visible=False)

    assert result.processed_bytes <= 2 * 1024
    assert result.pending_bytes >= 6 * 1024


def test_parser_exception_retires_ambiguous_slice_without_replaying_it() -> None:
    actor = TerminalOutputActor(
        capacity_bytes=8,
        max_chunk_bytes=8,
        max_turn_bytes=3,
    )
    assert actor.offer_output(b"abcdef").accepted is True

    with pytest.raises(RuntimeError, match="parser failed"):
        actor.process_parser_turn(
            lambda _value: (_ for _ in ()).throw(RuntimeError("parser failed")),
            visible=False,
        )

    assert actor.pending_bytes == 3
    consumed: list[bytes] = []
    result = actor.process_parser_turn(consumed.append, visible=False)
    assert consumed == [b"def"]
    assert result.pending_bytes == 0


def test_parser_turn_never_exceeds_global_byte_or_time_defaults() -> None:
    actor = TerminalOutputActor()
    chunk = b"x" * MAX_IO_CHUNK_BYTES
    for _ in range(8):
        assert actor.offer_output(chunk).accepted is True

    result = actor.process_parser_turn(lambda _: None, visible=False)

    assert result.processed_bytes <= MAX_PARSER_TURN_BYTES
    assert result.pending_bytes >= MAX_PENDING_OUTPUT_BYTES - MAX_PARSER_TURN_BYTES


def test_visible_refresh_is_coalesced_until_the_frame_is_acknowledged() -> None:
    actor = TerminalOutputActor(capacity_bytes=8, max_chunk_bytes=8)
    assert actor.offer_output(b"a").accepted is True
    first = actor.process_parser_turn(lambda _: None, visible=True)
    assert actor.offer_output(b"b").accepted is True
    second = actor.process_parser_turn(lambda _: None, visible=True)

    assert first.refresh_requested is True
    assert second.refresh_requested is False
    assert actor.acknowledge_visible_refresh() is True
    assert actor.acknowledge_visible_refresh() is False

    assert actor.offer_output(b"c").accepted is True
    third = actor.process_parser_turn(lambda _: None, visible=True)
    assert third.refresh_requested is True


def test_hidden_output_parses_without_scheduling_a_refresh() -> None:
    actor = TerminalOutputActor(capacity_bytes=8, max_chunk_bytes=8)
    assert actor.offer_output(b"hidden").accepted is True

    result = actor.process_parser_turn(lambda _: None, visible=False)

    assert result.processed_bytes == len(b"hidden")
    assert result.refresh_requested is False
    assert actor.acknowledge_visible_refresh() is False


def test_priority_close_is_independent_and_idempotent_when_both_paths_are_full() -> (
    None
):
    input_actor = TerminalInputActor(capacity_bytes=1)
    output_actor = TerminalOutputActor(capacity_bytes=1, max_chunk_bytes=1)
    priority = TerminalPriorityControl()
    assert input_actor.offer_key(b"x").accepted is True
    assert output_actor.offer_output(b"y").accepted is True

    assert priority.request_priority_close() is True
    assert priority.request_priority_close() is False
    assert priority.requested is True
    assert priority.wait(timeout=0) is True
    assert input_actor.pending_bytes == 1
    assert output_actor.pending_bytes == 1


@pytest.mark.asyncio
async def test_ten_second_ansi_flood_keeps_actors_bounded_and_reports_latency(
    request: pytest.FixtureRequest,
) -> None:
    actor = TerminalOutputActor()
    model = TerminalScreenModel(columns=80, rows=24)
    loop = asyncio.get_running_loop()
    duration = 10.0
    stop_at = loop.time() + duration
    sentinel_lateness: list[float] = []
    payload = (b"\x1b[32mterminal-flood\x1b[0m\r\n" * 2_048)[:MAX_IO_CHUNK_BYTES]
    assert actor.offer_output(payload).accepted is True

    async def sentinel() -> None:
        target = loop.time() + 0.1
        while target < stop_at:
            await asyncio.sleep(max(0.0, target - loop.time()))
            sentinel_lateness.append(max(0.0, loop.time() - target))
            target += 0.1

    sentinel_task = asyncio.create_task(sentinel())
    while loop.time() < stop_at:
        if actor.next_read_size >= len(payload):
            assert actor.offer_output(payload).accepted is True
        actor.process_parser_turn(model.feed, visible=False)
        assert actor.pending_bytes <= MAX_PENDING_OUTPUT_BYTES
        await asyncio.sleep(0)
    await sentinel_task

    assert len(sentinel_lateness) >= 90
    p95 = quantiles(sentinel_lateness, n=100, method="inclusive")[94]
    request.node.user_properties.append(("terminal_ansi_flood_p95_ms", p95 * 1_000))
    if os.environ.get("TLDW_TERMINAL_QUALIFICATION_HOST") == "1":
        assert p95 < 0.1
