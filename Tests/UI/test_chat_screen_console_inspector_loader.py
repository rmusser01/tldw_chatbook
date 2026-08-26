"""Unit tests for ``ChatScreen``'s Conversation Inspector ``exchanges_loader``
factory (task-8 review finding 6).

``_build_console_inspector_exchanges_loader`` (chat_screen.py) is a
standalone, module-level function -- extracted from what used to be a
method-local closure specifically so these branches are testable without
mounting a ``ChatScreen`` (which would otherwise require resolving
``_active_console_provider_model_display`` on a bare screen with no active
session, an unrelated and much heavier dependency chain). No Textual
App/pilot/mount anywhere in this file -- these are pure async unit tests
over the returned loader callable.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureCorruptError,
    ExchangeCapture,
    capture_from_storage,
    capture_to_blob,
)
from tldw_chatbook.UI.Screens.chat_screen import (
    ChatScreen,
    _build_console_inspector_exchanges_loader,
)


def _capture(seq: int = 1, run_tag: str = "run-1", model: str = "m") -> ExchangeCapture:
    return ExchangeCapture(
        run_tag=run_tag,
        seq=seq,
        created_at="2026-08-20T10:00:00Z",
        provider="anthropic",
        model=model,
        endpoint=None,
        request={"model": model},
        response={"id": "resp-1"},
        status="complete",
        usage_json=None,
        omitted_keys=(),
    )


def _message(
    *,
    native_id: str = "n1",
    persisted_message_id: str | None = "p1",
    exchanges: tuple[ExchangeCapture, ...] = (),
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="hi",
        id=native_id,
        persisted_message_id=persisted_message_id,
        exchanges=exchanges,
    )


def _raising_db_accessor() -> None:
    raise AssertionError(
        "DB should not have been accessed -- this branch must resolve "
        "without touching the DB"
    )


class _FakeExchangesDB:
    """Records every call so a test can assert it was (or wasn't) hit."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.calls: list[str] = []

    def get_message_exchanges(self, message_id: str) -> list[dict]:
        self.calls.append(message_id)
        return self._rows


@pytest.mark.asyncio
async def test_native_captures_win_without_touching_the_db() -> None:
    """(a) A message with in-memory ``exchanges`` returns them directly --
    the DB accessor must never even be CALLED, not just "called and
    ignored". Using an accessor that raises on any call is the proof."""
    native_capture = _capture(model="native-model")
    message = _message(native_id="n1", exchanges=(native_capture,))
    loader = _build_console_inspector_exchanges_loader(
        {"n1": message}, _raising_db_accessor
    )

    result = await loader("n1")

    assert result == [(native_capture, False)]


@pytest.mark.asyncio
async def test_native_captures_resolve_abandoned_via_the_optional_accessor() -> None:
    """task-9: when ``abandoned_run_tags_for`` IS supplied, a native
    capture's ``abandoned`` flag comes from it (keyed by ``run_tag``)
    rather than always ``False``."""
    abandoned_capture = _capture(model="native-model", run_tag="r-abandoned")
    kept_capture = _capture(model="native-model-2", run_tag="r-kept", seq=2)
    message = _message(
        native_id="n1", exchanges=(abandoned_capture, kept_capture)
    )
    loader = _build_console_inspector_exchanges_loader(
        {"n1": message},
        _raising_db_accessor,
        lambda _native_message_id: frozenset({"r-abandoned"}),
    )

    result = await loader("n1")

    # ExchangeCapture is unhashable (its request/response fields are
    # dicts) -- key off run_tag rather than using captures as dict keys.
    abandoned_by_run_tag = {capture.run_tag: abandoned for capture, abandoned in result}
    assert abandoned_by_run_tag == {"r-abandoned": True, "r-kept": False}


@pytest.mark.asyncio
async def test_native_captures_default_to_not_abandoned_without_an_accessor() -> None:
    """Omitting ``abandoned_run_tags_for`` (the default) preserves task-8's
    original behavior -- every native capture reports ``abandoned=False``."""
    native_capture = _capture(model="native-model")
    message = _message(native_id="n1", exchanges=(native_capture,))
    loader = _build_console_inspector_exchanges_loader(
        {"n1": message}, _raising_db_accessor
    )

    result = await loader("n1")

    assert result == [(native_capture, False)]


@pytest.mark.asyncio
async def test_db_fallback_decodes_captures_via_capture_from_blob() -> None:
    """(b) No native captures, but a persisted id -- falls back to the DB,
    decoding each row's ``capture_blob`` through ``capture_from_blob`` and
    pairing it with that row's real ``abandoned`` column."""
    db_capture = _capture(model="db-model", seq=3)
    db = _FakeExchangesDB(
        rows=[
            {
                "run_tag": "run-1",
                "seq": 3,
                "status": "complete",
                "abandoned": True,
                "capture_blob": capture_to_blob(db_capture),
                "created_at": "2026-08-20T10:00:00Z",
            }
        ]
    )
    message = _message(native_id="n1", persisted_message_id="p1", exchanges=())
    loader = _build_console_inspector_exchanges_loader(
        {"n1": message}, lambda: db
    )

    result = await loader("n1")

    assert db.calls == ["p1"]
    assert len(result) == 1
    decoded_capture, abandoned = result[0]
    assert decoded_capture == db_capture
    assert abandoned is True


@pytest.mark.asyncio
async def test_ephemeral_message_returns_empty_without_a_db_call() -> None:
    """(c) A message with no native captures AND no ``persisted_message_id``
    (an ephemeral, never-persisted session) has no DB row to fall back to
    -- returns ``[]`` without ever calling the DB accessor."""
    message = _message(native_id="n1", persisted_message_id=None, exchanges=())
    loader = _build_console_inspector_exchanges_loader(
        {"n1": message}, _raising_db_accessor
    )

    result = await loader("n1")

    assert result == []


@pytest.mark.asyncio
async def test_unknown_native_id_returns_empty_without_a_db_call() -> None:
    """(c, variant) A native id the loader has never seen (not in the
    ``messages_by_native_id`` map at all) must not blow up or touch the
    DB -- same "no message resolved" contract as the ephemeral case."""
    loader = _build_console_inspector_exchanges_loader({}, _raising_db_accessor)

    result = await loader("unknown-id")

    assert result == []


@pytest.mark.asyncio
async def test_db_returns_none_short_circuits_without_reading_rows() -> None:
    """(c, variant) ``app_instance.chachanotes_db`` can itself be ``None``
    (no DB wired up at all) -- the loader must return ``[]`` rather than
    calling ``.get_message_exchanges`` on ``None``."""
    message = _message(native_id="n1", persisted_message_id="p1", exchanges=())
    loader = _build_console_inspector_exchanges_loader(
        {"n1": message}, lambda: None
    )

    result = await loader("n1")

    assert result == []


@pytest.mark.asyncio
async def test_a_corrupt_blob_is_skipped_not_fatal_to_the_rest() -> None:
    """(d) One row's ``capture_blob`` fails to decode (corrupt bytes) --
    logged and skipped, while a sibling row's valid blob still comes
    through. A single bad row must not take down the whole turn."""
    good_capture = _capture(model="good-model", seq=1)
    db = _FakeExchangesDB(
        rows=[
            {
                "run_tag": "run-1",
                "seq": 1,
                "status": "complete",
                "abandoned": False,
                "capture_blob": capture_to_blob(good_capture),
                "created_at": "2026-08-20T10:00:00Z",
            },
            {
                "run_tag": "run-1",
                "seq": 2,
                "status": "complete",
                "abandoned": False,
                "capture_blob": b"not a valid zlib/json blob",
                "created_at": "2026-08-20T10:00:01Z",
            },
        ]
    )
    message = _message(native_id="n1", persisted_message_id="p1", exchanges=())
    loader = _build_console_inspector_exchanges_loader(
        {"n1": message}, lambda: db
    )

    result = await loader("n1")

    assert len(result) == 1
    decoded_capture, abandoned = result[0]
    assert decoded_capture == good_capture
    assert abandoned is False


@pytest.mark.asyncio
async def test_column_blob_provenance_mismatch_is_skipped() -> None:
    capture = _capture()
    with pytest.raises(CaptureCorruptError):
        capture_from_storage(capture_to_blob(capture), "full")
    db = _FakeExchangesDB(rows=[{
        "run_tag": "run-1", "seq": 1, "status": "complete", "abandoned": False,
        "capture_detail": "full", "capture_blob": capture_to_blob(capture),
        "created_at": "t",
    }])
    loader = _build_console_inspector_exchanges_loader(
        {"n1": _message(exchanges=())}, lambda: db
    )
    assert await loader("n1") == []


@pytest.mark.asyncio
async def test_corrupt_blob_diagnostic_omits_traceback_and_blob_bytes() -> None:
    """Review finding M8: the decode-failure log line used to call
    ``logger.opt(exception=True)`` in a frame holding the raw
    ``capture_blob`` bytes (and, mid-loop, already-decoded
    ``ExchangeCapture`` payloads from earlier rows) -- loguru's diagnose
    formatter would annotate the failing source line's names with their
    values across the whole frame chain. Mirrors the Exchange tab's own
    handlers (``console_conversation_inspector.py``'s
    ``_load_turn_captures``), which deliberately refuse tracebacks for the
    identical reason -- type(exc).__name__ plus the message id is enough
    to diagnose and retry."""
    marker_bytes = b"CANARY_BLOB_BYTES_SHOULD_NOT_APPEAR_IN_LOG"
    db = _FakeExchangesDB(
        rows=[
            {
                "run_tag": "run-1",
                "seq": 1,
                "status": "complete",
                "abandoned": False,
                "capture_blob": marker_bytes,
                "created_at": "2026-08-20T10:00:00Z",
            },
        ]
    )
    message = _message(native_id="n1", persisted_message_id="p1", exchanges=())
    loader = _build_console_inspector_exchanges_loader({"n1": message}, lambda: db)

    diagnostics: list[str] = []
    sink_id = loguru_logger.add(diagnostics.append, level="WARNING")
    try:
        result = await loader("n1")
    finally:
        loguru_logger.remove(sink_id)

    assert result == []
    assert any("exchange_blob_decode_failed" in d for d in diagnostics), diagnostics
    joined = "\n".join(diagnostics)
    assert "CANARY_BLOB_BYTES_SHOULD_NOT_APPEAR_IN_LOG" not in joined
    assert "Traceback" not in joined


def test_inspector_push_captures_immutable_revision_target() -> None:
    session = SimpleNamespace(
        id="session-at-open", persisted_conversation_id="conversation-at-open"
    )
    store = SimpleNamespace(
        active_session_id=session.id,
        sessions=lambda: [session],
    )
    capture_revision = Mock(return_value=11)
    controller = SimpleNamespace(store=store, capture_revision=capture_revision)
    pushed = Mock()
    screen = SimpleNamespace(
        _build_console_inspector_cost_data=lambda: (
            [],
            SimpleNamespace(),
            [],
            _empty_loader,
        ),
        _ensure_console_chat_controller=lambda: controller,
        _console_active_session_is_ephemeral=lambda: False,
        app=SimpleNamespace(push_screen=pushed),
    )

    async def snapshot_factory():
        return SimpleNamespace()

    ChatScreen._push_console_inspector(
        screen,
        initial_tab="inspector-costs",
        snapshot_factory=snapshot_factory,
    )

    inspector = pushed.call_args.args[0]
    store.active_session_id = "session-selected-later"
    assert inspector._target_session_id == "session-at-open"
    assert inspector._target_conversation_id == "conversation-at-open"
    assert inspector._capture_revision_at_open == 11
    assert inspector._capture_revision_provider() == 11
    capture_revision.assert_called_with("session-at-open")


async def _empty_loader(_native_message_id: str):
    return []
