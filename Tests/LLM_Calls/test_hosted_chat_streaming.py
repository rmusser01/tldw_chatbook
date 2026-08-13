"""Strict SSE framing contracts for hosted provider transports."""

from __future__ import annotations

import pytest

from tldw_chatbook.LLM_Calls.hosted_chat_streaming import (
    SSERecord,
    SSERecordDecoder,
)


def test_sse_decoder_preserves_event_and_multiline_data_across_chunks() -> None:
    decoder = SSERecordDecoder()

    assert decoder.feed(b"event: message\r\ndata: caf\xc3") == ()
    assert decoder.feed(b"\xa9\rdata: second\r") == ()
    assert decoder.feed(b"\r") == (
        SSERecord(event="message", data="caf\u00e9\nsecond"),
    )
    assert decoder.finish() == ()


def test_sse_decoder_supports_cr_lf_crlf_and_multiple_records() -> None:
    decoder = SSERecordDecoder()

    assert decoder.feed(b"data: one\r\ndata: two\n\ndata: three\r\r") == (
        SSERecord(event=None, data="one\ntwo"),
        SSERecord(event=None, data="three"),
    )


def test_sse_decoder_ignores_comments_and_non_data_fields() -> None:
    decoder = SSERecordDecoder()

    assert decoder.feed(
        b": heartbeat\nid: private-id\nretry: 10\nunknown: ignored\n"
        b"event: completion\ndata:first\ndata:  second\n\n"
    ) == (SSERecord(event="completion", data="first\n second"),)
    assert decoder.feed(b"event: ignored-without-data\n\n") == ()
    assert decoder.feed(b"data\n\n") == (SSERecord(event=None, data=""),)


def test_sse_decoder_requires_blank_record_boundary() -> None:
    decoder = SSERecordDecoder()

    assert decoder.feed(b"data: pending\n") == ()
    with pytest.raises(ValueError, match="incomplete"):
        decoder.finish()


def test_sse_decoder_rejects_invalid_utf8_and_non_bytes() -> None:
    with pytest.raises(UnicodeDecodeError):
        SSERecordDecoder().feed(b"data: \xff\n\n")
    with pytest.raises(TypeError, match="bytes"):
        SSERecordDecoder().feed("data: wrong\n\n")  # type: ignore[arg-type]


def test_sse_decoder_close_state_rejects_more_input() -> None:
    decoder = SSERecordDecoder()

    assert decoder.finish() == ()
    with pytest.raises(ValueError, match="finished"):
        decoder.feed(b"data: late\n\n")
    with pytest.raises(ValueError, match="finished"):
        decoder.finish()


@pytest.mark.parametrize(
    ("constant", "limit", "chunks", "message"),
    [
        ("_MAX_SSE_BYTES", 3, [b"data"], "byte"),
        ("_MAX_SSE_LINE_CHARS", 7, [b"data: abc\n\n"], "line"),
        ("_MAX_SSE_LINE_SEGMENTS", 2, [b"da", b"ta", b": x\n\n"], "segment"),
        ("_MAX_SSE_RECORD_CHARS", 3, [b"data: four\n\n"], "record"),
        ("_MAX_SSE_DATA_LINES", 1, [b"data: a\ndata: b\n\n"], "data line"),
        ("_MAX_SSE_RECORDS", 1, [b"data: a\n\ndata: b\n\n"], "record count"),
    ],
)
def test_sse_decoder_enforces_independent_bounds(
    constant: str,
    limit: int,
    chunks: list[bytes],
    message: str,
) -> None:
    argument = {
        "_MAX_SSE_BYTES": "max_bytes",
        "_MAX_SSE_LINE_CHARS": "max_line_chars",
        "_MAX_SSE_LINE_SEGMENTS": "max_line_segments",
        "_MAX_SSE_RECORD_CHARS": "max_record_chars",
        "_MAX_SSE_DATA_LINES": "max_data_lines",
        "_MAX_SSE_RECORDS": "max_records",
    }[constant]
    decoder = SSERecordDecoder(**{argument: limit})

    with pytest.raises(ValueError, match=message):
        for chunk in chunks:
            decoder.feed(chunk)


def test_sse_decoder_accepts_many_small_segments_without_quadratic_joining() -> None:
    decoder = SSERecordDecoder()

    records: tuple[SSERecord, ...] = ()
    for chunk in (bytes([value]) for value in b"data: linear\n\n"):
        records += decoder.feed(chunk)

    assert records == (SSERecord(event=None, data="linear"),)
