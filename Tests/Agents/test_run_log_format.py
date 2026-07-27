"""Pure record codec: round-trip, adversarial content, partial tails."""

from tldw_chatbook.Agents.run_log_format import (
    RECORD_ANCHOR,
    RunLogRecord,
    encode_record,
    iter_records,
)


def rec(number=1, content="hello", **kw):
    base = dict(
        number=number,
        run_id="a3f9c1",
        kind="primary",
        type="tool_result",
        ts="2026-07-27T18:22:31.004Z",
        content=content,
    )
    base.update(kw)
    return RunLogRecord(**base)


def test_round_trip_preserves_content_exactly():
    original = rec(content="line one\nline two\n")
    (parsed,) = list(iter_records(encode_record(original)))
    assert parsed.content == original.content
    assert parsed.number == 1
    assert parsed.run_id == "a3f9c1"


def test_header_is_one_physical_line():
    blob = encode_record(rec(tool="grep_files", status="ok", call_id="call_7"))
    header = blob.split(b"\n", 1)[0].decode()
    assert header.startswith(RECORD_ANCHOR + " ")
    assert "tool=grep_files" in header
    assert "bytes=5" in header


def test_content_containing_the_anchor_does_not_corrupt_parsing():
    # The whole point of bytes=N: content is sliced by length, never scanned.
    evil = f"{RECORD_ANCHOR} 999999 run=x kind=primary type=model ts=z bytes=0\nnope"
    blob = encode_record(rec(number=1, content=evil)) + encode_record(rec(number=2))
    parsed = list(iter_records(blob))
    assert len(parsed) == 2
    assert parsed[0].content == evil
    assert parsed[1].number == 2


def test_multibyte_content_counts_bytes_not_characters():
    original = rec(content="héllo — ✅")
    blob = encode_record(original)
    assert f"bytes={len(original.content.encode('utf-8'))}".encode() in blob
    (parsed,) = list(iter_records(blob))
    assert parsed.content == original.content


def test_partial_trailing_record_is_ignored():
    blob = encode_record(rec(number=1)) + encode_record(rec(number=2, content="abcdef"))
    truncated = blob[:-3]  # content cut mid-write
    parsed = list(iter_records(truncated))
    assert [p.number for p in parsed] == [1]


def test_record_missing_only_its_terminator_is_ignored():
    blob = encode_record(rec(number=1))
    parsed = list(iter_records(blob[:-1]))
    assert parsed == []


def test_truncated_field_round_trips_and_is_absent_otherwise():
    assert b"truncated=" not in encode_record(rec())
    blob = encode_record(rec(content="cut", truncated_from=9000))
    assert b"truncated=9000" in blob
    (parsed,) = list(iter_records(blob))
    assert parsed.truncated_from == 9000


def test_whitespace_in_header_values_is_sanitised():
    # A header field containing a space or newline would break single-line parsing.
    (parsed,) = list(iter_records(encode_record(rec(tool="bad name\nx"))))
    assert " " not in parsed.tool and "\n" not in parsed.tool


def test_empty_optional_fields_round_trip():
    # CRITICAL 1: Empty optional fields must round-trip as empty, not as "-".
    # A model record has tool="", status="", call_id="" by default.
    original = rec(number=1, tool="", status="", call_id="")
    (parsed,) = list(iter_records(encode_record(original)))
    assert parsed.tool == ""
    assert parsed.status == ""
    assert parsed.call_id == ""
    # Required fields should still be present.
    assert parsed.run_id == "a3f9c1"
    assert parsed.kind == "primary"


def test_malformed_bytes_field_does_not_discard_following_records():
    # CRITICAL 2a: Malformed header should resync, not return and lose all.
    # Good record 1, bad bytes field, good record 2.
    good1 = encode_record(rec(number=1, content="first"))
    bad = b"#@# 000002 run=a3f9c1 kind=primary type=model ts=z bytes=notanumber\nno-parse\n"
    good2 = encode_record(rec(number=3, content="third"))
    blob = good1 + bad + good2
    parsed = list(iter_records(blob))
    # Should yield only records 1 and 3, skipping the malformed record 2.
    assert len(parsed) == 2
    assert parsed[0].number == 1
    assert parsed[0].content == "first"
    assert parsed[1].number == 3
    assert parsed[1].content == "third"


def test_torn_record_does_not_stitch_following_record():
    # CRITICAL 2b: Torn record with overrun byte count must not yield stitched content.
    # Record 1: normal
    good1 = encode_record(rec(number=1, content="SHORT"))
    # Record 2: declares content is 100 bytes but only writes 10, then Record 3
    # This simulates a torn write: the parser would stitch record 3's start into record 2's content.
    # Manually craft record 2 with overrun declaration.
    # Content "123456789X" is 10 bytes, but declare bytes=100.
    torn_header = b"#@# 000002 run=a3f9c1 kind=primary type=tool_result tool=- status=- call=- ts=- bytes=100\n"
    torn_content = b"123456789X"  # Only 10 bytes, not 100
    # Add record 3 after torn content; resync will find it via "\n#@# "
    good3 = encode_record(rec(number=3, content="legitimate"))
    blob = good1 + torn_header + torn_content + b"\n" + good3
    parsed = list(iter_records(blob))
    # Record 1 should parse fine.
    # Record 2 is torn: declared 100 bytes, but at position end (10 bytes), we don't see '\n', so resync.
    # Record 3 should be recovered after resync.
    assert len(parsed) == 2
    assert parsed[0].number == 1
    assert parsed[1].number == 3
    assert parsed[1].content == "legitimate"
