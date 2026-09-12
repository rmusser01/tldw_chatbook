"""Real-file bounded paging and byte-framing regression evidence."""

from pathlib import Path

import pytest

from tldw_chatbook.Agents.run_log_format import RunLogRecord, encode_record


def record(number=1, content="hello", run_id="primary", **kwargs):
    return RunLogRecord(
        number, run_id, "primary", "tool_result", "now", content, **kwargs
    )


def pages(root, **kwargs):
    from tldw_chatbook.Agents.run_log_paging import load_record_page

    result = []
    cursor = None
    for _ in range(1000):
        page = load_record_page(root, cursor=cursor, **kwargs)
        result.append(page)
        if page.next_cursor is None:
            return result
        assert page.next_cursor != cursor
        cursor = page.next_cursor
    pytest.fail("cursor did not reach EOF")


@pytest.fixture
def reads(monkeypatch):
    sizes = []
    original = Path.open

    class RecordingFile:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, size=-1):
            assert 0 <= size <= 256_004
            data = self.file.read(size)
            sizes.append(len(data))
            return data

        def readline(self, size=-1):
            assert 0 <= size <= 16_385
            data = self.file.readline(size)
            sizes.append(len(data))
            return data

    def open_recording(path, *args, **kwargs):
        file = original(path, *args, **kwargs)
        return RecordingFile(file) if args and args[0] == "rb" else file

    monkeypatch.setattr(Path, "open", open_recording)
    monkeypatch.setattr(
        Path, "read_bytes", lambda *args: pytest.fail("whole segment read")
    )
    return sizes


def test_many_records_sparse_numeric_segments(tmp_path):
    for segment, start in [(2, 1), (10, 151)]:
        (tmp_path / f"logs.{segment}.txt").write_bytes(
            b"".join(encode_record(record(n)) for n in range(start, start + 150))
        )
    result = pages(tmp_path)
    assert [s.record.number for p in result for s in p.slices] == list(range(1, 301))
    assert all(len(p.slices) <= 100 for p in result)


def test_multimegabyte_utf8_reconstructs_with_bounded_reads(tmp_path, reads):
    original = ("aé✅😀" * 300_000).encode()
    (tmp_path / "logs.0001.txt").write_bytes(
        encode_record(record(content=original.decode()))
    )
    result = pages(tmp_path)
    assert (
        b"".join(s.record.content.encode() for p in result for s in p.slices)
        == original
    )
    assert all(
        sum(len(s.record.content.encode()) for s in p.slices) <= 256_000 for p in result
    )
    assert all(p.scanned_bytes <= 4_500_000 + 16_393 for p in result)
    assert sum(reads) == sum(p.scanned_bytes for p in result)
    assert all(
        s.stored_content_bytes == len(original) for p in result for s in p.slices
    )
    assert result[-1].slices[-1].slice_complete


def test_child_filter_seeks_over_large_sibling(tmp_path, reads):
    (tmp_path / "logs.0001.txt").write_bytes(
        encode_record(record(content="x" * 3_000_000))
        + encode_record(record(2, "child body", "child"))
    )
    result = pages(tmp_path, run_id="child")
    assert [s.record.content for p in result for s in p.slices] == ["child body"]
    assert sum(reads) < 1000


def test_scan_budget_empty_page_continues_and_diagnoses_corruption(tmp_path, reads):
    (tmp_path / "logs.0001.txt").write_bytes(
        b"bad secret\n" * 10_000 + encode_record(record())
    )
    result = pages(tmp_path, max_scan_bytes=20_000)
    assert not result[0].slices and result[0].next_cursor is not None
    assert result[0].diagnostics
    assert all("secret" not in d for p in result for d in p.diagnostics)
    assert all(p.scanned_bytes <= 20_000 + 16_393 for p in result)
    assert "".join(s.record.content for p in result for s in p.slices) == "hello"


@pytest.mark.parametrize("tail", [b"", b"WRONG"])
def test_complete_terminator_required_before_first_fragment(tmp_path, tail):
    blob = encode_record(record(content="x" * 300_000))
    (tmp_path / "logs.0001.txt").write_bytes(blob[:-1] + tail)
    result = pages(tmp_path)
    assert not any(p.slices for p in result)
    assert any(p.diagnostics for p in result)


def test_oversized_and_malformed_headers_resynchronize(tmp_path):
    blob = b"#@# " + b"secret" * 5000 + b"\n"
    blob += b"#@# nope bytes=-1\n" + encode_record(record(content="recovered"))
    (tmp_path / "logs.0001.txt").write_bytes(blob)
    result = pages(tmp_path, max_scan_bytes=20_000)
    assert [s.record.content for p in result for s in p.slices] == ["recovered"]
    assert any("header_too_large" in p.diagnostics for p in result)


@pytest.mark.parametrize(
    "values", [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (True, 0, 0), ("1", 0, 0)]
)
def test_invalid_cursor_rejected_before_io(tmp_path, values, monkeypatch):
    from tldw_chatbook.Agents.run_log_paging import RunLogPageCursor, load_record_page

    monkeypatch.setattr(
        Path, "open", lambda *a, **k: pytest.fail("opened invalid cursor")
    )
    with pytest.raises(ValueError):
        load_record_page(tmp_path, cursor=RunLogPageCursor(*values))


def test_format_preserves_heading_fragment_and_storage_notices(tmp_path):
    from tldw_chatbook.Agents.run_log_paging import format_record_page

    (tmp_path / "logs.0001.txt").write_bytes(
        encode_record(
            record(content="abcdefghij", tool="fetch", status="ok", truncated_from=20)
        )
    )
    result = pages(tmp_path, max_content_bytes=4)
    text = format_record_page(result[0])
    assert "record 000001 [tool_result/fetch/ok]" in text
    assert "abcd" in text and "0-4" in text and "10" in text
    assert "20" in text and "never stored" in text


def test_tiny_scan_budget_advances_utf8_fragments(tmp_path, reads):
    (tmp_path / "logs.0001.txt").write_bytes(encode_record(record(content="😀é✅")))
    result = pages(tmp_path, max_scan_bytes=1, max_content_bytes=4)
    assert "".join(s.record.content for p in result for s in p.slices) == "😀é✅"
    assert all(p.scanned_bytes <= 1 + 16_393 for p in result)


def test_torn_record_recovers_later_anchor(tmp_path):
    (tmp_path / "logs.0001.txt").write_bytes(
        encode_record(record(content="bad"))[:-1]
        + b"wrong\n"
        + encode_record(record(2, "good"))
    )
    result = pages(tmp_path)
    assert [s.record.content for p in result for s in p.slices] == ["good"]
    assert "torn_record" in result[0].diagnostics


def test_incomplete_record_can_be_read_after_append(tmp_path):
    path = tmp_path / "logs.0001.txt"
    complete = encode_record(record(content="😀" * 100_000))
    path.write_bytes(complete[:-1])
    assert not pages(tmp_path)[0].slices
    with path.open("ab") as file:
        file.write(b"\n")
    assert (
        "".join(s.record.content for p in pages(tmp_path) for s in p.slices)
        == "😀" * 100_000
    )


@pytest.mark.parametrize("cursor", ["bad", 1, {}, object()])
def test_cursor_object_type_rejected(tmp_path, cursor):
    from tldw_chatbook.Agents.run_log_paging import load_record_page

    with pytest.raises(ValueError):
        load_record_page(tmp_path, cursor=cursor)
