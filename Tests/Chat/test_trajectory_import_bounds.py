"""The trajectory import seam must bound its read and own its decode errors.

TASK-32806.6. `_read_document` read a user-picked file whole (the picker
offers an "All Files" filter) with no size ceiling, then parsed it whole,
and its `try` only caught `OSError` and `JSONDecodeError` -- so a non-UTF-8
file escaped as a bare `UnicodeDecodeError` and a deeply nested array as a
bare `RecursionError`, past the `TrajectoryImportError` seam its caller
catches.
"""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Chat.trajectory_import import (
    TrajectoryImportError,
    _read_document,
)

# `_MAX_TRACE_FILE_BYTES` is imported inside the one test that needs it, so
# this file still COLLECTS against a tree where the constant does not exist
# yet -- a test file that cannot be collected proves nothing.


def test_a_valid_document_still_reads(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"schema_version": 1, "hello": "world"}))
    assert _read_document(path) == {"schema_version": 1, "hello": "world"}


def test_an_oversized_file_is_refused_before_it_is_read(tmp_path):
    path = tmp_path / "huge.json"
    # One byte over the ceiling; content is irrelevant because the size
    # check runs on st_size before the read.
    from tldw_chatbook.Chat.trajectory_import import _MAX_TRACE_FILE_BYTES

    path.write_bytes(b"{" + b" " * (_MAX_TRACE_FILE_BYTES + 1))
    with pytest.raises(TrajectoryImportError, match="too large"):
        _read_document(path)


def test_a_non_utf8_file_surfaces_as_the_seam_error(tmp_path):
    path = tmp_path / "latin1.json"
    path.write_bytes(b'{"note": "\xff\xfe not utf-8"}')
    with pytest.raises(TrajectoryImportError, match="not UTF-8"):
        _read_document(path)


def test_a_deeply_nested_document_surfaces_as_the_seam_error(tmp_path):
    path = tmp_path / "nested.json"
    depth = 200_000
    path.write_text("[" * depth + "]" * depth)
    with pytest.raises(TrajectoryImportError):
        _read_document(path)


def test_invalid_json_still_surfaces_as_the_seam_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json")
    with pytest.raises(TrajectoryImportError, match="not valid JSON"):
        _read_document(path)
