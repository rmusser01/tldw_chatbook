"""TASK-19321 (ADR-029): chunker streaming diagnostics must not record user file paths.

The three `chunk_file_stream` diagnostic sites identify the file by a stable
content-free handle (`path_sha256=<sha256(resolved path)[:12]>`) plus safe
metadata (byte size, exception type, codec, byte offset). The full user path —
and exception text that embeds it or carries byte context from the file — must
never reach a log record. Capture idiom follows the TASK-15103 repairs
(see Tests/RAG/test_fusion.py's event-only pins).
"""

import hashlib

import pytest
from loguru import logger

from tldw_chatbook.Chunking.engine import Chunker, ChunkingError, InvalidInputError


def _expected_path_ref(path) -> str:
    return hashlib.sha256(
        str(path.resolve()).encode("utf-8", "surrogatepass")
    ).hexdigest()[:12]


def _capture(level: str):
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level=level)
    return messages, sink_id


class TestStreamDiagnosticsOmitUserPaths:
    def test_stream_processing_info_identifies_the_file_without_its_path(self, tmp_path):
        """The per-file INFO record keeps size + a stable handle, not the path."""
        secret_name = "medical-records-2026.txt"
        file_path = tmp_path / secret_name
        file_path.write_text("word " * 200, encoding="utf-8")

        messages, sink_id = _capture("INFO")
        try:
            chunks = list(
                Chunker().chunk_file_stream(file_path, method="words", max_size=10)
            )
        finally:
            logger.remove(sink_id)

        assert chunks, "streaming must still produce chunks"
        stream_records = [m for m in messages if "Stream processing file" in m]
        assert stream_records, f"the stream-start event must be logged: {messages}"
        assert not any(str(file_path) in m for m in messages), (
            f"the full user path must not appear in any record: {messages}"
        )
        assert not any(secret_name in m for m in messages), (
            f"the user file name must not appear in any record: {messages}"
        )
        # The record still identifies the file (stable handle) and its size.
        expected_ref = _expected_path_ref(file_path)
        assert any(
            f"path_sha256={expected_ref}" in m and "bytes)" in m
            for m in stream_records
        ), f"the stable handle and byte size must survive redaction: {stream_records}"

    def test_decode_failure_diagnostic_omits_path_and_raw_exception_text(self, tmp_path):
        """A UnicodeDecodeError's own text carries byte context from the file.

        The ERROR record keeps the failure class, codec, and byte offset; the
        path and str(e) stay out.
        """
        secret_name = "latin1-divorce-notes.txt"
        file_path = tmp_path / secret_name
        file_path.write_bytes("café monde ligne suivante".encode("cp1252"))

        messages, sink_id = _capture("ERROR")
        try:
            with pytest.raises(InvalidInputError):
                list(
                    Chunker().chunk_file_stream(file_path, method="words", max_size=10)
                )
        finally:
            logger.remove(sink_id)

        failure_records = [m for m in messages if "File stream decoding failed" in m]
        assert failure_records, f"the decode-failure event must be logged: {messages}"
        assert not any(str(file_path) in m for m in messages), (
            f"the full user path must not appear in any record: {messages}"
        )
        assert not any(secret_name in m for m in messages), (
            f"the user file name must not appear in any record: {messages}"
        )
        assert any("UnicodeDecodeError" in m for m in failure_records), (
            f"the failure class must survive redaction: {failure_records}"
        )
        assert not any("can't decode" in m for m in failure_records), (
            f"raw UnicodeDecodeError text must not be echoed: {failure_records}"
        )

    def test_oserror_diagnostic_does_not_leak_the_path_through_exception_text(
        self, tmp_path
    ):
        """An OSError stringifies with the filename embedded; log the TYPE only.

        Streaming a directory raises a real OSError subclass
        (IsADirectoryError on POSIX, PermissionError on Windows) whose str()
        names the path — the `{e}` interpolation was the leak here, not an
        explicit path variable.
        """
        secret_name = "private-therapy-journal"
        target = tmp_path / secret_name
        target.mkdir()

        messages, sink_id = _capture("ERROR")
        try:
            with pytest.raises(ChunkingError):
                list(Chunker().chunk_file_stream(target, method="words", max_size=10))
        finally:
            logger.remove(sink_id)

        failure_records = [m for m in messages if "File stream processing failed" in m]
        assert failure_records, f"the stream-failure event must be logged: {messages}"
        assert not any(str(target) in m for m in messages), (
            f"the full user path must not appear in any record: {messages}"
        )
        assert not any(secret_name in m for m in messages), (
            f"the user directory name must not appear in any record: {messages}"
        )
        assert any(
            "IsADirectoryError" in m or "PermissionError" in m
            for m in failure_records
        ), f"the exception class must survive redaction: {failure_records}"
