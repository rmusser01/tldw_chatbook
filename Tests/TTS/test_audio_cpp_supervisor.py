from __future__ import annotations

import logging
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.TTS.audio_cpp_supervisor import _AudioCppDiagnosticRing


def _snapshot_texts(ring: _AudioCppDiagnosticRing) -> tuple[str, ...]:
    lines, _dropped = ring.snapshot()
    return tuple(line.text for line in lines)


def test_diagnostics_bound_lines_total_utf8_bytes_and_each_line() -> None:
    line_bounded = _AudioCppDiagnosticRing()
    line_bounded.feed("stdout", (("😀" * 2_000) + "\n").encode())
    line_bounded.finish("stdout")
    line_lines, _ = line_bounded.snapshot()
    assert line_lines
    assert all(len(line.text.encode("utf-8")) <= 4_096 for line in line_lines)

    count_bounded = _AudioCppDiagnosticRing()
    count_bounded.feed("stdout", b"line\n" * 205)
    count_bounded.finish("stdout")
    count_lines, count_dropped = count_bounded.snapshot()
    assert len(count_lines) == 200
    assert count_dropped == 5

    byte_bounded = _AudioCppDiagnosticRing()
    byte_bounded.feed("stderr", (("x" * 2_048) + "\n").encode() * 40)
    byte_bounded.finish("stderr")
    byte_lines, byte_dropped = byte_bounded.snapshot()
    assert sum(len(line.text.encode("utf-8")) for line in byte_lines) <= 65_536
    assert byte_dropped > 0


def test_diagnostics_flush_an_overlong_stream_without_waiting_for_newline() -> None:
    ring = _AudioCppDiagnosticRing()

    ring.feed("stdout", b"x" * 5_000)

    assert _snapshot_texts(ring) == ("x" * 4_096,)
    ring.finish("stdout")
    assert _snapshot_texts(ring) == ("x" * 4_096, "x" * 904)


def test_diagnostics_replacement_decode_invalid_utf8() -> None:
    ring = _AudioCppDiagnosticRing()

    ring.feed("stderr", b"before\xffafter\n")

    assert _snapshot_texts(ring) == ("before\ufffdafter",)


def test_diagnostics_remove_ansi_controls_and_escape_rich_markup() -> None:
    ring = _AudioCppDiagnosticRing()

    ring.feed(
        "stdout",
        b"\x1b[31m[bold]danger[/bold]\x1b[0m\x00\x08\x7f\n",
    )

    assert _snapshot_texts(ring) == (r"\[bold]danger\[/bold]",)


def test_diagnostics_redact_credentials_and_normalize_home_prefix(
    tmp_path: Path,
) -> None:
    home = tmp_path / "synthetic-home"
    ring = _AudioCppDiagnosticRing(home_directory=home)
    secret = "SYNTHETIC_ASSIGNMENT_SECRET"
    bearer = "SYNTHETIC_BEARER_SECRET"
    token = "SYNTHETIC_QUOTED_SECRET"

    ring.feed(
        "stderr",
        (
            f"model={home}/models/model.gguf api_key={secret} "
            f"Authorization: Bearer {bearer} token='{token}'\n"
        ).encode(),
    )
    rendered = _snapshot_texts(ring)[0]

    assert "~/models/model.gguf" in rendered
    assert str(home) not in rendered
    assert secret not in rendered
    assert bearer not in rendered
    assert token not in rendered
    assert rendered.count("<redacted>") == 3


def test_diagnostics_report_eviction_count_and_clear_per_generation() -> None:
    ring = _AudioCppDiagnosticRing()
    ring.feed("stdout", b"line\n" * 201)

    lines, dropped = ring.snapshot()
    assert len(lines) == 200
    assert dropped == 1

    ring.clear()

    assert ring.snapshot() == ((), 0)
    ring.feed("stderr", b"new generation\n")
    assert _snapshot_texts(ring) == ("new generation",)


def test_diagnostics_never_emit_to_python_or_loguru_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_output = "SYNTHETIC_PRIVATE_CHILD_OUTPUT"
    loguru_messages: list[str] = []
    caplog.set_level(logging.DEBUG)
    sink_id = logger.add(loguru_messages.append, level="DEBUG", format="{message}")
    try:
        ring = _AudioCppDiagnosticRing()
        ring.feed("stderr", f"detail={private_output}\n".encode())
        ring.finish("stderr")
    finally:
        logger.remove(sink_id)

    assert private_output in _snapshot_texts(ring)[0]
    assert private_output not in caplog.text
    assert private_output not in "".join(loguru_messages)
