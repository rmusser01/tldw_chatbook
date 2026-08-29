#!/usr/bin/env python3
"""Qualify pyte 0.8.2 without retaining terminal content."""

from __future__ import annotations

import argparse
import codecs
import importlib.metadata
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from common import (
    SCHEMA_VERSION,
    OwnedProcessJob,
    QualificationError,
    artifact_manifest,
    command_facts,
    memory_facts,
    platform_facts,
    sha256_file,
    terminate_owned_group,
    utc_now,
    write_probe_result,
)


EXPECTED_VERSION = "0.8.2"
TERM_VALUE = "linux"
CAPTURE_LIMIT = 512 * 1024
CAPTURE_SECONDS = 3.0
CAPTURE_INPUT_QUIET_SECONDS = 0.1
PARSER_MEMORY_LIMIT = 64 * 1024 * 1024
CONTROL_SEQUENCE_BYTE_LIMIT = 4 * 1024
CSI_PARAMETER_LIMIT = 32
CSI_PARAMETER_DIGIT_LIMIT = 4
CSI_PARAMETER_VALUE_LIMIT = 9_999
CSI_PRIVATE_INTERMEDIATE_BYTE_LIMIT = 16
NON_CSI_CONTROL_BYTE_LIMIT = 16
STRING_CONTROL_BYTE_LIMIT = 4 * 1024


def _row(row_id: str, passed: bool, **facts: object) -> dict[str, object]:
    return {
        "id": row_id,
        "mandatory": True,
        "status": "PASS" if passed else "FAIL",
        **facts,
    }


class _AlternateScreenAdapter:
    """Route DEC 1049 mode changes between two bounded pyte screens."""

    def __init__(self, pyte_module: Any, columns: int, lines: int) -> None:
        self.primary = pyte_module.Screen(columns, lines)
        self.alternate = pyte_module.Screen(columns, lines)
        self.active = self.primary
        self.in_alternate = False
        self.entry_count = 0
        self.exit_count = 0

    def set_mode(self, *modes: int, **kwargs: object) -> None:
        private = kwargs.get("private") is True
        remaining = tuple(mode for mode in modes if not (private and mode == 1049))
        if private and 1049 in modes and not self.in_alternate:
            self.primary.save_cursor()
            self.alternate.reset()
            self.active = self.alternate
            self.in_alternate = True
            self.entry_count += 1
        if remaining:
            self.active.set_mode(*remaining, **kwargs)

    def reset_mode(self, *modes: int, **kwargs: object) -> None:
        private = kwargs.get("private") is True
        remaining = tuple(mode for mode in modes if not (private and mode == 1049))
        if remaining:
            self.active.reset_mode(*remaining, **kwargs)
        if private and 1049 in modes and self.in_alternate:
            self.active = self.primary
            self.primary.restore_cursor()
            self.in_alternate = False
            self.exit_count += 1

    def __getattr__(self, name: str) -> object:
        return getattr(self.active, name)


def _alternate_screen_facts(pyte_module: Any) -> dict[str, object]:
    """Exercise DEC 1049 entry and exit through one genuine pyte stream."""
    adapter = _AlternateScreenAdapter(pyte_module, columns=40, lines=4)
    stream = pyte_module.Stream(adapter)
    stream.feed("primary")
    primary_before = tuple(adapter.primary.display)
    cursor_before = (adapter.primary.cursor.x, adapter.primary.cursor.y)

    stream.feed("\x1b[?1049h")
    entered = adapter.in_alternate and adapter.active is adapter.alternate
    stream.feed("alternate")
    alternate_isolated = (
        "alternate" in "".join(adapter.alternate.display)
        and tuple(adapter.primary.display) == primary_before
    )

    stream.feed("\x1b[?1049l")
    exited = not adapter.in_alternate and adapter.active is adapter.primary
    primary_restored = (
        tuple(adapter.primary.display) == primary_before
        and (adapter.primary.cursor.x, adapter.primary.cursor.y) == cursor_before
    )
    return {
        "control_sequence_count": 2,
        "entered": entered,
        "entry_count": adapter.entry_count,
        "exited": exited,
        "exit_count": adapter.exit_count,
        "alternate_isolated": alternate_isolated,
        "primary_restored": primary_restored,
    }


def _csi_tokens(value: bytes) -> list[bytes]:
    tokens = [b""]
    for byte in value:
        if byte in (ord(";"), ord(":")):
            tokens.append(b"")
        elif 0x30 <= byte <= 0x3F:
            tokens[-1] += bytes((byte,))
    return tokens


def sequence_is_bounded(value: bytes) -> bool:
    """Return whether one complete sequence fits every ADR-099 pre-parser cap."""
    if not value.startswith(b"\x1b"):
        try:
            value.decode("utf-8", "strict")
        except UnicodeDecodeError:
            return False
        return len(value) <= 4
    if value.startswith(b"\x1b["):
        if len(value) > CONTROL_SEQUENCE_BYTE_LIMIT:
            return False
        body = value[2:]
        final_index = next(
            (index for index, byte in enumerate(body) if 0x40 <= byte <= 0x7E),
            None,
        )
        if final_index is None:
            return False
        parameters = body[:final_index]
        private_or_intermediate = sum(
            0x20 <= byte <= 0x2F or 0x3C <= byte <= 0x3F for byte in parameters
        )
        if private_or_intermediate > CSI_PRIVATE_INTERMEDIATE_BYTE_LIMIT:
            return False
        tokens = _csi_tokens(parameters)
        if len(tokens) > CSI_PARAMETER_LIMIT:
            return False
        for token in tokens:
            digits = bytes(byte for byte in token if 0x30 <= byte <= 0x39)
            if len(digits) > CSI_PARAMETER_DIGIT_LIMIT or (
                digits and int(digits) > CSI_PARAMETER_VALUE_LIMIT
            ):
                return False
        return True
    if value.startswith((b"\x1b]", b"\x1bP", b"\x1b_", b"\x1b^")):
        terminated = value.endswith(b"\x07") or value.endswith(b"\x1b\\")
        return terminated and len(value) <= STRING_CONTROL_BYTE_LIMIT
    return len(value) <= NON_CSI_CONTROL_BYTE_LIMIT


def _sequence_limit_facts(
    *, accepted_fixture_count: int, rejected_fixture_count: int
) -> dict[str, int]:
    return {
        "accepted_fixture_count": accepted_fixture_count,
        "rejected_fixture_count": rejected_fixture_count,
        "control_sequence_byte_limit": CONTROL_SEQUENCE_BYTE_LIMIT,
        "csi_parameter_limit": CSI_PARAMETER_LIMIT,
        "csi_parameter_digit_limit": CSI_PARAMETER_DIGIT_LIMIT,
        "csi_parameter_value_limit": CSI_PARAMETER_VALUE_LIMIT,
        "csi_private_intermediate_byte_limit": (CSI_PRIVATE_INTERMEDIATE_BYTE_LIMIT),
        "non_csi_control_byte_limit": NON_CSI_CONTROL_BYTE_LIMIT,
        "string_control_byte_limit": STRING_CONTROL_BYTE_LIMIT,
    }


def _installed_file_facts(
    distribution: importlib.metadata.Distribution,
) -> dict[str, str | None]:
    primary_name: str | None = None
    primary_hash: str | None = None
    record_name: str | None = None
    record_hash: str | None = None
    for relative in sorted(distribution.files or (), key=lambda item: str(item)):
        candidate = distribution.locate_file(relative)
        normalized = str(relative).replace("\\", "/")
        if candidate.is_file() and normalized.endswith(".dist-info/RECORD"):
            record_name = str(relative)
            record_hash = sha256_file(candidate)
        if (
            primary_name is None
            and candidate.is_file()
            and candidate.suffix.lower() in {".py", ".pyd", ".so"}
        ):
            primary_name = str(relative)
            primary_hash = sha256_file(candidate)
    return {
        "primary_file_name": primary_name,
        "primary_file_sha256": primary_hash,
        "record_file_name": record_name,
        "record_file_sha256": record_hash,
    }


def _artifact_binding(manifest: dict[str, Any]) -> tuple[bool, dict[str, object]]:
    distribution = importlib.metadata.distribution("pyte")
    prepared = [
        item
        for item in manifest["resolved_distributions"]
        if str(item.get("name", "")).lower() == "pyte"
    ]
    artifacts = [
        item
        for item in manifest["artifacts"]
        if str(item.get("name", "")).lower() == "pyte"
    ]
    installed = _installed_file_facts(distribution)
    prepared_item = prepared[0] if len(prepared) == 1 else {}
    artifact_item = artifacts[0] if len(artifacts) == 1 else {}
    passed = (
        distribution.version == EXPECTED_VERSION
        and len(prepared) == 1
        and len(artifacts) == 1
        and prepared_item.get("version") == EXPECTED_VERSION
        and prepared_item.get("primary_file") == installed["primary_file_name"]
        and prepared_item.get("primary_file_sha256") == installed["primary_file_sha256"]
        and prepared_item.get("record_file") == installed["record_file_name"]
        and prepared_item.get("record_file_sha256") == installed["record_file_sha256"]
        and isinstance(artifact_item.get("filename"), str)
        and isinstance(artifact_item.get("size_bytes"), int)
        and artifact_item.get("version") == EXPECTED_VERSION
        and len(
            {
                artifact_item.get("sha256"),
                artifact_item.get("sha256_before_install"),
                artifact_item.get("sha256_after_install"),
            }
        )
        == 1
        and re.fullmatch(r"[0-9a-f]{64}", str(artifact_item.get("sha256", "")))
        is not None
    )
    return passed, {
        "distribution_version": distribution.version,
        **installed,
        "artifact_filename": artifact_item.get("filename"),
        "artifact_sha256": artifact_item.get("sha256"),
        "artifact_size_bytes": artifact_item.get("size_bytes", 0),
        "artifact_verified_during_probe": passed,
    }


def _feed(pyte_module: Any, value: bytes, *, columns: int = 80, lines: int = 24) -> Any:
    screen = pyte_module.Screen(columns, lines)
    stream = pyte_module.Stream(screen)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    for offset in range(0, len(value), 7):
        stream.feed(decoder.decode(value[offset : offset + 7], final=False))
    stream.feed(decoder.decode(b"", final=True))
    return screen


@dataclass(frozen=True)
class _CaptureResult:
    """Content-capped PTY result with explicit exit and cleanup outcomes."""

    output: bytes
    exit_code: int
    timed_out: bool
    capture_within_bound: bool
    terminated: bool
    killed: bool


def _capture(argv: Sequence[str], input_bytes: bytes = b"") -> _CaptureResult:
    if os.name == "nt":
        return _CaptureResult(b"", 0, False, True, False, False)
    import pty

    master = -1
    slave = -1
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    terminated = False
    killed = False
    limit_hit = False
    captured_bytes = 0
    stream_closed = False
    deadline = time.monotonic() + CAPTURE_SECONDS
    job = OwnedProcessJob()
    with tempfile.TemporaryFile() as capture_file:
        try:
            master, slave = pty.openpty()
            process = subprocess.Popen(
                list(argv),
                stdin=slave,
                stdout=slave,
                stderr=slave,
                env={
                    "PATH": os.environ.get("PATH", os.defpath),
                    "HOME": os.environ.get("HOME", tempfile.gettempdir()),
                    "LANG": os.environ.get("LANG", "C.UTF-8"),
                    "TERM": TERM_VALUE,
                },
                close_fds=True,
                start_new_session=True,
            )
            job.assign(process)
            os.close(slave)
            slave = -1
            input_pending = input_bytes
            input_ready_at: float | None = None
            while True:
                remaining = CAPTURE_LIMIT - captured_bytes
                if remaining <= 0:
                    limit_hit = True
                    break
                now = time.monotonic()
                if (
                    input_pending
                    and input_ready_at is not None
                    and now >= input_ready_at
                ):
                    written = os.write(master, input_pending)
                    input_pending = input_pending[written:]
                    if input_pending:
                        input_ready_at = now + CAPTURE_INPUT_QUIET_SECONDS
                remaining_seconds = deadline - now
                if remaining_seconds <= 0:
                    timed_out = process.poll() is None
                    break
                select_seconds = min(0.1, remaining_seconds)
                if input_pending and input_ready_at is not None:
                    select_seconds = min(
                        select_seconds,
                        max(0.0, input_ready_at - now),
                    )
                readable, _, _ = select.select([master], [], [], select_seconds)
                if readable:
                    try:
                        chunk = os.read(master, min(8192, remaining))
                    except OSError:
                        stream_closed = True
                        break
                    if not chunk:
                        stream_closed = True
                        break
                    capture_file.write(chunk)
                    captured_bytes += len(chunk)
                    if input_pending:
                        input_ready_at = time.monotonic() + CAPTURE_INPUT_QUIET_SECONDS
                    continue
                if process.poll() is not None:
                    break
        finally:
            if slave >= 0:
                os.close(slave)
            if process is not None:
                if process.poll() is None and stream_closed:
                    try:
                        process.wait(
                            timeout=min(
                                1.0,
                                max(0.0, deadline - time.monotonic()),
                            )
                        )
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is None:
                    terminated, killed = terminate_owned_group(
                        process, job, grace_seconds=1.0
                    )
                else:
                    process.wait()
            job.close()
            if master >= 0:
                os.close(master)
        capture_file.seek(0)
        output = capture_file.read(CAPTURE_LIMIT)
    if process is None:
        raise QualificationError("PTY capture process did not start")
    return _CaptureResult(
        output=output,
        exit_code=int(process.returncode),
        timed_out=timed_out,
        capture_within_bound=not limit_hit and captured_bytes < CAPTURE_LIMIT,
        terminated=terminated,
        killed=killed,
    )


def _shell_matrix(pyte_module: Any) -> tuple[bool, dict[str, object]]:
    if os.name == "nt":
        return True, {
            "available_count": 0,
            "captured_count": 0,
            "captured_byte_count": 0,
        }
    candidates = (
        os.environ.get("SHELL"),
        shutil.which("bash"),
        shutil.which("zsh"),
    )
    available = 0
    captured = 0
    byte_count = 0
    for shell in dict.fromkeys(candidates):
        if not shell or not Path(shell).is_file():
            continue
        available += 1
        try:
            result = _capture(
                [shell, "-lc", "printf '\\033[32mterminal-qualification\\033[0m\\n'"]
            )
        except OSError:
            continue
        if (
            result.capture_within_bound
            and not result.timed_out
            and result.exit_code == 0
        ):
            _feed(pyte_module, result.output)
            captured += 1
            byte_count += len(result.output)
    return available > 0 and captured == available, {
        "available_count": available,
        "captured_count": captured,
        "captured_byte_count": byte_count,
    }


def _program_capture_markers(class_name: str, output: bytes) -> bool:
    alternate = b"\x1b[?1049h" in output and b"\x1b[?1049l" in output
    home = b"\x1b[H" in output or re.search(rb"\x1b\[\d+;\d+H", output) is not None
    clear = (
        b"\x1b[2J" in output
        or b"\x1b[J" in output
        or re.search(rb"\x1b\[[0-9;]*K", output) is not None
    )
    rendition = re.search(rb"\x1b\[[0-9;]*m", output) is not None
    if class_name == "editor":
        return alternate or (home and clear and rendition)
    if class_name == "pager":
        return alternate or (clear and rendition)
    if class_name == "monitor":
        return home and (clear or rendition)
    raise QualificationError("unknown full-screen program class")


def _program_capture_passes(class_name: str, result: _CaptureResult) -> bool:
    """Require a clean scripted exit plus class-specific interactive controls."""
    return (
        result.exit_code == 0
        and not result.timed_out
        and result.capture_within_bound
        and not result.terminated
        and not result.killed
        and _program_capture_markers(class_name, result.output)
    )


def _program_matrix(pyte_module: Any) -> tuple[bool, dict[str, object]]:
    if os.name == "nt":
        fixtures = {
            "editor": b"\x1b[?1049h\x1b[Heditor fixture\x1b[?1049l",
            "pager": b"\x1b[2J\x1b[Hpager fixture\r\n:",
            "monitor": b"\x1b[H\x1b[7mprocess monitor fixture\x1b[0m",
        }
        results = {
            name: _CaptureResult(value, 0, False, True, False, False)
            for name, value in fixtures.items()
        }
        for result in results.values():
            _feed(pyte_module, result.output)
        clean_exit = {name: result.exit_code == 0 for name, result in results.items()}
        markers = {
            name: _program_capture_markers(name, result.output)
            for name, result in results.items()
        }
        class_pass = {
            name: _program_capture_passes(name, result)
            for name, result in results.items()
        }
        return all(class_pass.values()), {
            "class_available_counts": {name: 0 for name in fixtures},
            "class_clean_exit": clean_exit,
            "class_interactive_markers": markers,
            "class_pass": class_pass,
            "captured_byte_count": sum(
                len(result.output) for result in results.values()
            ),
            "fixture_count": len(fixtures),
            "real_program_count": 0,
        }
    programs: dict[str, list[tuple[list[str], bytes]]] = {
        "editor": [],
        "pager": [],
        "monitor": [],
    }
    editor = shutil.which("vim") or shutil.which("vi") or shutil.which("nano")
    pager = shutil.which("less")
    monitor = shutil.which("top") or shutil.which("htop")
    if editor:
        if Path(editor).name == "nano":
            programs["editor"].append(([editor], b"\x18n"))
        else:
            programs["editor"].append(([editor, "-Nu", "NONE", "-n"], b":q!\r"))
    byte_count = 0
    with tempfile.NamedTemporaryFile(prefix="tldw-pager-", mode="w") as stream:
        stream.write("one\ntwo\nthree\n")
        stream.flush()
        if pager:
            programs["pager"].append(([pager, stream.name], b"q"))
        if monitor:
            argv = (
                [monitor, "-n", "1"]
                if sys.platform == "darwin" and Path(monitor).name == "top"
                else [monitor, "-n", "1"]
                if Path(monitor).name == "top"
                else [monitor, "-C", "-d", "1"]
            )
            programs["monitor"].append((argv, b"q"))
        class_pass: dict[str, bool] = {}
        class_clean_exit: dict[str, bool] = {}
        class_markers: dict[str, bool] = {}
        available_counts: dict[str, int] = {}
        for class_name, candidates in programs.items():
            available_counts[class_name] = len(candidates)
            class_pass[class_name] = False
            class_clean_exit[class_name] = False
            class_markers[class_name] = False
            for argv, input_bytes in candidates:
                try:
                    result = _capture(argv, input_bytes)
                except OSError:
                    continue
                clean_exit = (
                    result.exit_code == 0
                    and not result.timed_out
                    and not result.terminated
                    and not result.killed
                )
                markers = _program_capture_markers(class_name, result.output)
                class_clean_exit[class_name] = (
                    class_clean_exit[class_name] or clean_exit
                )
                class_markers[class_name] = class_markers[class_name] or markers
                if result.capture_within_bound:
                    _feed(pyte_module, result.output)
                    byte_count += len(result.output)
                class_pass[class_name] = class_pass[class_name] or (
                    _program_capture_passes(class_name, result)
                )
    return all(class_pass.values()), {
        "class_available_counts": available_counts,
        "class_clean_exit": class_clean_exit,
        "class_interactive_markers": class_markers,
        "class_pass": class_pass,
        "captured_byte_count": byte_count,
        "fixture_count": 0,
        "real_program_count": sum(available_counts.values()),
    }


def _parser_rows(pyte_module: Any) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    shell_passed, shell_facts = _shell_matrix(pyte_module)
    rows.append(_row("parser-shell-captures", shell_passed, **shell_facts))
    fixtures = (
        b"Microsoft Windows [Version 10.0.17763]\r\nC:\\>dir\r\n",
        b"\x1b[93mPS C:\\>\x1b[0m Get-Command python\r\n",
    )
    for fixture in fixtures:
        _feed(pyte_module, fixture)
    rows.append(
        _row(
            "parser-powershell-cmd-fixtures",
            True,
            fixture_count=len(fixtures),
            fixture_byte_count=sum(map(len, fixtures)),
        )
    )
    programs_passed, program_facts = _program_matrix(pyte_module)
    rows.append(_row("parser-full-screen-programs", programs_passed, **program_facts))

    unicode_screen = _feed(pyte_module, "A界e\u0301🙂".encode(), columns=20, lines=4)
    unicode_display = "".join(unicode_screen.display)
    normalized_display = unicodedata.normalize("NFD", unicode_display)
    wide_placeholders = sum(
        unicode_screen.buffer[0][column].data == "" for column in (2, 5)
    )
    combining_equivalent = "e\u0301" in normalized_display
    unicode_passed = (
        "界" in unicode_display
        and combining_equivalent
        and wide_placeholders == 2
        and unicode_screen.cursor.x == 6
    )
    rows.append(
        _row(
            "parser-unicode-cells",
            unicode_passed,
            cursor_column=unicode_screen.cursor.x,
            fixture_count=1,
            wide_placeholder_count=wide_placeholders,
            combining_normalized=combining_equivalent,
        )
    )
    alternate_facts = _alternate_screen_facts(pyte_module)
    alternate_passed = all(
        alternate_facts[key]
        for key in ("entered", "exited", "alternate_isolated", "primary_restored")
    )
    rows.append(
        _row(
            "parser-alternate-screen",
            alternate_passed,
            **alternate_facts,
        )
    )
    resized = _feed(pyte_module, b"resize", columns=80, lines=24)
    resized.resize(lines=40, columns=120)
    rows.append(
        _row(
            "parser-resize",
            resized.lines == 40 and resized.columns == 120,
            lines=resized.lines,
            columns=resized.columns,
        )
    )
    controls = {
        "parser-bracketed-paste": b"\x1b[?2004htext\x1b[?2004l",
        "parser-terminal-queries": b"\x1b[5n\x1b[6n\x1b[c",
        "parser-malformed-controls": b"\x1b[999999999999;::::m\xff\xfeplain\x1b]bad\x07",
    }
    for row_id, fixture in controls.items():
        try:
            _feed(pyte_module, fixture)
            passed = True
        except Exception:
            passed = False
        rows.append(_row(row_id, passed, fixture_byte_count=len(fixture)))

    rejected = (
        b"\xe2\x82",
        b"\x1b" + b"x" * 16,
        b"\x1b[" + b"1" * 257,
        b"\x1b[" + b"1;" * 33 + b"m",
        b"\x1b[10000m",
        b"\x1b[" + b" " * 17 + b"m",
        b"\x1b]" + b"x" * 4096,
        b"\x1bP" + b"x" * 4096,
        b"\x1b_" + b"x" * 4096,
        b"\x1b^" + b"x" * 4096,
    )
    accepted = (
        "€".encode(),
        b"\x1b[31m",
        b"\x1b[9999m",
        b"\x1b" + b"x" * 15,
        b"\x1b]title\x07",
        b"\x1bPdata\x1b\\",
    )
    bounded = all(not sequence_is_bounded(value) for value in rejected) and all(
        sequence_is_bounded(value) for value in accepted
    )
    rows.append(
        _row(
            "parser-incomplete-sequence-bounds",
            bounded,
            **_sequence_limit_facts(
                accepted_fixture_count=len(accepted),
                rejected_fixture_count=len(rejected),
            ),
        )
    )

    screen = pyte_module.Screen(80, 24)
    stream = pyte_module.Stream(screen)
    classifications = {
        "buffer": "viewport-bounded",
        "charset": "static",
        "dirty": "viewport-bounded",
        "mode": "static",
        "savepoints": "adapter-capped",
        "tabstops": "viewport-bounded",
    }
    observed = {
        name
        for owner in (screen, stream)
        for name, value in vars(owner).items()
        if isinstance(value, (dict, list, set, deque))
    }
    unknown = observed - classifications.keys()
    rows.append(
        _row(
            "parser-mutable-collections",
            not unknown,
            observed_mutable_names=sorted(observed),
            unknown_mutable_count=len(unknown),
            classifications=classifications,
        )
    )

    tracemalloc.start()
    tracemalloc.reset_peak()
    memory_screen = pyte_module.Screen(300, 120)
    memory_stream = pyte_module.Stream(memory_screen)
    for number in range(2000):
        memory_stream.feed(f"row {number:04d} terminal qualification\r\n")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rows.append(
        _row(
            "parser-memory-bound",
            peak <= PARSER_MEMORY_LIMIT,
            tracemalloc_peak_bytes=peak,
            limit_bytes=PARSER_MEMORY_LIMIT,
            viewport_columns=300,
            viewport_rows=120,
            feed_row_count=2000,
        )
    )
    return rows


def probe(manifest_path: Path, json_out: Path, *, replace: bool) -> bool:
    """Run the binding and parser matrix and write content-free facts."""
    started_at = utc_now()
    started = time.monotonic()
    manifest = artifact_manifest(manifest_path, required_distribution="pyte")
    import pyte

    binding_passed, binding_facts = _artifact_binding(manifest)
    rows = [_row("package-pyte-0.8.2", binding_passed, **binding_facts)]
    rows.extend(_parser_rows(pyte))
    passed = all(row["status"] == "PASS" for row in rows)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "row_id": manifest["row_id"],
        "probe": "pyte",
        "status": "PASS" if passed else "FAIL",
        "mandatory": True,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "command": command_facts(),
        "platform": platform_facts(),
        "measurements": memory_facts(),
        "runtime": manifest.get("runtime", {"kind": "host"}),
        "term": TERM_VALUE,
        "rows": rows,
    }
    write_probe_result(json_out, payload, replace=replace)
    return passed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return (
            0
            if probe(args.artifact_manifest, args.json_out, replace=args.replace)
            else 1
        )
    except (
        QualificationError,
        OSError,
        subprocess.SubprocessError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        print(f"pyte qualification failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
