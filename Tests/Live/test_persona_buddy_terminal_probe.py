"""Failure-evidence contract for the bounded Persona Buddy PTY probe."""

from __future__ import annotations

import importlib.util
import ast
import errno
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_CAPTURE_NAMES = {
    "normal.ansi",
    "alert.ansi",
    "folded.ansi",
    "constrained.ansi",
}
_VISUAL_CHECKS = {
    "pet_only_normal",
    "fixed_alert_replaces_pet",
    "real_folded_thumbnail",
    "constrained_two_icons",
}


def _load_probe_module():
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    spec = importlib.util.spec_from_file_location("persona_buddy_terminal_probe", probe)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_probe_persists_four_exact_visual_state_captures(tmp_path: Path) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report = tmp_path / "report.json"
    captures = tmp_path / "captures"
    captures.mkdir()
    report_collision = tmp_path / f".report.json.{os.getpid()}.tmp"
    capture_collision = captures / f".normal.ansi.{os.getpid()}.tmp"
    report_collision.write_bytes(b"caller-owned report collision")
    capture_collision.write_bytes(b"caller-owned capture collision")

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--report",
            str(report),
            "--capture-dir",
            str(captures),
            "--inject-startup-noise",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert report_collision.read_bytes() == b"caller-owned report collision"
    assert capture_collision.read_bytes() == b"caller-owned capture collision"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert all(payload["checks"][name] for name in _VISUAL_CHECKS)
    assert set(payload["regions"]) == {
        "normal",
        "alert",
        "folded",
        "constrained",
    }
    for region in payload["regions"].values():
        assert set(region) == {"x", "y", "width", "height"}
        assert all(type(value) is int for value in region.values())
        assert region["width"] > 0
        assert region["height"] > 0

    artifacts = {
        path.name for path in captures.iterdir() if path.name in _CAPTURE_NAMES
    }
    assert artifacts == _CAPTURE_NAMES
    capture_values = {(captures / name).read_bytes() for name in artifacts}
    assert len(capture_values) == len(_CAPTURE_NAMES)
    for value in capture_values:
        assert 0 < len(value) <= 256 * 1024
        assert value.decode("utf-8")
        assert b"\x1b[" in value
        for forbidden in (
            b"PERSONA_BUDDY_PRIVATE_STARTUP_MARKER",
            b"/private/tmp/private-checkout/config.toml",
            b"provider_inventory",
        ):
            assert forbidden not in value
    assert "Traceback" not in completed.stdout
    assert "Traceback" not in completed.stderr
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "/Users/",
        "/private/",
        "provider",
        "prompt",
        "assistant",
        "tool_arguments",
        "tool_result",
    ):
        assert forbidden not in serialized

    failed_report = tmp_path / "failed-check.json"
    failed_captures = tmp_path / "failed-captures"
    failed_captures.mkdir()
    for name in _CAPTURE_NAMES:
        (failed_captures / name).write_text("stale", encoding="utf-8")
    failed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--report",
            str(failed_report),
            "--capture-dir",
            str(failed_captures),
            "--inject-check-failure",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert failed.returncode == 1
    failure = json.loads(failed_report.read_text(encoding="utf-8"))
    assert failure["status"] == "FAIL"
    assert failure["category"] == "persona_buddy_terminal_check_failure"
    assert failure["phase"] == "parent:checks"
    assert failure["parent_return_code"] == 1
    assert failure["child_return_code"] == 0
    assert len(failure["checks"]) == 22
    assert set(failure["checks"]) == set(payload["checks"])
    assert failure["checks"]["drag"] is False
    assert all(
        value is True for name, value in failure["checks"].items() if name != "drag"
    )
    assert failure["check_statuses"]["drag"] == "failed"
    assert set(failure["check_statuses"].values()) == {"passed", "failed"}
    assert len(failure["diagnostic_tail"].encode("utf-8")) <= 16_000
    artifact = Path(failure["diagnostic_artifact"])
    assert artifact.is_file()
    assert artifact.parent == failed_report.parent
    assert failure["category"] in artifact.read_text(encoding="utf-8")
    assert failure["category"] in failed.stderr
    assert not any((failed_captures / name).exists() for name in _CAPTURE_NAMES)


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_success_requires_a_caller_owned_capture_directory(tmp_path: Path) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")

    completed = subprocess.run(
        [sys.executable, str(probe), "--report", str(tmp_path / "report.json")],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "--capture-dir" in completed.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_injected_failure_requires_capture_directory_before_parent_work(
    tmp_path: Path,
) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--inject-child-failure",
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "--capture-dir" in completed.stderr
    assert completed.stdout == ""
    assert not report.exists()
    assert list(tmp_path.glob("*.diagnostic.log")) == []


def test_diagnostic_tail_is_bounded_in_utf8_bytes_and_retains_category() -> None:
    probe = _load_probe_module()
    category = "persona_buddy_terminal_probe_failure"

    diagnostic = probe._bounded_utf8_tail(f"{category}\n" + "é" * 16_000)
    encoded = diagnostic.encode("utf-8")

    assert len(encoded) <= 16_384
    assert encoded.decode("utf-8") == diagnostic
    assert diagnostic.startswith(f"{category}\n")
    assert diagnostic.endswith("é")


def test_terminal_capture_rejects_overflow_private_content_and_partial_ansi() -> None:
    probe = _load_probe_module()

    assert probe._CAPTURE_BYTES <= 256 * 1024
    with pytest.raises(ValueError, match="capture_too_large"):
        probe._validate_terminal_capture(b"x" * (probe._CAPTURE_BYTES + 1))
    with pytest.raises(ValueError, match="capture_private_content"):
        probe._validate_terminal_capture(
            b"\x1b[2J/private/tmp/private-checkout/config.toml"
        )
    with pytest.raises(ValueError, match="capture_incomplete_ansi"):
        probe._validate_terminal_capture(b"\x1b[2J\x1b[")
    with pytest.raises(UnicodeDecodeError):
        probe._validate_terminal_capture(b"\x1b[2J\xc3")


def test_repaint_capture_discards_cumulative_stream_before_fresh_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    private_noise = (
        b"PERSONA_BUDDY_PRIVATE_STARTUP_MARKER "
        b"/private/tmp/private-checkout/config.toml provider_inventory\n"
        + b"x"
        * (probe._CAPTURE_BYTES * 2)
    )
    state = b"\x1b[2J\x1b[Hfresh exact state\x1b[0m"
    drains = iter((private_noise, state))
    writes: list[bytes] = []

    monkeypatch.setattr(
        probe, "_drain_until_quiet", lambda *_args, **_kwargs: next(drains)
    )
    monkeypatch.setattr(probe.os, "write", lambda _fd, value: writes.append(value))

    assert probe._capture_fresh_repaint(7) == state
    assert writes == [b"\x0c"]


def test_drain_helpers_treat_linux_eio_as_clean_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.select, "select", lambda *_args: ([7], [], []))

    def raise_eio(*_args):
        raise OSError(errno.EIO, "pty eof")

    monkeypatch.setattr(probe.os, "read", raise_eio)
    assert probe._drain_for(7, 0.01) == b""
    assert probe._drain_until_quiet(7, timeout=0.01) == b""

    source = Path(probe.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr == "read"
    ]
    assert len(direct_reads) == 1
    assert direct_reads[0].lineno == probe._read_pty.__code__.co_firstlineno + 4


def test_drain_helpers_propagate_non_eio_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.select, "select", lambda *_args: ([7], [], []))

    def raise_bad_fd(*_args):
        raise OSError(errno.EBADF, "bad fd")

    monkeypatch.setattr(probe.os, "read", raise_bad_fd)
    with pytest.raises(OSError, match="bad fd"):
        probe._drain_for(7, 0.01)
    with pytest.raises(OSError, match="bad fd"):
        probe._drain_until_quiet(7, timeout=0.01)

    class FailedProcess:
        returncode = 1

        def poll(self):
            return self.returncode

    monkeypatch.setattr(probe.pty, "openpty", lambda: (7, 8))
    monkeypatch.setattr(probe, "_set_size", lambda *_args: None)
    monkeypatch.setattr(probe.os, "close", lambda *_args: None)
    monkeypatch.setattr(
        probe.subprocess, "Popen", lambda *_args, **_kwargs: FailedProcess()
    )
    monkeypatch.setattr(probe, "_TIMEOUT_SECONDS", 0.0)
    with pytest.raises(probe._ProbeChildFailure) as caught:
        probe._run_child(
            root=tmp_path,
            preferences=tmp_path / "preferences.json",
            report=tmp_path / "report.json",
            drive=False,
            phase="runtime-read",
        )

    assert caught.value.category == "persona_buddy_terminal_child_exit"
    assert caught.value.phase == "runtime-read:startup"
    assert caught.value.child_return_code == 1
    assert "OSError: [Errno 9] bad fd" in caught.value.diagnostic_tail


def test_posix_modules_are_imported_only_below_the_platform_guard() -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    tree = ast.parse(probe.read_text(encoding="utf-8"))
    top_level = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert {"fcntl", "pty", "termios"}.isdisjoint(top_level)
    source = probe.read_text(encoding="utf-8")
    assert (
        'if os.name == "nt":\n        print("SKIP persona_buddy_terminal windows_no_posix_pty")'
        in source
    )


def test_output_preflight_never_removes_an_unrelated_collision(tmp_path: Path) -> None:
    probe = _load_probe_module()
    collision = tmp_path / f".capture.{os.getpid()}.admission.tmp"
    collision.write_text("caller-owned", encoding="utf-8")

    probe._preflight_atomic_directory(tmp_path, "capture")

    assert collision.read_text(encoding="utf-8") == "caller-owned"

    report = tmp_path / "report.json"
    report_collision = tmp_path / f".report.json.{os.getpid()}.tmp"
    report_collision.write_bytes(b"caller-owned report collision")
    probe._atomic_write_json(report, {"status": "PASS"})
    assert json.loads(report.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert report_collision.read_bytes() == b"caller-owned report collision"

    failed_report = tmp_path / "failed.json"
    failed_collision = tmp_path / f".failed.json.{os.getpid()}.tmp"
    failed_collision.write_bytes(b"caller-owned failed collision")
    with pytest.raises(OSError, match="report_publish_injected"):
        probe._atomic_write_json(
            failed_report,
            {"status": "FAIL"},
            inject_replace_failure=True,
        )
    assert not failed_report.exists()
    assert failed_collision.read_bytes() == b"caller-owned failed collision"

    capture = tmp_path / "capture.ansi"
    capture_collision = tmp_path / f".capture.ansi.{os.getpid()}.tmp"
    capture_collision.write_bytes(b"caller-owned capture collision")
    probe._atomic_write_bytes(capture, b"\x1b[2Jstate")
    assert capture.read_bytes() == b"\x1b[2Jstate"
    assert capture_collision.read_bytes() == b"caller-owned capture collision"


def test_last_resort_diagnostic_reserves_random_owned_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.tempfile, "gettempdir", lambda: str(tmp_path))
    old_fallback = tmp_path / f"persona-buddy-terminal-{os.getpid()}.diagnostic.log"
    old_fallback.write_bytes(b"caller-owned diagnostic sentinel")
    report = tmp_path / "failure.json"
    preferred_diagnostic = tmp_path / "failure.diagnostic.log"
    original_write = probe._atomic_write_text

    def fail_preferred_diagnostic(path, value, **kwargs):
        if path == preferred_diagnostic:
            raise OSError("diagnostic_publish_impossible")
        return original_write(path, value, **kwargs)

    monkeypatch.setattr(probe, "_atomic_write_text", fail_preferred_diagnostic)
    failure = probe._ProbeChildFailure(
        category="persona_buddy_terminal_probe_failure",
        phase="parent:report",
        child_return_code=1,
        diagnostic_tail=(
            "persona_buddy_terminal_probe_failure\n" + "é" * probe._DIAGNOSTIC_BYTES
        ),
    )

    result, diagnostic_path, actual_report = probe._persist_structured_failure(
        failure,
        report_output=report,
        root=tmp_path / "repo",
        temporary=None,
    )

    assert old_fallback.read_bytes() == b"caller-owned diagnostic sentinel"
    assert diagnostic_path != old_fallback
    assert diagnostic_path.parent == tmp_path
    assert diagnostic_path.name.startswith("persona-buddy-terminal-")
    assert diagnostic_path.name.endswith(".diagnostic.log")
    assert diagnostic_path.read_text(encoding="utf-8") == result["diagnostic_tail"]
    assert len(diagnostic_path.read_bytes()) <= probe._DIAGNOSTIC_BYTES
    assert result["diagnostic_artifact"] == str(diagnostic_path)
    assert actual_report == report
    assert not list(tmp_path.glob("*.tmp"))


def test_last_resort_report_reserves_random_owned_destination_and_cleans_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    monkeypatch.setattr(probe.tempfile, "gettempdir", lambda: str(tmp_path))
    old_fallback = tmp_path / f"persona-buddy-terminal-{os.getpid()}.failure.json"
    old_fallback.write_bytes(b"caller-owned report sentinel")
    report_attempts: list[Path] = []

    def fail_report_publication(path, _value, **_kwargs):
        report_attempts.append(path)
        raise OSError("report_publish_impossible")

    monkeypatch.setattr(probe, "_atomic_write_json", fail_report_publication)
    failure = probe._ProbeChildFailure(
        category="persona_buddy_terminal_probe_failure",
        phase="parent:report",
        child_return_code=1,
        diagnostic_tail="persona_buddy_terminal_probe_failure",
    )

    result, diagnostic_path, actual_report = probe._persist_structured_failure(
        failure,
        report_output=tmp_path / "failure.json",
        root=tmp_path / "repo",
        temporary=None,
    )

    assert len(report_attempts) == 2
    assert old_fallback.read_bytes() == b"caller-owned report sentinel"
    assert actual_report != old_fallback
    assert actual_report.parent == tmp_path
    assert actual_report.name.startswith("persona-buddy-terminal-")
    assert actual_report.name.endswith(".failure.json")
    assert json.loads(actual_report.read_text(encoding="utf-8")) == result
    assert diagnostic_path.is_file()
    assert not list(tmp_path.glob("*.tmp"))

    before_failed_write = set(tmp_path.iterdir())

    def fail_fsync(_descriptor):
        raise OSError("fallback_fsync_impossible")

    monkeypatch.setattr(probe.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fallback_fsync_impossible"):
        probe._write_exclusive_fallback_text("failure", suffix=".failure.json")
    assert set(tmp_path.iterdir()) == before_failed_write


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_injected_child_failure_persists_atomic_parent_evidence(tmp_path: Path) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report = tmp_path / "failure-report.json"
    captures = tmp_path / "captures"
    captures.mkdir()
    report.write_text('{"status":"STALE"}', encoding="utf-8")
    collision = tmp_path / f".failure-report.json.{os.getpid()}.tmp"
    collision.write_bytes(b"caller-owned failure collision")
    for name in _CAPTURE_NAMES:
        (captures / name).write_text("stale", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--inject-child-failure",
            "--report",
            str(report),
            "--capture-dir",
            str(captures),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["category"] == "persona_buddy_terminal_child_exit"
    assert payload["phase"] == "first:startup"
    assert payload["parent_return_code"] == 1
    assert payload["child_return_code"] == 1
    assert "Traceback (most recent call last)" in payload["diagnostic_tail"]
    assert "persona_buddy_injected_child_failure" in payload["diagnostic_tail"]
    assert len(payload["diagnostic_tail"].encode("utf-8")) <= 16_000
    assert "persona-buddy-terminal-" not in payload["diagnostic_tail"]
    assert payload["checks"] and not any(payload["checks"].values())
    assert set(payload["check_statuses"].values()) == {"not_run"}

    artifact = Path(payload["diagnostic_artifact"])
    assert artifact.is_file()
    assert artifact.parent == report.parent
    assert "persona_buddy_injected_child_failure" in artifact.read_text(
        encoding="utf-8"
    )
    assert payload["category"] in completed.stderr
    assert str(artifact) in completed.stderr
    assert collision.read_bytes() == b"caller-owned failure collision"
    assert not any((captures / name).exists() for name in _CAPTURE_NAMES)


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_capture_publication_failure_rolls_back_managed_group(tmp_path: Path) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report = tmp_path / "failure-report.json"
    captures = tmp_path / "captures"
    captures.mkdir()
    report.write_text('{"status":"STALE"}', encoding="utf-8")
    for name in _CAPTURE_NAMES:
        (captures / name).write_text("stale", encoding="utf-8")
    unrelated = captures / "keep.txt"
    unrelated.write_text("caller-owned", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--inject-publication-failure",
            "--report",
            str(report),
            "--capture-dir",
            str(captures),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["category"] == "persona_buddy_terminal_capture_publish"
    assert payload["phase"] == "parent:capture"
    assert payload["parent_return_code"] == 1
    assert payload["child_return_code"] == 0
    artifact = Path(payload["diagnostic_artifact"])
    assert artifact.is_file()
    assert payload["category"] in artifact.read_text(encoding="utf-8")
    assert payload["category"] in completed.stderr
    assert not any((captures / name).exists() for name in _CAPTURE_NAMES)
    assert unrelated.read_text(encoding="utf-8") == "caller-owned"


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_capture_directory_regular_file_fails_structured_before_child(
    tmp_path: Path,
) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report = tmp_path / "failure-report.json"
    report.write_text('{"status":"STALE"}', encoding="utf-8")
    capture_file = tmp_path / "captures"
    capture_file.write_text("caller-owned", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--inject-child-failure",
            "--report",
            str(report),
            "--capture-dir",
            str(capture_file),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["category"] == "persona_buddy_terminal_output_admission"
    assert payload["phase"] == "parent:admission"
    assert payload["child_return_code"] == 0
    assert capture_file.read_text(encoding="utf-8") == "caller-owned"
    assert "Traceback" not in completed.stderr
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_report_directory_fails_to_safe_sibling_before_child_and_rolls_back(
    tmp_path: Path,
) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report_dir = tmp_path / "reportdir"
    report_dir.mkdir()
    keeper = report_dir / "keep.txt"
    keeper.write_text("caller-owned", encoding="utf-8")
    captures = tmp_path / "captures"
    captures.mkdir()
    for name in _CAPTURE_NAMES:
        (captures / name).write_text("stale", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--inject-child-failure",
            "--report",
            str(report_dir),
            "--capture-dir",
            str(captures),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 1
    failure_reports = list(tmp_path.glob(".reportdir.*.failure.json"))
    assert len(failure_reports) == 1
    failure_report = failure_reports[0]
    payload = json.loads(failure_report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["category"] == "persona_buddy_terminal_output_admission"
    assert payload["phase"] == "parent:admission"
    assert payload["child_return_code"] == 0
    assert keeper.read_text(encoding="utf-8") == "caller-owned"
    assert report_dir.is_dir()
    assert not any((captures / name).exists() for name in _CAPTURE_NAMES)
    assert "Traceback" not in completed.stderr
    assert str(failure_report) in completed.stderr
    assert not list(tmp_path.glob(".reportdir.*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_report_replace_failure_is_structured_and_leaves_no_captures_or_temp(
    tmp_path: Path,
) -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")
    report = tmp_path / "report.json"
    report.write_text('{"status":"STALE"}', encoding="utf-8")
    captures = tmp_path / "captures"

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--inject-report-publication-failure",
            "--report",
            str(report),
            "--capture-dir",
            str(captures),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["category"] == "persona_buddy_terminal_report_publish"
    assert payload["phase"] == "parent:report"
    assert payload["child_return_code"] == 0
    assert not any((captures / name).exists() for name in _CAPTURE_NAMES)
    assert "Traceback" not in completed.stderr
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY capability required")
def test_probe_rejects_cli_paths_outside_workspace_and_temp_roots() -> None:
    probe = Path(__file__).with_name("persona_buddy_terminal_probe.py")

    completed = subprocess.run(
        [
            sys.executable,
            str(probe),
            "--child",
            "/etc/persona-buddy-preferences.json",
            "/etc/persona-buddy-report.json",
            "--inject-child-failure",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "persona_buddy_probe_path_invalid" in completed.stderr
    assert "/etc/" not in completed.stderr
