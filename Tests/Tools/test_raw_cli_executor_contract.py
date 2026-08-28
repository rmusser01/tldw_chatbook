"""Contracts for the raw one-shot CLI boundary.

The executor itself stays synthetic here; one regression launches a fresh
interpreter solely to prove the worker import graph is acyclic.
"""

from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import get_args

import pytest


@pytest.fixture
def raw_cli():
    """Load lazily so RED evidence is assertion-level, not a collection error."""
    module_name = "tldw_chatbook.Tools.raw_cli_executor"
    if importlib.util.find_spec(module_name) is None:
        return SimpleNamespace()
    return importlib.import_module(module_name)


def _required(module, name: str):
    value = getattr(module, name, None)
    assert value is not None, f"raw CLI contract {name} is missing"
    return value


def _request(raw_cli, directory: Path, **overrides):
    values = {
        "invocation_id": "inv-1",
        "caller": "user",
        "command": "printf hello",
        "shell": "auto",
        "initial_directory": directory,
        "timeout_seconds": 30.0,
        "console_session_id": "console-1",
        "transcript_anchor_id": None,
    }
    values.update(overrides)
    return _required(raw_cli, "RawCliRequest")(**values)


def test_raw_cli_executor_imports_in_a_fresh_interpreter() -> None:
    """The spawned worker must import its executor without parent preload order."""
    project_root = Path(__file__).resolve().parents[2]
    source = (
        "import sys; "
        f"sys.path.insert(0, {str(project_root)!r}); "
        "import tldw_chatbook.Tools.raw_cli_executor"
    )

    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=20.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_raw_cli_public_vocabulary_and_limits_are_pinned(raw_cli):
    assert get_args(_required(raw_cli, "RawCliCaller")) == ("user", "model")
    assert get_args(_required(raw_cli, "RawCliShell")) == (
        "auto",
        "bash",
        "powershell",
        "cmd",
    )
    assert get_args(_required(raw_cli, "RawCliTerminalState")) == (
        "refused",
        "shell_unavailable",
        "spawn_failed",
        "containment_unavailable",
        "exited",
        "timed_out",
        "cancelled",
        "cleanup_unproven",
    )
    assert _required(raw_cli, "MAX_RAW_COMMAND_BYTES") == 16 * 1024
    assert _required(raw_cli, "MAX_RAW_TIMEOUT_SECONDS") == 300.0
    assert _required(raw_cli, "MAX_RAW_PREVIEW_BYTES") == 32 * 1024


def test_raw_cli_value_objects_are_frozen_slotted_contracts(raw_cli, tmp_path):
    request = _request(raw_cli, tmp_path)
    assert is_dataclass(request)
    assert not hasattr(request, "__dict__")
    assert [field.name for field in fields(request)] == [
        "invocation_id",
        "caller",
        "command",
        "shell",
        "initial_directory",
        "timeout_seconds",
        "console_session_id",
        "transcript_anchor_id",
    ]
    with pytest.raises(FrozenInstanceError):
        request.command = "changed"

    event = _required(raw_cli, "RawCliStreamEvent")(
        stream="stdout", text="hello", total_bytes=5, truncated=False
    )
    result = _required(raw_cli, "RawCliResult")(
        invocation_id="inv-1",
        caller="user",
        resolved_shell="bash",
        initial_directory=tmp_path,
        elapsed_seconds=0.25,
        stdout_preview="hello",
        stderr_preview="",
        record_output="[stdout] hello",
        exit_code=0,
        terminal_state="exited",
        truncated=False,
        cleanup_proven=True,
    )
    for value in (event, result):
        assert is_dataclass(value)
        assert not hasattr(value, "__dict__")
        with pytest.raises(FrozenInstanceError):
            value.truncated = True

    with pytest.raises(ValueError, match="stdout.*stderr"):
        _required(raw_cli, "RawCliStreamEvent")(
            stream="combined", text="bad", total_bytes=3, truncated=False
        )


@pytest.mark.parametrize("command", ["", " \t\n", "echo\x00unsafe"])
def test_raw_request_rejects_empty_whitespace_and_nul(raw_cli, tmp_path, command):
    request = _request(raw_cli, tmp_path, command=command)

    with pytest.raises(ValueError):
        _required(raw_cli, "validate_raw_cli_request")(request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("caller", "system"),
        ("caller", None),
        ("shell", "zsh"),
        ("shell", None),
    ],
)
def test_raw_request_rejects_invalid_caller_or_shell(raw_cli, tmp_path, field, value):
    request = _request(raw_cli, tmp_path, **{field: value})

    with pytest.raises(ValueError, match=field):
        _required(raw_cli, "validate_raw_cli_request")(request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("invocation_id", None),
        ("invocation_id", 42),
        ("invocation_id", ""),
        ("invocation_id", " \t\n"),
        ("console_session_id", None),
        ("console_session_id", 42),
        ("console_session_id", ""),
        ("console_session_id", " \t\n"),
    ],
)
def test_raw_request_requires_nonblank_string_identifiers(
    raw_cli, tmp_path, field, value
):
    request = _request(raw_cli, tmp_path, **{field: value})

    with pytest.raises(ValueError, match=field):
        _required(raw_cli, "validate_raw_cli_request")(request)


def test_raw_request_enforces_16_kib_utf8_boundary(raw_cli, tmp_path):
    validate = _required(raw_cli, "validate_raw_cli_request")
    validate(_request(raw_cli, tmp_path, command="x" * (16 * 1024)))

    for command in ("x" * (16 * 1024 + 1), "é" * (8 * 1024 + 1)):
        with pytest.raises(ValueError, match="16 KiB"):
            validate(_request(raw_cli, tmp_path, command=command))


def test_raw_request_uses_shared_command_validator_result(
    raw_cli, tmp_path, monkeypatch
):
    calls = []

    def validate_command(command, *, max_bytes):
        calls.append((command, max_bytes))
        return "normalized command"

    monkeypatch.setattr(raw_cli, "validate_raw_cli_command", validate_command)

    request = _required(raw_cli, "validate_raw_cli_request")(
        _request(raw_cli, tmp_path, command="original command")
    )

    assert calls == [("original command", 16 * 1024)]
    assert request.command == "normalized command"


def test_raw_request_uses_central_normalized_directory(raw_cli, tmp_path, monkeypatch):
    selected = tmp_path / "selected"
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    calls = []

    def validate_directory(directory):
        calls.append(directory)
        return normalized

    monkeypatch.setattr(
        raw_cli,
        "validate_existing_absolute_directory",
        validate_directory,
    )

    request = _required(raw_cli, "validate_raw_cli_request")(
        _request(raw_cli, selected)
    )

    assert calls == [selected]
    assert request.initial_directory == normalized


def test_raw_request_requires_existing_absolute_initial_directory(
    raw_cli, tmp_path, monkeypatch
):
    validate = _required(raw_cli, "validate_raw_cli_request")
    validate(_request(raw_cli, tmp_path.resolve()))

    monkeypatch.chdir(tmp_path)
    relative_directory = Path("relative")
    relative_directory.mkdir()
    invalid_directories = (
        relative_directory,
        tmp_path / "missing",
        tmp_path / "file.txt",
    )
    (tmp_path / "file.txt").write_text("not a directory", encoding="utf-8")

    for directory in invalid_directories:
        with pytest.raises(ValueError, match="absolute.*directory"):
            validate(_request(raw_cli, directory))


@pytest.mark.parametrize("timeout", [0.0, -1.0, 300.0001, float("inf"), float("nan")])
def test_timeout_may_lower_but_not_exceed_300_seconds(raw_cli, tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        _required(raw_cli, "validate_raw_cli_request")(
            _request(raw_cli, tmp_path, timeout_seconds=timeout)
        )


@pytest.mark.parametrize("timeout", [0.001, 300.0])
def test_timeout_accepts_positive_values_through_maximum(raw_cli, tmp_path, timeout):
    _required(raw_cli, "validate_raw_cli_request")(
        _request(raw_cli, tmp_path, timeout_seconds=timeout)
    )


def _lookup(available: dict[str, str]):
    return available.get


def test_bash_argv_disables_profiles(raw_cli):
    argv = _required(raw_cli, "resolve_shell_argv")(
        "bash",
        "echo hello",
        executable_lookup=_lookup({"bash": "/bin/bash"}),
        platform_name="posix",
    )

    assert argv == ("/bin/bash", "--noprofile", "--norc", "-c", "echo hello")


@pytest.mark.parametrize("executable", ["pwsh", "powershell"])
def test_powershell_argv_disables_profiles(raw_cli, executable):
    executable_path = rf"C:\Tools\{executable}.exe"
    argv = _required(raw_cli, "resolve_shell_argv")(
        "powershell",
        "Write-Output hello",
        executable_lookup=_lookup({executable: executable_path}),
        platform_name="nt",
    )

    assert argv == (
        executable_path,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Write-Output hello",
    )


def test_cmd_argv_disables_autorun(raw_cli):
    argv = _required(raw_cli, "resolve_shell_argv")(
        "cmd",
        "echo hello",
        executable_lookup=_lookup({"cmd.exe": r"C:\Windows\cmd.exe"}),
        platform_name="nt",
    )

    assert argv == (r"C:\Windows\cmd.exe", "/D", "/S", "/C", "echo hello")


@pytest.mark.parametrize(
    ("platform_name", "available", "expected"),
    [
        (
            "posix",
            {"bash": "/bin/bash", "sh": "/bin/sh"},
            ("/bin/bash", "--noprofile", "--norc", "-c", "echo hello"),
        ),
        ("posix", {"sh": "/bin/sh"}, ("/bin/sh", "-c", "echo hello")),
        (
            "nt",
            {"pwsh": r"C:\Tools\pwsh.exe", "cmd.exe": r"C:\Windows\cmd.exe"},
            (
                r"C:\Tools\pwsh.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "echo hello",
            ),
        ),
        (
            "nt",
            {"powershell": r"C:\Windows\powershell.exe"},
            (
                r"C:\Windows\powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "echo hello",
            ),
        ),
        (
            "nt",
            {"cmd.exe": r"C:\Windows\cmd.exe"},
            (r"C:\Windows\cmd.exe", "/D", "/S", "/C", "echo hello"),
        ),
    ],
)
def test_auto_shell_resolution_uses_platform_fallback_order(
    raw_cli, platform_name, available, expected
):
    argv = _required(raw_cli, "resolve_shell_argv")(
        "auto",
        "echo hello",
        executable_lookup=_lookup(available),
        platform_name=platform_name,
    )

    assert argv == expected


def test_relative_posix_shell_lookup_is_made_absolute_at_lookup_time(
    raw_cli, monkeypatch
):
    expanded = "/resolved/tools/bash"
    expanded_paths = []
    monkeypatch.setattr(
        raw_cli.posixpath,
        "abspath",
        lambda path: expanded_paths.append(path) or expanded,
    )

    argv = _required(raw_cli, "resolve_shell_argv")(
        "bash",
        "echo hello",
        executable_lookup=_lookup({"bash": "tools/bash"}),
        platform_name="posix",
    )

    assert expanded_paths == ["tools/bash"]
    assert argv[0] == expanded


def test_relative_windows_shell_lookup_uses_windows_path_semantics(
    raw_cli, monkeypatch
):
    expanded = r"C:\resolved\tools\pwsh.exe"
    expanded_paths = []
    monkeypatch.setattr(
        raw_cli.ntpath,
        "abspath",
        lambda path: expanded_paths.append(path) or expanded,
    )

    argv = _required(raw_cli, "resolve_shell_argv")(
        "powershell",
        "Write-Output hello",
        executable_lookup=_lookup({"pwsh": r"tools\pwsh.exe"}),
        platform_name="nt",
    )

    assert expanded_paths == [r"tools\pwsh.exe"]
    assert argv[0] == expanded


def test_windows_absolute_path_is_expanded_under_posix_semantics(raw_cli, monkeypatch):
    expanded = "/resolved/C:\\Tools\\bash.exe"
    expanded_paths = []
    monkeypatch.setattr(
        raw_cli.posixpath,
        "abspath",
        lambda path: expanded_paths.append(path) or expanded,
    )

    argv = _required(raw_cli, "resolve_shell_argv")(
        "bash",
        "echo hello",
        executable_lookup=_lookup({"bash": r"C:\Tools\bash.exe"}),
        platform_name="posix",
    )

    assert expanded_paths == [r"C:\Tools\bash.exe"]
    assert argv[0] == expanded


@pytest.mark.parametrize(
    "executable",
    [r"/tools/pwsh.exe", r"\tools\pwsh.exe", r"C:tools\pwsh.exe"],
)
def test_non_fully_qualified_windows_path_is_expanded(raw_cli, monkeypatch, executable):
    expanded = r"C:\resolved\tools\pwsh.exe"
    expanded_paths = []
    monkeypatch.setattr(
        raw_cli.ntpath,
        "abspath",
        lambda path: expanded_paths.append(path) or expanded,
    )

    argv = _required(raw_cli, "resolve_shell_argv")(
        "powershell",
        "Write-Output hello",
        executable_lookup=_lookup({"pwsh": executable}),
        platform_name="nt",
    )

    assert expanded_paths == [executable]
    assert argv[0] == expanded


@pytest.mark.parametrize(
    "executable",
    [r"C:\Tools\pwsh.exe", r"\\server\share\pwsh.exe"],
)
def test_absolute_windows_shell_lookup_is_preserved(raw_cli, monkeypatch, executable):
    monkeypatch.setattr(
        raw_cli.ntpath,
        "abspath",
        lambda path: pytest.fail(f"unexpected expansion of {path}"),
    )
    argv = _required(raw_cli, "resolve_shell_argv")(
        "powershell",
        "Write-Output hello",
        executable_lookup=_lookup({"pwsh": executable}),
        platform_name="nt",
    )

    assert argv[0] == executable


@pytest.mark.parametrize("selector", ["bash", "powershell", "cmd", "auto"])
def test_unavailable_shell_fails_clearly(raw_cli, selector):
    with pytest.raises(FileNotFoundError, match="shell.*unavailable"):
        _required(raw_cli, "resolve_shell_argv")(
            selector,
            "echo hello",
            executable_lookup=_lookup({}),
            platform_name="posix",
        )


def test_build_scrubbed_environment_copies_only_usability_allowlist(raw_cli):
    source = {
        "PATH": "/usr/bin",
        "HOME": "/home/example",
        "USERPROFILE": r"C:\\Users\\example",
        "TMPDIR": "/tmp/example",
        "TEMP": r"C:\\Temp",
        "TMP": "/tmp",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C.UTF-8",
        "SYSTEMROOT": r"C:\\Windows",
        "WINDIR": r"C:\\Windows",
        "COMSPEC": r"C:\\Windows\\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "OPENAI_API_KEY": "secret",
        "HTTPS_PROXY": "http://credential@proxy",
        "OTEL_EXPORTER_OTLP_HEADERS": "authorization=secret",
        "PYTHONPATH": "/inject",
        "PYTHONINSPECT": "1",
        "BASH_ENV": "/inject.sh",
    }

    environment = _required(raw_cli, "build_scrubbed_environment")(source)

    assert environment == {
        key: source[key]
        for key in (
            "PATH",
            "HOME",
            "USERPROFILE",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
        )
    }


def test_build_scrubbed_environment_defaults_to_ambient_allowlist(raw_cli, monkeypatch):
    monkeypatch.setenv("PATH", "/safe-bin")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    environment = _required(raw_cli, "build_scrubbed_environment")()

    assert environment["PATH"] == "/safe-bin"
    assert "OPENAI_API_KEY" not in environment
