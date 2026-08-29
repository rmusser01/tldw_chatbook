#!/usr/bin/env python3
"""Probe shell startup from the ADR-099 content-free scrubbed environment."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from common import (
    SCHEMA_VERSION,
    BoundedResult,
    OwnedProcessJob,
    QualificationError,
    WindowsBootstrapSetup,
    command_facts,
    memory_facts,
    platform_facts,
    run_bounded,
    terminate_owned_group,
    utc_now,
    write_probe_result,
)

if os.name != "nt":
    import pwd
else:  # pragma: no cover - exercised by the required native Windows row
    pwd = None  # type: ignore[assignment]


SHELLS = ("default", "bash", "zsh", "powershell", "cmd")
MAX_CAPTURE_BYTES = 256 * 1024
SHELL_TIMEOUT_SECONDS = 12.0
STARTUP_QUIESCENCE_SECONDS = 0.1
_RESULT_NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
SENSITIVE_PREFIXES = (
    "ANTHROPIC_",
    "AWS_",
    "AZURE_",
    "COHERE_",
    "DD_",
    "GOOGLE_API_",
    "GROQ_",
    "LANGCHAIN_",
    "MISTRAL_",
    "OPENAI_",
    "OTEL_",
    "TRACE",
)
SENSITIVE_EXACT = {
    "ALL_PROXY",
    "GPG_AGENT_INFO",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "PSMODULEPATH",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "SSH_AGENT_PID",
    "SSH_AUTH_SOCK",
}


@dataclass(frozen=True)
class _ShellSelection:
    path: Path | None
    family: str


def _is_sensitive(key: str) -> bool:
    upper = key.upper()
    return upper in SENSITIVE_EXACT or upper.startswith(SENSITIVE_PREFIXES)


def _safe_scalar(value: str | None, *, maximum: int = 4096) -> str | None:
    if value is None or not value or "\x00" in value or len(value) > maximum:
        return None
    return value


def _validated_path(value: str | None) -> str:
    accepted: list[str] = []
    for raw in (value or os.defpath).split(os.pathsep):
        path = Path(raw)
        if path.is_absolute() and path.is_dir() and str(path) not in accepted:
            accepted.append(str(path))
    if not accepted:
        accepted.extend(
            part for part in os.defpath.split(os.pathsep) if Path(part).is_dir()
        )
    if not accepted:
        raise QualificationError("validated PATH has no existing absolute directory")
    return os.pathsep.join(accepted)


def _posix_environment() -> dict[str, str]:
    if pwd is None:
        raise QualificationError("POSIX account database is unavailable")
    account = pwd.getpwuid(os.getuid())
    home = Path(account.pw_dir)
    shell = Path(account.pw_shell)
    if not home.is_absolute() or not home.is_dir():
        raise QualificationError("account home is unavailable")
    if not shell.is_absolute() or not shell.is_file():
        raise QualificationError("account shell is unavailable")
    environment = {
        "PATH": _validated_path(os.environ.get("PATH")),
        "HOME": str(home),
        "USER": account.pw_name,
        "LOGNAME": account.pw_name,
        "SHELL": str(shell),
        "TERM": "linux",
        "TMPDIR": tempfile.gettempdir(),
    }
    for key, value in os.environ.items():
        if key == "LANG" or key == "LC_ALL" or key.startswith("LC_"):
            safe = _safe_scalar(value, maximum=256)
            if safe is not None:
                environment[key] = safe
    return environment


def _windows_environment() -> dict[str, str]:
    allowed = (
        "PATH",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "USERNAME",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
    )
    environment: dict[str, str] = {}
    ambient = {key.upper(): value for key, value in os.environ.items()}
    for key in allowed:
        safe = _safe_scalar(ambient.get(key))
        if safe is not None:
            environment[key] = _validated_path(safe) if key == "PATH" else safe
    environment["TERM"] = "linux"
    return environment


def scrubbed_environment() -> dict[str, str]:
    """Construct the qualification form of the terminal launch environment."""
    environment = _windows_environment() if os.name == "nt" else _posix_environment()
    if any(_is_sensitive(key) for key in environment):
        raise QualificationError("sensitive key entered the scrubbed environment")
    return environment


def _resolve_shell(
    requested: str,
    environment: dict[str, str],
    *,
    windows: bool,
    which: Callable[..., str | None] = shutil.which,
    is_file: Callable[[Path], bool] = Path.is_file,
) -> _ShellSelection:
    if windows:
        if requested in {"default", "powershell"}:
            for executable in ("pwsh.exe", "powershell.exe"):
                found = which(executable, path=environment.get("PATH"))
                if found:
                    return _ShellSelection(Path(found), "powershell")
            if requested == "powershell":
                return _ShellSelection(None, "powershell")
        if requested in {"default", "cmd"}:
            candidate = environment.get("COMSPEC")
            path = Path(candidate) if candidate else None
            return _ShellSelection(
                path if path is not None and is_file(path) else None,
                "cmd",
            )
        return _ShellSelection(None, requested)
    if requested == "default":
        if pwd is None:
            return _ShellSelection(None, "default")
        return _ShellSelection(Path(pwd.getpwuid(os.getuid()).pw_shell), "default")
    if requested in {"bash", "zsh"}:
        found = which(requested, path=environment.get("PATH"))
        return _ShellSelection(Path(found) if found else None, requested)
    return _ShellSelection(None, requested)


def _shell_path(requested: str, environment: dict[str, str]) -> Path | None:
    return _resolve_shell(requested, environment, windows=os.name == "nt").path


def _shell_is_mandatory(requested: str, *, available: bool, windows: bool) -> bool:
    """Return the internally consistent mandatory policy for one named shell."""
    if requested == "default":
        return True
    if windows:
        return requested in {"powershell", "cmd"}
    return requested in {"bash", "zsh"} and available


def _posix_argv(shell: Path) -> list[str]:
    name = shell.name
    if name == "bash":
        return [str(shell), "--login", "-i"]
    if name == "zsh":
        return [str(shell), "-l", "-i"]
    return [str(shell), "-l", "-i"]


def _result_marker(nonce: str) -> str:
    if _RESULT_NONCE_RE.fullmatch(nonce) is None:
        raise QualificationError("result marker nonce is invalid")
    return f"__TLDW_TASK22512_ENV_{nonce}__"


def _single_result_match(
    captured: bytes,
    nonce: str,
    *,
    windows: bool,
) -> re.Match[bytes] | None:
    group_count = 4 if windows else 3
    fields = rb",".join([rb"(\d+)"] * group_count)
    pattern = re.compile(
        rb"(?m)(?:^|(?<=\r))"
        + re.escape(_result_marker(nonce).encode("ascii"))
        + fields
        + rb"\r?$"
    )
    matches = list(pattern.finditer(captured))
    return matches[0] if len(matches) == 1 else None


def _read_pty(
    process: subprocess.Popen[bytes],
    master: int,
    nonce: str,
    input_bytes: bytes,
) -> tuple[bytes, bool]:
    captured = bytearray()
    limit_hit = False
    input_pending = input_bytes
    last_output_at: float | None = None
    deadline = time.monotonic() + SHELL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        now = time.monotonic()
        if (
            input_pending
            and last_output_at is not None
            and now - last_output_at >= STARTUP_QUIESCENCE_SECONDS
        ):
            while input_pending:
                written = os.write(master, input_pending)
                if written <= 0:
                    break
                input_pending = input_pending[written:]
        remaining = MAX_CAPTURE_BYTES - len(captured)
        if remaining <= 0:
            limit_hit = True
            break
        readable, _, _ = select.select([master], [], [], 0.1)
        if readable:
            try:
                chunk = os.read(master, min(4096, remaining))
            except OSError:
                break
            if not chunk:
                break
            captured.extend(chunk)
            last_output_at = time.monotonic()
            continue
        if process.poll() is not None:
            break
    output = bytes(captured)
    found = _single_result_match(output, nonce, windows=False) is not None
    return output, found and not limit_hit


def _stop_posix_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    job = OwnedProcessJob()
    try:
        terminate_owned_group(process, job, grace_seconds=1.0)
    finally:
        job.close()


def _run_posix_shell(shell: Path, environment: dict[str, str]) -> dict[str, bool | int]:
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))

    def child_setup() -> None:
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

    process: subprocess.Popen[bytes] | None = None
    nonce = uuid.uuid4().hex
    marker = _result_marker(nonce)
    captured = b""
    found = False
    try:
        process = subprocess.Popen(
            _posix_argv(shell),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=environment,
            close_fds=True,
            preexec_fn=child_setup,
        )
        os.close(slave)
        slave = -1
        command = (
            "unset HISTFILE; command -v sh >/dev/null 2>&1; _tldw_cmd=$?; "
            '[ "${TLDW_QUALIFICATION_PROFILE+x}" = x ]; _tldw_profile=$?; '
            '[ "${OPENAI_API_KEY+x}" = x ]; _tldw_repop=$?; '
            f"printf '{marker}%s,%s,%s\\n' "
            '"$_tldw_cmd" "$_tldw_profile" "$_tldw_repop"; exit\n'
        ).encode("ascii")
        captured, found = _read_pty(process, master, nonce, command)
    finally:
        if slave >= 0:
            os.close(slave)
        if process is not None:
            _stop_posix_process(process)
        os.close(master)
    match = _single_result_match(captured, nonce, windows=False)
    command_discovery = bool(match and match.group(1) == b"0")
    profile_discovery = bool(match and match.group(2) == b"0")
    return {
        "startup_completed": found and match is not None,
        "command_discovery": command_discovery,
        "profile_contract_applicable": True,
        "profile_marker_present": profile_discovery,
        "sensitive_key_repopulated_by_profile": bool(match and match.group(3) == b"0"),
        "module_discovery": command_discovery and profile_discovery,
        "default_module_discovery": command_discovery,
        "profile_extended_module_discovery": profile_discovery,
        "captured_byte_count": len(captured),
        "capture_within_bound": len(captured) < MAX_CAPTURE_BYTES,
    }


def _synthetic_profile(
    shell: Path, environment: dict[str, str]
) -> dict[str, bool | int]:
    with tempfile.TemporaryDirectory(prefix="tldw-terminal-profile-") as raw_home:
        home = Path(raw_home)
        body = "export TLDW_QUALIFICATION_PROFILE=1\nexport OPENAI_API_KEY=profile-restored\n"
        for name in (".profile", ".bash_profile", ".bashrc", ".zprofile", ".zshrc"):
            (home / name).write_text(body, encoding="utf-8")
        synthetic = dict(environment)
        synthetic["HOME"] = str(home)
        return _run_posix_shell(shell, synthetic)


def _run_shell_process(
    argv: Sequence[str],
    *,
    env: dict[str, str],
    input_bytes: bytes,
    result_nonce: str,
    bootstrap_setup: WindowsBootstrapSetup,
) -> BoundedResult:
    _result_marker(result_nonce)
    return run_bounded(
        argv,
        cwd=Path.cwd(),
        env=env,
        input_bytes=input_bytes,
        timeout_seconds=SHELL_TIMEOUT_SECONDS,
        output_limit=MAX_CAPTURE_BYTES // 2,
        operation=f"environment-{Path(argv[0]).stem}-startup",
        windows_profile_setup=bootstrap_setup,
    )


def _run_windows_shell(
    requested: str,
    shell: Path,
    environment: dict[str, str],
) -> dict[str, bool | int]:
    nonce = uuid.uuid4().hex
    marker = _result_marker(nonce)
    if requested == "powershell":
        module_body = (
            "function Get-TldwTask22512Marker { $true }\n"
            "Export-ModuleMember -Function Get-TldwTask22512Marker\n"
        )
        profile_body = (
            "$env:TLDW_QUALIFICATION_PROFILE = '1'\n"
            "$env:OPENAI_API_KEY = 'fixture-restored'\n"
            "$moduleRoot = Join-Path $HOME 'Modules'\n"
            "$env:PSModulePath = $env:PSModulePath + "
            "[IO.Path]::PathSeparator + $moduleRoot\n"
            "Import-Module TldwTask22512Probe -Force\n"
        )
        profile_files = [
            ("Modules/TldwTask22512Probe/TldwTask22512Probe.psm1", module_body)
        ]
        for profile_dir_name in ("WindowsPowerShell", "PowerShell"):
            for profile_name in ("profile.ps1", "Microsoft.PowerShell_profile.ps1"):
                profile_files.append(
                    (f"Documents/{profile_dir_name}/{profile_name}", profile_body)
                )
        bootstrap_setup = WindowsBootstrapSetup(profile_files=tuple(profile_files))
        script = (
            "$defaultModule=[bool](Get-Module -ListAvailable "
            "Microsoft.PowerShell.Management | Select-Object -First 1)\n"
            "$a=[int](-not ([bool](Get-Command cmd.exe "
            "-ErrorAction SilentlyContinue) -and $defaultModule))\n"
            "$b=[int](-not [bool]$env:TLDW_QUALIFICATION_PROFILE)\n"
            "$c=[int](-not [bool]$env:OPENAI_API_KEY)\n"
            "$d=[int](-not [bool](Get-Command Get-TldwTask22512Marker "
            "-ErrorAction SilentlyContinue))\n"
            f"[Console]::Out.WriteLine('{marker}' + $a + ',' + $b + ',' + "
            "$c + ',' + $d)\n"
            "exit\n"
        )
        argv = [str(shell), "-NoLogo"]
        completed = _run_shell_process(
            argv,
            env=dict(environment),
            input_bytes=script.encode("utf-8"),
            result_nonce=nonce,
            bootstrap_setup=bootstrap_setup,
        )
    else:
        autorun = (
            "@set TLDW_QUALIFICATION_PROFILE=1"
            "&@set OPENAI_API_KEY=fixture-restored"
            "&@set TLDW_TASK22512_AUTORUN=1"
        )
        script = (
            "where cmd.exe >nul 2>&1\r\n"
            'set "_tldw_a=%errorlevel%"\r\n'
            "if defined TLDW_QUALIFICATION_PROFILE "
            '(set "_tldw_b=0") else (set "_tldw_b=1")\r\n'
            "if defined OPENAI_API_KEY "
            '(set "_tldw_c=0") else (set "_tldw_c=1")\r\n'
            "if defined TLDW_TASK22512_AUTORUN "
            '(set "_tldw_d=0") else (set "_tldw_d=1")\r\n'
            f"echo {marker}%_tldw_a%,%_tldw_b%,%_tldw_c%,%_tldw_d%\r\n"
            "exit\r\n"
        )
        argv = [str(shell), "/Q"]
        bootstrap_setup = WindowsBootstrapSetup(
            registry_values=(
                (r"Software\Microsoft\Command Processor", "AutoRun", autorun),
            )
        )
        completed = _run_shell_process(
            argv,
            env=dict(environment),
            input_bytes=script.encode("ascii"),
            result_nonce=nonce,
            bootstrap_setup=bootstrap_setup,
        )
    captured = completed.stdout + completed.stderr
    match = _single_result_match(captured, nonce, windows=True)
    timed_out = bool(getattr(completed, "timed_out", False))
    output_overflowed = bool(getattr(completed, "overflowed", False))
    command_discovery = bool(match and match.group(1) == b"0")
    profile_marker_present = bool(match and match.group(2) == b"0")
    default_module_discovery = command_discovery
    profile_extended_module_discovery = bool(match and match.group(4) == b"0")
    return {
        "startup_completed": not timed_out
        and not output_overflowed
        and completed.returncode == 0
        and match is not None,
        "command_discovery": command_discovery,
        "profile_contract_applicable": True,
        "profile_marker_present": profile_marker_present,
        "sensitive_key_repopulated_by_profile": bool(match and match.group(3) == b"0"),
        "module_discovery": (
            default_module_discovery and profile_extended_module_discovery
        ),
        "default_module_discovery": default_module_discovery,
        "profile_extended_module_discovery": profile_extended_module_discovery,
        "captured_byte_count": len(captured),
        "capture_within_bound": (
            not timed_out
            and not output_overflowed
            and len(captured) < MAX_CAPTURE_BYTES
        ),
        "output_overflowed": output_overflowed,
    }


def _windows_shell_passed(requested: str, result: dict[str, bool | int]) -> bool:
    """Require normal startup and both default and profile-extended discovery."""
    del requested
    return (
        bool(result["startup_completed"])
        and bool(result["command_discovery"])
        and bool(result["capture_within_bound"])
        and not bool(result.get("output_overflowed", False))
        and result["profile_contract_applicable"] is True
        and bool(result["profile_marker_present"])
        and bool(result["sensitive_key_repopulated_by_profile"])
        and bool(result["module_discovery"])
        and bool(result["default_module_discovery"])
        and bool(result["profile_extended_module_discovery"])
    )


def _row_context(json_out: Path) -> tuple[str, dict[str, object]]:
    manifest = json_out.parent / "artifacts.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(
            "sibling artifacts.json is required for row identity"
        ) from exc
    row_id = payload.get("row_id") if isinstance(payload, dict) else None
    if not isinstance(row_id, str):
        raise QualificationError("artifact manifest row identity is invalid")
    runtime = payload.get("runtime", {"kind": "host"})
    if not isinstance(runtime, dict):
        raise QualificationError("artifact manifest runtime identity is invalid")
    return row_id, runtime


def probe(shell_name: str, json_out: Path, *, replace: bool) -> bool:
    """Run one named shell probe and write only booleans, counts, and platform facts."""
    started_at = utc_now()
    started = time.monotonic()
    row_id, runtime = _row_context(json_out)
    environment = scrubbed_environment()
    selection = _resolve_shell(shell_name, environment, windows=os.name == "nt")
    shell = selection.path
    mandatory = _shell_is_mandatory(
        shell_name,
        available=shell is not None,
        windows=os.name == "nt",
    )
    row_name = (
        f"environment-{shell_name}-shell"
        if shell_name == "default"
        else f"environment-{shell_name}"
    )
    if shell is None:
        status = "FAIL" if mandatory else "UNAVAILABLE"
        reason_category = (
            "mandatory-shell-unavailable" if mandatory else "optional-shell-unavailable"
        )
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "row_id": row_id,
            "probe": f"environment-{shell_name}",
            "status": status,
            "mandatory": mandatory,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "command": command_facts(),
            "platform": platform_facts(),
            "measurements": memory_facts(),
            "runtime": runtime,
            "initial_keys": sorted(environment),
            "reason_category": reason_category,
            "rows": [
                {
                    "id": row_name,
                    "mandatory": mandatory,
                    "status": status,
                    "available": False,
                    "reason_category": reason_category,
                }
            ],
        }
        if shell_name == "default" and os.name == "nt":
            payload["selected_shell_family"] = "unavailable"
        write_probe_result(json_out, payload, replace=replace)
        return not mandatory
    actual = (
        _run_windows_shell(selection.family, shell, environment)
        if os.name == "nt"
        else _run_posix_shell(shell, environment)
    )
    synthetic = actual if os.name == "nt" else _synthetic_profile(shell, environment)
    if os.name == "nt":
        passed = _windows_shell_passed(shell_name, actual)
    else:
        passed = (
            bool(actual["startup_completed"])
            and bool(actual["command_discovery"])
            and bool(actual["capture_within_bound"])
            and bool(synthetic["startup_completed"])
            and bool(synthetic["profile_marker_present"])
            and bool(synthetic["sensitive_key_repopulated_by_profile"])
        )
    passed = passed and not any(_is_sensitive(key) for key in environment)
    status = "PASS" if passed else "FAIL"
    profile_candidates = (
        (".profile", ".bash_profile", ".bashrc", ".zprofile", ".zshrc")
        if os.name != "nt"
        else ()
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "row_id": row_id,
        "probe": f"environment-{shell_name}",
        "status": status,
        "mandatory": mandatory,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "command": command_facts(),
        "platform": platform_facts(),
        "measurements": memory_facts(),
        "runtime": runtime,
        "initial_keys": sorted(environment),
        "initial_key_count": len(environment),
        "sensitive_initial_key_count": sum(_is_sensitive(key) for key in environment),
        "account_profile_candidate_count": sum(
            (Path(environment.get("HOME", "")) / name).is_file()
            for name in profile_candidates
        ),
        "actual_startup": actual,
        "synthetic_profile": synthetic,
        "rows": [
            {
                "id": row_name,
                "mandatory": mandatory,
                "status": status,
                "available": True,
                "initial_key_count": len(environment),
                "sensitive_initial_key_count": sum(
                    _is_sensitive(key) for key in environment
                ),
            }
        ],
    }
    if shell_name == "default" and os.name == "nt":
        payload["selected_shell_family"] = selection.family
    write_probe_result(json_out, payload, replace=replace)
    return passed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell", choices=SHELLS, required=True)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return 0 if probe(args.shell, args.json_out, replace=args.replace) else 1
    except (QualificationError, OSError, subprocess.SubprocessError) as exc:
        print(
            f"environment qualification failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if exc.__cause__ is not None:
            print(
                "environment qualification cause: "
                f"{type(exc.__cause__).__name__}: {exc.__cause__}",
                file=sys.stderr,
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
