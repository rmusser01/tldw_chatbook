"""Fresh-process admission gate for POSIX controlling-terminal launch."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import termios
from typing import Any

import psutil
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
)


_MAX_CONTROL_BYTES = 64 * 1024
_REFUSED_EXIT = 125
_FAILED_EXIT = 126


class _LauncherConfig(BaseModel):
    """Strict launch values accepted from the parent control pipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    executable: StrictStr = Field(min_length=1)
    argv: list[StrictStr] = Field(min_length=1, max_length=64, strict=True)
    cwd: StrictStr = Field(min_length=1)
    environment: dict[StrictStr, StrictStr] = Field(max_length=128, strict=True)
    token: StrictStr = Field(min_length=1, max_length=1024)

    @field_validator("executable")
    @classmethod
    def _validate_executable(cls, value: str) -> str:
        if not os.path.isabs(value) or "\0" in value:
            raise ValueError("invalid executable")
        return value

    @field_validator("argv")
    @classmethod
    def _validate_argv(cls, value: list[str]) -> list[str]:
        if any(not item or "\0" in item for item in value):
            raise ValueError("invalid argv")
        return value

    @field_validator("cwd")
    @classmethod
    def _validate_cwd(cls, value: str) -> str:
        if "\0" in value or not os.path.isabs(value) or not Path(value).is_dir():
            raise ValueError("invalid cwd")
        return value

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not key or "=" in key or "\0" in key or "\0" in item
            for key, item in value.items()
        ):
            raise ValueError("invalid environment")
        return value

    @field_validator("token")
    @classmethod
    def _validate_token(cls, value: str) -> str:
        if "\0" in value:
            raise ValueError("invalid token")
        return value


def _read_control(fd: int) -> dict[str, Any]:
    """Read one bounded JSON control object from a trusted parent pipe."""
    payload = bytearray()
    while len(payload) <= _MAX_CONTROL_BYTES:
        chunk = os.read(fd, min(4096, _MAX_CONTROL_BYTES + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
        if b"\n" in chunk:
            break
    if len(payload) > _MAX_CONTROL_BYTES:
        raise ValueError("launcher control is invalid")
    line, separator, remainder = bytes(payload).partition(b"\n")
    if not separator or remainder:
        raise ValueError("launcher control is invalid")
    value = json.loads(line)
    if type(value) is not dict:
        raise ValueError("launcher control is invalid")
    return value


def _validated_config(
    value: dict[str, Any],
) -> _LauncherConfig:
    """Validate one untrusted launcher control object with strict field types."""
    try:
        return _LauncherConfig.model_validate(value, strict=True)
    except (TypeError, ValueError, ValidationError):
        raise ValueError("launcher config is invalid") from None


def _write_json(fd: int, value: dict[str, object]) -> None:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _set_close_on_exec(fd: int) -> None:
    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
    fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)


def _close_unrelated_fds(keep_fd: int) -> None:
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit == resource.RLIM_INFINITY:
        soft_limit = 1_048_576
    maximum = max(3, int(soft_limit))
    if keep_fd > 3:
        os.closerange(3, keep_fd)
    os.closerange(max(3, keep_fd + 1), maximum)


def _run(
    *,
    slave_fd: int,
    config_fd: int,
    admission_fd: int,
    report_fd: int,
    exec_status_fd: int,
) -> int:
    """Enter a new session, await admission, and exec the configured shell."""
    config = _validated_config(_read_control(config_fd))
    os.close(config_fd)

    os.setsid()
    pid = os.getpid()
    _write_json(
        report_fd,
        {
            "birth_time": psutil.Process(pid).create_time(),
            "pgid": os.getpgrp(),
            "pid": pid,
            "sid": os.getsid(0),
        },
    )
    os.close(report_fd)

    decision = _read_control(admission_fd)
    os.close(admission_fd)
    if decision != {"admitted": True, "token": config.token}:
        os.close(slave_fd)
        os.close(exec_status_fd)
        return _REFUSED_EXIT

    fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
    os.tcsetpgrp(slave_fd, os.getpgrp())
    for target in (0, 1, 2):
        if slave_fd != target:
            os.dup2(slave_fd, target)
    if slave_fd > 2:
        os.close(slave_fd)
    os.chdir(config.cwd)
    _set_close_on_exec(exec_status_fd)
    _close_unrelated_fds(exec_status_fd)
    os.execve(config.executable, config.argv, config.environment)
    return _FAILED_EXIT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--slave-fd", required=True, type=int)
    parser.add_argument("--config-fd", required=True, type=int)
    parser.add_argument("--admission-fd", required=True, type=int)
    parser.add_argument("--report-fd", required=True, type=int)
    parser.add_argument("--exec-status-fd", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    """Run the content-free launcher protocol.

    Returns:
        The bounded launcher process exit code.
    """
    arguments = _parse_args()
    try:
        return _run(
            slave_fd=arguments.slave_fd,
            config_fd=arguments.config_fd,
            admission_fd=arguments.admission_fd,
            report_fd=arguments.report_fd,
            exec_status_fd=arguments.exec_status_fd,
        )
    except BaseException:
        try:
            os.write(arguments.exec_status_fd, b"1")
        except OSError:
            pass
        return _FAILED_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
