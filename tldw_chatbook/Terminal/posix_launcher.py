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


_MAX_CONTROL_BYTES = 64 * 1024
_REFUSED_EXIT = 125
_FAILED_EXIT = 126


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
) -> tuple[str, list[str], str, dict[str, str], str]:
    executable = value.get("executable")
    argv = value.get("argv")
    cwd = value.get("cwd")
    environment = value.get("environment")
    token = value.get("token")
    if not isinstance(argv, list) or not argv or len(argv) > 64:
        raise ValueError("launcher config is invalid")
    if not all(isinstance(item, str) and item and "\0" not in item for item in argv):
        raise ValueError("launcher config is invalid")
    if executable is None:
        executable = argv[0]
    if (
        not isinstance(executable, str)
        or not os.path.isabs(executable)
        or "\0" in executable
    ):
        raise ValueError("launcher config is invalid")
    if not isinstance(cwd, str) or not os.path.isabs(cwd) or not Path(cwd).is_dir():
        raise ValueError("launcher config is invalid")
    if type(environment) is not dict or len(environment) > 128:
        raise ValueError("launcher config is invalid")
    if not all(
        isinstance(key, str)
        and key
        and "=" not in key
        and "\0" not in key
        and isinstance(item, str)
        and "\0" not in item
        for key, item in environment.items()
    ):
        raise ValueError("launcher config is invalid")
    if not isinstance(token, str) or not token or len(token) > 1024:
        raise ValueError("launcher config is invalid")
    return executable, argv, cwd, environment, token


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
    executable, argv, cwd, environment, expected_token = _validated_config(
        _read_control(config_fd)
    )
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
    if decision != {"admitted": True, "token": expected_token}:
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
    os.chdir(cwd)
    _set_close_on_exec(exec_status_fd)
    _close_unrelated_fds(exec_status_fd)
    os.execve(executable, argv, environment)
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
    """Run the content-free launcher protocol and return a bounded exit code."""
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
