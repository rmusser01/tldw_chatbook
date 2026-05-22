"""ACP-owned runtime process lifecycle management."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .runtime_session import ACPRuntimeSessionState


class ACPRuntimeProcessStatus(StrEnum):
    """Runtime lifecycle states surfaced across ACP, Console, and Home."""

    NOT_CONFIGURED = "not_configured"
    CONFIGURED = "configured"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"


def _clean_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _coerce_args(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part for part in value.split() if part)
    if isinstance(value, Sequence):
        return tuple(str(part) for part in value)
    return ()


@dataclass(frozen=True)
class ACPRuntimeProcessConfig:
    """Configuration for a shell-free ACP runtime process launch."""

    command: str = ""
    args: tuple[str, ...] = ()
    cwd: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    runtime_id: str = "local-acp-runtime"
    runtime_label: str = "Local ACP Runtime"
    runtime_version: str = ""
    startup_timeout_seconds: float = 2.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ACPRuntimeProcessConfig":
        """Build config from the `[acp.runtime]` mapping."""
        if not isinstance(value, Mapping):
            return cls()
        return cls(
            command=_clean_text(value.get("command")),
            args=_coerce_args(value.get("args")),
            cwd=_clean_text(value.get("cwd")),
            env=dict(value.get("env") or {}) if isinstance(value.get("env"), Mapping) else {},
            runtime_id=_clean_text(value.get("runtime_id"), "local-acp-runtime"),
            runtime_label=_clean_text(value.get("runtime_label"), "Local ACP Runtime"),
            runtime_version=_clean_text(value.get("runtime_version")),
            startup_timeout_seconds=float(value.get("startup_timeout_seconds") or 2.0),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.command)

    @property
    def disabled_reason(self) -> str:
        if self.is_configured:
            return "ACP runtime is configured."
        return "Configure an ACP runtime command in ACP before launch."

    @property
    def command_display(self) -> str:
        if not self.command:
            return "not configured"
        parts = (self.command, *self.args)
        return " ".join(parts)


@dataclass(frozen=True)
class ACPRuntimeProcessResult:
    """Result from a process lifecycle operation."""

    status: ACPRuntimeProcessStatus
    recovery: str
    session_state: ACPRuntimeSessionState = field(default_factory=ACPRuntimeSessionState)
    return_code: int | None = None


class ACPRuntimeProcessManager:
    """Start, stop, and summarize one ACP-compatible local runtime process."""

    def __init__(self, *, config: ACPRuntimeProcessConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._status = (
            ACPRuntimeProcessStatus.CONFIGURED
            if config.is_configured
            else ACPRuntimeProcessStatus.NOT_CONFIGURED
        )
        self._last_recovery = config.disabled_reason
        self._session_state = ACPRuntimeSessionState(
            runtime_id=config.runtime_id if config.is_configured else "",
            runtime_label=config.runtime_label if config.is_configured else "",
            runtime_version=config.runtime_version if config.is_configured else "",
        )

    @classmethod
    def from_app_config(cls, app_config: Mapping[str, Any] | None) -> "ACPRuntimeProcessManager":
        """Build a manager from `app_config['acp']['runtime']` if present."""
        acp_config = app_config.get("acp") if isinstance(app_config, Mapping) else {}
        if not acp_config and isinstance(app_config, Mapping):
            raw_config = app_config.get("COMPREHENSIVE_CONFIG_RAW")
            if isinstance(raw_config, Mapping):
                acp_config = raw_config.get("acp")
        runtime_config = acp_config.get("runtime") if isinstance(acp_config, Mapping) else {}
        return cls(config=ACPRuntimeProcessConfig.from_mapping(runtime_config))

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable state snapshot for UI surfaces."""
        self._refresh_process_state()
        return {
            "status": self._status.value,
            "runtime_id": self.config.runtime_id if self.config.is_configured else "",
            "runtime_label": self.config.runtime_label if self.config.is_configured else "",
            "runtime_version": self.config.runtime_version,
            "session_id": self._session_state.session_id,
            "session_title": self._session_state.session_title,
            "session_status": self._session_state.session_status,
            "session_payload": dict(self._session_state.session_payload),
            "launch_available": self.config.is_configured
            and self._status
            in {
                ACPRuntimeProcessStatus.CONFIGURED,
                ACPRuntimeProcessStatus.FAILED,
                ACPRuntimeProcessStatus.STOPPED,
            },
            "stop_available": self._status == ACPRuntimeProcessStatus.RUNNING,
            "recovery": self._last_recovery,
            "command_display": self.config.command_display,
        }

    def session_state(self) -> ACPRuntimeSessionState:
        """Return the current runtime/session state."""
        self._refresh_process_state()
        return self._session_state

    def start_session(self, *, title: str = "ACP session") -> ACPRuntimeProcessResult:
        """Start the runtime and create a Console-followable session payload."""
        if not self.config.is_configured:
            self._status = ACPRuntimeProcessStatus.NOT_CONFIGURED
            self._last_recovery = self.config.disabled_reason
            self._session_state = ACPRuntimeSessionState()
            return ACPRuntimeProcessResult(self._status, self._last_recovery)

        self.stop() if self._process and self._process.poll() is None else None
        self._status = ACPRuntimeProcessStatus.STARTING
        self._last_recovery = "ACP runtime is starting."
        try:
            env = os.environ.copy()
            env.update({str(key): str(value) for key, value in self.config.env.items()})
            cwd = str(Path(self.config.cwd).expanduser()) if self.config.cwd else None
            self._process = subprocess.Popen(
                [self.config.command, *self.config.args],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                shell=False,
            )
        except OSError as exc:
            self._status = ACPRuntimeProcessStatus.FAILED
            self._last_recovery = f"ACP runtime could not start: {exc}"
            self._session_state = self._base_session_state()
            return ACPRuntimeProcessResult(self._status, self._last_recovery, self._session_state)

        deadline = time.monotonic() + max(0.05, self.config.startup_timeout_seconds)
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                self._status = ACPRuntimeProcessStatus.FAILED
                self._last_recovery = (
                    f"ACP runtime exited before it became ready with code {return_code}."
                )
                self._session_state = self._base_session_state()
                return ACPRuntimeProcessResult(
                    self._status,
                    self._last_recovery,
                    self._session_state,
                    return_code=return_code,
                )
            if time.monotonic() + 0.05 >= deadline:
                break
            time.sleep(0.01)

        self._status = ACPRuntimeProcessStatus.RUNNING
        session_id = uuid.uuid4().hex
        pid = int(self._process.pid)
        self._last_recovery = "ACP runtime is running."
        self._session_state = ACPRuntimeSessionState(
            runtime_id=self.config.runtime_id,
            runtime_label=self.config.runtime_label,
            runtime_version=self.config.runtime_version,
            session_id=session_id,
            session_title=_clean_text(title, "ACP session"),
            session_status="running",
            session_payload={
                "pid": pid,
                "command": self.config.command,
                "args": list(self.config.args),
                "cwd": self.config.cwd,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        return ACPRuntimeProcessResult(self._status, self._last_recovery, self._session_state)

    def stop(self) -> ACPRuntimeProcessResult:
        """Stop the runtime process if it is running."""
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self._status = (
            ACPRuntimeProcessStatus.STOPPED
            if self.config.is_configured
            else ACPRuntimeProcessStatus.NOT_CONFIGURED
        )
        self._last_recovery = (
            "ACP runtime stopped."
            if self.config.is_configured
            else self.config.disabled_reason
        )
        self._session_state = self._base_session_state()
        return ACPRuntimeProcessResult(self._status, self._last_recovery, self._session_state)

    def _refresh_process_state(self) -> None:
        if self._status != ACPRuntimeProcessStatus.RUNNING or self._process is None:
            return
        return_code = self._process.poll()
        if return_code is None:
            return
        self._status = ACPRuntimeProcessStatus.FAILED
        self._last_recovery = f"ACP runtime exited with code {return_code}."
        self._session_state = self._base_session_state()

    def _base_session_state(self) -> ACPRuntimeSessionState:
        return ACPRuntimeSessionState(
            runtime_id=self.config.runtime_id if self.config.is_configured else "",
            runtime_label=self.config.runtime_label if self.config.is_configured else "",
            runtime_version=self.config.runtime_version if self.config.is_configured else "",
        )
