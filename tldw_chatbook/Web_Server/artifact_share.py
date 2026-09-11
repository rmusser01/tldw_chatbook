# artifact_share.py
"""App-side orchestration for artifact share sessions (no Textual imports)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from loguru import logger

from . import is_web_server_available
from .artifact_share_manifest import (
    ArtifactShareError,
    build_share_auth,
    share_root_dir,
    stage_share,
    sweep_stale_shares,
)

_CHILD_READY_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class ShareStatus:
    """Snapshot of one running share session (no credentials, ever)."""

    share_name: str
    urls: tuple[str, ...]
    artifact_count: int
    share_dir: Path


class ArtifactShareController:
    """Owns at most one running share: staging, child process, teardown."""

    def __init__(
        self, *, status_callback: Callable[[ShareStatus | None], None] | None = None
    ) -> None:
        """Prepare an idle controller.

        Args:
            status_callback: Optional sink invoked (from the starting/stopping
                thread) with the new ``ShareStatus``, or None on stop.
        """
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._share_dir: Path | None = None
        self._status: ShareStatus | None = None
        self._status_callback = status_callback

    @property
    def status(self) -> ShareStatus | None:
        """The running share's status, or None when no share is active."""
        with self._lock:
            return self._status

    def startup_sweep(self) -> list[Path]:
        """Remove share directories left behind by dead server processes.

        Returns:
            The removed share directories.
        """
        removed = sweep_stale_shares()
        if removed:
            logger.info(f"Artifact share sweep removed {len(removed)} stale share dir(s)")
        return removed

    def start_share(
        self,
        *,
        records: list[dict[str, Any]],
        share_name: str,
        username: str | None = None,
        password: str | None = None,
        bind: str = "127.0.0.1",
        port: int = 0,
    ) -> ShareStatus:
        """Stage the selected artifacts and start the share server child.

        Args:
            records: Artifact records to stage (see ``stage_share``).
            share_name: Human-facing share title.
            username: Optional shared-login username (with ``password``).
            password: Optional shared-login password.
            bind: Bind host; "0.0.0.0" exposes on all interfaces.
            port: Bind port (0 = ephemeral).

        Returns:
            The running share's ``ShareStatus``.

        Raises:
            ArtifactShareError: Web extras missing, no records, an invalid
                username, staging failure, or the child failed/was too slow
                to report readiness (staging is cleaned in every case).
        """
        if not is_web_server_available():
            raise ArtifactShareError(
                "Web sharing requires extra packages: pip install tldw_chatbook[web]"
            )
        if not records:
            raise ArtifactShareError("No artifacts selected to share.")
        if username and ":" in username:
            # Defensive twin of the dialog's check (Qodo #14): Basic auth
            # splits on the first ':', so a colon username can never match.
            raise ArtifactShareError("Username cannot contain ':'.")
        with self._lock:
            self.stop_share()  # single active share; starting a new one stops the old
            auth = (
                build_share_auth(username, password)
                if username and password
                else None
            )
            manifest = stage_share(
                records, share_name=share_name, auth=auth, share_root=share_root_dir()
            )
            self._share_dir = share_root_dir() / manifest.share_id
            manifest_path = self._share_dir / "manifest.json"
            # The child's stdout/stderr go to an anonymous temp file, never a
            # pipe: nothing in steady state drains a pipe, and a chatty child
            # (loguru/aiohttp tracebacks) would fill it and wedge the server.
            # A file can absorb unlimited output; we read it back only on the
            # early-death path for diagnostics. The parent's fd closes when
            # the readiness phase ends; the child keeps its own descriptor
            # until it exits, so writes never block on either side.
            with tempfile.TemporaryFile() as child_log:
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "tldw_chatbook.Web_Server.artifact_share_server",
                            str(manifest_path),
                            "--host",
                            bind,
                            "--port",
                            str(port),
                        ],
                        start_new_session=True,
                        cwd=str(Path(__file__).resolve().parents[2]),
                        stdout=child_log,
                        stderr=subprocess.STDOUT,
                    )
                except OSError as exc:
                    # Qodo #9: a failed spawn must not strand the staged
                    # directory; fail with one error and clean up.
                    self._cleanup_staging()
                    raise ArtifactShareError(
                        f"Could not start share server: {exc}"
                    ) from exc
                self._process = process
                status_path = self._share_dir / "status.json"
                deadline = time.monotonic() + _CHILD_READY_TIMEOUT_SECONDS
                url: str | None = None
                while time.monotonic() < deadline:
                    if status_path.is_file():
                        try:
                            url = str(
                                json.loads(status_path.read_text(encoding="utf-8"))["url"]
                            )
                            break
                        except (ValueError, KeyError, OSError):
                            pass
                    if process.poll() is not None:
                        output = ""
                        try:
                            child_log.seek(0)
                            output = child_log.read().decode(errors="replace")
                        except OSError:
                            pass
                        self._cleanup_staging()
                        raise ArtifactShareError(
                            f"Artifact share server failed to start: {output[-500:]}"
                        )
                    time.sleep(0.1)
            if url is None:
                self.stop_share()
                raise ArtifactShareError(
                    "Artifact share server did not report readiness in time."
                )
            bound = urlparse(url)
            self._status = ShareStatus(
                share_name=manifest.share_name,
                urls=tuple(compute_display_urls(bind, bound.port or port)),
                artifact_count=len(manifest.artifacts),
                share_dir=self._share_dir,
            )
            self._emit_status()
            return self._status

    def stop_share(self) -> None:
        """Stop the running share (terminate child, remove staging), if any."""
        with self._lock:
            process, self._process = self._process, None
            if process is not None:
                self._terminate_child(process)
            self._cleanup_staging()
            if self._status is not None:
                self._status = None
                self._emit_status()

    def _terminate_child(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - pathological
            logger.error(f"Artifact share child {process.pid} refused to die")

    def _cleanup_staging(self) -> None:
        if self._share_dir is not None:
            shutil.rmtree(self._share_dir, ignore_errors=True)
            self._share_dir = None

    def _emit_status(self) -> None:
        """Invoke the status callback, logging failures with share context.

        Qodo #7: the failure log names the share (never credentials) so a
        broken UI callback can be tied to the session it belongs to.
        """
        callback = self._status_callback
        if callback is None:
            return
        share_name = self._status.share_name if self._status is not None else "<stopped>"
        try:
            callback(self._status)
        except Exception:  # noqa: BLE001 - UI callback must not break sharing
            logger.exception(
                f"Artifact share status callback failed (share='{share_name}')"
            )


def compute_display_urls(bind: str, port: int) -> list[str]:
    """Human-usable URLs for the bound port: loopback always, LAN route when wide."""
    urls = [f"http://127.0.0.1:{port}"]
    if bind not in ("127.0.0.1", "localhost", "::1"):
        lan_ip = _primary_route_ip()
        if lan_ip:
            urls.append(f"http://{lan_ip}:{port}")
    return urls


def _primary_route_ip() -> str | None:
    """Best-effort LAN address for display URLs; never raises, sends nothing.

    Hostname resolution is tried first because it needs no socket egress at
    all; the UDP route probe (which picks the default-route interface without
    sending a packet) stays as the fallback for hosts whose name maps only to
    loopback. Resolution order matters under the test suite's network guard
    (Tests/network_guard.py): a UDP connect to a non-loopback target is
    blocked and recorded even when swallowed here, which would fail the
    loopback-marked tests at teardown.
    """
    try:
        for info in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM
        ):
            candidate = info[4][0]
            if candidate and not candidate.startswith("127."):
                return candidate
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            candidate = probe.getsockname()[0]
        if candidate and not candidate.startswith("127."):
            return candidate
    except OSError:
        pass
    return None
