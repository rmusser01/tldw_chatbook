"""Contain one local worker generation and all of its descendants."""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Utils.windows_job import WindowsJobApi as _WindowsJobApi


class ProcessContainmentError(RuntimeError):
    """Raised when a worker cannot be safely admitted to containment."""


@dataclass(frozen=True, slots=True)
class WorkerContainmentIdentity:
    """Worker-reported identity needed for parent-side containment."""

    pid: int
    process_group_id: int | None

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("pid must be a positive integer")
        if self.process_group_id is not None and (
            type(self.process_group_id) is not int or self.process_group_id <= 0
        ):
            raise ValueError("process_group_id must be a positive integer")


def enter_worker_containment() -> WorkerContainmentIdentity:
    """Enter the worker-side containment boundary before admission."""

    if os.name == "posix":
        os.setsid()
        return WorkerContainmentIdentity(
            pid=os.getpid(),
            process_group_id=os.getpgrp(),
        )
    return WorkerContainmentIdentity(pid=os.getpid(), process_group_id=None)


class ExecutorProcessTree:
    """Parent-owned containment lifecycle for one worker generation."""

    def __init__(
        self,
        process: Any,
        admission_event: Any,
        identity: WorkerContainmentIdentity,
        *,
        platform_name: str | None = None,
        windows_api: Any | None = None,
    ) -> None:
        if type(identity) is not WorkerContainmentIdentity:
            raise TypeError("identity must be a WorkerContainmentIdentity")
        if getattr(process, "pid", None) != identity.pid:
            raise ValueError("worker identity does not match process")
        self._process = process
        self._admission_event = admission_event
        self._identity = identity
        self._platform_name = platform_name or os.name
        if self._platform_name not in {"posix", "nt"}:
            raise ValueError("unsupported process containment platform")
        self._windows_api = windows_api
        self._lock = threading.RLock()
        self._job_handle = 0
        self._admitted = False
        self._quarantined = False
        self._closed = False

    @property
    def admitted(self) -> bool:
        """Return whether this generation passed parent admission."""

        with self._lock:
            return self._admitted

    @property
    def quarantined(self) -> bool:
        """Return whether worker-tree death could not be proven."""

        with self._lock:
            return self._quarantined

    def admit(self) -> None:
        """Establish platform containment before releasing the worker."""

        with self._lock:
            self._admit_locked()

    def _admit_locked(self) -> None:
        if self._closed or self._quarantined:
            raise ProcessContainmentError("worker containment is unavailable")
        if self._admitted:
            return
        if not self._process.is_alive():
            raise ProcessContainmentError("worker exited before containment admission")
        try:
            if self._platform_name == "nt":
                api = self._windows_api or _WindowsJobApi()
                self._windows_api = api
                self._job_handle = api.create_kill_on_close_job()
                api.assign_process(self._job_handle, self._identity.pid)
            elif self._identity.process_group_id != self._identity.pid:
                raise ProcessContainmentError(
                    "POSIX worker did not establish its own process group"
                )
            self._admission_event.set()
            self._admitted = True
        except BaseException as error:
            self._admitted = False
            self._terminate_unadmitted()
            self._close_job_handle()
            self._quarantined = self._process.is_alive()
            if isinstance(error, ProcessContainmentError):
                raise
            raise ProcessContainmentError(
                "worker process-tree admission failed"
            ) from error

    def terminate_tree(
        self,
        *,
        term_timeout: float = 2.0,
        kill_timeout: float = 2.0,
    ) -> bool:
        """Terminate the contained tree and return only after proven death."""

        with self._lock:
            if self._closed:
                if self._job_handle:
                    try:
                        self._close_job_handle()
                    except OSError:
                        self._quarantined = True
                return not self._quarantined
            self._admitted = False
            self._closed = True
            try:
                return self._terminate_tree_locked(
                    term_timeout=term_timeout,
                    kill_timeout=kill_timeout,
                )
            except BaseException:
                self._quarantined = True
                raise

    def _terminate_tree_locked(
        self,
        *,
        term_timeout: float,
        kill_timeout: float,
    ) -> bool:
        if self._platform_name == "posix":
            return self._terminate_posix_group(
                term_timeout=term_timeout,
                kill_timeout=kill_timeout,
            )

        job_proven_dead = not self._job_handle
        if self._job_handle:
            try:
                self._windows_api.terminate_job(self._job_handle)
                job_proven_dead = self._windows_api.wait_for_job_empty(
                    self._job_handle,
                    max(0.0, term_timeout),
                )
            except OSError:
                job_proven_dead = False
        if self._process.is_alive() and not self._job_handle:
            self._process.terminate()
        self._process.join(max(0.0, term_timeout))

        if self._process.is_alive() or not job_proven_dead:
            if self._job_handle:
                try:
                    self._windows_api.terminate_job(self._job_handle)
                    job_proven_dead = self._windows_api.wait_for_job_empty(
                        self._job_handle,
                        max(0.0, kill_timeout),
                    )
                except OSError:
                    job_proven_dead = False
            elif self._process.is_alive():
                try:
                    self._process.kill()
                except OSError:
                    pass
            self._process.join(max(0.0, kill_timeout))

        cleanup_proven = not self._process.is_alive() and job_proven_dead
        if not cleanup_proven:
            self._quarantined = True
        try:
            self._close_job_handle()
        except OSError:
            self._quarantined = True
            return False
        return cleanup_proven

    def _terminate_posix_group(
        self,
        *,
        term_timeout: float,
        kill_timeout: float,
    ) -> bool:
        """Signal and prove death of the group independently of its leader."""

        group_id = self._identity.process_group_id
        if group_id is None:
            self._quarantined = True
            return False
        if self._posix_group_exists(group_id):
            try:
                os.killpg(group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self._process.join(max(0.0, term_timeout))
        group_dead = self._wait_for_posix_group_exit(group_id, term_timeout)
        if self._process.is_alive() or not group_dead:
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.join(max(0.0, kill_timeout))
            group_dead = self._wait_for_posix_group_exit(group_id, kill_timeout)
        if self._process.is_alive() or not group_dead:
            self._quarantined = True
            return False
        self._closed = True
        return True

    @staticmethod
    def _posix_group_exists(group_id: int) -> bool:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @classmethod
    def _wait_for_posix_group_exit(cls, group_id: int, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while cls._posix_group_exists(group_id) and time.monotonic() < deadline:
            time.sleep(0.01)
        return not cls._posix_group_exists(group_id)

    def close(self) -> bool:
        """Idempotently close containment, terminating a live worker if needed."""

        with self._lock:
            if self._closed:
                if self._job_handle:
                    try:
                        self._close_job_handle()
                    except OSError:
                        self._quarantined = True
                return not self._quarantined
            return self.terminate_tree()

    def _terminate_unadmitted(self) -> None:
        if not self._process.is_alive():
            return
        self._process.terminate()
        self._process.join(2.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(2.0)

    def _close_job_handle(self) -> None:
        with self._lock:
            if not self._job_handle:
                return
            job_handle = self._job_handle
            self._windows_api.close_handle(job_handle)
            self._job_handle = 0


__all__ = [
    "ExecutorProcessTree",
    "ProcessContainmentError",
    "WorkerContainmentIdentity",
    "enter_worker_containment",
]
