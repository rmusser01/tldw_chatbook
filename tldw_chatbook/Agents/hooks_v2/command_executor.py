"""Bounded argv-only commands with retained launch and terminal custody."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from .budgets import HookTicket
from .models import HookEvent, HookHandler, HookResult
from .ownership import HookProcessOwner
from .validation import decode_result

STDOUT_BYTES = 16 * 1024
STDERR_BYTES = 4 * 1024
REAP_SECONDS = 5.0


class _Capture(asyncio.SubprocessProtocol):
    def __init__(self, on_transport=None) -> None:
        self.on_transport = on_transport
        self.transport = None
        self.done = asyncio.get_running_loop().create_future()
        self.exited = asyncio.get_running_loop().create_future()
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.stdout_overflow = False
        self.stderr_truncated = False

    def connection_made(self, transport) -> None:
        # Save even before subprocess_exec returns: cancellation during spawn
        # cannot erase a handle already delivered by the event loop.
        self.transport = transport
        if self.on_transport:
            self.on_transport()

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        if fd not in (1, 2):
            return
        output, cap = (
            (self.stdout, STDOUT_BYTES) if fd == 1 else (self.stderr, STDERR_BYTES)
        )
        overflow = len(output) + len(data) > cap
        output.extend(data[: max(0, cap - len(output))])
        if fd == 1:
            self.stdout_overflow |= overflow
        else:
            self.stderr_truncated |= overflow

    def process_exited(self) -> None:
        if not self.exited.done():
            self.exited.set_result(None)

    def connection_lost(self, exc) -> None:
        if not self.done.done():
            self.done.set_result(None)


@dataclass
class CommandResult:
    result: HookResult | None = None
    failure: str | None = None
    cleanup_pending: bool = False
    duration: float = 0.0
    stderr_bytes: int = 0
    stderr_truncated: bool = False


@dataclass
class CommandJob:
    id: str
    ticket: HookTicket
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    capture: _Capture | None = None
    token: str | None = None
    task: asyncio.Task | None = None
    cleanup_deadline: float | None = None

    def stop(self) -> None:
        if not self.cancel.is_set():
            self.cleanup_deadline = time.monotonic() + REAP_SECONDS
            self.cancel.set()
        self.signal_child()

    def signal_child(self) -> None:
        if (
            self.cancel.is_set()
            and self.capture
            and self.capture.transport
            and os.name != "nt"
        ):
            try:
                os.killpg(self.capture.transport.get_pid(), signal.SIGKILL)
            except OSError:
                pass


class CommandExecutor:
    """Own tasks/handles until positive terminal evidence and owner settlement.

    Owner callbacks run on workers, never under scheduler/admission locks. Their
    exceptions are reduced to fixed codes: exception messages can contain secrets.
    """

    def __init__(
        self, process_owner: HookProcessOwner, *, launch_guard: Callable | None = None
    ) -> None:
        self.owner = process_owner
        self._launch_guard = launch_guard
        self.records: dict[str, CommandJob] = {}

    def start(
        self,
        handler: HookHandler,
        event: HookEvent,
        ticket: HookTicket,
        payload: bytes,
        authority_check: Callable,
        environment: Callable,
        deadline: float,
    ) -> CommandJob:
        job = CommandJob(uuid4().hex, ticket)
        self.records[job.id] = job
        job.task = asyncio.create_task(
            self._run(
                job, handler, event, payload, authority_check, environment, deadline
            )
        )
        return job

    async def wait(self, job: CommandJob, deadline: float) -> CommandResult:
        try:
            done, _ = await asyncio.wait(
                {job.task}, timeout=max(0, deadline - time.monotonic())
            )
            if done:
                return job.task.result()
            job.stop()
            done, _ = await asyncio.wait(
                {job.task}, timeout=max(0, job.cleanup_deadline - time.monotonic())
            )
            if done:
                value = job.task.result()
                value.result = None
                value.failure = value.failure or "timeout"
                return value
            return CommandResult(failure="timeout", cleanup_pending=True)
        except asyncio.CancelledError:
            job.stop()
            raise

    async def _run(
        self, job, handler, event, payload, authority_check, environment, deadline
    ):
        started = time.monotonic()
        outcome = CommandResult()
        launched = False
        terminal = False
        capture = job.capture = _Capture(job.signal_child)
        try:
            # R47: without whole-tree terminal proof, a Windows launch would
            # permanently consume custody. Refuse before root/owner/env access.
            if sys.platform == "win32":
                terminal = True
                outcome.failure = "unsupported_platform"
                return outcome
            job.token = await asyncio.to_thread(self.owner.reserve_launch, event)
            env = await asyncio.to_thread(environment, handler, event)
            if job.cancel.is_set() or time.monotonic() >= deadline:
                outcome.failure = "cancelled"
                terminal = True  # reserve succeeded, no launch attempted
                return outcome
            if not await asyncio.to_thread(authority_check, handler, event, "launch"):
                outcome.failure = "authority_refused"
                terminal = True
                return outcome
            # Recheck cancellation immediately before the launch transaction.
            if job.cancel.is_set() or time.monotonic() >= deadline:
                outcome.failure = "cancelled"
                terminal = True
                return outcome
            try:
                loop = asyncio.get_running_loop()

                async def launch():
                    return await loop.subprocess_exec(
                        lambda: capture,
                        *handler.argv,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=handler.cwd,
                        env=env,
                        start_new_session=os.name != "nt",
                    )

                if self._launch_guard is None:
                    transport, _ = await launch()
                else:
                    # Same consent transaction as legacy Popen. The worker owns
                    # its thread-affine locks; the retained job owns any late
                    # transport publication through cancellation/teardown.
                    def guarded_launch():
                        with self._launch_guard(handler, event):
                            if job.cancel.is_set() or time.monotonic() >= deadline:
                                raise ValueError("launch cancelled")
                            return asyncio.run_coroutine_threadsafe(
                                launch(), loop
                            ).result()

                    transport, _ = await asyncio.to_thread(guarded_launch)
                launched = True
            except (OSError, ValueError):
                # subprocess_exec has positively reported no child; a transport
                # already delivered still needs the terminal path below.
                terminal = capture.transport is None
                outcome.failure = "launch_failed"
                return outcome
            await asyncio.to_thread(
                self.owner.publish_process,
                job.token,
                {
                    "pid": transport.get_pid(),
                    "process_group": transport.get_pid() if os.name != "nt" else None,
                    "platform": os.name,
                },
            )
            if not job.cancel.is_set():
                stdin = transport.get_pipe_transport(0)
                stdin.write(payload)
                stdin.close()
            cancelled = asyncio.create_task(job.cancel.wait())
            try:
                done, _ = await asyncio.wait(
                    {capture.done, cancelled},
                    timeout=max(0, deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancelled in done or capture.done not in done or job.cancel.is_set():
                    outcome.failure = "cancelled" if job.cancel.is_set() else "timeout"
                    job.stop()
                elif transport.get_returncode() != 0:
                    outcome.failure = "nonzero_exit"
                elif capture.stdout_overflow:
                    outcome.failure = "stdout_overflow"
                elif not await asyncio.to_thread(
                    authority_check, handler, event, "accept"
                ):
                    outcome.failure = "authority_refused"
                elif time.monotonic() >= deadline or job.cancel.is_set():
                    outcome.failure = "timeout"
                else:
                    try:
                        outcome.result = decode_result(bytes(capture.stdout), handler)
                    except ValueError:
                        outcome.failure = "invalid_output"
            finally:
                cancelled.cancel()
                await asyncio.gather(cancelled, return_exceptions=True)
        except asyncio.CancelledError:
            # Loop teardown is not launch-failure evidence. Never release an
            # uncertain token/ticket even if no transport has arrived yet.
            job.stop()
            outcome.failure = "owner_loop_cancelled"
            raise
        except Exception:  # noqa: BLE001 - callbacks may contain secret exception text
            outcome.failure = "owner_or_launch_refused"
            terminal = not launched and capture.transport is None
        finally:
            outcome.duration = time.monotonic() - started
            if capture.transport is not None:
                job.stop()
                terminal = await self._terminate(job)
            if job.token is not None:
                try:
                    await asyncio.to_thread(
                        self.owner.settle_process, job.token, terminal
                    )
                except Exception:  # noqa: BLE001
                    # Any owner failure preserves unresolved custody.
                    terminal = False
                    outcome.failure = outcome.failure or "settlement_failed"
            outcome.stderr_bytes = len(capture.stderr)
            outcome.stderr_truncated = capture.stderr_truncated
            outcome.cleanup_pending = not terminal
            if terminal:
                job.ticket.release()
                self.records.pop(job.id, None)
            else:
                outcome.result = None
                outcome.failure = outcome.failure or "cleanup_pending"
        return outcome

    async def _terminate(self, job: CommandJob) -> bool:
        capture = job.capture
        transport = capture.transport
        pid = transport.get_pid()
        if os.name == "nt":
            # Existing host tree-kill mechanics. Windows tree terminal proof is
            # not qualified here; retain unresolved custody after the attempt.
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    timeout=1,
                    capture_output=True,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                # Darwin can transiently return EPERM for an already-signalled
                # process group before reap. Attempted kill is not proof either
                # way; retain custody and inspect actual terminal evidence.
                pass
        for fd in (0, 1, 2):
            pipe = transport.get_pipe_transport(fd)
            if pipe is not None:
                pipe.close()
        while time.monotonic() < job.cleanup_deadline:
            if (
                capture.exited.done()
                and transport.get_returncode() is not None
                and os.name != "nt"
            ):
                try:
                    os.killpg(pid, 0)
                except ProcessLookupError:
                    transport.close()
                    return True
                except OSError:
                    pass
            await asyncio.sleep(0.01)
        return False

    async def reap_pending(self) -> None:
        """Retry retained terminal evidence without granting a new wait window."""
        for job in tuple(self.records.values()):
            if (
                job.task is not None
                and job.task.done()
                and job.capture.transport is not None
            ):
                # A new caller can observe terminal evidence now; do not wait or
                # extend the original cleanup allowance.
                capture = job.capture
                if capture.exited.done() and os.name != "nt":
                    try:
                        os.killpg(capture.transport.get_pid(), 0)
                    except ProcessLookupError:
                        try:
                            await asyncio.to_thread(
                                self.owner.settle_process, job.token, True
                            )
                        except Exception:  # noqa: BLE001, S112
                            # Retained metadata is the diagnostic.
                            continue
                        capture.transport.close()
                        job.ticket.release()
                        self.records.pop(job.id, None)
