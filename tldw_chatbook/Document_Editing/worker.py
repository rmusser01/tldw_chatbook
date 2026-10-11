"""One app-owned document process lane; no save or document-path authority."""

import asyncio
import json
import struct
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import CheckResult, DocumentRequest, FormatResult
from .scheduler import DocumentCheckScheduler
from .worker_process import (
    MAX_FRAME,
    admit,
    decode_request,
    decode_result,
    encode,
    refusal,
)

WORKER_PATH = Path(__file__).with_name("worker_process.py")


class DocumentWorker:
    """Own lazy single-request children and the application's active editor lease."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[Any] | None = None
        self._lease: DocumentCheckScheduler | None = None
        self._closed = False
        self._lease_generation = 0
        self._lease_lock = asyncio.Lock()

    async def acquire(
        self, on_result: Callable[[CheckResult], None]
    ) -> DocumentCheckScheduler:
        """Retire the previous editor before handing the lane to its successor."""
        if self._closed:
            raise RuntimeError("Document worker is closed")
        self._lease_generation += 1
        generation = self._lease_generation
        async with self._lease_lock:
            if self._lease is not None:
                await self._lease.close()
            lease = DocumentCheckScheduler(self, on_result)
            if self._closed or generation != self._lease_generation:
                lease.retire()
            else:
                self._lease = lease
            return lease

    async def check(self, request: DocumentRequest) -> CheckResult:
        """Check one bounded snapshot through the isolated child."""
        return await self._run(request, "check")

    async def format(self, request: DocumentRequest) -> FormatResult:
        """Format through the same process lane and authoritative engine check."""
        return await self._run(request, "format")

    async def _run(self, request, operation):
        check = admit(request)
        if check is None and (self._closed or self._task is not None):
            check = refusal(request, "failed")
        if check is not None:
            return (
                check
                if operation == "check"
                else FormatResult(
                    request.key,
                    (
                        check.state
                        if check.state in ("failed", "resource_limited")
                        else "blocked"
                    ),
                    None,
                    check,
                )
            )
        self._task = asyncio.current_task()
        launch = None
        process = None
        try:
            value = {"operation": operation, "request": asdict(request)}
            decode_request(value)
            payload = encode(value)
            async with asyncio.timeout(2):
                launch = asyncio.create_task(
                    asyncio.create_subprocess_exec(
                        sys.executable,
                        "-I",
                        str(WORKER_PATH),
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                        env={},
                        cwd=WORKER_PATH.parent,
                        close_fds=True,
                    )
                )
                process = await asyncio.shield(launch)
                self._process = process
                if process.stdin is None or process.stdout is None:
                    raise RuntimeError("Missing worker pipes")
                process.stdin.write(payload)
                await process.stdin.drain()
                process.stdin.close()
                size = struct.unpack("!I", await process.stdout.readexactly(4))[0]
                if size > MAX_FRAME:
                    raise ValueError("Reply exceeds limit")
                body = await process.stdout.readexactly(size)
                result = decode_result(request, operation, json.loads(body))
                if await process.wait() != 0:
                    raise ValueError("Worker exited unsuccessfully")
                return result
        except TimeoutError:
            check = refusal(request, "resource_limited")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - source-safe process boundary
            check = refusal(request, "failed")
        finally:
            cleanup = asyncio.create_task(self._cleanup(process, launch))
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            self._process = None
            self._task = None
            if cancelled:
                raise asyncio.CancelledError
        return (
            check
            if operation == "check"
            else FormatResult(request.key, check.state, None, check)
        )

    async def _cleanup(self, process, launch):
        if process is None and launch is not None:
            try:
                process = await launch
            except Exception:  # noqa: BLE001 - source-safe launch failure
                return
        if process is None:
            return
        reaped = asyncio.create_task(process.wait())
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(asyncio.shield(reaped), 0.1)
        except TimeoutError:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        # CPython owns these transports; inherited fixture pipes must not hold wait open.
        process._transport.close()
        await reaped

    async def close(self) -> None:
        """Permanently retire the lease, cancel work and reap its child."""
        self._closed = True
        if self._lease is not None:
            await self._lease.close()
            self._lease = None
        if self._task is not None and self._task is not asyncio.current_task():
            task = self._task
            if not task.cancelling():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
