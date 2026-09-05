"""Fresh local preview process with bounded pipes and deterministic reaping.

This is an engineering resource boundary, not a security sandbox. POSIX pipe
supervision is qualified; unsupported platforms refuse execution. The 32 MiB
working-payload estimate is deliberately conservative, not a bound on Python RSS.
OS address-space enforcement is reported only if setrlimit actually succeeds.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import selectors
import string
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .lab_models import RunRequest, RunResult, canonical_json
from .lab_preflight import PreviewUnsupportedError, prepare_recipe

_WORKING_BYTES = 33554432
_ADDRESS_SPACE_BYTES = 1073741824
_REAP_SECONDS = 0.5
_POLL_SECONDS = 0.025
_FRAME_HEADER = 8
_MAX_INPUT_BYTES = 33554432
_MAX_SAMPLE_BYTES = 2097152
_MAX_CHUNKS = 10000
_MAX_RESULT_BYTES = 33554432
_MAX_WALL_SECONDS = 60.0


@dataclass(frozen=True)
class PreviewLimits:
    """Hard parent budgets; lower values support explicit smaller previews."""

    sample_bytes: int = _MAX_SAMPLE_BYTES
    chunks: int = _MAX_CHUNKS
    result_bytes: int = _MAX_RESULT_BYTES
    wall_seconds: float = _MAX_WALL_SECONDS

    def __post_init__(self):
        for name, ceiling in (
            ("sample_bytes", _MAX_SAMPLE_BYTES),
            ("chunks", _MAX_CHUNKS),
            ("result_bytes", _MAX_RESULT_BYTES),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError(f"{name} must be positive and within the v1 ceiling")
        if (
            not math.isfinite(self.wall_seconds)
            or not 0 < self.wall_seconds <= _MAX_WALL_SECONDS
        ):
            raise ValueError(
                f"wall_seconds must be positive and at most {_MAX_WALL_SECONDS:g}"
            )


class _Limited(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def terminal_result(
    request: RunRequest,
    status: str,
    code: str,
    *,
    started: str | None = None,
    elapsed_ms: float = 0,
) -> RunResult:
    """Create a content-free terminal diagnostic retaining the exact request."""
    now = datetime.now(UTC).isoformat()
    return RunResult(
        request=request,
        status=status,
        report=None,
        started_at=started or now,
        finished_at=now,
        elapsed_ms=elapsed_ms,
        error={"code": code},
    )


def _encoded(result: RunResult) -> bytes:
    # Identical canonical accounting to the recovery store, including request.
    return canonical_json(result.model_dump(mode="json")).encode("utf-8")


def _check_working(size: int) -> None:
    if size > _WORKING_BYTES:
        raise _Limited("intermediate_limit")


def _format_bound(template: str, chunk_bytes: int, count: int) -> int:
    size = 0
    for literal, field, _, _ in string.Formatter().parse(template):
        size += len(literal.encode("utf-8"))
        if field is not None:
            size += chunk_bytes if field == "chunk" else len(str(count))
        _check_working(size)
    return size


def _admit_text(
    request: RunRequest, text: str, limits: PreviewLimits, pre_metadata: int = 0
) -> int:
    """Conservative aggregate text, token-index, and attributed-history estimate.

    Counts do not decrease after filtering/merging, and merged histories are
    charged to every surviving record. Repeated overlap doubles contributor and
    history bounds. This can refuse a recipe whose actual output would fit.
    """
    body = json.loads(request.recipe.effective_json)
    config = body["chunking"]["config"]
    words = body["chunking"]["method"] == "words"
    units = sum(1 for _ in re.finditer(r"\S+", text)) if words else len(text)
    step = config["max_size"] - config["overlap"]
    count = (
        (units + step - 1) // step
        if words
        else (1 + max(0, units - config["max_size"] + step - 1) // step if units else 0)
    )
    if count > limits.chunks:
        raise _Limited("chunk_limit")
    source_bytes = len(text.encode("utf-8"))
    duplication = (config["max_size"] + step - 1) // step
    total_text = source_bytes * duplication
    largest = source_bytes if words else min(source_bytes, 4 * config["max_size"])
    # Python token arrays and spans are a separate cost from emitted text.
    token_cost = units * (160 + 16 * duplication) if words else 0
    history = count * (1024 + pre_metadata)
    contributors = count
    peak = 2 * total_text + history + token_cost
    _check_working(peak)
    for operation in body["postprocessing"]:
        name, options = operation["operation"], operation["config"]
        config_bytes = len(canonical_json(options).encode("utf-8"))
        previous_text = total_text
        if name == "format_chunks":
            occurrences = sum(
                field == "chunk"
                for _, field, _, _ in string.Formatter().parse(options["template"])
            )
            literal = _format_bound(options["template"], 0, count)
            total_text = total_text * occurrences + count * literal
            largest = _format_bound(options["template"], largest, count)
        elif name == "add_metadata":
            extra = sum(
                _format_bound(options[key], 0, count) for key in ("prefix", "suffix")
            )
            total_text += count * extra
            largest += extra
        elif name == "add_overlap" and options["size"]:
            extra = min(largest, 4 * options["size"]) + (
                2 * len(options["marker"].encode("utf-8")) + 3
                if options["marker"]
                else 0
            )
            total_text += count * extra
            largest += extra
            history = 2 * history + count * extra
            contributors *= 2
        elif name == "merge_small":
            total_text += count * len(options["separator"].encode("utf-8"))
            largest = total_text
            # Each record can carry all prior history after a merge. Charging
            # that worst case avoids simulating the vendor's merge algorithm.
            history *= max(1, count)
            contributors *= max(1, count)
        history += count * (512 + config_bytes) + 16 * contributors
        peak = max(peak, previous_text + total_text + 2 * history + token_cost)
        _check_working(peak)
    return peak


def _checked_request(request: RunRequest, limits: PreviewLimits) -> RunRequest:
    request = RunRequest.model_validate(request.model_dump(mode="json"))
    if len(request.sample.text.encode("utf-8")) > limits.sample_bytes:
        raise _Limited("sample_limit")
    try:
        recipe = prepare_recipe(
            json.loads(request.recipe.authored_json), runtime=request.recipe.runtime
        )
    except PreviewUnsupportedError as exc:
        if exc.field.startswith("resource."):
            raise _Limited(
                "recipe_limit" if exc.field == "resource.recipe" else "operation_limit"
            ) from exc
        raise
    if recipe != request.recipe:
        raise PreviewUnsupportedError("snapshot", "Captured runtime or recipe changed")
    body = json.loads(recipe.effective_json)
    for operation in body["preprocessing"]:
        if operation["operation"] == "normalize_whitespace":
            _check_working(
                operation["config"]["max_line_breaks"]
                + 2 * len(request.sample.text.encode("utf-8"))
            )
    # Parent performs only bounded arithmetic; all regex/preprocessing stays in
    # the supervised child. No source sanitation assumptions cross this gate.
    if not body["preprocessing"] and all(
        char not in request.sample.text
        for char in (
            "\x00",
            "\u202a",
            "\u202b",
            "\u202c",
            "\u202d",
            "\u202e",
            "\u2066",
            "\u2067",
            "\u2068",
            "\u2069",
        )
    ):
        _admit_text(request, request.sample.text, limits)
    return request


def _child_admission(request: RunRequest, limits: PreviewLimits) -> int:
    from .template_runtime import (
        run_template_preprocessing_operation,
        sanitize_template_input,
    )

    text = request.sample.text
    metadata_bytes = 0
    pre_peak = len(text.encode("utf-8"))
    for operation in json.loads(request.recipe.effective_json)["preprocessing"]:
        name, config = operation["operation"], operation["config"]
        if name == "normalize_whitespace":
            pre_peak = max(
                pre_peak,
                2 * len(text.encode("utf-8"))
                + config["max_line_breaks"]
                + metadata_bytes,
            )
            _check_working(pre_peak)
        if name == "extract_sections":
            # Prescan incrementally before vendor finditer materializes the list;
            # overlapping captures are charged individually, including JSON cost.
            for count, match in enumerate(
                re.finditer(config["pattern"], text, re.MULTILINE), 1
            ):
                if count > 10000:
                    raise _Limited("section_limit")
                title = match.group(1) if match.re.groups else match.group(0)
                metadata_bytes += (
                    len(
                        canonical_json(
                            {"title": title, "position": match.start()}
                        ).encode("utf-8")
                    )
                    + 128
                )
                _check_working(metadata_bytes)
        result = run_template_preprocessing_operation(text, name, config)
        if isinstance(result, dict):
            text = result["text"]
            if name != "extract_sections":
                metadata_bytes += (
                    len(canonical_json(result.get("metadata", {})).encode("utf-8"))
                    + 128
                )
        else:
            text = result
        pre_peak = max(pre_peak, 2 * len(text.encode("utf-8")) + metadata_bytes)
        _check_working(pre_peak)
    sanitized = sanitize_template_input(text)
    return max(pre_peak, _admit_text(request, sanitized, limits, metadata_bytes))


def _resource_limits(limits: PreviewLimits) -> dict:
    applied = {}
    try:
        import resource
    except ImportError:
        return applied
    for name, value in (
        ("RLIMIT_AS", _ADDRESS_SPACE_BYTES),
        ("RLIMIT_CPU", math.ceil(limits.wall_seconds) + 1),
    ):
        if hasattr(resource, name):
            try:
                resource.setrlimit(getattr(resource, name), (value, value))
                applied[name] = value
            except (OSError, ValueError):
                pass
    return applied


def _worker_command() -> list[str]:
    return [sys.executable, "-m", "tldw_chatbook.Chunking.lab_runner"]


def _worker_main() -> None:
    """Top-level fresh-interpreter entry; protocol is data, never executable code."""
    header = sys.stdin.buffer.read(_FRAME_HEADER)
    if len(header) != _FRAME_HEADER:
        return
    size = struct.unpack("!Q", header)[0]
    if size > _MAX_INPUT_BYTES:
        return
    data = json.loads(sys.stdin.buffer.read(size))
    request = RunRequest.model_validate(data["request"])
    limits = PreviewLimits(**data["limits"])
    start = time.monotonic()
    started = datetime.now(UTC).isoformat()
    applied = _resource_limits(limits)
    try:
        from .template_runtime import execute_prepared

        request = _checked_request(request, limits)
        estimate = _child_admission(request, limits)
        report = execute_prepared(request.recipe, request.sample.text)
        if len(report.chunks) > limits.chunks:
            raise _Limited("chunk_limit")
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
        report = report.model_copy(
            update={
                "diagnostics": (
                    *report.diagnostics,
                    {
                        "kind": "resources",
                        "estimated_working_bytes": estimate,
                        "peak_rss_bytes": rss_bytes,
                        "applied_limits": applied,
                    },
                )
            }
        )
        result = RunResult(
            request=request,
            status="completed",
            report=report,
            started_at=started,
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_ms=(time.monotonic() - start) * 1000,
            error=None,
        )
        if len(_encoded(result)) > limits.result_bytes:
            raise _Limited("result_limit")
    except _Limited as exc:
        result = terminal_result(
            request,
            "limited",
            exc.code,
            started=started,
            elapsed_ms=(time.monotonic() - start) * 1000,
        )
    except MemoryError:
        result = terminal_result(request, "limited", "memory_limit", started=started)
    except Exception:  # noqa: BLE001 - child failures are sanitized terminal outcomes.
        result = terminal_result(request, "failed", "execution_failed", started=started)
    payload = _encoded(result)
    if len(payload) > limits.result_bytes:
        # Send only an over-limit length. Parent materializes the diagnostic from
        # its original request without accepting an oversized allocation.
        sys.stdout.buffer.write(struct.pack("!Q", len(payload)))
    else:
        sys.stdout.buffer.write(struct.pack("!Q", len(payload)) + payload)
    sys.stdout.buffer.flush()


class LocalPreviewRunner:
    """Own exactly one child through termination/reaping, including canceled awaits."""

    def __init__(self, limits: PreviewLimits):
        self.limits = limits
        self._task: asyncio.Task | None = None
        self._stop = threading.Event()
        self._closed = False

    async def run(self, request: RunRequest) -> RunResult:
        """Execute a captured request off-loop; never queue a concurrent call."""
        if self._closed or self._task is not None:
            raise RuntimeError("Preview runner is closed or busy")
        self._stop.clear()
        self._task = asyncio.create_task(asyncio.to_thread(self._supervise, request))
        owned = self._task
        try:
            return await asyncio.shield(owned)
        except asyncio.CancelledError:
            self._stop.set()
            await self._settle(owned)
            raise
        finally:
            self._task = None

    @staticmethod
    async def _settle(task: asyncio.Task) -> None:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue

    async def cancel(self) -> None:
        """Return only after the owned supervisor has killed/reaped its child."""
        self._stop.set()
        if self._task is not None:
            await self._settle(self._task)

    async def close(self) -> None:
        """Refuse new work and stop current work."""
        self._closed = True
        await self.cancel()

    def _supervise(self, request: RunRequest) -> RunResult:
        started = datetime.now(UTC).isoformat()
        start = time.monotonic()
        process = None
        usage = None
        try:
            request = _checked_request(request, self.limits)
            if os.name != "posix":
                return terminal_result(request, "failed", "platform_unqualified")
            payload = canonical_json(
                {
                    "request": request.model_dump(mode="json"),
                    "limits": asdict(self.limits),
                }
            ).encode("utf-8")
            if len(payload) > _MAX_INPUT_BYTES:
                raise _Limited("request_limit")
            if self._stop.is_set():
                return terminal_result(request, "canceled", "canceled")
            if time.monotonic() - start > self.limits.wall_seconds:
                raise _Limited("time_limit")
            process = subprocess.Popen(
                _worker_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parents[2],
                bufsize=0,
            )
            packet = memoryview(struct.pack("!Q", len(payload)) + payload)
            output = bytearray()
            expected = None
            with selectors.DefaultSelector() as selector:
                for stream, event in (
                    (process.stdin, selectors.EVENT_WRITE),
                    (process.stdout, selectors.EVENT_READ),
                ):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, event)
                while True:
                    if self._stop.is_set():
                        return terminal_result(request, "canceled", "canceled")
                    if time.monotonic() - start > self.limits.wall_seconds:
                        raise _Limited("time_limit")
                    for key, _ in selector.select(_POLL_SECONDS):
                        if key.fileobj is process.stdin:
                            try:
                                packet = packet[os.write(key.fd, packet[:65536]) :]
                            except BrokenPipeError:
                                packet = packet[len(packet) :]
                            if not packet:
                                selector.unregister(process.stdin)
                                process.stdin.close()
                        else:
                            cap = (
                                _FRAME_HEADER
                                if expected is None
                                else _FRAME_HEADER + expected
                            )
                            chunk = os.read(key.fd, min(65536, cap - len(output) + 1))
                            if not chunk:
                                selector.unregister(process.stdout)
                            output.extend(chunk)
                            if expected is None and len(output) >= _FRAME_HEADER:
                                expected = struct.unpack("!Q", output[:_FRAME_HEADER])[
                                    0
                                ]
                                if expected > self.limits.result_bytes:
                                    raise _Limited("result_limit")
                            if (
                                expected is not None
                                and len(output) > expected + _FRAME_HEADER
                            ):
                                raise ValueError("Invalid worker frame")
                    if process.returncode is None:
                        pid, status, measured = os.wait4(process.pid, os.WNOHANG)
                        if pid:
                            process.returncode = os.waitstatus_to_exitcode(status)
                            usage = measured
                    if process.returncode is not None:
                        # Drain the final pipe bytes before examining exit status.
                        if any(
                            key.fileobj is process.stdout
                            for key in selector.get_map().values()
                        ):
                            continue
                        if (
                            process.returncode
                            or expected is None
                            or len(output) != expected + _FRAME_HEADER
                        ):
                            raise ValueError("Worker exited without a complete result")
                        result = RunResult.model_validate_json(output[_FRAME_HEADER:])
                        if result.request != request:
                            raise ValueError("Worker result is not the captured member")
                        if (
                            result.report
                            and len(result.report.chunks) > self.limits.chunks
                        ):
                            raise _Limited("chunk_limit")
                        if result.report and usage is not None:
                            peak = int(
                                usage.ru_maxrss
                                if sys.platform == "darwin"
                                else usage.ru_maxrss * 1024
                            )
                            diagnostics = tuple(
                                {**item, "peak_rss_bytes": peak}
                                if item.get("kind") == "resources"
                                else item
                                for item in result.report.diagnostics
                            )
                            result = result.model_copy(
                                update={
                                    "report": result.report.model_copy(
                                        update={"diagnostics": diagnostics}
                                    )
                                }
                            )
                        if len(_encoded(result)) > self.limits.result_bytes:
                            raise _Limited("result_limit")
                        if time.monotonic() - start > self.limits.wall_seconds:
                            raise _Limited("time_limit")
                        return result
        except _Limited as exc:
            return terminal_result(
                request,
                "limited",
                exc.code,
                started=started,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        except Exception:  # noqa: BLE001 - process/protocol failures never expose private payloads.
            return terminal_result(
                request,
                "failed",
                "worker_failed",
                started=started,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=_REAP_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        # A slow OS reap must not relinquish ownership or allow
                        # a second process. Each wait is bounded and off-loop;
                        # cancel returns only once termination is observable.
                        while True:
                            try:
                                process.wait(timeout=_REAP_SECONDS)
                                break
                            except subprocess.TimeoutExpired:
                                continue
                else:
                    process.wait(timeout=_REAP_SECONDS)
                for stream in (process.stdin, process.stdout):
                    with contextlib.suppress(OSError):
                        stream.close()


if __name__ == "__main__":
    _worker_main()
