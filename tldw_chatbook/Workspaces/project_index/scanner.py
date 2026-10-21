"""Bounded parent transport; durable frontier and publication belong to K3."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tldw_chatbook.STT.executor_process_tree import (
    ExecutorProcessTree,
    WorkerContainmentIdentity,
)
from tldw_chatbook.Tools.project_index_scan_protocol import (
    PAGE_TEXT_BYTES,
    SOURCE_BUFFER_HEADROOM,
    ScanProtocolError,
    authority_fields,
    encode_frame,
    frame,
    initialization,
    read_frame,
    write_frame,
)
from tldw_chatbook.Tools.workspace_tool_executor import (
    _PopenProcessAdapter,
    workspace_worker_environment,
)
from tldw_chatbook.Workspaces.project_index.models import (
    FileCapture,
    MAX_FILE_BYTES,
    ProjectIndexAuthority,
    ProjectIndexPolicy,
    ProjectIndexScope,
    ScanEntry,
    ScanPage,
    ScanRunResult,
    relative_path,
)


class ProjectScanner:
    """One operation at a time, with observable physical retirement evidence.

    The caller supplies fresh code-issued authority for every unit. A return from
    scan alone never grants publication; the later store/registry fence does.
    Metrics and retirement observations describe the most recent operation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.retired = True
        self.worker_pid = 0
        self.process_returncode = None
        self.pipes_closed = True
        self.max_queued_text_bytes = 0
        self.max_active_reads = 0
        self.max_frame_bytes = 0
        self._ack = 0
        self._deadline = 0.0
        self._timed_out = False

    def _write(self, stream, value):
        self.max_frame_bytes = max(self.max_frame_bytes, write_frame(stream, value))

    def _read(self, stream):
        value = read_frame(stream)
        self.max_frame_bytes = max(self.max_frame_bytes, len(encode_frame(value)))
        if value["type"] == "Error":
            raise ScanProtocolError("worker_refused")
        return value

    @contextmanager
    def _worker(self, authority, policy, stop):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("scanner_busy")
        self.retired = False
        self.pipes_closed = False
        self.process_returncode = None
        self.max_queued_text_bytes = self.max_active_reads = self.max_frame_bytes = 0
        self._ack = 0
        self._timed_out = False
        process = tree = watcher = None
        finished = threading.Event()
        try:
            # Windows venv python.exe is a launcher that can create its real
            # interpreter before Job Object assignment. Start the native runtime
            # directly; -I -S excludes cwd, environment Python paths and .pth/site.
            # Only this installed code location enters bootstrap argv, never the
            # admitted project locator, exclusions, policy or file bodies.
            process = subprocess.Popen(
                [
                    getattr(sys, "_base_executable", sys.executable),
                    "-I",
                    "-S",
                    "-c",
                    "import runpy,sys;sys.path.insert(0,"
                    + repr(str(Path(__file__).resolve().parents[3]))
                    + ");runpy.run_module('tldw_chatbook.Tools.project_index_scan_worker',run_name='__main__')",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=workspace_worker_environment(authority.root.path),
                shell=False,
                start_new_session=os.name == "posix",
            )
            self.worker_pid = process.pid
            tree = ExecutorProcessTree(
                _PopenProcessAdapter(process),
                threading.Event(),
                WorkerContainmentIdentity(
                    process.pid, process.pid if os.name == "posix" else None
                ),
            )
            tree.admit()
            self._deadline = time.monotonic() + 300

            def monitor():
                while not finished.wait(0.02):
                    if stop.is_set() or time.monotonic() >= self._deadline:
                        self._timed_out = not stop.is_set()
                        tree.terminate_tree(term_timeout=1, kill_timeout=1)
                        return

            watcher = threading.Thread(
                target=monitor, name="project-index-retirement", daemon=True
            )
            watcher.start()
            self._write(process.stdin, initialization(authority, policy))
            if self._read(process.stdout) != frame("Started", revision=-1, ack=0):
                raise ScanProtocolError("init_not_acknowledged")
            yield process
        finally:
            # Stop publication/admission first (the calling loop has ended), then
            # kill the complete tree, close every pipe and join the sole watcher.
            finished.set()
            proven = False
            try:
                if tree is not None:
                    proven = tree.terminate_tree(term_timeout=1, kill_timeout=1)
                elif process is not None:
                    process.kill()
                    process.wait(timeout=2)
                    proven = process.returncode is not None
                else:
                    proven = True
            finally:
                if watcher is not None:
                    watcher.join(timeout=5)
                if process is not None:
                    for pipe in (process.stdin, process.stdout):
                        if pipe is not None:
                            pipe.close()
                    self.process_returncode = process.poll()
                    self.pipes_closed = all(
                        p is None or p.closed for p in (process.stdin, process.stdout)
                    )
                else:
                    self.pipes_closed = True
                self.retired = (
                    proven
                    and self.pipes_closed
                    and (watcher is None or not watcher.is_alive())
                )
                self._lock.release()
            if not self.retired:
                raise RuntimeError("worker_retirement_unproven")

    def _unit(self, process, request):
        self._deadline = time.monotonic() + 300
        request["ack"] = self._ack
        self._write(process.stdin, request)
        self._ack += 1
        if self._read(process.stdout) != frame(
            "Started", revision=request["revision"], ack=self._ack
        ):
            raise ScanProtocolError("admission_not_acknowledged")
        entries = None
        captures = []
        parts = []
        metadata = None
        part_bytes = retained_bytes = 0
        ordinal = 0
        while True:
            value = self._read(process.stdout)
            kind = value["type"]
            if kind == "File":
                if entries is not None or len(captures) >= 256:
                    raise ScanProtocolError("unexpected_file")
                stamp = (value["entry"], value["sha256"])
                if value["part"] != ordinal or (
                    metadata is not None and metadata != stamp
                ):
                    raise ScanProtocolError("file_part_order")
                metadata = stamp
                ordinal += 1
                part_bytes += len(value["text"].encode("utf-8"))
                if (
                    part_bytes > MAX_FILE_BYTES
                    or retained_bytes + part_bytes > PAGE_TEXT_BYTES
                ):
                    raise ScanProtocolError("capture_limit")
                self.max_queued_text_bytes = max(
                    self.max_queued_text_bytes,
                    retained_bytes + part_bytes + SOURCE_BUFFER_HEADROOM,
                )
                parts.append(value["text"])
                if value["final"]:
                    entry = value["entry"]
                    text = "".join(parts)
                    digest = value["sha256"]
                    parts.clear()
                    value = None
                    captures.append(
                        FileCapture(
                            entry["relative_path"],
                            entry["identity"],
                            entry["mtime_ns"],
                            entry["size_bytes"],
                            digest,
                            text,
                        )
                    )
                    del text
                    retained_bytes += part_bytes
                    metadata = None
                    part_bytes = ordinal = 0
            elif kind == "Entries":
                if entries is not None or metadata is not None:
                    raise ScanProtocolError("unexpected_entries")
                entries = tuple(ScanEntry(**e) for e in value["entries"])
            elif kind == "DirectoryDone":
                if metadata is not None or (
                    request["type"] == "Directory" and entries is None
                ):
                    raise ScanProtocolError("incomplete_page")
                self.max_active_reads = max(self.max_active_reads, value["reads"])
                self.max_queued_text_bytes = max(
                    self.max_queued_text_bytes, value["queued"]
                )
                return entries or (), tuple(captures), value
            else:
                raise ScanProtocolError("unexpected_response")

    def scan(
        self,
        authority: ProjectIndexAuthority,
        scope: ProjectIndexScope,
        job_id: str,
        frontier: Iterator[str],
        emit: Callable[[ScanPage], bool],
        stop: threading.Event,
        *,
        admit_unit: Callable[[str], ProjectIndexAuthority | None],
        policy: ProjectIndexPolicy,
    ) -> ScanRunResult:
        """Consume a lazy durable frontier, stopping after rejected publication.

        Args:
            authority: Initial exact root and authority floor.
            scope: Exact index scope already claimed by the caller.
            job_id: Opaque claimed job identity (never sent on argv).
            frontier: Caller-owned iterator of incomplete relative directories.
            emit: Synchronous bounded handoff; False stops all further admission.
            stop: Cooperative cancellation, also monitored during blocked pipe IO.
            admit_unit: Fresh authority callback before every page request.
            policy: Immutable per-job processing policy snapshot.
        Returns:
            Truthful coverage outcome, with process retirement completed first.
        """
        if scope.key != authority.key or not job_id:
            raise ValueError("scope_mismatch")
        if stop.is_set():
            return ScanRunResult("interrupted", 0, 0, ())
        directories = discovered = 0
        issues = set()
        status = "complete"
        floor = set(authority.exclusions)
        revision = authority.authority_revision
        try:
            with self._worker(authority, policy, stop) as process:
                for raw_directory in frontier:
                    directory = relative_path(raw_directory, root=True)
                    continuation = False
                    while True:
                        if stop.is_set():
                            return ScanRunResult(
                                "interrupted",
                                directories,
                                discovered,
                                tuple(sorted(issues)),
                            )
                        admitted = admit_unit(directory)
                        if (
                            admitted is None
                            or admitted.key != authority.key
                            or admitted.chain != authority.chain
                            or admitted.authority_revision < revision
                        ):
                            return ScanRunResult(
                                "interrupted",
                                directories,
                                discovered,
                                ("authority_changed",),
                            )
                        revision = admitted.authority_revision
                        floor.update(admitted.exclusions)
                        admitted = replace(admitted, exclusions=tuple(sorted(floor)))
                        entries, captures, done = self._unit(
                            process,
                            frame(
                                "Directory",
                                directory=directory,
                                continuation=continuation,
                                ack=0,
                                **authority_fields(admitted),
                            ),
                        )
                        if done["directory"] != directory:
                            raise ScanProtocolError("directory_mismatch")
                        if stop.is_set():
                            return ScanRunResult(
                                "interrupted",
                                directories,
                                discovered,
                                tuple(sorted(issues)),
                            )
                        page = ScanPage(
                            directory,
                            done["identity"],
                            entries,
                            captures,
                            done["complete"],
                            revision,
                        )
                        if not emit(page):
                            return ScanRunResult(
                                "paused", directories, discovered, tuple(sorted(issues))
                            )
                        discovered += len(entries)
                        directories += int(done["complete"])
                        issues.update(done["issues"])
                        if done["issues"] or not done["complete"] and not done["more"]:
                            status = "partial"
                        if "entry_limit" in issues:
                            return ScanRunResult(
                                "partial",
                                directories,
                                discovered,
                                tuple(sorted(issues)),
                            )
                        # Release the previous handoff before any next-directory or
                        # continuation admission, including the last page of a directory.
                        del page, captures, entries
                        if not done["more"]:
                            break
                        continuation = True
                self._write(process.stdin, frame("Close"))
        except (OSError, ValueError):
            status = "interrupted" if stop.is_set() else "error"
            issues.add("worker_timeout" if self._timed_out else "scan_unavailable")
        return ScanRunResult(status, directories, discovered, tuple(sorted(issues)))

    def capture_file(
        self,
        authority: ProjectIndexAuthority,
        relative_path: str,
        *,
        stop: threading.Event,
    ) -> FileCapture | None:
        """Validate one already policy-selected candidate through the private wire."""
        if stop.is_set():
            return None
        try:
            with self._worker(authority, ProjectIndexPolicy((), ()), stop) as process:
                _, captures, done = self._unit(
                    process,
                    frame(
                        "Capture",
                        path=relative_path,
                        ack=0,
                        **authority_fields(authority),
                    ),
                )
                self._write(process.stdin, frame("Close"))
                if (
                    not stop.is_set()
                    and done["complete"]
                    and not done["issues"]
                    and len(captures) == 1
                    and captures[0].relative_path == relative_path
                ):
                    return captures[0]
        except (OSError, ValueError):
            return None
        return None
