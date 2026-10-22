"""Stdlib-only, pinned, single-job project scanner. No database or UI imports."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import BinaryIO

from tldw_chatbook.Tools.local_tool_impls import _is_relative_sensitive_path
from tldw_chatbook.Tools.project_index_scan_protocol import (
    PAGE_TEXT_BYTES,
    SOURCE_BUFFER_HEADROOM,
    TEXT_PART_BYTES,
    ScanProtocolError,
    frame,
    read_frame,
    write_frame,
)
from tldw_chatbook.Tools.workspace_root_pin import (
    pin_workspace_root,
    WorkspaceRootPinError,
)
from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryChain,
    DirectoryIdentity,
    capture_directory_chain,
)
from tldw_chatbook.Utils.sensitive_paths import (
    SensitiveExclusion,
    _SENSITIVE_FILE_NAMES,
)
from tldw_chatbook.Workspaces.project_index.models import (
    FileCapture,
    MAX_ENTRIES,
    MAX_FILE_BYTES,
    MAX_PAGE_ENTRIES,
    ProjectIndexPolicy,
    ScanEntry,
    relative_path,
)

PROCESSING_EXCLUSIONS = frozenset(
    {"node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".git"}
)
ARCHIVE_SUFFIXES = frozenset(
    {".zip", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tar", ".whl", ".jar"}
)


class UnsafeCapture(ValueError):
    """The candidate changed or cannot be safely accessed."""


def _safe_stat(value, *, directory=False):
    if stat.S_ISLNK(value.st_mode) or (
        os.name == "nt" and getattr(value, "st_file_attributes", 0x400) & 0x400
    ):
        raise UnsafeCapture("unsafe_metadata")
    if not (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)):
        raise UnsafeCapture("unsafe_metadata")
    return value


def _identity(value):
    return hashlib.sha256(
        f"{value.st_dev}:{value.st_ino}:{stat.S_IFMT(value.st_mode)}".encode()
    ).hexdigest()


def _state(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)


def _allowed(path, exclusions, *, directory, policy=None):
    parts = Path(path).parts
    if any(p.casefold() in PROCESSING_EXCLUSIONS for p in parts):
        return False
    if not directory and Path(path).name.casefold() in _SENSITIVE_FILE_NAMES:
        return False
    if _is_relative_sensitive_path(Path(path), exclusions, is_directory=directory):
        return False
    if policy is None:
        return True
    folded = path.casefold()
    prefixes = tuple(p.casefold() for p in policy.include_prefixes)
    included = not prefixes or any(
        not p or folded == p or folded.startswith(p + "/") for p in prefixes
    )
    if directory:
        return included or not path or any(p.startswith(folded + "/") for p in prefixes)
    suffix = Path(path).suffix.casefold()
    return (
        included
        and suffix not in ARCHIVE_SUFFIXES
        and (not policy.text_suffixes or suffix in policy.text_suffixes)
    )


def _windows_open(path, *, directory):
    """Open without following the final reparse point or sharing deletion.

    Holding every ancestor without FILE_SHARE_DELETE prevents junction/rename
    replacement during a child open. The returned CRT fd owns the native handle.
    """
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    flags = 0x00200000 | (0x02000000 if directory else 0)
    handle = kernel.CreateFileW(
        str(path), 0x0080 if directory else 0x80000000, 3, None, 3, flags, None
    )
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), "candidate_open_failed")
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle(handle)
        raise


@contextmanager
def _directory(path, exclusions):
    """Retain each ancestor; POSIX child opens are relative to retained fds."""
    path = relative_path(path, root=True)
    if not _allowed(path, exclusions, directory=True):
        raise UnsafeCapture("excluded_directory")
    descriptors = []
    try:
        base = Path(".")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        fd = (
            _windows_open(base, directory=True)
            if os.name == "nt"
            else os.open(base, flags)
        )
        descriptors.append(fd)
        _safe_stat(os.fstat(fd), directory=True)
        for part in Path(path).parts:
            base = base / part
            if not _allowed(base.as_posix(), exclusions, directory=True):
                raise UnsafeCapture("excluded_directory")
            before = (
                _safe_stat(os.lstat(base), directory=True)
                if os.name == "nt"
                else _safe_stat(
                    os.stat(part, dir_fd=fd, follow_symlinks=False), directory=True
                )
            )
            child = (
                _windows_open(base, directory=True)
                if os.name == "nt"
                else os.open(part, flags, dir_fd=fd)
            )
            descriptors.append(child)
            opened = _safe_stat(os.fstat(child), directory=True)
            if _identity(before) != _identity(opened):
                raise UnsafeCapture("directory_changed")
            fd = child
        yield base if os.name == "nt" else fd, fd
        # Retained POSIX descriptors cannot prevent renames. Verify every
        # lexical parent-child link again before a capture can leave this scope.
        base = Path(".")
        for index, part in enumerate(Path(path).parts):
            base = base / part
            current = (
                os.lstat(base)
                if os.name == "nt"
                else os.stat(part, dir_fd=descriptors[index], follow_symlinks=False)
            )
            if _identity(_safe_stat(current, directory=True)) != _identity(
                os.fstat(descriptors[index + 1])
            ):
                raise UnsafeCapture("directory_changed")
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def capture_candidate(
    path: str, exclusions: tuple[SensitiveExclusion, ...]
) -> FileCapture | None:
    """Capture strict UTF-8, identity and digest from one bounded descriptor read.

    Args:
        path: Canonical root-relative file path.
        exclusions: Complete admitted mandatory and binding exclusion floor.
    Returns:
        Immutable capture, or None for an ineligible/unsafe file.
    """
    path = relative_path(path)
    if not _allowed(path, exclusions, directory=False):
        return None
    candidate = Path(path)
    try:
        with _directory(
            candidate.parent.as_posix() if candidate.parent != Path(".") else "",
            exclusions,
        ) as (base, parent_fd):
            kwargs = {} if os.name == "nt" else {"dir_fd": parent_fd}
            target = base / candidate.name if os.name == "nt" else candidate.name
            before = _safe_stat(os.stat(target, follow_symlinks=False, **kwargs))
            if before.st_size > MAX_FILE_BYTES:
                return None
            flags = (
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            fd = (
                _windows_open(target, directory=False)
                if os.name == "nt"
                else os.open(target, flags, **kwargs)
            )
            try:
                opened = _safe_stat(os.fstat(fd))
                if _state(before) != _state(opened):
                    raise UnsafeCapture("file_changed")
                # A regular file may still yield short reads; never read past 1MiB+1.
                raw = bytearray()
                while len(raw) <= MAX_FILE_BYTES:
                    part = os.read(
                        fd, min(TEXT_PART_BYTES, MAX_FILE_BYTES + 1 - len(raw))
                    )
                    if not part:
                        break
                    raw.extend(part)
                finished = _safe_stat(os.fstat(fd))
            finally:
                os.close(fd)
            after = _safe_stat(os.stat(target, follow_symlinks=False, **kwargs))
            # Python 3.12 Windows stat uses creation time for ctime, while CRT
            # fstat may report mtime. Compare ctime within each source only.
            if (
                _state(before) != _state(finished)
                or _state(before) != _state(after)
                or before.st_ctime_ns != after.st_ctime_ns
                or opened.st_ctime_ns != finished.st_ctime_ns
            ):
                raise UnsafeCapture("file_changed")
            if (
                len(raw) != before.st_size
                or len(raw) > MAX_FILE_BYTES
                or b"\x00" in raw
            ):
                return None
            text = raw.decode("utf-8", errors="strict")
            digest = hashlib.sha256(raw).hexdigest()
            del raw
            return FileCapture(
                path,
                _identity(before),
                before.st_mtime_ns,
                before.st_size,
                digest,
                text,
            )
    except UnicodeDecodeError:
        return None


def _send_capture(stdout, capture):
    entry = asdict(
        ScanEntry(
            capture.relative_path,
            "file",
            capture.identity,
            capture.mtime_ns,
            capture.size_bytes,
        )
    )
    # 16K Unicode code points are <=64KiB UTF-8, even for astral characters.
    width = TEXT_PART_BYTES // 4
    for ordinal, offset in enumerate(range(0, max(1, len(capture.text)), width)):
        part = capture.text[offset : offset + width]
        write_frame(
            stdout,
            frame(
                "File",
                entry=entry,
                sha256=capture.content_sha256,
                text=part,
                part=ordinal,
                final=offset + width >= len(capture.text),
            ),
        )


def run_project_scan_worker(stdin: BinaryIO, stdout: BinaryIO) -> int:
    """Serve one pinned job and bounded, revision-acknowledged directory units."""
    iterator = None
    directory_context = None
    try:
        initial = read_frame(stdin)
        if initial["type"] != "Init":
            raise ScanProtocolError("init_required")
        chain = DirectoryChain(
            Path(initial["root"]),
            tuple(DirectoryIdentity(**i) for i in initial["identities"]),
        )
        policy = ProjectIndexPolicy(
            tuple(initial["policy"]["include_prefixes"]),
            tuple(initial["policy"]["text_suffixes"]),
        )
        if capture_directory_chain(chain.canonical_root) != chain:
            raise UnsafeCapture("root_changed")
        with pin_workspace_root(chain.canonical_root, chain):
            write_frame(stdout, frame("Started", revision=-1, ack=0))
            ack = revision = 0
            floor = set()
            discovered = 0
            current = None
            directory_before = None
            directory_fd = None
            while True:
                request = read_frame(stdin)
                kind = request["type"]
                if kind == "Close":
                    return 0
                if (
                    kind not in {"Directory", "Unit", "Capture"}
                    or request["ack"] != ack
                    or request["revision"] < revision
                ):
                    raise ScanProtocolError("unexpected_request")
                ack += 1
                revision = request["revision"]
                floor.update(SensitiveExclusion(*e) for e in request["exclusions"])
                exclusions = tuple(sorted(floor))
                write_frame(stdout, frame("Started", revision=revision, ack=ack))
                if capture_directory_chain(chain.canonical_root) != chain:
                    raise UnsafeCapture("root_changed")
                if kind == "Capture":
                    if iterator is not None:
                        raise ScanProtocolError("capture_during_directory")
                    try:
                        capture = capture_candidate(request["path"], exclusions)
                        if capture_directory_chain(chain.canonical_root) != chain:
                            raise UnsafeCapture("root_changed")
                        if capture:
                            _send_capture(stdout, capture)
                        write_frame(
                            stdout,
                            frame(
                                "DirectoryDone",
                                directory="",
                                identity="",
                                complete=True,
                                more=False,
                                issues=[],
                                reads=int(capture is not None),
                                queued=(
                                    capture.size_bytes + SOURCE_BUFFER_HEADROOM
                                    if capture
                                    else 0
                                ),
                            ),
                        )
                    except (OSError, UnsafeCapture):
                        write_frame(
                            stdout,
                            frame(
                                "DirectoryDone",
                                directory="",
                                identity="",
                                complete=False,
                                more=False,
                                issues=["capture_unavailable"],
                                reads=1,
                                queued=0,
                            ),
                        )
                    continue
                pending = (
                    list(request["directories"])
                    if kind == "Unit"
                    else [request["directory"]]
                )
                continuation = request["continuation"]
                inspections = unit_entries = sent_bytes = metadata_bytes = 0
                attempted = set()  # At most one bounded unit, never the whole tree.
                while pending and len(attempted) < MAX_PAGE_ENTRIES:
                    directory = pending.pop(0)
                    attempted.add(directory)
                    entries = []
                    issues = set()
                    complete = more = False
                    active_reads = 0
                    identity = ""
                    try:
                        if not _allowed(
                            directory, exclusions, directory=True, policy=policy
                        ):
                            raise UnsafeCapture("excluded_directory")
                        if continuation:
                            if iterator is None or current != directory:
                                raise ScanProtocolError("invalid_continuation")
                            with _directory(directory, exclusions) as (_, verified_fd):
                                if _identity(os.fstat(verified_fd)) != _identity(
                                    directory_before
                                ):
                                    raise UnsafeCapture("directory_changed")
                        else:
                            if iterator is not None:
                                raise ScanProtocolError("unacknowledged_directory")
                            directory_context = _directory(directory, exclusions)
                            base, directory_fd = directory_context.__enter__()
                            directory_before = os.fstat(directory_fd)
                            iterator = os.scandir(base)
                            current = directory
                        identity = _identity(directory_before)
                        if _state(os.fstat(directory_fd)) != _state(directory_before):
                            raise UnsafeCapture("directory_changed")
                        for _ in range(MAX_PAGE_ENTRIES - inspections):
                            if (
                                sent_bytes > PAGE_TEXT_BYTES - MAX_FILE_BYTES
                                or metadata_bytes > 512 * 1024
                            ):
                                more = True
                                break
                            try:
                                inspections += 1
                                item = next(iterator)
                            except StopIteration:
                                complete = True
                                break
                            path = (
                                f"{directory}/{item.name}" if directory else item.name
                            )
                            # Filter lexical exclusions before metadata/read, including ancestors.
                            if not _allowed(
                                path, exclusions, directory=True, policy=policy
                            ) and not _allowed(
                                path, exclusions, directory=False, policy=policy
                            ):
                                continue
                            info = item.stat(follow_symlinks=False)
                            is_dir = stat.S_ISDIR(info.st_mode)
                            try:
                                _safe_stat(info, directory=is_dir)
                            except UnsafeCapture:
                                continue
                            if not _allowed(
                                path, exclusions, directory=is_dir, policy=policy
                            ):
                                continue
                            if discovered >= MAX_ENTRIES:
                                issues.add("entry_limit")
                                break
                            if is_dir:
                                entry = ScanEntry(
                                    path,
                                    "directory",
                                    _identity(info),
                                    info.st_mtime_ns,
                                    0,
                                )
                            else:
                                if info.st_size > MAX_FILE_BYTES:
                                    continue
                                active_reads = 1
                                capture = capture_candidate(path, exclusions)
                                if capture is None:
                                    continue
                                entry = ScanEntry(
                                    path,
                                    "file",
                                    capture.identity,
                                    capture.mtime_ns,
                                    capture.size_bytes,
                                )
                                _send_capture(stdout, capture)
                                sent_bytes += capture.size_bytes
                                del capture
                            entries.append(asdict(entry))
                            metadata_bytes += len(
                                json.dumps(entries[-1], ensure_ascii=True)
                            )
                            discovered += 1
                            unit_entries += 1
                            if (
                                is_dir
                                and kind == "Unit"
                                and path not in pending
                                and path not in attempted
                            ):
                                if len(pending) < MAX_PAGE_ENTRIES:
                                    pending.append(path)
                        else:
                            more = True
                        if _state(os.fstat(directory_fd)) != _state(directory_before):
                            raise UnsafeCapture("directory_changed")
                    except (OSError, UnsafeCapture) as error:
                        issues.add(
                            str(error)
                            if isinstance(error, UnsafeCapture)
                            else "directory_io"
                        )
                        complete = more = False
                    if issues:
                        complete = more = False
                    if not more:
                        if iterator is not None:
                            iterator.close()
                            iterator = None
                        if directory_context is not None:
                            try:
                                directory_context.__exit__(None, None, None)
                            except (OSError, UnsafeCapture):
                                issues.add("directory_changed")
                                complete = False
                            finally:
                                directory_context = None
                    write_frame(stdout, frame("Entries", entries=entries))
                    write_frame(
                        stdout,
                        frame(
                            "DirectoryDone",
                            directory=directory,
                            identity=identity,
                            complete=complete,
                            more=more,
                            issues=sorted(issues),
                            reads=active_reads,
                            queued=sent_bytes
                            + (SOURCE_BUFFER_HEADROOM if active_reads else 0),
                        ),
                    )
                    if (
                        more
                        or inspections >= MAX_PAGE_ENTRIES
                        or unit_entries >= MAX_PAGE_ENTRIES
                        or sent_bytes > PAGE_TEXT_BYTES - MAX_FILE_BYTES
                        or metadata_bytes > 512 * 1024
                        or "entry_limit" in issues
                    ):
                        break
                    continuation = False
                if kind == "Unit":
                    write_frame(
                        stdout,
                        frame(
                            "UnitDone",
                            inspections=inspections,
                            attempts=len(attempted),
                            entries=unit_entries,
                        ),
                    )
    except (OSError, ValueError, WorkspaceRootPinError):
        try:
            write_frame(stdout, frame("Error", code="scan_unavailable"))
        except (OSError, ValueError):
            pass
        return 2
    finally:
        if iterator is not None:
            iterator.close()
        if directory_context is not None:
            directory_context.__exit__(RuntimeError, RuntimeError("scan_abort"), None)


if __name__ == "__main__":
    # The raw inherited reference is deliberately never closed/unlocked by
    # Python cleanup. Native exit retires it, even before Init/containment.
    if len(sys.argv) == 2:
        from tldw_chatbook.Workspaces.project_index.lifetime import retain_inherited

        _lifetime_capability = retain_inherited(sys.argv[1])
    elif len(sys.argv) != 1:
        raise SystemExit(2)
    raise SystemExit(run_project_scan_worker(sys.stdin.buffer, sys.stdout.buffer))
