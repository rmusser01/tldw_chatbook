"""Private local-file lifecycle primitives.

This module is deliberately dependency-leaf: callers choose failure policy and
diagnostics while this module performs lexical selection and filesystem checks.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import stat
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, TextIO, TypeAlias

from tldw_chatbook.Utils.platform_files import os

PathInput: TypeAlias = str | os.PathLike[str]

_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIRECTORY_MODE = 0o700
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_NOCTTY = getattr(os, "O_NOCTTY", 0)
_PRIVATE_FILE_OPEN_FLAGS = os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _NOCTTY
_WINDOWS_PLATFORM = os.name == "nt"
# Matches the usual kernel ELOOP budget, so a symlink cycle terminates instead
# of walking forever.
_MAX_TRUSTED_SYMLINK_HOPS = 8
_ORIGINAL_NATIVE_OPEN = os.open


class PrivatePathStatus(StrEnum):
    CREATED_PRIVATE = "created_private"
    HARDENED_PRIVATE = "hardened_private"
    ALREADY_PRIVATE = "already_private"
    UNSAFE_PARENT = "unsafe_parent"
    WRONG_OWNER = "wrong_owner"
    LINK_OR_NON_REGULAR = "link_or_non_regular"
    OPERATION_FAILED = "operation_failed"
    TRUSTED_DIRECTORY = "trusted_directory"
    UNVERIFIED_PLATFORM = "unverified_platform"


@dataclass(frozen=True)
class PrivatePathResult:
    """Outcome of a private-path lifecycle check.

    Attributes:
        lexical_path: The normalized absolute path the caller selected. It is
            lexical on purpose: nothing here resolves symlinks.
        status: The posture decision (see ``PrivatePathStatus``).
        reason: Machine-stable refusal identifier, or ``None`` on success.
        offender_path: Diagnostics only (task-32900) -- best-effort path of
            the existing directory whose mode/owner failed a walk check.
            Never consulted by trust decisions, and absent when the walk
            could not attribute the refusal to a component.
        offender_detail: Diagnostics-only human-readable mode/owner summary
            of ``offender_path``, when the refusing entry was stat'd.
        offender_mode: Diagnostics-only ``stat.S_IMODE`` value of the
            refusing entry, when it was stat'd; lets callers reason about
            the write bits without parsing ``offender_detail``.
    """

    lexical_path: Path
    status: PrivatePathStatus
    reason: str | None = None
    offender_path: Path | None = None
    offender_detail: str | None = None
    offender_mode: int | None = None

    @property
    def verified_private(self) -> bool:
        return self.status in {
            PrivatePathStatus.CREATED_PRIVATE,
            PrivatePathStatus.HARDENED_PRIVATE,
            PrivatePathStatus.ALREADY_PRIVATE,
        }

    @property
    def usable(self) -> bool:
        return self.verified_private or self.status in {
            PrivatePathStatus.TRUSTED_DIRECTORY,
            PrivatePathStatus.UNVERIFIED_PLATFORM,
        }


class PrivatePathError(OSError):
    """Raised when a private-path lifecycle check refuses or fails.

    Attributes:
        result: The bounded ``PrivatePathResult`` decision. It carries the
            status, refusal reason, and any diagnostics-only offender
            information; there is no chained exception state to inspect.
    """

    def __init__(self, result: PrivatePathResult) -> None:
        self.result = result
        reason = f": {result.reason}" if result.reason else ""
        offender = ""
        if result.offender_path is not None:
            detail = f" ({result.offender_detail})" if result.offender_detail else ""
            offender = f" [{result.offender_path}{detail}]"
        super().__init__(f"{result.status.value}{reason}{offender}")


@dataclass
class PrivateBinaryFile:
    """A pinned binary stream paired with its verified privacy posture."""

    stream: BinaryIO
    result: PrivatePathResult


@dataclass(frozen=True)
class PrivateFileWritePrecondition:
    """Pinned target identity required by a later atomic replacement."""

    target_identity: tuple[int, int] | None

    def __post_init__(self) -> None:
        identity = self.target_identity
        if identity is not None and (
            type(identity) is not tuple
            or len(identity) != 2
            or any(type(part) is not int or part < 0 for part in identity)
        ):
            raise ValueError("target_identity must be a device/inode pair or None")

    @classmethod
    def from_opened(cls, opened: PrivateBinaryFile) -> PrivateFileWritePrecondition:
        """Capture the identity of a pinned private file."""

        if not isinstance(opened, PrivateBinaryFile):
            raise TypeError("opened must be PrivateBinaryFile")
        opened_stat = os.fstat(opened.stream.fileno())
        return cls((opened_stat.st_dev, opened_stat.st_ino))

    @classmethod
    def missing(cls) -> PrivateFileWritePrecondition:
        """Require the atomic replacement target to remain absent."""

        return cls(None)


def lexical_path(path: PathInput) -> Path:
    raw = os.fspath(path)
    if "\x00" in raw:
        raise ValueError("Path must not contain NUL")
    expanded = os.path.expanduser(raw)
    return Path(os.path.abspath(os.path.normpath(expanded)))


def _posix_guards_available() -> bool:
    # `os.readlink` joined this set when trusted-symlink traversal was added:
    # the walk now depends on it being descriptor-relative, and a platform
    # without that support must fall through to the unverified-platform
    # branch rather than raising TypeError out of the walk.
    required_dir_fd = {os.open, os.stat, os.mkdir, os.readlink}
    return (
        os.name in {"posix", "nt"}
        and _NOFOLLOW != 0
        and _NONBLOCK != 0
        and _NOCTTY != 0
        and getattr(os, "O_DIRECTORY", 0) != 0
        and required_dir_fd.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
        and hasattr(os, "geteuid")
        and hasattr(os, "fstat")
        and hasattr(os, "fchmod")
        and hasattr(os, "fsync")
    )


def _atomic_posix_guards_available() -> bool:
    return _posix_guards_available() and {
        os.rename,
        os.unlink,
    }.issubset(os.supports_dir_fd)


def _classify_private_file_stat(
    file_stat: os.stat_result,
    *,
    expected_uid: int,
) -> PrivatePathStatus | None:
    if not stat.S_ISREG(file_stat.st_mode):
        return PrivatePathStatus.LINK_OR_NON_REGULAR
    if file_stat.st_nlink != 1:
        return PrivatePathStatus.LINK_OR_NON_REGULAR
    if file_stat.st_uid != expected_uid:
        return PrivatePathStatus.WRONG_OWNER
    return None


def _trusted_directory_owner(directory_stat: os.stat_result, euid: int) -> bool:
    return directory_stat.st_uid in {0, euid}


def _open_directory_component(parent_fd: int, component: str, *, _open=None) -> int:
    return (_open or _native_open)(
        component,
        _DIRECTORY_OPEN_FLAGS | _NOFOLLOW,
        dir_fd=parent_fd,
    )


def _trusted_symlink(link_stat: os.stat_result) -> bool:
    """Report whether a symlink component may be traversed.

    Only *root-owned* symlinks qualify, and only when no other user could have
    written them. This is deliberately narrower than `_trusted_directory_owner`,
    which also accepts the effective uid: a directory owned by the caller is the
    caller's own storage, whereas following a caller-owned symlink would silently
    relocate the application's private files behind a lexical alias. Platform
    symlinks such as macOS `/var -> private/var` are root-owned, so this is the
    smallest rule that lets the walk reach the system temporary directory.

    Note that Linux reports every symlink as mode 0o777, so the write-bit gate
    rejects all symlinks there. That is intentional: Linux has no root-owned
    symlink on the paths this module walks, so no traversal is needed.
    """

    if link_stat.st_uid != 0:
        return False
    return not bool(stat.S_IMODE(link_stat.st_mode) & 0o022)


def _read_trusted_symlink(
    parent_fd: int,
    component: str,
    selected: Path,
    exc: OSError,
    traversed: list[str],
) -> str:
    """Return a trusted symlink component's target, or re-raise the open error.

    Both the `lstat` and the `readlink` go through `dir_fd`, so the component is
    never re-derived from a path string that an attacker could repoint between
    the two calls. Refusals carry the component as `offender_path` (plus its
    lstat detail when the link itself was the problem) so symlink rejections
    name their blocker like every other refusal (task-32900).
    """

    offender = _offender_path(traversed, component)
    if exc.errno not in {errno.ELOOP, errno.ENOTDIR}:
        raise _private_path_error_from_oserror(
            selected, exc, offender_path=offender
        ) from None
    try:
        link_stat = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise _private_path_error_from_oserror(
            selected, exc, offender_path=offender
        ) from None
    if not stat.S_ISLNK(link_stat.st_mode) or not _trusted_symlink(link_stat):
        raise _private_path_error_from_oserror(
            selected,
            exc,
            offender_path=offender,
            offender_detail=_describe_stat(link_stat),
            offender_mode=_offender_mode(link_stat),
        ) from None
    # `TypeError` as well as `OSError`: `os.readlink` raises TypeError, not
    # OSError, on a build where it does not accept `dir_fd`. The probe above
    # should stop us reaching here on such a platform, but a guard that
    # crashes rather than refusing is the wrong failure mode -- fail closed.
    try:
        return os.readlink(component, dir_fd=parent_fd)
    except (OSError, TypeError):
        raise _private_path_error_from_oserror(
            selected, exc, offender_path=offender
        ) from None


def _symlink_walk_components(target: str) -> tuple[bool, list[str]]:
    absolute = target.startswith(os.sep)
    components = [part for part in target.split(os.sep) if part not in {"", os.curdir}]
    # A target that names its own directory ("." or "/") must still produce one
    # walk step so the per-component trust checks run on the resulting directory.
    return absolute, components or [os.curdir]


def _describe_stat(entry_stat: os.stat_result | None) -> str | None:
    """Best-effort human-readable mode/owner for refusal diagnostics."""

    if entry_stat is None:
        return None
    mode = stat.S_IMODE(entry_stat.st_mode)
    owner = f"uid={entry_stat.st_uid}"
    group = f"gid={entry_stat.st_gid}"
    try:  # Names are a display nicety; unknown ids must not break the refusal.
        import pwd

        owner = f"{pwd.getpwuid(entry_stat.st_uid).pw_name} ({owner})"
    except Exception:
        pass
    try:
        import grp

        group = f"{grp.getgrgid(entry_stat.st_gid).gr_name} ({group})"
    except Exception:
        pass
    return (
        f"mode {stat.filemode(entry_stat.st_mode)} ({oct(mode)}), "
        f"owner {owner}, group {group}"
    )


def _offender_mode(entry_stat: os.stat_result | None) -> int | None:
    """Diagnostics-only permission bits of a refusing entry."""

    if entry_stat is None:
        return None
    return stat.S_IMODE(entry_stat.st_mode)


def _offender_path(traversed: list[str], *extra: str) -> Path:
    """Best-effort lexical path of the directory a walk refused."""

    return Path(os.sep, *traversed, *extra)


def _follow_trusted_symlink(
    *,
    current_fd: int,
    component: str,
    pending: list[str],
    hops: int,
    selected: Path,
    exc: OSError,
    traversed: list[str],
    _close: Callable[[int], None] | None = None,
    _open: Callable[..., int] | None = None,
) -> tuple[int, int]:
    """Splice a trusted symlink's target into the pending walk.

    Returns the descriptor the walk should continue from and the new hop count.
    `current_fd` is only replaced once its successor is open, so the caller
    always has exactly one descriptor to close on failure. `traversed` is the
    caller's diagnostics breadcrumb; an absolute splice restarts the walk at
    the root, so it is cleared to keep the reported path honest.
    """

    # Classify first, so an unrelated open failure keeps reporting its own cause
    # even once symlinks have been followed.
    target = _read_trusted_symlink(current_fd, component, selected, exc, traversed)
    if hops >= _MAX_TRUSTED_SYMLINK_HOPS:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.LINK_OR_NON_REGULAR,
                reason="symlink_hop_limit_exceeded",
                offender_path=_offender_path(traversed, component),
            )
        )
    absolute, components = _symlink_walk_components(target)
    pending[:0] = components
    if not absolute:
        return current_fd, hops + 1
    traversed.clear()
    root_fd = (_open or _native_open)(
        selected.anchor, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW
    )
    (_close or _native_close)(current_fd)
    return root_fd, hops + 1


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _private_file_postcondition_holds(
    file_fd: int,
    parent_fd: int,
    leaf: str,
    *,
    expected_identity: os.stat_result,
) -> bool:
    opened = os.fstat(file_fd)
    entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    return (
        _same_identity(opened, expected_identity)
        and _same_identity(entry, expected_identity)
        and stat.S_ISREG(opened.st_mode)
        and opened.st_nlink == 1
        and entry.st_nlink == 1
        and opened.st_uid == os.geteuid()
        and stat.S_IMODE(opened.st_mode) == _PRIVATE_FILE_MODE
    )


def _private_directory_postcondition_holds(
    directory_fd: int,
    parent_fd: int,
    component: str,
    *,
    expected_identity: os.stat_result,
) -> bool:
    opened = os.fstat(directory_fd)
    entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    return (
        _same_identity(opened, expected_identity)
        and _same_identity(entry, expected_identity)
        and stat.S_ISDIR(opened.st_mode)
        and opened.st_uid == os.geteuid()
        and stat.S_IMODE(opened.st_mode) == _PRIVATE_DIRECTORY_MODE
    )


def _trusted_directory_postcondition_holds(
    directory_fd: int,
    parent_fd: int,
    component: str,
    *,
    expected_identity: os.stat_result,
) -> bool:
    opened = os.fstat(directory_fd)
    entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    return (
        _same_identity(opened, expected_identity)
        and _same_identity(entry, expected_identity)
        and stat.S_ISDIR(opened.st_mode)
        and _trusted_directory_owner(opened, os.geteuid())
    )


def _walk_verified_parent(
    selected: Path,
    *,
    missing_leaf_allowed: bool,
    _close: Callable[[int], None] | None = None,
    _open: Callable[..., int] | None = None,
) -> tuple[int, str]:
    close = _close or _native_close
    parts = selected.parts
    if len(parts) < 2 or parts[0] != selected.anchor:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="invalid_absolute_path",
            )
        )

    euid = os.geteuid()
    current_fd = (_open or _native_open)(
        selected.anchor, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW
    )
    try:
        current_stat = os.fstat(current_fd)
        pending = list(parts[1:-1])
        traversed: list[str] = []
        symlink_hops = 0
        while pending:
            component = pending.pop(0)
            if not _trusted_directory_owner(current_stat, euid):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="untrusted_directory_owner",
                        offender_path=_offender_path(traversed),
                        offender_detail=_describe_stat(current_stat),
                        offender_mode=_offender_mode(current_stat),
                    )
                )
            current_mode = stat.S_IMODE(current_stat.st_mode)
            current_writable = bool(current_mode & 0o022)
            current_sticky = bool(current_mode & stat.S_ISVTX)
            if current_writable and not current_sticky:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="shared_writable_parent",
                        offender_path=_offender_path(traversed),
                        offender_detail=_describe_stat(current_stat),
                        offender_mode=_offender_mode(current_stat),
                    )
                )
            try:
                next_fd = _open_directory_component(
                    current_fd,
                    component,
                    **({"_open": _open} if _open is not None else {}),
                )
            except FileNotFoundError:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="missing_parent",
                        offender_path=_offender_path(traversed),
                    )
                ) from None
            except OSError as exc:
                current_fd, symlink_hops = _follow_trusted_symlink(
                    current_fd=current_fd,
                    component=component,
                    pending=pending,
                    hops=symlink_hops,
                    selected=selected,
                    exc=exc,
                    traversed=traversed,
                    _close=_close,
                    _open=_open,
                )
                current_stat = os.fstat(current_fd)
                continue

            transferred = False
            try:
                next_stat = os.fstat(next_fd)
                if not stat.S_ISDIR(next_stat.st_mode):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.LINK_OR_NON_REGULAR,
                            reason="non_directory_parent",
                            offender_path=_offender_path(traversed, component),
                        )
                    )
                if not _trusted_directory_owner(next_stat, euid):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="untrusted_directory_owner",
                            offender_path=_offender_path(traversed, component),
                            offender_detail=_describe_stat(next_stat),
                            offender_mode=_offender_mode(next_stat),
                        )
                    )
                old_fd = current_fd
                current_fd = next_fd
                transferred = True
                close(old_fd)
                current_stat = next_stat
                traversed.append(component)
            finally:
                if not transferred:
                    close(next_fd)

        if not _trusted_directory_owner(current_stat, euid):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.UNSAFE_PARENT,
                    reason="untrusted_directory_owner",
                    offender_path=_offender_path(traversed),
                    offender_detail=_describe_stat(current_stat),
                    offender_mode=_offender_mode(current_stat),
                )
            )
        final_parent_mode = stat.S_IMODE(current_stat.st_mode)
        final_parent_writable = bool(final_parent_mode & 0o022)
        final_parent_sticky = bool(final_parent_mode & stat.S_ISVTX)
        if final_parent_writable and (not final_parent_sticky or missing_leaf_allowed):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.UNSAFE_PARENT,
                    reason=(
                        "missing_leaf_in_shared_sticky_parent"
                        if final_parent_sticky and missing_leaf_allowed
                        else "shared_writable_parent"
                    ),
                    offender_path=_offender_path(traversed),
                    offender_detail=_describe_stat(current_stat),
                    offender_mode=_offender_mode(current_stat),
                )
            )
        return current_fd, parts[-1]
    except BaseException:
        close(current_fd)
        raise


def _private_path_error_from_oserror(
    selected: Path,
    exc: OSError,
    *,
    offender_path: Path | None = None,
    offender_detail: str | None = None,
    offender_mode: int | None = None,
) -> PrivatePathError:
    status = (
        PrivatePathStatus.LINK_OR_NON_REGULAR
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}
        else PrivatePathStatus.OPERATION_FAILED
    )
    return PrivatePathError(
        PrivatePathResult(
            selected,
            status,
            reason=type(exc).__name__,
            offender_path=offender_path,
            offender_detail=offender_detail,
            offender_mode=offender_mode,
        )
    )


def _open_leaf_for_create(parent_fd: int, leaf: str) -> int:
    return _native_open(
        leaf,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
        _PRIVATE_FILE_MODE,
        dir_fd=parent_fd,
    )


def _runtime_operation(path=None):
    raw = sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants")
    return raw._runtime_operation(path) if raw is not None else None


def _active_visual_source():
    for name in ("persona_visual_participants", "visual_identity_participants"):
        visual = sys.modules.get("tldw_chatbook.Backup_Recovery." + name)
        # A concurrent cold import publishes the module before its thread-local
        # state exists. No visual operation can be active at that point.
        state = getattr(getattr(visual, "_local", None), "state", None)
        if state is not None:
            return visual, state
    return None, None


def _visual_native_scope():
    visual, state = _active_visual_source()
    return (
        (visual, state)
        if state is not None and state.helper is not None
        else (None, None)
    )


def _voice_operation(*, native=False):
    module = sys.modules.get("tldw_chatbook.TTS.loose_voice_lifetime")
    return (
        (module.native() if native else module.active()) if module is not None else None
    )


def _native_mkdir(*args, **kwargs):
    visual, state = _visual_native_scope()
    if state is not None:
        return visual.mkdir(*args, **kwargs)
    return os.mkdir(*args, **kwargs)


@dataclass
class _NativeOpenOutcome:
    """Per-call native allocation result; no source or permission authority."""

    descriptor: int | None = None
    rejected: bool = False


def _native_open(*args, _outcome: _NativeOpenOutcome | None = None, **kwargs):
    visual, state = _visual_native_scope()
    if state is not None:
        fd = visual.native_open(*args, **kwargs)
        if _outcome is not None:
            _outcome.descriptor = fd
        return fd
    operation = _runtime_operation()
    return _allocate_native_descriptor(operation, args, kwargs, _outcome)


def _allocate_native_descriptor(operation, args, kwargs, _outcome):
    open = os.open  # Pin the exact callable used at this native edge.
    try:
        fd = open(*args, **kwargs)
    except OSError:
        # Only the original native primitive's actual rejection proves absence.
        # A substituted provider can allocate and raise the same exception.
        if _outcome is not None and open is _ORIGINAL_NATIVE_OPEN:
            _outcome.rejected = True
        raise
    if _outcome is not None:
        _outcome.descriptor = fd
    voice = _voice_operation(native=True)
    if voice is not None:
        voice.opened(fd)
    if operation is not None:
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        raw._states[operation].descriptors.add(fd)
    return fd


def _native_close(fd):
    voice = _voice_operation(native=True)
    if voice is not None:
        return voice.close(fd)
    visual, state = _visual_native_scope()
    if state is not None:
        return visual.native_close(fd)
    raw = sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants")
    state = raw._owned_descriptor_retirement_state(fd) if raw is not None else None
    if state is not None:
        return raw._close_descriptor(state, fd)
    operation = _runtime_operation()
    if operation is None:
        os.close(fd)
    else:
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        raw._close_descriptor(raw._states[operation], fd)


# Capture definitions once; optional module metadata cannot restamp this witness.
_PARENT_WALK_SOURCE = tuple(
    (
        name, function, getattr(function, "__code__", None),
        getattr(function, "__defaults__", None),
        None if getattr(function, "__kwdefaults__", None) is None else tuple(function.__kwdefaults__.items()),
        getattr(function, "__globals__", None),
        None if getattr(function, "__closure__", None) is None else tuple(cell.cell_contents for cell in function.__closure__),
    )
    for name, function in (
        ("_walk_verified_parent", _walk_verified_parent),
        ("_open_directory_component", _open_directory_component),
        ("_follow_trusted_symlink", _follow_trusted_symlink),
        ("_native_open", _native_open),
        ("_native_close", _native_close),
        ("_allocate_native_descriptor", _allocate_native_descriptor),
        ("_runtime_operation", _runtime_operation),
        ("_visual_native_scope", _visual_native_scope),
        ("_voice_operation", _voice_operation),
        ("_posix_guards_available", _posix_guards_available),
        ("os.open", _ORIGINAL_NATIVE_OPEN),
    )
)


def _bind_parent_walk(definition):
    namespace = globals()
    native_os = os
    directory_flags = _DIRECTORY_OPEN_FLAGS | _NOFOLLOW

    def source_current():
        if namespace.get("_PARENT_WALK_SOURCE") is not definition or namespace.get("os") is not native_os:
            return False
        if _DIRECTORY_OPEN_FLAGS | _NOFOLLOW != directory_flags:
            return False
        for name, function, code, defaults, keywords, defining_globals, closure in definition:
            actual = native_os.open if name == "os.open" else namespace.get(name)
            if (
                actual is not function or getattr(function, "__code__", None) is not code
                or getattr(function, "__defaults__", None) is not defaults
                or getattr(function, "__globals__", None) is not defining_globals
            ):
                return False
            current_keywords = getattr(function, "__kwdefaults__", None)
            if keywords is None:
                if current_keywords is not None:
                    return False
            elif type(current_keywords) is not dict or len(current_keywords) != len(keywords) or any(
                key not in current_keywords or current_keywords[key] is not value
                for key, value in keywords
            ):
                return False
            cells = getattr(function, "__closure__", None)
            if closure is None:
                if cells is not None:
                    return False
            elif cells is None or len(cells) != len(closure) or any(
                cell.cell_contents is not value for cell, value in zip(cells, closure, strict=True)
            ):
                return False
        return True

    @contextlib.contextmanager
    def _prepared_parent_walk(selected, open, close):
        raw = sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants")
        selector = getattr(raw, "_parent_walk_operation", None)
        allocation_check = getattr(raw, "_check_directory_allocation", None)
        if (
            open is not None or close is not None
            or source_current.__code__ is not current_code
            or source_current.__globals__ is not namespace
            or source_current.__defaults__ is not None
            or source_current.__kwdefaults__ is not None
            or source_current.__closure__ is not current_cells
            or any(cell.cell_contents is not value for cell, value in current_values)
            or not source_current()
            or not _posix_guards_available()
            or _voice_operation() is not None
            or _visual_native_scope()[1] is not None
            or selector is None or allocation_check is None
        ):
            yield open, close, None
            return
        raw_callbacks = tuple(
            (name, callback, callback.__code__, callback.__defaults__,
             None if callback.__kwdefaults__ is None else tuple(callback.__kwdefaults__.items()),
             callback.__globals__, callback.__closure__,
             tuple((cell, cell.cell_contents) for cell in callback.__closure__ or ()))
            for name, callback in (("_parent_walk_operation", selector), ("_check_directory_allocation", allocation_check))
        )
        owner = selector(selected)
        if owner is None:
            yield open, close, None
            return
        operation, state, source, participant, pin = owner
        active, entered = True, False

        def current():
            if (
                not active
                or source_current.__code__ is not current_code
                or source_current.__globals__ is not namespace
                or source_current.__defaults__ is not None
                or source_current.__kwdefaults__ is not None
                or source_current.__closure__ is not current_cells
                or any(cell.cell_contents is not value for cell, value in current_values)
                or not source_current()
                or sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants") is not raw
                or _voice_operation() is not None
                or _visual_native_scope()[1] is not None
                or getattr(raw._local, "operation", None) is not operation
            ):
                raise raw.bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
            for name, callback, code, defaults, keywords, defining_globals, cells, values in raw_callbacks:
                if (
                    vars(raw).get(name) is not callback or callback.__code__ is not code
                    or callback.__defaults__ is not defaults or callback.__globals__ is not defining_globals
                    or callback.__closure__ is not cells
                    or any(cell.cell_contents is not value for cell, value in values)
                ):
                    raise raw.bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
                actual = callback.__kwdefaults__
                if (keywords is None and actual is not None) or (
                    keywords is not None and (
                        type(actual) is not dict or len(actual) != len(keywords)
                        or any(key not in actual or actual[key] is not value for key, value in keywords)
                    )
                ):
                    raise raw.bootstrap.RecoveryRequired("raw_operation_provenance_invalid")

        def directory_open(path, flags, *, dir_fd=None):
            nonlocal entered
            current()
            if type(flags) is not int or flags != directory_flags:
                raise raw.bootstrap.RecoveryRequired("raw_path_outside_scope")
            if not entered:
                if _runtime_operation() is not operation:
                    raise raw.bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
                entered = True
            allocation_check(operation, state, source, participant)
            current()
            kwargs = {} if dir_fd is None else {"dir_fd": dir_fd}
            return _allocate_native_descriptor(operation, (path, flags), kwargs, None)

        def owned_close(fd):
            # Keep creator cleanup independent of revoked source metadata.
            if not active or raw._owned_descriptor_retirement_state(fd) is not state:
                raise raw.bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
            raw._close_descriptor(state, fd)

        def finish(parent):
            current()
            checked = raw._check(operation, writing=True)
            if checked is not state or state.source is not source or state.participant is not participant:
                raise raw.bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
            if type(state.pins.get(selected.parent)) is not int or state.pins[selected.parent] != pin:
                raise raw.bootstrap.RecoveryRequired("raw_parent_identity_changed")
            if type(parent) is not int or parent not in state.descriptors:
                raise raw.bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
            opened, pinned = os.fstat(parent), os.fstat(pin)
            if (opened.st_dev, opened.st_ino) != (pinned.st_dev, pinned.st_ino):
                raise raw.bootstrap.RecoveryRequired("raw_parent_identity_changed")
            allocation_check(operation, state, source, participant)
            current()
            if type(state.pins.get(selected.parent)) is not int or state.pins[selected.parent] != pin:
                raise raw.bootstrap.RecoveryRequired("raw_parent_identity_changed")
            if type(parent) is not int or parent not in state.descriptors:
                raise raw.bootstrap.RecoveryRequired("raw_operation_provenance_invalid")

        try:
            current()
            yield directory_open, owned_close, finish
        finally:
            active = False

    def _open_verified_parent(
        selected: Path,
        *,
        missing_leaf_allowed: bool,
        _close: Callable[[int], None] | None = None,
        _open: Callable[..., int] | None = None,
    ) -> tuple[int, str]:
        # Invalid input and changed preparation retain the original error route.
        parts = selected.parts
        if len(parts) < 2 or parts[0] != selected.anchor or any(
            function.__code__ is not code or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not keywords or function.__globals__ is not defining_globals
            or function.__closure__ is not cells
            or any(cell.cell_contents is not value for cell, value in values)
            for function, code, defaults, keywords, defining_globals, cells, values in preparation
        ) or namespace.get("_prepared_parent_walk") is not _prepared_parent_walk:
            return _walk_verified_parent(selected, missing_leaf_allowed=missing_leaf_allowed, _close=_close, _open=_open)
        with _prepared_parent_walk(selected, _open, _close) as (open, close, finish):
            parent, leaf = _walk_verified_parent(
                selected, missing_leaf_allowed=missing_leaf_allowed, _close=close, _open=open,
            )
            try:
                if finish is not None:
                    finish(parent)
                return parent, leaf
            except BaseException:
                (close or _native_close)(parent)
                raise

    current_code = source_current.__code__
    current_cells = source_current.__closure__
    current_values = tuple((cell, cell.cell_contents) for cell in current_cells or ())
    preparation = tuple(
        (function, function.__code__, function.__defaults__, function.__kwdefaults__,
         function.__globals__, function.__closure__,
         tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()))
        for function in (_prepared_parent_walk, _prepared_parent_walk.__wrapped__)
    )
    return _open_verified_parent, _prepared_parent_walk


_open_verified_parent, _prepared_parent_walk = _bind_parent_walk(_PARENT_WALK_SOURCE)
del _bind_parent_walk


def _close_runtime_stream(operation, stream):
    """Retire a source stream only after an explicitly successful close."""
    raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
    state = raw._states[operation]
    try:
        stream.close()
        if not stream.closed:
            raise RuntimeError("raw_resources_not_retired")
    except BaseException:
        state.uncertain = True
        raise
    state.files.remove(stream)


def _admitted_file(function):
    @functools.wraps(function)
    def admitted(path, *args, **kwargs):
        voice = _voice_operation()
        if voice is not None:
            voice.check(path, external=True)
            with voice.native_scope():
                return function(path, *args, **kwargs)
        from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

        visual, visual_state = _active_visual_source()
        if visual_state is not None:
            if function.__name__ != "secure_private_directory":
                raise RuntimeError("persona_visual_helper_not_supported")
            state = visual.private_directory(path)
            previous_helper = state.helper
            state.helper = lexical_path(path)
            try:
                return function(path, *args, **kwargs)
            finally:
                state.helper = previous_helper
        operation = _runtime_operation(lexical_path(path))
        if operation is not None:
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            state = raw._check(operation)
            config = sys.modules.get("tldw_chatbook.config")
            if raw.mcp_sources.history_operation(state):
                if not raw.mcp_sources.helper_allowed(state, function.__name__, path):
                    raise RuntimeError("raw_source_helper_not_supported")
                if function.__name__ == "atomic_private_write_bytes":
                    raw.mcp_sources.check_destination(state, path)
                result = function(path, *args, **kwargs)
                if function.__name__ == "atomic_private_write_bytes":
                    raw.mcp_sources.published(state, path)
                return result
            if state.route == "hook_permissions":
                selected = lexical_path(path)
                allowed = (
                    function.__name__ == "secure_private_directory"
                    and selected == state.selected.parent
                    or function.__name__ == "create_private_text"
                    and selected
                    == state.selected.with_name(state.selected.name + ".lock")
                    or function.__name__ == "atomic_private_write_bytes"
                    and selected == state.selected
                    and state.temporary is not None
                )
                if not allowed:
                    raise RuntimeError("raw_source_helper_not_supported")
            elif config is not None and state.source is config:
                selected = lexical_path(path)
                allowed = (
                    function.__name__ == "secure_private_directory"
                    and selected in state.directories
                    or function.__name__ == "create_private_text"
                    and selected
                    in (
                        state.selected,
                        state.selected.with_name(state.selected.name + ".lock"),
                    )
                    or function.__name__ == "atomic_private_write_bytes"
                    and selected in state.temporaries
                )
                if not allowed:
                    raise RuntimeError("raw_source_helper_not_supported")
            else:
                expected = (
                    state.source.application_owned_directory
                    if function.__name__ == "secure_private_directory"
                    else state.selected
                )
                if (
                    function.__name__
                    not in {"atomic_private_write_bytes", "secure_private_directory"}
                    or expected is None
                    or lexical_path(path) != lexical_path(expected)
                    or (
                        function.__name__ == "atomic_private_write_bytes"
                        and state.temporary is None
                    )
                ):
                    raise RuntimeError("raw_source_helper_not_supported")
            return function(path, *args, **kwargs)
        with acquire_storage(lexical_path(path)):
            return function(path, *args, **kwargs)

    return admitted


class _AdmittedStream:
    def __init__(self, stream, lease):
        self._stream = stream
        self._lease = lease

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._stream.close()
        self._lease.close()

    def __iter__(self):
        return iter(self._stream)

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass


class _ConfigStream:
    """A config lock stream whose native lifetime has no GC retirement path."""

    def __init__(self, stream, operation):
        self._stream = stream
        self._operation = operation
        self._fd = stream.fileno()
        self._retired = False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def close(self):
        if self._retired:
            return
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        raw._check(self._operation)
        _close_runtime_stream(self._operation, self._stream)
        _native_close(self._fd)
        self._retired = True


class _MCPAppendStream(_ConfigStream):
    """The exact history append stream retains FD ownership through close."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, value):
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        state = raw._check(self._operation, writing=True)
        state.mcp_effects = True
        return self._stream.write(value)

    def close(self):
        if self._retired:
            return
        info = os.fstat(self._fd)
        super().close()
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        state = raw._check(self._operation)
        state.mcp_publications[state.selected] = (info.st_dev, info.st_ino)
        raw.mcp_sources.published(state, state.selected)


def _operation_temporary(operation, selected):
    raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
    state = raw._check(operation, selected, writing=True)
    return state.temporaries.get(selected, state.temporary)


def _admitted_stream(function):
    @functools.wraps(function)
    def admitted(path, *args, **kwargs):
        from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

        operation = _runtime_operation()
        if operation is not None:
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            state = raw._check(operation, lexical_path(path), writing=True)
            config = sys.modules.get("tldw_chatbook.config")
            if raw.mcp_sources.history_operation(state):
                if not raw.mcp_sources.helper_allowed(state, function.__name__, path):
                    raise RuntimeError("raw_source_helper_not_supported")
                raw.mcp_sources.check_destination(state, path)
                return _MCPAppendStream(function(path, *args, **kwargs), operation)
            if (
                state.route != "hook_permissions" and state.source is not config
            ) or lexical_path(path) != (
                state.selected
                if state.route == "config_data_lock"
                else state.selected.with_name(state.selected.name + ".lock")
            ):
                raise RuntimeError("raw_source_helper_not_supported")
            return _ConfigStream(function(path, *args, **kwargs), operation)
        lease = acquire_storage(lexical_path(path))
        try:
            return _AdmittedStream(function(path, *args, **kwargs), lease)
        except BaseException:
            lease.close()
            raise

    return admitted


def _admitted_reader(function):
    @functools.wraps(function)
    def admitted(path, *args, **kwargs):
        voice = _voice_operation()
        if voice is not None:
            with voice.native_scope():
                yield from function(path, *args, **kwargs)
            return
        from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

        operation = _runtime_operation(lexical_path(path))
        if operation is not None:
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            state = raw._check(operation)
            config = sys.modules.get("tldw_chatbook.config")
            if raw.mcp_sources.history_operation(state):
                if not raw.mcp_sources.helper_allowed(state, function.__name__, path):
                    raise RuntimeError("raw_source_helper_not_supported")
                raw.mcp_sources.check_destination(state, path)
                yield from function(path, *args, **kwargs)
                return
            allowed = (state.selected,)
            if config is not None and state.source is config:
                allowed += (config._advanced_backup_path(state.selected),)
            if lexical_path(path) not in allowed:
                raise RuntimeError("raw_source_helper_not_supported")
            yield from function(path, *args, **kwargs)
            return
        with acquire_storage(lexical_path(path)):
            yield from function(path, *args, **kwargs)

    return admitted


@_admitted_file
def create_private_text(
    path: PathInput,
    text: str,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
) -> PrivatePathResult:
    """Create a new private text file without replacing an existing entry."""

    selected = lexical_path(path)
    payload = text.encode(encoding)
    if application_owned_directory is not None:
        owned_dir = lexical_path(application_owned_directory)
        if selected.parent != owned_dir:
            raise ValueError("Application-owned directory must be the target parent")
        secure_private_directory(
            owned_dir,
            create=True,
            application_owned=True,
        )

    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            with selected.open("xb") as handle:
                handle.write(payload)
                handle.flush()
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=True,
    )
    file_fd = -1
    try:
        file_fd = _open_leaf_for_create(parent_fd, leaf)
        created_stat = os.fstat(file_fd)
        os.fchmod(file_fd, _PRIVATE_FILE_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(file_fd, view)
            if written == 0:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="zero_byte_write",
                    )
                )
            view = view[written:]
        os.fsync(file_fd)
        if not _private_file_postcondition_holds(
            file_fd,
            parent_fd,
            leaf,
            expected_identity=created_stat,
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="private_file_postcondition_failed",
                )
            )
        return PrivatePathResult(selected, PrivatePathStatus.CREATED_PRIVATE)
    except FileExistsError:
        raise
    except PrivatePathError:
        raise
    except OSError as exc:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason=type(exc).__name__,
            )
        ) from None
    finally:
        if file_fd >= 0:
            _native_close(file_fd)
        _native_close(parent_fd)


def _prepare_application_owned_parent(
    selected: Path,
    application_owned_directory: PathInput | None,
) -> None:
    if application_owned_directory is None:
        return
    owned_dir = lexical_path(application_owned_directory)
    if selected.parent != owned_dir:
        raise ValueError("Application-owned directory must be the target parent")
    secure_private_directory(
        owned_dir,
        create=True,
        application_owned=True,
    )


@_admitted_file
def atomic_private_write_bytes(
    path: PathInput,
    payload: bytes,
    *,
    application_owned_directory: PathInput | None = None,
    target_precondition: PrivateFileWritePrecondition | None = None,
) -> PrivatePathResult:
    """Atomically replace a private file without following its target."""

    import secrets
    import tempfile

    selected = lexical_path(path)
    operation = _runtime_operation(selected)
    raw = sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants")

    if target_precondition is not None and not isinstance(
        target_precondition, PrivateFileWritePrecondition
    ):
        raise TypeError("target_precondition must be PrivateFileWritePrecondition")
    _prepare_application_owned_parent(selected, application_owned_directory)

    if not _atomic_posix_guards_available():
        if target_precondition is not None:
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="atomic_target_precondition_unavailable",
                )
            )
        if _WINDOWS_PLATFORM:
            try:
                if operation is None:
                    fd, temporary = tempfile.mkstemp(
                        dir=selected.parent,
                        prefix=f".{selected.name}.",
                        suffix=".tmp",
                    )
                else:
                    temporary = _operation_temporary(operation, selected)
                    fd = _native_open(
                        temporary,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        _PRIVATE_FILE_MODE,
                    )
                    info = os.fstat(fd)
                    raw._check(operation).created_files[temporary] = (
                        info.st_dev,
                        info.st_ino,
                    )
            except OSError as exc:
                raise _private_path_error_from_oserror(selected, exc) from None
            try:
                with os.fdopen(fd, "wb", closefd=operation is None) as stream:
                    if operation is not None:
                        raw._check(operation).files.append(stream)
                    stream.write(payload)
                    stream.flush()
                if operation is not None:
                    raw._check(operation).files.remove(stream)
                    _native_close(fd)
                    raw._replace(operation, temporary, selected)
                else:
                    os.replace(temporary, selected)
            except BaseException:
                if operation is not None:
                    state = raw._states[operation]
                    if state.files or fd in state.descriptors:
                        state.uncertain = True
                    if not state.uncertain:
                        raw._remove_temporary(operation, temporary)
                else:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
                raise
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=True,
    )
    operation = _runtime_operation(selected)
    raw = sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants")
    temporary_leaf = (
        _operation_temporary(operation, selected).name
        if operation is not None
        else f".{leaf}.{secrets.token_hex(8)}.tmp"
    )
    temporary_fd = -1
    temporary_exists = False
    published = False
    try:
        try:
            existing_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing_stat = None
            prior_mode = None

        if target_precondition is not None:
            expected_identity = target_precondition.target_identity
            if expected_identity is None:
                if existing_stat is not None:
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.OPERATION_FAILED,
                            reason="target_appeared",
                        )
                    )
            elif (
                existing_stat is None
                or (
                    existing_stat.st_dev,
                    existing_stat.st_ino,
                )
                != expected_identity
            ):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="target_replaced",
                    )
                )

        if existing_stat is not None:
            rejected = _classify_private_file_stat(
                existing_stat,
                expected_uid=os.geteuid(),
            )
            if rejected is not None:
                raise PrivatePathError(PrivatePathResult(selected, rejected))
            prior_mode = stat.S_IMODE(existing_stat.st_mode)
            existing_fd = _native_open(
                leaf,
                _PRIVATE_FILE_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
            try:
                opened_stat = os.fstat(existing_fd)
                if not _same_identity(opened_stat, existing_stat):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.OPERATION_FAILED,
                            reason="target_replaced",
                        )
                    )
                if prior_mode != _PRIVATE_FILE_MODE:
                    os.fchmod(existing_fd, _PRIVATE_FILE_MODE)
            finally:
                _native_close(existing_fd)

        temporary_fd = _open_leaf_for_create(parent_fd, temporary_leaf)
        temporary_exists = True
        temporary_stat = os.fstat(temporary_fd)
        if operation is not None:
            raw._check(operation).created_files[selected.parent / temporary_leaf] = (
                temporary_stat.st_dev,
                temporary_stat.st_ino,
            )
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            if written == 0:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="zero_byte_write",
                    )
                )
            view = view[written:]
        os.fchmod(temporary_fd, _PRIVATE_FILE_MODE)
        os.fsync(temporary_fd)

        try:
            current_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            current_stat = None
        if existing_stat is None:
            if current_stat is not None:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="target_appeared",
                    )
                )
        elif current_stat is None or not _same_identity(current_stat, existing_stat):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="target_replaced",
                )
            )

        if operation is not None:
            raw._check_temporary_identity(
                raw._check(operation), selected.parent / temporary_leaf
            )
            state = raw._check(operation)
            if raw.mcp_sources.history_operation(state):
                raw.mcp_sources.check_destination(state, selected)
        try:
            os.replace(
                temporary_leaf,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except BaseException:
            voice = _voice_operation()
            if voice is not None:
                voice.uncertain = True
            if operation is not None:
                raw._states[operation].uncertain = True
            raise
        if _voice_operation() is not None:
            _voice_operation().changed += 1
        temporary_exists = False
        published = True
        if operation is not None:
            raw._check(operation).created_files.pop(selected.parent / temporary_leaf)
        os.fsync(parent_fd)
        if not _private_file_postcondition_holds(
            temporary_fd,
            parent_fd,
            leaf,
            expected_identity=temporary_stat,
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="private_file_postcondition_failed",
                )
            )
        if operation is not None and raw.mcp_sources.history_operation(
            raw._check(operation)
        ):
            raw._states[operation].mcp_publications[selected] = (
                temporary_stat.st_dev,
                temporary_stat.st_ino,
            )
        if operation is not None:
            state = raw._check(operation)
            if state.route == "config" and state.selected == selected:
                state.config_publication = (
                    selected,
                    (temporary_stat.st_dev, temporary_stat.st_ino),
                )
        if existing_stat is None:
            status = PrivatePathStatus.CREATED_PRIVATE
        elif prior_mode != _PRIVATE_FILE_MODE:
            status = PrivatePathStatus.HARDENED_PRIVATE
        else:
            status = PrivatePathStatus.ALREADY_PRIVATE
        return PrivatePathResult(selected, status)
    except BaseException as exc:
        if published and _voice_operation() is not None:
            _voice_operation().uncertain = True
        if operation is not None and published:
            raw._states[operation].uncertain = True
        if isinstance(exc, OSError) and not isinstance(exc, PrivatePathError):
            raise _private_path_error_from_oserror(selected, exc) from None
        raise
    finally:
        if temporary_fd >= 0:
            _native_close(temporary_fd)
        try:
            if temporary_exists:
                try:
                    if operation is not None:
                        raw._remove_temporary(
                            operation, selected.parent / temporary_leaf
                        )
                    else:
                        os.unlink(temporary_leaf, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                except BaseException:
                    # The temporary was allocated by this voice publication.
                    # Failed cleanup leaves its bytes/ownership unresolved even
                    # when the write never reached rename and all fds close.
                    voice = _voice_operation(native=True)
                    if voice is not None:
                        voice.uncertain = True
                    raise
        finally:
            _native_close(parent_fd)


def atomic_private_write_text(
    path: PathInput,
    text: str,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
    target_precondition: PrivateFileWritePrecondition | None = None,
) -> PrivatePathResult:
    """Atomically replace a private text file."""

    return atomic_private_write_bytes(
        path,
        text.encode(encoding),
        application_owned_directory=application_owned_directory,
        target_precondition=target_precondition,
    )


@contextlib.contextmanager
def open_private_text_append(
    path: PathInput,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
    errors: str | None = None,
) -> Iterator[TextIO]:
    """Open a private text file for append without following its target."""

    stream = open_private_text_append_stream(
        path,
        application_owned_directory=application_owned_directory,
        encoding=encoding,
        errors=errors,
    )
    with stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())


@_admitted_stream
def open_private_text_append_stream(
    path: PathInput,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
    errors: str | None = None,
) -> TextIO:
    """Return a pinned private text stream opened for append.

    Every stream this returns writes ``\\n`` as a single LF byte on every
    platform (``newline="\\n"``). The default (``newline=None``) translates
    each ``\\n`` a caller writes into ``os.linesep``, which on Windows is
    ``\\r\\n`` -- so a caller that encodes a line, writes it as text, and then
    reasons about the file's SIZE (or diffs the file against the bytes it
    just wrote) is wrong by one byte per line there. ``MCPExecutionLog``
    does exactly that: it caches a generation's sanitized bytes keyed on
    ``st_size``, so under CRLF translation the cache was dropped on every
    append and the whole log was re-parsed and rewritten each time -- the
    exact per-tool-call cost TASK-21134's cache removed on POSIX. These are
    line-oriented JSONL/log files, so LF is also the shape every reader here
    already assumes.
    """

    selected = lexical_path(path)
    _prepare_application_owned_parent(selected, application_owned_directory)

    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            selected.parent.mkdir(parents=True, exist_ok=True)
            operation = _runtime_operation(selected)
            if operation is None:
                return selected.open(
                    "a", encoding=encoding, errors=errors, newline="\n"
                )
            fd = _native_open(
                selected, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _PRIVATE_FILE_MODE
            )
            try:
                stream = os.fdopen(
                    fd,
                    "a",
                    encoding=encoding,
                    errors=errors,
                    newline="\n",
                    closefd=False,
                )
            except BaseException:
                _native_close(fd)
                raise
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            raw._check(operation).files.append(stream)
            return stream
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=True,
    )
    file_fd = -1
    try:
        try:
            entry_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            entry_stat = None
        else:
            rejected = _classify_private_file_stat(
                entry_stat,
                expected_uid=os.geteuid(),
            )
            if rejected is not None:
                raise PrivatePathError(PrivatePathResult(selected, rejected))

        file_fd = _native_open(
            leaf,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW | _NONBLOCK | _NOCTTY,
            _PRIVATE_FILE_MODE,
            dir_fd=parent_fd,
        )
        file_stat = os.fstat(file_fd)
        rejected = _classify_private_file_stat(
            file_stat,
            expected_uid=os.geteuid(),
        )
        if rejected is not None:
            raise PrivatePathError(PrivatePathResult(selected, rejected))
        if entry_stat is not None and not _same_identity(file_stat, entry_stat):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="target_replaced",
                )
            )
        os.fchmod(file_fd, _PRIVATE_FILE_MODE)
        if not _private_file_postcondition_holds(
            file_fd,
            parent_fd,
            leaf,
            expected_identity=file_stat,
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="private_file_postcondition_failed",
                )
            )
        operation = _runtime_operation(selected)
        stream = os.fdopen(
            file_fd,
            "a",
            encoding=encoding,
            errors=errors,
            # Same LF contract as the Windows branch above. A no-op on this
            # platform (``os.linesep`` is already LF), stated so the two
            # branches cannot drift apart on the one property callers of
            # this helper reason about in bytes.
            newline="\n",
            closefd=operation is None,
        )
        if operation is not None:
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            raw._check(operation).files.append(stream)
        file_fd = -1
        return stream
    except PrivatePathError:
        raise
    except OSError as exc:
        raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        if file_fd >= 0:
            _native_close(file_fd)
        _native_close(parent_fd)


@_admitted_stream
def open_private_lock_stream(
    path: PathInput,
    *,
    application_owned_directory: PathInput | None = None,
    encoding: str = "utf-8",
    errors: str | None = None,
) -> TextIO:
    """Create or open a checked lock; retain the same native FD through close."""

    lock_create = True

    if lock_create:
        "".encode(encoding)
    selected = lexical_path(path)
    _prepare_application_owned_parent(selected, application_owned_directory)

    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            if lock_create:
                try:
                    create_private_text(
                        selected,
                        "",
                        application_owned_directory=application_owned_directory,
                        encoding=encoding,
                    )
                except FileExistsError:
                    pass
            selected.parent.mkdir(parents=True, exist_ok=True)
            operation = _runtime_operation(selected)
            if operation is None:
                return selected.open(
                    "a", encoding=encoding, errors=errors, newline="\n"
                )
            fd = _native_open(
                selected, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _PRIVATE_FILE_MODE
            )
            try:
                stream = os.fdopen(
                    fd,
                    "a",
                    encoding=encoding,
                    errors=errors,
                    newline="\n",
                    closefd=False,
                )
            except BaseException:
                _native_close(fd)
                raise
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            raw._check(operation).files.append(stream)
            return stream
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=True,
    )
    file_fd = -1
    created = False
    created_stat = None
    try:
        if lock_create:
            outcome = _NativeOpenOutcome()
            try:
                file_fd = _native_open(
                    leaf,
                    os.O_WRONLY
                    | os.O_APPEND
                    | os.O_CREAT
                    | os.O_EXCL
                    | _NOFOLLOW
                    | _NONBLOCK
                    | _NOCTTY,
                    _PRIVATE_FILE_MODE,
                    dir_fd=parent_fd,
                    _outcome=outcome,
                )
                created = True
            except FileExistsError:
                if not outcome.rejected or outcome.descriptor is not None:
                    raise
            if created:
                created_stat = os.fstat(file_fd)
                rejected = _classify_private_file_stat(
                    created_stat,
                    expected_uid=os.geteuid(),
                )
                if rejected is not None:
                    raise PrivatePathError(PrivatePathResult(selected, rejected))
                os.fchmod(file_fd, _PRIVATE_FILE_MODE)
                os.fsync(file_fd)
            # Preserve the former append helper's fresh ancestor check after
            # either actual creation/fsync or confirmed existing-file rejection.
            reopened, reopened_leaf = _open_verified_parent(
                selected,
                missing_leaf_allowed=True,
            )
            try:
                old_info, new_info = os.fstat(parent_fd), os.fstat(reopened)
                if not _same_identity(old_info, new_info):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.OPERATION_FAILED,
                            reason="target_parent_replaced",
                        )
                    )
                if application_owned_directory is not None and (
                    not stat.S_ISDIR(new_info.st_mode)
                    or new_info.st_uid != os.geteuid()
                    or stat.S_IMODE(new_info.st_mode) != _PRIVATE_DIRECTORY_MODE
                ):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="application_owned_parent_changed",
                        )
                    )
            except BaseException:
                _native_close(reopened)
                raise
            previous_parent, parent_fd = parent_fd, reopened
            leaf = reopened_leaf
            _native_close(previous_parent)
        entry_stat = created_stat
        if file_fd < 0:
            try:
                entry_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                if lock_create:
                    raise
            else:
                rejected = _classify_private_file_stat(
                    entry_stat,
                    expected_uid=os.geteuid(),
                )
                if rejected is not None:
                    raise PrivatePathError(PrivatePathResult(selected, rejected))
            file_fd = _native_open(
                leaf,
                os.O_WRONLY
                | os.O_APPEND
                | (0 if lock_create else os.O_CREAT)
                | _NOFOLLOW
                | _NONBLOCK
                | _NOCTTY,
                _PRIVATE_FILE_MODE,
                dir_fd=parent_fd,
            )
        file_stat = os.fstat(file_fd)
        rejected = _classify_private_file_stat(
            file_stat,
            expected_uid=os.geteuid(),
        )
        if rejected is not None:
            raise PrivatePathError(PrivatePathResult(selected, rejected))
        if entry_stat is not None and not _same_identity(file_stat, entry_stat):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="target_replaced",
                )
            )
        if not created:
            os.fchmod(file_fd, _PRIVATE_FILE_MODE)
        if not _private_file_postcondition_holds(
            file_fd,
            parent_fd,
            leaf,
            expected_identity=file_stat,
        ):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    reason="private_file_postcondition_failed",
                )
            )
        operation = _runtime_operation(selected)
        stream = os.fdopen(
            file_fd,
            "a",
            encoding=encoding,
            errors=errors,
            # Same LF contract as the Windows branch above. A no-op on this
            # platform (``os.linesep`` is already LF), stated so the two
            # branches cannot drift apart on the one property callers of
            # this helper reason about in bytes.
            newline="\n",
            closefd=operation is None,
        )
        if operation is not None:
            raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
            raw._check(operation).files.append(stream)
        file_fd = -1
        return stream
    except PrivatePathError:
        raise
    except OSError as exc:
        raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        if file_fd >= 0:
            _native_close(file_fd)
        _native_close(parent_fd)

# These exact callable records select only the optional combined lock route.
# They grant no native/source authority and never replace admission checks.
_LOCK_STREAM_BINDING = tuple(
    (
        name,
        function,
        function.__code__,
        function.__defaults__,
        None if function.__kwdefaults__ is None else tuple(function.__kwdefaults__.items()),
        None if function.__closure__ is None else tuple(cell.cell_contents for cell in function.__closure__),
        function.__globals__,
    )
    for name, function in (
        ("create_private_text", create_private_text),
        ("create_private_text.__wrapped__", create_private_text.__wrapped__),
        ("open_private_text_append_stream", open_private_text_append_stream),
        ("open_private_text_append_stream.__wrapped__", open_private_text_append_stream.__wrapped__),
        ("open_private_lock_stream", open_private_lock_stream),
        ("open_private_lock_stream.__wrapped__", open_private_lock_stream.__wrapped__),
    )
)
def _make_lock_stream_selector(original):
    """Capture only this optional lock schedule's original collaborators."""
    namespace = globals()
    admitted = _admitted_stream
    admitted_code = admitted.__code__

    def current(caller, caller_binding):
        if namespace.get("_LOCK_STREAM_BINDING") is not original:
            return False
        if namespace.get("_lock_stream_for_callbacks") is not _lock_stream_for_callbacks:
            return False
        if _lock_stream_for_callbacks.__code__ is not select_code or _lock_stream_for_callbacks.__globals__ is not namespace:
            return False
        if namespace.get("_admitted_stream") is not admitted or admitted.__code__ is not admitted_code:
            return False
        if caller is not None and (
            caller.get("_LOCK_STREAM_SELECTOR_BINDING") is not caller_binding
            or caller.get("_lock_stream_for_callbacks") is not _lock_stream_for_callbacks
            or caller.get("create_private_text") is not original[0][1]
            or caller.get("open_private_text_append_stream") is not original[2][1]
            or caller.get("open_private_lock_stream") is not original[4][1]
        ):
            return False
        if caller is not None:
            # The caller captured this exact record before selecting us. Recheck
            # that selected function after admission, before any leaf effects.
            if type(caller_binding) is not tuple or len(caller_binding) != 4:
                return False
            selector, code, defining_globals, closure = caller_binding
            if (
                selector is not _lock_stream_for_callbacks
                or code is not select_code
                or defining_globals is not namespace
                or selector.__defaults__ is not None
                or selector.__kwdefaults__ is not None
                or type(closure) is not tuple
                or selector.__closure__ is None
                or len(selector.__closure__) != len(closure)
                or any(
                    cell.cell_contents is not value
                    for cell, value in zip(selector.__closure__, closure, strict=True)
                )
            ):
                return False
        for name, function, code, defaults, kwdefaults, closure, defining_globals in original:
            live = namespace.get(name.split(".")[0])
            if "." in name:
                live = getattr(live, "__wrapped__", None)
            if live is not function or function.__code__ is not code or function.__globals__ is not defining_globals:
                return False
            if function.__defaults__ is not defaults:
                return False
            keywords = function.__kwdefaults__
            if kwdefaults is None:
                if keywords is not None:
                    return False
            elif type(keywords) is not dict or len(keywords) != len(kwdefaults) or any(
                key not in keywords or keywords[key] is not value for key, value in kwdefaults
            ):
                return False
            cells = function.__closure__
            if closure is None:
                if cells is not None:
                    return False
            elif cells is None or len(cells) != len(closure) or any(
                cell.cell_contents is not value for cell, value in zip(cells, closure, strict=True)
            ):
                return False
        return True

    def _lock_stream_for_callbacks(create, append, lock, caller):
        if type(caller) is not dict:
            return None
        if (
            current.__code__ is not current_code
            or current.__globals__ is not namespace
            or current.__defaults__ is not current_defaults
            or current.__kwdefaults__ is not current_keywords
            or current.__closure__ is not current_cells
            or any(cell.cell_contents is not value for cell, value in current_values)
        ):
            return None
        caller_binding = caller.get("_LOCK_STREAM_SELECTOR_BINDING")
        if (
            create is not original[0][1]
            or append is not original[2][1]
            or lock is not original[4][1]
            or not current(caller, caller_binding)
        ):
            return None
        body = original[5][1]

        def finish(path, *args, **kwargs):
            # The unchanged admitted wrapper has completed its real checks.
            # Stale selection cannot invoke a mutable body or bypass callbacks.
            if (
                current.__code__ is not current_code
                or current.__globals__ is not namespace
                or current.__defaults__ is not current_defaults
                or current.__kwdefaults__ is not current_keywords
                or current.__closure__ is not current_cells
                or any(cell.cell_contents is not value for cell, value in current_values)
                or not current(caller, caller_binding)
            ):
                raise RuntimeError("private_lock_source_changed")
            return body(path, *args, **kwargs)

        return admitted(finish)

    select_code = _lock_stream_for_callbacks.__code__
    current_code = current.__code__
    current_defaults = current.__defaults__
    current_keywords = current.__kwdefaults__
    current_cells = current.__closure__
    current_values = tuple((cell, cell.cell_contents) for cell in current_cells or ())
    return _lock_stream_for_callbacks


_lock_stream_for_callbacks = _make_lock_stream_selector(_LOCK_STREAM_BINDING)



def unlink_private_file(
    path: PathInput,
    *,
    application_owned_directory: PathInput | None = None,
) -> bool:
    """Delete one verified private regular file without following links."""

    selected = lexical_path(path)
    _prepare_application_owned_parent(selected, application_owned_directory)
    if not _atomic_posix_guards_available():
        try:
            with open_private_binary(selected):
                pass
            os.unlink(selected)
        except FileNotFoundError:
            return False
        return True

    try:
        parent_fd, leaf = _open_verified_parent(
            selected,
            missing_leaf_allowed=False,
        )
    except FileNotFoundError:
        return False
    try:
        try:
            target_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        rejected = _classify_private_file_stat(
            target_stat,
            expected_uid=os.geteuid(),
        )
        if rejected is not None:
            raise PrivatePathError(PrivatePathResult(selected, rejected))
        os.unlink(leaf, dir_fd=parent_fd)
        return True
    finally:
        os.close(parent_fd)


@contextlib.contextmanager
@_admitted_reader
def open_private_binary(path: PathInput) -> Iterator[PrivateBinaryFile]:
    """Open and harden an existing private file without following links."""

    selected = lexical_path(path)
    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            operation = _runtime_operation(selected)
            if operation is None:
                with selected.open("rb") as stream:
                    yield PrivateBinaryFile(
                        stream=stream,
                        result=PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNVERIFIED_PLATFORM,
                            reason="native_acl_not_verified",
                        ),
                    )
            else:
                raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
                state = raw._check(operation)
                fd = _native_open(selected, os.O_RDONLY | getattr(os, "O_BINARY", 0))
                try:
                    stream = os.fdopen(fd, "rb", closefd=False)
                    state.files.append(stream)
                    try:
                        yield PrivateBinaryFile(
                            stream=stream,
                            result=PrivatePathResult(
                                selected,
                                PrivatePathStatus.UNVERIFIED_PLATFORM,
                                reason="native_acl_not_verified",
                            ),
                        )
                    finally:
                        _close_runtime_stream(operation, stream)
                finally:
                    _native_close(fd)
            return
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parent_fd, leaf = _open_verified_parent(
        selected,
        missing_leaf_allowed=False,
    )
    file_fd = -1
    try:
        try:
            entry_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise _private_path_error_from_oserror(selected, exc) from None

        if not stat.S_ISREG(entry_stat.st_mode):
            raise PrivatePathError(
                PrivatePathResult(
                    selected,
                    PrivatePathStatus.LINK_OR_NON_REGULAR,
                )
            )

        try:
            file_fd = _native_open(
                leaf,
                _PRIVATE_FILE_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
        except FileNotFoundError as exc:
            raise _private_path_error_from_oserror(selected, exc) from None
        except OSError as exc:
            raise _private_path_error_from_oserror(selected, exc) from None

        try:
            file_stat = os.fstat(file_fd)
            rejected = _classify_private_file_stat(
                file_stat,
                expected_uid=os.geteuid(),
            )
            if rejected is not None:
                raise PrivatePathError(PrivatePathResult(selected, rejected))
            prior_mode = stat.S_IMODE(file_stat.st_mode)
            if prior_mode != _PRIVATE_FILE_MODE:
                os.fchmod(file_fd, _PRIVATE_FILE_MODE)
                status = PrivatePathStatus.HARDENED_PRIVATE
            else:
                status = PrivatePathStatus.ALREADY_PRIVATE
            if not _private_file_postcondition_holds(
                file_fd,
                parent_fd,
                leaf,
                expected_identity=file_stat,
            ):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="private_file_postcondition_failed",
                    )
                )
            operation = _runtime_operation(selected)
            stream = os.fdopen(
                file_fd, "rb", closefd=operation is None and _voice_operation() is None
            )
            stream_fd = file_fd
            if operation is not None:
                raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
                raw._check(operation).files.append(stream)
            file_fd = -1
        except PrivatePathError:
            raise
        except OSError as exc:
            raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        if file_fd >= 0:
            _native_close(file_fd)
        _native_close(parent_fd)

    if _voice_operation() is not None:
        try:
            with stream:
                yield PrivateBinaryFile(
                    stream=stream, result=PrivatePathResult(selected, status)
                )
        finally:
            _native_close(stream_fd)
    elif operation is None:
        with stream:
            yield PrivateBinaryFile(
                stream=stream,
                result=PrivatePathResult(selected, status),
            )
    else:
        try:
            yield PrivateBinaryFile(
                stream=stream,
                result=PrivatePathResult(selected, status),
            )
        finally:
            _close_runtime_stream(operation, stream)
            _native_close(stream_fd)


@_admitted_file
def secure_private_directory(
    path: PathInput,
    *,
    create: bool,
    application_owned: bool,
    _open: Callable[..., int] | None = None,
    _close: Callable[[int], None] | None = None,
) -> PrivatePathResult:
    """Create or harden an application-owned directory."""

    close = _close or _native_close
    open_options = {} if _open is None else {"_open": _open}
    observer_options = dict(open_options)
    if _close is not None:
        observer_options["_close"] = _close
    selected = lexical_path(path)
    if not application_owned:
        raise ValueError("Only application-owned directories may be changed")
    if selected == Path(selected.anchor):
        raise ValueError("The filesystem root cannot be application-owned")
    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            if create:
                selected.mkdir(parents=True, exist_ok=True)
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    euid = os.geteuid()
    parts = selected.parts
    current_fd = (_open or _native_open)(
        selected.anchor, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW
    )
    created_final = False
    hardened_final = False
    try:
        current_stat = os.fstat(current_fd)
        pending = list(parts[1:])
        traversed: list[str] = []
        symlink_hops = 0
        while pending:
            component = pending.pop(0)
            is_final = not pending
            if not _trusted_directory_owner(current_stat, euid):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="untrusted_directory_owner",
                        offender_path=_offender_path(traversed),
                        offender_detail=_describe_stat(current_stat),
                        offender_mode=_offender_mode(current_stat),
                    )
                )
            current_mode = stat.S_IMODE(current_stat.st_mode)
            shared_writable = bool(current_mode & 0o022)
            sticky = bool(current_mode & stat.S_ISVTX)
            if shared_writable and not sticky:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="shared_writable_parent",
                        offender_path=_offender_path(traversed),
                        offender_detail=_describe_stat(current_stat),
                        offender_mode=_offender_mode(current_stat),
                    )
                )

            created_component = False
            try:
                next_fd = _open_directory_component(
                    current_fd, component, **open_options
                )
            except FileNotFoundError:
                if not create:
                    raise
                if shared_writable:
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="missing_component_in_shared_sticky_parent",
                            offender_path=_offender_path(traversed),
                            offender_detail=_describe_stat(current_stat),
                            offender_mode=_offender_mode(current_stat),
                        )
                    ) from None
                try:
                    _native_mkdir(
                        component,
                        mode=_PRIVATE_DIRECTORY_MODE,
                        dir_fd=current_fd,
                    )
                    created_component = True
                except FileExistsError:
                    # Another caller may have created this entry after our
                    # open. Reopen without following links and apply the same
                    # owner/type/mode checks below; existence is not trust.
                    pass
                next_fd = _open_directory_component(
                    current_fd, component, **open_options
                )
            except OSError as exc:
                current_fd, symlink_hops = _follow_trusted_symlink(
                    current_fd=current_fd,
                    component=component,
                    pending=pending,
                    hops=symlink_hops,
                    selected=selected,
                    exc=exc,
                    traversed=traversed,
                    **observer_options,
                )
                current_stat = os.fstat(current_fd)
                continue

            transferred = False
            try:
                next_stat = os.fstat(next_fd)
                if not stat.S_ISDIR(next_stat.st_mode):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.LINK_OR_NON_REGULAR,
                            reason="non_directory_component",
                            offender_path=_offender_path(traversed, component),
                        )
                    )
                if next_stat.st_uid != euid and (created_component or is_final):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.WRONG_OWNER,
                            reason="application_directory_wrong_owner",
                            offender_path=_offender_path(traversed, component),
                            offender_detail=_describe_stat(next_stat),
                            offender_mode=_offender_mode(next_stat),
                        )
                    )
                if not _trusted_directory_owner(next_stat, euid):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="untrusted_directory_owner",
                            offender_path=_offender_path(traversed, component),
                            offender_detail=_describe_stat(next_stat),
                            offender_mode=_offender_mode(next_stat),
                        )
                    )
                if shared_writable and next_stat.st_uid not in {0, euid}:
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason="sticky_child_wrong_owner",
                            offender_path=_offender_path(traversed, component),
                            offender_detail=_describe_stat(next_stat),
                            offender_mode=_offender_mode(next_stat),
                        )
                    )

                if created_component or is_final:
                    before = stat.S_IMODE(next_stat.st_mode)
                    if before != _PRIVATE_DIRECTORY_MODE:
                        os.fchmod(next_fd, _PRIVATE_DIRECTORY_MODE)
                        if is_final:
                            hardened_final = True
                    if not _private_directory_postcondition_holds(
                        next_fd,
                        current_fd,
                        component,
                        expected_identity=next_stat,
                    ):
                        raise PrivatePathError(
                            PrivatePathResult(
                                selected,
                                PrivatePathStatus.OPERATION_FAILED,
                                reason="private_directory_postcondition_failed",
                            )
                        )
                if is_final and created_component:
                    created_final = True

                old_fd = current_fd
                current_fd = next_fd
                transferred = True
                close(old_fd)
                current_stat = os.fstat(current_fd)
                traversed.append(component)
            finally:
                if not transferred:
                    close(next_fd)

        status = (
            PrivatePathStatus.CREATED_PRIVATE
            if created_final
            else (
                PrivatePathStatus.HARDENED_PRIVATE
                if hardened_final
                else PrivatePathStatus.ALREADY_PRIVATE
            )
        )
        return PrivatePathResult(selected, status)
    except PrivatePathError:
        raise
    except OSError as exc:
        raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        close(current_fd)


def verify_trusted_directory(
    path: PathInput,
    *,
    allow_shared_sticky: bool,
    _close: Callable[[int], None] | None = None,
    _open: Callable[..., int] | None = None,
) -> PrivatePathResult:
    """Verify an existing lexical directory without creating or changing it.

    Security intent — do not weaken this to make a test pass:

    The walk starts at `/` and opens one component at a time through `dir_fd`,
    with `O_NOFOLLOW`, checking every directory it crosses is owned by root or
    the effective uid and is not group/world-writable without the sticky bit.
    The point is that no other local user can ever redirect the final directory,
    and that the answer cannot change between the check and the open, because
    the path string is never re-derived after the first component.

    A symlink component is crossed only when the symlink *itself* passes the
    same trust test (see `_trusted_symlink`: root-owned, no group/world write
    bits), and traversal then resumes from its target under the identical
    per-component rules, capped at `_MAX_TRUSTED_SYMLINK_HOPS`. That is what
    lets macOS reach `/var/folders/...` across `/var -> private/var`. Anything
    else — a caller-owned alias, a link in a shared directory, a non-directory —
    still fails with `LINK_OR_NON_REGULAR`.

    Resolving the path up front (`Path.resolve()`, `os.path.realpath`) would be
    a much simpler way to reach the same directories and is the wrong answer: it
    follows every symlink before anything has been vetted and re-opens a name
    that may have changed since. `lexical_path` deliberately does not resolve.
    """

    close = _close or _native_close
    selected = lexical_path(path)
    operation = _runtime_operation()
    if operation is not None:
        raw = sys.modules["tldw_chatbook.Backup_Recovery.raw_participants"]
        state = raw._check(operation)
        if selected != state.selected.parent and selected not in state.directories:
            raise RuntimeError("raw_source_helper_not_supported")

    if not _posix_guards_available():
        if _WINDOWS_PLATFORM:
            if not selected.is_dir():
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.OPERATION_FAILED,
                        reason="missing_or_non_directory",
                    )
                )
            return PrivatePathResult(
                selected,
                PrivatePathStatus.UNVERIFIED_PLATFORM,
                reason="native_acl_not_verified",
            )
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="required_posix_guards_unavailable",
            )
        )

    parts = selected.parts
    if len(parts) < 2 or parts[0] != selected.anchor:
        raise PrivatePathError(
            PrivatePathResult(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                reason="invalid_absolute_path",
            )
        )

    euid = os.geteuid()
    current_fd = (_open or _native_open)(
        selected.anchor, _DIRECTORY_OPEN_FLAGS | _NOFOLLOW
    )
    try:
        current_stat = os.fstat(current_fd)
        pending = list(parts[1:])
        traversed: list[str] = []
        symlink_hops = 0
        while pending:
            component = pending.pop(0)
            is_final = not pending
            if not _trusted_directory_owner(current_stat, euid):
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="untrusted_directory_owner",
                        offender_path=_offender_path(traversed),
                        offender_detail=_describe_stat(current_stat),
                        offender_mode=_offender_mode(current_stat),
                    )
                )
            current_mode = stat.S_IMODE(current_stat.st_mode)
            current_writable = bool(current_mode & 0o022)
            current_sticky = bool(current_mode & stat.S_ISVTX)
            if current_writable and not current_sticky:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="shared_writable_parent",
                        offender_path=_offender_path(traversed),
                        offender_detail=_describe_stat(current_stat),
                        offender_mode=_offender_mode(current_stat),
                    )
                )

            try:
                next_fd = _open_directory_component(
                    current_fd,
                    component,
                    **({"_open": _open} if _open is not None else {}),
                )
            except FileNotFoundError:
                raise PrivatePathError(
                    PrivatePathResult(
                        selected,
                        PrivatePathStatus.UNSAFE_PARENT,
                        reason="missing_directory",
                        offender_path=_offender_path(traversed, component),
                    )
                ) from None
            except OSError as exc:
                current_fd, symlink_hops = _follow_trusted_symlink(
                    current_fd=current_fd,
                    component=component,
                    pending=pending,
                    hops=symlink_hops,
                    selected=selected,
                    exc=exc,
                    traversed=traversed,
                    _close=_close,
                    _open=_open,
                )
                current_stat = os.fstat(current_fd)
                continue

            transferred = False
            try:
                next_stat = os.fstat(next_fd)
                if not stat.S_ISDIR(next_stat.st_mode):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.LINK_OR_NON_REGULAR,
                            reason="non_directory_component",
                            offender_path=_offender_path(traversed, component),
                        )
                    )
                if not _trusted_directory_owner(next_stat, euid):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            (
                                PrivatePathStatus.WRONG_OWNER
                                if is_final
                                else PrivatePathStatus.UNSAFE_PARENT
                            ),
                            reason="untrusted_directory_owner",
                            offender_path=_offender_path(traversed, component),
                            offender_detail=_describe_stat(next_stat),
                            offender_mode=_offender_mode(next_stat),
                        )
                    )
                if not _trusted_directory_postcondition_holds(
                    next_fd,
                    current_fd,
                    component,
                    expected_identity=next_stat,
                ):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.OPERATION_FAILED,
                            reason="trusted_directory_postcondition_failed",
                        )
                    )

                next_mode = stat.S_IMODE(next_stat.st_mode)
                next_writable = bool(next_mode & 0o022)
                next_sticky = bool(next_mode & stat.S_ISVTX)
                if next_writable and (
                    not next_sticky or (is_final and not allow_shared_sticky)
                ):
                    raise PrivatePathError(
                        PrivatePathResult(
                            selected,
                            PrivatePathStatus.UNSAFE_PARENT,
                            reason=(
                                "shared_sticky_directory_not_allowed"
                                if next_sticky and is_final
                                else "shared_writable_parent"
                            ),
                            offender_path=_offender_path(traversed, component),
                            offender_detail=_describe_stat(next_stat),
                            offender_mode=_offender_mode(next_stat),
                        )
                    )

                old_fd = current_fd
                current_fd = next_fd
                transferred = True
                close(old_fd)
                current_stat = next_stat
                traversed.append(component)
            finally:
                if not transferred:
                    close(next_fd)

        return PrivatePathResult(
            selected,
            PrivatePathStatus.TRUSTED_DIRECTORY,
            reason="trusted_existing_directory",
        )
    except PrivatePathError:
        raise
    except OSError as exc:
        raise _private_path_error_from_oserror(selected, exc) from None
    finally:
        close(current_fd)
