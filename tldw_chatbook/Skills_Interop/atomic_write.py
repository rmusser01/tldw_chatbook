"""Shared writer-unique atomic write-then-replace primitive (TASK-17963).

Several skills-subsystem stores do the same write-to-temp-then-`Path.replace`
atomic write, and all of them shared the same flaw: a FIXED temp filename.
Two writers touching the same target at close to the same moment -- two app
instances, or two threads/async callers inside one instance -- would race on
that one temp path: one writer's `temp.replace(target)` could consume the
other's still-being-written temp file (or one writer's write could clobber
the other's in-flight temp file), surfacing as a stray `FileNotFoundError`
from `replace()` or a silently corrupted target.

`ProjectSkillsPromptLedger.record()` (`project_skills_prompt.py`, shipped in
TASK-18705) solved this first by folding the writer's pid + thread id into
its temp name. This module lifts that scheme into one shared place so the
other fixed-name sites (`local_skills_service.py`'s `_save_index`/
`_write_text_atomic`/`_write_bytes_atomic`, `skill_trust_store.py`'s
`_atomic_write_json`/`_atomic_write_bytes`) get the same fix instead of each
re-deriving it. It lives directly under `Skills_Interop/` rather than as a
method on either store: `local_skills_service.py` and `skill_trust_store.py`
do not import each other (checked -- no cycle either way), but a shared
helper used by both is preferable to two near-duplicate private
implementations, so this module has no dependency on either of them and both
import from it instead.

Only the temp-name COLLISION class disappears here -- a genuine write or
replace failure (disk full, permission denied, the target's directory
vanishing, etc.) still raises. Unlike the advisory prompt ledger above (which
swallows `OSError` because losing one record just means one extra prompt
later), the stores that use this module expect a real write failure to
surface to their caller; `replace_atomically` only ever best-effort cleans up
the stray temp file it left behind, never the exception itself.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

_OWNER_ONLY_FILE_MODE = 0o600


def _owner_only_open_flags() -> int:
    """Return flags for exclusive owner-only temp-file creation."""
    return (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def unique_temp_path(path: Path, *, hidden: bool = False) -> Path:
    """Build a writer-unique temp path alongside ``path``.

    Args:
        path: The eventual write target; the temp file is a sibling of it
            (same parent directory, derived from its name).
        hidden: Dot-prefix the temp name. Preserves each call site's
            existing visibility convention -- `local_skills_service.py`
            never dot-prefixed its temp files, `skill_trust_store.py`
            always has (its "hidden trust-material file" convention);
            pass the matching value rather than picking one globally.

    Returns:
        `<name>.<pid>.<tid>.tmp` (or `.<name>.<pid>.<tid>.tmp` when
        `hidden`) -- a name no other writer, another process or another
        thread in this one, can produce for the same target at the same
        time, so two concurrent writers to the same target never share a
        temp path.
    """
    name = f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    if hidden:
        name = f".{name}"
    return path.with_name(name)


def replace_atomically(
    temp_path: Path,
    target_path: Path,
    write_fn: Callable[[Path], None],
    *,
    owner_only: bool = False,
) -> None:
    """Call ``write_fn(temp_path)`` then atomically replace ``target_path``.

    By default, behavior is unchanged: ``write_fn`` creates the temp file, and
    any write or replace failure triggers best-effort cleanup. With
    ``owner_only=True``, the temp file is instead created exclusively with mode
    ``0o600`` before content is written. On POSIX, ``fchmod`` confirms that mode
    before the descriptor is closed, the callback runs, and the atomic replace
    occurs. An exclusive-open collision happens before this writer owns the
    temp path, so the unexplained existing file is preserved.

    Once this call creates or delegates creation of its own temp file, any
    exception from the setup, write, or replace best-effort unlinks that temp
    file and re-raises the original exception unchanged. Cleanup errors alone
    are suppressed.

    Args:
        temp_path: Writer-unique temp file to write through, normally from
            ``unique_temp_path``. Must be on the same filesystem as
            ``target_path`` for the replace to be atomic.
        target_path: Final destination, replaced in one step once the write
            has completed.
        write_fn: Callable given ``temp_path``; performs the actual write.
        owner_only: Exclusively precreate ``temp_path`` with owner-only mode
            before invoking ``write_fn``. The default leaves creation to
            ``write_fn`` for compatibility with existing callers.

    Raises:
        BaseException: Whatever secure temp setup, ``write_fn``, or
            ``Path.replace`` raised, re-raised unchanged after cleanup of a
            temp file owned by this call.
    """
    created_temp = False
    try:
        if owner_only:
            fd = os.open(temp_path, _owner_only_open_flags(), _OWNER_ONLY_FILE_MODE)
            created_temp = True
            try:
                if os.name == "posix" and hasattr(os, "fchmod"):
                    os.fchmod(fd, _OWNER_ONLY_FILE_MODE)
            finally:
                os.close(fd)
        write_fn(temp_path)
        temp_path.replace(target_path)
    except BaseException:
        if not owner_only or created_temp:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_text_atomic(
    path: Path, content: str, *, encoding: str = "utf-8", hidden: bool = False
) -> None:
    """Atomically write text to ``path`` via a writer-unique temp file.

    Args:
        path: Destination file, replaced atomically once written.
        content: Text to write.
        encoding: Text encoding for the write.
        hidden: Dot-prefix the temp file, for stores whose directory
            convention hides transient artifacts.

    Raises:
        OSError: Propagated unchanged from the write or the replace; the
            temp file is cleaned up first.
    """
    temp_path = unique_temp_path(path, hidden=hidden)
    replace_atomically(
        temp_path, path, lambda t: t.write_text(content, encoding=encoding)
    )


def write_bytes_atomic(path: Path, data: bytes, *, hidden: bool = False) -> None:
    """Atomically write bytes to ``path`` via a writer-unique temp file.

    Args:
        path: Destination file, replaced atomically once written.
        data: Bytes to write.
        hidden: Dot-prefix the temp file, for stores whose directory
            convention hides transient artifacts.

    Raises:
        OSError: Propagated unchanged from the write or the replace; the
            temp file is cleaned up first.
    """
    temp_path = unique_temp_path(path, hidden=hidden)
    replace_atomically(temp_path, path, lambda t: t.write_bytes(data))
