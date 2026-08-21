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
    temp_path: Path, target_path: Path, write_fn: Callable[[Path], None]
) -> None:
    """Call ``write_fn(temp_path)`` then atomically replace ``target_path``.

    On any exception from either the write or the replace, the temp file is
    best-effort unlinked (it is this writer's own temp file -- writer-unique
    naming means no other writer could be relying on it) and the original
    exception is re-raised unchanged. This never swallows a genuine failure;
    it only ever prevents a stray temp file from being left behind by one.

    Args:
        temp_path: Writer-unique temp file to write through, normally from
            ``unique_temp_path``. Must be on the same filesystem as
            ``target_path`` for the replace to be atomic.
        target_path: Final destination, replaced in one step once the write
            has completed.
        write_fn: Callable given ``temp_path``; performs the actual write.

    Raises:
        BaseException: Whatever ``write_fn`` or ``Path.replace`` raised,
            re-raised unchanged after the temp file is cleaned up.
    """
    try:
        write_fn(temp_path)
        temp_path.replace(target_path)
    except BaseException:
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
