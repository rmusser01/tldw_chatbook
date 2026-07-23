"""Pure import/export logic for a character's expression image SET
(idle/thinking/speaking/error). No Textual and no DB-module imports -- takes a
db object as a parameter where needed. Reused by P3d-3's .vpack extractor.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

from PIL import Image

from tldw_chatbook.Chat.console_expression_state import (
    EXPRESSION_STATES,       # ("idle","thinking","speaking","error")
    EXPRESSION_IMAGE_STATES, # ("thinking","speaking","error")
)

MAX_ZIP_MEMBERS = 64
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024

_STATE_SET = {s.lower() for s in EXPRESSION_STATES}


@dataclass(frozen=True)
class ExpressionSetResolution:
    """Outcome of resolving a set of import candidates into character
    expression-state images.

    Attributes:
        images: {state: bytes} for each candidate that resolved to a known
            expression state (a subset of ``EXPRESSION_STATES``) with valid,
            PIL-decodable image bytes.
        skipped: (name, reason) pairs for candidates that were rejected --
            e.g. an unrecognized filename, invalid image bytes, or a
            size-cap violation.
        notes: Informational messages about the resolution, e.g. size/count
            caps that were applied or which file won a same-state naming
            conflict.
    """

    images: dict[str, bytes] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (name, reason)
    notes: list[str] = field(default_factory=list)


def _valid_image(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return True
    except Exception:
        return False


def _prefer(existing_name: str, new_name: str) -> bool:
    """Return True if new_name should replace existing for the same state."""
    ext_e, ext_n = Path(existing_name).suffix.lower(), Path(new_name).suffix.lower()
    if ext_n == ".png" and ext_e != ".png":
        return True
    if ext_e == ".png" and ext_n != ".png":
        return False
    return new_name.lower() < existing_name.lower()  # first alphabetically


def _candidate_pairs(paths: list[Path]) -> tuple[list[tuple[str, bytes]], list[tuple[str, str]], list[str]]:
    """Yield (base_name, bytes) for every candidate file across the inputs.

    Handles a .zip (in-memory, security-capped), a directory, or standalone
    image files. ``MAX_MEMBER_BYTES``/``MAX_TOTAL_BYTES`` are enforced up
    front -- against each entry's on-disk/declared size, before any bytes
    are read -- using a single running total shared across every input in
    ``paths`` (a zip, a directory, and standalone files all draw from the
    same total-size budget).
    """
    pairs: list[tuple[str, bytes]] = []
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []
    total = 0
    for path in paths:
        try:
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    if not child.is_file():
                        continue
                    size = child.stat().st_size
                    if size > MAX_MEMBER_BYTES:
                        skipped.append((child.name, "file too large"))
                        continue
                    if total + size > MAX_TOTAL_BYTES:
                        notes.append("Total size cap exceeded; remaining files were skipped.")
                        break
                    total += size
                    pairs.append((child.name, child.read_bytes()))
            elif zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as zf:
                    infos = [i for i in zf.infolist() if not i.is_dir()]
                    if len(infos) > MAX_ZIP_MEMBERS:
                        notes.append(f"Archive has {len(infos)} members; only the first {MAX_ZIP_MEMBERS} were read.")
                        infos = infos[:MAX_ZIP_MEMBERS]
                    for info in infos:
                        # Normalize Windows-style separators: a member written
                        # on Windows as "dir\\idle.png" has no POSIX path
                        # separator, so a plain Path(...).name would keep the
                        # whole "dir\\idle.png" string as the basename.
                        member = PurePosixPath(info.filename.replace("\\", "/")).name
                        if info.file_size > MAX_MEMBER_BYTES:
                            skipped.append((member, "file too large"))
                            continue
                        if total + info.file_size > MAX_TOTAL_BYTES:
                            notes.append("Archive exceeds the total size cap; remaining members skipped.")
                            break
                        total += info.file_size
                        pairs.append((member, zf.read(info)))
            elif path.is_file():
                size = path.stat().st_size
                if size > MAX_MEMBER_BYTES:
                    skipped.append((path.name, "file too large"))
                elif total + size > MAX_TOTAL_BYTES:
                    notes.append("Total size cap exceeded; remaining files were skipped.")
                else:
                    total += size
                    pairs.append((path.name, path.read_bytes()))
            else:
                skipped.append((str(path), "not found"))
        except Exception as exc:   # broken/encrypted zip, unreadable dir, etc.
            skipped.append((str(path), f"could not read: {exc}"))
    return pairs, skipped, notes


def resolve_local_expression_set(paths: list[Path]) -> ExpressionSetResolution:
    """Resolve selected inputs into a validated {state: bytes} set.

    Args:
        paths: Filesystem paths to resolve -- any mix of a single ``.zip``
            archive, a directory of image files, or standalone image files.
            Callers are responsible for any path validation (this module is
            pure and never validates against the filesystem policy layer).

    Returns:
        An ``ExpressionSetResolution`` holding the resolved {state: bytes}
        images, plus the candidates that were skipped and any informational
        notes. Never raises.
    """
    pairs, skipped, notes = _candidate_pairs(paths)
    chosen: dict[str, tuple[str, bytes]] = {}   # state -> (name, bytes)
    for name, data in pairs:
        state = Path(name).stem.lower()
        if state not in _STATE_SET:
            skipped.append((name, "filename is not a known state"))
            continue
        if not _valid_image(data):
            skipped.append((name, "not a valid image"))
            continue
        if state in chosen:
            keep = chosen[state][0]
            if _prefer(keep, name):
                notes.append(f"Multiple files for {state}; used {name}.")
                chosen[state] = (name, data)
            else:
                notes.append(f"Multiple files for {state}; used {keep}.")
            continue
        chosen[state] = (name, data)
    return ExpressionSetResolution(
        images={state: data for state, (_, data) in chosen.items()},
        skipped=skipped,
        notes=notes,
    )


_FORMAT_TO_EXT = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "GIF": "gif"}


def _detect_ext(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return _FORMAT_TO_EXT.get((im.format or "").upper(), "png")
    except Exception:
        return "png"


def build_expression_set_zip(character_name: str, images: dict[str, bytes]) -> bytes:
    """Build a .zip (bytes) of a character's expression set.

    Args:
        character_name: The character's display name (for the provenance marker).
        images: {state: bytes} for any subset of EXPRESSION_STATES.

    Returns:
        The zip archive bytes: one ``{state}.{ext}`` per present state (ext
        PIL-detected from the bytes) plus an ``expression_set.json`` provenance
        marker. Always a valid zip, even for an empty set.
    """
    present = [s for s in EXPRESSION_STATES if s in images and images[s]]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for state in present:
            zf.writestr(f"{state}.{_detect_ext(images[state])}", images[state])
        manifest = {
            "format": "tldw-expression-set/1",
            "character": character_name,
            "states": present,
        }
        zf.writestr("expression_set.json", json.dumps(manifest, indent=2))
    return buf.getvalue()


class _ExpressionImageDB(Protocol):
    """Minimal shape this pure module needs from a characters DB. Defined
    here (instead of importing the real DB class) to keep this module free
    of DB-layer imports."""

    def set_character_expression_image(
        self, character_id: int, state_id: str, image: bytes, mime: object = None
    ) -> None: ...


def apply_expression_images_to_db(
    db: _ExpressionImageDB, character_id: int, images: dict[str, bytes]
) -> tuple[list[str], list[tuple[str, str]]]:
    """Write the non-idle states to the expression table, best-effort.

    Args:
        db: a CharactersRAGDB-like object exposing ``set_character_expression_image``.
        character_id: the owning character id.
        images: {state: bytes}; only EXPRESSION_IMAGE_STATES are written (idle skipped).

    Returns:
        (applied_states, skipped[(state, reason)]).
    """
    applied: list[str] = []
    skipped: list[tuple[str, str]] = []
    for state in EXPRESSION_IMAGE_STATES:
        data = images.get(state)
        if not data:
            continue
        try:
            db.set_character_expression_image(character_id, state, data, None)
            applied.append(state)
        except Exception as exc:
            skipped.append((state, f"write failed: {exc}"))
    return applied, skipped


@dataclass(frozen=True)
class ExpressionSetApplyResult:
    """Outcome of applying a resolved expression set to a character:
    idle staged in the editor, the reactive states written to the DB."""
    applied: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (state, reason)
