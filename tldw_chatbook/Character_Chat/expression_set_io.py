"""Pure import/export logic for a character's expression image SET
(idle/thinking/speaking/error). No Textual and no DB-module imports -- takes a
db object as a parameter where needed. Reused by P3d-3's .vpack extractor.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

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
    Handles a .zip (in-memory, security-capped), a directory, or image files.
    """
    pairs: list[tuple[str, bytes]] = []
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []
    for path in paths:
        try:
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    if child.is_file():
                        pairs.append((child.name, child.read_bytes()))
            elif zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as zf:
                    infos = [i for i in zf.infolist() if not i.is_dir()]
                    if len(infos) > MAX_ZIP_MEMBERS:
                        notes.append(f"Archive has {len(infos)} members; only the first {MAX_ZIP_MEMBERS} were read.")
                        infos = infos[:MAX_ZIP_MEMBERS]
                    total = 0
                    for info in infos:
                        if info.file_size > MAX_MEMBER_BYTES:
                            skipped.append((info.filename, "file too large"))
                            continue
                        if total + info.file_size > MAX_TOTAL_BYTES:
                            notes.append("Archive exceeds the total size cap; remaining members skipped.")
                            break
                        total += info.file_size
                        pairs.append((Path(info.filename).name, zf.read(info)))
            elif path.is_file():
                pairs.append((path.name, path.read_bytes()))
            else:
                skipped.append((str(path), "not found"))
        except Exception as exc:   # broken/encrypted zip, unreadable dir, etc.
            skipped.append((str(path), f"could not read: {exc}"))
    return pairs, skipped, notes


def resolve_local_expression_set(paths: list[Path]) -> ExpressionSetResolution:
    """Resolve selected inputs into a validated {state: bytes} set. Never raises."""
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


def apply_expression_images_to_db(db, character_id: int, images: dict[str, bytes]):
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
