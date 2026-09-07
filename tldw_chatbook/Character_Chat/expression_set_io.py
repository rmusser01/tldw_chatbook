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


def _detect_vpack(zf: zipfile.ZipFile) -> str | None:
    """Detect a persona visual pack by CONTENT (never by file extension).

    Returns the member prefix to use for lookups: "" for a root-level pack,
    "Root/" when the pack sits under a single shared top-level directory
    (the extract-and-re-zip papercut), or None when the archive is not a
    pack (deeper nesting and multiple roots fall through to stem mapping).
    A pack requires all three of ``manifest.json``, ``metadata/assets.json``,
    and ``metadata/pack.json`` (the server's REQUIRED_MEMBERS) to be present
    -- a plain zip that happens to carry only the first two names must still
    fall through to stem mapping. The sniff reads the namelist only -- zero
    member reads.
    """
    names = set(zf.namelist())
    if "manifest.json" in names and "metadata/assets.json" in names and "metadata/pack.json" in names:
        return ""
    roots = {n.split("/", 1)[0] for n in names if n and not n.startswith("/")}
    if len(roots) == 1:
        root = next(iter(roots))
        if (
            f"{root}/manifest.json" in names
            and f"{root}/metadata/assets.json" in names
            and f"{root}/metadata/pack.json" in names
        ):
            return f"{root}/"
    return None


def _candidate_pairs(
    paths: list[Path], start_total: int = 0
) -> tuple[list[tuple[str, bytes]], list[tuple[str, str]], list[str], int]:
    """Yield (base_name, bytes) for every candidate file across the inputs.

    Handles a .zip (in-memory, security-capped), a directory, or standalone
    image files. ``MAX_MEMBER_BYTES``/``MAX_TOTAL_BYTES`` are enforced up
    front -- against each entry's on-disk/declared size, before any bytes
    are read -- using a single running total shared across every input in
    ``paths`` (a zip, a directory, and standalone files all draw from the
    same total-size budget), starting from ``start_total`` so callers can
    thread a budget already partially consumed elsewhere (e.g. by vpack
    reads earlier in the same resolver call).
    """
    pairs: list[tuple[str, bytes]] = []
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []
    total = start_total
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
    return pairs, skipped, notes, total


def resolve_local_expression_set(paths: list[Path]) -> ExpressionSetResolution:
    """Resolve selected inputs into a validated {state: bytes} set.

    Any input that is a persona visual pack (``.tldw-persona-vpack`` or a
    plain ``.zip`` -- detected by CONTENT via ``_detect_vpack``, never by
    file extension) is resolved through the targeted vpack extractor;
    everything else is batched into the generic stem-mapping path below.

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
    images: dict[str, bytes] = {}
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []
    total = 0
    generic: list[Path] = []
    for path in paths:
        handled = False
        try:
            if path.is_file() and zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as zf:
                    vprefix = _detect_vpack(zf)
                    if vprefix is not None:
                        res, total = _resolve_vpack_expression_set(
                            zf, prefix=vprefix, start_total=total
                        )
                        for state, data in res.images.items():
                            images.setdefault(state, data)   # first-writer-wins
                        skipped.extend(res.skipped)
                        notes.extend(res.notes)
                        handled = True
        except Exception as exc:
            skipped.append((str(path), f"could not read: {exc}"))
            handled = True
        if not handled:
            generic.append(path)
    if generic:
        pairs, g_skipped, g_notes, total = _candidate_pairs(generic, start_total=total)
        skipped.extend(g_skipped)
        notes.extend(g_notes)
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
        for state, (_, data) in chosen.items():
            images.setdefault(state, data)   # vpack states win over generic
    return ExpressionSetResolution(images=images, skipped=skipped, notes=notes)


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


# ---------- P3d-3: .tldw-persona-vpack extraction ----------


def _read_member_capped(
    zf: zipfile.ZipFile,
    member: str,
    bytes_cache: dict[str, bytes],
    total: int,
    skipped: list[tuple[str, str]],
) -> tuple[bytes | None, int]:
    """Read one zip member through the size caps, with a per-call cache.

    ``member`` is only ever used as a zip-member KEY -- a traversal-shaped
    value simply fails the ``getinfo`` lookup and is skipped; nothing here
    touches the filesystem. A cache hit does not re-charge the budget (the
    common pack shape is ONE sprite sheet referenced by every state).
    """
    if member in bytes_cache:
        return bytes_cache[member], total
    try:
        info = zf.getinfo(member)
    except KeyError:
        skipped.append((member, "not found in archive"))
        return None, total
    if info.file_size > MAX_MEMBER_BYTES:
        skipped.append((member, "file too large"))
        return None, total
    if total + info.file_size > MAX_TOTAL_BYTES:
        skipped.append((member, "total size cap exceeded"))
        return None, total
    try:
        data = zf.read(info)
    except Exception:
        skipped.append((member, "could not be read"))
        return None, total
    bytes_cache[member] = data
    return data, total + info.file_size


def _read_vpack_json(
    zf: zipfile.ZipFile,
    member: str,
    bytes_cache: dict[str, bytes],
    total: int,
    skipped: list[tuple[str, str]],
) -> tuple[dict | None, int]:
    """Parse a JSON member under the size caps. Returns (obj|None, total).
    Parse failures (bad JSON, RecursionError on pathological nesting) are
    caught -- the vpack path never raises."""
    data, total = _read_member_capped(zf, member, bytes_cache, total, skipped)
    if data is None:
        return None, total
    try:
        obj = json.loads(data)
    except Exception:
        skipped.append((member, "invalid JSON"))
        return None, total
    return obj if isinstance(obj, dict) else None, total


def _resolve_vpack_expression_set(
    zf: zipfile.ZipFile, *, prefix: str = "", start_total: int = 0
) -> tuple[ExpressionSetResolution, int]:
    """Extract one static image per expression state from a persona visual pack.

    Per state: ``manifest.states[state]`` -> ``animations[id]`` (honoring the
    ``asset_ids`` shorthand) -> ``frames[preview_frame]`` if that is a valid
    in-range int else ``frames[0]`` -> assets-index (``source_asset_id`` ->
    ``asset_path``; skip only on explicit ``asset_bytes_status == "missing"``)
    -> capped cached member read -> optional ``region`` crop (bounds-checked,
    re-encoded as PNG) -> ``_valid_image`` -> ``images[state]``.

    Lenient/best-effort: every failure skips that state with a reason; a
    broken manifest degrades to an empty resolution. NEVER raises.
    ``frame_rate``/``duration_ms``/``fallbacks``/``alignment``/checksums are
    inert (all four of our states are in the server's REQUIRED_VISUAL_STATES,
    so any server-valid pack defines them directly).

    Args:
        zf: The open archive.
        prefix: Member-path prefix ("" for a root-level pack; "Pack/" when
            the pack sits under a single shared top-level directory).
        start_total: The resolver call's running size total so far.

    Returns:
        (resolution, new_running_total).
    """
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []
    total = start_total
    bytes_cache: dict[str, bytes] = {}
    image_cache: dict[str, Image.Image] = {}

    manifest, total = _read_vpack_json(zf, f"{prefix}manifest.json", bytes_cache, total, skipped)
    assets_doc, total = _read_vpack_json(zf, f"{prefix}metadata/assets.json", bytes_cache, total, skipped)
    if manifest is None or assets_doc is None:
        notes.append("Archive looks like a persona visual pack, but its manifest could not be read.")
        return ExpressionSetResolution(images={}, skipped=skipped, notes=notes), total

    states = manifest.get("states")
    animations = manifest.get("animations")
    states = states if isinstance(states, dict) else {}
    animations = animations if isinstance(animations, dict) else {}
    assets_index: dict[str, dict] = {}
    entries = assets_doc.get("assets")
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and entry.get("source_asset_id"):
            assets_index[str(entry["source_asset_id"])] = entry

    images: dict[str, bytes] = {}
    for state in EXPRESSION_STATES:
        try:
            animation_id = states.get(state)
            if not animation_id:
                skipped.append((state, "state not in pack"))
                continue
            animation = animations.get(animation_id) if isinstance(animation_id, str) else None
            if not isinstance(animation, dict):
                skipped.append((state, "unknown animation"))
                continue
            frames = animation.get("frames")
            if not (isinstance(frames, list) and frames):
                asset_ids = animation.get("asset_ids")
                frames = (
                    [{"asset_id": a} for a in asset_ids]
                    if isinstance(asset_ids, list) and asset_ids
                    else None
                )
            if not frames:
                skipped.append((state, "animation has no frames"))
                continue
            idx = animation.get("preview_frame")
            frame = frames[idx] if type(idx) is int and 0 <= idx < len(frames) else frames[0]
            if not isinstance(frame, dict):
                skipped.append((state, "invalid frame"))
                continue
            asset_id = frame.get("asset_id")
            entry = assets_index.get(str(asset_id)) if asset_id else None
            if entry is None:
                skipped.append((state, "unknown asset"))
                continue
            # Skip only an EXPLICIT "missing"; an absent status key is
            # tolerated (mirrors the server's own defensive .get reads).
            if entry.get("asset_bytes_status") == "missing":
                skipped.append((state, "asset bytes missing from archive"))
                continue
            asset_path = str(entry.get("asset_path") or "")
            if not asset_path:
                skipped.append((state, "asset has no archive path"))
                continue
            member = f"{prefix}{asset_path}"
            data, total = _read_member_capped(zf, member, bytes_cache, total, skipped)
            if data is None:
                skipped.append((state, "asset could not be read"))
                continue
            region = frame.get("region")
            if region is not None:
                img = image_cache.get(member)
                if img is None:
                    # Decoding an untrusted image: PIL's default
                    # MAX_IMAGE_PIXELS decompression-bomb guard applies, and
                    # any decode failure lands in this state's try/except.
                    img = Image.open(io.BytesIO(data))
                    img.load()
                    image_cache[member] = img
                if not (
                    isinstance(region, dict)
                    and all(type(region.get(k)) is int for k in ("x", "y", "width", "height"))
                    and region["x"] >= 0 and region["y"] >= 0
                    and region["width"] > 0 and region["height"] > 0
                    and region["x"] + region["width"] <= img.width
                    and region["y"] + region["height"] <= img.height
                ):
                    skipped.append((state, "invalid region"))
                    continue
                crop = img.crop((
                    region["x"], region["y"],
                    region["x"] + region["width"], region["y"] + region["height"],
                ))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")   # a crop must be standalone bytes
                out = buf.getvalue()
            else:
                out = data   # whole-asset bytes pass through verbatim
            if not _valid_image(out):
                skipped.append((state, "not a valid image"))
                continue
            images[state] = out
        except Exception as exc:
            skipped.append((state, f"extraction failed: {exc}"))
    return ExpressionSetResolution(images=images, skipped=skipped, notes=notes), total
