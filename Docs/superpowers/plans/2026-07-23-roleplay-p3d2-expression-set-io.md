# Roleplay P3d-2 — Bulk Import + Export of a Character's Expression Set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user import a whole character expression set (idle+thinking+speaking+error) from one `.zip`, and export a character's set to a `.zip`, instead of P3d-1's three one-at-a-time upload slots.

**Architecture:** A new pure module `Character_Chat/expression_set_io.py` (zipfile + PIL only) resolves a `.zip`/dir/file list into a validated `{state: bytes}` set, builds an export zip, and applies the non-idle states to a DB. A screen-level `_apply_expression_set` orchestrator stages idle in the editor (persists on card save) + writes the three via the pure DB helper, bumping the render token once. Two editor buttons drive import (single-`.zip` picker) and export (write to the exports dir).

**Tech Stack:** Python ≥3.11, `zipfile`, PIL, Textual, CharactersRAGDB, loguru.

## Global Constraints

- **NO migration** — `character_expression_images` (P3d-1) + `character_cards.image` already exist (schema v23). **Pre-flight re-verify `_CURRENT_SCHEMA_VERSION == 23`** at plan+merge time.
- **idle is STAGED** via `editor.set_avatar_image(bytes)` (persists on card save, a version-bumping `update_character_card`); the **three reactive states write IMMEDIATELY** via `db.set_character_expression_image` (card-version-independent — P3d-1 invariant). Export reads idle from `editor.current_avatar_bytes()`.
- **Export PIL-detects the format from the bytes** (`Image.open(BytesIO(b)).format`) — `get_character_expression_image` returns bytes only; the stored mime is unreadable.
- **Import UI = a single `.zip`** (`EnhancedFileOpen` returns one file). The resolver stays general (`.zip` / dir / image list) for tests + future.
- **Untrusted-zip security:** read members **into memory** via `zipfile` (never extract attacker paths to disk); enforce `MAX_ZIP_MEMBERS=64`, `MAX_MEMBER_BYTES=16MiB`, `MAX_TOTAL_BYTES=64MiB` against `ZipInfo.file_size` **before** reading each member; PIL-validate every image; skip a bad member (best-effort); a broken/encrypted zip fails cleanly (notification), **never raises into the worker**.
- **Best-effort partial:** each state applies independently; one bad state never fails the whole op; the summary reports skips.
- **Reuse P3d-1**: bump `_character_editor_generation` **ONCE** per import then re-render affected slots + avatar (never per-state — that races); mirror `_expression_upload_dialog_worker` (dialog worker) + `_stage_character_expression_from_path` (dialog-free path method) + `_dictionary_export_worker` (exports-dir + atomic temp-replace) + the `_io_dialog_active` guard + `group="personas-io"`.
- **Saved-character-only** (`editor.expression_character_id() is not None`) + **characters-only** (`_character_editor_is_active()`).
- **`expression_set_io.py` imports NO Textual and NO DB module** (it may take a `db` object as a parameter and call methods on it; it may import the pure state constants from `Chat/console_expression_state.py`).
- Never let an exception escape into an `exit_on_error=True` worker (mirror the P3d-1 IO workers' `try/finally` + inner `try/except`).
- DB tests use a file-backed `CharactersRAGDB(tmp_path / "...db", "test-client")`, **never `:memory:`**.
- `Tests/UI/pytest.ini` sets `asyncio_mode=auto`: keep async tests in `Tests/UI/` OR add `@pytest.mark.asyncio`; don't mix `Tests/UI` with another dir in one pytest invocation.
- **Test env prefix:** `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ... -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign`; subagents PREPEND `cd <worktree> && ` to EVERY Bash call. CONCURRENT-SESSION HAZARD: `personas_screen.py` / `personas_character_editor_widget.py` heavily edited elsewhere — localize; expect a rebase. Stage ONLY task files (never `git add -A`, never `.superpowers/`). NO background/broad sweeps; NEVER pkill.

---

## File Structure

- `tldw_chatbook/Character_Chat/expression_set_io.py` — **create**: pure logic. `resolve_local_expression_set` (Task 1), `build_expression_set_zip` (Task 2), `apply_expression_images_to_db` + the dataclasses + constants (Tasks 1-3).
- `tldw_chatbook/UI/Screens/personas_screen.py` — **modify**: `_apply_expression_set` orchestrator + the import/export path methods + dialog/export workers + message handlers (Tasks 3-4).
- `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` — **modify**: two buttons + their `post_message` (Task 4).
- `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py` — **modify**: two new message classes (Task 4).

---

## Task 1: `resolve_local_expression_set` + security (pure)

**Files:**
- Create: `tldw_chatbook/Character_Chat/expression_set_io.py`
- Test: `Tests/Character_Chat/test_expression_set_io.py` (create)

**Interfaces:**
- Produces:
  - `EXPRESSION_STATES` / `EXPRESSION_IMAGE_STATES` (imported from `tldw_chatbook.Chat.console_expression_state`).
  - `@dataclass(frozen=True) class ExpressionSetResolution: images: dict[str, bytes]; skipped: list[tuple[str, str]]; notes: list[str]`
  - `resolve_local_expression_set(paths: list[Path]) -> ExpressionSetResolution`
  - Caps: `MAX_ZIP_MEMBERS = 64`, `MAX_MEMBER_BYTES = 16 * 1024 * 1024`, `MAX_TOTAL_BYTES = 64 * 1024 * 1024`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Character_Chat/test_expression_set_io.py`:
```python
import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat.expression_set_io import (
    resolve_local_expression_set,
    MAX_ZIP_MEMBERS,
    MAX_TOTAL_BYTES,
)


def _png(color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, format="JPEG")
    return buf.getvalue()


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_zip_maps_by_filename_stem(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"idle.png": _png(), "speaking.jpg": _jpg()}))
    res = resolve_local_expression_set([z])
    assert set(res.images) == {"idle", "speaking"}
    assert res.images["idle"] == _png()  # bytes preserved verbatim


def test_case_insensitive_stem(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"IDLE.PNG": _png(), "Thinking.png": _png()}))
    res = resolve_local_expression_set([z])
    assert set(res.images) == {"idle", "thinking"}


def test_non_matching_and_non_image_skipped(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"speaking.png": _png(), "notes.txt": b"hello", "random.png": _png()}))
    res = resolve_local_expression_set([z])
    assert set(res.images) == {"speaking"}
    assert any(name == "notes.txt" for name, _ in res.skipped)  # not an image / no state


def test_bad_image_bytes_skipped_with_reason(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"error.png": b"not-an-image"}))
    res = resolve_local_expression_set([z])
    assert "error" not in res.images
    assert any(name == "error.png" for name, _ in res.skipped)


def test_two_files_one_state_prefers_png(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"speaking.jpg": _jpg(), "speaking.png": _png()}))
    res = resolve_local_expression_set([z])
    assert res.images["speaking"] == _png()  # .png wins
    assert res.notes  # tie recorded


def test_directory_of_images(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "idle.png").write_bytes(_png())
    (d / "error.png").write_bytes(_png())
    res = resolve_local_expression_set([d])
    assert set(res.images) == {"idle", "error"}


def test_list_of_image_files(tmp_path):
    a = tmp_path / "thinking.png"; a.write_bytes(_png())
    b = tmp_path / "speaking.png"; b.write_bytes(_png())
    res = resolve_local_expression_set([a, b])
    assert set(res.images) == {"thinking", "speaking"}


def test_member_count_cap(tmp_path):
    z = tmp_path / "big.zip"
    z.write_bytes(_zip({f"file{i}.png": _png() for i in range(MAX_ZIP_MEMBERS + 5)}))
    res = resolve_local_expression_set([z])
    # capped: does not read all members; the 4 states may still resolve, but the
    # resolution notes/skips the cap. At minimum it must not raise and must be bounded.
    assert isinstance(res.images, dict)
    assert res.notes  # cap recorded


def test_total_size_cap_rejects(tmp_path, monkeypatch):
    # One member whose declared uncompressed size exceeds the total cap is skipped.
    import tldw_chatbook.Character_Chat.expression_set_io as mod
    monkeypatch.setattr(mod, "MAX_TOTAL_BYTES", 100)
    z = tmp_path / "bomb.zip"
    z.write_bytes(_zip({"idle.png": _png() * 50}))  # > 100 bytes uncompressed
    res = resolve_local_expression_set([z])
    assert "idle" not in res.images
    assert res.notes or res.skipped


def test_not_a_zip_fails_cleanly(tmp_path):
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"this is not a zip")
    res = resolve_local_expression_set([bad])  # must not raise
    assert res.images == {}
    assert res.skipped or res.notes
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_expression_set_io.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Character_Chat.expression_set_io`.

- [ ] **Step 3: Create the module + resolver**

Create `tldw_chatbook/Character_Chat/expression_set_io.py`:
```python
"""Pure import/export logic for a character's expression image SET
(idle/thinking/speaking/error). No Textual and no DB-module imports -- takes a
db object as a parameter where needed. Reused by P3d-3's .vpack extractor.
"""
from __future__ import annotations

import io
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run the same command as Step 2. Expected: PASS (11 tests). Also `python -c "import tldw_chatbook.app"` (env prefix) exits 0.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/expression_set_io.py Tests/Character_Chat/test_expression_set_io.py
git commit -m "feat(personas): P3d-2 Task 1 — resolve_local_expression_set (zip/dir/files, security-capped)"
```

---

## Task 2: `build_expression_set_zip` + round-trip (pure)

**Files:**
- Modify: `tldw_chatbook/Character_Chat/expression_set_io.py`
- Test: `Tests/Character_Chat/test_expression_set_io.py` (extend)

**Interfaces:**
- Consumes: `resolve_local_expression_set` (Task 1).
- Produces: `build_expression_set_zip(character_name: str, images: dict[str, bytes]) -> bytes`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Character_Chat/test_expression_set_io.py`:
```python
def test_build_zip_round_trips_through_resolver(tmp_path):
    from tldw_chatbook.Character_Chat.expression_set_io import build_expression_set_zip
    images = {"idle": _png(), "speaking": _jpg()}
    blob = build_expression_set_zip("Ada Lovelace", images)
    out = tmp_path / "ada.zip"
    out.write_bytes(blob)
    res = resolve_local_expression_set([out])
    assert set(res.images) == {"idle", "speaking"}


def test_build_zip_uses_detected_extension(tmp_path):
    from tldw_chatbook.Character_Chat.expression_set_io import build_expression_set_zip
    blob = build_expression_set_zip("Ada", {"speaking": _jpg()})
    names = zipfile.ZipFile(io.BytesIO(blob)).namelist()
    assert "speaking.jpg" in names          # JPEG bytes -> .jpg, not .png
    assert "expression_set.json" in names    # provenance marker present


def test_build_zip_empty_set_is_valid_zip():
    from tldw_chatbook.Character_Chat.expression_set_io import build_expression_set_zip
    blob = build_expression_set_zip("Ada", {})
    zf = zipfile.ZipFile(io.BytesIO(blob))
    assert zf.namelist() == ["expression_set.json"]
```

- [ ] **Step 2: Run to verify it fails**

Run the file (Step-2 command from Task 1). Expected: FAIL — `build_expression_set_zip` not defined.

- [ ] **Step 3: Implement the exporter**

Add to `expression_set_io.py`:
```python
import json

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
```

- [ ] **Step 4: Run to verify it passes**

Run the file. Expected: PASS (14 tests total).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/expression_set_io.py Tests/Character_Chat/test_expression_set_io.py
git commit -m "feat(personas): P3d-2 Task 2 — build_expression_set_zip (format-detected, round-trips)"
```

---

## Task 3: `apply_expression_images_to_db` helper + `_apply_expression_set` orchestrator

**Files:**
- Modify: `tldw_chatbook/Character_Chat/expression_set_io.py` (the DB helper)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (the orchestrator — READ FRESH; `_apply_expression_upload` ~:4393, `_render_character_expression_slot` ~:4271, `_character_editor_generation`, `_render_character_editor_avatar`/avatar-thumb render)
- Test: `Tests/Character_Chat/test_expression_set_io.py` (helper) + `Tests/UI/test_personas_expression_slots.py` (orchestrator)

**Interfaces:**
- Consumes: `db.set_character_expression_image(character_id, state, bytes, mime)` (P3d-1); `EXPRESSION_IMAGE_STATES`; `editor.set_avatar_image(bytes)`.
- Produces:
  - `apply_expression_images_to_db(db, character_id: int, images: dict[str, bytes]) -> tuple[list[str], list[tuple[str, str]]]` (applied, skipped) — writes only `EXPRESSION_IMAGE_STATES` (ignores idle).
  - `@dataclass(frozen=True) class ExpressionSetApplyResult: applied: list[str]; skipped: list[tuple[str, str]]`
  - `_apply_expression_set(self, character_id: int, images: dict[str, bytes]) -> ExpressionSetApplyResult` (screen method).

- [ ] **Step 1: Write the failing test (DB helper)**

Append to `Tests/Character_Chat/test_expression_set_io.py`:
```python
@pytest.fixture
def db(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    return CharactersRAGDB(tmp_path / "expr.db", "test-client")   # file-backed, not :memory:


def test_apply_images_to_db_writes_only_non_idle(db):
    from tldw_chatbook.Character_Chat.expression_set_io import apply_expression_images_to_db
    cid = db.add_character_card({"name": "Ada"})
    applied, skipped = apply_expression_images_to_db(
        db, cid, {"idle": _png(), "speaking": _png(), "thinking": _png()}
    )
    assert set(applied) == {"speaking", "thinking"}     # idle NOT written to the table
    assert db.get_character_expression_image(cid, "speaking") is not None
    assert db.get_character_expression_image(cid, "idle") is None


def test_apply_images_to_db_best_effort(db, monkeypatch):
    from tldw_chatbook.Character_Chat.expression_set_io import apply_expression_images_to_db
    cid = db.add_character_card({"name": "Ada"})
    orig = db.set_character_expression_image
    def boom(c, s, i, m=None):
        if s == "error":
            raise RuntimeError("disk full")
        return orig(c, s, i, m)
    monkeypatch.setattr(db, "set_character_expression_image", boom)
    applied, skipped = apply_expression_images_to_db(db, cid, {"speaking": _png(), "error": _png()})
    assert applied == ["speaking"]
    assert any(s == "error" for s, _ in skipped)
```

- [ ] **Step 2: Run to verify it fails**

Run the file (Task 1 Step-2 command). Expected: FAIL — `apply_expression_images_to_db` not defined.

- [ ] **Step 3: Implement the DB helper**

Add to `expression_set_io.py`:
```python
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
```

- [ ] **Step 4: Write the failing test (orchestrator)**

Append to `Tests/UI/test_personas_expression_slots.py` (reuse its existing editor harness — grep the file for how it mounts the editor for a saved character; mirror that). Adjust fixture/helper names to the file's:
```python
@pytest.mark.asyncio
async def test_apply_expression_set_stages_idle_and_writes_three(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    import io as _io
    from PIL import Image as _Img
    def _png(c=(1, 2, 3)):
        b = _io.BytesIO(); _Img.new("RGB", (8, 8), c).save(b, format="PNG"); return b.getvalue()

    result = await screen._apply_expression_set(
        char_id, {"idle": _png((9, 9, 9)), "speaking": _png(), "thinking": _png()}
    )
    # idle STAGED in the editor (not the table); three -> table
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import PersonasCharacterEditorWidget
    editor = screen.query_one(PersonasCharacterEditorWidget)
    assert editor.current_avatar_bytes() == _png((9, 9, 9))      # idle staged
    assert db.get_character_expression_image(char_id, "speaking") is not None
    assert db.get_character_expression_image(char_id, "idle") is None
    assert set(result.applied) >= {"idle", "speaking", "thinking"}
```

- [ ] **Step 5: Run to verify it fails**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_expression_slots.py::test_apply_expression_set_stages_idle_and_writes_three -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `_apply_expression_set` not defined.

- [ ] **Step 6: Implement the orchestrator**

In `personas_screen.py`, add `_apply_expression_set` (place it near `_apply_expression_upload`). It bumps `_character_editor_generation` **once** (NOT per state — per-state bumps race the re-renders) and re-renders the affected slots + the avatar:
```python
    async def _apply_expression_set(self, character_id: int, images: dict):
        """Apply a resolved expression set: idle staged in the editor (persists on
        card save), the three reactive states written immediately. Bumps the render
        token ONCE, then re-renders the affected slots + the avatar thumbnail."""
        from ...Character_Chat.expression_set_io import (
            apply_expression_images_to_db,
            ExpressionSetApplyResult,
        )
        applied: list[str] = []
        skipped: list = []
        db = getattr(self.app_instance, "chachanotes_db", None)
        # idle -> stage in the editor (like a manual avatar upload)
        idle = images.get("idle")
        editor = None
        try:
            editor = self.query_one(PersonasCharacterEditorWidget)
        except QueryError:
            editor = None
        if idle and editor is not None:
            editor.set_avatar_image(idle)
            applied.append("idle")
        # three -> DB (immediate), off-thread
        if db is not None:
            db_applied, db_skipped = await asyncio.to_thread(
                apply_expression_images_to_db, db, character_id, images
            )
            applied.extend(db_applied)
            skipped.extend(db_skipped)
        # single generation bump, then ONE re-render of the avatar + all 3 slots
        self._character_editor_generation += 1
        await self._render_all_character_editor_thumbnails(character_id)
        return ExpressionSetApplyResult(applied=applied, skipped=skipped)
```
NOTE: `_render_all_character_editor_thumbnails(self, character_id: int | None)` (personas_screen ~:4244) re-renders the avatar thumbnail + all three expression-slot thumbnails, and does NOT bump the token itself — so bump ONCE above, then call it once; all re-renders share the fresh token. Do NOT call `_apply_expression_upload` per state (it bumps the token each time → drops the prior slot's in-flight render — the P3d-1 render-race). READ FRESH to confirm the method name/signature hasn't drifted.

- [ ] **Step 7: Run both test files to verify pass**

Run the DB-helper file and the UI file (separate invocations — different rootdirs):
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_expression_set_io.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_expression_slots.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS both. Then `python -c "import tldw_chatbook.app"`.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Character_Chat/expression_set_io.py tldw_chatbook/UI/Screens/personas_screen.py Tests/Character_Chat/test_expression_set_io.py Tests/UI/test_personas_expression_slots.py
git commit -m "feat(personas): P3d-2 Task 3 — apply_expression_images_to_db + _apply_expression_set orchestrator"
```

---

## Task 4: Import / Export buttons + workers

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py` (2 messages)
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` (2 buttons + post_message; READ the expression-slot compose + the `Button.Pressed`/action that posts `CharacterExpressionUploadRequested` ~:1036 to mirror)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (handlers + import worker + import path method + export worker; READ `_expression_upload_dialog_worker` ~:4485, `_import_dialog_worker` ~:4622, `_dictionary_export_worker`, `_io_dialog_active`, `_character_editor_is_active`, `_notify`)
- Test: `Tests/UI/test_personas_expression_slots.py` (extend)

**Interfaces:**
- Consumes: `_apply_expression_set` (Task 3); `resolve_local_expression_set` / `build_expression_set_zip` (Tasks 1-2); `editor.expression_character_id()`, `editor.current_avatar_bytes()`, `db.get_character_expression_image`.
- Produces: `CharacterExpressionSetImportRequested`, `CharacterExpressionSetExportRequested` messages; `_import_expression_set_from_path(character_id, path)` (dialog-free, testable) + `_export_expression_set(character_id, name)` (dialog-free) + their workers/handlers.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_personas_expression_slots.py` (mirror the file's editor harness). Test the **dialog-free** path methods directly (the P3d-1 pattern):
```python
@pytest.mark.asyncio
async def test_import_expression_set_from_zip_path(personas_editor_with_saved_character, tmp_path):
    app, screen, db, char_id = personas_editor_with_saved_character
    from tldw_chatbook.Character_Chat.expression_set_io import build_expression_set_zip
    import io as _io
    from PIL import Image as _Img
    def _png(): b=_io.BytesIO(); _Img.new("RGB",(8,8)).save(b,format="PNG"); return b.getvalue()
    z = tmp_path / "set.zip"
    z.write_bytes(build_expression_set_zip("Ada", {"idle": _png(), "speaking": _png()}))

    await screen._import_expression_set_from_path(char_id, str(z))

    assert db.get_character_expression_image(char_id, "speaking") is not None
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import PersonasCharacterEditorWidget
    assert screen.query_one(PersonasCharacterEditorWidget).current_avatar_bytes() is not None  # idle staged


@pytest.mark.asyncio
async def test_export_expression_set_writes_a_zip(personas_editor_with_saved_character):
    app, screen, db, char_id = personas_editor_with_saved_character
    import io as _io
    from PIL import Image as _Img
    def _png(): b=_io.BytesIO(); _Img.new("RGB",(8,8)).save(b,format="PNG"); return b.getvalue()
    db.set_character_expression_image(char_id, "speaking", _png())
    target = await screen._export_expression_set(char_id, "Ada")
    assert target is not None
    from pathlib import Path
    import zipfile
    assert zipfile.is_zipfile(Path(target))
    assert "speaking.png" in zipfile.ZipFile(target).namelist()
```

- [ ] **Step 2: Run to verify it fails**

Run the UI file (Task 3 Step-7 UI command). Expected: FAIL — the path methods don't exist.

- [ ] **Step 3: Add the two messages**

In `personas_pane_messages.py`, mirror `CharacterExpressionUploadRequested` (no payload needed):
```python
class CharacterExpressionSetImportRequested(Message):
    """Roleplay P3d-2: import a whole expression set from a .zip."""


class CharacterExpressionSetExportRequested(Message):
    """Roleplay P3d-2: export the character's expression set to a .zip."""
```

- [ ] **Step 4: Add the two buttons**

In `personas_character_editor_widget.py`, next to the expression slots, add an "Import set…" and "Export set…" button; on press, `self.post_message(CharacterExpressionSetImportRequested())` / `...ExportRequested()` (mirror the `CharacterExpressionUploadRequested(state)` post at ~:1036). Import the two message classes at the top (~:27).

- [ ] **Step 5: Add the handlers + path methods + workers**

In `personas_screen.py`:
```python
    @on(CharacterExpressionSetImportRequested)
    def _handle_expression_set_import_requested(self, message) -> None:
        message.stop()
        if not self._character_editor_is_active():
            self._notify("Open a character editor before importing an expression set.", "warning")
            return
        editor = self.query_one(PersonasCharacterEditorWidget)
        character_id = editor.expression_character_id()
        if character_id is None:
            self._notify("Save the character to import an expression set.", "warning")
            return
        if self._io_dialog_active:
            return
        self._io_dialog_active = True
        self.run_worker(self._expression_set_import_dialog_worker(character_id), group="personas-io")

    async def _expression_set_import_dialog_worker(self, character_id: int) -> None:
        from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters
        try:
            picker = EnhancedFileOpen(
                title="Import Expression Set (.zip)",
                filters=Filters(("Zip Archives", lambda p: p.suffix.lower() == ".zip")),
                context="character_expression_set_import",
            )
            try:
                file_path = await self.app.push_screen_wait(picker)
            except Exception:
                logger.opt(exception=True).warning("Could not show the expression-set import dialog.")
                return
            if file_path:
                await self._import_expression_set_from_path(character_id, str(file_path))
        finally:
            self._io_dialog_active = False

    async def _import_expression_set_from_path(self, character_id: int, path: str) -> None:
        from ...Character_Chat.expression_set_io import resolve_local_expression_set
        from pathlib import Path
        res = await asyncio.to_thread(resolve_local_expression_set, [Path(path)])
        if not res.images:
            reason = "; ".join(n for n, _ in res.skipped[:2]) or "; ".join(res.notes[:1]) or "no matching images"
            self._notify(f"Nothing imported ({reason}).", "warning")
            return
        result = await self._apply_expression_set(character_id, res.images)
        applied = ", ".join(result.applied) or "nothing"
        note = " — save the character to keep idle" if "idle" in result.applied else ""
        self._notify(f"Imported: {applied}.{note}", "information")

    @on(CharacterExpressionSetExportRequested)
    def _handle_expression_set_export_requested(self, message) -> None:
        message.stop()
        if not self._character_editor_is_active():
            return
        editor = self.query_one(PersonasCharacterEditorWidget)
        character_id = editor.expression_character_id()
        if character_id is None:
            self._notify("Save the character before exporting its expression set.", "warning")
            return
        name = str(self.state.selected_entity_name or "character")   # screen state (mirrors _dictionary_export_worker); the editor has NO name accessor
        self.run_worker(self._export_expression_set_worker(character_id, name), group="personas-io", exit_on_error=False)

    async def _export_expression_set_worker(self, character_id: int, name: str) -> None:
        try:
            target = await self._export_expression_set(character_id, name)
            if target:
                self._notify(f"Expression set exported to {target}.", "information")
        except Exception as exc:
            logger.opt(exception=True).error(f"Expression-set export failed: {exc}")
            self._notify(f"Export failed: {exc}", "error")

    async def _export_expression_set(self, character_id: int, name: str) -> str | None:
        """Build the export zip and write it to the exports dir (atomic). Returns the path."""
        from ...Character_Chat.expression_set_io import build_expression_set_zip
        images: dict[str, bytes] = {}
        try:
            editor = self.query_one(PersonasCharacterEditorWidget)
            idle = editor.current_avatar_bytes()
            if idle:
                images["idle"] = idle
        except QueryError:
            pass
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is not None:
            for state in ("thinking", "speaking", "error"):
                data = await asyncio.to_thread(db.get_character_expression_image, character_id, state)
                if data:
                    images[state] = data
        if not images:
            self._notify("This character has no expression images to export.", "warning")
            return None
        blob = await asyncio.to_thread(build_expression_set_zip, name, images)
        # Mirror _dictionary_export_worker: exports dir + atomic temp-replace.
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "character"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exports_dir = get_user_data_dir() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        target = exports_dir / f"{slug}-expressions-{stamp}.zip"
        temp = exports_dir / f".{slug}-expressions-{stamp}.zip.tmp"
        temp.write_bytes(blob)
        temp.replace(target)
        return str(target)
```
NOTE: verify the exact names `_character_editor_is_active`, `_notify`, `get_user_data_dir`, `PersonasCharacterEditorWidget`, `QueryError`, `re`, `datetime` imports already exist in `personas_screen.py` (they do — `_dictionary_export_worker` uses them). The character NAME comes from `self.state.selected_entity_name` (the screen state `_dictionary_export_worker` uses) — the editor widget has NO name accessor.

- [ ] **Step 6: Run to verify pass**

Run the UI file. Expected: PASS. Then run the DB module file + `python -c "import tldw_chatbook.app"`. Also re-run `Tests/UI/test_personas_character_editor_avatar.py` for no regression.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_expression_slots.py
git commit -m "feat(personas): P3d-2 Task 4 — Import/Export expression set buttons + workers"
```

---

## Self-Review (author)

- **Spec coverage:** resolver + security (Task 1) ✓; exporter + round-trip + format-detection (Task 2) ✓; DB helper + orchestrator with idle-staged/3-immediate + single-generation-bump (Task 3) ✓; UI buttons + single-.zip import + exports-dir export + saved-char/characters-only gate + best-effort summary (Task 4) ✓. `.vpack` extractor explicitly deferred to P3d-3 (no task). No migration (no task) ✓.
- **Type consistency:** `resolve_local_expression_set(list[Path]) -> ExpressionSetResolution{images,skipped,notes}` (T1) consumed in T4; `build_expression_set_zip(str, dict) -> bytes` (T2) consumed in T4; `apply_expression_images_to_db(db, int, dict) -> (applied, skipped)` (T3) consumed by `_apply_expression_set` (T3) consumed in T4; `ExpressionSetApplyResult{applied, skipped}` consistent T3→T4. `EXPRESSION_STATES`/`EXPRESSION_IMAGE_STATES` imported from `console_expression_state` throughout.
- **Placeholder scan:** every code step shows code; the "READ FRESH / confirm the exact name" notes point at specific named methods (avatar-thumb render method, editor name accessor) the implementer must bind — flagged because `personas_screen.py` drifts under concurrent edits, not vague hand-waving.
- **Known drift:** all `personas_screen.py` / `personas_character_editor_widget.py` line numbers are approximate (concurrent edits) — every task says READ FRESH. The P3d-1 patterns (`_expression_upload_dialog_worker`, `_stage_character_expression_from_path`, `_dictionary_export_worker`, `_render_character_expression_slot`, `_character_editor_generation`, `editor.set_avatar_image/current_avatar_bytes/expression_character_id`) are verified against the current file.
