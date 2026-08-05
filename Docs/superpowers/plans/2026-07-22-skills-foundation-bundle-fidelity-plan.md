# Skills Foundation — Full-Bundle Supporting-File Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tldw's local-skills subsystem faithfully import, store, trust-verify, and export a spec-conformant Agent Skills bundle — arbitrary nested directories, binary files, and executable scripts — without disturbing any currently-trusted skill.

**Architecture:** Directory-native (the skill dir on disk is the source of truth). The trust scanner becomes recursive, binary/mode-aware, and junk-pruning, with per-file SHA-256 kept inside the existing HMAC-authenticated manifest. `supporting_files` stays `dict[str,str]` (text, nested keys); a new metadata-only `bundle_files` list surfaces binaries. Binaries never cross the JSON wire.

**Tech Stack:** Python 3.11+, pydantic v2, `os.walk`, `zipfile`, `hashlib`/`hmac`, pytest. No new dependencies.

**Design doc:** `Docs/superpowers/specs/2026-07-22-skills-foundation-bundle-fidelity-design.md`.

## Global Constraints

- **Backward-compat (load-bearing):** the canonical trust snapshot MUST be byte-identical to the pre-change form for any skill with no nested paths, no binaries, and no executable bits. Achieve this by (a) sorting the recursive walk by **POSIX relative path**, (b) including `executable` in a fingerprint dict **only when `True`**, (c) leaving `file_type` unchanged for text.
- **Three fingerprint builders must change together:** `SkillFileFingerprint.as_manifest_entry` (`skill_trust_models.py`), the scanner constructor (`skill_trust_scanner.py`), and `_content_manifest_entry` (`skill_trust_service.py`). They must produce identical dicts for the same file.
- **Binaries never enter `supporting_files`** (the `dict[str,str]` text view) or cross the JSON boundary. They live on disk and are surfaced via `bundle_files` metadata only.
- **Path rules:** relative POSIX subpath; validate **each segment** against `^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$`; reject absolute, leading `/`, backslash, `..`/`.`/empty segments, symlinks, any-case `skill.md`, depth > 8, total length > 255.
- **Size caps:** per-file 5 MB, per-bundle total 25 MB, max 500 files. Raise the legacy `MAX_SUPPORTING_FILES_COUNT`/`MAX_SUPPORTING_FILE_BYTES`/`MAX_SUPPORTING_FILES_TOTAL_BYTES` to match.
- **Junk ignore-list:** dirs `.git`, `.github`, `.hg`, `.svn`, `node_modules`, `__pycache__`; files `.DS_Store`, `Thumbs.db`; suffixes `.pyc`, `.pyo`, `~`, `.bak`, `.orig`, `.tmp`, `.swp`, `.part`. Pruned entries are skipped entirely (never recorded in `unsupported_paths`).
- **Tolerate-and-surface:** trust operations never hard-raise on unsupported files; they trust the supported files and leave residuals as recoverable `needs-review`.
- **Symlinks are skip-not-fail** on import; never followed on scan (`os.walk(followlinks=False)`).
- **venv-only tests:** run pytest via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`. Force the file-marker keyring fallback where trust is exercised: prefix `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`.
- **No trust-data migration; no config change.** Additive fields only.

## File Structure

| File | Change |
|------|--------|
| `tldw_chatbook/tldw_api/skills_schemas.py` | New `validate_supporting_file_path`; per-segment nested validation in `_validate_supporting_files`; raised caps; new `BundleFileInfo` model + `bundle_files` field on `SkillResponse` |
| `tldw_chatbook/Skills_Interop/skill_trust_models.py` | `executable` field on `SkillFileFingerprint`; conditional `as_manifest_entry` |
| `tldw_chatbook/Skills_Interop/skill_trust_scanner.py` | Recursive `os.walk`, junk ignore-list, per-segment validation, relative-path body classification, binary fingerprint, exec bit, symlink skip |
| `tldw_chatbook/Skills_Interop/skill_trust_service.py` | `_content_manifest_entry` binary+exec parity; `bootstrap_trust`/`trust_current_skill` tolerate-and-surface |
| `tldw_chatbook/Skills_Interop/local_skills_service.py` | Recursive binary-safe `_read_supporting_files` + `_read_bundle_manifest`; bytes-mode atomic writer; `_apply_supporting_files` guard; `import_skill_directory`; nested/binary zip import + export; junk prune; caps |
| `tldw_chatbook/UI/Screens/library_screen.py` | Folder import calls `import_skill_directory` |
| `tldw_chatbook/Widgets/Library/library_skills_canvas.py` | Review preview binary-vs-deleted; editor lists nested + view-only binaries |
| `Tests/Skills/`, `Tests/Library/`, `Tests/UI/` | New tests; flip `test_skills_import.py:525-527` |

---

### Task 1: Path validator, `BundleFileInfo` schema, raised caps

**Files:**
- Modify: `tldw_chatbook/tldw_api/skills_schemas.py:10-56` (patterns, caps, `_validate_supporting_files`) and the `SkillResponse` class
- Test: `Tests/tldw_api/test_skills_schemas_bundle.py` (create)

**Interfaces:**
- Produces: `validate_supporting_file_path(path: str) -> str` (returns normalized POSIX path, raises `ValueError`); `SEGMENT_PATTERN` (the per-segment regex); `BundleFileInfo` (pydantic: `path: str`, `size: int`, `executable: bool`, `is_text: bool`); constants `MAX_SUPPORTING_FILES_COUNT=500`, `MAX_SUPPORTING_FILE_BYTES=5*1024*1024`, `MAX_SUPPORTING_FILES_TOTAL_BYTES=25*1024*1024`, `MAX_SUPPORTING_FILE_PATH_DEPTH=8`, `MAX_SUPPORTING_FILE_PATH_LEN=255`.

- [ ] **Step 1: Write the failing test**

Create `Tests/tldw_api/test_skills_schemas_bundle.py`:

```python
import pytest

from tldw_chatbook.tldw_api.skills_schemas import (
    validate_supporting_file_path,
    BundleFileInfo,
    MAX_SUPPORTING_FILES_COUNT,
)


@pytest.mark.parametrize("good", [
    "notes.md",
    "scripts/build.sh",
    "references/api/reference.md",
    "assets/img/logo.png",
])
def test_validate_supporting_file_path_accepts_nested(good):
    assert validate_supporting_file_path(good) == good


@pytest.mark.parametrize("bad", [
    "/abs.md",                # absolute
    "../escape.md",           # traversal
    "a/../b.md",              # traversal segment
    "a/./b.md",               # dot segment
    "a//b.md",                # empty segment
    "back\\slash.md",         # backslash
    "SKILL.md",               # reserved body (root)
    "refs/SKILL.md",          # nested shadow (any case)
    "refs/skill.md",          # nested shadow wrong-case
    "-leading-dash.md",       # segment must start alnum
    ".dotfile",               # leading dot
    "a/" * 9 + "deep.md",     # depth > 8
    "x" * 256,                # length > 255
])
def test_validate_supporting_file_path_rejects(bad):
    with pytest.raises(ValueError):
        validate_supporting_file_path(bad)


def test_bundle_file_info_shape():
    info = BundleFileInfo(path="assets/logo.png", size=1234, executable=False, is_text=False)
    assert info.model_dump() == {
        "path": "assets/logo.png", "size": 1234, "executable": False, "is_text": False,
    }


def test_caps_raised():
    assert MAX_SUPPORTING_FILES_COUNT == 500
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/tldw_api/test_skills_schemas_bundle.py -q`
Expected: FAIL — `ImportError: cannot import name 'validate_supporting_file_path'`.

- [ ] **Step 3: Implement**

In `skills_schemas.py`, replace the cap constants (lines 12-14) and add the validator + segment pattern after `SUPPORTING_FILE_NAME_PATTERN` (line 11):

```python
SUPPORTING_FILE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$")
SEGMENT_PATTERN = SUPPORTING_FILE_NAME_PATTERN  # each path segment obeys the same rule
MAX_SUPPORTING_FILES_COUNT = 500
MAX_SUPPORTING_FILE_BYTES = 5 * 1024 * 1024
MAX_SUPPORTING_FILES_TOTAL_BYTES = 25 * 1024 * 1024
MAX_SUPPORTING_FILE_PATH_DEPTH = 8
MAX_SUPPORTING_FILE_PATH_LEN = 255
_RESERVED_BODY_BASENAME = "skill.md"


def validate_supporting_file_path(path: str) -> str:
    """Validate a relative POSIX supporting-file subpath, returning it normalized.

    Args:
        path: Candidate relative POSIX path (e.g. ``scripts/build.sh``).

    Returns:
        The same path when valid.

    Raises:
        ValueError: On absolute paths, ``..``/``.``/empty segments, backslashes,
            a segment failing ``SEGMENT_PATTERN``, any-case ``skill.md`` basename,
            depth greater than ``MAX_SUPPORTING_FILE_PATH_DEPTH``, or a total
            length exceeding ``MAX_SUPPORTING_FILE_PATH_LEN``.
    """
    if not path or path != path.strip():
        raise ValueError(f"Invalid supporting file path: {path!r}")
    if "\\" in path or path.startswith("/"):
        raise ValueError(f"Invalid supporting file path: {path!r}")
    if len(path.encode("utf-8")) > MAX_SUPPORTING_FILE_PATH_LEN:
        raise ValueError(f"Supporting file path too long: {path!r}")
    segments = path.split("/")
    if len(segments) > MAX_SUPPORTING_FILE_PATH_DEPTH:
        raise ValueError(f"Supporting file path too deep: {path!r}")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise ValueError(f"Invalid path segment in {path!r}")
        if not SEGMENT_PATTERN.fullmatch(segment):
            raise ValueError(f"Invalid path segment {segment!r} in {path!r}")
    if segments[-1].lower() == _RESERVED_BODY_BASENAME:
        raise ValueError("SKILL.md is the skill body, not a supporting file")
    return path
```

Then rewrite the loop body of `_validate_supporting_files` (lines 36-53) to validate nested paths and count via the new caps:

```python
    for filename, content in value.items():
        validate_supporting_file_path(filename)  # replaces pattern + skill.md checks
        if content is None:
            if allow_deletes:
                continue
            raise ValueError(f"Supporting file {filename} cannot be null")
        non_null_count += 1
        if non_null_count > MAX_SUPPORTING_FILES_COUNT:
            raise ValueError(
                f"Too many supporting files ({non_null_count}); maximum is {MAX_SUPPORTING_FILES_COUNT}"
            )
        file_bytes = len(content.encode("utf-8"))
        if file_bytes > MAX_SUPPORTING_FILE_BYTES:
            raise ValueError(f"Supporting file {filename} exceeds file size limit")
        total_bytes += file_bytes
    if total_bytes > MAX_SUPPORTING_FILES_TOTAL_BYTES:
        raise ValueError("Total supporting files size exceeds bundle limit")
    return value
```

Add the `BundleFileInfo` model near the other schema classes and a `bundle_files` field on `SkillResponse` (which already has `model_config = ConfigDict(extra="allow")`):

```python
class BundleFileInfo(BaseModel):
    path: str
    size: int
    executable: bool = False
    is_text: bool = True
```
Add to `SkillResponse`: `bundle_files: list[BundleFileInfo] | None = None`.

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/tldw_api/test_skills_schemas_bundle.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/skills_schemas.py Tests/tldw_api/test_skills_schemas_bundle.py
git commit -m "feat(skills): path-aware supporting-file validator, BundleFileInfo, raised caps"
```

---

### Task 2: `executable` fingerprint field (conditional serialization)

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_models.py:32-49`
- Test: `Tests/Skills/test_skill_fingerprint_executable.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `SkillFileFingerprint(relative_path, file_type, byte_length, sha256, executable=False)`; `as_manifest_entry()` includes `"executable": True` **only when** `executable` is True.

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillFileFingerprint


def test_manifest_entry_omits_executable_when_false():
    fp = SkillFileFingerprint(
        relative_path="notes.md", file_type="supporting_text", byte_length=3, sha256="ab",
    )
    # BACKWARD-COMPAT: byte-identical to the pre-change 4-key dict.
    assert fp.as_manifest_entry() == {
        "relative_path": "notes.md", "file_type": "supporting_text",
        "byte_length": 3, "sha256": "ab",
    }


def test_manifest_entry_includes_executable_when_true():
    fp = SkillFileFingerprint(
        relative_path="scripts/run.sh", file_type="supporting_text",
        byte_length=3, sha256="ab", executable=True,
    )
    assert fp.as_manifest_entry()["executable"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_fingerprint_executable.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'executable'`.

- [ ] **Step 3: Implement**

Add the field (with default so existing constructors stay valid) and the conditional key:

```python
@dataclass(frozen=True, slots=True)
class SkillFileFingerprint:
    """Stable fingerprint metadata for one local skill file."""

    relative_path: str
    file_type: str
    byte_length: int
    sha256: str
    executable: bool = False

    def as_manifest_entry(self) -> dict[str, Any]:
        """Return the JSON-safe manifest representation for this fingerprint.

        The ``executable`` key is emitted ONLY when True so that a flat,
        all-text, non-executable skill serializes byte-identically to the
        pre-Spec-2 form (preserving existing manifests' HMAC validity).
        """
        entry = {
            "relative_path": self.relative_path,
            "file_type": self.file_type,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }
        if self.executable:
            entry["executable"] = True
        return entry
```

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_fingerprint_executable.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_models.py Tests/Skills/test_skill_fingerprint_executable.py
git commit -m "feat(skills): conditional executable field on SkillFileFingerprint"
```

---

### Task 3: Recursive, junk-aware, binary/mode-aware trust scanner

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_scanner.py` (whole `scan_skill_directory` + helpers)
- Test: `Tests/Skills/test_skill_trust_scanner_recursive.py` (create)

**Interfaces:**
- Consumes: `validate_supporting_file_path` (Task 1); `SkillFileFingerprint(..., executable=)` (Task 2).
- Produces: `scan_skill_directory(skill_name, skill_dir)` unchanged signature, now recursive; `SUPPORTING_JUNK_DIRS`, `SUPPORTING_JUNK_FILES`, `SUPPORTING_JUNK_SUFFIXES` constants; a file is `file_type="supporting_binary"` when non-UTF-8/null-byte, `"skill"` iff `relative_path == "SKILL.md"`, else `"supporting_text"`.

- [ ] **Step 1: Write the failing test**

```python
import os
from pathlib import Path

from tldw_chatbook.Skills_Interop.skill_trust_scanner import scan_skill_directory


def _write(p: Path, data: bytes, mode: int | None = None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if mode is not None:
        os.chmod(p, mode)


def test_recurses_and_classifies(tmp_path):
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"---\nname: demo\n---\nbody\n")
    _write(d / "references" / "api.md", b"# api\n")
    _write(d / "scripts" / "run.sh", b"#!/bin/sh\necho hi\n", mode=0o755)
    _write(d / "assets" / "logo.png", b"\x89PNG\x00\x01binary")
    snap = scan_skill_directory("demo", d)
    fps = {f.relative_path: f for f in snap.fingerprints}
    assert set(fps) == {"SKILL.md", "references/api.md", "scripts/run.sh", "assets/logo.png"}
    assert fps["SKILL.md"].file_type == "skill"
    assert fps["assets/logo.png"].file_type == "supporting_binary"
    assert "assets/logo.png" not in snap.text_files       # binary not decoded
    assert "references/api.md" in snap.text_files
    assert fps["scripts/run.sh"].executable is True
    assert snap.unsupported_paths == ()


def test_prunes_junk(tmp_path):
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"x")
    _write(d / ".git" / "config", b"junk")
    _write(d / "__pycache__" / "m.pyc", b"\x00pyc")
    _write(d / ".DS_Store", b"\x00")
    _write(d / "keep.md~", b"backup")
    snap = scan_skill_directory("demo", d)
    paths = {f.relative_path for f in snap.fingerprints}
    assert paths == {"SKILL.md"}
    assert snap.unsupported_paths == ()   # junk pruned, NOT recorded


def test_nested_skill_md_is_not_the_body(tmp_path):
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"body")
    _write(d / "references" / "SKILL.md", b"nested")
    snap = scan_skill_directory("demo", d)
    # A nested SKILL.md is rejected as a shadow (validate_supporting_file_path),
    # so it lands in unsupported_paths, never classified as the body.
    fps = {f.relative_path: f for f in snap.fingerprints}
    assert fps["SKILL.md"].file_type == "skill"
    assert "references/SKILL.md" not in fps
    assert "references/SKILL.md" in snap.unsupported_paths


def test_flat_snapshot_byte_identical_to_legacy_ordering(tmp_path):
    # Guards the backward-compat guarantee: a flat all-text skill's fingerprint
    # entries are in iterdir name-order and carry no executable key.
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"body")
    _write(d / "b.md", b"bbb")
    _write(d / "a.md", b"aaa")
    snap = scan_skill_directory("demo", d)
    entries = [f.as_manifest_entry() for f in snap.fingerprints]
    assert [e["relative_path"] for e in entries] == ["SKILL.md", "a.md", "b.md"]
    assert all("executable" not in e for e in entries)


def test_symlink_skipped_not_followed(tmp_path):
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"body")
    (d / "link.md").symlink_to(d / "SKILL.md")
    snap = scan_skill_directory("demo", d)
    assert "link.md" not in {f.relative_path for f in snap.fingerprints}
    assert "link.md" in snap.unsupported_paths
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_scanner_recursive.py -q`
Expected: FAIL (nested files missing / classified wrong; today's flat scan skips subdirs).

- [ ] **Step 3: Implement**

Rewrite `skill_trust_scanner.py`. Keep the `_SKILL_FILENAME`, `sha256_hex`, model imports. Replace `_TEMP_SUFFIXES` handling with a junk ignore-list, walk recursively, validate per path, fingerprint binaries:

```python
"""Deterministic local skill directory scanning for trust verification."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath

from .skill_trust_crypto import sha256_hex
from .skill_trust_models import SkillDirectorySnapshot, SkillFileFingerprint

_SKILL_FILENAME = "SKILL.md"
SUPPORTING_JUNK_DIRS = frozenset({".git", ".github", ".hg", ".svn", "node_modules", "__pycache__"})
SUPPORTING_JUNK_FILES = frozenset({".DS_Store", "Thumbs.db"})
SUPPORTING_JUNK_SUFFIXES = (".pyc", ".pyo", "~", ".bak", ".orig", ".tmp", ".swp", ".part")


def _is_junk(name: str) -> bool:
    return name in SUPPORTING_JUNK_FILES or name.lower().endswith(SUPPORTING_JUNK_SUFFIXES)


def _validate_relative_path(relative_path: str) -> bool:
    # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
    from ..tldw_api.skills_schemas import validate_supporting_file_path

    try:
        validate_supporting_file_path(relative_path)
        return True
    except ValueError:
        return False


def scan_skill_directory(skill_name: str, skill_dir: Path) -> SkillDirectorySnapshot:
    """Return a deterministic trust snapshot for a skill directory tree.

    Walks ``skill_dir`` recursively (never following symlinks), pruning
    VCS/OS/build junk, sorting by POSIX relative path. Text files are decoded
    into ``text_files``; binaries are fingerprinted (``file_type=
    "supporting_binary"``) but not decoded. Symlinks and files failing path
    validation are recorded in ``unsupported_paths`` (they are NOT trust
    material) but never raise here.

    Args:
        skill_name: Local skill name represented by the directory.
        skill_dir: Directory to scan.

    Returns:
        Snapshot with fingerprints (sorted by relative path), decoded text
        files, and unsupported path names.
    """
    fingerprints: list[SkillFileFingerprint] = []
    text_files: dict[str, str] = {}
    unsupported_paths: list[str] = []

    collected: list[tuple[str, Path]] = []
    for root, dirs, files in os.walk(skill_dir, followlinks=False):
        # Prune junk directories in-place so os.walk does not descend them.
        dirs[:] = sorted(d for d in dirs if d not in SUPPORTING_JUNK_DIRS)
        for name in files:
            if _is_junk(name):
                continue
            abs_path = Path(root) / name
            rel = PurePosixPath(abs_path.relative_to(skill_dir).as_posix())
            collected.append((str(rel), abs_path))
    collected.sort(key=lambda item: item[0])

    for relative_path, path in collected:
        if path.is_symlink() or not _validate_relative_path(relative_path):
            unsupported_paths.append(relative_path)
            continue
        try:
            if not path.is_file():
                unsupported_paths.append(relative_path)
                continue
            raw = path.read_bytes()
        except OSError:
            unsupported_paths.append(relative_path)
            continue

        is_binary = b"\x00" in raw
        text: str | None = None
        if not is_binary:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                is_binary = True

        if relative_path == _SKILL_FILENAME:
            file_type = "skill"
        elif is_binary:
            file_type = "supporting_binary"
        else:
            file_type = "supporting_text"

        executable = bool(path.stat().st_mode & stat.S_IXUSR)
        fingerprints.append(
            SkillFileFingerprint(
                relative_path=relative_path,
                file_type=file_type,
                byte_length=len(raw),
                sha256=sha256_hex(raw),
                executable=executable,
            )
        )
        if text is not None:
            text_files[relative_path] = text

    return SkillDirectorySnapshot(
        skill_name=skill_name,
        fingerprints=tuple(fingerprints),
        text_files=text_files,
        unsupported_paths=tuple(unsupported_paths),
    )
```

Note: the `SKILL.md` body keeps `executable=False` in practice (markdown), so its entry stays byte-identical.

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_scanner_recursive.py -q`
Expected: PASS.

- [ ] **Step 5: Regression — existing scanner tests still pass**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ -q -k "scanner or trust"`
Expected: PASS (backward-compat: flat all-text snapshots unchanged).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_scanner.py Tests/Skills/test_skill_trust_scanner_recursive.py
git commit -m "feat(skills): recursive junk-aware binary/mode trust scanner"
```

---

### Task 4: Fix `verify_skill_content` so binary/executable files don't false-block execute

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_service.py:399-410` (`verify_skill_content` comparison)
- Test: `Tests/Skills/test_verify_content_binary.py` (create)

**Interfaces:**
- Consumes: scanner (Task 3), `bootstrap_trust` (Task 5 tolerance not required here), the manifest's per-file `as_manifest_entry` dicts (which now carry `file_type="supporting_binary"` and `executable`).
- Produces: `verify_skill_content` no longer raises when the trusted manifest contains binary or executable files that the text-only in-memory reconstruction cannot represent.

**Context (the bug this fixes):** `verify_skill_content` (`:399-410`) runs on **every** `execute_skill`. It builds `current_files` from `_fingerprint_in_memory_files(skill_content, supporting_files)` where `supporting_files` is the **text-only** view (binaries excluded; executable bit unknown). It then compares against `trusted_files` (the full manifest, which post-Spec-2 includes binary + executable fingerprints). So `missing = trusted − current` would contain every binary file, and `modified` would contain every executable text file (manifest has `executable:True`, in-memory omits it) → `verify_skill_content` raises `SkillTrustBlockedError` on every run. The authoritative full-bundle check already happens in `ensure_skill_trusted` (`:371`) → `status_for_skill` → recursive disk scan (binary/exec included). So the in-memory check must be **scoped to files it can faithfully reconstruct** (text, non-executable) and leave binary/exec/full-set verification to the disk scan.

- [ ] **Step 1: Write the failing test**

```python
import secrets, pytest
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore, SkillTrustStore,
)


def _svc(tmp_path):
    (tmp_path / "trust").mkdir()
    (tmp_path / "skills").mkdir()
    return SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "trust" / "m.json"),
        ),
        key_cache=None,
    )


def test_verify_content_tolerates_binary_and_exec(tmp_path):
    import os, stat
    d = tmp_path / "skills" / "demo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("body\n", encoding="utf-8")
    (d / "assets").mkdir()
    (d / "assets" / "logo.png").write_bytes(b"\x89PNG\x00bin")   # binary → in manifest, not in text view
    (d / "scripts").mkdir()
    (d / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(d / "scripts" / "run.sh", 0o755)                    # executable text
    svc = _svc(tmp_path)
    svc.bootstrap_trust("pw", salt=secrets.token_bytes(32))      # trusts the full bundle
    # verify_skill_content receives the TEXT view (no binary; run.sh present but mode unknown).
    svc.verify_skill_content(
        "demo",
        skill_content="body\n",
        supporting_files={"scripts/run.sh": "#!/bin/sh\n"},
    )   # must NOT raise (was raising: assets/logo.png "missing", scripts/run.sh "modified")


def test_verify_content_still_catches_modified_text_body(tmp_path):
    d = tmp_path / "skills" / "demo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("body\n", encoding="utf-8")
    svc = _svc(tmp_path)
    svc.bootstrap_trust("pw", salt=secrets.token_bytes(32))
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustBlockedError
    with pytest.raises(SkillTrustBlockedError):
        svc.verify_skill_content("demo", skill_content="TAMPERED\n", supporting_files=None)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_verify_content_binary.py -q`
Expected: FAIL — `test_verify_content_tolerates_binary_and_exec` raises `SkillTrustBlockedError` (binary "missing", exec "modified").

- [ ] **Step 3: Implement**

Replace the comparison block (`:403-410`). Scope `missing`/`modified` to trusted files that the in-memory text view can faithfully reconstruct — i.e. exclude `supporting_binary` and executable entries (the disk scan in `ensure_skill_trusted` is the authority for those):

```python
        current_files = self._fingerprint_in_memory_files(
            skill_content=skill_content,
            supporting_files=supporting_files,
        )
        # The in-memory reconstruction is text-only and mode-blind. Binary and
        # executable files cannot be represented here, so scope the in-memory
        # comparison to reconstructable (text, non-executable) trusted files;
        # ensure_skill_trusted() above already verified the full on-disk bundle
        # (binaries + exec bits) against the manifest via a recursive scan.
        reconstructable = {
            path
            for path, entry in trusted_files.items()
            if entry.get("file_type") != "supporting_binary"
            and not entry.get("executable")
        }
        missing = reconstructable - set(current_files)
        added = set(current_files) - set(trusted_files)
        modified = {
            path
            for path in reconstructable & current_files.keys()
            if trusted_files[path] != current_files[path]
        }
        changed = tuple(sorted(missing | added | modified))
```

Leave the trailing `if not changed:` / raise logic (`:411+`) unchanged. Do NOT modify `_fingerprint_in_memory_files` or `_content_manifest_entry` — they correctly emit text entries (executable omitted), which now compare only against `reconstructable` trusted entries.

- [ ] **Step 4: Run to verify pass**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_verify_content_binary.py -q`
Expected: PASS (both tests: binary/exec tolerated; tampered text body still blocked).

- [ ] **Step 5: Regression**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_service.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_service.py Tests/Skills/test_verify_content_binary.py
git commit -m "fix(skills): scope in-memory verify to text files so binary/exec don't false-block execute"
```

---

### Task 5: Tolerate-and-surface unsupported files (no hard-raise)

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_service.py` — `bootstrap_trust` (the `if snapshot.unsupported_paths: raise` at ~:221) and `trust_current_skill` (~:507)
- Test: `Tests/Skills/test_trust_tolerates_unsupported.py` (create)

**Interfaces:**
- Consumes: scanner (Task 3).
- Produces: `bootstrap_trust`/`trust_current_skill` no longer raise `ValueError(TRUST_REASON_UNSUPPORTED_PATH)`; they trust the supported fingerprints and leave any residual unsupported file surfaced via `status_for_skill` (already returns `quarantined_unsupported_path`).

- [ ] **Step 1: Write the failing test**

```python
import secrets
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore, SkillTrustStore,
)


def _svc(tmp_path):
    (tmp_path / "trust").mkdir()
    (tmp_path / "skills").mkdir()
    return SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "trust" / "m.json"),
        ),
        key_cache=None,
    )


def test_bootstrap_does_not_raise_on_residual_unsupported(tmp_path):
    d = tmp_path / "skills" / "demo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("body\n", encoding="utf-8")
    (d / "good.md").write_text("ok\n", encoding="utf-8")
    (d / "link.md").symlink_to(d / "good.md")   # residual unsupported (symlink)
    svc = _svc(tmp_path)
    svc.bootstrap_trust("pw", salt=secrets.token_bytes(32))   # must NOT raise
    status = svc.status_for_skill("demo")
    assert status.trust_status == "quarantined_unsupported_path"  # surfaced, recoverable
    # Removing the residual clears it.
    (d / "link.md").unlink()
    svc.trust_current_skill("demo")   # must NOT raise
    assert svc.status_for_skill("demo").trust_status == "trusted"
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_trust_tolerates_unsupported.py -q`
Expected: FAIL — `ValueError: unsupported_path` raised by `bootstrap_trust`.

- [ ] **Step 3: Implement**

In `bootstrap_trust`, delete the `if snapshot.unsupported_paths: raise ValueError(TRUST_REASON_UNSUPPORTED_PATH)` guard (~:221-222) — proceed to build the manifest entry from `snapshot.fingerprints` (supported files) regardless. Do the same in `trust_current_skill` (~:507-508). The residual unsupported files remain visible because `status_for_skill` still returns `quarantined_unsupported_path` when `current.unsupported_paths` is non-empty (unchanged, ~:276-285). Do not remove the `TRUST_STATUS_QUARANTINED_UNSUPPORTED_PATH` status or `status_for_skill`'s handling.

- [ ] **Step 4: Run to verify pass**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_trust_tolerates_unsupported.py -q`
Expected: PASS.

- [ ] **Step 5: Regression — existing trust tests**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skill_trust_service.py -q`
Expected: PASS. If a test asserted the old hard-raise on unsupported paths, update it to assert the tolerate-and-surface behavior (quarantined status, no raise) — this is an intended behavior change per the spec.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Skills_Interop/skill_trust_service.py Tests/Skills/test_trust_tolerates_unsupported.py
git commit -m "feat(skills): tolerate-and-surface unsupported files instead of hard-raise"
```

---

### Task 6: Recursive binary-safe service read + bundle manifest + safe writers

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py` — `_read_supporting_files` (:391-405), `_read_text_preserving_newlines` (:407-419), `_apply_supporting_files` (:559-571), `_write_text_atomic` (:148-154), `_response_for_record` (:421-435)
- Test: `Tests/Skills/test_local_skills_bundle_io.py` (create)

**Interfaces:**
- Consumes: `validate_supporting_file_path`, `BundleFileInfo` (Task 1); junk constants (Task 3).
- Produces: `_read_supporting_files(skill_dir) -> dict[str,str] | None` (recursive, nested keys, TEXT only, never raises on binary); `_read_bundle_manifest(skill_dir) -> list[dict] | None` (all files metadata); `_write_bytes_atomic(path, data: bytes)`; `_apply_supporting_files` validates + contains each key.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.mark.asyncio
async def test_get_skill_with_nested_and_binary_never_raises(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    await svc.create_skill(name="demo", content="---\nname: demo\n---\nbody\n")
    d = svc._skill_dir("demo")
    (d / "references").mkdir(parents=True, exist_ok=True)
    (d / "references" / "api.md").write_text("# api\n", encoding="utf-8")
    (d / "assets").mkdir(parents=True, exist_ok=True)
    (d / "assets" / "logo.png").write_bytes(b"\x89PNG\x00binary")
    skill = await svc.get_skill("demo")      # must NOT raise
    assert skill["supporting_files"]["references/api.md"] == "# api\n"
    assert "assets/logo.png" not in skill["supporting_files"]   # binary excluded from text view
    paths = {b["path"]: b for b in skill["bundle_files"]}
    assert paths["assets/logo.png"]["is_text"] is False
    assert paths["references/api.md"]["is_text"] is True


def test_apply_supporting_files_rejects_traversal(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    d = svc._skill_dir("demo")
    d.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        svc._apply_supporting_files(d, {"../escape.md": "x"})
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_local_skills_bundle_io.py -q`
Expected: FAIL — `UnicodeDecodeError` from `get_skill` (top-level binary) and/or `bundle_files` KeyError.

- [ ] **Step 3: Implement**

Add a bytes-mode atomic writer beside `_write_text_atomic`:

```python
    @staticmethod
    def _write_bytes_atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("wb") as handle:
            handle.write(data)
        temp_path.replace(path)
```

Rewrite `_read_supporting_files` to walk recursively (junk-pruned, text-only, never raise), and add `_read_bundle_manifest`:

```python
    @staticmethod
    def _iter_bundle_files(skill_dir: Path):
        """Yield (relative_posix, abs_path) for every non-junk file, junk dirs pruned."""
        from .skill_trust_scanner import SUPPORTING_JUNK_DIRS, _is_junk  # reuse
        import os
        from pathlib import PurePosixPath
        if not skill_dir.exists():
            return
        for root, dirs, files in os.walk(skill_dir, followlinks=False):
            dirs[:] = [d for d in dirs if d not in SUPPORTING_JUNK_DIRS]
            for name in files:
                if _is_junk(name) or name == _SKILL_FILENAME and Path(root) == skill_dir:
                    continue
                abs_path = Path(root) / name
                rel = PurePosixPath(abs_path.relative_to(skill_dir).as_posix())
                yield str(rel), abs_path

    @staticmethod
    def _read_supporting_files(skill_dir: Path) -> dict[str, str] | None:
        from ..tldw_api.skills_schemas import validate_supporting_file_path
        supporting_files: dict[str, str] = {}
        for relative_path, path in sorted(
            LocalSkillsService._iter_bundle_files(skill_dir), key=lambda x: x[0]
        ):
            if path.is_symlink():
                continue
            try:
                validate_supporting_file_path(relative_path)
            except ValueError:
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw:
                continue
            try:
                supporting_files[relative_path] = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue   # binary — excluded from the text view, never raises
        return supporting_files or None

    @staticmethod
    def _read_bundle_manifest(skill_dir: Path) -> list[dict[str, Any]] | None:
        import stat
        from ..tldw_api.skills_schemas import validate_supporting_file_path
        manifest: list[dict[str, Any]] = []
        for relative_path, path in sorted(
            LocalSkillsService._iter_bundle_files(skill_dir), key=lambda x: x[0]
        ):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                validate_supporting_file_path(relative_path)
                raw = path.read_bytes()
            except (ValueError, OSError):
                continue
            is_text = b"\x00" not in raw
            if is_text:
                try:
                    raw.decode("utf-8")
                except UnicodeDecodeError:
                    is_text = False
            manifest.append({
                "path": relative_path,
                "size": len(raw),
                "executable": bool(path.stat().st_mode & stat.S_IXUSR),
                "is_text": is_text,
            })
        return manifest or None
```

Rewrite `_apply_supporting_files` to validate + contain each key and write bytes-safely for text (it only receives text via the editor):

```python
    @staticmethod
    def _apply_supporting_files(
        skill_dir: Path, supporting_files: dict[str, str | None] | None
    ) -> None:
        from ..tldw_api.skills_schemas import validate_supporting_file_path
        if supporting_files is None:
            return
        base = skill_dir.resolve()
        for filename, content in supporting_files.items():
            validate_supporting_file_path(filename)     # raises on traversal/bad name
            path = (skill_dir / filename)
            if base not in path.resolve().parents and path.resolve() != base:
                raise ValueError(f"unsafe supporting file path: {filename}")
            if content is None:
                if path.exists():
                    path.unlink()
                continue
            LocalSkillsService._write_text_atomic(path, content)
```

In `_response_for_record` (:431-435), add the bundle manifest:

```python
        response = SkillResponse(
            **record,
            content=content,
            supporting_files=self._read_supporting_files(skill_dir),
            bundle_files=self._read_bundle_manifest(skill_dir),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_local_skills_bundle_io.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/local_skills_service.py Tests/Skills/test_local_skills_bundle_io.py
git commit -m "feat(skills): recursive binary-safe read + bundle manifest + safe writers"
```

---

### Task 7: Faithful folder import (`import_skill_directory`) + UI wiring

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py` (add `import_skill_directory`); `tldw_chatbook/UI/Screens/library_screen.py:7267-7314` (folder branch)
- Test: `Tests/Skills/test_import_skill_directory.py` (create)

**Interfaces:**
- Consumes: `_iter_bundle_files`, `_write_bytes_atomic`, `validate_supporting_file_path`, caps (Tasks 1, 6).
- Produces: `async import_skill_directory(self, source_dir: Path, *, name: str, overwrite: bool = False, trust_approved: bool = False) -> dict[str,Any]` — faithful recursive copy (nested, binary, exec bit from real mode), junk-pruned, symlink-skipped, cap-enforced, trust-pending.

- [ ] **Step 1: Write the failing test**

```python
import os, stat, pytest
from pathlib import Path
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.mark.asyncio
async def test_import_skill_directory_faithful(tmp_path):
    src = tmp_path / "src" / "demo"
    (src / "scripts").mkdir(parents=True)
    (src / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")
    (src / "scripts" / "run.sh").write_bytes(b"#!/bin/sh\necho hi\n")
    os.chmod(src / "scripts" / "run.sh", 0o755)
    (src / "assets").mkdir()
    (src / "assets" / "logo.png").write_bytes(b"\x89PNG\x00bin")
    (src / ".git").mkdir()
    (src / ".git" / "config").write_text("junk", encoding="utf-8")

    svc = LocalSkillsService(store_dir=tmp_path / "store")
    await svc.import_skill_directory(src, name="demo")
    d = svc._skill_dir("demo")
    assert (d / "scripts" / "run.sh").read_bytes() == b"#!/bin/sh\necho hi\n"
    assert d.joinpath("scripts", "run.sh").stat().st_mode & stat.S_IXUSR
    assert (d / "assets" / "logo.png").read_bytes() == b"\x89PNG\x00bin"
    assert not (d / ".git").exists()          # junk pruned
    skill = await svc.get_skill("demo")
    assert skill["trust_status"] != "trusted"  # trust-pending
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_import_skill_directory.py -q`
Expected: FAIL — `AttributeError: 'LocalSkillsService' object has no attribute 'import_skill_directory'`.

- [ ] **Step 3: Implement**

Add to `LocalSkillsService`, modeled on `import_skill` (:779-825) but copying the tree faithfully:

```python
    async def import_skill_directory(
        self,
        source_dir: Path,
        *,
        name: str,
        overwrite: bool = False,
        trust_approved: bool = False,
    ) -> dict[str, Any]:
        import os, shutil, stat
        from pathlib import PurePosixPath
        from ..tldw_api.skills_schemas import (
            _normalize_skill_name, validate_supporting_file_path,
            MAX_SUPPORTING_FILES_COUNT, MAX_SUPPORTING_FILE_BYTES,
            MAX_SUPPORTING_FILES_TOTAL_BYTES,
        )
        self._enforce("skills.import.launch.local")
        skill_name = _normalize_skill_name(name)
        source_dir = Path(source_dir)
        body = source_dir / _SKILL_FILENAME
        if not body.is_file():
            raise ValueError("local_skill_missing_skill_md")
        # Collect the faithful file set (junk pruned, symlinks skipped, caps enforced).
        files: list[tuple[str, Path]] = []
        total = 0
        for relative_path, abs_path in self._iter_bundle_files(source_dir):
            if abs_path.is_symlink():
                continue                     # skip-not-fail
            try:
                validate_supporting_file_path(relative_path)
            except ValueError:
                continue
            size = abs_path.stat().st_size
            if size > MAX_SUPPORTING_FILE_BYTES:
                raise ValueError(f"local_skill_file_too_large:{relative_path}")
            total += size
            files.append((relative_path, abs_path))
        if len(files) > MAX_SUPPORTING_FILES_COUNT:
            raise ValueError("local_skill_too_many_files")
        if total > MAX_SUPPORTING_FILES_TOTAL_BYTES:
            raise ValueError("local_skill_bundle_too_large")
        async with self._lock:
            records = self._load_index()
            if skill_name in records and not overwrite:
                raise ValueError(f"local_skill_exists:{skill_name}")
            skill_dir = self._skill_dir(skill_name)
            if overwrite and skill_dir.exists():
                shutil.rmtree(skill_dir)
            skill_dir.mkdir(parents=True, exist_ok=True)
            existing = records.get(skill_name) if overwrite else None
            content = body.read_text(encoding="utf-8", errors="strict")
            self._write_text_atomic(skill_dir / _SKILL_FILENAME, content)
            for relative_path, abs_path in files:
                dest = skill_dir / PurePosixPath(relative_path)
                self._write_bytes_atomic(dest, abs_path.read_bytes())
                if abs_path.stat().st_mode & stat.S_IXUSR:
                    os.chmod(dest, dest.stat().st_mode | 0o755)
            record = self._metadata_from_content(
                name=skill_name, content=content, skill_dir=skill_dir, existing=existing,
            )
            if existing is not None:
                record["version"] = int(existing.get("version", 0)) + 1
            records[skill_name] = record
            self._save_index(records)
            self._trust_after_approved_mutation(skill_name, trust_approved=trust_approved)
            return self._response_for_record(record)
```

In `library_screen.py`, the directory branch of `_run_library_skills_import` (:7267-7314) currently reads flat siblings and calls `import_skill`. Replace it to call `import_skill_directory`:

```python
        if validated_path.is_dir():
            skill_dir = validated_path
            if self._find_skill_md_in_dir(skill_dir) is None:
                self._apply_library_skills_import_status("No SKILL.md found in that folder.")
                return
            import_skill_directory = getattr(service, "import_skill_directory", None)
            if not callable(import_skill_directory):
                self._apply_library_skills_import_status("Skill import is unavailable.")
                return
            skill_name = skill_dir.name
            try:
                await self._run_library_service_call(
                    import_skill_directory, skill_dir, mode="local",
                    name=skill_name, trust_approved=False, isolate_in_worker=True,
                )
            except Exception as exc:
                self._apply_library_skills_import_outcome_from_exception(skill_name, exc)
                return
            self._apply_library_skills_import_success(skill_name)
            return
```

Delete `_read_library_skill_import_supporting_files` if now unused (grep first). Keep the SKILL.md-file and loose-file branches unchanged.

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_import_skill_directory.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/local_skills_service.py tldw_chatbook/UI/Screens/library_screen.py Tests/Skills/test_import_skill_directory.py
git commit -m "feat(skills): faithful folder import (import_skill_directory) + UI wiring"
```

---

### Task 8: Faithful zip import (nested + binary + exec + hardening)

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py` — `_validate_archive_member` (:588-600), `import_skill_file` (:827-869)
- Test: `Tests/Skills/test_zip_import_bundle.py` (create)

**Interfaces:**
- Consumes: `validate_supporting_file_path`, `_write_bytes_atomic`, caps.
- Produces: `import_skill_file` extracts nested/binary/exec faithfully to `skill_dir`; `_validate_archive_member(name) -> str` allows nested relative paths, rejects traversal/absolute/reserved.

- [ ] **Step 1: Write the failing test**

```python
import io, zipfile, stat, pytest
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data, mode in members:
            info = zipfile.ZipInfo(name)
            if mode:
                info.external_attr = mode << 16
            z.writestr(info, data)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_zip_import_nested_binary_exec(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip([
        ("SKILL.md", b"---\nname: z\n---\nbody\n", 0),
        ("scripts/run.sh", b"#!/bin/sh\n", 0o755),
        ("assets/logo.png", b"\x89PNG\x00bin", 0),
    ])
    await svc.import_skill_file(data, filename="z.zip", content_type="application/zip")
    d = svc._skill_dir("z")
    assert (d / "scripts" / "run.sh").read_bytes() == b"#!/bin/sh\n"
    assert d.joinpath("scripts", "run.sh").stat().st_mode & stat.S_IXUSR
    assert (d / "assets" / "logo.png").read_bytes() == b"\x89PNG\x00bin"


@pytest.mark.asyncio
async def test_zip_slip_rejected(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    data = _zip([("SKILL.md", b"body", 0), ("../evil.md", b"x", 0)])
    with pytest.raises(ValueError):
        await svc.import_skill_file(data, filename="z.zip", content_type="application/zip")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_zip_import_bundle.py -q`
Expected: FAIL — current `_validate_archive_member` rejects `scripts/run.sh` (`len(parts) != 1`).

- [ ] **Step 3: Implement**

Replace `_validate_archive_member` to allow nested via the shared validator:

```python
    @staticmethod
    def _validate_archive_member(name: str) -> str:
        from ..tldw_api.skills_schemas import validate_supporting_file_path
        posix = PurePosixPath(name)
        if str(posix) == _SKILL_FILENAME:
            return _SKILL_FILENAME
        return validate_supporting_file_path(str(posix))
```

Rewrite the zip branch of `import_skill_file` (:849-869) to extract faithfully to disk (not through the text dict), preserving mode, with junk prune, caps, symlink rejection, and zip-slip containment. Use `import_skill` only for the body creation, then write files:

```python
        import stat as _stat
        from .skill_trust_scanner import SUPPORTING_JUNK_DIRS, _is_junk
        from ..tldw_api.skills_schemas import (
            MAX_SUPPORTING_FILES_COUNT, MAX_SUPPORTING_FILE_BYTES,
            MAX_SUPPORTING_FILES_TOTAL_BYTES,
        )
        skill_name = self._derive_name_from_filename(filename)
        members: list[tuple[str, bytes, bool]] = []
        skill_content: str | None = None
        total = 0
        seen_lower: set[str] = set()
        with zipfile.ZipFile(io.BytesIO(file_content), "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                mode = (member.external_attr >> 16) & 0xFFFF
                if _stat.S_ISLNK(mode):
                    continue                       # symlink member: skip-not-fail
                parts = PurePosixPath(member.filename).parts
                if any(p in SUPPORTING_JUNK_DIRS for p in parts) or _is_junk(parts[-1]):
                    continue                       # junk pruned
                member_name = self._validate_archive_member(member.filename)  # raises on zip-slip
                lower = member_name.lower()
                if lower in seen_lower:             # case-fold collision on a case-insensitive FS
                    raise ValueError(f"local_skill_invalid_archive:case_collision:{member_name}")
                seen_lower.add(lower)
                data = archive.read(member)
                if len(data) > MAX_SUPPORTING_FILE_BYTES:
                    raise ValueError(f"local_skill_file_too_large:{member_name}")
                total += len(data)
                if member_name == _SKILL_FILENAME:
                    skill_content = data.decode("utf-8")
                else:
                    members.append((member_name, data, bool(mode & 0o111)))
        if skill_content is None:
            raise ValueError("local_skill_invalid_archive:missing_skill_md")
        if len(members) > MAX_SUPPORTING_FILES_COUNT:
            raise ValueError("local_skill_too_many_files")
        if total > MAX_SUPPORTING_FILES_TOTAL_BYTES:
            raise ValueError("local_skill_bundle_too_large")
        result = await self.import_skill(
            name=skill_name, content=skill_content, overwrite=overwrite,
            trust_approved=False,   # re-trusted below only if approved
        )
        skill_dir = self._skill_dir(skill_name)
        base = skill_dir.resolve()
        import os as _os
        from pathlib import PurePosixPath as _PP
        for member_name, data, executable in members:
            dest = skill_dir / _PP(member_name)
            if base not in dest.resolve().parents:
                raise ValueError(f"local_skill_invalid_archive:{member_name}")
            self._write_bytes_atomic(dest, data)
            if executable:
                _os.chmod(dest, dest.stat().st_mode | 0o755)
        # Re-derive trust state now that the full bundle is on disk.
        self._trust_after_approved_mutation(skill_name, trust_approved=trust_approved)
        return self._response_for_record(self._load_index()[skill_name])
```

(Keep the non-zip loose-file branch at :841-847 unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_zip_import_bundle.py -q`
Expected: PASS.

- [ ] **Step 5: Regression — existing import tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_skills_import.py -q`
Expected: PASS after flipping the nested-drop assertion (Task 12 does this) — if it fails only on `test_skills_import.py:525-527`, that flip is expected; note it and continue.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Skills_Interop/local_skills_service.py Tests/Skills/test_zip_import_bundle.py
git commit -m "feat(skills): faithful nested/binary/exec zip import with zip-slip hardening"
```

---

### Task 9: Faithful zip export

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py` — `export_skill` (:871-887)
- Test: `Tests/Skills/test_zip_export_roundtrip.py` (create)

**Interfaces:**
- Consumes: `_iter_bundle_files`.
- Produces: `export_skill` walks the dir recursively, writes raw bytes at nested paths preserving the exec bit.

- [ ] **Step 1: Write the failing test**

```python
import io, os, stat, zipfile, pytest
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.mark.asyncio
async def test_export_roundtrip_preserves_tree_and_mode(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    await svc.create_skill(name="demo", content="---\nname: demo\n---\nbody\n")
    d = svc._skill_dir("demo")
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    (d / "scripts" / "run.sh").write_bytes(b"#!/bin/sh\n")
    os.chmod(d / "scripts" / "run.sh", 0o755)
    (d / "assets").mkdir(exist_ok=True)
    (d / "assets" / "logo.png").write_bytes(b"\x89PNG\x00bin")

    export = await svc.export_skill("demo")
    with zipfile.ZipFile(io.BytesIO(export["content"])) as z:
        names = set(z.namelist())
        assert {"SKILL.md", "scripts/run.sh", "assets/logo.png"} <= names
        assert z.read("assets/logo.png") == b"\x89PNG\x00bin"
        info = z.getinfo("scripts/run.sh")
        assert (info.external_attr >> 16) & stat.S_IXUSR
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_zip_export_roundtrip.py -q`
Expected: FAIL — current export is flat and drops nested files.

- [ ] **Step 3: Implement**

```python
    async def export_skill(self, skill_name: str) -> Any:
        import stat
        self._enforce("skills.export.launch.local")
        normalized = _normalize_skill_name(skill_name)
        skill_dir = self._skill_dir(normalized)
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            body = skill_dir / _SKILL_FILENAME
            archive.writestr(_SKILL_FILENAME, body.read_bytes())
            for relative_path, path in sorted(
                self._iter_bundle_files(skill_dir), key=lambda x: x[0]
            ):
                if path.is_symlink() or not path.is_file():
                    continue
                info = zipfile.ZipInfo(relative_path)
                mode = path.stat().st_mode
                info.external_attr = (mode & 0xFFFF) << 16
                archive.writestr(info, path.read_bytes())
        return {
            "content": archive_buffer.getvalue(),
            "filename": f"{normalized}.zip",
            "content_type": "application/zip",
        }
```

(`_normalize_skill_name` is imported at module scope or via the existing deferred import — match the file's existing usage.)

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_zip_export_roundtrip.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Skills_Interop/local_skills_service.py Tests/Skills/test_zip_export_roundtrip.py
git commit -m "feat(skills): faithful recursive zip export preserving tree and exec bit"
```

---

### Task 10: Trust review preview — binary vs deleted disambiguation

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_skills_canvas.py:242-304` (`skill_trust_review_preview`); `tldw_chatbook/Skills_Interop/skill_trust_service.py` `capture_review` (ensure `current_fingerprints` is in the payload — it already is per the design review)
- Test: `Tests/Library/test_skill_trust_review_preview.py` (create)

**Interfaces:**
- Consumes: `active_review` mapping with `changed_files`, `current_files` (text), `current_fingerprints` (list of manifest entries incl. `relative_path`, `byte_length`, `sha256`, `file_type`).
- Produces: a present binary renders `binary file, N bytes, sha256 …`; a deleted file renders `(deleted …)`.

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Widgets.Library.library_skills_canvas import skill_trust_review_preview


def test_present_binary_shows_metadata_not_deleted():
    review = {
        "changed_files": ["assets/logo.png"],
        "current_files": {},   # binary absent from text view
        "current_fingerprints": [
            {"relative_path": "assets/logo.png", "file_type": "supporting_binary",
             "byte_length": 2048, "sha256": "deadbeef"},
        ],
    }
    out = skill_trust_review_preview(review)
    assert "assets/logo.png" in out
    assert "binary" in out.lower()
    assert "2048" in out
    assert "deadbeef"[:8] in out
    assert "deleted" not in out.lower()


def test_genuinely_deleted_still_shows_deleted():
    review = {
        "changed_files": ["gone.md"],
        "current_files": {},
        "current_fingerprints": [],   # not on disk at all
    }
    out = skill_trust_review_preview(review)
    assert "deleted" in out.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_skill_trust_review_preview.py -q`
Expected: FAIL — a present binary currently renders "(deleted — no longer on disk)".

- [ ] **Step 3: Implement**

In `skill_trust_review_preview`, build a fingerprint lookup and disambiguate the `content is None` branch:

```python
    raw_files = active_review.get("current_files")
    current_files = dict(raw_files) if isinstance(raw_files, Mapping) else {}
    fingerprints = {
        str(fp.get("relative_path")): fp
        for fp in (active_review.get("current_fingerprints") or [])
        if isinstance(fp, Mapping)
    }
    ...
        content = current_files.get(file_name)
        if content is None:
            fp = fingerprints.get(file_name)
            if fp is not None:
                block = (
                    f"── {file_name} ──\n"
                    f"(binary file — {fp.get('byte_length', 0)} bytes, "
                    f"sha256 {str(fp.get('sha256', ''))[:12]}…)"
                )
            else:
                block = f"── {file_name} ──\n(deleted — no longer on disk)"
        else:
            ... # unchanged text rendering
```

Verify `capture_review` includes `current_fingerprints` in its returned payload; if not present, add `"current_fingerprints": [fp.as_manifest_entry() for fp in current.fingerprints]` alongside `current_files`.

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_skill_trust_review_preview.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_skills_canvas.py tldw_chatbook/Skills_Interop/skill_trust_service.py Tests/Library/test_skill_trust_review_preview.py
git commit -m "feat(skills): review preview disambiguates present binary from deleted"
```

---

### Task 11: Editor canvas — nested paths + view-only binaries

**Files:**
- Modify: `tldw_chatbook/Library/library_skills_state.py:381-387` (`build_skill_editor_state` supporting-files reduction), `tldw_chatbook/Widgets/Library/library_skills_canvas.py:409-413` (`skill_supporting_files_text`)
- Test: `Tests/Library/test_library_skills_state.py` (extend)

**Interfaces:**
- Consumes: `bundle_files` (Task 1/6) on the skill response.
- Produces: editor lists nested paths; binaries labeled view-only (`path — N bytes (binary)`).

- [ ] **Step 1: Write the failing test** (add to `Tests/Library/test_library_skills_state.py`)

```python
def test_editor_lists_nested_and_marks_binary():
    from tldw_chatbook.Library.library_skills_state import build_skill_editor_state
    detail = {
        "name": "demo", "content": "body", "version": 1,
        "supporting_files": {"references/api.md": "# api\n"},
        "bundle_files": [
            {"path": "references/api.md", "size": 6, "executable": False, "is_text": True},
            {"path": "assets/logo.png", "size": 2048, "executable": False, "is_text": False},
        ],
        "trust_status": "trusted", "trust_blocked": False,
    }
    state = build_skill_editor_state(detail)
    names = [f.name for f in state.supporting_files]
    assert "references/api.md" in names
    assert "assets/logo.png" in names          # binary listed
    binary = next(f for f in state.supporting_files if f.name == "assets/logo.png")
    assert binary.is_text is False             # view-only marker
```

- [ ] **Step 2: Run to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_skills_state.py -q -k nested_and_marks_binary`
Expected: FAIL — `is_text` attr missing / binary not listed.

- [ ] **Step 3: Implement**

Extend the supporting-file row model in `library_skills_state.py` with an `is_text: bool = True` field, and build the list from `bundle_files` (falling back to `supporting_files` when `bundle_files` is absent, e.g. remote). In `skill_supporting_files_text` (canvas :409-413), render binaries as `"{path} — {size} bytes (binary)"` and text as before. Keep the list read-only for binaries (the editor already renders supporting files as a read-only list per the design).

- [ ] **Step 4: Run to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_skills_state.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Library/library_skills_state.py tldw_chatbook/Widgets/Library/library_skills_canvas.py Tests/Library/test_library_skills_state.py
git commit -m "feat(skills): editor lists nested supporting paths, marks binaries view-only"
```

---

### Task 12: Integration — real bundle end-to-end + flip the legacy assertion

**Files:**
- Modify: `Tests/Skills/test_skills_import.py:525-527` (flip nested-drop assertion)
- Test: `Tests/Skills/test_bundle_fidelity_integration.py` (create)

**Interfaces:** consumes the whole stack.

- [ ] **Step 1: Flip the legacy assertion**

In `Tests/Skills/test_skills_import.py:525-527`, the test currently asserts a nested `references/note.md` is NOT in `supporting_files`. Change it to assert it IS imported:

```python
    assert "references/note.md" in result["supporting_files"]
    assert result["supporting_files"]["references/note.md"] == "..."  # match the fixture content
```

Leave `Tests/Skills/test_local_skills_service.py:320` (rejects `../escape.md`) unchanged.

- [ ] **Step 2: Write the integration test**

```python
import io, zipfile, pytest
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.mark.asyncio
async def test_bundle_roundtrip_and_tamper_detection(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    # Build a bundle with a nested executable script and a binary asset.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SKILL.md", "---\nname: sdd\n---\nRun scripts/run.sh\n")
        info = zipfile.ZipInfo("scripts/run.sh"); info.external_attr = 0o755 << 16
        z.writestr(info, "#!/bin/sh\necho hi\n")
        z.writestr("assets/logo.png", b"\x89PNG\x00bin")
    await svc.import_skill_file(buf.getvalue(), filename="sdd.zip",
                               content_type="application/zip")
    skill = await svc.get_skill("sdd")
    assert "scripts/run.sh" in skill["supporting_files"]
    assert any(b["path"] == "assets/logo.png" and not b["is_text"]
               for b in skill["bundle_files"])
    # Export → re-import is byte-identical for the binary asset.
    export = await svc.export_skill("sdd")
    with zipfile.ZipFile(io.BytesIO(export["content"])) as z:
        assert z.read("assets/logo.png") == b"\x89PNG\x00bin"
```

- [ ] **Step 3: Run**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/test_bundle_fidelity_integration.py Tests/Skills/test_skills_import.py -q`
Expected: PASS.

- [ ] **Step 4: Full suite regression**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Skills/ Tests/Library/ Tests/UI/test_library_skills_canvas.py Tests/tldw_api/test_skills_schemas_bundle.py -q`
Expected: PASS (note any pre-existing baseline failures unrelated to this branch).

- [ ] **Step 5: Commit**

```bash
git add Tests/Skills/test_bundle_fidelity_integration.py Tests/Skills/test_skills_import.py
git commit -m "test(skills): end-to-end bundle fidelity integration + flip nested-drop assertion"
```

---

## Notes for the executor

- **Server proxy:** `server_skills_service.py` forwards the same pydantic models; nested text keys pass through, `bundle_files` is optional metadata. No binary crosses the wire — do not add base64.
- **Do NOT recurse** `skill_trust_service.py` `_iter_skill_dirs` (:539), `_known_and_current_skill_names` (:557), `_skill_dir_for_normalized_name` (:643) — they list child skill dirs, not one skill's contents.
- **Reachability is out of scope:** `execute_skill` still returns only the rendered SKILL.md body. Exposing the stored `scripts/` to the agent is a later layer.
- After each task, the reviewer checks: backward-compat (flat snapshot unchanged), the three fingerprint builders agree, binaries never enter `supporting_files`, and no hard-raise path remains for unsupported files.
