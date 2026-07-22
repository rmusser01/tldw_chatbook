# Skills Foundation — Spec 2: Full-Bundle Supporting-File Fidelity

**Status:** Approved design (2026-07-22)
**Program:** Skills-install program, Layer 0. Sibling of Spec 1 (trust isolation,
recovery & discoverability — PR #762, merged). North star: *"a user asks an agent
to install a skill/pack from a GitHub link."*

## Problem

tldw's local-skills subsystem stores a skill as a real directory on disk, but its
*supporting-file* model is **flat and text-only**, so a spec-conformant
[Agent Skills](https://agentskills.io/specification) bundle loses data at every
stage of its lifecycle:

| Stage | Code | Behaviour with nested / binary / executable files |
|-------|------|---------------------------------------------------|
| Read from dir | `local_skills_service.py` `_read_supporting_files` | `iterdir()` + `is_file()` — **silently skips subdirectories** |
| Write to dir | `_apply_supporting_files` | `skill_dir / filename` — no parent `mkdir`, assumes flat |
| Zip import | `_validate_archive_member` | **rejects** any member where `len(parts) != 1` |
| Zip export | `export_skill` | flat `writestr(filename, …)` |
| Folder import (UI) | `library_screen.py` `_read_library_skill_import_supporting_files` | reads flat siblings only; explicitly does **not** recurse |
| Trust snapshot | `skill_trust_scanner.py` `scan_skill_directory` | `iterdir()` + `is_file()` — **also flat**, text-only |
| Data model | `supporting_files: dict[str, str]` | keyed by bare `path.name` — cannot represent `scripts/x` vs `refs/x`, no bytes, no mode |

This is the confirmed UAT finding #3: importing the real `obra/superpowers`
`subagent-driven-development` skill silently drops its `scripts/review-package`,
`scripts/task-brief`, and `scripts/sdd-workspace` files. Per the
[Agent Skills spec](https://agentskills.io/specification), a bundle is a directory
of `SKILL.md` plus optional nested `scripts/` (**executable**), `references/`
(docs), `assets/` (**images / binary**, data files), *"and any additional files
or directories."*

## Goal

Faithfully **import, store, trust-verify, and export** a spec-conformant Agent
Skills bundle — arbitrary nested directories, binary files, and executable
scripts — with cryptographic tamper-evidence over every file (text and binary),
without disturbing any currently-trusted skill.

## Scope

**In:**
- Path-aware supporting-file model: relative POSIX subpaths, arbitrary depth.
- Binary files stored faithfully (raw bytes), not decoded.
- Executable bit preserved and trust-fingerprinted.
- Recursive, binary-aware, mode-aware trust scanner with per-file SHA-256 inside
  the existing HMAC-authenticated manifest.
- Both existing import paths (**zip** and **folder**) and export made faithful,
  with zip-slip / symlink / traversal hardening and size caps.
- Editor lists nested paths; text files stay editable, binaries are view-only.

**Out (later layers, noted so deferral is deliberate):**
- SKILL.md **frontmatter** conformance (`license`, `compatibility`, `metadata`
  map, `--` rule, 1024-char description) — a separate sibling spec.
- A new **folder / GitHub picker UX** and multi-skill **pack** import — Layers 1–2.
- **Asymmetric publisher signatures** (proving a bundle came from a given GitHub
  publisher) — a supply-chain concern for the remote-fetch layer. Spec 2 provides
  *local* keyed-MAC tamper-evidence, not provenance.

## Design

### 1. Bundle representation (directory-native)

The **skill directory on disk is the source of truth.** Import writes the bundle
faithfully into `skill_dir`; trust scans the dir; export zips the dir. The API and
editor keep two *derived* views, so no byte round-trips through JSON:

- `supporting_files: dict[str, str]` — **text files only**, keys now **nested
  POSIX paths** (`scripts/build.sh`, `references/api.md`). Backward-compatible
  shape; keys simply gain `/`. This is what the editor reads/writes.
- `bundle_files: list[BundleFileInfo]` — the **full manifest view**, one entry per
  file: `{path: str, size: int, executable: bool, is_text: bool}`. Binaries
  appear here only, so the UI can show `assets/logo.png — 12 KB` without inlining
  bytes.

Rationale: skills are already stored as directories and trust already scans the
tree, so the tree is the natural source of truth. This avoids base64 bloat, keeps
the editor's text-only reality honest, and confines binary/mode concerns to the
filesystem and the trust fingerprint.

### 2. Path validation — `validate_supporting_file_path(path: str) -> str`

A single validator (new, in the skills schema module) reused by read, write,
zip-import, and folder-import. Accepts a **relative POSIX subpath** where **each
segment** matches the existing safe filename pattern
`^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$`. Returns the normalized POSIX path.

Rejects (each raises a typed `ValueError`):
- absolute paths, a leading `/`, backslashes (`\`);
- any `..` or `.` segment; empty segments (`a//b`);
- symlinks (checked against the on-disk target during read/scan/copy);
- `SKILL.md` at the root (reserved — it is the skill body, not a supporting file);
- depth > **8** segments; total path length > **255** bytes.

### 3. Trust scanner — recursive, binary-aware, mode-aware, backward-identical for flat bundles

`scan_skill_directory` walks the tree recursively, yielding files sorted by their
POSIX relative path. Per file:
- reject symlinks and traversal → `unsupported_paths` (never trust material);
- fingerprint `SkillFileFingerprint{relative_path, file_type, byte_length,
  sha256, executable}` — `executable` is a **new field**;
- decode into `text_files[relative_path]` **only** when UTF-8-decodable and
  null-byte-free; binaries are fingerprinted (`file_type="supporting_binary"`)
  but never decoded.

**Cryptographic integrity (the tamper-evidence requirement).** The snapshot's
per-file `sha256` values live inside the canonical manifest payload that is
authenticated by `manifest_mac(payload, manifest_mac_key)` =
`HMAC-SHA256(canonical_json(payload), key)`, where `manifest_mac_key` is derived
from the user's passphrase (`skill_trust_crypto.py`). Making every binary a
first-class fingerprint (`sha256` + `byte_length` + `executable`) therefore means:
- tampering with a binary's **content** changes its SHA-256 → the on-disk file no
  longer matches its recorded fingerprint on the next scan;
- flipping a file's **executable bit** changes the `executable` fingerprint field
  → same mismatch;
- either mismatch drives the skill to **blocked / needs-review** — the user is
  *made aware*; and the manifest itself cannot be forged without the
  passphrase-derived MAC key.

**Backward-compatibility guarantee (load-bearing).** The canonical snapshot MUST
be **byte-identical to the pre-Spec-2 form** for any skill that has **no nested
paths, no binary files, and no executable bits**. Concretely: `executable` is
omitted from canonicalization when `False`; `file_type` is unchanged for text
files; and a flat directory's recursive walk reproduces today's `iterdir()`
name-ordering (for a flat dir, `relative_path == path.name`). As a result **every
currently-trusted skill stays trusted** — only bundles that actually use nesting,
binaries, or exec bits are re-scanned. No mass re-review.

### 4. Import round-trip (both existing paths; no new picker)

- **Zip** (`import_skill_file` / `_validate_archive_member`): allow nested members
  (validated by §2), extract raw bytes faithfully (no UTF-8 decode of binaries),
  preserve the executable bit from the zip member's unix mode
  (`external_attr >> 16`). **Zip-slip hardening**: resolve each target and assert
  it stays within `skill_dir` before writing.
- **Folder** (`_run_library_skills_import` → a new
  `import_skill_directory(source_dir, name, …)` service method): replace the
  flat-siblings read with a **safe recursive tree copy** into `skill_dir` —
  validate every relative path (§2), preserve mode, reject symlinks. This is the
  direct finding #3 fix.
- Both land **trust-pending** (`trust_approved=False`); the trust scan runs on the
  stored directory.

### 5. Export — `export_skill`

Walk `skill_dir` recursively; write each file at its POSIX relative path into the
zip as raw bytes, preserving the executable bit via `ZipInfo.external_attr`.
Export→import is byte-identical (round-trip test).

### 6. Service read / write

- `_read_supporting_files` → recursive; returns the **text view** (nested keys).
  A companion `_read_bundle_manifest` returns the `bundle_files` listing (text +
  binary + mode + size).
- `_apply_supporting_files` (editor text edits) → `mkdir` parents, path-validate
  (§2), atomic write; text-only. Binary and mode changes never arrive through the
  editor. The faithful import copy (§4) is a separate, tree-aware path.

### 7. Editor display

The canvas lists supporting files by nested path. **Text files stay editable**;
binaries are **view-only** (`path — size (binary)`), not rendered or edited. No
hex editor, no binary preview (YAGNI). Deleting a supporting file remains
available for both kinds.

### 8. Security & limits

- Symlinks rejected everywhere (read, scan, copy, zip in/out).
- Zip-slip contained by resolve-and-assert-within-`skill_dir` on every member.
- Per-file cap **5 MB**; per-bundle total **25 MB**; max **500 files** — bounding
  memory and DoS from a hostile archive. Enforced during import (zip and folder)
  before writing. Over-limit is a clear, non-partial failure (nothing is left
  half-written).
- Binary detection: a null byte **or** a UTF-8 decode failure marks a file binary
  (the scanner's existing heuristic, now applied recursively).

## Components & boundaries

| Unit | File | Responsibility |
|------|------|----------------|
| Path validator | `tldw_api/skills_schemas.py` (or new `skill_bundle_paths.py`) | `validate_supporting_file_path`; the one place path rules live |
| Trust scanner | `Skills_Interop/skill_trust_scanner.py` | recursive walk, binary/mode fingerprints, backward-identical canonical form |
| Fingerprint model | `Skills_Interop/skill_trust_models.py` | add `executable` field; canonicalization omits it when `False` |
| Service bundle I/O | `Skills_Interop/local_skills_service.py` | recursive read, safe tree copy (`import_skill_directory`), faithful zip in/out, size caps |
| Import archive validation | `Skills_Interop/local_skills_service.py` | nested-aware `_validate_archive_member` + zip-slip guard |
| Folder import path | `UI/Screens/library_screen.py` | call `import_skill_directory`; drop the flat-siblings read |
| Editor canvas | `Widgets/Library/library_skills_canvas.py` | nested-path list, view-only binaries |
| API schema | `tldw_api/skills_schemas.py` | `bundle_files` / `BundleFileInfo`; nested keys in `supporting_files` |

## Testing strategy

- **Unit — validator:** nested-ok; reject traversal, absolute, symlink, backslash,
  depth > 8, over-length, reserved `SKILL.md`.
- **Unit — scanner:** recursion into `scripts/`/`references/`/`assets/`; binary
  fingerprinted-not-decoded; executable bit captured; **a flat all-text skill's
  snapshot is byte-identical to the pre-Spec-2 output** (guards existing trust);
  content or mode change → fingerprint mismatch.
- **Unit — zip import:** nested + binary + exec preserved; **zip-slip member
  rejected**; over-cap archive rejected.
- **Unit — export:** recursive round-trip preserves paths, bytes, exec bit.
- **Unit — folder import:** faithful recursive copy; symlink rejected; caps.
- **Integration:** import the real `subagent-driven-development` bundle (has
  `scripts/`) → nested executable scripts present, blocked until approved, trusted
  after approval; **export→import is byte-identical**; tampering with an approved
  binary flips the skill to needs-review.

## Migration

No trust-data migration and no config change. The backward-identical canonical
form (§3) means existing trusted skills are undisturbed. The new `executable`
fingerprint field and `bundle_files` listing are additive.
