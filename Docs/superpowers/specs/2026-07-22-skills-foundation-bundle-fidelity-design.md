# Skills Foundation — Spec 2: Full-Bundle Supporting-File Fidelity

**Status:** Approved design (2026-07-22), hardened after a code-verified review.
**Program:** Skills-install program, Layer 0. Sibling of Spec 1 (trust isolation,
recovery & discoverability — PR #762, merged). North star: *"a user asks an agent
to install a skill/pack from a GitHub link."*

## Problem

tldw stores a skill as a real directory on disk, but its *supporting-file* model
is **flat and text-only**, so a spec-conformant
[Agent Skills](https://agentskills.io/specification) bundle loses data at every
stage of its lifecycle:

| Stage | Code | Behaviour with nested / binary / executable files |
|-------|------|---------------------------------------------------|
| Read from dir | `local_skills_service.py` `_read_supporting_files` | `iterdir()` + `is_file()` — **silently skips subdirectories**; UTF-8-decodes every file, so a **top-level binary crashes `get_skill`** |
| Write to dir | `_apply_supporting_files` | `skill_dir / filename` — no parent `mkdir`, assumes flat |
| Zip import | `_validate_archive_member` | **rejects** any member where `len(parts) != 1` |
| Zip export | `export_skill` | flat `writestr(filename, …)` |
| Folder import (UI) | `library_screen.py` `_read_library_skill_import_supporting_files` | reads flat siblings only; explicitly does **not** recurse |
| Trust snapshot | `skill_trust_scanner.py` `scan_skill_directory` | `iterdir()` + `is_file()` — **flat**, text-only; body classified by **basename** |
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

**In:** path-aware supporting-file model (relative POSIX subpaths, arbitrary
depth); binary files stored faithfully (raw bytes); executable bit preserved and
trust-fingerprinted; recursive, binary-aware, mode-aware trust scanner with
per-file SHA-256 inside the existing HMAC-authenticated manifest; both existing
import paths (**zip** and **folder**) and export made faithful, with
zip-slip / symlink / traversal / case-fold hardening and size caps; editor lists
nested paths, text editable, binaries view-only.

**Out (later layers, noted so deferral is deliberate):**
- SKILL.md **frontmatter** conformance (`license`, `compatibility`, `metadata`
  map, `--` rule, 1024-char description) — a separate sibling spec.
- A new **folder / GitHub picker UX** and multi-skill **pack** import — Layers 1–2.
- **Asymmetric publisher signatures** (proving provenance) — a supply-chain
  concern for the remote-fetch layer. Spec 2 provides *local* keyed-MAC
  tamper-evidence, not provenance.
- **Full binary restore on rollback** — trust rollback restores text files only
  (see §9). Binaries are tamper-*detected* but not auto-reverted.

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
  bytes. Optional/local-first: the remote skills service may not populate it, so
  every consumer degrades gracefully to an empty list.

Rationale: skills are already stored as directories and trust already scans the
tree, so the tree is the natural source of truth — avoiding base64 bloat, keeping
the editor's text-only reality honest, and confining binary/mode concerns to the
filesystem and the trust fingerprint.

### 2. Path validation — `validate_supporting_file_path(path: str) -> str`

A single validator (new, in the skills schema module) reused by read, write,
zip-import, and folder-import. Accepts a **relative POSIX subpath** where **each
segment** matches the existing safe filename pattern
`^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$`. Returns the normalized POSIX path.

Rejects (each raises a typed `ValueError`):
- absolute paths, a leading `/`, backslashes (`\`);
- any `..` or `.` segment; empty segments (`a//b`);
- symlinks (rejected, never followed — §3/§4);
- **any** file whose basename case-insensitively equals `skill.md`: the top-level
  `SKILL.md` (exact case, per the Agent Skills spec) is the skill body, and no
  other file — a wrong-case `skill.md` at the root, or a nested
  `references/SKILL.md` — may shadow it or be confused with it;
- depth > **8** segments; total path length > **255** bytes.

This single validator resolves the current inconsistency where the scanner treats
`SKILL.md` case-sensitively but `skills_schemas.py` rejects supporting `skill.md`
case-insensitively.

### 3. Trust scanner — recursive, binary-aware, mode-aware, backward-identical for flat bundles

`scan_skill_directory` walks the tree with `os.walk(followlinks=False)` (never
following symlinked directories — no loops, no escape), emitting files **sorted by
POSIX relative path** (the ordering choice that preserves backward-compat — not
absolute path, not case-folded, not raw traversal order). Per file:
- reject symlinks and traversal → `unsupported_paths` (never trust material);
- **classify the body by the top-level relative path**: `file_type="skill"` iff
  `relative_path == "SKILL.md"` (no `/`). A nested `references/SKILL.md` is never
  the body (today's basename check would misclassify it — this is a change from
  `relative_path = path.name`);
- fingerprint `SkillFileFingerprint{relative_path, file_type, byte_length,
  sha256, executable}` — `executable` is a **new field**;
- decode into `text_files[relative_path]` **only** when UTF-8-decodable and
  null-byte-free; binaries are fingerprinted (`file_type="supporting_binary"`)
  but never decoded and never raise.

**Three fingerprint sites must change together.** The canonical fingerprint dict
is built explicitly (field-by-field) in three independent places; all three must
add `executable` with the *identical* conditional rule or verification diverges:
1. `SkillFileFingerprint.as_manifest_entry` (`skill_trust_models.py`) — the shared
   manifest/digest/review serializer;
2. the scanner constructor (`skill_trust_scanner.py`) — sets `executable` from the
   file mode;
3. `_content_manifest_entry` / `_fingerprint_in_memory_files`
   (`skill_trust_service.py`) — the parallel *in-memory* builder used by
   `verify_skill_content`, which must hash raw bytes for binaries and set
   `executable` the same way.

**Cryptographic integrity (the tamper-evidence requirement).** Per-file `sha256`
values live inside the canonical manifest payload authenticated by
`manifest_mac(payload, manifest_mac_key)` =
`HMAC-SHA256(canonical_json(payload), key)`, where `manifest_mac_key` is derived
from the user's passphrase. Verification compares **only** the fingerprint dicts
(never decoded content), so making every binary a first-class fingerprint
(`sha256` + `byte_length` + `executable`) fully covers it:
- tampering with a binary's **content** changes its SHA-256;
- flipping a file's **executable bit** changes the `executable` field;
- either mismatch drives the skill to **blocked / needs-review** — the user is
  *made aware* — and the manifest cannot be forged without the passphrase-derived
  MAC key.

**Backward-compatibility guarantee (load-bearing).** The canonical snapshot MUST
be **byte-identical to the pre-Spec-2 form** for any skill with **no nested paths,
no binaries, and no executable bits**. This is achievable because serialization is
explicit: `executable` is **included in the fingerprint dict only when `True`**
(`if self.executable: entry["executable"] = True`); `file_type` is unchanged for
text; and a flat directory's `os.walk` sorted by relative path reproduces today's
`iterdir()` name-order (for a flat dir `relative_path == path.name`). As a result
**every currently-trusted skill stays trusted** — only bundles that actually use
nesting, binaries, or exec bits are re-scanned. No mass re-review.

### 4. Import round-trip (both existing paths; no new picker)

- **Zip** (`import_skill_file` / `_validate_archive_member`): allow nested members
  (validated by §2), extract raw bytes faithfully (no UTF-8 decode of binaries).
  Hardening on every member: **zip-slip** (resolve target, assert within
  `skill_dir`), **symlink members** rejected (unix mode `S_ISLNK`), **case-fold
  collisions** rejected (two members whose paths differ only in case would collide
  on a case-insensitive filesystem — reject rather than silently last-wins).
  Executable bit is **best-effort** from the member's unix mode
  (`external_attr >> 16`): honored when the archive carries unix attributes
  (GitHub archive zips do), otherwise defaults non-executable.
- **Folder** (`_run_library_skills_import` → a new
  `import_skill_directory(source_dir, name, …)` service method): replace the
  flat-siblings read with a **safe recursive tree copy** into `skill_dir` —
  `os.walk(followlinks=False)`, validate every relative path (§2), reject
  symlinks, and read the **real file mode** for the exec bit (reliable, unlike
  zip). This is the direct finding #3 fix.
- Both land **trust-pending** (`trust_approved=False`); the trust scan runs on the
  stored directory.

### 5. Export — `export_skill`

Walk `skill_dir` recursively; write each file at its POSIX relative path into the
zip as raw bytes, preserving the executable bit via `ZipInfo.external_attr`. **All
files** round-trip byte-identically (export→import). Empty directories carry no
content and are not preserved — an accepted, documented limitation.

### 6. Service read / write

- `_read_supporting_files` → recursive (`os.walk(followlinks=False)`); returns the
  **text view** (nested keys). **It MUST catch UTF-8 decode failures / null bytes
  and route those files to the binary manifest — never raise.** This also fixes a
  pre-existing latent bug: today a single top-level binary makes `get_skill` throw
  `UnicodeDecodeError`, leaving the skill un-openable in the editor.
- A companion `_read_bundle_manifest` returns the `bundle_files` listing (text +
  binary + mode + size).
- `_apply_supporting_files` (editor text edits) → `mkdir` parents, path-validate
  (§2), atomic write; text-only. **Invariant:** the editor save must never pass a
  full-replacement `supporting_files` dict (today it passes `None`, a no-op), so
  binaries and nested files it never loaded are never deleted or reconciled away.

### 7. Editor display

The canvas lists supporting files by nested path. **Text files stay editable**;
binaries are **view-only** (`path — size (binary)`), not rendered or edited. No
hex editor, no binary preview (YAGNI). Delete remains available for both kinds.

### 8. Security & limits

- Symlinks rejected everywhere and never followed (`os.walk(followlinks=False)` on
  read, scan, and folder copy; `S_ISLNK` member rejection on zip in/out).
- Zip-slip contained (resolve-and-assert-within-`skill_dir` per member).
- Case-fold collisions rejected on import (zip and folder).
- Size caps: per-file **5 MB**, per-bundle total **25 MB**, max **500 files** —
  enforced on the archive's **declared** `file_size` *before* extracting **and**
  bounded on the **bytes actually read** during extraction (defence against a
  header that lies / a zip bomb). Over-limit is a clean, non-partial failure
  (nothing left half-written).
- Binary detection: a null byte **or** a UTF-8 decode failure marks a file binary
  (the scanner's existing heuristic, applied recursively).

### 9. Trust rollback / restore

Trust rollback restores the trusted baseline from the encrypted `text_files`
snapshot — **text files only**, as today. Binaries are fingerprinted for
tamper-*detection* (a changed binary flips the skill to needs-review) but are
**not** stored in the encrypted snapshot and so are **not auto-reverted** on
rollback; they are left in place for the user to re-review. Keeping binary bytes
out of the encrypted snapshot avoids a new blob format and storage growth; the
integrity guarantee (§3) is unaffected. This limitation is called out in the
release note.

## Components & boundaries

| Unit | File | Responsibility |
|------|------|----------------|
| Path validator | `tldw_api/skills_schemas.py` (or new `skill_bundle_paths.py`) | `validate_supporting_file_path`; the one place path + `skill.md` rules live |
| Trust scanner | `Skills_Interop/skill_trust_scanner.py` | recursive `os.walk`, relative-path body classification, binary/mode fingerprints, backward-identical canonical form |
| Fingerprint model | `Skills_Interop/skill_trust_models.py` | add `executable`; `as_manifest_entry` includes it only when `True` |
| In-memory verify path | `Skills_Interop/skill_trust_service.py` | `_content_manifest_entry` / `_fingerprint_in_memory_files` — same conditional `executable`, hash raw bytes for binaries |
| Service bundle I/O | `Skills_Interop/local_skills_service.py` | recursive binary-safe read (never raises), `_read_bundle_manifest`, `import_skill_directory` safe tree copy, faithful zip in/out, size caps |
| Import archive validation | `Skills_Interop/local_skills_service.py` | nested-aware `_validate_archive_member` + zip-slip + symlink + case-fold guards |
| Folder import path | `UI/Screens/library_screen.py` | call `import_skill_directory`; drop the flat-siblings read |
| Editor canvas | `Widgets/Library/library_skills_canvas.py` | nested-path list, view-only binaries |
| API schema | `tldw_api/skills_schemas.py` | `bundle_files` / `BundleFileInfo` (optional); nested keys in `supporting_files` |

**Do NOT recurse** the three `skill_trust_service.py` enumerations at `:539`
(`_iter_skill_dirs`), `:557` (`_known_and_current_skill_names`), and `:643`
(`_skill_dir_for_normalized_name`) — they list the *child skill directories* under
`skills_dir`, not the contents of one skill, and must stay flat.

## Testing strategy

- **Unit — validator:** nested-ok; reject traversal, absolute, symlink, backslash,
  depth > 8, over-length, and any-case `skill.md`.
- **Unit — scanner:** recursion into `scripts/`/`references/`/`assets/`; binary
  fingerprinted-not-decoded; executable bit captured; **nested `references/SKILL.md`
  is NOT classified as the body**; **a flat all-text skill's snapshot is
  byte-identical to the pre-Spec-2 output** (guards existing trust); content OR
  mode change → fingerprint mismatch.
- **Unit — verify_skill_content parity:** the in-memory builder and the scanner
  produce identical fingerprint dicts for the same nested/binary/exec bundle.
- **Unit — read never crashes:** `get_skill` on a skill containing a top-level
  binary returns (binary in `bundle_files`, not in `supporting_files`), never
  raises.
- **Unit — zip import:** nested + binary + exec preserved; **zip-slip, symlink,
  and case-fold members rejected**; over-cap / declared-vs-actual-size mismatch
  rejected.
- **Unit — export:** recursive round-trip preserves paths, bytes, exec bit.
- **Unit — folder import:** faithful recursive copy; symlink rejected; real mode
  read; caps.
- **Unit — rollback:** text files restored from snapshot; a tampered binary is
  flagged needs-review and left in place (not reverted).
- **Integration:** import the real `subagent-driven-development` bundle (has
  `scripts/`) → nested executable scripts present, blocked until approved, trusted
  after approval; **export→import byte-identical**; tampering with an approved
  binary flips the skill to needs-review.

## Migration

No trust-data migration and no config change. The backward-identical canonical
form (§3) means existing trusted skills are undisturbed. The new `executable`
fingerprint field and `bundle_files` listing are additive.

**Release note:** skills may now contain nested directories, binary assets, and
executable scripts, imported and exported faithfully with per-file tamper-evidence.
Trust rollback restores text files; a tampered binary is flagged for re-review
rather than auto-reverted.
