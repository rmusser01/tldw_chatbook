# Skills Foundation — Spec 2: Full-Bundle Supporting-File Fidelity

**Status:** Approved design (2026-07-22), hardened after two rounds of code-verified
review.
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
| Write to dir | `_apply_supporting_files` | text-mode write (`open("w")`), no parent `mkdir`, **no traversal guard** (relies on the schema pattern) |
| Zip import | `_validate_archive_member` | **rejects** any member where `len(parts) != 1`; `.decode("utf-8")` on every member |
| Zip export | `export_skill` | flat `writestr(filename, …)`, content assumed text |
| Folder import (UI) | `library_screen.py` `_read_library_skill_import_supporting_files` | reads flat siblings only; explicitly does **not** recurse |
| Trust snapshot | `skill_trust_scanner.py` `scan_skill_directory` | `iterdir()` + `is_file()` — **flat**, text-only; body classified by **basename**; nested/binary → `unsupported_paths` |
| Trust manifest | `skill_trust_service.py` `bootstrap_trust` / `trust_current_skill` | **hard-raise `ValueError("unsupported_path")`** on any unsupported file |
| Schema | `skills_schemas.py` | `supporting_files: dict[str, str]`; key pattern forbids `/`; caps **20 files / 500 KB / 5 MB**; byte counts assume `str` |
| Review panel | `library_skills_canvas.py` `skill_trust_review_preview` | text-only preview; a present-but-non-text file renders as "(deleted)" |

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
without disturbing any currently-trusted skill and without making real-world
bundles (which carry VCS/OS/build junk) untrustable.

## Position in the program (important framing)

Spec 2 delivers **storage fidelity** — a *prerequisite* for the north star, not
the whole of it. Verified gap: `execute_skill` returns only the rendered `SKILL.md`
body; `supporting_files` and the skill `directory_path` are **not** propagated to
the runtime, and an agent fork gets no working-directory binding or file handle to
the skill dir. So a `SKILL.md` instruction like "run `scripts/extract.py`" cannot
resolve today even after faithful storage. **Exposing the skill directory to the
agent at execution time is a separate, later layer** (Layer "skill-runtime
reachability"). Spec 2 makes the files *exist correctly*; a later layer makes them
*reachable*.

## Scope

**In:** path-aware supporting-file model (relative POSIX subpaths, arbitrary
depth); binary files stored faithfully on disk (raw bytes); executable bit
preserved and trust-fingerprinted; recursive, binary-aware, mode-aware trust
scanner with per-file SHA-256 inside the existing HMAC-authenticated manifest;
junk-pruning + tolerate-and-surface handling of unsupported files; both existing
import paths (**zip** and **folder**) and export made faithful, with
zip-slip / symlink / traversal / case-fold hardening and size caps; editor and
**trust review panel** handle nested paths and binaries.

**Out (later layers, noted so deferral is deliberate):**
- **Skill-runtime reachability** — exposing the stored skill dir / nested files to
  the agent at execution (see "Position in the program"). Necessary for the north
  star; not part of storage fidelity.
- SKILL.md **frontmatter** conformance (`license`, `compatibility`, `metadata`
  map, `--` rule, 1024-char description) — a separate sibling spec.
- A new **folder / GitHub picker UX** and multi-skill **pack** import — Layers 1–2.
- **Asymmetric publisher signatures** (provenance) — the remote-fetch layer. Spec 2
  provides *local* keyed-MAC tamper-evidence, not provenance.
- **Full binary restore on rollback** — rollback restores text files only (§9).
- **Binary content over the remote/server wire** — see §1; only metadata crosses.

## Design

### 1. Bundle representation (directory-native) — binaries never cross the JSON wire

The **skill directory on disk is the source of truth.** Import writes the bundle
faithfully into `skill_dir`; trust scans the dir; export zips the dir. Two derived
views serve the API and editor, and — crucially — **no raw bytes ever cross the
pydantic/JSON boundary**:

- `supporting_files: dict[str, str]` — **text files only**, keys now **nested
  POSIX paths** (`scripts/build.sh`, `references/api.md`). Stays `dict[str, str]`,
  so the existing JSON transport (`model_dump(mode="json")`) and the remote server
  proxy keep working; only the key *pattern* loosens to allow `/`.
- `bundle_files: list[BundleFileInfo]` — the **full manifest view**, JSON-safe
  metadata only: `{path: str, size: int, executable: bool, is_text: bool}`.
  Binaries appear here (never in `supporting_files`), so the UI shows
  `assets/logo.png — 12 KB` without inlining bytes. Optional / local-first: the
  remote proxy may leave it empty (its bytes live on a remote server; downloading
  them is the deferred remote layer). `SkillResponse` already has `extra="allow"`,
  so a server returning `bundle_files` passes through.

Because binaries live only on disk (local) or as metadata (remote), Approach B
sidesteps base64/multipart entirely. Rationale: skills are already stored as
directories and trust already scans the tree, so the tree is the natural source of
truth — avoiding byte bloat and keeping the editor's text-only reality honest.

### 2. Path validation — `validate_supporting_file_path(path: str) -> str`

A single validator (new, in the skills schema module) reused by read, write,
zip-import, and folder-import. Validates **each path segment separately** against
the safe pattern `^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$` (a slash-joined string would
fail the pattern outright — the segment-wise check is mandatory for nesting).
Returns the normalized POSIX path.

Rejects (each raises a typed `ValueError`):
- absolute paths, a leading `/`, backslashes (`\`);
- any `..` or `.` segment; empty segments (`a//b`);
- symlinks (rejected, never followed — §3/§4);
- **any** file whose basename case-insensitively equals `skill.md`: the top-level
  `SKILL.md` (exact case) is the skill body; nothing else — a wrong-case root
  `skill.md` or a nested `references/SKILL.md` — may shadow it;
- depth > **8** segments; total path length > **255** bytes.

The **write path** (`_apply_supporting_files`) must *also* enforce containment
(resolve target, assert within `skill_dir`) rather than trusting the schema —
today it has no traversal guard of its own.

### 3. Trust scanner — recursive, binary-aware, mode-aware, backward-identical for flat bundles

`scan_skill_directory` walks the tree with `os.walk(followlinks=False)` (never
following symlinked directories — no loops, no escape), emitting files **sorted by
POSIX relative path** (the ordering choice that preserves backward-compat — not
absolute path, not case-folded, not raw traversal order). Per file:
- apply the **junk ignore-list** (§4a) — pruned entries are skipped entirely, never
  recorded;
- reject symlinks and traversal → `unsupported_paths`;
- **classify the body by the top-level relative path**: `file_type="skill"` iff
  `relative_path == "SKILL.md"` (no `/`). A nested `references/SKILL.md` is never
  the body (today's basename check, `relative_path = path.name`, would misclassify
  it);
- fingerprint `SkillFileFingerprint{relative_path, file_type, byte_length,
  sha256, executable}` — `executable` is a **new field**;
- decode into `text_files[relative_path]` **only** when UTF-8-decodable and
  null-byte-free; binaries are fingerprinted (`file_type="supporting_binary"`)
  but never decoded and never raise.

**Three fingerprint sites must change together.** The canonical fingerprint dict is
built explicitly (field-by-field) in three independent places; all must add
`executable` with the *identical* conditional rule or verification diverges:
1. `SkillFileFingerprint.as_manifest_entry` (`skill_trust_models.py`) — shared
   manifest/digest/review serializer;
2. the scanner constructor (`skill_trust_scanner.py`) — sets `executable` from mode;
3. `_content_manifest_entry` / `_fingerprint_in_memory_files`
   (`skill_trust_service.py`) — the in-memory builder used by `verify_skill_content`
   (run on every execute), which must hash **raw bytes** for binaries and set
   `executable` the same way.

**Cryptographic integrity.** Per-file `sha256` values live inside the canonical
manifest payload authenticated by `manifest_mac(payload, manifest_mac_key)` =
`HMAC-SHA256(canonical_json(payload), key)` (`manifest_mac_key` derived from the
passphrase). Verification compares **only** the fingerprint dicts (never decoded
content), so a binary is fully covered by `sha256` + `byte_length` + `executable`:
tampering with content OR the exec bit → fingerprint mismatch → skill flips to
**blocked / needs-review** (the user is made aware); the manifest can't be forged
without the passphrase-derived MAC key.

**Backward-compatibility guarantee (load-bearing).** The canonical snapshot MUST be
**byte-identical to the pre-Spec-2 form** for any skill with no nested paths, no
binaries, and no executable bits: `executable` is included in the fingerprint dict
**only when `True`** (`if self.executable: entry["executable"] = True`), `file_type`
is unchanged for text, and a flat dir's `os.walk` sorted by relative path
reproduces today's `iterdir()` name-order. **Every currently-trusted skill stays
trusted** — no mass re-review.

### 4. Junk pruning and unsupported-file tolerance

Recursion exposes real bundles' VCS/OS/build junk, and today `bootstrap_trust`
(`service.py:221`) and `trust_current_skill` (`:507`) **hard-raise** on any
unsupported file — which would make real GitHub/superpowers bundles untrustable
dead-ends (capture succeeds, approve raises). Two coordinated changes:

**(a) Junk ignore-list** applied during the walk on both import *and* scan (so junk
is neither stored nor scanned): directories `.git`, `.github`, `.hg`, `.svn`,
`node_modules`, `__pycache__`; files `.DS_Store`, `Thumbs.db`; suffixes `.pyc`,
`.pyo`, `~`, `.bak`, `.orig`, plus the existing `.tmp`/`.swp`/`.part`. Ignored
entries are skipped entirely — **never** recorded in `unsupported_paths`.

**(b) Tolerate-and-surface** instead of hard-raise. `bootstrap_trust` and
`trust_current_skill` establish trust over the **supported (fingerprinted)** files
and leave any *residual* unsupported file (a symlink, or a genuinely bad-named
file that survived pruning) as a **visible, recoverable** `needs-review`
(`status_for_skill` already models `quarantined_unsupported_path`) — never a crash.
Removing the offending file and re-scanning clears it. This is security-neutral:
unsupported files were never fingerprinted trust material either way; the change
only removes the permanent-crash dead-end. Under Spec 2, binaries are *supported*,
so the residual unsupported set shrinks to symlinks + dotfile/bad-name junk.

### 5. Import round-trip (both existing paths; no new picker)

- **Zip** (`import_skill_file` / `_validate_archive_member`): allow nested members
  (validated by §2), prune junk (§4a), extract raw bytes faithfully via a
  **bytes-mode atomic writer** (no UTF-8 decode of binaries). Hardening per member:
  **zip-slip** (resolve, assert within `skill_dir`), **symlink members** rejected
  (`S_ISLNK`), **case-fold collisions** rejected. Executable bit **best-effort**
  from the member's unix mode (`external_attr >> 16`; honored when the archive
  carries unix attrs — GitHub archive zips do — else default non-executable).
- **Folder** (`_run_library_skills_import` → new
  `import_skill_directory(source_dir, name, …)`): safe recursive tree copy into
  `skill_dir` — `os.walk(followlinks=False)`, prune junk, validate each path (§2),
  reject symlinks, read the **real file mode** for the exec bit (reliable). Direct
  finding #3 fix.
- Both land **trust-pending** (`trust_approved=False`); the trust scan runs on the
  stored directory.

### 6. Export — `export_skill`

Walk `skill_dir` recursively; write each file at its POSIX relative path into the
zip as raw bytes, preserving the exec bit via `ZipInfo.external_attr`. **All files**
round-trip byte-identically (export→import). Empty directories carry no content and
are not preserved — an accepted, documented limitation.

### 7. Service read / write

- `_read_supporting_files` → recursive (`os.walk(followlinks=False)`, junk-pruned);
  returns the **text view** (nested keys). **MUST catch UTF-8 decode failures /
  null bytes and route those files to the binary manifest — never raise.** This
  also fixes a pre-existing latent bug: today a single top-level binary makes
  `get_skill` throw `UnicodeDecodeError`, leaving the skill un-openable.
- `_read_bundle_manifest` returns the `bundle_files` listing (text + binary + mode
  + size), sharing the same binary/`is_text` heuristic as the scanner.
- `_apply_supporting_files` (editor text edits): `mkdir` parents, path-validate +
  **containment guard** (§2), atomic write; text-only. **Invariant:** the editor
  save must never pass a full-replacement `supporting_files` dict (today it passes
  `None`, a no-op), so binaries and nested files it never loaded are never deleted.

### 8. Editor and trust-review surfaces

- **Editor canvas:** lists supporting files by nested path; **text editable**,
  binaries **view-only** (`path — size (binary)`). No hex editor / binary preview.
- **Trust review panel** (distinct surface — was previously unaddressed): approval
  is already binary-safe (`trust_reviewed_snapshot` compares `_fingerprints_digest`
  hashes, no text decode/diff). But the preview (`skill_trust_review_preview`) must
  **disambiguate** a present binary from a deleted file: today a file absent from
  `text_files` renders "(deleted — no longer on disk)". Join `current_files` with
  `current_fingerprints` so a present binary renders `binary file, N bytes,
  sha256 …` while genuinely-deleted keeps "(deleted)". `capture_review`'s payload
  carries `current_fingerprints` already; nested paths already work (dict lookups
  are path-agnostic). Binaries reach a reviewable state automatically once they are
  *supported* (fingerprinted) rather than `quarantined_unsupported_path`.

### 9. Security & limits

- Symlinks rejected everywhere and never followed (`os.walk(followlinks=False)`;
  `S_ISLNK` member rejection on zip in/out).
- Zip-slip contained (resolve-and-assert-within-`skill_dir` per member); write-path
  containment guard in `_apply_supporting_files`.
- Case-fold collisions rejected on import.
- Junk ignore-list (§4a) applied on import and scan.
- **Size caps** (raising the legacy 20 / 500 KB / 5 MB): per-file **5 MB**,
  per-bundle total **25 MB**, max **500 files**. Enforced on the archive's
  **declared** sizes *before* extracting **and** bounded on **bytes actually read**
  (defence against a lying header / zip bomb). The legacy `skills_schemas.py` caps
  are raised to match so the text-view round-trip of a real bundle validates. A
  `bundle_files` count/size ceiling is added (none exists today). Over-limit is a
  clean, non-partial failure.
- Binary detection: null byte **or** UTF-8 decode failure (the scanner heuristic,
  applied recursively and shared with `bundle_files.is_text`).

### 10. Trust rollback / restore

Rollback restores the trusted baseline from the encrypted `text_files` snapshot —
**text files only**, as today. Binaries are fingerprinted for tamper-*detection*
but are **not** in the encrypted snapshot, so they are **not** auto-reverted on
rollback; a tampered binary is flagged `needs-review` and left in place for the
user to re-review. Keeping binary bytes out of the snapshot avoids a new blob
format; integrity (§3) is unaffected.

## Components & boundaries

| Unit | File | Responsibility |
|------|------|----------------|
| Path validator | `tldw_api/skills_schemas.py` (or new `skill_bundle_paths.py`) | `validate_supporting_file_path` (per-segment), `skill.md` rule, raised caps; the one place path + cap rules live |
| Trust scanner | `Skills_Interop/skill_trust_scanner.py` | recursive `os.walk`, junk-prune, relative-path body classification, binary/mode fingerprints, backward-identical canonical form |
| Fingerprint model | `Skills_Interop/skill_trust_models.py` | add `executable`; `as_manifest_entry` includes it only when `True` |
| Trust manifest/tolerance | `Skills_Interop/skill_trust_service.py` | `bootstrap_trust`/`trust_current_skill` tolerate-and-surface (no hard-raise); `_content_manifest_entry`/`_fingerprint_in_memory_files` hash raw bytes + conditional `executable` |
| Service bundle I/O | `Skills_Interop/local_skills_service.py` | recursive binary-safe read (never raises), `_read_bundle_manifest`, `import_skill_directory` safe tree copy, bytes-mode atomic writer, faithful zip in/out, junk-prune, caps |
| Import archive validation | `Skills_Interop/local_skills_service.py` | nested-aware `_validate_archive_member` + zip-slip + symlink + case-fold + junk guards |
| Folder import path | `UI/Screens/library_screen.py` | call `import_skill_directory`; drop the flat-siblings read |
| Editor + review UI | `Widgets/Library/library_skills_canvas.py`, `UI/Screens/library_screen.py` | nested-path list, view-only binaries, review preview binary-vs-deleted disambiguation |
| API schema | `tldw_api/skills_schemas.py` | `bundle_files`/`BundleFileInfo` (optional, metadata-only); nested keys in `supporting_files`; str-only byte counts unaffected (binaries never enter the dict) |

**Do NOT recurse** the three `skill_trust_service.py` enumerations at `:539`
(`_iter_skill_dirs`), `:557` (`_known_and_current_skill_names`), `:643`
(`_skill_dir_for_normalized_name`) — they list the *child skill directories* under
`skills_dir`, not one skill's contents, and must stay flat.

## Testing strategy

- **Unit — validator:** nested-ok (per-segment); reject traversal, absolute,
  symlink, backslash, depth > 8, over-length, any-case `skill.md`.
- **Unit — scanner:** recursion into `scripts/`/`references/`/`assets/`; junk
  (`.git`, `.DS_Store`, `__pycache__`, `*.pyc`, `~`) pruned (not in
  `unsupported_paths`); binary fingerprinted-not-decoded; exec bit captured; nested
  `references/SKILL.md` NOT the body; **flat all-text snapshot byte-identical to
  pre-Spec-2** (guards existing trust); content OR mode change → mismatch.
- **Unit — tolerance:** a skill with a residual unsupported file (e.g. a symlink)
  trusts its supported files and is a recoverable `needs-review`, NOT a raise;
  removing the file → trusted.
- **Unit — verify parity:** in-memory builder and scanner produce identical
  fingerprint dicts for the same nested/binary/exec bundle.
- **Unit — read never crashes:** `get_skill` on a skill with a top-level binary
  returns (binary in `bundle_files`, not `supporting_files`), never raises.
- **Unit — zip import:** nested + binary + exec preserved; junk pruned; zip-slip,
  symlink, case-fold members rejected; over-cap / declared-vs-actual mismatch
  rejected.
- **Unit — export:** recursive round-trip preserves paths, bytes, exec bit.
- **Unit — folder import:** faithful recursive copy; junk pruned; symlink rejected;
  real mode read; caps.
- **Unit — review preview:** a present changed binary renders "binary … sha256",
  NOT "(deleted)"; a genuinely deleted file still renders "(deleted)".
- **Unit — rollback:** text restored from snapshot; a tampered binary flagged
  needs-review and left in place.
- **Flip:** `Tests/Skills/test_skills_import.py:525-527` (currently asserts nested
  files dropped) → asserts they are imported. **Keep** `test_local_skills_service.py:320`
  (`../escape.md` rejected).
- **Integration:** import the real `subagent-driven-development` bundle (has
  `scripts/`) → nested executable scripts present, blocked until approved, trusted
  after approval; **export→import byte-identical**; tampering with an approved
  binary flips it to needs-review.

## Migration

No trust-data migration and no config change. The backward-identical canonical form
(§3) keeps existing trusted skills undisturbed. The `executable` fingerprint field
and `bundle_files` listing are additive; the raised caps and loosened key pattern
are backward-compatible (they only *accept* more).

**Release note:** skills may now contain nested directories, binary assets, and
executable scripts, imported/exported faithfully with per-file tamper-evidence.
VCS/OS/build junk (`.git`, `__pycache__`, `.DS_Store`, …) is ignored on import.
Trust rollback restores text files; a tampered binary is flagged for re-review
rather than auto-reverted. Running a skill's bundled `scripts/` from within the
agent is a later enhancement (this release stores them faithfully).
