---
id: TASK-17963
title: >-
  Shared fixed-name temp files race across app instances in skills store and
  trust store
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18 00:00'
updated_date: '2026-08-21 16:31'
labels:
  - skills
  - reliability
dependencies:
  - TASK-18705
priority: medium
---

## Description (the why)

TASK-18705's live verification and its own new prompt ledger
(`ProjectSkillsPromptLedger.record()` in
`tldw_chatbook/Skills_Interop/project_skills_prompt.py`) had to route around a
pre-existing atomic-write pattern used elsewhere in the skills subsystem: a
fixed, non-writer-unique temp filename for the atomic write-then-`replace()`
sequence. Two writes with the SAME fixed temp name racing across concurrent
writers means one writer's `temp_path.replace(path)` can consume the other's
still-being-written temp file out from under it (or the second writer's own
open/write can clobber the first's in-flight temp file), raising
`FileNotFoundError` or silently corrupting the write.

Two pre-existing sites still have this pattern and were deliberately left
untouched by TASK-18705 (out of that task's scope):

- `tldw_chatbook/Skills_Interop/local_skills_service.py:303`
  (`LocalSkillsService._save_index`, `temp_path =
  self.index_path.with_suffix(".json.tmp")`) — called by every skill
  create/import/edit/delete that touches the shared skills index, so two app
  instances (or two concurrent async callers in the same instance) mutating
  the skills store at close to the same moment can race on this one fixed
  path. The same file's `_write_text_atomic`/`_write_bytes_atomic` (lines
  ~316-329, `temp_path = path.with_name(f"{path.name}.tmp")`) have the
  identical shape, scoped per target file rather than per store, but still
  fixed-name and racy if two writers touch the same skill file concurrently.
- `tldw_chatbook/Skills_Interop/skill_trust_store.py:599` and `:611`
  (`_atomic_write_json`/`_atomic_write_bytes`, `temp_path =
  path.with_name(f".{path.name}.tmp")`) — the trust store's core write
  primitive, used for every trust mutation (bootstrap, approve, generation
  marker updates, encrypted snapshots).

This project's new ledger avoided the bug entirely by including the writer's
PID and thread id in its temp filename
(`project_prompts.json.<pid>.<tid>.tmp`) before the write-and-replace, per
its own inline comment explaining why. The two sites above should get the
same treatment.

## Acceptance Criteria (the what)

- [x] `LocalSkillsService._save_index`'s temp file name includes a
      writer-unique component (PID + thread id, or equivalent) so two
      concurrent callers never share a temp path
- [x] `LocalSkillsService._write_text_atomic`/`_write_bytes_atomic` get the
      same writer-unique temp naming
- [x] `skill_trust_store.py`'s `_atomic_write_json`/`_atomic_write_bytes` get
      the same writer-unique temp naming, preserving the existing
      `_validated_trust_file_path` containment check on the (now
      writer-unique) temp path
- [x] A test reproduces the race for at least one of the two modules (two
      concurrent writers to the same target path never raise
      `FileNotFoundError` and the final file is one writer's complete,
      valid content) and passes after the fix
- [x] Existing `Tests/Skills/` suite remains green

## Implementation Plan (the how)

1. Read the reference implementation (`ProjectSkillsPromptLedger.record()` in
   `project_skills_prompt.py`) and the two target sites' exact current
   semantics (text vs bytes, encoding, `indent=2, sort_keys=True` +
   trailing newline, the trust store's dot-prefix + containment check).
   Confirm neither module imports the other (no cycle risk) and check for
   any `chmod`/permission-bit logic on the trust material.
2. Add one shared module `Skills_Interop/atomic_write.py` with
   `unique_temp_path(path, *, hidden=False)`, `replace_atomically(temp,
   target, write_fn)`, and `write_text_atomic`/`write_bytes_atomic`
   convenience wrappers, all with cleanup-on-failure and error
   propagation (mirroring the ledger's naming scheme, but never
   swallowing the exception).
3. Write `Tests/Skills/test_atomic_write_concurrency.py` (uniqueness,
   cleanup-on-failure, semantics-preservation, and a 6-thread x
   30-iteration race reproduction against each of the 5 (still
   unconverted) call sites) and run it to capture RED evidence of the
   real race (`FileNotFoundError`) against the old fixed-name code.
4. Convert all 5 call sites to the shared helper, preserving each site's
   exact on-disk format and its mkdir/validation steps around the write.
5. Re-run the new test file (GREEN), then the full `Tests/Skills/` and
   `Tests/Workspaces/` gates and a full-suite `--collect-only` sweep.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Shared-helper choice:** a new standalone module,
`tldw_chatbook/Skills_Interop/atomic_write.py`, rather than a shared
implementation living inside either `local_skills_service.py` or
`skill_trust_store.py`. The two target modules do not import each other
(verified via their import lists), so no cycle would have forced this
choice, but a third, dependency-free module is still cleaner: both sites
import from it symmetrically instead of one site importing the other's
private helper. It exports `unique_temp_path(path, *, hidden=False)`
(`<name>.<pid>.<tid>.tmp`, or dot-prefixed when `hidden=True`),
`replace_atomically(temp_path, target_path, write_fn)` (write, replace,
best-effort `unlink` the temp file on any exception, always re-raise), and
`write_text_atomic`/`write_bytes_atomic` convenience wrappers over both.

**Preserved semantics (verified per site before changing it):**
- `_save_index`: still `json.dumps(payload, indent=2, sort_keys=True) +
  "\n"` written as UTF-8 text before the atomic replace -- same bytes on
  disk as the old `json.dump(..., indent=2, sort_keys=True)` +
  `handle.write("\n")` pair, verified by a round-trip test
  (`_save_index` → `_load_index` → equal, plus sorted-key + trailing-
  newline checks on the raw text).
- `_write_text_atomic`/`_write_bytes_atomic`: unchanged text/bytes split,
  same UTF-8 default encoding, still not dot-prefixed (matches this
  module's pre-existing visible-temp-file convention, unlike the trust
  store).
- `skill_trust_store._atomic_write_json`/`_atomic_write_bytes`: unchanged
  `indent`/`sort_keys=True` + trailing-newline JSON shape;
  `_validated_trust_file_path`'s containment check still re-runs against
  the temp path *after* the writer-unique name is built (only the name
  changed, not when/whether it's validated); the dot-prefix ("hidden
  trust file") convention is preserved via `hidden=True`. Checked for
  permission bits: neither `_atomic_write_json`/`_atomic_write_bytes` nor
  any other code in `skill_trust_store.py`/`skill_trust_crypto.py`/
  `skill_trust_models.py` calls `chmod` or sets a file mode on the
  manifest/snapshot/marker material -- confirmed by grep, so there is no
  restrictive-permissions window to preserve at these two sites (the
  `os.chmod` calls that DO exist in the skills subsystem, in
  `local_skills_service.py`, only mark extracted script files
  owner-executable and are unrelated to these atomic-write helpers).

**Error propagation unchanged:** unlike `ProjectSkillsPromptLedger.record()`
(advisory; swallows `OSError`), all 5 converted sites still let a genuine
write/replace failure propagate to their caller -- `replace_atomically`
only ever suppresses the stray-temp-file cleanup's own `OSError`, never the
original exception, which is always re-raised. Only the fixed-temp-name
COLLISION class is eliminated.

**TDD evidence:** `Tests/Skills/test_atomic_write_concurrency.py` (17
tests) was run against the *unconverted* sites first: the 5 race tests (6
threads x 30 iterations hammering the same target through each of
`LocalSkillsService._write_text_atomic`/`_write_bytes_atomic`/
`_save_index` and `skill_trust_store._atomic_write_json`/
`_atomic_write_bytes`) failed with 60-125 captured `FileNotFoundError(2,
'No such file or directory')` instances out of 180 attempts each (RED,
tee'd to `/tmp/17963-red2.txt`); after converting the 5 sites, the same
run is 17/17 green, reproduced across 4 consecutive runs with no flakes.

**Gate:** `Tests/Skills/ -q` → 453 passed (436 pre-existing + 17 new);
`Tests/Workspaces/ -q` → 279 passed; `Tests/ --collect-only -q` → 53418
tests collected, no collection errors.

**Files:** `tldw_chatbook/Skills_Interop/atomic_write.py` (new),
`tldw_chatbook/Skills_Interop/local_skills_service.py`,
`tldw_chatbook/Skills_Interop/skill_trust_store.py`,
`Tests/Skills/test_atomic_write_concurrency.py` (new).
<!-- SECTION:NOTES:END -->
