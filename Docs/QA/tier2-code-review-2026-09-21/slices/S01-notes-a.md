# S01 — Notes A (sync engine, conflict resolution, file services)

**Coverage:** files read in full: 19 | sampled: 1 | mechanical only: 0 (of 20; 28,599 of 40,126 lines read line-by-line).
Sampled file: `Notes/file_notes_git_service.py` (11,527 lines) — full symbol map plus four clusters read in
full (1–1500, 1828–2925 `AsyncGitProcessRunner`, 7930–8170, 10150–10600), ~3,400 lines.

## Findings

### P1 [D1/D4] — The File Notes editor-save path writes without `fsync`, so a crash between `os.replace` and writeback loses both the old and new note bytes
- Where: `Notes/file_notes_service.py:576-594` (`save_file`); same shape at `:746-766` (`export_exact_file`)
- Evidence: `grep -n "fsync\|flush()" Notes/file_notes_service.py` → only `583: temporary.flush()`, `761: target.flush()`;
  `grep -c fsync` → `sync_paths.py:12`, `file_notes_service.py:0`, `file_notes_replica.py:0`.
  `Utils/atomic_file_ops.atomic_write_bytes` (17 importer modules) does `f.flush(); os.fsync(f.fileno())` before
  `os.replace` (`atomic_file_ops.py:176-184`).
- Why it matters: `flush()` only pushes the Python buffer to the OS. `os.replace` then publishes a name that may
  point at an inode with no data blocks committed; on power loss the user's note is zero-length or torn, and the old
  content is already unlinked. This is the primary save path for every File Notes edit, and the sibling module in the
  same package (`sync_paths.PinnedSyncRoot.replace_bytes`) fsyncs both the temp file and the parent directory.
- Recommended correction: `os.fsync(temporary.fileno())` after the write inside the `with` (line 583), and `os.fsync`
  on a parent-directory descriptor before returning. Canonical home is `Utils/atomic_file_ops.atomic_write_bytes`, but
  the TOCTOU recheck at `:591-593` and mode preservation at `:584-589` mean it cannot be dropped in verbatim — add the
  fsync in place and record why the helper is not used.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none. `Tests/Notes/test_file_notes_service.py` does not assert durability.
- Already covered: partially — TASK-32806.6 (In Progress) and TASK-32808.5 (**Done**). This is a site .5 did not
  reach, and the gap is durability, not atomicity, so a sweep keyed on `os.replace` adoption passes over it.

### P1 [D1] — The reconciler's `stale_observation` safety gate is unreachable: the only production caller derives both generations from the identical expression
- Where: `Notes/notes_sync_reconciler.py:700-708` (gate) fed by `Notes/notes_sync_runtime.py:1079-1090` (only
  constructor). Second dead copy: `Notes/notes_sync_legacy.py:1155-1157`.
- Evidence: AST walk over `tldw_chatbook/` for `ReconciliationInput(...)` call sites → one production site
  (`notes_sync_runtime.py:1079`) where `observation_generation` and `expected_generation` are both
  `max((item.note_version for item in observed), default=0)` — IDENTICAL EXPRESSION.
- Why it matters: the comparison is `x != x` and can never fire. The legacy activation authorizer makes the same
  comparison on the same object and is equally inert. What actually protects the apply path is the separate
  observation-token recomputation (`assert_review_current`, `notes_sync_runtime.py:2801`) — so this is a
  redundant-but-advertised gate, not an open hole; the risk is a future change relying on it.
- Recommended correction: feed `expected_generation` from a genuinely independent source (the binding generation the
  review was taken under), or delete both fields and both checks and let `assert_review_current` be the single
  documented staleness authority. Do not leave a validated field whose check is a tautology.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none pinning production behaviour. `Tests/Notes/test_notes_sync_conflict_runtime.py:130,526,1182` and
  `test_notes_sync_legacy_migration.py:778-884` construct `ReconciliationInput` with hand-picked unequal generations —
  the gate is exercised only by tests that build the input production never builds.
- Already covered: none.

### P2 [D1] — `_ProductionRuntimeAdapter._bundles` has a hard cap of 8 with a raise, no eviction, and a leak window: eight cancelled Checks wedge every sync pass for the process lifetime
- Where: `Notes/notes_sync_runtime.py:1092-1094` (insert + cap), `:1322` (only removal), `:1135` (awaited window)
- Evidence: one insert (1094), one `pop` (1322), five reads. All six `release_observation` call sites (2207, 2311,
  2543, 2621, 2738, 2866) are in a `finally` guarded on a value computed **after** `observe_root` returns.
- Why it matters: `observe_root` registers the bundle at 1094 then awaits `asyncio.to_thread(build_reuse)` at 1135. A
  cancellation or exception in that window leaves the token in `_bundles` with no owner and no expiry. After eight such
  events `observe_root` raises `RuntimeError("observation_capacity_exceeded")` unconditionally for every root, and sync
  never plans again until restart. `_reconcile` catches it, sets `failed/review_changes`, and the root is durably blocked.
- Recommended correction: register the bundle as the last statement of `observe_root`, or give `_bundles` LRU eviction
  instead of a raise. Prefer eviction — losing the oldest bundle costs a re-observe, which is what a miss already costs.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none. `Tests/Notes/test_notes_sync_observation_reuse.py:212` covers the happy path only.
- Already covered: none.

### P2 [D1/D2] — `FileNotesService.reconcile()` walks the whole tree uncancellably while holding the service operation lock; the task-32121 fix covers only `scan()`
- Where: `Notes/file_notes_service.py:1174` (`self._walk_candidates()` with no `should_cancel`) vs `:407-410`
  (`scan` passes `should_cancel=`/`on_progress=`). Lock held by `@_serialized` (`:62-66`).
- Evidence: callers `Widgets/Library/library_file_notes_workspace.py:2514` and `:6180`
  (`asyncio.to_thread(service.reconcile)`); the cancellable scan is at `:5857`.
- Why it matters: the `ScanCancelled` docstring (`file_notes_service.py:185-194`) describes exactly this wedge —
  "scan holds the service's operation lock for its whole run … every later folder change queued behind it for the rest
  of the session". `reconcile()` does the same `os.walk` with two `lstat`s per file, runs on every folder refresh and
  catch-up, and has no cancel seam. Picking a large root reproduces the original wedge through the other door.
- Recommended correction: give `reconcile()` the same optional `should_cancel`/`on_progress` and thread them into
  `_walk_candidates`; the workspace already owns a cancel token for `scan`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Notes/test_file_notes_service.py` covers `ScanCancelled` for `scan`; nothing for `reconcile`.
- Already covered: none.

### P2 [D3/D4] — ~370 lines of production-dead code inside the Notes trust-boundary module, including a second hand-rolled atomic write and an uncapped whole-file read
- Where: `Notes/sync_paths.py` — `PinnedSyncRoot.scan` (`:601-660`), `.read_file` (`:662-692`), `.write_text`
  (`:1019-1106`), `.create_new_text` (`:932-1017`), `.validate_relative` (`:352-356`), `._read_file` (`:433-476`),
  and the `SafeSyncFile` (`:55`) / `SyncPathIssue` (`:47`) dataclasses.
- Evidence: production caller counts across all four importing modules:
  `scan 0 | read_file 0 | write_text 0 | create_new_text 0 | validate_relative 0`;
  `read_bytes 1 | replace_bytes 2 | move_file 2 | cleanup_private_file 2`. `SafeSyncFile`/`SyncPathIssue` → 0 hits
  outside `sync_paths.py`. Reached only by `Tests/Notes/test_sync_containment.py` (23 call sites).
- Why it matters: two dead methods carry real defects a reader must re-triage. `_read_file` (`:457-463`) reads the
  whole file with **no** `max_bytes` — unlike its live sibling `_read_bytes` (`:526`,
  `max_bytes=_DEFAULT_MAX_SYNC_FILE_BYTES`) — and lets `UnicodeDecodeError` escape `read_file`, whose handler catches
  only `OSError`. `write_text` is a third hand-rolled tmp+rename in this package.
- Recommended correction: delete the five methods and two dataclasses; move any remaining value in
  `test_sync_containment.py` onto `read_bytes`/`replace_bytes`. If `scan()` is kept as a deliberate seam, cap
  `_read_file` with the same `max_bytes` and say so.
- Size: M · ADR: no · Confidence: verified
- Pinning test: `Tests/Notes/test_sync_containment.py` covers the dead methods (that is the evidence they are test-only).
- Already covered: **not** on TASK-32807 `.1`/`.2`/`.6` — a different shape (dead methods inside a live module).

### P2 [D4] — Three divergent root-overlap implementations inside this slice; the one guarding sync-root admission is lexical and the legacy module does not trust it
- Where: `Notes/notes_sync_filesystem.py:104-106` `_overlaps` (lexical:
  `left == right or left in right.parents or right in left.parents`), used by `validate_sync_root_admission`
  (`:108-130`); `Notes/notes_sync_coordinator.py:74-82` `_overlaps` (inode-based `Path.samefile`);
  `Notes/notes_sync_legacy.py:290-295` `_filesystem_paths_overlap` (inode-based, third copy).
- Evidence: `notes_sync_legacy._root_evidence` calls `validate_sync_root_admission` (lexical) at `:348`, then
  **re-runs the identical three overlap comparisons** with its own `samefile` version at `:369-386`. The module does
  not trust the lexical answer.
- Why it matters: `Path.resolve()` collapses symlinks but not bind mounts, and on macOS not firmlinks — so
  `/Users/me/Notes` and `/System/Volumes/Data/Users/me/Notes` compare unequal lexically and equal by inode. The lexical
  check decides whether a candidate sync root overlaps an existing sync root, a File Notes root, or a private path; a
  miss admits two writers for the same directory.
- Recommended correction: one `_overlaps` in `Notes/` (or `Utils/filesystem_identity.py`, which already exists),
  inode-based with an explicit `OSError` policy. Delete the lexical copy; `notes_sync_legacy` then drops its second pass.
- Size: M · ADR: no · Confidence: inferred (drift and double-check verified by reading; no firmlink/bind-mount case
  constructed — see UNVERIFIED)
- Pinning test: none asserting the lexical semantics as a requirement.
- Already covered: none (TASK-32808.9, To Do, may be the right home).

### P2 [D1] — 18 of the 20 slice files have no diagnostics at all; every failure path publishes a UI status string and records nothing
- Where: `Notes/notes_sync_runtime.py:1988, 2010, 2036, 2054` (`except Exception:` → `self._status = "failed"`),
  `:3336-3338`, `:1873-1874` (`except Exception: pass`); `notes_sync_executor.py:3672`;
  `notes_sync_coordinator.py:440,545,567,633`; `notes_sync_authority.py:411,440,512`.
- Evidence: per-file `logger.` call counts: `file_notes_service: 2`; every other slice file: `0`.
- Why it matters: `_start_once` has four `except Exception:` handlers collapsing *any* failure — schema error,
  corrupted store, bug — into `status="failed", next_action="review_settings"`, exception discarded, nothing written.
  A user reporting "sync says failed" gives a support path of zero bits. The reason codes this package computes are
  already bounded, path-free machine codes (`NotesSyncRootRefused.reason_code`, `_bounded_reason`,
  `_TYPED_REASON_CODES`) — precisely the values safe to log.
- Recommended correction: log the bounded reason code (never exception text, never a path) at the four `_start_once`
  handlers, `_run_hint`'s two, and `_inspect_resolution_undo`. This is a privacy stance taken one step too far.
- Size: S · ADR: no · Confidence: verified (counts executed); intent inferred from the module's privacy comments
- Already covered: none. Orthogonal to TASK-32806.7, which pushes the other way.

### P2 [D3] — `file_notes_git_service.py` is a god module: 11,527 lines, one class of ~8,600 lines and ~180 methods
- Where: `Notes/file_notes_git_service.py`; `class FileNotesGitService` spans `:2890` to EOF.
- Evidence: `wc -l` → 11,527. Symbol map shows 15+ responsibilities (discovery, push destination policy, push
  preflight/execution/recovery, status, stage, unstage, commit review, commit, commit recovery, orphaned-commit
  lifecycle, hooks-directory lifecycle, shutdown settlement, private proof directory). It also contains a complete
  second subsystem, `AsyncGitProcessRunner` (`:1828-2873`, ~1,050 lines) and `_PrivatePushProofDirectory`
  (`:593-1044`, ~450 lines), neither of which needs the service.
- Why it matters: largest file in the slice by 2x (second is `notes_sync_executor.py` at 5,454). Reviewers fall back
  to symbol-cluster reading, which is how the cross-cutting defects above survive.
- Recommended correction: the two self-contained subsystems lift out with no inward dependency —
  `AsyncGitProcessRunner` + `_RetainedChildRecord` + the `GitProcessRunner` protocol into
  `Notes/git_process_runner.py`; `_PrivatePushProofDirectory` + `_PrivateProofEntry` + `_ExternalProofDirectory` +
  the `_create_private_*` / `_hooks_parent_is_safe` helpers into `Notes/git_private_proof.py`. ~1,900 lines, zero
  behaviour change. Shape per `backlog/docs/library-decomposition-recipe.md`.
- Size: L · ADR: yes (new, or an entry under TASK-32809.3) · Confidence: verified
- Already covered: TASK-32809 `.2`/`.3` (In Progress). This file has **no** budget row and no decomposition note.

### P3 [D4] — `Notes/` produces timestamps by hand and imports `Utils/timestamps.py` zero times; the ADR-173 guard structurally cannot see the shape it emits
- Where: `Notes/file_notes_replica.py:766-767` (`datetime.now(timezone.utc).isoformat()`, stored in `files.deleted_at`
  and `revisions.created_at`, ordered lexically by `list_deleted` at `:389`); `Notes/notes_sync_runtime.py:1937-1938`
  (`datetime.fromtimestamp(value / 1e9, UTC).isoformat()`, display only).
- Evidence: `grep -rn "Utils.timestamps" tldw_chatbook/Notes/` → 0; repo-wide → 28 importers.
  `grep -c file_notes_replica scripts/timestamp_writer_census.tsv` → 0; **the census file is 4 lines and contains only
  comments — no data rows at all**. `scripts/check_timestamp_writers.py:21-23` states why: *"An **aware**
  `datetime.now(timezone.utc).isoformat()` does not match and is fine."*
- Why it matters: ADR-173 fixes the canonical stored shape as millisecond `Z` precisely so lexical `TEXT` ordering
  equals chronological ordering, and names microsecond-`+00:00` as one of the shapes to eliminate. The guard only
  forbids `utcnow()` and *naive* `.now().isoformat()`, so an aware-but-non-canonical writer passes silently. Today the
  column is self-consistent (every caller leaves the timestamp defaulted, so `_utc_now` is the sole writer), so nothing
  is broken now; the exposure is that an empty census reads as "zero writers left" when the true statement is "zero
  writers of the two shapes we check for".
- Recommended correction: `file_notes_replica._utc_now` → `Utils.timestamps.utc_now_iso`. Separately, extend
  `check_timestamp_writers.py` with a third kind for aware-non-canonical `.isoformat()`, or the ratchet keeps reading
  green while ADR-173 drifts.
- Size: S · ADR: no — ADR-173 already decides it · Confidence: verified
- Pinning test: `Tests/Notes/test_file_notes_replica.py` (51 passed) does not assert the timestamp shape.
- Already covered: TASK-32803.5 is marked **Done** — this shows that closure was measured with an instrument blind to
  this shape class.

### P3 [D2] — `ctypes.CDLL(None)` is constructed inside every xattr / ACL / guarded-rename call on the per-file read path
- Where: `Notes/sync_paths.py:139` (`_xattr_call`), `:208`, `:225`, `:258`, `:282`
- Evidence: `_read_bytes` (`:520-582`) calls `_read_extended_attributes` + `_has_extended_acl` twice per file (before
  and after the read); `_read_extended_attributes` calls `_xattr_call` once for the name list and once per attribute.
  For one file with M xattrs that is `2 × (2 + M)` `CDLL(None)` constructions — a fresh `dlopen` handle and ctypes
  function-cache each time. `PosixNotesSyncFilesystem.observe` runs once per file in `observe_root`'s cold path
  (`notes_sync_runtime.py:899`), bounded at 1,000 files per root.
- Recommended correction: one module-level `_LIBC = ctypes.CDLL(None, use_errno=True)`; hoist the
  `argtypes`/`restype` assignments to import time, as `git_process_containment._WindowsKernel._bind_functions`
  already does for the Windows side of the same package.
- Size: S · ADR: no · Confidence: inferred (operation count verified; cost not profiled)

### P3 [D1] — `_run_keep_both` alone does not persist attention on cancellation, unlike its two sibling run paths
- Where: `Notes/notes_sync_executor.py:3191-3192` vs `_run` at `:3086-3091` and `_run_create_or_move` at `:3140-3145`
- Evidence: both siblings do `if admitted: self._persist_attention_best_effort(request.operation_id,
  "cancelled_after_admission")` before re-raising; `_run_keep_both` computes `admitted` (`:3162, 3173, 3176`) then
  discards it on the `CancelledError` branch.
- Why it matters: a keep-both resolution cancelled mid-flight leaves the durable operation in a non-attention
  intermediate state with no marker. Still recoverable (`_restore_keep_both_operation_state` at `:4702` rebuilds from
  `conflict_substage`), so asymmetry rather than loss — reported because `admitted` is computed-and-unused there.
- Size: S · ADR: no · Confidence: inferred

### P3 [docs] — `CLAUDE.md` describes Notes sync as "last-write-wins conflict resolution"; the engine never picks a winner
- Where: `CLAUDE.md` ("Notes Sync — … Last-write-wins conflict resolution") vs `Notes/notes_sync_reconciler.py:598-609`
  and `Notes/notes_sync_conflicts.py:35-41`
- Evidence: `_plan_bound` returns `ReconciliationAttention(CONFLICT, "both_sides_changed")` when both digests moved —
  no winner. Resolution requires an explicit `NotesSyncConflictChoice` (`KEEP_FILE`/`KEEP_NOTE`/`KEEP_BOTH`/`SKIP`)
  supplied through `apply_reviewed`. The old policy names survive only as legacy config the migration drops:
  `notes_sync_legacy.py:58` `_VALID_CONFLICT_POLICIES = {"ask","disk_wins","db_wins","newer_wins"}`, reported as
  `legacy_policy_ignored` at `:570`.
- Why it matters: cost this reviewer a full first pass hunting a silent-winner data-loss shape the design does not have.
- Size: S · Confidence: verified

## Candidate triage
| Candidate row | Verdict |
|---|---|
| `except_exception_pass` `file_notes_git_service:2416,2428` | retired — swallow around `terminate()`/`kill()` in `_stop_record`, followed by `_refresh_containment_proof`, which is the actual proof |
| `except_exception_pass` `git_process_containment:1208` | retired — terminate-on-cleanup during `_create_process` unwind; re-raised on the next line |
| `except_exception_pass` `notes_sync_authority:411,440,512` | retired — create-or-verify race; each swallow is followed by an unconditional re-read that decides the outcome. A real DB error surfaces as `*_mutation_failed` |
| `except_exception_pass` `notes_sync_coordinator:440,545,567,633` | retired — `_close_quietly` and unlock-failure swallows on error-unwind paths only |
| `except_exception_pass` `notes_sync_executor:3672` | retired — `_persist_attention_best_effort` is best-effort by contract; folded into the no-diagnostics finding |
| `except_exception_pass` `notes_sync_runtime:1873,3793,3814` | 1873 **confirmed** (folded into no-diagnostics P2); 3793/3814 retired (shutdown-path closes) |
| `except_exception_return` `file_notes_git_service:7956,7966,7977,8061` | retired — fail-closed: every `return False` means "do not release retained evidence" |
| `except_exception_return` `file_notes_service:366,566,1039,1112,1363,1399,1616,1656,1712,1720` | retired — each returns a typed `OperationResult`/`_replica_warning` the UI surfaces |
| `except_exception_return` `git_process_containment:600` | retired — `owns_native_process` returns True on error, i.e. assumes ownership (fail-safe) |
| `except_exception_return` `notes_sync_executor:1166`, `notes_sync_legacy:419`, `notes_sync_runtime:814,836,3629` | retired — typed bounded projections |
| `fetchall_no_limit` `file_notes_replica:169,384` | retired — metadata-only projections bounded by files in one root; `reconcile` needs the complete set to compute deletions. A LIMIT would be a correctness bug |
| `function_body_import` `file_notes_replica:53`, `file_notes_service:252,271,282,289`, `notes_sync_runtime:1594,3845` | retired — ADR-097 boot-path avoidance, documented at `notes_sync_runtime.py:3827-3833` |
| `function_body_import` `git_process_containment:859,860` | retired — Windows-only `ctypes`/`wintypes` behind `os.name != "nt"` |
| `function_body_import` `notes_sync_conflicts:165`, `notes_sync_runtime:384` | retired — circular-import break and a lazy parser import |
| `inline_path_check_no_pv` `file_notes_git_network:2812` | retired — `_path_is_within` is an *exclusion* test returning True (= excluded) on error. Fails closed; `path_validation` has no equivalent |
| `inline_path_check_no_pv` `file_notes_git_service:10269,10367` | retired — same shape, inside an identity-pinned tree |
| `inline_path_check_no_pv` `notes_sync_runtime:3290` | confirmed, low — `folder_is_sync_root` is advisory; returns False on error, so a resolution failure silently drops the warning. Noted, not filed |
| `legacy_markers` (31 rows) | retired — all false positives (the word "retired" in prose/variable names; `notes_sync_legacy.py`/`sync_paths.py` hits are the modules' documented purpose) |
| `mutable_class_attr` `git_process_containment:1254-1330` (8 rows) | retired — `ctypes.Structure._fields_` |
| `os_replace_no_atomic` `file_notes_service:594` | **confirmed** — see P1 (durability, not atomicity) |
| `raw_1024x1024` `file_notes_git_service:234,235,241,7190,7192` | retired — named byte-limit constants at module scope |
| `raw_1024x1024` `sync_paths:459` | confirmed — `os.read(file_fd, 1024*1024)` in the **dead** `_read_file`; the uncapped read named in the dead-code P2 |
| `tempfile_no_secure` `file_notes_git_network:1610,2604`, `file_notes_git_service:10235,10262,10357` | retired — `mkdtemp`/`gettempdir` + `chmod(0o700)` + full ancestor identity pinning. Stronger than `Utils/secure_temp_files` |
| `tempfile_no_secure` `file_notes_service:576,746` | confirmed — `mkstemp` is fine; the finding is the missing fsync (P1) |
| DUP_VERBATIM `Library/library_notes_lasting_sync_state.py:444/471` ↔ `notes_sync_runtime.py:250/466` | unverified — both halves are `__post_init__` validators; Library half outside this slice. Check: `diff <(sed -n '444,470p' …) <(sed -n '250,262p' …)` |
| DUP_SHAPE `_require_db`-family (26 members) | retired for this slice — the Notes member is a one-line "raise if unset" guard; repo-wide idiom |

## D4 observations for repo-wide Phase 3
1. **Two `build_conflict_comparison` functions**, same name, same four bounds, drifted output.
   `Notes/notes_sync_conflicts.py:254-336` (+ `_bound_diff_output` at `:421`) and
   `Notes/file_notes_conflict_compare.py:169-203` (+ `_bounded_output` at `:148`). Both define
   `*_MAX_INPUT_CHARS = 200_000`, `*_MAX_INPUT_LINES = 10_000`, `*_MAX_OUTPUT_CHARS = 120_000`,
   `*_MAX_OUTPUT_LINES = 2_000` and the identical elision marker
   `"… comparison output elided at the bounded display limit."`. Drift: the sync version truncates a single over-long
   line mid-line and appends the marker (`:441-445`); the File Notes version drops the whole line and appends it
   (`:154-165`). Two "the diff was cut here" semantics, one marker string. Canonical home: `Utils/` (a
   `bounded_unified_diff` taking the four limits). Missed by the DUP rows — structurally different enough to miss the
   shape hash.
2. **Filesystem path-overlap:** three implementations, two semantics, all in this slice. `Utils/filesystem_identity.py`
   already exists and is the natural home; worth a repo-wide census of `in .parents` / `is_relative_to` / `samefile`
   overlap checks.
3. **Atomic-write count in `Notes/` alone is three**: `sync_paths.replace_bytes` (descriptor-pinned, guarded rename,
   fsync — genuinely stronger, keep), `sync_paths.write_text` (dead, delete), `file_notes_service.save_file`
   (hand-rolled, no fsync). TASK-32808.5's sweep appears keyed on modules with **no** existing atomic write; a module
   that already had one was skipped even where a second, weaker one lived alongside it. Re-run .5's census with "does
   this module contain more than one tmp+rename idiom" as the key.
4. **`Utils/timestamps.py`: 28 importers, `Notes/` 0**, and the enforcing guard's census file has no data rows. Treat
   an empty ratchet as a signal to widen the instrument, not as closure.
5. **`run_worker_coroutine` (`notes_sync_executor.py:92-121`) keeps one permanent asyncio loop per worker thread.**
   Each lazily gets its own default `ThreadPoolExecutor` the first time an inner `asyncio.to_thread` runs under it, and
   `run_until_complete` never shuts it down. Worst case ~32 loops each owning up to 32 non-daemon threads. The
   docstring names the bound it fixed (per-call loops) but not this one.

## Left UNVERIFIED
| Claim | Why | Command that would settle it |
|---|---|---|
| The lexical `_overlaps` admits a pair the `samefile` version rejects | needs a bind mount / firmlink; review is read-only | `sudo mount --bind /tmp/a /tmp/b` then `.venv/bin/python -c "from pathlib import Path; from tldw_chatbook.Notes.notes_sync_filesystem import _overlaps as lex; from tldw_chatbook.Notes.notes_sync_coordinator import _overlaps as ino; a,b=Path('/tmp/a').resolve(),Path('/tmp/b').resolve(); print('lexical',lex(a,b),'inode',ino(a,b))"` |
| `save_file` loses data on a real crash window | needs a power-cut / `dm-flakey` harness | not reproducible in `Tests/`; the argument rests on documented POSIX `os.replace` behaviour that the sibling `sync_paths` module guards against with 12 `fsync` calls |
| The `_bundles` leak is reachable from a real UI cancel | would need the app running | `.venv/bin/python -m pytest Tests/Notes/test_notes_sync_observation_reuse.py -q` after adding a test that cancels `check_root` while `build_reuse` is in flight, then asserts `len(adapter._bundles) == 0` |
| `run_worker_coroutine`'s per-thread loops each hold a live `ThreadPoolExecutor` | requires observing thread counts under a real sync pass | `.venv/bin/python -m pytest Tests/Notes/test_notes_sync_executor.py -q -p no:randomly` with a `threading.active_count()` probe before/after |
| The remaining ~8,100 unread lines of `file_notes_git_service.py` | budget; the runner and proof directory proved uniformly defensive | `sed -n '3311,7510p' tldw_chatbook/Notes/file_notes_git_service.py` — target the `_publish_successful_commit` / `_retain_uncertain_*` state transitions |
