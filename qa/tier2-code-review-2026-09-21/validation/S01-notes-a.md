# S01 — Notes A validation

## 1. P1 — save_file/export_exact_file write without fsync
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/file_notes_service.py:583` (`temporary.flush()` in `save_file`, def at :486), `:761` (`target.flush()` in `export_exact_file`, def at :689) — line numbers essentially unchanged from review.
- Proof: `grep -n "fsync" tldw_chatbook/Notes/file_notes_service.py` → no matches; `grep -n "fsync" tldw_chatbook/Notes/sync_paths.py` → 12 matches (846,869,878,986,1065,1203,1238,1317,1330,1356,1511,1516), confirming the sibling module fsyncs and this one doesn't.

## 2. P1 — stale_observation gate fed by identical expression (unreachable tautology)
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/notes_sync_runtime.py:1079-1087` (only production `ReconciliationInput(...)` call), gate at `tldw_chatbook/Notes/notes_sync_reconciler.py:700-708`; legacy dead copy at `notes_sync_legacy.py:1155-1157`.
- Proof: read of `notes_sync_runtime.py:1082-1087` shows both `observation_generation=max((item.note_version for item in observed), default=0)` and `expected_generation=max((item.note_version for item in observed), default=0)` — byte-identical expressions, so `request.observation_generation != request.expected_generation` (reconciler.py:700) can never be true.

## 3. P2 — `_bundles` cap of 8 with raise, leak window on cancellation
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/notes_sync_runtime.py:1092` (cap check `if len(self._bundles) >= _OBSERVATION_BUNDLE_LIMIT: raise RuntimeError("observation_capacity_exceeded")`), `:1094` (insert), `:1135` (`await asyncio.to_thread(build_reuse)`, awaited after insert), `:1322` (only `pop`). `_OBSERVATION_BUNDLE_LIMIT = 8` at `:114`. All 6 `release_observation` call sites (2207,2311,2543,2621,2738,2866) unchanged from review.
- Proof: `_fresh_authority` (`notes_sync_runtime.py:2182-2207`) does `observations = await self._adapter.observe_root(root)` then `plan: ReconciliationPlan | None = None` then `try: plan = plan_reconciliation(observations)` — so `plan` is only non-`None` after `observe_root` already returned; the `finally: if plan is not None and callable(release): release(...)` guard at 2205-2207 cannot fire if `observe_root` itself is cancelled after registering the bundle at line 1094 but before returning at 1136.

## 4. P2 — `FileNotesService.reconcile()` walks uncancellably while holding the service lock
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/file_notes_service.py:1174` (`self._walk_candidates()` — reconcile calls it with zero args) vs `scan()` at `:407-409` (passes `should_cancel=`, `on_progress=`); `reconcile` decorated `@_serialized` at `:1151`.
- Proof: `grep -n "\.reconcile(" tldw_chatbook/Widgets/Library/library_file_notes_workspace.py` → lines 2514 and 6180, both `asyncio.to_thread(service.reconcile)` — identical to the review's cited call sites, confirming `reconcile()` is invoked with no cancel token from the two production call sites.

## 5. P2 — ~370 lines of dead code in `sync_paths.py` (second hand-rolled atomic write, uncapped read)
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/sync_paths.py` — `scan` :601, `read_file` :662, `write_text` :1019, `create_new_text` :932, `validate_relative` :353, `_read_file` :433-475 (unchanged from review).
- Proof: the only importer of `sync_paths.PinnedSyncRoot` is `Notes/notes_sync_filesystem.py`, where `self._root.` is called only as `.read_bytes` (1x, :218), `.replace_bytes` (2x, :286,:346), `.move_file` (2x, :397,:508), `.cleanup_private_file` (2x, :414,:475) — `.scan/.read_file/.write_text/.create_new_text` never called. `_read_file` (:433-475) loops `os.read(file_fd, 1024*1024)` to EOF with no `max_bytes` ceiling, then `content = b"".join(chunks).decode("utf-8")` (can raise `UnicodeDecodeError`); the caller `read_file` (:662-691) catches only `SyncPathError`/`FileNotFoundError`/`OSError` — `UnicodeDecodeError` escapes uncaught, exactly as claimed.

## 6. P2 — three divergent root-overlap implementations; lexical one guards sync-root admission
- Verdict: CONFIRMED
- Site now: `notes_sync_filesystem.py:104-106` `_overlaps` (lexical `left == right or left in right.parents or right in left.parents`, used by `validate_sync_root_admission`); `notes_sync_coordinator.py:74-82` `_overlaps` (inode `samefile`); `notes_sync_legacy.py:290-295` `_filesystem_paths_overlap` (inode `samefile`, third copy).
- Proof: `notes_sync_legacy.py:348` calls `validate_sync_root_admission` (lexical), then `:369-386` re-runs the same three-role overlap check (`sync_roots`→root_overlap, `file_notes_roots`→file_notes_overlap, `private_paths`→private_path_overlap) using `_filesystem_paths_overlap` (inode-based) — the legacy module does not trust the lexical answer, exactly as claimed.

## 7. P2 — 18/20 slice files have no diagnostics; failure paths publish only a UI status string
- Verdict: CONFIRMED
- Site now: `notes_sync_runtime.py:1988,2010,2036,2054` (`except Exception:` handlers in `_start_once`) — unchanged from review's line numbers.
- Proof: `grep -c "logger\." <file>` for the 12 core Notes sync modules: `file_notes_service.py`=2, all of `notes_sync_runtime.py`, `notes_sync_executor.py`, `notes_sync_coordinator.py`, `notes_sync_authority.py`, `notes_sync_legacy.py`, `sync_paths.py`, `file_notes_replica.py`, `notes_sync_reconciler.py`, `notes_sync_conflicts.py`, `notes_sync_filesystem.py`, `file_notes_git_service.py`=0.

## 8. P2 — `file_notes_git_service.py` is a god module (11,527 lines, ~8,600-line class)
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/file_notes_git_service.py`, `wc -l`=11527; `class _PrivatePushProofDirectory` at :593, `class AsyncGitProcessRunner` at :1828, `class FileNotesGitService` at :2890 (spans to EOF at 11527, ~8,637 lines) — line numbers unchanged from review.
- Proof: as above (grep for the three class defs).

## 9. P3 — `Notes/` hand-writes timestamps, imports `Utils/timestamps.py` zero times
- Verdict: CONFIRMED
- Site now: `file_notes_replica.py:766-767` (`_utc_now` = `datetime.now(timezone.utc).isoformat()`).
- Proof: `grep -rn "Utils.timestamps" tldw_chatbook/Notes/` → 0 hits.

## 10. P3 — `ctypes.CDLL(None)` constructed fresh on every xattr/ACL/rename call
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/sync_paths.py:139,208,225,258,282` — identical line numbers to the review.
- Proof: `grep -n "ctypes.CDLL(None" tldw_chatbook/Notes/sync_paths.py` → exactly those 5 lines, each a fresh `CDLL(None, use_errno=True)` (or bare `CDLL(None)` at 282) construction inside a per-call function (`_xattr_call`, `_write_extended_attributes`, `_has_extended_acl`, `_rename_with_flags`, `guarded_rename_available`), no module-level caching.

## 11. P3 — `_run_keep_both` alone does not persist attention on cancellation
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Notes/notes_sync_executor.py:3156-3192` (`_run_keep_both`, `CancelledError` branch at :3191-3192 is bare `raise`) vs `_run` at :3024-3096 (CancelledError branch :3086-3091 calls `self._persist_attention_best_effort(...)`) and `_run_create_or_move` at :3102-3150 (:3140-3145 same call). Line numbers essentially unchanged.
- Proof: direct read of all three `except asyncio.CancelledError:` blocks — only `_run_keep_both`'s omits the `if admitted: self._persist_attention_best_effort(request.operation_id, "cancelled_after_admission")` call present in both siblings.

## 12. P3 [docs] — CLAUDE.md says "last-write-wins"; engine never picks a winner
- Verdict: CONFIRMED
- Site now: `CLAUDE.md:219` ("Last-write-wins conflict resolution") vs `Notes/notes_sync_reconciler.py:605` (`ReconciliationAttentionKind.CONFLICT, "both_sides_changed"` — requires explicit user choice via `NotesSyncConflictChoice`, no automatic winner).
- Proof: grep for "Last-write-wins" in CLAUDE.md (line 219, under the "Notes Sync" bullet list) and for "both_sides_changed" in notes_sync_reconciler.py (line 605, inside `_plan_bound`, a CONFLICT attention with no digest-based winner selection).

TOTALS: confirmed=12 fixed=0 wrong=0 demoted=0 promoted=0
