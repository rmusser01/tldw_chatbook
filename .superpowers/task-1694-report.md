# TASK-1694 report — adopt the service-owned payload-subtree finalization seam

## Summary

Ported the download-stage API (`_ManagedDownloadStage`, `_download_stage_for`,
`_finalize_download_stage`, `_discard_download_stage`, and their supporting
marker/containment/layout/state validation and retirement helpers) from
`codexclone/task-595-v2`'s `tldw_chatbook/Model_Artifacts/service.py` into this
branch's `service.py`, and retargeted `ArtifactAcquisitionService._install_artifact`
(in `acquisition.py`) at `core._finalize_download_stage` instead of
`core.install(..., consume_source=True)`. `install()` itself is unchanged and
still serves local-import callers.

## What was ported vs. reimplemented, and why

**Ported verbatim (behavior-preserving):**
- `service.py`: the entire download-stage block (`_ManagedDownloadStage`
  dataclass, `_DOWNLOAD_STAGE_SCHEMA_VERSION`/`_DOWNLOAD_STAGE_KEYS`,
  `_download_stage_for`, `_finalize_download_stage`, `_discard_download_stage`,
  `_discard_temporary_download_stage`, `_download_stage_paths`,
  `_open_download_stage`, `_validate_download_stage_handle*`,
  `_download_stage_node_identity`, `_validate_download_stage_layout`,
  `_remove_incomplete_download_stage`, `_read_download_stage_marker`,
  `_validate_download_stage_state`, `_remove_finalized_download_stage`,
  `_retire_download_stage_operation`). All helper symbols it depends on
  (`_node_identity`, `ArtifactOperationLease`, `atomic_write_json`,
  `_assert_managed_path`, `_managed_path_exists`, `_promote`, `_verify_payload`,
  `_verify_existing_destination`, `_ensure_final_parent`, `_read_manifest`,
  the error hierarchy, `_reject_duplicate_json_keys`, `_require_exact_keys`,
  `_parse_reference`, `_LOWERCASE_SHA256`) already existed on this branch with
  identical signatures/behavior, so the port required only adding `import
  tempfile` and inserting the block between `disk_usage()` and `install()`.
  `locks_path` (needed by the ported code's docstrings/history) was already
  present on this branch from an earlier TASK-595 task.
- `test_service.py`: all 31 stage tests from the source branch's
  `test_service.py`, ported unmodified in intent (same assertions, same
  fixture helpers — `descriptor`, `artifact_file`, `install_inputs`,
  `symlink_or_skip` already existed here with matching signatures). Added the
  one companion assertion the source branch's diff also added
  (`service.locks_path == root.resolve() / "locks"` in
  `test_service_validates_root_and_creates_only_owned_layout`).

**Reimplemented (the acquisition.py retargeting — no equivalent exists on the
source branch, which has no downloader):**
- `provision()`'s per-artifact loop now creates/reopens one marked stage per
  artifact via `core._download_stage_for(descriptor, create=True)` before
  fetch starts, and threads `stage.payload` through `_fetch_artifact`/
  `_preverify_artifact` (unchanged `staging_dir: Path` signatures — they
  remain stage-agnostic, plain file I/O) and the `stage` object itself into
  `_install_artifact` (new signature: `(descriptor, stage)` instead of
  `(descriptor, staging_dir)`).
- `_install_artifact` is now a thin wrapper: `core._finalize_download_stage(descriptor, stage)`
  via the existing `_run_core_call` executor-hop/error-wrapping pattern. All
  the old manual sidecar-unlink + `shutil.rmtree(staging_dir)` cleanup is
  gone — `_finalize_download_stage` retires the whole stage operation
  (payload having been promoted, plus marker and state) on success.
- `_run_core_call`'s `operation` literal gained `"stage"` and dropped
  `"install"` in favor of `"finalize"` (error messages now say `"finalize
  failed for ..."` / `"stage failed for ..."`).
- `_staged_bytes_for` (preflight's already-staged credit) now resolves the
  stage via `core._download_stage_for(descriptor, create=False)` (best-effort:
  any `ArtifactError`, or a missing stage, reads as zero credit) instead of
  computing a bare `staging/managed/<id>/<rev>/<variant>` path directly.

## The sidecar decision

`_FETCH_SIDECAR_SUFFIX` and the old `_fetch_sidecar_path` (which computed a
sibling **file** of the staging directory, `<variant>.fetch-state.json`) are
**deleted**. The stage's own `state/` subdirectory supersedes the workaround
by construction: `_validate_download_stage_layout` requires an operation
directory's immediate entries to be *exactly* `{marker, payload, state}`, so a
stray sidecar file directly inside the operation dir would fail validation —
the sidecar has to live inside `state/`. `_fetch_sidecar_path(staging_dir)` now
returns `staging_dir.parent / "state" / "fetch-state.json"`, which resolves
correctly whether `staging_dir` is a real stage's `payload/` (parent is the
stage's `operation/`, so `.../state/...` is the real `state/` dir) or an
ad-hoc test path (the sibling `state/` dir is auto-created by
`atomic_write_json`'s `mkdir(parents=True)`). This is the one substantive
design decision in the retarget: it means resume metadata is now provably
outside the promoted subtree by construction, not by convention.

I left the **old** `_MANAGED_FETCH_SIDECAR_SUFFIX` / `_gc_managed_staging` /
`_is_valid_managed_staging_entry` GC machinery **untouched** in `service.py` —
it still recognizes and reclaims the old bare `staging/managed/<id>/<rev>/<variant>`
+ sibling-sidecar shape (now unused by the new downloader, but still a valid
defensive cleanup for that shape if it ever appears, e.g. from a filesystem
that has this layout from before this change). Porting that GC to the new
`download-*` stage layout is explicitly item 4 in the reconciliation doc's
priority list, not item 1, and is out of this task's scope.

## Integration conflicts and how they were resolved

- **`install(consume_source=True)` vs. finalize.** The branches agreed this
  was exactly what to fix (reconciliation doc item 1). Kept this branch's
  `install()` unchanged (per the task brief: local import keeps it), retargeted
  only the remote-acquisition path.
- **`_install_artifact` signature.** The two candidate designs were (a) keep
  `(descriptor, staging_dir: Path)` and have `_install_artifact` internally
  re-derive/reopen the stage via `core._download_stage_for(descriptor,
  create=False)`, or (b) thread the *already-opened* `stage` object straight
  through from `provision()`'s loop. Chose (b): it avoids a redundant
  reopen+revalidate per artifact, and every direct-call test needed rewriting
  either way (their *setup* has to create a real stage now, regardless of
  which shape `_install_artifact` itself takes). `_fetch_artifact`/
  `_preverify_artifact` keep the plain `staging_dir: Path` signature since
  they never touch stage validation — this kept ~26 of their existing unit
  tests (which call them directly with an ad-hoc `tmp_path`-based dir, not a
  real stage) passing with only a sidecar-path-formula fixture update.
- **Reconcile's staging GC.** Untouched (see sidecar decision above) — this
  branch's GC categories (`install-*` orphans, `managed/` orphans) simply
  don't recognize the new `download-*` top-level shape, so it's silently
  left alone by `_gc_staging`'s "unrecognized top-level names are left
  alone" behavior. No conflict; confirmed by
  `test_reconcile_after_crash_removes_only_orphans_leaves_everything_else`
  (rewritten to check the new stage instead of the old `managed/...` path,
  but its actual claim — reconcile touches only genuine orphans — is
  unchanged and still exercises the old GC's orphan-removal correctness
  side-by-side with a live new-style stage).

## Test evidence

```
PYTHONPATH=<worktree> <worktree>/.venv/bin/pytest Tests/Model_Artifacts/ Tests/STT/test_boundaries.py -q
409 passed in ~30s
```

Baseline was 379; net +30 = +31 ported service.py stage tests, −1 (the
`_FETCH_SIDECAR_SUFFIX` drift-guard test in `test_reconcile_staging_gc.py`,
removed — its premise no longer holds, see below).

Also spot-checked unaffected adjacent suites that reference `Model_Artifacts`
(`Tests/STT/test_persistence.py`, `Tests/STT/test_contracts.py`,
`Tests/Library/test_stt_retry_lineage.py` — all reference only
`Model_Artifacts.leases`, never `service`/`acquisition`): 257 passed,
untouched.

### Files touched
- `tldw_chatbook/Model_Artifacts/service.py` — ported stage API (+691 lines)
- `tldw_chatbook/Model_Artifacts/acquisition.py` — retargeted `_install_artifact`,
  `provision()`, `_staged_bytes_for`, `_run_core_call`; removed
  `_FETCH_SIDECAR_SUFFIX`; rewrote `_fetch_sidecar_path`
- `Tests/Model_Artifacts/test_service.py` — ported 31 stage tests (+540 lines)
- `Tests/Model_Artifacts/test_provision_install.py` — rewrote the
  `_install_artifact` unit tests around real stages (including the AC #3
  regression test); rewrote the `provision()` end-to-end staging-cleanup
  assertion; fixed sidecar-path-formula fixtures
- `Tests/Model_Artifacts/test_provision_fetch.py`,
  `Tests/Model_Artifacts/test_preflight.py` — mechanical sidecar-path-formula
  fixture updates (old sibling-file convention → `state/fetch-state.json`)
- `Tests/Model_Artifacts/test_provision_crash_recovery.py` — the two
  staging-path assertions that hardcoded the old `managed/<id>/<rev>/<variant>`
  shape now locate the real stage via `core._download_stage_for`; the
  underlying claims (valid partial survives a real SIGKILL, reconcile leaves
  it alone, a fresh `provision()` resumes via Range) are unchanged
- `Tests/Model_Artifacts/test_credentials_and_boundaries.py` — one
  sidecar-existence assertion updated to the stage lookup
- `Tests/Model_Artifacts/test_reconcile_staging_gc.py` — removed the
  `_FETCH_SIDECAR_SUFFIX` drift-guard test (see below); every other test in
  this file (pure `_gc_managed_staging`/`_is_valid_managed_staging_entry`
  unit coverage, independent of acquisition.py) is untouched and still passes

## Deviations from a literal reading of the task

- **Removed one test** (`test_fetch_sidecar_suffix_mirror_matches_acquisition`
  in `test_reconcile_staging_gc.py`) rather than editing it to keep passing.
  Its only content was `assert acquisition._FETCH_SIDECAR_SUFFIX ==
  service_module._MANAGED_FETCH_SIDECAR_SUFFIX` — a drift guard whose premise
  (both modules must agree on one sibling-file sidecar suffix) is retired by
  this port, not merely relocated. I judged this is not one of the
  "crash-recovery, resume, or credential tests" the constraints call out as
  the off-limits safety net — it is a hygiene guard for a naming convention
  that no longer exists — so I removed it rather than stopping to ask,
  replacing it with a comment explaining why. Flagging it here per the
  instruction to report such calls.
- **Renamed the `_run_core_call` operation label** from `"install"` to
  `"finalize"`, which changes the *substring* two existing tests asserted on
  (`assert "install" in str(excinfo.value)` → `"finalize"`). The tests
  (`test_install_artifact_wraps_core_integrity_error_as_non_retryable`,
  `test_install_artifact_wraps_core_path_error_as_non_retryable`) still
  assert the same underlying claim (core `ArtifactError` subclasses are
  wrapped as `TransferError` with the correct `retryable` flag and
  `__cause__`); only the operation-name substring changed because the
  operation itself was renamed. I did not treat this as an "expectations"
  change requiring a stop — the assertion's *purpose* (never let a raw core
  error escape) is intact — but flagging it since it touches exactly the
  wrapping behavior the constraints singled out for caution.
- Did **not** touch `service.py`'s old `managed/` staging-GC code
  (`_MANAGED_FETCH_SIDECAR_SUFFIX`, `_gc_managed_staging`,
  `_is_valid_managed_staging_entry`) — reconciliation doc item 4, explicitly
  out of scope for item 1.

## Acceptance criteria

- [x] #1 Verified payload files are promoted by renaming the stage's payload
  subtree; no second staging copy occurs for remote acquisition —
  `_finalize_download_stage` calls `self._promote(checked.payload,
  destination)` (`os.rename`); `_install_artifact` no longer calls
  `core.install`/`_copy_payload` at all.
- [x] #2 Resume metadata cannot reside inside what gets promoted — the
  fetch-state sidecar lives under `stage.state/`, a sibling of `stage.payload/`
  enforced by `_validate_download_stage_layout`'s exact-entries check;
  structurally impossible for it to end up inside the promoted subtree.
- [x] #3 A retryable install/finalization failure leaves the durable partial
  resumable — regression test
  `test_retryable_finalize_failure_leaves_staged_bytes_resumable_via_range`
  in `test_provision_install.py` (monkeypatches
  `core._finalize_download_stage` to raise a retryable `ArtifactStateError`,
  asserts the stage's payload bytes and sidecar checkpoint survive
  untouched, then proves a fresh `_fetch_artifact` resumes via Range instead
  of re-downloading).
- [x] #4 The ported stage tests from the parallel branch pass unmodified in
  intent — all 31 pass; `Tests/Model_Artifacts/test_service.py` diff shows
  no assertion changes from the source branch's versions.
