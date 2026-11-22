---
id: TASK-42191
title: "Repair stale Tool Pack tests broken by later product changes"
status: Done
assignee:
- '@claude'
created_date: 2026-11-21 13:40
updated_date: 2026-11-22 19:14
labels:
- tests
- tool-packs
priority: medium
references:
- Tests/Tool_Packs/test_catalog_snapshot.py
- Tests/Tool_Packs/test_export.py
- Tests/Tool_Packs/test_binding_guard.py
- Tests/Tool_Packs/test_permission_profile_authority.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A triage of the 109 `Tests/Tool_Packs` tests that fail on Windows (2026-11-21) found that four causes are not Windows bugs. The tests are stale: later product changes broke them, and nobody noticed because no PR-time CI lane runs `Tests/Tool_Packs`.

- **C7** (2 tests). `b5251e9a6e` (TASK-32628, raw source selection) makes the real V1 inventory fail under the per-test config redirect. `3725b2b653` put the word "root_alias" into tool prose, so `b"root_alias" not in schema_bytes` is false on every platform. The path-privacy check also passes vacuously on Windows, because canonical JSON doubles backslashes.
- **C8** (1 test). Since `03cd621579` (TASK-32801.5), the store transaction rolls back on `BaseException`. A `KeyboardInterrupt` inside the write block therefore leaves no binding, but the test still expects one.
- **C9** (1 test). Since `37273cff7b`, `_PATH_LOCKS` holds `_PermissionPathFence` values, and the guarded wrapper calls `acquire(timeout=...)`/`release()`. The test's `_ContestedLockProbe` has neither shape.
- **C10** (1 test, Windows only). The directory-fsync failure is injected through stdlib `os`. On Windows the store opens and flushes the directory through the native barrier (TASK-33422), so the injection never fires. Its tolerance sibling passes vacuously there.

This task repairs the tests only. The product behaviour they now contradict is correct.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The C7 tests run against the collection-time config profile and pass; the `root_alias` check matches the JSON property name only; the path-privacy check also rejects the JSON-escaped sentinel path
- [x] #2 The C8 test expects the rolled-back state (no binding) after an interrupt inside the write block
- [x] #3 The C9 test injects a real `_PermissionPathFence` whose lock is a probe supporting both `acquire`/`release` and the context manager, and passes
- [x] #4 On Windows, the C10 test injects its directory-fsync failure at the native barrier and passes; `test_save_tolerates_unsupported_directory_fsync` is skipped on Windows rather than passing vacuously
- [x] #5 The four touched test files show no new failures against `dev` on Windows
- [x] #6 No product code changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Apply the triage's validated test patches for C7, C8 and C9/C10.
2. Run the five target tests on Windows with the private-root runner.
3. Run the four touched files on this branch and on a `dev` baseline; compare the failure sets.
4. Run `./scripts/preflight.sh`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
This change touches tests only. Each stale test now encodes the current, correct product behaviour.

- **C7** (`test_catalog_snapshot.py`, `test_export.py`): both tests are marked `@pytest.mark.bootstrap_profile`, following the precedent in `Tests/Agents/test_local_tool_provider.py`. The real V1 inventory builds `LocalToolProvider` default specs, which read the config participant admitted at collection; under the per-test redirect, that read raises `raw_source_selection_changed`. The leak check now looks for the JSON property `"root_alias"`, not the bare word that tool prose has contained since `3725b2b653`. A second check rejects the JSON-escaped sentinel path, because canonical JSON doubles Windows backslashes and the raw-path check passed vacuously there.
- **C8** (`test_binding_guard.py`): an interrupt inside the write block now expects `assistant_defaults is None`, matching the rollback-on-`BaseException` transaction (TASK-32801.5) and its sibling test.
- **C9** (`test_permission_profile_authority.py`): `_ContestedLockProbe` gains `acquire(blocking, timeout)`/`release()` with the same instrumentation, keeping the context manager on top of them. The test injects a real `_PermissionPathFence` with `.lock = probe`.
- **C10** (same file): on `nt`, `_fail_directory_fsync` records the fd that `platform_files.os.open` returns for the target directory, and makes `file_durability.flush_directory` raise for it. The POSIX branch is unchanged. `test_save_tolerates_unsupported_directory_fsync[*]` is skipped on `nt`, where the native barrier propagates every errno by design.

Verification (Windows 10, Python 3.12, `Tests/windows_private_fixture_runner` private root, 2026-11-21, branch `claude/tool-pack-stale-tests` against `dev` at `e5626ea043`):

- The four touched files on `dev`: 5 failed (exactly the five targets), 80 passed, 1 skipped.
- On the branch: 0 failed, 81 passed, 5 skipped. The 4 new skips are the tolerance cases; the 1 existing skip is the symlink-privilege case.
- Negative control: un-skipping the tolerance cases on this branch makes all four fail on `nt` (errno 22, 40, 129 and 10045 propagate). So the skip removes a vacuous pass and hides no regression.
- The triage confirmed C8's cause: restoring the old `except Exception` transaction made the old assertion pass again.
- Linux was not run, because no Linux host was available. C7, C8 and C9 have no platform-specific paths, so they are expected to fail on Linux on `dev` and to pass with this change. Nothing caught them because no PR-time lane runs `Tests/Tool_Packs`.
- `ruff check` on the four files reports only the pre-existing B009 at `test_binding_guard.py:202`, which `dev` reports too.

**Port (2026-11-22).** PR #3687's branch was 141+ commits behind `dev` with red CI, so its commit (63e840579a) was cherry-picked with `-x` onto `fu/r21-staletests` from `dev` ddd3c02052. The Windows evidence above is from the original branch on `dev` e5626ea043 and was not re-run on this port. `dev` has not repaired any of the five tests another way.
- **Red/green on macOS** (CPython 3.12.11), the four touched files, `-n 4`: on `dev` ddd3c02052's versions of the files, 9 failed / 77 passed, including all four POSIX-reachable targets (C7 `test_concrete_v1_provider_inventory_is_complete_unbound_and_path_private` and `test_v1_zero_tool_namespace_receives_a_portable_fallback`, C8 `test_process_control_exception_propagates_without_marker_clear`, C9 `test_two_store_instances_share_one_path_fence`). With the port, 5 failed / 81 passed, and all four targets pass. C10's `nt` branch and skip cannot run here.
- **The other 5 reds were not Windows-only either; fixed here as an addition beyond the criteria** (coordinator decision, 2026-11-22). The reason: the same file's stale fixture left this repair incomplete on POSIX. `test_save_propagates_real_directory_fsync_failure` and the four `test_save_tolerates_unsupported_directory_fsync[*]` cases failed on macOS on `dev` and on the cherry-picked port. `_fail_directory_fsync`'s POSIX branch remembered the directory's fd *number*. Storage admission (`storage_admission._resolved`) now opens and closes that directory first, so `mkstemp` reused the number. The injected failure then fired on the temp file's own `os.fsync` in `_save_locked`, which tolerates nothing. `controlled_fsync` now matches the directory by identity (`os.path.samestat(os.fstat(fd), os.stat(directory))`). Red/green, macOS, `-n 4`: the file went from 5 failed / 16 passed to 21 passed. The propagate case asserts the injected `EIO`, which shows the matcher still fires on the directory. POSIX always hands out the lowest free fd, so Linux should behave the same (not run).
- Whole `Tests/Tool_Packs` on this branch: 420 passed, 0 failed. The other two Tool Pack ports (TASK-42225, PR #3684) fix Windows-only paths, so no reds remain for them here.
<!-- SECTION:NOTES:END -->
