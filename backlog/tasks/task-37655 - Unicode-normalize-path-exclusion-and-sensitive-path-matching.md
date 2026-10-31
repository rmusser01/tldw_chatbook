---
id: TASK-37655
title: Unicode-normalize path exclusion and sensitive-path matching
status: Done
assignee:
  - '@claude'
created_date: 2026-10-30 10:35
updated_date: '2026-10-31T09:45:00Z'
labels:
- security
- tools
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On APFS, and other filesystems that ignore Unicode normalization when opening files, an NFD spelling of a path reaches a file that the user excluded under its NFC spelling. The shared exclusion and sensitive-path matcher (local_tool_impls._relative_target_is_safe / _is_relative_sensitive_path) casefolds but never normalizes. So a user's binding exclusion with a non-ASCII name (for example naïve.txt, or a folder like Données/) can be bypassed, and the file's content reaches a model. The same applies to names git reports in the other normalization. Found during the TASK-35582 (PR #3286) local review: a reviewer captured an excluded NFD-spelled file as review source. The PR closed its own selection path with an exact-name listing check, but the shared helper is older and also serves fs_read. Built-in sensitive names are ASCII and unaffected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An exclusion or sensitive-path rule matches a path regardless of the NFC/NFD spelling of either the rule or the candidate path
- [x] #2 fs_read and code-review source capture both refuse an NFD-spelled selection of an NFC-excluded file, and an NFC-spelled selection of an NFD-excluded one
- [x] #3 Regression tests prove both directions and that ASCII case-insensitive matching is unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep every caller of `_relative_target_is_safe` / `_is_relative_sensitive_path` (fs_*, git_*, dispatch, project-index scan, code-review source, remote bundle) and the folding sites that feed them.
2. Write regression tests first (both NFC/NFD directions, ASCII case folding unchanged) for the matcher, fs_read (executor and worker core), code-review capture, git exclusion translation and the absolute denylist; show them red.
3. Add one fold, `sensitive_paths.fold_path_part` (canonical caseless: `NFD(casefold(NFD(x)))`), and use it for rule and candidate in `_compare_key`, `_folded_exclusions` / `_is_relative_sensitive_path` (name rule included) and git's `_repo_relative_exclusions`.
4. Regenerate the remote worker bundle, adjust its size-ratchet row with a dated comment, run the affected suites on branch and base, preflight, ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: the shared matcher folded case but not Unicode normalization, so `naïve.txt` (NFC) and `naïve.txt` (NFD) compared unequal while APFS opens either as the same stored file. Fixed once at the fold, not per caller.

- `Utils/sensitive_paths.py`: new `fold_path_part` (canonical caseless match, `NFD(casefold(NFD(x)))`; ASCII folds exactly as `casefold`). `_compare_key` uses it, so the absolute denylist, `_is_within`/`_relative_within` (the root-relative translation that produces the exclusions) and the name rule all fold normalization too.
- `Tools/local_tool_impls.py`: `_folded_exclusions` / `_is_relative_sensitive_path` fold rule parts, candidate parts and the name-rule value with `fold_path_part`. This serves fs_read/list/glob/grep/mkdir/write, the pinned dispatch, the project-index scan, git path checks and code-review `_safe`. The code-review exact-listing check from PR #3286 is untouched; it now runs after the matcher already refused.
- `Tools/git_tool_impls.py`: `_repo_relative_exclusions` folds the same way, so a repo root git reports in the other normalization keeps its exclusions.
- `Tools/remote_worker_bundle.py` regenerated. The artifact on dev 94a8a85f03 was already stale (PR #3286's git review reader never re-bundled; `test_bundle_rebuild_matches_committed_artifact` failed on base). Size-ratchet row 5706 -> 5950: +16 this fix, +228 that pre-existing drift, recorded in a dated comment.
- Tests: matcher both directions for every rule kind plus case+normalization combined, ASCII case folding unchanged (`Tests/Tools/test_local_tool_impls.py`); fs_read through `WorkspaceToolExecutor` with a user exclusion (`test_local_tool_user_exclusions.py`) and through the worker's `_read_relative_file`; code-review capture refuses with `source_path_unavailable` (the matcher, ahead of the listing check) both ways (`Tests/Agents/test_code_review_source.py`, replacing the NFD-only test); git translation (`test_git_tool_sensitive_paths.py`); absolute denylist + `sensitive_exclusions_under` (`test_local_tool_sensitive_paths.py`). 20 of the 25 new cases were red before the fix (the 5 ASCII controls passed), all 25 green after.
- Verified 2026-10-31 on claude/burn-37655 (base origin/dev 94a8a85f03) through the scratch wrapper. Suites that die at setup with `RecoveryRequired: raw_source_selection_changed` were compared on both trees with the bootstrap_profile probe plugin (lessons-testing-evidence): identical failing ids, no branch-only failure.
- Not changed (same class, separate re-implementations): `Agents/project_instruction_resolver.py` `path_is_excluded` / `remote_path_is_excluded` and the remote preflight mirror `refused` in `Agents/local_tool_provider.py` still casefold without normalizing. The worker re-enforces for the latter; the former decides which AGENTS.md files load. Candidates for a follow-up.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
