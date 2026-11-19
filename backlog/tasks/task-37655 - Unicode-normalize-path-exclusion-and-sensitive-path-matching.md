---
id: TASK-37655
title: Unicode-normalize path exclusion and sensitive-path matching
status: Done
assignee:
  - '@claude'
created_date: 2026-10-30 10:35
updated_date: '2026-10-31T18:45:00Z'
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
3. Add one fold, `fold_path_part` (first in `Utils/sensitive_paths.py`, moved to `Utils/path_validation.py` in round 2; canonical caseless: `NFD(casefold(NFD(x)))`), and use it for rule and candidate in `_compare_key`, `_folded_exclusions` / `_is_relative_sensitive_path` (name rule included) and git's `_repo_relative_exclusions`.
4. Regenerate the remote worker bundle, adjust its size-ratchet row with a dated comment, run the affected suites on branch and base, preflight, ruff.
5. Round 2: fold the remaining exclusion matchers (project-instruction `path_is_excluded`/`remote_path_is_excluded`, the remote preflight mirror in `local_tool_provider`), test git pathspecs against both spellings, fix the unbound `re` annotation in `local_tool_impls`.
6. Review round: fold the project index's pre-top-k helpers, the Workspace Files badge and the registry's exclusion dedupe/remove; drop the matcher's per-rule name re-fold.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: the shared matcher folded case but not Unicode normalization, so `naïve.txt` (NFC) and `naïve.txt` (NFD) compared unequal while APFS opens either as the same stored file. Fixed once at the fold, not per caller.

- `fold_path_part` (now in `Utils/path_validation.py`, see round 2; first added to `Utils/sensitive_paths.py`) is the canonical caseless match, `NFD(casefold(NFD(x)))`; ASCII folds exactly as `casefold`. `_compare_key` uses it, so the absolute denylist, `_is_within`/`_relative_within` (the root-relative translation that produces the exclusions) and the name rule all fold normalization too.
- `Tools/local_tool_impls.py`: `_folded_exclusions` / `_is_relative_sensitive_path` fold rule parts, candidate parts and the name-rule value with `fold_path_part`. This serves fs_read/list/glob/grep/mkdir/write, the pinned dispatch, the project-index scan, git path checks and code-review `_safe`. The code-review exact-listing check from PR #3286 is untouched; it now runs after the matcher already refused.
- `Tools/git_tool_impls.py`: `_repo_relative_exclusions` folds the same way, so a repo root git reports in the other normalization keeps its exclusions.
- `Tools/remote_worker_bundle.py` regenerated. The artifact on dev 94a8a85f03 was already stale (PR #3286's git review reader never re-bundled; `test_bundle_rebuild_matches_committed_artifact` failed on base). Size-ratchet row 5706 -> 5950: +16 this fix, +228 that pre-existing drift, recorded in a dated comment.
- Tests: matcher both directions for every rule kind plus case+normalization combined, ASCII case folding unchanged (`Tests/Tools/test_local_tool_impls.py`); fs_read through `WorkspaceToolExecutor` with a user exclusion (`test_local_tool_user_exclusions.py`) and through the worker's `_read_relative_file`; code-review capture refuses with `source_path_unavailable` (the matcher, ahead of the listing check) both ways (`Tests/Agents/test_code_review_source.py`, replacing the NFD-only test); git translation (`test_git_tool_sensitive_paths.py`); absolute denylist + `sensitive_exclusions_under` (`test_local_tool_sensitive_paths.py`). 20 of the 25 new cases were red before the fix (the 5 ASCII controls passed), all 25 green after.
- Verified 2026-10-31 on claude/burn-37655 (base origin/dev 94a8a85f03) through the scratch wrapper. Suites that die at setup with `RecoveryRequired: raw_source_selection_changed` were compared on both trees with the bootstrap_profile probe plugin (lessons-testing-evidence): identical failing ids, no branch-only failure.
- Round 2 (same branch, second commit):
  - `fold_path_part` moved to `Utils/path_validation.py`, which is already in the agent-runtime import footprint and the bundle closure. `sensitive_paths`, `local_tool_impls` and `git_tool_impls` import it from there, so there is still one definition and `test_agent_runtime_import_footprint` stays green.
  - `Agents/project_instruction_resolver.py`: `path_is_excluded` and `remote_path_is_excluded` fold with it, so an excluded folder's AGENTS.md does not load under the other spelling (`Tests/Chat/test_project_instruction_exclusions.py`).
  - `Agents/local_tool_provider.py`: the remote preflight mirror `refused` folds the same way, so the refusal happens before an approval card (`Tests/Agents/test_local_tool_provider.py`). Its ratchet row goes 5738 -> 5741 (+3), with a dated comment.
  - Git pathspecs: a scratch repo on git 2.39.5 (Apple) with a modified tracked `naïve.txt` and an untracked `new_naïve.txt` showed that with `core.precomposeunicode=true` (what `git init` writes on macOS) both spellings are excluded. With it `false`, an NFD rule missed an NFC file and an NFC rule missed an NFD file in `git diff` (content leaked) and in `git status`. `_denylist_pathspecs` now emits each exclusion's NFC and NFD spellings, and identical ASCII spellings dedupe, so ASCII output is unchanged. A name stored in a mixed normalization is still not covered (marked `ponytail:`). Test: `test_an_exclusion_hides_the_other_normalization_from_git` (precompose off, both directions).
  - `local_tool_impls.py` F821: `_grep_relative_files` annotates `re.Pattern` while `re` was only imported inside function bodies. With `from __future__ import annotations` this never fails at runtime; it fails only when the annotation is evaluated (`typing.get_type_hints`), and no production code does that for this function. `re` is now imported at module scope (the two function-local imports are gone); `test_grep_core_annotations_resolve` failed with NameError before.
  - Bundle row 5950 -> 5951 (+1).
- Review round (2026-10-31, claude/burn-fixA, base f58ff6e2a8):
  - `DB/Workspace_Project_Index_DB.py`: `_path_allowed` (the live re-check in `project_index/service.py`) and `_eligibility_predicate.allowed` (the SQL pre-filter for `lexical_candidates`/`has_compatible_files`, ahead of LIMIT) fold rule and path with `fold_path_part`, so an excluded row in the other normalization no longer takes a top-k slot or reports coverage. Folding the whole string equals folding each component (`/` is a starter neither fold touches; checked over every code point). Include prefixes stay casefold-only, as in the scan worker.
  - `Widgets/Console/console_workspace_files_modal.py` `_is_excluded` (badge and unexclude toggle) and `Workspaces/registry_service.py` exclusion dedupe/remove fold the same way, so an NFD duplicate of an NFC exclusion is refused and removing by either spelling works. Ratchet row 4232 -> 4233 (+1, the import).
  - `local_tool_impls._is_relative_sensitive_path`: the name rule compares the already-folded `parts[-1]` instead of re-folding `relative.name` per rule. Bundle regenerated (line count unchanged).
  - Tests, both directions: `Tests/DB/test_workspace_project_index_db.py` (both helpers per rule kind, plus an end-to-end top-k/coverage check), `Tests/Widgets/test_console_workspace_files_modal_exclusions.py`, `Tests/Workspaces/test_folder_binding_exclusions.py`. All 16 new cases were red before the fix and are green after.
- Final size-ratchet rows, restated 2026-11-02 against dev 88af9f9379 (supersede the numbers above): local_tool_provider 5,851 -> 5,854; remote_worker_bundle 5,722 (dev's fresh build 5,705 + 17); registry_service 4,304 -> 4,305.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
