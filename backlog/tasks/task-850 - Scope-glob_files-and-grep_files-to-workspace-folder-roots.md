---
id: TASK-850
title: Scope glob_files and grep_files to workspace folder roots
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 02:36'
updated_date: '2026-07-27 05:54'
labels:
  - tools
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
read_file, write_file and list_directory honour dev's allowed_file_roots workspace folders, but glob_files and grep_files were scoped to the tool sandbox root only when they were added. The result is strictly narrower than their siblings, so it is safe but inconsistent -- an agent can read a file it cannot find by search. Filed from the PR #953 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 glob_files and grep_files honour the same root set as read_file,Sandbox-only configurations behave exactly as before,A test covers a workspace-bound folder for both tools
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read how ReadFileTool/WriteFileTool/ListDirectoryTool resolve allowed_file_roots() and reuse the exact same call for GlobFiles/GrepFiles.
2. Add a shared _iter_candidates_across_roots() generator that globs each usable root in turn, applying containment (is_within), the sensitive-path denylist, and the hidden-component rule to every candidate from every root, with _MAX_CANDIDATES enforced globally across all roots (not per root) and results deduplicated by resolved identity.
3. Decide and implement the dotted-root rule for a root SET: check _sandbox_root_is_hidden per root, exclude a dotted root from the search rather than failing the whole call; refuse the call only when zero roots survive that filter (preserves the existing single-root/sandbox-only behavior exactly).
4. Resolve SensitivePathContext once per call in both GlobFiles.execute and GrepFiles.execute, threaded through to every candidate check.
5. Add tests: workspace-bound-folder coverage for both tools, cross-root merge without duplication, shared candidate bound across roots, the dotted-root-is-skipped-not-fatal decision, and a proof that a path outside every configured root stays refused by all five file tools (read/write/list directly, glob/grep via a symlink planted inside an allowed root).
6. Run Tests/Utils Tests/Tools Tests/Agents to confirm no regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
GlobFiles.execute/GrepFiles.execute now call allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root()) -- the SAME accessor read_file/write_file/list_directory already use -- instead of globbing the sandbox root alone.

A new shared generator, _iter_candidates_across_roots(pattern, roots, sensitive_ctx), globs each usable root in turn and yields validated candidates: containment (is_within, which also applies the sensitive-path denylist) and the hidden-component rule (_is_hidden_within) apply to every candidate from every root. _MAX_CANDIDATES is enforced as a single counter shared across all roots (not reset per root), so N configured roots cannot multiply the worst-case walk by N, and a candidate reachable through more than one root is deduplicated by resolved identity so it is never reported twice.

Dotted-root rule, decided and documented: _sandbox_root_is_hidden is now checked once per root (via a usable_roots = [r for r in roots if not _sandbox_root_is_hidden(r)] filter) rather than once against a single sandbox root. A dotted root is excluded from the search; the whole call is refused only when NO root survives that filter -- which is exactly the pre-existing single-root (sandbox-only, dotted) case, so that behavior is unchanged. This is documented in _sandbox_root_is_hidden's and _iter_candidates_across_roots's docstrings and pinned by test_dotted_workspace_root_is_skipped_not_fatal_to_other_roots.

SensitivePathContext is still resolved exactly once per call (resolve_sensitive_context()), now threaded through every root's candidates rather than just the sandbox's.

Tests added in Tests/Tools/test_file_tools_workspace_roots.py: glob/grep find files in a bound workspace folder, results merge across sandbox+bound folder without duplication, _MAX_CANDIDATES bounds the total examined across roots (not per root), the dotted-root decision, and a proof that a path outside every configured root stays refused across all five file tools (read/write/list directly against the path; glob/grep via a symlink planted inside an allowed root, since they take no target-path argument at all).

Modified: tldw_chatbook/Tools/file_operation_tools.py (GlobFiles.execute, GrepFiles.execute, new _iter_candidates_across_roots, _sandbox_root_is_hidden docstring update).
Added tests: Tests/Tools/test_file_tools_workspace_roots.py.

This task was implemented together with TASK-843 (same file, same commit sequence) since both touch GrepFiles.execute.
<!-- SECTION:NOTES:END -->
