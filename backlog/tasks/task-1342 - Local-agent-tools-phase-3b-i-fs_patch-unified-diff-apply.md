---
id: TASK-1342
title: 'Local agent tools phase 3b-i: fs_patch (unified-diff apply)'
status: Done
assignee: []
created_date: '2026-08-05 17:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md §2.4. Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3b-i.md. ADR-032. Port of tldw_server filesystem_diff.py @ 5605b9d9.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 fs_patch applies multi-file multi-hunk unified diffs confined to the workspace root
- [x] #2 Context mismatches, deletes, renames, and malformed diffs return model-actionable errors without writing
- [x] #3 dry_run returns the would-be result and writes nothing
- [x] #4 Diff size/file/hunk limits enforced; writes are encode-before-write and newline-preserving
- [x] #5 All new tests pass
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3b-i.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented on branch `feat/local-agent-tools-p2` (stacked on PRs #1352/#1358) via subagent-driven development with per-task spec + quality review.

- `Tools/patch_tool_impls.py` (new): near-verbatim port of tldw_server's `filesystem_diff.py` @ 5605b9d9 (attribution header per re-plan §5) — `parse_unified_diff`/`apply_patch_to_text` with all 12 reason codes, limits (256 KiB / 20 files / 200 hunks) — plus the `patch_files` workspace wrapper: per-file `resolve_workspace_path` confinement, modify-exists/create-absent/parent-exists checks, `newline=""` reads, encode-before-write, dry_run, reason-code-preserving `LocalToolError` translation, per-file summary, sequential-apply atomicity non-goal (documented in docstring AND the model-facing tool description).
- `Agents/local_tool_provider.py`: `fs_patch` spec — `diff` required, `dry_run` optional, `tags=("mutates",)`; description teaches the diff format (anti-hallucination) and defers single replacements to `fs_edit`.

Review-driven fixes beyond the plan — three bugs INHERITED from the reference, fixed as documented deviations (module header "Deviations from reference" list, GNU-verified):
1. Pure-insertion hunks (`@@ -N,0`) applied off-by-one — silent corruption; now byte-identical to GNU `patch` output (cross-check test).
2. Real multi-file `git diff` output (`diff --git`/`index` separators) was unparseable; preamble now skipped between file sections (real git-generated fixture test).
3. Removing a line whose content starts with `-- ` broke sentinel-based hunk parsing; hunk bodies now terminate on header-count satisfaction (truncated hunks still raise `invalid_hunk_line_count`).
Plus: BOM-prefixed diffs stripped before parsing.

Known limitations (recorded, non-blocking): (a) a hunk removing `-- x` immediately followed by adding `++ y` trips a header-pair heuristic and fails loudly (`invalid_hunk_line_count`) though GNU patch accepts it; (b) hunk bodies with MORE lines than declared have the surplus silently skipped (inherent to the count-based fix); (c) multi-file apply is non-atomic by design.

Tests: 23 core + 4 provider + integration catalog updates; full suites green (383 passed in Tests/Tools+Tests/Agents at final review).

Final whole-phase review: Ready to merge; all 5 ACs verified.
