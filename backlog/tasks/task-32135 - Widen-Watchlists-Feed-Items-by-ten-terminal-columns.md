---
id: TASK-32135
title: Widen Watchlists Feed Items by ten terminal columns
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 07:07'
updated_date: '2026-09-09 07:11'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-09-09-watchlists-feed-items-width-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give the middle Feed Items pane more room for article titles while preserving the existing Reader and responsive pane behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Read-mode Feed Items uses a 42-column minimum and 50-column preferred maximum.
- [x] #2 Read collapse and reopen thresholds account for the extra ten columns; management layout behavior remains unchanged.
- [x] #3 Focused layout tests and CSS bundle checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Apply the existing Feed Items width design: CSS bounds 42–50, matching resolver minimum 42.
2. Update existing Read-mode layout and hysteresis expectations; keep management expectations.
3. Regenerate CSS, run focused layout and bundle tests, and review the scoped diff.
ADR required: no
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: Small presentation adjustment within the accepted Reader and responsive side-pane policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Widened Read-mode Feed Items from 32–40 to 42–50 terminal cells in features/_watchlists.tcss and matched the 42-cell resolver minimum in region_layout.py. Regenerated tldw_cli_modular.tcss; comparison against the pre-build bundle confirmed only the Watchlists width rules and generated timestamp changed. Updated existing geometry, responsive, hysteresis, and scoped-rebuild expectations; management thresholds remain unchanged.

Validation: 172 targeted tests passed (2 warnings) across test_watchlists_responsive_layout.py, test_watchlists_workbench.py, test_watchlists_layout_hysteresis_probe.py, test_watchlists_scoped_rebuilds.py, test_css_bundle_sync_guard.py, and test_css_build_integrity.py. Ruff lint and git diff --check passed. Range-scoped Ruff formatting passed for all 71 edited ranges in five Python files. Whole-file Ruff format checks also flag all five original HEAD versions, so existing formatting was preserved. Self-review confirmed the CSS/resolver bounds agree and unrelated workspace edits were preserved.

ADR required: no; this implements the existing width design within backlog/decisions/042-watchlists-reader-first-ia.md. No new logic, dependencies, storage, or security boundaries were introduced.

PR verification against dev at 9faf96d9f8: 192 of 193 targeted tests passed.
The unchanged test_management_scope_invalidates_reader_return_to_read_failure_is_honest
fails because ArticleListPane.display is True where False is expected. The same
single-test failure was reproduced independently in a clean checkout of that dev
commit, establishing a pre-existing baseline failure. All edited geometry and
responsive tests, CSS bundle checks, Ruff lint, 71 edited formatting ranges,
whitespace checks, and the backlog ID/path guard passed. The bundle regenerated
on dev changes only the four Watchlists width declarations.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

The CLI assigned TASK-31384, already used by the older Console unified interrupt
surface task on remote branches. Renumbered this new task to TASK-32135 after
sweeping local remote refs and worktrees (highest observed ID: 32115).
