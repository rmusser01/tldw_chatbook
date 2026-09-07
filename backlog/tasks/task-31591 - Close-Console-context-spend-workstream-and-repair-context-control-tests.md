---
id: TASK-31591
title: Close Console context spend workstream and repair context control tests
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 03:40'
updated_date: '2026-09-05 05:24'
labels:
  - console
  - tests
  - documentation
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the merged automatic-compaction and context/spend workstream with authoritative records and current settings-contract test coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The six pre-existing context-control failures pass using the current settings submission and provider-default ownership contracts.
- [x] #2 The merged compaction and Current/On next send behavior is documented with accurate validation evidence and a unique task ID.
- [x] #3 Focused test and static checks pass, with the workstream cleanup targets documented separately from unrelated user work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the six failures on merged dev and compare their fixtures and assertions with the current settings submission/default-persistence contracts.
2. Reuse the mounted popover fixture and migrate assertions to committed submissions while retaining provider-default isolation and geometry coverage.
3. Publish the corrected context/spend design and completed implementation record, including PR #2397 evidence and the local TASK-31382 collision provenance.
4. Run focused context-control, popover, and settings-default tests plus scoped formatting/lint and documentation checks; review, merge into dev, then clean up only this workstream's branches/worktrees with recoverable preservation of its conflicted attempt.

ADR required: no
ADR path: backlog/decisions/095-conversation-owned-console-generation-settings.md
Reason: test and documentation repair within the existing settings ownership and merged context/spend contracts; no production interface or policy changes.

## Task ID provenance

The CLI initially offered TASK-31430. A remote-ref and worktree census found IDs through TASK-31567, so this task was assigned TASK-31568 before implementation. The original local context/spend task used TASK-31382, which belongs to the unrelated ask_user attribution task on dev. This closeout record references the original work by PR #2397 and does not replace dev's TASK-31382.

## Renumbering provenance

PR #2401 landed while PR #2403 was completing CI and introduced the older Library Reader TASK-31568 (created 2026-09-05 03:22, versus this task at 03:40). Following the older-keeps-ID rule, this closeout moved from TASK-31568 to TASK-31585. A fresh sweep across all remote refs and worktrees found a maximum of TASK-31584; the spec and completed implementation record were updated together. The existing Library task remains unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced the six stale failures on merged dev (25 passed, 6 failed), then updated the context-control tests to the conversation-owned settings submission contract. Reused the existing popover fixture, asserted explicit Automatic reselection, retained the provider-default ownership exclusion through the defaults intent builder, and corrected the action-row selector. The mounted harness now loads the same consolidated and split stylesheets as production. No application behavior changed.

Published the corrected Current/On next send design and completed implementation record, preserving the feature PR's evidence separately from this follow-up. ADR-095 already governs the tested boundary; no new ADR was required. The existing testing lessons already describe the stale-fixture and consolidated-CSS traps, so no duplicate lesson was added.

Validation: 158 focused tests passed across context controls, rail/popover settings, and the real settings-default persistence service, including after rebasing onto dev's interrupt-host change. Scoped Ruff lint/format, diff whitespace checks, and the backlog ID/path guard passed. The subsequent Library follow-up merge changes documentation only; its task-ID collision is reconciled above. The full suite was not run. Pytest emitted a Requests dependency-version warning and temporary-directory cleanup warnings; neither was a test failure.

Cleanup after merge is limited to the Console spend feature/port/baseline/closeout worktrees and topic/backup refs. Preserve the conflicted port and branch history in a verified recovery archive first. The original records in the shared main checkout are tracked on an unrelated branch and remain untouched; the authoritative dev replacements above supersede them.

Renumbering provenance (PR #2404, 2026-09-05): moved this Console closeout from TASK-31585 to TASK-31591 after PR #2403 merged alongside the older Buddy task. Buddy created_date is 03:32; this closeout is 03:40. TASK-19601 gives the older task the ID regardless of Done status. A fresh sweep of 334 refs and 63 worktrees found max TASK-31590. Updated both spend plan/spec links; earlier TASK-31568-to-31585 provenance is retained as history. No implementation or acceptance status changed.
<!-- SECTION:NOTES:END -->
