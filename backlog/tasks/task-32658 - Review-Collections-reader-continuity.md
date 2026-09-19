---
id: TASK-32658
title: Review Collections reader continuity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 21:47'
updated_date: '2026-09-15 22:25'
labels:
  - library
  - collections
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through Local capture browsing and reader actions, keeping visible capture identity, annotations and keyboard navigation coherent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reader content and highlights belong to the selected capture across traversal and status changes.
- [x] #2 Unsaved annotation fields survive same-capture reader controls and refreshes without crossing capture authority.
- [x] #3 Reviewed controls and return focus remain readable and reachable at 170x48 and 80x24 in both themes.
- [x] #4 Targeted regression checks and a private native journey document persistence, normal shutdown and review findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/113-collections-capture-authority-and-legacy-boundary.md; backlog/decisions/055-library-destructive-action-reversibility-rule.md; backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Repair existing reader continuity and token-based controls while preserving the capture service, authority and durable storage contracts.

1. Reproduce Local capture traversal, annotation recomposition and status/Archive successor behavior through real services and production CSS; retain failing checks before fixes.
2. Repair verified identity, draft and keyboard return defects using existing controller ownership and design patterns.
3. Run affected controller/reader/service and governance checks plus a private native size/theme journey, verifying exact saved data and normal exit.
4. Update guide, QA and workflow audit, review the bounded diff, complete task and commit locally. No full suite or remote/provider execution.

Allocation: fresh origin fetch; all-history and 30 live-worktree path sweep plus content scan across 255 unique ref commits. Maximum 32657; 32658 has no prior content references. Evidence: .superpowers/sdd/2026-09-15-collections-reader/task-id-sweep.json.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved per-capture annotation drafts across reader recomposition and mode/selection changes; fenced highlight publication by identity and request generation; loaded the selected successor after filtered status mutations. Made mutation failures and stale Undo recovery explicit, kept Undo within its receipt, distinguished committed highlight writes from failed list refreshes while retaining newer drafts, and restored visible compact Save focus without stealing a newer focus choice. Existing tokens and capture authority/revision contracts remain in force.

Added 12 real-service reader journeys and repaired the geometry harness to load app CSS. 95 distinct targeted checks pass; the inherited LibraryScreen ceiling check remains red at 35,210 lines / 1,320 methods (base 35,202 / 1,319, ceilings 33,204 / 1,276). This slice adds one eight-line event forwarder; neither screen nor controller budgets were raised. Both Collections controller size checks pass. No new Ruff diagnostics; new Python files pass lint/format, changed existing ranges were formatted, and git diff --check passes. No full suite was run.

Final private native run-003 passed at 170x48 dark and 80x24 light; six rendered captures were inspected. Terminal Ctrl+Q completed normal exit 0 after the runner-posted key did not exit; the owned shell was observed and closed. Read-only checks confirm exact saved notes, four highlights, two Saved captures, ten SQLite integrity results and zero messages. Final independent review found no actionable finding. Controls use explicit focus plus Enter and direct capture selection; full Tab traversal, remote/provider calls and restart are not claimed.

Updated Collections guide, workflow audit and Docs/superpowers/qa/2026-09-15-collections-reader/README.md with evidence and limits. Preserved historical extraction docstrings in the QA appendix and retained current ownership rationale in source. ADR required: no; existing ADRs 113, 055, 086, 150 and 161 apply, as linked in the plan and QA. TASK-32659 records Clear/search, More saved searches and repeated Archive receipt follow-ups. No new general lesson beyond existing live-verification guidance; integration into dev remains pending.
<!-- SECTION:NOTES:END -->
