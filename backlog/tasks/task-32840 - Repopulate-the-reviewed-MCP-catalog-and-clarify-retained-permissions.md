---
id: TASK-32840
title: Repopulate the reviewed MCP catalog and clarify retained permissions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 14:42'
updated_date: '2026-09-19 14:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep reviewed server definitions visible and explain the fresh-defaults outcome without suggesting that historical grants become active.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current successful review shows the approved local server definitions and fresh permission state without discovery, connection or tool grants.
- [x] #2 Navigation or service replacement during a pending catalog read prevents stale publication; a failed catalog read preserves successful approval and provides a retry action.
- [x] #3 Guidance remains accurate before and after review; real-owner targeted tests and native dark/light evidence verify the bounded workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the post-review empty catalog and verify the actual local catalog reader remains passive after owner approval.
2. Load reviewed local definitions without reload/discovery/connection, publish only while PR2727 view ownership remains current, and distinguish a catalog-read failure from successful owner approval. Clarify that historical grants remain inactive before and after review.
3. Run real-owner success/failure/navigation tests and adjacent controls, static/derived guards and independent review. Verify dark/light native post-review server rows and guidance; save a bounded draft PR stacked on PR2727.
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: routine UI publication and copy repair using the existing passive local catalog service and approval boundary; no new owner policy, persistence or service contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented passive catalog repopulation after restored-root review. Fresh local definitions and built-in readiness publish only while the completion token remains current; a read failure preserves successful approval and offers the existing r retry. Historical rules/grants remain inactive in the updated guidance. No discovery, connection, grant or service-boundary change.

Verification: 49 distinct targeted cases pass (6 new, 18 parent publication, 17 controls, 3 actual owner and 5 adjacent checks); seven preflight guards pass; no introduced Ruff diagnostics; changed ranges and new files formatted. Independent review found no blockers. Eight native captures cover dark/light at 120x40 and 170x48. The final private app exits 0 with absent process, released lock, ten healthy databases, unchanged defaults and matching final source/runner hashes. One unrelated Evals owner-enrollment diagnostic is documented. Initial native harness assumed an absent disconnected client existed; corrected assertion and final-source rerun pass. No full suite ran.

Reviewed the sole new warning through the existing secret/path redactor and updated the diagnostic inventory count/digest. No sink changed. Production files: mcp_workbench.py and mcp_permissions_mode.py. Regression tests, QA README/gallery and component ledgers retain evidence and remaining scope.

ADR required: no; implements existing backlog/decisions/126-complete-local-backup-and-recovery.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md. This branch stacks on PR2727 at d6c0312bedff15d8c2e8e9589a70c5be7c60379b; the draft targets codex/mcp-restored-roots-review and should retarget dev after parent merge. Current-head CI and final visual approval still gate merging. Next bounded review: bulk permission actions and approval workflows.

Evidence: Docs/superpowers/qa/2026-09-19-mcp-recovery-catalog/README.md.
<!-- SECTION:NOTES:END -->
