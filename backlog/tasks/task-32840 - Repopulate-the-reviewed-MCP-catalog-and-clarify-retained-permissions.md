---
id: TASK-32840
title: Repopulate the reviewed MCP catalog and clarify retained permissions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 14:42'
updated_date: '2026-09-20 06:32'
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

Current-dev integration (after PR2727 merge):
4. Rebase this saved draft onto verified merged dev ebee42fab8, retain both ledger histories and regenerate the reviewed diagnostic inventory from combined source.
5. Reuse the accepted shared native CLI validation and app warm-up entry point; prove runner boundary rejection before replacing the old entry point. Rerun current product/parent/adjacent tests and seven derived guards.
6. Run the real private native journey in both themes and sizes, retain shutdown/default fingerprints, independently review integration and publish an updated PR against dev. This PR requires its own current-head CI/review and owner visual approval before merge.
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: existing passive publication and QA boundaries; no new product or service contract.

Qodo follow-up: add isolated direct-method coverage for successful catalog conversion, malformed returns, reader exceptions and stale receipt ownership. Trace the existing store/profile validation and add a real store/service regression for malformed nested input and valid siblings before accepting a duplicate UI schema. Preserve the current product boundary and gallery if behavior does not change; document the verified disposition of each finding.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented passive catalog repopulation after restored-root review. Fresh local definitions and built-in readiness publish only while the completion token remains current; a read failure preserves successful approval and offers the existing r retry. Historical rules/grants remain inactive in the updated guidance. No discovery, connection, grant or service-boundary change.

Verification: 49 distinct targeted cases pass (6 new, 18 parent publication, 17 controls, 3 actual owner and 5 adjacent checks); seven preflight guards pass; no introduced Ruff diagnostics; changed ranges and new files formatted. Independent review found no blockers. Eight native captures cover dark/light at 120x40 and 170x48. The final private app exits 0 with absent process, released lock, ten healthy databases, unchanged defaults and matching final source/runner hashes. One unrelated Evals owner-enrollment diagnostic is documented. Initial native harness assumed an absent disconnected client existed; corrected assertion and final-source rerun pass. No full suite ran.

Reviewed the sole new warning through the existing secret/path redactor and updated the diagnostic inventory count/digest. No sink changed. Production files: mcp_workbench.py and mcp_permissions_mode.py. Regression tests, QA README/gallery and component ledgers retain evidence and remaining scope.

ADR required: no; implements existing backlog/decisions/126-complete-local-backup-and-recovery.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md. This branch stacks on PR2727 at d6c0312bedff15d8c2e8e9589a70c5be7c60379b; the draft targets codex/mcp-restored-roots-review and should retarget dev after parent merge. Current-head CI and final visual approval still gate merging. Next bounded review: bulk permission actions and approval workflows.

Evidence: Docs/superpowers/qa/2026-09-19-mcp-recovery-catalog/README.md.

Current-dev integration: PR2727 merged at ebee42fab8 after owner approval and current-head CI/review. Concurrent PR2751 changed unrelated UI files during merge; the actual combined app passed 85 affected checks and its native journey before this draft resumed. Saved PR2728 is rebased onto that merge at 0489b76d7d. Retained every dev ledger line plus the saved checkpoint; regenerated the sole reviewed diagnostic addition to 7,761 total calls. No product conflicts or extra product changes.

The old native runner now reuses shared CLI validation before effects, uses validated arguments/PATH tmux and the app image warm-up. Seven malformed-CLI failures reproduced before the fix; 77 runner cases now pass. Product/parent/adjacent tests pass 91 cases, including all 49 original selected cases (168 distinct current passes overall). Seven guards pass, no new Ruff diagnostics, changed ranges/new files formatted. Two independent read-only reviews found no blockers. Eight final native captures were individually inspected in dark/light at 120x40 and 170x48; real approval/cancellation, fresh defaults, retained history and disconnected catalog detail pass. Normal shutdown, ten healthy DBs, released lock, zero conversations/messages and unchanged defaults verified; one known unrelated Evals enrollment diagnostic retained. No full suite ran.

Current evidence: Docs/superpowers/qa/2026-09-19-mcp-recovery-catalog/CURRENT-DEV-REVIEW.md. Existing ADR-126/150/161 apply; no new ADR needed. PR2728 targets dev after this push and retains its own current-head CI/review and visual approval gate. Remaining scope: long-path presentation, inspector refresh, connected-runtime journeys and saved PR2730.

Qodo e57b3f6dc7: zero bugs and two rule findings. Added 14 isolated completion/storage cases; the focused run passes all 20 cases including the six existing owner journeys, bringing the distinct total to 182. The reported non-mapping env_placeholders field is already normalized through LocalExternalMCPProfile.from_storage_dict before the service emits it; four real store/service regressions retain both profiles and verify passive readiness. Independent review confirms no duplicate UI Pydantic schema is needed for that reported case. QODO-REVIEW.md records the narrow evidence and does not claim strict validation of every nested field. New test lint/format pass. Product and runner bytes are unchanged; current native hashes still match. Final current-head CI/review and owner visual approval remain.
<!-- SECTION:NOTES:END -->
