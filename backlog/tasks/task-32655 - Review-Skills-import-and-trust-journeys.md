---
id: TASK-32655
title: Review Skills import and trust journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 19:04'
updated_date: '2026-09-15 20:08'
labels:
  - library
  - skills
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through package import and exact trust review with readable dialogs, truthful recovery and usable keyboard focus.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Trust setup and passphrase dialogs use token-backed surfaces and readable text, errors and focus in both themes at wide and compact sizes.
- [x] #2 Import errors, candidate choice, Cancel and successful Review handoff preserve authoritative operation state and reach usable controls.
- [x] #3 Trust review displays the captured file contents; approval accepts only that exact snapshot and stale approval remains blocked with a recoverable next action.
- [x] #4 Targeted UI, import, trust and token checks plus private native and persistence evidence qualify the reviewed journeys.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/009-local-skill-trust-boundary.md; backlog/decisions/076-library-lifecycle-progressive-disclosure.md; backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Repair existing presentation, focus and recovery journeys without changing import ownership, trust authority or storage.

1. Reproduce trust-dialog readability and exercise import error/recovery, candidate Cancel/selection and Review handoff with real local services and production CSS.
2. Repair only reproduced dialog/focus defects using existing component tokens; add forced stale-snapshot approval and keyboard coverage at 170x48/80x24 in both themes.
3. Run targeted import/trust/UI/token/bundle checks. Verify final native TldwCli in a private profile with synthetic Skills, exact persisted contents/trust state and normal exit.
4. Update guide, audit, QA and task notes, obtain bounded review and commit locally. No full suite, provider execution or dev integration.

Allocation: fresh fetch; all-ref and live-worktree max 32654; candidate 32655 absent from all-ref content sweep. Evidence: .superpowers/sdd/2026-09-15-skills-import-trust/task-id-sweep.json.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed Skills import and exact trust review: token-backed dialogs, compact action rows, selected-Skill Import with dirty veto, guarded keyboard returns, durable import receipts and fresh retained Items after approval. Items refreshes independently of Work, preserving newer drafts.

355 distinct targeted checks pass. Two legacy backdrop failures were repaired by loading production app CSS in their modal harnesses; all 13 modal contracts pass. Wiring/token/bundle/controller checks pass; the two inherited governance failures remain unchanged (22 CSS allowlist offenders and LibraryScreen size 35,202 vs 33,204, one line smaller than HEAD). No new Ruff diagnostics; new files/Work pane are formatted. No full suite.

Private native run-005 passes 170x48 dark and 80x24 light, including setup errors, candidate Cancel/selection, stale rejection, fresh approval and natural focus. Twelve rendered captures inspected. Exact hashes, absent unselected packages, fresh locked-to-trusted service reopen, ten SQLite integrity checks and zero messages verified. Normal quit returned exit 0; owned shell closed. This is not full-app restart or provider/script-execution qualification.

Updated Skills guide, workflow audit, QA evidence and the retained-Items/delayed-event testing lesson. Production changes are in the Skills controller/screen, Work/list widgets and source CSS with regenerated bundles; targeted harnesses and journey tests document the regressions. Independent review found no remaining issue.

ADR required: no; existing backlog/decisions/009-local-skill-trust-boundary.md, 076-library-lifecycle-progressive-disclosure.md, 086-library-adaptive-reader-shell.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md apply. No ownership, trust or storage boundary changed.

Evidence: Docs/superpowers/qa/2026-09-15-skills-import-trust/README.md. Next: Skills Files/supporting-file interactions. Dev integration remains pending.
<!-- SECTION:NOTES:END -->
