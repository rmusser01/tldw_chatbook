---
id: TASK-32824
title: Close PR2707 with current dev integration and final visual review
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-18 21:53'
updated_date: '2026-09-18 23:22'
labels:
  - ui
  - design-system
  - integration
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the existing PR at its approved scope boundary, verify current dev integration and accumulated findings, and present exact visual/conflict evidence for merge approval while preserving remaining reviews for follow-up PRs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unfinished inspector work is safely preserved outside PR2707 and remaining destination reviews are explicitly deferred without becoming closeout blockers.
- [ ] #2 Current dev is integrated with an exact conflict-resolution record; targeted affected checks and current-head CI are reviewed with any remaining failures identified.
- [x] #3 A concrete final visual and conflict review identifies the proposed merge result; PR remains unmerged until this review receives user approval.
- [x] #4 Qodo review findings are resolved with bounded complete-catalog Persona navigation, retained selected identities, documented public contracts and targeted evidence before the authorized merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/079-workspace-assistant-defaults.md; presentation ADR-150/161. Reason: integration and review corrections preserve existing service, identity, consent and UI contracts. 1. Preserve TASK32823 separately and integrate latest dev. 2. Record exact rebase conflict decisions and verify the rebased approved tree. 3. Address all Qodo findings: document the nine public API groups; replace unbounded workspace Persona options with finite previous/next pages using existing service offsets, pin saved or staged identity, and materialize only the requested service page. 4. Verify page boundaries, catalog reach, saved/pending identity, failed listing and memory consent with real-service regressions; run focused native dark/light compact/wide journeys, scoped lint and required generated checks. 5. Push the fixes, respond with evidence, obtain current-head Qodo review and required CI, then merge under the user approval granted on 2026-09-18. 6. Resume preserved inspector work on a fresh branch from merged dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved TASK32823 separately at 135f226888. Rebased onto latest dev53b56384cc; retained both testing lessons and recomputed combined diagnostic counts, then restored merge-only composer tests. Rebased tree654e39956b18 matched the visually approved tree exactly. Addressed all ten Qodo findings: nine documented API groups are executable-AST equivalent; both Persona pickers use finite pages with selected identity retained, and the service normalizes only the requested page. Native QA caught clipped compact actions; added visible-region regression, constrained the form and kept errors/actions outside its scroller.59 distinct targeted cases pass, including final22 UI/governance checks. Seven preflight guards and scoped lint pass; generated CSS reverified after final layout correction. Twenty final native captures inspected across dark/light and compact/wide, clean exit0,11 healthy private databases, matching source hashes and unchanged defaults. Independent review found no blocker. ADR required:no, existing ADR079/139/150/161 apply. QA: Docs/superpowers/qa/2026-09-18-pr-2707-qodo/README.md. User approved merge; final pushed-head Qodo/CI qualification remains pending.
<!-- SECTION:NOTES:END -->
