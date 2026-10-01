---
id: TASK-33645
title: Preserve evaluation case sensitivity in typed API requests
status: In Progress
created_date: 2026-10-01 20:11
references:
- https://github.com/rmusser01/tldw_server/pull/2979
updated_date: 2026-10-01 20:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Typed evaluation create and update requests discard case_sensitive, so resaving a case-sensitive server evaluation can reset its behavior. Preserve the server setting in the shared client schema after server PR2979 landed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Explicit true and false survive typed create and update JSON serialization
- [ ] #2 An omitted setting retains the server-compatible false default and partial updates preserve omission
- [ ] #3 Existing evaluation schema and client contracts remain passing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no; ADR path: N/A; reason: routine missing-field parity fix within the existing evaluation API boundary. Stage1: verify fresh main and original tests, then add failing flag round-trip/default controls. Stage2: add the smallest shared EvaluationSpec field and verify targeted schema/client tests, lint and Bandit. Stage3: independent review, normal commit and companion PR linked to server UAT523. Plan document: Docs/superpowers/plans/2026-10-01-evaluation-case-sensitive-followup.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fresh main b0dadf original schema/client baseline16passes. Test-first controls produce5 missing-key failures and7passes; adding one shared bool field with false default produces20passes, zero skips, natural exit0 in0.99s. Explicit true/false create and response-to-update round trips, default false and exclude_unset omission verified. All17 original assertion ASTs retained; four new assertions. Ruff candidate/baseline each155 inherited UP045 notices in production and zero test findings, no new notices; both files format clean and compile. Bandit actualexit1 from21LOW B101 test assertions, zero errors/HIGH/MEDIUM and zero production findings. ADR not required: routine parity fix in existing API contract. Independent review/publication in progress; keep all task criteria open until companion PR is linked. No full sweep or macOS investigation. Source patch /private/tmp/pr2979-chatbook-2004-reviewed-source.patch; static proof /private/tmp/pr2979-chatbook-2004-static-proof.json.
Independent scoped review CLEAR: no actionable P1/P2. Reviewer verified exact immutable patch/source hashes, create and response-to-update true/false preservation, false default and partial omission, and retained red/green artifacts. Reviewer did not rerun tests or perform broad/native/server/PG qualification. Publication pending; criteria and task remain open until companion linkage and hosted disposition.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
