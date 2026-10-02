---
id: TASK-33645
title: Preserve evaluation case sensitivity in typed API requests
status: In Progress
created_date: 2026-10-01 20:11
references:
- https://github.com/rmusser01/tldw_server/pull/2979
updated_date: 2026-10-01 20:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Typed evaluation create and update requests discard case_sensitive, so resaving a case-sensitive server evaluation can reset its behavior. Preserve the server setting in the shared client schema after server PR2979 landed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Explicit true and false survive typed create and update JSON serialization
- [x] #2 An omitted setting retains the server-compatible false default and partial updates preserve omission
- [x] #3 Existing evaluation schema and client contracts remain passing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no; ADR path: N/A; reason: routine missing-field parity fix within the existing evaluation API boundary. Stage1: verify fresh main and original tests, then add failing flag round-trip/default controls. Stage2: add the smallest shared EvaluationSpec field and verify targeted schema/client tests, lint and Bandit. Stage3: independent review, normal commit and companion PR linked to server UAT523. Plan document: Docs/superpowers/plans/2026-10-01-evaluation-case-sensitive-followup.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fresh main b0dadf original schema/client baseline16passes. Test-first controls produce5 missing-key failures and7passes; adding one shared bool field with false default produces20passes, zero skips, natural exit0 in0.99s. Explicit true/false create and response-to-update round trips, default false and exclude_unset omission verified. All17 original assertion ASTs retained; four new assertions. Ruff candidate/baseline each155 inherited UP045 notices in production and zero test findings, no new notices; both files format clean and compile. Bandit actualexit1 from21LOW B101 test assertions, zero errors/HIGH/MEDIUM and zero production findings. ADR not required: routine parity fix in existing API contract. Independent review/publication in progress; keep all task criteria open until companion PR is linked. No full sweep or macOS investigation. Source patch /private/tmp/pr2979-chatbook-2004-reviewed-source.patch; static proof /private/tmp/pr2979-chatbook-2004-static-proof.json.
Independent scoped review CLEAR: no actionable P1/P2. Reviewer verified exact immutable patch/source hashes, create and response-to-update true/false preservation, false default and partial omission, and retained red/green artifacts. Reviewer did not rerun tests or perform broad/native/server/PG qualification. Publication pending; criteria and task remain open until companion linkage and hosted disposition.
PR #2950 review follow-up: add explicit true/false payload assertions to the existing public-client CRUD test, including reconstruction from the parsed response before update. Qodo server-side field claim is contradicted by landed server commit 88f8b81: EvaluationSpec already declares case_sensitive and retained server schema/runner controls cover it. Preserve all original CRUD assertions, prove the added boundary checks fail without the field, then rerun only the two evaluation schema/client modules and scoped static checks. Production fix remains the same single field.
Final reviewed candidate for already-published companion PR https://github.com/rmusser01/tldw_chatbook/pull/2950: all three task criteria verified. Existing public client CRUD test now proves explicit true/false create and response-to-update JSON. Exact old schema causal red2fail/7deselected; actual restored candidate21passes/zero skips/errors/failures/natural0. All131 original schema/client assertions preserved; six added. Compile/format clean; Ruff155 production UP045 and one client I001 unchanged from exact base, zero new. Bandit137LOW B101 test assertions/zero errors/production findings/actual1; six new asserts, no suppression. Independent follow-up review CLEAR/no actionableP1/P2. Qodo server-field claim disproved by exact landed server88f8b81 schema and existing schema/runner controls. Evidence /private/tmp/pr2979-chatbook-2027-boundary-*. Companion merge not yet claimed; status remains In Progress until verified landing.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
