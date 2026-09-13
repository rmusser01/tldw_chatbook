# Backup UAT remediation — TASK-32562

Scope: correct the failures documented in `Docs/Development/backup-uat-2026-09-13.md` and the preceding backup review. Retain Python and existing archive/recovery protections on macOS, Linux, and Windows. No new backup modes or unrelated application work.

ADR required: no
ADR path: N/A
Reason: Correct observed defects within the approved backup design and existing ownership, publication, and user-review contracts.

## Stage 1: Live capture and ordinary configuration files
**Goal:** Capture the normal running application, including a profile created by onboarding.
**Success Criteria:** The known configuration lock is classified safely; an actual running app can finish a Complete backup and resume ordinary writes.
**Tests:** Reproduce the UAT failure with test-only diagnostics; add regression tests for the demonstrated cause and exact lock classification; rerun live installed capture.
**Status:** Complete — native mounted Console, Settings and Library capture and independent scoped review passed. Actual Library follow-up corrected its installed retained-state contract and released newly acquired native caches at measured finite worker boundaries. Persona acceptance remains Stage 5.

## Stage 2: Valid, understandable restore destinations
**Goal:** Let users select a new local location per profile and explicit locations for independent external content; derive required shared/owner paths from installed policy.
**Success Criteria:** A normal profile restores without entering hashed internal roots; the reviewed destinations pass owner relocation checks and open with the saved note. Existing explicit low-level restore API remains checked.
**Tests:** Real archive fixtures, shared DB and owner path layout, multiple profiles, custom paths, and GUI keyboard workflow.
**Status:** Complete — derived destinations, earlier refusal, isolated restore/open, and native replacement passed; final review also corrected empty generated-image roots and preserved unused optional stores. Native replacement and later rollback passed after those corrections; independent review approved.

## Stage 3: Review and failure guidance
**Goal:** Explain coverage, credential requirements, extraction contents, and failures using user-facing descriptions; correct F9 documentation.
**Success Criteria:** Acknowledged Partial coverage is accurately shown; users can identify nonempty files for recovery; known errors give corrective actions without exposing arbitrary exception text.
**Tests:** Relevant UI/service behavior and nonempty extraction; inspect actual terminal output.
**Status:** Complete — bounded guidance, coverage and extraction labels verified by focused UI tests; terminal acceptance remains Stage 5.

## Stage 4: Large archives and earlier review findings
**Goal:** Keep valid larger backups restorable within explicit bounded descriptor/journal limits; correct the stale directory metadata assertion.
**Success Criteria:** The 1,800-file collection completes recovery; oversized untrusted records remain rejected; archive metadata tests pass.
**Tests:** Boundary and large-file-count staging/publication regression, journal recovery checks, archive writer/reader tests.
**Status:** Complete — bounded records and actual 1,800-file publication passed independent review; platform repeats remain Stage 5.

## Stage 5: Native verification and persona acceptance
**Goal:** Rebuild the installed artifact, rerun affected native checks and the failed/blocked user journeys, and publish evidence on PR #2642.
**Success Criteria:** First-time and power-user backup/restore, encrypted credentials, nonempty extraction, opening restored data, replacement/later rollback, cancellation and interruption recovery have explicit results. Run Linux over the authorized SSH host and Windows on the authorized GitHub Actions runner. Record remaining limitations honestly.
**Tests:** Scoped pytest, Ruff and Bandit, required repository checks, independent code review, actual keyboard UAT and native platform workflows.
**Status:** In Progress
