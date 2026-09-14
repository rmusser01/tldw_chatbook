# Backup UAT remediation — TASK-32562

Scope: correct the failures documented in `Docs/Development/backup-uat-2026-09-13.md` and the preceding backup review. Retain Python and existing archive/recovery protections on macOS, Linux, and Windows. No new backup modes or unrelated application work.

ADR required: no
ADR path: N/A
Reason: Correct observed defects within the approved backup design and existing ownership, publication, and user-review contracts.

## Stage 1: Live capture and ordinary configuration files
**Goal:** Capture the normal running application, including a profile created by onboarding.
**Success Criteria:** The known configuration lock is classified safely; an actual running app can finish a Complete backup and resume ordinary writes.
**Tests:** Reproduce the UAT failure with test-only diagnostics; add regression tests for the demonstrated cause and exact lock classification; rerun live installed capture.
**Status:** Complete — exact retained Home notification worker boundary corrected and reviewed. Fresh immutable merged tree f69a11ecb0720db043d9e89616ff46dc789f8bad passed the six-step keyboard onboarding → first note → Complete capture → archived note readback → resumed live write journey. Linux lifetime/capture checks passed on the same verified tree. Windows affected repeats remain Stage 5.

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
**Status:** In Progress — reviewed product corrections are pushed in `230073f527b66d4528b41aa48bfc44473eaa6c59`; the PR is mergeable against dev and remains unmerged. All three requester approvals are applied. Config-write continuity, missing preferences publication/retirement, binding/capture lifetimes, destination guidance and filename review have regression and independent review evidence. Fresh immutable keyboard UAT (`f41682c6`) passed onboarding, saved-note handoff, explicit safety selection, credential-review Abort, normal Console/Library restart and original-note readback. Reviewed replacement retry now advances beyond the earlier fingerprint refusal; replacement completion/later rollback remain pending. Exact-commit Linux passed full default replacement/later rollback and first-note capture/handoff; final mounted capture cases are running. Windows34810093007 passed plain, encrypted, encrypted-credentials and roundtrip; replacement/rollback selections fail at the same ordinary-reopen45s child deadline after successful Abort. A reviewed test-only observer adds fixed phases and bounded stacks while preserving that deadline. Support and final diagnostic repeats remain pending.

Stage 5 current immutable keyboard checkpoint (`ab24f6b0`): clean credential-review Abort, normal Console/Library startup, exact original-note readback and normal Quit passed. Retrying replacement afterward refuses `target_unverified`; the default profile binding has a stale fingerprint and no activation. Independent causal review continues. No replacement/later-rollback pass is claimed for this artifact.

Independent Stage 5 causal review confirmed automatic Console `console.rail_state` persistence → stale unactivated binding → `mcp_recovery_binding_changed` → `target_unverified`. Two native default-binding RED tests reproduce the same write invalidation. The existing fingerprint-repair approval must cover both activated and unactivated existing bindings; no new roots or activation changes are proposed.


Stage 5 approved execution: requester authorized all three previously blocked actions. Configuration-owner binding continuity (default and restored) is implemented and independently approved, including native descriptor retirement and a two-process concurrent-profile regression. Focused39 and consolidated profile-open27 passed; inventories pass; no new Bandit findings. The two exact Linux diagnostic files are uploaded with verified hashes; preserved-build default replacement passed with2,865 installed files unchanged. Full Linux workflow, current installed keyboard repeat and fresh Windows run remain pending; task stays In Progress.


Stage 5 Windows ordinary-reopen diagnostic: independent review approved SHA `50e14d609c077557beb7044dfcf61f84e4db695b52ce72078380d09e0e121c76`. Short checks verify separate checkpoint logs, error preservation, metadata-only output and collection. Ruff passes; Bandit has the same eight existing test-scope findings as HEAD. Fresh full native workflow verification is running before the diagnostic commit.

Stage5 diagnostic verification complete: ordinary-reopen observer passed fresh explicit replacement/Abort/reopen (165.12s); Notes-retirement observer passed all three installed mounted capture routes (156.02s), six independent cases and native transaction-preservation probe. No product operation, close, namespace, assertion or timeout changed. Current Linux completed251pass/1Console admission timeout; actual immutable keyboard replacement/reopen/new-note/quit passed, but later rollback review failed before publication and is under read-only diagnosis.

Stage 5 access-time correction: native reproduction proved the later-review failure followed a config binding refresh rejected by its own read-induced atime update at Quit. Stable identity comparisons now exclude only atime and retain exact content and authority verification. Independent review approved; 61 binding/activation and 26 pair/default lifecycle cases pass, Bandit zero product findings. Fresh immutable keyboard replacement/Quit/later rollback and affected native platform checks remain pending. The latest Linux Console diagnostic passed once; the prior intermittent timeout remains under review.

Stage 5 subsequent verification: Notes initialization cleanup and three Windows fixture/diagnostic corrections are independently reviewed and tested; see the remediation report for exact passing and baseline-failing scopes. Linux 84cf passed 90/91; captured metadata identifies a one-shot cache-retirement race after a worker finishes. Its bounded same-predicate retry is independently design-approved and being tested. Fresh keyboard replacement is at safety-copy review; later rollback and final native platform acceptance remain unfinished.

Stage 5 2026-09-14: Same-pause cache revisit is independently approved; deterministic native regressions and all three installed macOS capture routes pass (153.90s, preserved fixture). The Windows replacement observer is adjusted only for its proven constructor progress, to the existing 135s recovered-open allowance; enclosing F9 and product deadlines are retained. Native Linux/Windows repeats and actual keyboard later rollback remain pending.
