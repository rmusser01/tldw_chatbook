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
**Status:** In Progress — fresh first-note capture and restored Console Open/normal Quit passed. Actual interruption → normal restart → retry passed with all 1,800 research files and the saved note verified. Required safety-copy selection, saved Library handoff and keyboard focus fixes passed independent review. Installed default replacement/later rollback passed (328.90s), but actual keyboard confirmation exposed an absent incoming UI preferences file in the default config container; publication repair now has independent approval; its immutable keyboard repeat is in progress. Config companion and first user-data binding corrections passed independent review (24 and 76 checks respectively). Ordinary app construction, mounting and saved-note readback passed after Abort, replacement and later rollback in the source-only workflow; its mixed-source caveat requires a final immutable repeat. Runtime config-sibling writes and duplicate sidebar persistence correction passed independent review (85 checks); installed Console capture/readback/resumed writes passed in 47.96s. The bound writer → Complete capture → writer regression is fixed: 15 independent native checks and 48 existing capture/SQLite checks passed, preserving requested capture contents while holding every related config group. Default config-file publication, authenticated later retirement and activation-pair recovery passed independent review (51 checks); the full native default lifecycle passed 22 cases. A separate config-only missing-user-directory Open correction passed 29 native and six independent checks. Final installed and platform acceptance remains pending. Legitimate restored-profile settings writes still invalidate its fingerprint. Linux passed 243 focused checks but default replacement entered automatic reversal before its subprocess deadline; exact finalization error is under diagnosis. Windows latest support run has 11 failures; bounded test diagnostics and the ACL fixture correction have been reviewed, native repeat pending. Specific approvals for the path-inventory comparison entries, two-file Linux diagnostic transfer, and fingerprint-only settings-continuity repair remain pending; those rejected changes are unapplied.

Stage 5 current immutable keyboard checkpoint (`ab24f6b0`): clean credential-review Abort, normal Console/Library startup, exact original-note readback and normal Quit passed. Retrying replacement afterward refuses `target_unverified`; the default profile binding has a stale fingerprint and no activation. Independent causal review continues. No replacement/later-rollback pass is claimed for this artifact.

Independent Stage 5 causal review confirmed automatic Console `console.rail_state` persistence → stale unactivated binding → `mcp_recovery_binding_changed` → `target_unverified`. Two native default-binding RED tests reproduce the same write invalidation. The existing fingerprint-repair approval must cover both activated and unactivated existing bindings; no new roots or activation changes are proposed.


Stage 5 approved execution: requester authorized all three previously blocked actions. Configuration-owner binding continuity (default and restored) is implemented and independently approved, including native descriptor retirement and a two-process concurrent-profile regression. Focused39 and consolidated profile-open27 passed; inventories pass; no new Bandit findings. The two exact Linux diagnostic files are uploaded with verified hashes; preserved-build default replacement passed with2,865 installed files unchanged. Full Linux workflow, current installed keyboard repeat and fresh Windows run remain pending; task stays In Progress.
