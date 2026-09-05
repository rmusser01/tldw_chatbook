---
id: TASK-31741
title: Complete Canvas integration artifact guards
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 20:02'
updated_date: '2026-09-05 21:32'
labels:
  - canvas
  - testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Canvas merge candidate satisfy repository artifact guards with reviewed diagnostic metadata and measured SQLite index coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every changed diagnostic inventory owner is reviewed for content or credential disclosure and the generated inventory reproduces.
- [x] #2 All six Canvas indexes are classified in the census with real no-statistics query-plan evidence for read indexes and explicit constraint rationale for uniqueness enforcers.
- [x] #3 Targeted checks and repository preflight pass on the latest-dev merge candidate; independent review and implementation notes document evidence.
- [x] #4 Canvas startup integration stays within existing app-import, first-paint module, CSS, and pre-import payload ratchets without raising thresholds or hiding new residents.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md. Reason: verify existing schema and privacy contracts, no new indexes or runtime privileges.
1. Review each diagnostic statement drift against the pinned inventory revision, including exception traceback and user-controlled argument paths; add failing privacy regression and bounded correction for any newly confirmed disclosure before regenerating metadata.
2. Add isolated real-SQLite tests asserting sqlite_stat1 absent and representative Canvas queries use the four query indexes; document the two composite unique foreign-key target constraints as correctness enforcers in the census. Reuse existing repository fixtures; do not add or remove indexes to appease the guard.
3. Run targeted tests and artifact generators, inspect inventory/census diff, and repeat preflight. Root owns executable checks; agents stay static-only.
4. Rebase latest dev with recoverable branch state, resolve only scoped conflicts, rerun affected checks and preflight, obtain independent static review and record commands/limitations. Complete before opening the authorized PR.
5. Post-rebase fixture reconciliation: preserve the required selection-generation contract by completing the three served compiler snapshot doubles, and remove duplicate prefill/generation-settings imports retained during conflict resolution while keeping dev's endpoint-policy imports. Loopback route failures under the restricted executor are environmental: the unchanged gateway module passes all72 tests with local-listener permission; do not alter network markers or guards to hide them. Run the corrected affected selection with that permission and retain the initial results honestly.
6. Complete the existing bare TldwCli shutdown fake with the current meeting-session ownership slot exposed by the rebased lifecycle test; preserve the shutdown ordering assertion and production lifecycle. This is fixture compatibility, not a new fallback or a weakened ownership invariant.
7. Startup ratchet repair (existing ADR-097 plus ADR-121, no new architecture or threshold increase): compare isolated untouched-dev and feature first-paint module censuses; retain the real app/store Canvas lifecycle owner, defer unused browser/control implementation imports to served admission or the first actual Canvas publication/open, and preserve view rebinding, settlement-driven auto-open, kill-switch revocation, and shutdown. Add first-use/no-use regression controls before implementation, then measure rather than assume the reduction. If necessary, defer existing Chatbook archive import/export implementation edges to their existing first-use entry points while preserving public exports and service ownership; do not invent a replacement Canvas owner or relax validation. Root alone runs isolated pytest/browser checks. Independently review the bounded changes and rerun all four startup guards plus affected lifecycle, tool, archive, and browser tests before PR creation.
8. Measured startup follow-up: Group A saves six modules and Group B saves five browser modules plus one first-tick resident (978 still exceeds972). Add discriminating tests before sharing the strict enabled-flag normalization between the execution-only getter and full Settings policy; the getter must not construct remote-access policy, but all malformed values remain fail-closed and Settings still validates admission. Defer compiler implementation imports at all remaining controller/service/exception-only UI parents together, preserving callable patch seams, injected compilers, and exact exception types. Keep concrete controller/service/compilation owners. The CI CSS census also reports276 broad ancestor rules vs274: re-key the two new Canvas card/recovery control subjects with exact classes, retain declarations/appearance, regenerate the widget bundle, and prove matching controls retain computed styles. No limit, snapshot, scheduler safety or interface privilege change.
9. Measured Group C first paint is975, with compiler/web_auth deferred and one safety first-tick resident absent only by timing. Under existing ADR-097, shed equivalent first-paint work by moving SideChat, ReviewNotes and PromptQueue dialog imports to their actual user-open handlers/callbacks, and ConflictResolution to local Chatbook import. Preserve concrete class/enum identity, callbacks, singleton/dirty-edit protections and import validation; do not defer into compose/mount or add timing workarounds. Record negative import closure and existing real first-open behavior tests before implementation, then measure actual census and rerun affected controls. Keep ScopePicker unchanged because alternate eager parents defeat a single-edge deferral.
10. Group D positive-control reconciliation: untouched dev reproduces the same SideChat timeout and two ReviewNotes workspace-registry refusals (3 failures). Complete the offline gateway fake with the current route keyword contract and provide the existing real app workspace registry to the two isolated persistence fixtures. Preserve actual sends, real SQLite assertions and workspace validation; establish green positives plus the two expected closure failures before changing production imports. Local Fast Lane process-lease tests also reproduce all six SemLock errno28 failures on untouched dev; record this host verification limit, do not modify OS resources or test gates, and require protected CI success before merge.
11. Post-open dev rebase66a1cbf8f includes Stop-dispatch drain and interaction import deferrals; preserve both plus all Canvas contribution semantics. Only manual rebase conflict is additive lessons documentation. Two restore-lock fixture writes reproduce semantic-mutation guard failures on untouched dev; authorize only the exact assistant message_update within each out-of-band test setup transaction using the existing private test-fixture authorization pattern, leaving production guards and measured reconcile transactions untouched. Add verification that setup authority is gone before reconciliation, retain read/write-lock assertions, then rerun complete affected module and new Stop/startup controls. Preserve published-tip recovery ref and update remote only with an explicit expected-head force-with-lease.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented reviewed diagnostic metadata, six Canvas index classifications and real no-statistics query-plan coverage; corrected four new Canvas traceback disclosures with fixed diagnostics. Restored startup ratchets through first-use archive/browser/compiler/dialog imports and exact CSS subjects, preserving concrete ownership, first-create delivery, strict policy and compiler contracts. Reconciled current dev fixture contracts without weakening production gates. Existing ADR-121 and ADR-097 apply; no new architecture, thresholds or snapshots. Latest base d9d5763d6: all116 commits range-diff equal, recovery ref retained; exact candidate970c86da4 preflight passes, startup+first-use36passed, prior affected1517passed, browser89passed (optional Firefox/WebKit unavailable), compiler/service139passed, finaldialog32passed, latency9passed. New/tiny owned-file Ruff/format passes; legacy diagnostics unchanged or reduced. Local PR Fast Lane749passed/6SemLock errno28 failures reproduced on untouched dev; no OS changes, protected CI still required. Evidence and warning qualifications: Docs/Canvas/V1_VERIFICATION.md. All scoped changes independently reviewed with PASS. Remains In Progress until protected integration checks complete; no full-suite claim. Modified Canvas/Console/config/Chatbook tests and owners, CSS bundle, diagnostic/index inventories, evidence and task documentation. Lesson added for baseline-verified OS allocation failures.
<!-- SECTION:NOTES:END -->
