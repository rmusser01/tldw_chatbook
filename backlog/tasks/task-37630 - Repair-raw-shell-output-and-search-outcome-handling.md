---
id: TASK-37630
title: Repair raw shell output and search outcome handling
status: Done
assignee:
  - '@codex'
created_date: 2026-10-29 22:34
updated_date: '2026-10-31T18:37:00Z'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep raw shell error details available to models, let empty searches complete without exhausting retries, and preserve Windows command text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shell results retain stderr and all executor-provided output beyond 4000 characters.
- [x] #2 Unambiguous no-match grep, rg and findstr calls do not trigger the ordinary failure streak; genuine and ambiguous command failures still do.
- [x] #3 Windows cmd commands retain quoting and preserve intended output and exit codes.
- [x] #4 Focused regression tests, lint, documentation and ADR review are complete.
- [x] #5 The new CLI regressions run automatically in the existing Linux Python 3.12 and 3.14 PR workflow.
- [x] #6 Cancellation settlement does not access a worker-owned synchronization lock after confirmed worker termination, while late launch stays refused.
- [x] #7 The PR satisfies the existing UI-ready module census and source-size ratchets without increasing their limits; first-use tool construction remains functional.
- [x] #8 The Console rename dialog public export and Rename action preserve behavior while package/session-controller imports defer its implementation until first use.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/254-cli-search-outcomes-and-listing-compatibility.md. Reason: refine search outcomes and optional listing worker arguments while retaining ADR-078/094/101 authority. Follow Docs/superpowers/plans/2026-10-29-cli-tool-recovery.md with focused red/green tests, implementation, documentation, targeted checks and fresh review.
PR routine follow-up: add a fail-first CI routing contract, select the new regression modules in bounded separate invocations of the existing Tool Recovery job, retain the two-version matrix and timeout, and verify the contract before publication. ADR required: no new ADR; existing ADR-254 and CI execution structure apply.

Current-head review follow-up: reproduce PowerShell backslash/quote parsing with simple full-path searches and compound failure controls; add failing classifier and real-shell regressions, preserve backslashes for PowerShell without broadening recognized command forms, run scoped verification, and have the reviewer verify the fix. ADR required: no new ADR; implements the existing ADR-254 unambiguous-command rule.

CI ferry follow-up: investigate the Python 3.14 cancellation-test timeout and UI-ready module-census failure independently. Replace unsafe inspection of a multiprocessing event after worker termination with an observation of real parent-side launch signaling; retain the actual worker and cancellation behavior. Attribute the census delta without raising ADR-097 ratchets. Verify scoped controls and obtain review before publication. ADR required: no new ADR for the test observation repair; ADR-254 and ADR-097 apply.

Local cancellation diagnostics also exposed the executor finally block re-signaling an abort event after worker death. Add a deterministic post-cleanup signal guard to the real-worker regression, observe RED, then skip abort signaling for a worker already proven dead and retain all bounded tree cleanup.

Boot census follow-up: base/head source overlays reproduce identical native module sets; background character reads explain Linux timing variation. Defer the controller-only WatchlistsToolService import to its existing first-use local-provider composition site, preserving service/root/kill-switch checks and source line count. Verify fail-before/pass-after census plus actual provider composition controls; use existing ADR-097, no new ADR.

Verification rejected the controller import move: it shifted two modules into the screen-preimport payload (560/558). Revert that candidate. Instead defer ConsoleRenameSessionModal through the Console package existing lazy-export facade, preserving __all__ and first Rename behavior. Confirm existing census, preimport, first-send, lazy export and real Rename controls with all limits unchanged. This is an ADR-097 paydown using an established facade, no new ADR.

Full boot verification found a second eager public import in UI/Console_Modules/session.py. Extend the chosen lazy-modal repair by relocating that one import to _open_console_session_rename_modal. Extend the subprocess regression to import the session controller before checking cold implementation absence, observe RED, then verify actual census and Rename persistence separately from profile-sensitive mixed collections. Preserve controller line count and all existing budget limits; ADR-097 applies.

The existing real Rename persistence test also reproduces raw_source_selection_changed alone before its action. Enroll only that case in the repository established private_profile_test harness so it exercises real persistence under an admitted isolated profile; retain its UI and database assertions. This is a test fixture repair, with no production authority/config gate changes.

Select the new cold package/session Rename export regression in the existing Perf Guard boot-ratchet step and trigger that workflow for changes to the regression itself. Preserve the Python version, permissions, concurrency and timeout; verify existing workflow safety contracts. This keeps the new import-boundary check exercised on PRs.

Fast Lane found only generated remote worker growth: 5730 lines versus 5706. Preserve the generator and ratchet. Reduce duplicated canonical write-source logic: combine optional open flags with equivalent getattr defaults, share complete descriptor reads using the stdlib sentinel iterator, and unify successful temp cleanup bookkeeping. Add multi-chunk preview/CAS coverage, regenerate the artifact, and verify size, real local writes, native confinement and bundled worker parity. Select existing bundle ceiling/slack checks in the focused Linux workflow so this gate runs early. ADR required: no new ADR; behavior-preserving implementation under ADR-254 and existing bundle contract.

Direct merge review: reproduce the real cmd caret-escaped quote compound before changing production; add pure/native empty-search and failure controls; conservatively reject cmd caret syntax, preserving other shells and execution text; run focused recovery/provider/integration checks and fresh exact-head review. Existing ADR-254 applies; no new ADR or CI-policy change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the provider-only 4000-character slice while retaining executor/storage bounds. Simple complete grep/rg/findstr no-match outcomes no longer consume the failure streak; diagnostics, compounds, truncated previews and ambiguous commands remain failures. Native cmd text is preserved. The user-selected requesting-code-review replacement for Qodo found PowerShell quote/backslash issues; eight cases failed before the fix, then 33 recovery and 88 provider/integration checks passed. The reviewer confirmed the fix and closed-stdin native controls.

CI follow-up repaired cancellation settlement: the regression observes actual parent signals instead of reading an Event after worker death, and executor finalization no longer re-signals a dead worker. A bounded fault guard reproduced the unsafe call; all 46 native process checks passed with 7 POSIX skips. Independent review accepted the repair and independently passed 10 controls.

The inherited boot census was paid down using the Console package's existing lazy Rename export and a handler-local session import. Public class identity, real Rename behavior, session size and all thresholds remain unchanged. Seven budget/export/switcher/size checks passed (1031/1031 ready modules, 558/558 preimport, 93/94 first-send), two current real SQLite Rename/changed-profile controls passed, and five workflow-safety checks passed. The new import regression is selected in Perf Guard; Linux tool regressions remain selected in the bounded Python 3.12/3.14 workflow. Fresh review found no remaining issues. Scoped lint adds no findings; inherited ownership E721 checks remain unchanged.

The controller-import candidate was reverted after it moved cost into preimport. Tests/UI/test_console_rename_persistence.py::test_rename_modal_persists_saved_conversation_title remains unchanged: its profile admission and stale fake persistence-owner route failed; authoritative test_console_rename_consistency.py controls passed. No thresholds or assertions were weakened. No full suite was run. GitHub checks must still pass on the published head.

Modified raw shell projection/execution, native and parser regression fixtures, the Console lazy-export/session import boundary, CI selection, architecture docs and testing lessons. Existing ADR-254 and ADR-097 govern the changes; no new ADR is required. Tasks 37631/37632 retain virtual listing and Windows write evidence. PR #3314 is the integration target.

Fast Lane artifact follow-up: the unchanged worker size guard failed at 5730/5706. Canonical write-source deduplication restored exactly 5706 lines with builder/budget unchanged. Multi-chunk preview/CAS, native write/listing/loopback, write/edit/locking, size and workflow checks pass; bare Python 3.10.5 source/artifact execution is verified. Full POSIX bundle contracts are explicitly selected in Linux CI. Fresh independent review found no findings and independently passed the size/routing and multi-chunk controls. No full suite or budget override was used; publication still requires fresh GitHub checks.

The user-requested fresh full-PR review found one cmd caret-escape ambiguity: findstr with caret-escaped quotes executed COMPOUND_RAN and an explicit exit1, but shlex hid its compound operators and projected success. Two pure regressions and the real cmd/provider reproduction failed first (3failed6passed controls). The classifier now conservatively rejects cmd caret syntax alongside percent expansion. A plain cmd no-match and POSIX caret-regex controls remain successful. All93 focused recovery/provider/integration checks passed; scoped formatter/lint passed. No execution text, shell launch, permissions, bounds, builder, worker artifact, CI workflow or guard changed. ADR-254 documents the existing conservative boundary. Fresh exact-head review gates the requested direct merge attempt. Evidence: cmd-caret-review-red.txt and cmd-caret-review-green.txt. No full suite or hosted-workflow dispatch.
<!-- SECTION:NOTES:END -->
