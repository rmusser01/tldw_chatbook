---
id: TASK-31243
title: Add trusted character navigation recovery and Roleplay browse
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 02:08'
updated_date: '2026-09-06 02:35'
labels:
  - console
  - roleplay
  - library
  - navigation
dependencies:
  - TASK-31242
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the first complete local character-conversation navigation slice: draft-safe departure, typed exact Console activation, Library-owned unavailable-link repair, and full Roleplay browse/search/preview.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-31235 on 2026-09-04. The final pre-commit worktree sweep
found the older `Sort chooser renders every option` task created at 01:50; it
keeps TASK-31235 under the older-arrival rule. This unshipped task moves with
all plan and dependency references.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every caller uses one typed cancellable activation contract and Console changes destination only after commit, with rollback preserving the prior tab on failure.
- [ ] #2 Escape cancels only before commit; duplicate activation cannot open duplicate sessions; success is returned only after the exact destination is visible.
- [ ] #3 Roleplay navigation captures all incumbent card, Persona visual, attachment, and in-flight-save drafts and requires Save and continue, Discard and continue, or Stay.
- [ ] #4 Roleplay provides local per-character keyset browse, Keyword search, read-only preview, exact resume, Back to Console, and stable focus in the approved 52x20 progression.
- [ ] #5 Library accepts a typed repair context, shows historical evidence and same-authority candidates, requires explicit confirmation, and performs compare-and-set repair.
- [ ] #6 Repair failure preserves source data and focuses Refresh; success invalidates indexes and restores the requested return anchor.
- [ ] #7 Existing Roleplay card editing, imports, exports, visual and attachment workflows, and transcript-to-Console draft remain unchanged.
- [ ] #8 Targeted race, focus, compact-layout, keyboard, pointer, draft-loss, exact-resume, and unavailable-recovery tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md
Reason: Implements existing activation, draft-veto, repair and surface ownership contracts.
1. Start TASK-31243 after Task 2 merges; capture targeted incumbent baseline.
2. Write failing navigation-payload validation tests.
3. Implement versioned payload module and navigation keys.
4. Write failing activation state-machine tests.
5. Implement Console-owned activation coordinator.
6. Write failing aggregate Roleplay draft-veto tests.
7. Implement app-owned pre-navigation coordinator.
8. Write failing Library repair interaction tests.
9. Implement Library-owned repair presentation.
10. Write failing Roleplay browse and compact-flow tests.
11. Extend incumbent Roleplay controller and widgets.
12. Build CSS and run targeted Task 3 gate.
13. Perform isolated real-TUI verification and commit.
Delivery adaptation: reuse seven original Task3 commits through 653b9e3c0 plus owning later fixes; preserve merged Task2 schema/APIs and newer dev ownership. New adaptations require fresh RED/GREEN evidence. Task3 Console bodies belong in workspace/modules. Native evidence remains pending; Pilot is compositor evidence only. No full-suite runs or remote actions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Packaged the approved Task3 implementation on merged Task2 (2b4973971e5dcf101c5a6ddcc55aa082ff22f814): typed exact activation with cached-runtime rollback/token ownership, aggregate Roleplay draft veto, complete three-field keyset browse, retained Keyword snapshot-time copy, and Library-owned confirmed CAS repair with bounded candidate continuation. Preserved owning late b157/8a3/c1 fixes without Task4/5 consumers. New Console bodies live in workspace with named late-bound wiring; screen changes remain lifecycle/state glue. ADR required: no; existing backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md governs. Fresh adaptation RED/GREEN and current targeted evidence are documented in Docs/QA/task-31243/README.md and local task-3-report.md. Main selected gate:569passed1fixturefailure; corrected core68passed, resource fixture10passed with zero registered worker connections, final compact/capture9passed, queue-veto7passed. Focused Ruff/format and CSS reproducibility pass; inherited screen ratchet remains red and was not raised. One bounded compact-preview paint correction preserves full labels/body and pointer Back focus. Native isolated-terminal walkthrough remains unavailable; do not mark Done or claim all acceptance evidence complete until controller review and missing platform evidence are addressed.

Final six-capture controller confirmation: full Back/Send/Open labels and visible transcript are resolved at52x20; a right-edge In inspector-area fragment remains beside Send. This is a known compact visual concern, not a visual pass. No further polish pass was started; independent review will triage it alongside inherited ratchet deltas. Generated SVG whitespace-only blank lines were normalized for git diff --check without changing rendered content.

Fix round1: I1 clears selected identity and revokes/reset confirmation on blank or changed candidate; I2 restores wide center and removes narrow inline sizes; I3 sizes late Back to Console outside breakpoint caching; M1 hit-test traced orphaned In to personas-inspector-rail-open and synchronizes rails after center-view commit. Fresh real-dialog/resize/real-DB deep-link/paint RED-GREEN recorded locally. Seven original SVGs regenerated, independent re-review pending. I4 attribution correction: Library41393/1321 at merged base grew to41494/1325 in Task3, so its ratchet failure was NEW, not inherited. Controller authorized extraction only of directly extended apply_navigation_context plus new repair orchestration/state into lightweight LibraryNavigationController; named media/collections callables resolve late and shared admission/generation/monkeypatch seams remain. Final41356/1321 restores unchanged41393/1321 budget; owning25tests and navigation14tests pass, full ratchet retains only2pre-existing Console failures. Behavior and extraction are separate commits. Carryover attribution correction: cached Console/reuse/ordered gate/fresh-candidate protections belong to8a3f3478, notb157b1d; b157 owns NoMatches visibility/boundary catches/first-use Library repair. Native/live timing and aggregate FD warning attribution remain unverified. No Done claim.

Latest-dev qualification: both scoped code/UI re-reviews passed all five findings; rebased the three reviewed commits without conflicts onto frozen dev 3ff76c89e0ffc3fc947be9cdba31d1379f911e52, with three patch-equivalent range-diff entries and safety ref codex/task-31243-pre-rebase-eed0bd582 preserved. Qualified runtime HEAD b574082c35ad3f953ede7c18d5d147dd0696a1cd. Fresh serial gates:79 functional,9 focused Personas,14 Library admission,10 wizard overlap and6 startup checks pass. Current startup gates are GREEN, superseding historical inherited-startup wording: dev-to-feature ui_ready969->970/972, CSS786624->787554/804000 bytes, preimport498->498/500 modules and366177->367182/378740LOC, fattest Library112023->111986/123319LOC. Library41356/1321 remains below unchanged41393/1321 cap. Console22615->22643/698 contributes28 new lines while its two threshold failures pre-exist; no cap raised. Focused Ruff/format, shared-diagnostic comparison, CSS sync and task-ID checks pass. Controller approved derived inventory refresh after reviewing all six drifting owners and20 Task3-added literal-message calls:3 TASK492 and17 TASK494; no direct payload/path/URL interpolation or new sink/path metadata. Seventeen calls retain exception tracebacks under unchanged incumbent logging/redaction policy; inventory registration is not new privacy clearance. Requests/utcnow and upstream wizard harness unawaited-coroutine warning remain; no out-of-scope cleanup. Native/platform/human/live timing and historical aggregateFD attribution remain pending. No all-programme/Done claim; all AC boxes remain open. Full exact commands/results/provenance are appended to local task-3-report.md.

Derived-artifact closeout: regenerated inventory guard reports no drift (582 owners,1339 TASK492,30 TASK31551,7634 TASK494,12 sink files); exact JSON comparison confirms only six reviewed owner rows and three summary counters changed, with716 path candidates and all sink/schema/classification metadata unchanged. Three focused digest/multiplicity/chained-logger checker tests pass. Runtime source and reviewed captures are unchanged by this artifact/notes commit.

External Qodo correction round 2 from published 0e9e797c42f740047b2beda3803c09936f0bd760: addressed all eight verified comments plus exact CI CSS275>274 finding. Strict lazy private Pydantic models validate both wire shapes/nested identities before domain construction; routing uses Constants and Returns docs are complete. Discard restores saved/new Character and Persona forms across cached-screen return without persistence. Exact links fence authority/identity/revision before selection and after page rendering, retain ownership through later-page focus, expose stale/missing recovery, and refresh only the same identity on explicit Retry; new search abandons the old seek. Fast compact completion reveals and scrolls the exact row after layout. Terminal Library profile mismatch clears repair ownership; precommit cancellation returns the domain target. Ten existing diagnostic statements add bounded ephemeral tokens/literal context only; reviewed artifact diff changes exactly three owner digests, no counts/sinks/path metadata or traceback-policy changes. Fresh gates:101 functional passed,32 focused Personas passed; final compact/race12 and last-change browse/form/startup20 passed. Startup remains GREEN at970/972 modules,498/500 preimport modules,367336/378740LOC,787546/804000CSS bytes; bare-selector census274. Library41356/1321 and Console22643/698 unchanged from FIX_BASE; only two inherited Console ratchet failures remain. Focused Ruff/format, zero-new shared Ruff comparison, CSS sync, inventory guard and3 digest tests pass. Original seven SVGs are historical; no new native claim. User-authorized ignored native packet remains HOLD and un-repinned pending stable-head review; no native app or remote operations in this round. ADR required:no; existing ADR120 governs. Full RED/GREEN commands, invalid fixture attempts, hashes and limits are in local task-3-qodo-fix-2-report.md. Task stays In Progress with AC boxes open for controller review/native evidence.
<!-- SECTION:NOTES:END -->
