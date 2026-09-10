---
id: TASK-31941
title: Qualify and admit Canvas V2 Mermaid
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 22:15'
updated_date: '2026-09-09 05:41'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31940
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - >-
    Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
  - Docs/superpowers/reviews/2026-09-08-canvas-v2-mermaid-acceptance-closeout.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the first immutable diagram profile only after end-to-end security and usability qualification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mandatory real Chromium zero-egress and containment gates include diagram attacks, combined budgets, useful mixed documents and positive controls.
- [x] #2 Native and same-origin served workflows, two-browser isolation, archive recovery, restart revocation, packaging and reproducibility have fresh recorded evidence.
- [x] #3 Only a fully qualified immutable profile is admitted; failures leave V2 unavailable without changing V1 limits, and user and operator documentation report actual scope and coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (existing ADR applies)
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
Reason: Qualification and admission of the exact approved immutable profile under ADR124/ADR121, no new authority.
1. Follow Task8 of Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md and plan-scoped task8 integration notes.
2. Add missing release gates for adversarial/quota behavior, real process restart and recovery, exact examples, packaging/reproducibility and actual CI browser collection.
3. Run targeted Canvas native/served/browser/lifecycle/archive/static checks, inspect required visual fixtures, record exact scope and limits in Docs/Canvas/V2_VERIFICATION.md.
4. Only after required gates pass, observe production-admission RED then freeze/admit exact candidate and rerun final targeted checks; otherwise keep candidate disabled and report failing design gate.
5. Update docs and backlog evidence, commit and obtain independent task review; whole-branch review follows.
6. Address final review I1 under existing ADR124 item6: preserve ordinary startup and authenticated source-only recovery when the complete packaged snapshot is unavailable, retain strict integrity rejection and parent/child fencing, observe targeted regression RED/GREEN, then obtain one scoped fix review. Existing warning cleanup stays separately scoped.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current: the exact Mermaid profile is locally qualified and admitted. Task review,
whole-branch review, and the scoped recovery-fix review are complete, with no open
Critical or Important findings. Existing M1 warning cleanup remains separately
disclosed. The earlier blocked gates and prerequisite handoff below are historical
checkpoints, not the current qualification outcome.

### Historical qualification — 2026-09-07

Release gate BLOCKED; task remains In Progress. Corrected candidate selection passed 1350 tests with 2 optional browser skips (578.29s), but final admitted selection failed 4 tests (1346 passed, 2 skips, 539.80s). Restored V2 executable=false and default_diagram_profile=null; no admission claim. Snapshot-refusal fixture now uses a deliberately distinct fixed test-only revoked policy. Bounded diagnostics captured owned Python child SIGBUS in SQLite WAL recovery/frame lookup and Console trace-maintenance SQL; cause is not proven environmental or Canvas-specific, and earlier untraced failures are not retroactively attributed. No shared DB/security/native dependency fix was attempted. Final disabled/profile/offline reproduction selection: 77 passed, 1 existing dependency warning, no skips, 5.85s. Packaging/source closure, CI Chromium installation, immutable identity/policy reproduction, actual same-origin durable-source restart with parent/all children stopped, explicit V1 recovery, real unsent repair, bounded resources and useful browser fixtures are recorded in Docs/Canvas/V2_VERIFICATION.md. ADR124 clarifies catalog-only policy and bounded trusted selection reconciliation; manifest metadata-only identity is 17717bcab7c7bba4a28e0069354f6ecbf895d2ca58f4b8d1c0355b7726e2f466, with executable/library/V1 bytes unchanged. AC2 remains open until final release workflow failures are resolved. Independent review and separately authorized SQLite concurrency investigation are required.

### Prerequisite handoff — 2026-09-08

The SQLite prerequisite is now Done with reviewed acceptance closeout at Docs/superpowers/reviews/2026-09-08-sqlite-acceptance-closeout.md. The missing paired actual Canvas measurement passed five samples per arm and independent fix-only review; full workflow median36.834s current vs34.186s baseline (+7.75%), with stated limits. This only permits resuming the existing Task8 qualification: AC2 remains open, V2 disabled, prior admitted1346-pass/4-fail run stays failed. No candidate/admitted suite or V2 admission was performed by SQLite closeout. Before reproducibility runs, revalidate pinned input availability; the earlier runtime archive cache was empty on preflight, not passing evidence. No PR/push/rebase/merge action here.

### Fresh qualification — 2026-09-08

2026-09-08 continuation: fresh five-file Chromium candidate gate passed 176 tests with 2 optional engine skips (475.87s). Complete Canvas candidate/admitted selections, including the 19 CI workflow contracts, both passed 1383 tests with 2 optional Firefox/WebKit skips (661.78s/767.12s); admitted run had only the existing RequestsDependencyWarning. Required actual-child create/update, confirmed unsent repair, parent/all-child restart revocation, two-browser isolation, Chatbook archives, twice-rebuilt offline assets and wheel/sdist closure passed. Two packaged admission assertions were observed RED before the four-field catalog-only change and GREEN afterward. Exact V2 manifest and all executable/library/V1 bytes and limits remain unchanged; admitted policy is cd4f0cdd756732e686b05031ce12c6bd086473cc72ff2f9d58340d8528b40f15. Two touched test files pass Ruff/format after baseline-proven mechanical lint cleanup. Root inspected all 16 useful candidate screenshots and recorded bounded resource measurements and coverage limitations in Docs/Canvas/V2_VERIFICATION.md. Evidence: /private/tmp/mermaid-qualification.lga1Y7, including complete invocations/logs/JUnit and phase snapshots. ADR124/ADR121 still govern; no new ADR. Earlier failed admission remains failed historical evidence. AC2 is now evidenced, but task remains In Progress pending independent task/whole-branch review and final closeout. No PR/push/rebase/merge or evidence cleanup.

Final whole-branch review identified I1: an unavailable packaged profile snapshot propagates through parent, child and native Console startup, preventing ADR124 source-only recovery. The strict loader is correct; application-owner recovery needs a bounded correction before closeout. Task remains In Progress. Prior passing qualification and admitted immutable bytes remain recorded; no PR or merge.

Final review I1 correction implemented under ADR124 item6: the strict profile loader is unchanged; native Console, served-parent and served-child owners retain one inert source-free snapshot on packaged integrity failure. This preserves ordinary startup and authenticated source/history/download while denying HTML mutations and all runtime delivery, with matched unavailable control state and healthy/unavailable mismatch rejection. Fresh frozen covering gate: 1400 passed, 2 optional engine skips, 1 existing warning, 712.32s, exit0. Final test-only picker/lineage fixture correction passed all 17 focused cases, 6.07s, exit0; product bytes remained identical to the covering gate. New/small files and changed legacy logical ranges pass scoped static checks with zero new diagnostics over the recorded baseline. Evidence /private/tmp/mermaid-recovery.MikmXD; recovery compatibility docs and testing lesson updated. Independent scoped fix review remains before Done. No PR, push, merge or cleanup.

### Reviewed closeout — 2026-09-08

Completed TASK-31941 under ADR124/ADR121. Admission745811a335 and recovery7464fe0c6c are independently reviewed; scoped review marks I1 addressed with no new breakage. Final covering gate1400pass/2optionalengine skips/1existingwarning and final17focusedpass are documented with exact unchanged runtime bytes, static scope, intermediate failures, and evidence hashes. All acceptance criteria are checked. The testing lesson records why strict-loader tests alone missed startup recovery. Acceptance closeout: Docs/superpowers/reviews/2026-09-08-canvas-v2-mermaid-acceptance-closeout.md. No new ADR, runtime privileges, dependencies, schema or immutable asset changes in the correction. M1 and existing lint debt remain explicit non-blocking limitations. Branch and evidence preserved; PR/rebase/push/merge remain a separate integration choice.
<!-- SECTION:NOTES:END -->
