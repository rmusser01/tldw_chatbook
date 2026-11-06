---
id: TASK-36762
title: Admit phone-owned Console input and explicit agent commands
status: Done
created_date: 2026-10-25 17:39
dependencies:
- TASK-36759
- TASK-36760
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
updated_date: 2026-10-25 21:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let phone users send text and correct exact existing main agents or subagents without borrowing the desktop composer or changing its active chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Phone Send uses immutable command-owned text through native per-session admission, checked preparation and queue custody while preserving desktop drafts, attachments, one-shot prefill and staged RAG.
- [x] #2 Desktop typing, view changes or another device's edits alone do not reject or consume phone input; real session, settings, binding, permission and admission changes still refuse stale work.
- [x] #3 Main Send, Steer, Redirect and Stop address exact session incarnations and applicable runs; existing active-tab wrappers use the same native implementation.
- [x] #4 Subagent guidance and Stop remain scoped to the exact conversation/handle/run, distinguish queued from consumed guidance and refuse finished or replaced targets without relaunching them.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Direct implementation of approved command-owned input and exact native targeting contracts.

1. Add failing independent-phone-composition and native preparation/custody race tests using existing store and received-intent fixtures.
2. Extend the existing input snapshot origin, native witnesses and claim/preparation path without transferring desktop composition or introducing another admission owner.
3. Extract explicit-session main controls with exact turn/run checks; retain active wrappers and conversation-owned subagent guidance/cancellation semantics.
4. Add the runtime command adapter using Task2 identities and Task1 execution authority; preserve native hooks/permissions/configuration/currentness/custody.
5. Run specified targeted input/custody/dispatch/control/subagent checks and scoped static/format verification, update living docs, commit explicit paths and obtain independent spec+quality review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented command-owned text on the existing received-intent/claim/preparation/queue path, retaining original settings/context/binding witnesses through queued promotion and preserving desktop composition and staged evidence. Added ConsoleRemoteCommands with explicit host/session-incarnation/run/turn/handle targets; main active wrappers share native controls, and subagent commands reuse native mailboxes/cancellation/approval revocation. The live primary control witness checks the actual service mailbox rather than last-bound run metadata. Updated Docs/Architecture/console.md and agent-runtime.md. ADR required: no; directly implements ADR-244 (backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md). Targeted verification: 265 passed with one independently reproduced inherited closed-session node deselected, plus 2 native worker-completion/config-source race cases passed. Earlier connected bridge/Stop/queue batch: 432 passed; all eight then-failing run-state nodes reproduced unchanged at exact BASE96d6c88669 before the documented bootstrap_profile fixture correction. The remaining closed-session handoff failure reproduces identically on BASE with the same marker ('Draft custody is unconfirmed.'); it is outside this task and unchanged. Scoped AST14, full-default Ruff per-file comparison (no added diagnostics), new-file format, existing-file format_ratchet vs exact BASE, and diff checks pass. Full suite and served/browser integration were not run. Controller owns independent review, final AC and Done. Detailed evidence: .superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-4-report.md.
Independent Task4 spec and quality review approved commit45304b13fb with no blocking findings. Verified native command origin/currentness/custody, retained queued witnesses, active-wrapper behavior, exact live-primary/subagent ownership and honest pending/queued/accepted results. Final265 passes +2 completion-race passes, no warnings;14 Python paths parse with zero introduced per-file Ruff diagnostics, formatting ratchet and diff checks pass. One independently reproduced inherited closed-session handoff node remains red on exact BASE with same documented bootstrap_profile marker; final selection deselects only that node. No new regression was attributed to it. Controller timing/scope rulings preserve async configuration capture after native claim/Preparing custody and allow two existing queue-owner files to retain origin/witnesses. ADR-244 applies. Later Tasks5/6/9/11 own projection/decision/receipt/cutover/served qualification; Task4 does not claim public mobile integration.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
