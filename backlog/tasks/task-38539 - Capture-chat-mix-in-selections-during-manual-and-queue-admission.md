---
id: TASK-38539
title: Capture chat mix-in selections during manual and queue admission
status: Done
created_date: 2026-11-01 19:30
dependencies:
- TASK-38537
- TASK-38538
labels:
- prompts
- console
references:
- backlog/decisions/286-console-prompt-mix-ins.md
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md
- backlog/decisions/286-console-prompt-mix-ins.md
assignee:
- '@codex'
updated_date: 2026-11-01 22:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every human-submitted message retain the mix-in choices and definitions accepted for its owning chat despite asynchronous preparation or later library edits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chat-owned add/toggle/remove/reorder operations preserve disabled positions and exact-origin revisions; placement edits move an entry to its new group's end without changing other order.
- [x] #2 Manual and queued inputs freeze selected references before awaited preparation and resolve coherent literal definitions before acceptance; queued drain never reloads captured bodies.
- [x] #3 Missing enabled entries, stale source/session/revisions and unreadable definitions refuse without losing drafts; disabled unavailable entries do not block; machine follow-ups do not newly resolve chat-selected mix-ins.
- [x] #4 Selection save failure appears in the existing settings recovery surface with exact-session, exact-revision retry, without a new manager or timer.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused placement and held-admission regressions.
2. Implement chat-owned live preference CAS, bounded settings persistence and exact-source capture/retry.
3. Carry frozen snapshots through configuration, received preparation, custody and queue; qualify races and failure recovery.
4. Update Console architecture and self-review targeted evidence.
ADR required: no
ADR path: backlog/decisions/286-console-prompt-mix-ins.md
Reason: Direct implementation of approved ADR-286; preserve existing runtime and persistence boundaries.
5. Per controller ruling, extend the existing left-rail settings recovery pattern with a failure-only PROMPT_MIXINS row and exact revision retry; no normal selection controls. Read design language/component patterns and qualify existing recovery-row tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented chat-owned selection CAS with exact session/incarnation/binding fences, placement reconciliation, current-source coherent capture and strict original-retry lookup. Frozen values flow through received/direct configuration and queue custody; queue text edits retain their accepted mix-ins, machine follow-ups receive empty values. Reused app settings durability admission and the existing coalesced component drain/retry surface; added failure-only recovery row and off-loop preference hydration plus atomic temporary promotion seed.

ADR: direct implementation of backlog/decisions/286-console-prompt-mix-ins.md; no new ADR/schema/dependency. Source authority uses the selected local Prompt DB lexical path digest plus exact backend/DB fences; server capture remains unsupported. Already-complete inputs remain authoritative, including empty snapshots. Composition, dispatch attempt contributions and normal Workbench controls remain later slices.

Evidence: final focused 34 passed; affected context/direct/received custody 101 passed; source/settings/hydration/UI governance 285 passed; extra duplicate-receipt race 1 passed. Earlier combined operational-storage run: 119 passed plus three fixture-anchor failures subsequently corrected and six native replay cases passed; it retained an aggregate FD-growth warning also present in prior slice evidence. Ruff exact-BASE differential: 667 baseline / 666 current, zero introduced diagnostics; new/extended mix-in files lint/format pass; diff whitespace clean. Full command/result details: .superpowers/sdd/2026-11-01-prompt-mix-ins-implementation/task-4-report.md.

Introduced direct-call attachment regression was fixed by carrying only the new snapshot/witness until the original configuration-resolution point; focused attachment and full context owner pass. Status remains In Progress for independent review; no full sweep, push or merge.
Review fix I1 (BASE 32df824beb): retained the service's private exact source-currentness guard across uncustodied direct Queue/manual hooks, later preparation/provider waits, pending pause/resume, and the owning-loop durable handoff drain. Guards never enter frozen/serialized configuration; supplied complete and accepted envelopes remain authoritative. Stale COMMITTING work uses the existing pause-to-abandon cleanup, retaining draft/attachments and failing optimistic echoes. Evidence: native source replacement RED 8 failed; final I1/compatibility proof 15 passed; final cleanup 4 passed. Connected context/native-commit/compaction run had 141 passed plus one new-test assertion expecting an exception instead of the existing continuation refusal result; corrected assertion and both cases passed. Exact fix-BASE Ruff: 182/182 diagnostics, zero introduced; focused owner lint/format and diff checks pass. Details/evidence appended to task-4-report.md. Task stays In Progress for independent re-review.
Independent review cleared after a0c2c4bc32 retained exact source/session/preference witnesses through uncustodied direct hook/config/provider/handoff and pause/resume waits. Barrier regressions qualify refusal cleanup and preserve drafts/attachments. Accepted queue and supplied complete snapshots stay authoritative; no library recapture. Targeted evidence and qualified aggregate FD warning remain above. ADR-286 governs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
