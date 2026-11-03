---
id: TASK-38534
title: Design prompt mix-ins for Console sends
status: Done
created_date: 2026-11-01 17:47
assignee:
- '@codex'
labels:
- design
- prompts
- console
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- backlog/decisions/286-console-prompt-mix-ins.md
updated_date: 2026-11-01 19:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define reusable, independently toggled prompt modifiers that users manage through the existing prompt library and composer Prompts button, with predictable per-chat selection, current-send-only instructions, and honest retry and cache behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A linked design spec records the approved library type, literal Before/After bodies, existing composer Prompts entry point, per-chat toggles and ordering, and empty selections for new chats.
- [x] #2 The spec defines library edits, reopen, queue capture, retry-original versus regenerate-current, operational snapshot retention, and current-send-only history semantics.
- [x] #3 A linked ADR records artifact/source compatibility, conversation ownership, request preparation, exact retry snapshots independent of diagnostic capture, and stable-prefix caching trade-offs.
- [x] #4 The documents define failure recovery, targeted automated and live verification, v1 exclusions, and affected subsystem owners without claiming runtime implementation.
- [x] #5 Self-review and scoped documentation checks pass, and the user reviews the written spec before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/286-console-prompt-mix-ins.md
Reason: Adds a prompt artifact contract and durable chat/attempt ownership, changes request preparation and caching, and defines source compatibility and privacy boundaries. Extends ADR-040 and ADR-057 while preserving ADR-097, ADR-098 and ADR-210.

1. Read the existing prompt artifact/source contracts, Console request/queue ownership, retry storage and provider caching behavior.
2. Write the approved design spec and ADR, with explicit v1 scope, lifecycle and failure rules, integration owners and verification scenarios.
3. Self-review for contradictions, placeholders and scope; validate local links, task metadata, identifiers and scoped whitespace.
4. Commit only these design artifacts and present the written spec for user review. Product code, implementation planning, runtime tests, push and merge remain outside this design task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote and committed the prompt mix-in design spec and ADR-286, with an ADR index entry and scoped source-backed verification/failure requirements. The user approved both written documents on 2026-11-01 and authorized implementation planning. The spec/ADR now record Accepted status and link the implementation plan.

ADR required: yes; backlog/decisions/286-console-prompt-mix-ins.md. Scoped document links, placeholder/whitespace checks and Backlog metadata/identity checks passed. All design-task acceptance criteria are complete. Planning proceeds separately under TASK-38535; no product implementation, migrations or runtime/provider/cache qualification was performed by this design task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
