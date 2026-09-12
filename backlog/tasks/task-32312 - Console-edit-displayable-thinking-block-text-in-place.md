---
id: TASK-32312
title: 'Console: edit displayable thinking block text in place'
status: Done
assignee:
  - '@robert'
created_date: '2026-09-11'
updated_date: '2026-09-11 03:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need to correct or clean up a model's visible thinking text (typos, sensitive fragments, export hygiene) without discarding the whole generation. ADR-090 currently clears thinking entirely when the answer content is edited; this task adds a block-scoped in-place edit for displayable thinking blocks. Amends ADR-090: edited blocks keep identity/provenance, stay replay-eligible in their original encoding under existing serialization safety checks, and content-edit-clears-thinking behavior is unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Edit action appears on selected displayable thinking rows (never proprietary) and opens a modal prefilled with the block text
- [x] #2 Save persists edited text via the atomic generation projection: content untouched, no descendant purge, variant envelope and message envelope both updated
- [x] #3 Blank edits and unsafe start-anchored text (`<think`/`</think`) rejected with clear messages; envelope bounds re-validated (256 KiB block, 1 MiB envelope)
- [x] #4 Edits refused while streaming/pending, for opaque or quarantined generations, and for non-assistant owners
- [x] #5 Edited blocks remain replay-eligible under existing policy with existing safety checks (auto skips unsafe, include fails before send)
- [x] #6 Existing content-edit-clears-thinking behavior unchanged (its test stays green)
- [x] #7 ADR-090 amendment records user-editable block text semantics
- [x] #8 User docs updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline targeted tests (thinking persistence/blocks/history/presentation, edit modal, message actions) — 272 passed
2. Write ADR-090 amendment
3. TDD store method: update_message_thinking_block with envelope revalidation, dual-home write, atomic persist, guards
4. TDD ConsoleEditThinkingModal + transcript thinking-row edit action + screen wiring
5. Extend thinking-history tests with edited-text serialization cases
6. Update user docs; run targeted tests; task close-out

ADR required: yes
ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md (amendment)
Reason: amends ADR-090's accepted "assistant edit clears thinking" boundary by adding block-scoped user edits of displayable thinking text with replay-eligibility semantics.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented block-scoped displayable thinking edit per the ADR-090 amendment: store update_message_thinking_block (guards + envelope revalidation + atomic generation projection + variant dual-home), ConsoleEditThinkingModal, keyboard e seam on thinking rows via ConsoleThinkingEditRequested event (ChatScreen handler -> ConsoleMessageController), proprietary rows gated. +23 targeted tests, zero regressions vs clean dev (35 failures pre-exist on this machine).
<!-- SECTION:NOTES:END -->
