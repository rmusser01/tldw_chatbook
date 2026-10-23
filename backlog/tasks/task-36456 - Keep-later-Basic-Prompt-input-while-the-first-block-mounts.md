---
id: TASK-36456
title: Keep later Basic Prompt input while the first block mounts
status: In Progress
created_date: 2026-10-23 16:39
labels:
- textual
- book-audit
priority: medium
references:
- backlog/tasks/task-19024 - Simplify-Library-Prompt-editing.md
- backlog/tasks/task-1775 - Deliver-unified-Prompt-Recipe-and-Library-user-interfaces.md
- backlog/tasks/task-32603 - Review-Library-Prompt-save-and-browse-continuity.md
- tldw_chatbook/Widgets/Library/library_prompts_canvas.py
- tldw_chatbook/Widgets/Prompts/prompt_block_editor.py
documentation:
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
- backlog/decisions/076-library-lifecycle-progressive-disclosure.md
- Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md
assignee:
- '@codex'
updated_date: 2026-10-23 19:09
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first edit into an empty Basic Prompt lane awaits a real block mount, then projects its older captured text back into the still-enabled editor. Actual keyboard input accepted during that wait can disappear from both the live TextArea and canonical blocks. Keep the incremental editor coherent with its newest accepted text.

Priority: Medium (P2), rank 3 of five in this audit. See the report for impact and ordering.

Observed evidence: Real first User-lane block reconciliation/mount completes but delivery is held at the existing await; actual keyboard b reaches the same enabled child (a -> ab), then older completion restores a in live and canonical text. No-new-key control retains a. User lane measured; System/cross-lane/clear remain regression requirements. Controlled interleaving, not natural race-frequency measurement.

Existing scope: TASK-19024/TASK-1775 own the existing incremental Basic/Advanced composition, and TASK-32603 covers ordinary Save/browse continuity. This failure occurs before Save at the first empty-lane reconciliation. Keep its scope independent from pending Save acknowledgement.

Related tasks: [TASK-19024](task-19024%20-%20Simplify-Library-Prompt-editing.md), [TASK-1775](task-1775%20-%20Deliver-unified-Prompt-Recipe-and-Library-user-interfaces.md), [TASK-32603](task-32603%20-%20Review-Library-Prompt-save-and-browse-continuity.md).

ADR check: no new ADR expected for a bounded repair within existing owners and contracts. Applicable decisions: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md), [040-versioned-prompt-artifacts-and-safe-improvement-transactions](../decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md), [076-library-lifecycle-progressive-disclosure](../decisions/076-library-lifecycle-progressive-disclosure.md). Re-evaluate if implementation changes a boundary.

[Textual book audit: evidence, controls, limits and overlap review](../../Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md). The audit records the pre-fix behavior; implementation evidence is recorded below.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Basic System and User input accepted while their first block mounts survives the earlier reconciliation, including a later clear and edits across either lane.
- [ ] #2 Canonical blocks and Basic/Advanced projections agree with the newest live text while preserving block identity, metadata and order, with no duplicate block.
- [ ] #3 Ordinary first and subsequent edits continue to work without replacing the full stateful editor or changing persistence/provider boundaries.
- [ ] #4 Mounted controls cover no-later-input and queued later-key interleavings around the original structural reconciliation, witnessing delivery to the live child before releasing the earlier handler.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce mounted first-block reconciliation loss in both Basic lanes with actual child delivery, clear, and cross-lane controls.
2. Reconcile newest live Basic text after structural awaits using the existing stable block identities; preserve metadata and incremental widgets.
3. Verify canonical/Basic/Advanced agreement, unchanged controls, stable identities and subsequent edits with targeted regressions.
ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md; backlog/decisions/076-library-lifecycle-progressive-disclosure.md
Reason: bounded stale projection repair inside the existing Prompt canvas; no storage or service boundary changes.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Basic first-block reconciliation now reads both live TextAreas after structural awaits and reconciles the newest accepted text on the existing block identities. Later keys, clear and cross-lane input remain consistent in canonical and Advanced state; the full stateful editor is retained.

Eight mounted controls first produced two ordinary passes and six exact loss failures, then all eight passed after the repair and on the final relevant source. Subsequent editing preserves child/block identity, metadata and order. See Docs/superpowers/qa/2026-10-23-textual-pending-draft-followups.md. Existing ADR-033 application state and ADR-040/076 govern this bounded repair; no new ADR. Independent review and hosted CI remain pending.
<!-- SECTION:NOTES:END -->
