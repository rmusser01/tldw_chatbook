---
id: TASK-36454
title: Keep newer Library Prompt drafts when an earlier Save completes
status: In Progress
created_date: 2026-10-23 16:33
labels:
- textual
- book-audit
priority: medium
references:
- backlog/tasks/task-32603 - Review-Library-Prompt-save-and-browse-continuity.md
- backlog/tasks/task-19024 - Simplify-Library-Prompt-editing.md
- backlog/tasks/task-1775 - Deliver-unified-Prompt-Recipe-and-Library-user-interfaces.md
- backlog/tasks/task-32630 - Review-Library-Prompt-retained-history-journeys.md
- backlog/tasks/task-2702 - Library-unsaved-prompt-silently-blocks-screen-navigation.md
- backlog/tasks/task-32393 - Library-Prompts-Escape-on-a-dirty-prompt-editor-does-nothing-at-all.md
- backlog/tasks/task-32461 - Library-Prompts-three-more-dirty-vetoes-refuse-without-a-word.md
- backlog/tasks/task-32646 - Review-Library-Skills-browsing-and-editor-journeys.md
- tldw_chatbook/UI/Library_Modules/library_prompts_controller.py
- tldw_chatbook/Widgets/Library/library_prompts_canvas.py
documentation:
- backlog/decisions/033-application-session-state-ownership.md
- backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
- backlog/decisions/049-local-prompt-retained-version-history.md
- backlog/decisions/076-library-lifecycle-progressive-disclosure.md
- backlog/decisions/086-library-adaptive-reader-shell.md
- Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md
assignee:
- '@codex'
updated_date: 2026-10-23 19:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library Prompt Save can falsely mark a later edit as saved. A real SQLite update stored the submitted body, then completion cleared dirty state and restored older canonical blocks while newer text remained visible. Back left without a warning and reopening lost that text. Preserve accepted edits while acknowledging the actual committed record.

Priority: Medium (P2), rank 1 of five in this audit. See the report for impact and ordering.

Observed evidence: Submitted body -> accepted keyboard x -> visible Submitted bodyx; persisted v2 and canonical blocks remained Submitted body; dirty false, Back exits, reopen loses x. First-create acknowledgement also fails; its separate Back/reopen continuation had a harness settlement failure and is excluded. Advanced/metadata/next Save/quit are future regression requirements.

Existing scope: TASK-32603 covers ordinary first-save continuity; TASK-19024/TASK-1775 own the incremental Basic/Advanced editor. Existing dirty-veto tasks depend on the predicate this acknowledgement clears. TASK-32646 protects Skills, a different owner; both Skills create controls passed. No new draft store, history or service boundary is proposed.

Related tasks: [TASK-32603](task-32603%20-%20Review-Library-Prompt-save-and-browse-continuity.md), [TASK-19024](task-19024%20-%20Simplify-Library-Prompt-editing.md), [TASK-1775](task-1775%20-%20Deliver-unified-Prompt-Recipe-and-Library-user-interfaces.md), [TASK-32630](task-32630%20-%20Review-Library-Prompt-retained-history-journeys.md), [TASK-2702](task-2702%20-%20Library-unsaved-prompt-silently-blocks-screen-navigation.md), [TASK-32393](task-32393%20-%20Library-Prompts-Escape-on-a-dirty-prompt-editor-does-nothing-at-all.md), [TASK-32461](task-32461%20-%20Library-Prompts-three-more-dirty-vetoes-refuse-without-a-word.md), [TASK-32646](task-32646%20-%20Review-Library-Skills-browsing-and-editor-journeys.md).

ADR check: no new ADR expected for a bounded repair within existing owners and contracts. Applicable decisions: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md), [040-versioned-prompt-artifacts-and-safe-improvement-transactions](../decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md), [049-local-prompt-retained-version-history](../decisions/049-local-prompt-retained-version-history.md), [076-library-lifecycle-progressive-disclosure](../decisions/076-library-lifecycle-progressive-disclosure.md), [086-library-adaptive-reader-shell](../decisions/086-library-adaptive-reader-shell.md). Re-evaluate if implementation changes a boundary.

[Textual book audit: evidence, controls, limits and overlap review](../../Docs/superpowers/qa/2026-10-23-textual-book-after-pr3219.md). The audit records the pre-fix behavior; implementation evidence is recorded below.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create and update Save acknowledge only the submitted snapshot and adopt the actual persisted identity/version; later raw metadata and canonical block edits remain visible and dirty.
- [ ] #2 The next explicit Save writes the retained draft to the same prompt using its current version. An unchanged successful Save becomes clean; definite failure preserves current edits for retry.
- [ ] #3 Basic and Advanced projections agree after acknowledgement, including live child input already accepted while its Changed notification is queued. Existing Back, navigation and quit protections detect the retained draft.
- [ ] #4 Mounted real-store regressions distinguish ordinary create/update, later accepted input, queued notification, definite failure, next Save and guarded exit; actual child input delivery and committed rows are witnessed.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce create/update acknowledgement with mounted real-store tests, including later Basic/Advanced/raw metadata edits, queued Changed, definite failure and next Save.
2. Capture submitted and current live drafts at the existing UI owner. Adopt actual saved identity/version and retain newer blocks/raw controls, dirty/action/history projections and stable children.
3. Verify current-version retry and Back/navigation/quit guards; run targeted editor/service/governance checks, update documentation and obtain Superpowers review.

ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md; backlog/decisions/049-local-prompt-retained-version-history.md; backlog/decisions/076-library-lifecycle-progressive-disclosure.md; backlog/decisions/086-library-adaptive-reader-shell.md
Reason: bounded acknowledgement repair inside existing editor/persistence/history owners; no boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Prompt Save now captures live Basic/Advanced blocks and raw metadata, adopts the actual persisted identity/version, and retains later edits as a dirty working copy. The existing first-save capability refresh keeps the next Save on the same row; old queued semantic projections are rejected. Completion moved to UI/Library_Modules/prompt_save_ack.py to lower the controller ceiling to5,100. Child editors and existing history/guard/service boundaries stay in place.

Mounted real-SQLite regressions cover create/update, later and queued input, definite failure, second Save, guarded Back/navigation/quit flush and a held older semantic message. See Docs/superpowers/qa/2026-10-23-textual-pending-draft-followups.md for RED/GREEN receipts and fixture qualifications. Existing ADR-033 application state and ADR-040/049/076/086 apply; no new ADR. Independent Superpowers review and hosted CI remain pending.
<!-- SECTION:NOTES:END -->
