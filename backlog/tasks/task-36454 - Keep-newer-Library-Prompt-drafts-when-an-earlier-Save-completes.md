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
updated_date: 2026-10-23 20:36
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
- [x] #1 Create and update Save acknowledge only the submitted snapshot and adopt the actual persisted identity/version; later raw metadata and canonical block edits remain visible and dirty.
- [x] #2 The next explicit Save writes the retained draft to the same prompt using its current version. An unchanged successful Save becomes clean; definite failure preserves current edits for retry.
- [x] #3 Basic and Advanced projections agree after acknowledgement, including live child input already accepted while its Changed notification is queued. Existing Back, navigation and quit protections detect the retained draft.
- [x] #4 Mounted real-store regressions distinguish ordinary create/update, later accepted input, queued notification, definite failure, next Save and guarded exit; actual child input delivery and committed rows are witnessed.
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
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Prompt Save now captures raw live metadata and Basic/Advanced blocks, acknowledges only submitted input, and adopts the committed identity/version. Newer edits stay visible and dirty, obsolete semantic messages cannot replay old blocks, and the next Save updates the same row with the current version. The existing create-success action refresh is retained. Completion moved to UI/Library_Modules/prompt_save_ack.py; the controller ceiling drops from5,199 to5,100. Stateful children, history, services and exit guards remain with their existing owners.

Mounted real-SQLite controls cover create/update, later and queued body/metadata input, definite failure, guarded exit, next Save and old semantic publication. Superpowers review caught a stale other-lane Advanced projection; both convergence loops now synchronize all captured cards. Three new review regressions pass, plus11 neighboring Prompt controls. The combined Prompt module contains22 exact census cases. Full counts, RED/GREEN and fixture qualifications are in the [verification report](../../Docs/superpowers/qa/2026-10-23-textual-pending-draft-followups.md). Existing ADR-033 application state and ADR-040/049/076/086 apply; no new ADR. Independent review approved the implementation; required current-head PR #3228 CI is pending, so the task stays In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->