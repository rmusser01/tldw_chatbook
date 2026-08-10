---
id: TASK-14808
title: Deliver the visible Console prompt queue experience
status: To Do
created_date: 2026-08-10 06:11
dependencies:
- TASK-14806
labels:
- console
- agents
- ux
priority: high
references:
- backlog/decisions/046-visible-bounded-console-prompt-queue.md
- backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
documentation:
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the already-safe bounded queue through an honest keyboard-first Console shelf and manager, document it, and prove the complete experience live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a turn is accepted, Send becomes Queue and Enter or the button admits the exact text draft; Preparing, Queue full, Send, slash-command, attachment, staged-evidence, and boundary-race states use the approved routing and preserve refused drafts exactly.
- [ ] #2 A one-row queue shelf and focused manager expose count, state, safe previews, edit, reorder, remove, clear waiting, pause, resume, retry, skip, resume next, retry stopped turn, review, and current-context confirmation with stable focus and stale-revision recovery.
- [ ] #3 Collapsed and background presentations reveal only content-free counts and states; previews strip controls, escape Rich markup, and truncate by terminal cell width, while full prompt text is fetched only for the entry actively edited.
- [ ] #4 Composer, mouse, keyboard, voice, hands-free, and programmatic sends share one UI dispatcher and one exact per-session Textual chain worker, while chat_screen size and method ratchets do not increase.
- [ ] #5 F1 help, fleet and session markers, Console user guides, collapsed-composer documentation, and close, leave, and quit documentation match the shipped queue vocabulary and controls.
- [ ] #6 Mounted tests prove rendering, key routing, draft transactions, revision-gated polling, safe previews, privacy, lifecycle-dialog integration, and neighboring control geometry at 80x24, 100x30, and 160x40.
- [ ] #7 An isolated live-provider walkthrough verifies sequential draining, management, approval, pause, failure, Stop recovery, background isolation, one final notification, and close, leave, and quit warnings.
- [ ] #8 Unsent queue text remains absent from database persistence, screen snapshots, prompt history, diagnostics, and logs; accepted queued turns persist through the normal message path.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
