---
id: TASK-32082
title: Reply to a conversation through its Buddy
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 19:31'
updated_date: '2026-09-08 21:03'
labels:
  - buddy
  - console
dependencies:
  - TASK-32081
references:
  - Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow context and replies to a pinned conversation from another application destination.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Click opens the named conversation transcript/activity/decisions and a text composer without navigating the underlying screen.
- [x] #2 Replies and approvals target the explicit bound conversation even when Console has another active conversation.
- [x] #3 Per-conversation drafts survive closing; Open in Console is available and close restores focus.
- [x] #4 Supported conversation voice uses existing guarded voice facilities; microphone capture stops when interaction closes.
- [x] #5 Closing, hiding or rebinding never stops accepted work; deleted/inaccessible targets never fall back.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: directly implement approved explicit conversation interaction using retained Console execution and existing guarded decisions/voice.
1. Add an app-owned exact-binding coordinator and mounted projection modal; preserve per-binding drafts and opener focus.
2. Add a narrow composer-preserving manual send option with visible refusal of unseen staged inputs, retaining all normal admission/permission and durable recovery checks.
3. Project existing approval/question cards and resolve only exact current owner rounds. Extend answerable-time claims for the visible Buddy projection.
4. Reuse guarded Console streaming dictation with explicit start/finish, review-before-send, owner/generation fencing, and microphone discard on close/suspend.
5. Add targeted two-session mounted text/approval, stale/deleted owner, draft/focus and microphone-close coverage; record exact results and limitations.
Detailed plan: Docs/superpowers/plans/2026-09-08-buddy-conversation-interaction.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented app-owned conversation Buddy interaction over retained Console execution: named transcript/activity, existing question/tool/skill cards, a private composer, Close/focus restoration and exact Open Console handoff. Replies preserve both Console drafts and refuse unseen staged attachments, evidence, one-shot prefill, slash commands and @ routing. The preserve_composer intent survives prepared/durable recovery; ordinary admission, permission and provider rules remain authoritative.

Explicit saved-row interaction uses shared full-tree hydration with activate=False and repeated durable/profile/owner checks. Cold Home opening may initialize the established launch runtime; passive inbox reads do not. Bootstrap failure keeps Open Console recovery. Both newly hydrated and concurrently restored exact sessions retain a weak session/revision decision capability after modal close; wake-only siblings retain their existing no-view refusal. Per-kind visible claims spend finite decision time only on answerable cards; worktree decisions retain their existing Console review.

Dictation reuses ConsoleStreamingDictationSession guards, explicit start/finish, editable text and separate Send. Closing or suspending releases the microphone; late results cannot alter a closed/replaced owner. Shared speech controls hold playback during microphone capture. Drafts transfer when the same durable conversation gains a canonical binding. Accepted runs remain app-owned through close/navigation; explicit Stop and shutdown still cancel.

Validation: final race/cold/retained-decision/bootstrap-failure subset 6 passed in 14.37s (/private/tmp/buddy-loader-race-green.log); clock tests 30 passed in 1.84s; runtime lifetime/interrupt/ask_user/durable/clock gate 105 passed in 30.73s; controller submission selection 22 passed in 4.76s. Parent combined stable-source gate passed every conversation and clock case; its three unrelated failures remained with root. Earlier mounted conversation/clock gate:50 passed in 56.40s. All five owned host/new module/test files pass Ruff lint and full-file formatting; shared controller introduces no Ruff diagnostics versus HEAD (178 preexisting). git diff --check clean.

Files: UI/Navigation/buddy_conversation.py; Widgets/Persona_Widgets/buddy_conversation_modal.py; narrow Chat/console_chat_controller.py and console_interrupt_rounds.py changes; Tests/UI/test_buddy_conversation_modal.py; Tests/Chat/test_console_decision_clock.py; Docs/User_Guide/console/buddy-conversation.md; Docs/superpowers/plans/2026-09-08-buddy-conversation-interaction.md; testing-evidence lesson. Existing ADR139 amended for explicit cold bootstrap and retained decision availability.

Evidence uses mounted Textual and real SQLite with injected provider/recorder boundaries. No full sweep, live provider, physical microphone, audible playback or realtime-voice claim. No commits made.
<!-- SECTION:NOTES:END -->
