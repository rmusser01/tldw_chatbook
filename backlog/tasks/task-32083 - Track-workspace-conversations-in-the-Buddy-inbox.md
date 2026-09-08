---
id: TASK-32083
title: Track workspace conversations in the Buddy inbox
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 19:32'
updated_date: '2026-09-08 21:06'
labels:
  - buddy
  - console
dependencies:
  - TASK-32082
references:
  - Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Present active workspace conversations and directed replies through a nonintrusive Buddy inbox.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inbox separates Needs you, Running and unseen Results from historical inactive chats.
- [x] #2 Selecting a conversation permits directed text reply and review without leaving the current screen.
- [x] #3 Workspace mode has no voice input, including inside a selected conversation.
- [x] #4 Opening inbox does not mark all results read or resolve questions; acknowledgement is result-specific.
- [x] #5 Updates are scoped to the bound workspace and never steal focus.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes; existing ADR applies. ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md. Reason: independent Buddy scope and existing app-owned Console activity authority.
1. Project active local sessions and existing durable unseen receipts into exact workspace inbox entries.
2. Add a native keyboard/compact-safe inbox with Needs you, Running and Results and explicit result-specific acknowledgement.
3. Route row interaction to the shared conversation modal with voice disabled and preserve focus on refresh.
4. Verify workspace filtering, stale targets, frozen acknowledgement, no mutation on open and mounted directed row interaction; document behavior.
5. Fix cold-start receipt access through a narrow app-owned ConsoleRuntime.ensure_activity_receipt_service seam: reuse one AgentRunsDB/lazy service across inbox and later bridge creation, fence creation against disposal, close worker-owned connections, and report loading/degraded storage without false empty results. Verify persisted ordinary receipts from a fresh runtime on Home with no bridge/provider/selection and targeted resource-lifetime races.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review follow-up (buddy_ui_review): cold workspace inbox currently treats an uninitialized receipt owner as no unread results. Approved bounded change will initialize existing local receipt authority without creating an agent bridge/provider or selecting Console. ADR-139 applies; no new storage schema or provider boundary. Implementation and targeted evidence to follow.

Cold-start receipt fix (buddy_ui_review): ConsoleRuntime.ensure_activity_receipt_service() now initializes and retains one app-owned AgentRunsDB/lazy receipt service without constructing a Console store, controller, provider or agent bridge. Later bridge initialization reuses the same hydrated receipt identity. A serialized constructor and the existing disposal publication latch fence creation against shutdown; background initialization and hydration close their own thread-local SQLite connections, and disposal waits away from the UI loop for any in-flight construction.

BuddyWorkspaceCoordinator.snapshot initializes storage off-loop and projects a frozen receipt snapshot only after hydration is ready. Cold/degraded/unavailable storage reports actionable loading/retry copy rather than an empty inbox; runtime disposal/profile replacement still rejects stale snapshots. Opening remains read-only and does not acknowledge receipts or change selection. ADR-139 applies; no schema or provider boundary change.

Verification: 16 targeted tests passed in 7.32s: new Tests/Chat/test_console_receipt_bootstrap.py (real SQLite persisted ordinary result from mounted fresh Home with no Console/provider/bridge, later bridge reuse, concurrent construction, first-creation/dispose race, per-thread cleanup, absent/in-memory DB, loading/degraded states), existing Buddy workspace coordinator cases, receipt ownership/hydration cases from test_console_runtime_ownership, and three selected runtime disposal/resource guards. New tests were red before implementation (missing API, absent degraded error, Home inbox missing persisted result). Ruff and formatter pass for new test file; zero Ruff diagnostics in owned implementation methods, with 9 existing diagnostics elsewhere in the two production files. All three Python files parse and scoped git diff --check passes. Existing RequestsDependencyWarning remains. No commits; parent owns remaining task acceptance and final status.

Root implementation: added read-only workspace projection over current local session/queue/decision snapshots and immutable existing unread result receipts, native Needs you/Running/Results inbox, exact frozen-ID acknowledgement and directed row opening with voice disabled. Refresh preserves selected row and focus; opening or speaking never acknowledges. Hidden selected-session ordinary and queued completions now publish unread receipts using existing authority. Fixed Close awaiting safe dismissal and bounded row-open storage errors. Six workspace coordinator/normal and compact modal checks with speech controls, 6 projection checks and 4 hidden/visible direct/queue receipt cases pass. Agent cold receipt initialization evidence is recorded above. Saved-row cold interaction is being completed under TASK-32082; final combined gate pending. ADR-139 and Docs/User_Guide/buddies.md document scope and limitations.

Final integration complete: cold Home result opening now lazily boots the existing headless runtime only on explicit interaction, hydrates exact saved rows without selecting Console, and retains their later questions/confirmations. Concurrent loader admission is verified by the TASK-32082 regression. Root receipt, projection, exact acknowledgement, no-microphone, normal/compact speech controls and focus checks pass; combined runtime/UI gate includes the actual cold saved transcript -> Send -> Close -> late question/tool approval journey. Independent final review is clear after fixing the cold opener and retained target race. No new run owner, automatic acknowledgement or voice input was added. ADR-139 and user guide capture shipped limits.
<!-- SECTION:NOTES:END -->
