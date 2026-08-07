---
id: TASK-406
title: Wire RAG context injection into the native Console send path
status: In Progress
assignee: []
created_date: '2026-07-21 09:48'
updated_date: '2026-08-07 20:44'
labels:
  - rag
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified during the RAG scope program (plan verification V4, re-confirmed by two independent reviews): the native Console send path (ConsoleChatController.submit_draft) performs NO RAG context injection at all — get_rag_context_for_chat's only production caller is the legacy chat_events send path, which is unreachable in the live app (routes/dead-sites traced in the Task 5 review). Users of the native Console therefore get no chat-RAG injection regardless of settings. Mirror the chat-dictionaries precedent (PR 664: transform applied at all native send sites): resolve scope + inject RAG context in the native path, honoring conversation scope (Chat/rag_scope.py resolution seams exist and are public as resolve_effective_scope_for_chat).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native Console sends inject RAG context only when the 'Auto-retrieve on send' toggle in the Console RAG chip modal is ON (default OFF; enablement is re-homed from the legacy sidebar checkbox to this modal toggle, persisted as a global config key); when ON, injection routes through the existing staged-evidence pipeline -- a visible 'Evidence sent · N' strip, consumed only on send -- honoring conversation scope end-to-end, never as invisible prompt injection.
- [x] #2 EMPTY scope short-circuits with the shared notice copy on the native path
- [x] #3 Legacy path behavior unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Status note: all 3 ACs above are ticked as CODE-COMPLETE. Left In Progress,
not Done -- TASK-3170's Task 11 (gates + a live TUI walkthrough) still owes
the Definition of Done's live-verification evidence for this specific send
path before either task closes.

Approach: implemented as TASK-3170's tasks 7 (config key + RAG chip modal
toggle, `chat_screen.py` + `console_rag_settings_modal.py`) and 8 (the actual
send-path seam, `chat_screen.py`). AC #1's toggle persists the instant it is
flipped (a modal callback wired to a `@work(thread=True)` writer), not tied
to the modal's Run/Cancel draft-discard semantics -- an earlier draft
persisted only on dismiss and lost the change on Escape; fixed in a review
round before this task counted as code-complete.

`ChatScreen._maybe_auto_retrieve_for_send` is called from
`_capture_console_staged_rag`, immediately before `_consume_pending_console_
launch()`, inside the same exclusive `run_worker(..., group=f"console-run-
{session_id}")` the send already dispatches under -- no second worker, so a
double-send cannot double-retrieve. Reached only after every send-blocking
gate (provider readiness, workspace policy, skill-substitution refusal), so
a blocked/refused send never auto-retrieves. Gate order (each mutation-
tested to red exactly its own test): toggle off -> return; non-plain-text
send (slash command, `$skill` invocation, second-Enter literal `/word`) ->
return; evidence already staged (resident launch OR an unclaimed
`HandoffChannel.CONSOLE_LIVE_WORK` entry) -> return; EMPTY resolved scope ->
notify with the shared `SCOPE_EMPTY_NOTICE_TEMPLATE` copy (AC #2), return;
otherwise stage a "Retrieving..." placeholder and await the search under a
5s `asyncio.timeout`, with `top_k` from the active profile's `default_top_k`
(task 9 gave the chip's manual run the same source, so the two paths cannot
disagree about depth).

Two departures from the original sketch, both because reusing the manual
chip run's own recovery path wholesale would have let a single empty auto-
retrieve lock the composer (the existing block-while-`available_count==0`
guard is screen-level, not query-specific): a zero-result outcome clears the
placeholder silently rather than staging the manual run's blocking recovery
card (tracked as a UX follow-up, TASK-3504, not a defect); a failed/timeout
outcome shows a quiet notice distinguishing "RAG service still initializing"
(no cached runtime -- the timeout is paying the first-run embedding-model
load) from "retrieval failed" (a real error), and the auto-retrieve await
itself is wrapped in try/except so an exploding retrieval cannot also
discard a manually staged evidence bundle the same send was about to
consume. AC #3 (legacy path untouched): confirmed by diff -- no
`chat_events`/`get_rag_context_for_chat` file was touched by tasks 7-8.

Files: tldw_chatbook/config.py, Widgets/Console/console_rag_settings_modal.py,
UI/Screens/chat_screen.py. Full test/mutation evidence in
.superpowers/sdd/2026-08-07-rag-port-p0-foundations/task-{7,8,9}-report.md.
<!-- SECTION:NOTES:END -->
