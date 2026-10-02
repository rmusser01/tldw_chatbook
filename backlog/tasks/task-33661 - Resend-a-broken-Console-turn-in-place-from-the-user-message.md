---
id: TASK-33661
title: Resend a broken Console turn in place from the user message
status: To Do
assignee: []
created_date: '2026-10-01 17:30'
labels:
  - console
  - transcript
  - owner-request
dependencies: []
references:
  - tldw_chatbook/Chat/console_message_actions.py
  - tldw_chatbook/Widgets/Console/console_transcript.py
  - tldw_chatbook/UI/Console_Modules/message.py
  - tldw_chatbook/Chat/console_chat_controller.py
  - tldw_chatbook/Chat/console_chat_store.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: when a Console send fails or gets stuck, the user sees an error and has no direct way to try again. Today the user has to choose Edit, then Edit & resend. That forks a new sibling branch, and for some failure shapes it is the only path left. The owner asked (2026-10-01) for a one-click **Resend** on the user's own message that resumes the broken chat.

Owner decisions:
- Resend appears only when that turn failed or got no reply.
- It re-runs the same turn in place: no fork, no sibling branch, no second copy of the user message.

Lead defaults (the owner can override): Resend applies to the last turn of the active path only, since a failed reply higher up keeps its own Retry. An empty reply that the user stopped counts as "no reply", which makes stuck → Stop → Resend work. A partial reply does not count, because Continue covers it.

How the code models a broken turn today (research against dev 84247cb843, 2026-10-01):
- Provider or HTTP error: the user message and the assistant row are both persisted. The assistant is marked failed, and a transcript-only SYSTEM "Provider stream failed" row follows. The assistant row offers Retry, but nothing on the user row does.
- Empty stream: the assistant is marked failed with no SYSTEM row.
- Stop on a hung or stuck run: the assistant is marked stopped and a SYSTEM "Response stopped by user." row follows. It offers ♻ (which forks) and Continue, but no Retry.
- Readiness or other refusal before the send is accepted: the user echo is unpersisted and marked failed, a SYSTEM block row follows, and there is no assistant row. The draft and staged attachments stay in the composer. The user row shows Continue, which today parents the new reply under the SYSTEM row and leaves the echo failed (an untested hazard).
- After a restart: hydration rebuilds nodes as complete, keeping only assistant_generation_state. A failed reply then reads "Response failed." and loses Retry.
- Dispatch recovery after a restart (a reply left in flight) has its own card with Retry response and Discard. Discard settles the reply as discarded.

Existing machinery that resend can reuse without forking:
- The assistant in-place retry (retry_message / prepare_message_retry) for a failed reply.
- continue_from_message, which re-runs from a persisted user message and appends a reply at the active leaf.
- delete_message (subtree tombstone).
- The normal send path, for re-dispatching a refused echo.

Related open tasks, not absorbed here: TASK-370 (resume or retry for interrupted replies), TASK-33620.3 (a failed durable commit leaves a user message with no assistant row, stuck on Running), TASK-33621.22 (Stop during "Connecting tools…" wedges the next send).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user message that is the last turn on the active path shows a Resend action only when its turn is broken. Broken means: its send was refused before acceptance; or it has no assistant reply; or its reply failed; or its reply is empty and stopped, discarded, or restored as "Response failed." after a restart. Resend never shows on a healthy turn, on a partial (non-empty) stopped reply, on a turn above the last one, or on an assistant row.
- [ ] #2 Resend re-runs that turn in place, with no new branch, no sibling and no second copy of the user message. The broken or empty reply and its trailing failure or stop SYSTEM rows are cleared, and the new reply appears directly under the same user message. A refused echo is re-sent with the same text and attachments as exactly one user message, and the composer is not left holding a duplicate of that text.
- [ ] #3 Resend is not offered while a run is live in that tab (Stop comes first) or while a dispatch-recovery card is unresolved (the card owns that case). Every normal send gate applies, including readiness, vision and skill refusal, and a refused Resend shows the same refusal copy as a normal send.
- [ ] #4 Resend works after an app restart for a persisted broken last turn: a restored failed or discarded reply, or a last user message with no reply.
- [ ] #5 Keyboard: with a broken user row selected, `r` runs Resend. With a failed assistant row selected, `r` runs Retry; today it does nothing there. The action's tooltip and the help or footer text describe Resend, and no hint names a key that does not work (ADR-031).
- [ ] #6 The selected action row still fits the reference terminals at 211x44 and 235x52. Tests that pin the user-row and assistant-row action sets are updated on purpose and named in the notes.
- [ ] #7 Tests cover each broken shape with a real controller and store: provider error, empty stream, refused echo, empty stopped reply, restored after restart, and discarded. Each proves the re-run is in place by message count, ids and parentage, with no sibling. There are negative tests for a healthy turn, a mid-path turn, a live run and an unresolved dispatch recovery. A pilot test clicks Resend in the real Console.
- [ ] #8 The Console User Guide documents Resend: when it appears and what it does.
- [ ] #9 No size ratchet rises (ADR-097). The resend logic lives in its own module. console_chat_controller.py, console_transcript.py and console_chat_store.py each net 0 or fewer lines, and preflight passes.
<!-- AC:END -->
