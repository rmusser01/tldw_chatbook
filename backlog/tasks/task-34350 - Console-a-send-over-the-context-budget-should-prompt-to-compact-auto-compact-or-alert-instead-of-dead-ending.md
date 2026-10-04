---
id: TASK-34350
title: >-
  Console: a send over the context budget should prompt to compact, auto-compact, or alert, instead of dead-ending
status: To Do
assignee: []
created_date: '2026-10-03 18:37'
labels:
  - console
  - context-compaction
  - follow-up-33940
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-33940.5, which removed a stale 4,096-token fallback after it made the shipped default model refuse every send in a fresh profile. That fix removed the bad number but left the underlying policy unchanged: a send that does not fit is stopped, and the user is not given a way forward at the point where it stopped.

On dev `01a2020981` the Console's preflight (`_apply_conversation_memory_preflight` in `Chat/console_chat_controller.py`, deciding via `decide_compaction` in `Chat/console_context_compaction.py`) behaves like this:

- **Compaction mode Ask (the default).** When the conversation reaches the compaction threshold, the send is blocked with a failure row reading "Conversation context reached its compaction threshold. Review and approve compaction before sending again." Nothing on that row or the composer lets the user approve anything. The only Compact now control is in Conversation settings ▸ Context and memory.
- **Compaction mode Automatic.** The preflight compacts and then sends.
- **Compaction cannot help** (no complete older turns to summarise, or the response reservation plus safety margin plus mandatory input already exceed the window). The send is refused with "This request cannot fit the selected model. …" This is also what fires when the window is only an *estimate*: an unverified window is treated as hard fact, as the 4,096 fallback showed.

**Owner ruling (2026-10-03):** a send that exceeds the budget should **prompt the user to compact**, or **trigger an automatic compaction if that is enabled**, or **alert the user if the conversation is too large to compact**.

Compaction mode Off keeps its current behaviour; it is not part of this ruling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With compaction mode Ask, a send that reaches the compaction threshold shows a prompt at the send surface offering to compact and send, or cancel; choosing compact runs the same compaction as Compact now and then sends the held message without the user retyping it
- [ ] #2 Cancelling that prompt leaves the user's message recoverable in the composer, with nothing sent and nothing compacted
- [ ] #3 With compaction mode Automatic, such a send compacts and sends without a prompt (current behaviour, pinned by a test)
- [ ] #4 When compaction cannot make enough room, the user gets an alert that names the limiting reason and the specific setting to change (response maximum, mandatory context, or model context window), and their message is not lost
- [ ] #5 An estimated (unverified) context window takes the same prompt / automatic / alert route as a verified one, and the alert says when the window is an estimate and where to set the real value
- [ ] #6 Tests drive the real controller through the prompt, automatic and alert outcomes, including one on an estimated window; the prompt and estimated-window tests fail on dev `01a2020981`
- [ ] #7 The prompt and the alert are verified in the running app against a real provider, and `Docs/User_Guide/console/context-and-rag.md` describes the new behaviour
<!-- AC:END -->
