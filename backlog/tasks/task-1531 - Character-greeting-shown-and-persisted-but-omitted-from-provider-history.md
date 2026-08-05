---
id: TASK-1531
title: 'Character greeting shown and persisted but omitted from provider history'
status: Done
assignee: []
created_date: '2026-07-30 15:30'
labels: [bug, roleplay, console, P2]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified live against origin/dev @ c329def0d (2026-07-30, stub provider
capturing payloads): Roleplay → Start Chat → Console. The character's
first_mes renders in the transcript as the opening Assistant message AND is
persisted to the DB as the conversation's first assistant message — but
outbound requests omit it. Captured turn-2 payload:
`[system, user(turn1), assistant(reply1), user(turn2)]` — the greeting
assistant turn is missing while every later assistant turn is included.

Impact: the model never knows the greeting happened and contradicts the
transcript the user is reading (e.g. this card's greeting asks the user to
fill in a character sheet; the model re-introduces the world instead of
acknowledging the sheet). Since the greeting IS in the DB history, whatever
builds the request history is filtering it (or slicing from the first user
message) — align it with the persisted transcript.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

1. Add pure `fold_greeting_into_system_prompt` to Chat/console_chat_models.py (shared by Console + preview).
2. Controller: `_seeded_greeting_text` mirrors the leading-assistant drop rule; `_leading_system_message(greeting=...)` folds it; wire both send seams (`_provider_messages_for_session`, `_provider_messages_through_message`). Message array stays user-first (task-427 constraint kept).
3. Preview: fold the active seeded greeting inside `build_preview_system_prompt`.
4. TDD: revise `test_leading_greeting_excluded_from_provider_payload` to the new contract (system row carries greeting; array user-first) + add with-system-prompt variant.
5. Live stub-provider payload re-verification (greeting text present in system row).

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The greeting content reaches the provider on every character-session send (AC reworded: it is folded into the SYSTEM row rather than sent as an assistant-first message, because strict providers (Anthropic, Gemini) reject assistant-first arrays -- the task-427 constraint that motivated the original drop).
- [x] #2 Roleplay preview requests include the displayed greeting the same way.
- [x] #3 Regression test asserts greeting presence at the request-payload seam.
<!-- AC:END -->

## Implementation Notes

Added pure `fold_greeting_into_system_prompt` to Chat/console_chat_models.py
(shared by Console and preview). Console: `_seeded_greeting_text` mirrors the
payload builder's leading-assistant drop rule (skips failed turns, joins
multiple leading turns) and both send seams
(`_provider_messages_for_session`, `_provider_messages_through_message`) fold
it via `_leading_system_message(greeting=...)`; a greeting now produces a
system row even when no session system prompt is set. The message array stays
user-first, preserving task-427. Preview: the active seeded greeting is folded
inside `build_preview_system_prompt`.

Revised `test_leading_greeting_excluded_from_provider_payload` to the new
contract (`test_leading_greeting_folds_into_system_row_not_message_array`)
and added the with-system-prompt variant; both watched RED first. The
regenerate/continue-blocked greeting tests still pass unchanged.

Live-verified against the stub provider: Console turn 1/2 and preview payloads
all carry the greeting text inside the system row with a user-first array;
DB transcript and provider view now agree.
