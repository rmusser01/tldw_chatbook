---
id: TASK-406
title: Wire RAG context injection into the native Console send path
status: Done
assignee: []
created_date: '2026-07-21 09:48'
updated_date: '2026-08-07 21:36'
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

LIVE VERIFICATION (2026-08-07, TASK-3170 Task 11) -- the status note above is
now discharged; this task is Done. Driven in the real TUI (tmux 235x52) on a
scratch profile (`TLDW_CONFIG_PATH`, `users_name = verify_ragp0`) holding a
copy of the real ChaChaNotes + media DBs and the real `chromadb/` directory,
against a real Anthropic account (claude-haiku-4-5-20251001) and the default
Hybrid Basic profile:

- AC #1 toggle + persistence: flipping "Auto-retrieve on send" in the RAG chip
  modal wrote `[chat_defaults] rag_auto_retrieve_on_send = true` to the config
  file at toggle time -- confirmed by reading the file WHILE the modal was
  still open -- and Escape left it set (the dismiss path is write-free).
- AC #1 injection through the staged pipeline, end to end: a plain-text send
  showed "Auto-retrieving Library evidence for this message." at t=1.0s with
  the footer chip reading `RAG: on · Sources: 1 staged`, then "Evidence sent
  with this message · 15 sources" at t=2.1s. The reply that came back named
  the injected block back to us -- "the evidence sections [S1] through [S15]
  contain only repeated Latin placeholder words..." (the corpus fixture that
  dominates this profile's index) -- which is the proof that the staged
  evidence actually reached the provider rather than being staged and dropped.
  Depth 15 is the profile's `default_top_k`, i.e. the TASK-3170 AC#8 parity.
- AC #1 gating: a send beginning with a slash command fired NO retrieval --
  no placeholder, no chip flip, no evidence line (polled at 0.15s for 18s).
- AC #2/#3 are covered headlessly (Tests/UI/test_console_auto_rag_on_send.py);
  the empty-scope short-circuit was not driven live because the scratch
  profile could not cheaply produce an empty resolved scope.

One live observation worth carrying forward, already filed: an auto-retrieve
that returns zero rows is completely silent (placeholder cleared, no notice),
so the model answers unaided and the user cannot tell retrieval was attempted.
Observed on a first send whose phrasing matched nothing; that is TASK-3504.
Evidence captures: scratchpad ragp0-evidence/04a..04d.
<!-- SECTION:NOTES:END -->
