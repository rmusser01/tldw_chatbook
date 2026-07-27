---
id: TASK-553.15
title: Add provisional citation checking and one visible repair transition
status: Done
assignee: []
created_date: '2026-07-26 23:03'
updated_date: '2026-07-27 03:54'
labels:
  - rag
  - citations
  - provenance
  - console
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-26-local-citation-repair-transition-design.md
  - Docs/superpowers/plans/2026-07-26-local-citation-repair-transition.md
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep local RAG answers visibly provisional until citation markers are structurally checked, and make one bounded repair attempt without changing claims or overstating grounded trust.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A local RAG answer remains one provisional assistant message until structural citation checking and any repair select the visible body
- [x] #2 Valid markers complete without repair while missing or invalid markers trigger at most one direct tool-free repair using the same resolved provider and model
- [x] #3 A repaired body is selected only when its non-marker text is unchanged and its markers validate; otherwise the original body is selected with honest failure or cancellation copy
- [x] #4 Successful repair visibly replaces the same message and offers a keyboard-accessible current-session original-attempt preview without mutating message content persistence or provider history
- [x] #5 Citation checking remains available independently of canonical-write readiness and all repair prompts, outputs, buffers, and diagnostics are bounded and privacy-safe
- [x] #6 Direct-provider and agent-generated local answers preserve existing stop and session-close compatibility and pass scoped regression coverage
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed implementation plan:
`Docs/superpowers/plans/2026-07-26-local-citation-repair-transition.md`

1. Define pure bounded repair contracts, structural validation, unchanged-claim selection, exact prompt construction, and model-window checks.
2. Carry repair eligibility independently of canonical-builder readiness and add a content-free synthesized-fallback signal across direct and agent gateway calls.
3. Add explicit terminal persistence deferral, atomic repaired-body replacement, and safe transient presentation state.
4. Wire one controller-owned repair session through the shared direct/agent post-generation seam with phase-aware cancellation and one terminal write.
5. Add honest notices and a bounded current-session original-attempt preview without changing message content, persistence, provider history, TTS, or exports.
6. Run only scoped touched-code tests/static checks, perform self-review, and record verification and implementation notes.

ADR required: no
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This directly implements ADR-024 streaming and repair behavior and introduces no new architecture decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented a controller-owned, one-attempt citation transition over one provisional assistant message. Pure bounded contracts validate marker structure, construct the exact direct tool-free request, enforce the model window, and select repaired text only when non-marker text is byte-for-byte unchanged.
- Carried repair eligibility independently of canonical-write readiness, deferred terminal persistence until selection, reused the resolved provider/model and shared direct/agent lifecycle, and atomically persisted only the selected body. Cancellation, late chunks, provider failure, invalid output, fallback provenance, session close, and shutdown fail closed to the original body or closed-session result without a repair-phase stopped-message write.
- Added content-free stream signals and presentation metadata plus an eight-entry current-session original-attempt preview. Preview state does not alter message content, provider history, persistence, export, or TTS. One outer stream-entry cleanup boundary now clears the request-scoped evidence contract and provider resolution on every exit, including missing-owner returns, pre-dispatch compaction/window exceptions, direct/agent completion, cancellation, session close, and shutdown.
- Privacy coverage uses distinct initial-body, repaired-body, evidence, source-identity, locator, complete-repair-prompt-only, and provider-exception sentinels. A controlled builder seam inserts the prompt sentinel only into the exact two-message request, proves that exact payload is dispatched, and then excludes it with the other sentinels from stdlib/Loguru diagnostics, signals, presentation, cleaned session fields, run state/history, cleanup state, and governed persistence. The final matrix also covers request-fit failure, provider raise, empty/oversized output, invalid markers, changed claims, user cancellation (direct and agent), late chunks, session close, shutdown, and fallback bypass.
- Modified production surfaces: `tldw_chatbook/Chat/citation_repair.py`, `console_chat_controller.py`, `console_chat_store.py`, `console_chat_models.py`, `console_provider_gateway.py`, `console_agent_bridge.py`, `console_message_actions.py`, `Event_Handlers/Chat_Events/chat_rag_events.py`, `Widgets/Console/console_transcript.py`, and `UI/Screens/chat_screen.py`. Added or extended the scoped RAG, Console controller/store/gateway/agent/action, transcript, and native-flow tests named in the linked implementation plan; updated this task and the approved design spec.
- Scoped verification with `PYTHONPATH=/tmp/codex_task55315_no_mlx`: initial privacy TDD proof RED `1 failed, 366 deselected`; follow-up outer-cleanup RED `3 failed, 2 passed` and GREEN `5 passed`; final privacy matrix `22 passed, 364 deselected`; complete unchanged pure file `Tests/Chat/test_citation_repair.py` previously passed `117 passed`; final focused groups `18 passed, 91 deselected`, `17 passed, 152 deselected`, `108 passed, 250 deselected`, and `20 passed, 318 deselected`. The three groups that import `requests` emitted one existing dependency-version warning each.
- Exact 22-file Ruff check passed. The exact 11-file format check reported inherited whole-file drift in `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, and `tldw_chatbook/Widgets/Console/console_transcript.py`; the other seven files passed, and changed-range checks found every new controller/test hunk formatted, so unrelated baseline files were not rewritten.
- Task 7's broader native-flow run reached `208 passed` with two unrelated search-timing failures that pass together in isolation; no new baseline task was created because no Task 8 scoped gate failed. Unstubbed native collection remains separately tracked as TASK-839 due to the optional MLX import abort, so Task 8 used the required MLX guard and did not broaden to a full suite.
- ADR required: no. ADR path: `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`. Reason: this implements ADR-024's approved streaming/repair behavior without changing storage, authority, provider, or application boundaries.
<!-- SECTION:NOTES:END -->
