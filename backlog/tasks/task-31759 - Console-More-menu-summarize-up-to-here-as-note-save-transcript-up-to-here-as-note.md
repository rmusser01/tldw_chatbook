---
id: TASK-31759
title: >-
  Console More-menu: summarize up to here as note + save transcript up to here
  as note
status: In Progress
assignee:
  - '@robert'
created_date: '2026-09-06 02:45'
updated_date: '2026-09-06 02:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add two per-message actions to the Console message More-menu: (1) summarize the active-path conversation span up to and including the selected message into a note, and (2) save the formatted transcript of that span as a note. Both write to the notes library via notes_scope_service and are independent of the /rewind compaction state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] More-menu on a completed USER/ASSISTANT message offers 'Summarize up to here as note' and 'Save transcript up to here as note'
- [x] Summarize action uses a dedicated internal prompt (console.summarize_note) and a stateless provider call that does NOT write context_summary or move the rewind boundary
- [x] Save action writes a role-prefixed Markdown transcript with provenance header, inclusive of the selected message, active-path only
- [x] Both notes are created via notes_scope_service.save_note with keywords=['console']
- [x] Oversized summarize spans are blocked with a user-visible notice (no silent truncation)
- [x] Actions are blocked with a notice while a run is active; summarize requires a configured provider
- [x] Unit tests cover action availability, dispatch, gates, note content, and no-compaction-side-effects
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Register console.summarize_note internal prompt
2. Add public stateless summarize_span_to_text on the console_context_compaction service
3. Controller: span slicing helpers + transcript-note builder + summarize_span_as_note
4. Action service: two overflow entries + dispatch branches in the message controller
5. UI wiring: button-id parser entries, exclusive workers, transient notices
6. Tests mirroring existing console action/summarize suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

- **Approach**: Reused the existing origin/dev More-menu overflow (`_overflow_actions`) — the two actions are overflow-only entries behind the existing `More…` popup, dispatched via the pre-`dispatch()` interception pattern used by `save-as`. No new modal or menu widget.
- **Stateless summarize seam**: `ConsoleCompactionService.summarize_span_to_text` wraps the existing `_summary_completion` (bounded, aux-model routed) without the attempt ledger, branch-memory admission, or boundary commit that `summarize_manual` always performs. Controller method `summarize_span_as_note` uses it with the new `console.summarize_note` prompt; the `/rewind` context summary and boundary are provably untouched (asserted in tests, including zero `console_auxiliary_attempts` rows).
- **Span semantics**: active-path messages up to AND INCLUDING the selected message (USER or ASSISTANT target), skipping TOOL rows, failed rows, and empty rows. Oversized spans (> `_SUMMARY_SPAN_TOKEN_BUDGET`) block with a notice rather than silently trimming — a note must not quietly omit turns.
- **Notes write**: both drafts flow through one shared `_save_console_note_draft` tail that mirrors `_save_console_message_as_note` (`notes_scope_service.save_note`, `scope=LOCAL_NOTE`, `keywords=["console"]`), running in the `console-note-actions` exclusive worker group (never `console-run`, so it cannot cancel a live stream).
- **Gotcha found**: `_parse_console_message_action_button_id` has an explicit prefix table; the new action ids had to be registered there or presses fall through unhandled.
- **Files modified**: `Internal_Prompts/console_prompts.py` (new prompt), `Chat/console_context_compaction.py` (stateless method), `Chat/console_chat_controller.py` (`ConsoleNoteDraft`, `_note_span_messages`, `_note_provenance_header`, `build_transcript_note`, `summarize_span_as_note`, `_NOTE_SUMMARY_OUTPUT_CAP`), `Chat/console_message_actions.py` (two `_COMPLETED_ACTIONS` + overflow entries), `UI/Console_Modules/message.py` (router interception, worker methods, parser prefixes), tests: `Tests/Chat/test_console_note_span_actions.py` (new, 10 tests) + expectation updates in `Tests/Chat/test_console_message_actions.py`.
- **ADR check**: ADR required: no — no schema/sync/boundary changes; direct use of existing store/notes/compaction seams within existing surfaces.
- **Verification**: targeted runs — `Tests/Chat/test_console_note_span_actions.py` (10 passed), `test_console_message_actions.py` (106), `test_console_rewind_summarize.py` + `test_internal_prompts_panel.py` (76), `test_console_context_compaction.py` (143), `test_console_run_gate.py` (in 334-run), chat-flow action subset (21). `Tests/UI/test_console_native_transcript.py` shows 31 pre-existing failures identical on clean origin/dev (verified via stash baseline) — not regressions. Full suite not run (per repo policy).
