---
id: TASK-18921
title: 'Slash-command popup: rank suggestions by usage'
status: Done
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-10-11 00:00'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's "slash menu leads with your most-used skills" idea (2026-08-19 hermes-release review). The Console slash-command popup (ConsoleCommandPopup, fed by Chat/console_command_suggestions.py) orders suggestions statically. Rank suggestions within the current prefix-filtered set by observed usage — command dispatches and `$skill` invocations counted from existing local records — so frequently used commands surface first. Display-only; matching and filtering semantics unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Suggestion order within a filtered set is usage-ranked (most-used first) with deterministic tie-breaking (alphabetical) so tests are stable — `rank_suggestions_by_usage` sorts by `(-count, label.lower(), label, insert_text)`; pinned by `test_rank_most_used_first_with_alphabetical_ties` and `test_rank_is_a_permutation_of_the_filtered_set`
- [x] #2 Usage counts derive from existing local data (prompt history JSONL / run records); no new network calls, no telemetry, nothing leaves the machine — counts are mined by `count_suggestion_usage` over the shared `prompt_history.jsonl` store. Decision recorded: command dispatches were NOT previously in any local record (verified — command sends return before the turn-submit path that records history), so accepted dispatches now append a minimal content-free token (`/<name>`, plus the skill name for `/skills <name>`) to that same existing JSONL at the `_dispatch_console_command` choke point. No new store, no network, no args recorded. Ghost text is unaffected (slash drafts never get ghost suggestions); Up-arrow recall gaining past commands is the standard-CLI reading. Pinned by `test_count_mines_slash_commands_and_skills_invocations`, `test_command_dispatch_records_usage_token`, `test_unknown_command_dispatch_records_nothing`
- [x] #3 First-run behavior with zero usage data is identical to today's ordering — `rank_suggestions_by_usage` returns the input order whenever no row in the set carries a non-zero count; pinned at the pure seam (`test_rank_with_no_usage_data_returns_input_order`) and at the UI seam (`test_popup_order_is_registry_order_with_no_usage_data`)
- [x] #4 Unit + UI tests pin ordering, tie-breaking, and that prefix filtering is unchanged — 8 new unit tests in `Tests/Chat/test_console_command_suggestions.py`, 4 new UI tests in `Tests/UI/test_console_command_popup.py`; `test_ranked_suggestions_keep_prefix_filtering_semantics` and `test_popup_orders_most_used_command_first` pin that ranking never adds or drops rows
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: display-order-only change over an existing local data source; no schema or boundary change.

1. Add a small usage-count source over the existing prompt history / run records
2. Feed ranked suggestions into ConsoleCommandPopup (keep the idempotent same-suggestions guard intact)
3. Tests + brief docs note in the Console chat-basics page
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Approach: three layers, mirroring the module's purity discipline. (1) Pure ranking/counting in `Chat/console_command_suggestions.py`: `suggestion_usage_key` (canonical `command:<name>` / `skill:<name>` keys), `count_suggestion_usage` (mines history texts: leading `/name` tokens, `/skills <name>` first args, `$name` mentions once per text, no code-span masking — documented simplification), `rank_suggestions_by_usage` (identity when no row has usage; else `(-count, label asc)`). (2) A `PromptHistory.entry_texts()` sync snapshot accessor (in-memory only, IO-free, `revision`-keyed) — the UI path never triggers a load. (3) ChatScreen wiring: `_console_suggestion_usage_counts` caches `(revision, counts)` (class-attribute default for the hand-built-fixture discipline, per-instance on write, best-effort try/except so ranking can never break the popup), `_rank_console_command_suggestions` applies ranking in `_sync_console_command_popup` after filtering, and `_record_console_command_dispatch` fire-and-forget-records accepted dispatches (content-free token, args dropped except the `/skills` skill name) at the `_dispatch_console_command` choke point after handler resolution, so unknown/blocked commands record nothing.
- Key decision: recording goes into the existing prompt-history JSONL, not a new counts store — the AC names that file as the source; it already stores every accepted draft, command dispatches are accepted drafts of a different kind, and no new telemetry surface is created. Verified pre-change that command sends never reached `_record_prompt_history` (they return at the command branch before the turn-submit path), so without recording, command ranking could never work.
- Matching/filtering semantics are untouched: ranking runs on the already-filtered list; the popup's idempotent same-suggestions guard still sees the ranked list as value-equal on re-syncs.
- Tests: 8 unit (Tests/Chat/test_console_command_suggestions.py) + 4 UI (Tests/UI/test_console_command_popup.py, marked `@pytest.mark.bootstrap_profile` per the TASK-32873 pattern because the file's `_build_test_app` harness is red at dev without it — 10 of its 11 pre-existing nodes fail identically at base `9f8533686c`, unchanged by this work).
- Modified files: `tldw_chatbook/Chat/console_command_suggestions.py`, `tldw_chatbook/Chat/prompt_history.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/Chat/test_console_command_suggestions.py`, `Tests/UI/test_console_command_popup.py`, `Docs/User_Guide/console/chat-basics.md`, this task file.
