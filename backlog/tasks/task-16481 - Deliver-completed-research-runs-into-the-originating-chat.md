---
id: TASK-16481
title: Deliver completed research runs into the originating chat
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 03:00'
updated_date: '2026-08-16 03:18'
labels:
  - research
  - console
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The deep-research flow lives isolated in the Research window: nothing can launch a run from where users actually work (the Console chat), and completed reports land in a window the user must remember to check. The run schema already carries an unused chat_handoff_json, and the server solved exactly this with its chat_handoff machinery (assistant message with deep_research_completion metadata on packaging, notification fallback).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A run launched with a chat handoff target records it in chat_handoff_json,The engine fires an injectable completion callback with the report and bundle summary when such a run completes (callback failures never fail the run),The app wires the callback to insert an assistant message into the target conversation carrying the report and deep_research_completion metadata, with the existing terminal-run notification as fallback when insertion is impossible,A Console chat entry point launches a local research run with the handoff wired to the current conversation,Tests cover the engine callback contract (fire on completion with handoff, silence without, callback errors swallowed) and the message insertion wiring
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Engine (`local_research_engine.py`): `completion_handoff` constructor seam; fired after `complete_run` when the run's `chat_handoff` dict is non-empty and the run completed. Payload: `{run_id, question, chat_handoff, report_markdown, bundle, verification_summary}`. Awaitable callbacks are awaited; handoff exceptions are warnings — the terminal state is already recorded and a delivery problem must never retroactively fail a run.
- New `Research_Interop/chat_handoff.py`: `insert_research_completion_message(db, payload)` inserts an assistant message into `chat_handoff.conversation_id` with the report content and a `deep_research_completion` metadata block (run id, question, source count, confidence, gate verdict). Missing target or DB failure returns None (the existing terminal-run notification stays the fallback) — never raises.
- Console entry point: `/research <question>` registered in the command grammar + suggestions description ("Run deep research in the background; the report is delivered into this conversation") + `chat_screen` dispatch (`_console_command_research`). Guards: empty question, no active conversation, missing local service. The run launches in a worker with the tool-assembled LLM params, the academic lane when enabled, and the handoff wired to `chachanotes_db`; a native system message confirms the start and where the report will land.
- Verified TDD: 3 engine handoff tests + 3 inserter tests + 1 registration test written first and watched failing; command-grammar/suggestions pins updated for the new built-in (their own invariant test correctly caught the missing popup description — added). Suites: Research + grammar + suggestions = 142 passed; screen_navigation = 129 passed (one earlier failure was the known cross-run flake, green on re-run); ruff findings on touched files are pre-existing only.
<!-- SECTION:NOTES:END -->
