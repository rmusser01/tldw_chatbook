---
id: TASK-16334
title: Expose budgets and follow-up Q&A in the Research window
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 22:39'
updated_date: '2026-08-15 22:43'
labels:
  - research
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The engine enforces limits_json budgets (task-16323) and answers follow-up questions from stored claims (task-16325), but neither has window controls: every local run launches unbudgeted and follow-up Q&A is unreachable from the UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Research window has a limits input parsed into limits_json on run creation (numeric key=value pairs; invalid pairs warn in the status line without blocking creation),Follow-up questions can be asked against the selected completed local run and the answer or explicit insufficient-evidence verdict is displayed,Follow-up ask requires a selected run on the local source and reports a clear status otherwise,The limits text persists through save_state and restore_state,Tests cover limits parsing, payload wiring, follow-up display, and the no-selection guard
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD a module-level limits parser in Research_Window (numeric key=value pairs; invalid pairs excluded with warnings)
2. TDD window wiring: limits input in the create row parsed into limits_json on create_run (warnings to the status line without blocking), limits text persisted via save_state/restore_state
3. TDD follow-up Q&A: question input plus Ask button calling LocalResearchEngine.answer_follow_up on the selected local run with the tool-assembled synthesis LLM params; answer or explicit insufficient verdict rendered to a dedicated Static; no-selection and non-local guards report status without constructing the engine
4. Tests plus lint plus task close
ADR required: no - UI controls over existing engine seams
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `_parse_limits_text` (module-level in Research_Window): numeric `key=value` pairs; invalid pairs are excluded with warnings surfaced in the status line — one typo never blocks creation. The limits input sits in the create row (`#research-limits-input`), parsed into `limits_json` on `create_run`, and persists via save/restore state (`limits` key).
- Follow-up Q&A: `#research-followup-input` + Ask button + `#research-followup-answer` Static. `ask_follow_up(question)` guards no-selection (the existing `_selected_run_id` raises — caught), non-local source, missing local service, and empty question before constructing a transient `LocalResearchEngine` with the tool-assembled `final_answer_llm` search params (so the default answerer uses the configured pipeline). Answered results render `Q: …` + answer; insufficient results render the engine's explicit verdict + suggestion — never a fabricated answer.
- Guard ordering note surfaced by the tests: `switch_source` clears the selection, so the no-selection guard legitimately precedes the source guard when both apply.
- Verified TDD: 8 new tests written first and watched failing (parser valid/invalid/empty, limits payload wiring, state persistence, answered + insufficient rendering, no-selection and source guards); `Tests/UI/test_research_screen.py + Tests/Research/` = 128 passed; ruff clean.
<!-- SECTION:NOTES:END -->
