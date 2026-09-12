# Task 1 — Project generation blocks into trusted Assistant activities

## Ownership

Own only:

- `tldw_chatbook/Chat/console_chat_models.py`
- `tldw_chatbook/Chat/console_turn_grouping.py`
- `Tests/Chat/test_console_thinking_presentation.py`
- `Tests/Chat/test_console_turn_grouping.py`
- this Task 1 report

Do not edit disclosure widgets, transcript behavior, CSS, Settings, Context controls,
Planning labels, Backlog tasks, or later-task tests. You are not alone in the codebase;
do not revert other changes and adapt to the current worktree.

## Required behavior

- Follow Task 1 in `Docs/superpowers/plans/2026-08-26-console-thinking-blocks-ui-settings.md`, ADR-090, the approved spec, and the child acceptance criteria.
- TDD: write the focused failing projection tests first, run them and record the expected RED, then implement the smallest production change and record GREEN.
- Add the exact application constant:
  `PROPRIETARY_THINKING_NOTICE = "Proprietary thinking obfuscated - not available"`.
- Project only actual supported displayable/proprietary envelope blocks. No envelope means no synthetic activity.
- Use deterministic trusted internal IDs derived from session, assistant owner, and block identity. Imported/hostile/raw block IDs must never become Textual DOM/selection/CSS identities.
- Expand activity status vocabulary only as required: live, done, stopped, failed, unavailable while retaining existing statuses.
- Map proprietary evidence to label `Thinking` plus unavailable status; never put the unavailable wording into the label and never source a body from storage.
- Preserve exact Assistant activity order across multiple model rounds and interleaved TOOL activity: a model block precedes the first tool activity belonging to its round.
- Cover duplicate block IDs and same IDs across sessions/assistant owners.
- Keep the projection pure and minimal; no dependency and no parallel presentation system.

## Verification and handoff

- Run the focused presentation and grouping suites plus the nearest pure model regressions touched by the status type.
- Run Ruff format/lint on owned files and `git diff --check`.
- Write `.superpowers/sdd/2026-08-26-console-thinking-blocks-ui-settings/task-1-report.md` with RED/GREEN commands, counts, decisions, and known concerns.
- Commit owned code/tests/report cleanly. Do not update Backlog status.
- Do not spawn subagents.
