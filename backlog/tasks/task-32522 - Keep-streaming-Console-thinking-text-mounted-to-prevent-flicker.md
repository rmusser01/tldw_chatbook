---
id: TASK-32522
title: Keep streaming Console thinking text mounted to prevent flicker
status: Done
assignee:
  - '@codex'
created_date: '2026-09-13 02:52'
updated_date: '2026-09-13 03:05'
labels:
  - console
  - bug
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expanded model thinking briefly disappears while receiving streamed reasoning. Preserve continuous readable thinking text through updates and verify the result in the rendered UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Expanded thinking remains painted continuously during same-block streaming updates and line wrapping.
- [x] #2 Thinking body updates preserve widget identity, disclosure focus, and existing collapse, reopen, and answer-boundary behavior.
- [x] #3 Targeted regressions, lint and formatter checks, and streaming UAT pass with recorded evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the mounted thinking-disclosure regression to pin the actual text widget and observe rendered frames while streamed text wraps. Confirm failure on the current implementation.
2. Update same-block expanded thinking content in place, retaining structural replacement only for mount/unmount transitions.
3. Run targeted disclosure, thinking-edit, and assistant-turn tests plus Ruff and formatter checks.
4. Perform isolated live streaming UAT against the available local llama.cpp server; inspect thinking frames and exercise collapse/reopen and the answer transition. Record evidence and self-review the diff.

ADR required: no
ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md
Reason: Routine rendering bug fix preserving the existing disclosure and generation-ownership contract; no new architecture decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retained the expanded thinking Static during same-block text changes, removing the asynchronous remove/mount gap while preserving structural mount/unmount for collapse and ownership transitions. Extended the body-identity regression and added actual compositor continuity checks at 60/100 columns; corrected the relevant styled test harnesses to load production split CSS.

Verification: original renderer fails the three new/strengthened regressions (production-CSS negative control: 16/15 blank frames). Fixed targeted disclosure, edit-wiring, and assistant-turn suites: 74 passed; final thinking/edit rerun after test cleanup: 31 passed. Test files pass Ruff and whole-file formatting; production changed range passes formatting and adds zero Ruff findings against 27 pre-existing module findings. Full-suite and unrelated module-wide cleanup were not run. Self-review and independent review found no actionable issues.

Native UAT: real TldwCli at 140x45 with temporary SQLite profile and local llama.cpp/Gemma 4; two actual composer/Enter sends completed, 215 observed live thinking frames with zero blanking or body replacement. Automatic collapse, full trace reopen, and manual collapse/reopen during live streaming passed. A prior longer prompt had 458 clean frames but ended in provider HTTP 502; not counted as a successful end-to-end run. Evidence and screenshots: qa/thinking-flicker-32515/report.md.

ADR: backlog/decisions/090-console-thinking-block-ownership-and-replay.md applies unchanged; no new ADR required. Updated lessons-testing-evidence.md with the observed wrapper-versus-body verification trap. Production change: console_transcript.py; tests: test_console_thinking_disclosures.py and test_console_assistant_turn.py; QA artifacts under qa/thinking-flicker-32515.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered from TASK-32515 to TASK-32522 while preparing the PR against dev on
2026-09-13. The older ChaChaNotes index-census task (created 2026-09-11 20:21)
retains TASK-32515; this thinking task was created 2026-09-13 02:52. A scan of
176 local/remote ref tips and available worktrees found maximum ID 32521.
Historical UAT artifacts retain their original directory name
`qa/thinking-flicker-32515/` so existing evidence links remain usable.

PR verification on fresh dev base a3142cb356: all 74 targeted tests passed.
