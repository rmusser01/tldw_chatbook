# Inspect Rail Critique Burn-Down Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Each task's requirements live in its backlog file (the brief); this plan carries order, constraints, and code landmarks.

**Goal:** Burn down the 2026-09-05 UX critique of the Console Inspect rail (18/40): tasks 31660-31665, owner-ranked state-honesty first.

**Spec:** the critique snapshot `.impeccable/critique/2026-09-05T06-46-15Z__tldw-chatbook-ui-console-modules-right-rail-py.md` (main checkout) + each task's ACs. The task file is the binding brief; this plan's landmarks are advisory.

## Global Constraints

- Base `origin/dev @ 7e904737c7` (the Environment panel merge). Branch `feat/inspect-rail-critique-burndown`. PR to dev at arc end.
- TDD per task; both-seams tests (projection AND wiring) wherever a screen behavior changes; deferred-fake style for controller cadence work (`Tests/UI/test_console_environment_controller.py` shows the pattern).
- Foreground pytest only (`> .pytest-out.txt 2>.pytest-warn.txt` + grep; never commit these). NEVER `git stash`. No subagents from implementers.
- CSS edits in `css/components/_agentic_terminal.tcss` only; regenerate via `.venv/bin/python -m tldw_chatbook.css.build_css`; both-halves rule: any style change to a widget pair (open/collapsed, inline/CSS) must touch both halves.
- Owner rulings that BIND: the `Local` row stays as designed (31662 must not cut it); state-honesty cluster ships first.
- Focus work: decide synchronously at call sites; never a flag around `focus()`; TASK-24702 prior art — a background tint CANNOT clear 3:1 on this theme, use shape/glyph carriers.
- Known pre-existing failure clusters (baseline before blaming a diff): reaction_picker, parallel_runs, inspector_compact_access, workbench_contract, live_work_handoffs, roleplay_writer, agent_rail, fleet_survivor_tick, console_workspace_reconcile.
- `.impeccable/` is TRACKED — never delete/modify it.
- Backlog hygiene: tick ACs + Implementation Notes per task; preflight before the PR.

## Task order and landmarks

### Task 1 = TASK-31660 (high): UNBOUND/PENDING state honesty
Brief: `backlog/tasks/task-31660 - Environment-panel-state-honesty-UNBOUND-and-PENDING-states.md`
Landmarks: controller `tldw_chatbook/UI/Console_Modules/environment.py` — `poll_tick`/`request_refresh` early-return when `workspace_root_accessor()` is None (that early return is the bug: land an explicit state instead). `EnvSourceAvailability` + projections in `tldw_chatbook/Chat/console_environment_state.py` (ERROR already has its own row; add UNBOUND + PENDING renderings; PENDING may alternatively keep the section hidden). Copy source: Change Review's unbound-workspace copy in `change_review_screen.py` ("No folder is bound… not a report that nothing changed"). Wiring: `_land_console_environment` in `chat_screen.py`. Cold-start: the default `EnvironmentSnapshot()` must project as PENDING, not "No git workspace" — NOT_APPLICABLE keeps meaning "checked: not a repo".

### Task 2 = TASK-31661 (high): poll must not steal focus
Brief: `backlog/tasks/task-31661 - Environment-poll-must-not-steal-rail-focus.md`
Landmarks: `_land_console_environment` (chat_screen) syncs both sections; the activation path already restores by row_id via `_request_console_environment_row_focus`/`_focus_console_environment_row` — reuse that pair on the sync path (capture focused row_id across `sync_state`, restore if it survives; nearest-surviving-row fallback per AC).

### Task 3 = TASK-31662: single-line rows + density
Brief: `backlog/tasks/task-31662 - Inspect-rail-single-line-rows-and-density.md`
Landmarks: `Widgets/Console/console_inspector_section.py` — `ConsoleInspectorSectionRow` `min_height = 2` and always-mounted secondary Static; CSS `.console-inspector-section-row*` (~L5030). Consumers to sweep before changing row geometry: fleet section tests, `Tests/UI/test_console_inspector_section.py` (its geometry assertions), any `render_line(0/1)` pins. Header-summary suppression while open lives in the section widget (it knows `open`); Tasks counts-row removal in `project_tasks_section`. Budgets: rail content width is 30 (80×24) to 36 (200×50) — derive, don't assume 34 (31629 #12/#13 context).

### Task 4 = TASK-31663: focus visibility + reachability
Brief: `backlog/tasks/task-31663 - Inspect-rail-focus-visibility-and-reachability.md`
Landmarks: rows carry corner-bracket focus (shape — good); buttons/chevrons are tint-only (`.console-inspector-section-toggle:focus`, view-all, etc.) — give them a glyph/shape carrier. The indication-free Tab stop is the send-authority summary block (stop 3 after Alt+I). Scrollbar thumb: section body scrollbar colors in the TCSS. Tab-route breaker: a hidden-but-focusable left-rail widget ("Review changes", renders blank) absorbs Tab — fix display/focusability coupling or file with evidence per AC #3.

### Task 5 = TASK-31664: affordances + consequence legibility
Brief: `backlog/tasks/task-31664 - Environment-panel-affordance-and-consequence-legibility.md`
Landmarks: all row copy in `console_environment_state.py` (markers are projection-side); rename `Commit or push` → `Review & commit… · N files` (sweep tests for the literal); Refresh acknowledgment: ViewAllRequested handler in chat_screen + a transient row or label state (must not flicker the 10s poll); stale text marker in `_git_status_class` call sites; `$text-error` vs `$ds-status-error` in the row status CSS.

### Task 6 = TASK-31665: minors batch + banding investigation
Brief: `backlog/tasks/task-31665 - Inspect-rail-critique-minor-batch-and-banding.md`
Landmarks: INVESTIGATE FIRST (AC #1): the `#2d2d2d` band originates in the left rail (cols 20-32) and bleeds to col 233 — find the widget/rule painting it (suspect: a full-width row/container background in left_rail or the workspace shell) and fix or document; AC #9 depends on it. Then the mechanical minors (frontmatter titles: `environment_status.py` `_TASK_FILENAME_RE` group-2 → parse title from the frontmatter already in hand; child indent via a new `InspectorSectionRow.indent` field; glyph unification; Refresh attachment/tooltip; vocab; Change Review plural + transient flash; canonical-opener decision may be a documented ruling rather than code).
