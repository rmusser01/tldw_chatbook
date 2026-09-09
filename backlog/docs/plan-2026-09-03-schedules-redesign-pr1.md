# Schedules Redesign — PR-1: DetailValueRow + detail-pane regrammar (read-only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The redesign's first slice: one reusable `DetailValueRow` widget + collapsible groups, and both detail surfaces regrammared to the spec §5 per-family table — read-only (in-pane editing is PR-3). Definitions get their FIRST real detail pane (today the Automations tab has none, by its own in-source admission).

**Architecture:** A new reusable widget pair (`DetailValueRow`: label left, value right, dormant `▾` affordance + error-line slot built now, activated in PR-3; `DetailGroup`: titled collapsible container modeled on `ConsoleInspectorSection`'s pattern) styled through `$ds-*` tokens in `css/features/_scheduling.tcss`. `TaskDetail` (reminders) re-renders its field display through the new grammar while preserving every existing behavior (transfer actions/badges, incidents, run history, footer shortcuts). The Automations tab gains a definitions detail widget fed on row highlight, rendering the recurring-question column of the spec table from the JSON config seams plus two small DB count additions. The redesign spec itself is committed in this PR (program convention).

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite, pytest.

**Spec:** `backlog/docs/spec-2026-09-02-schedules-screen-redesign.md` §5 (+§9 copy rules) — committed by Task 5. Planning rulings (binding):
1. **Read-only-first is strict**: no new interactivity in PR-1 — the `▾` affordance renders dimmed/disabled, the error-line slot exists but never populates, no new bindings. History-group inline read/dismiss (spec §5 table) is deliberately deferred to PR-3 with the rest of editing; PR-1's History group shows last-run outcome, run count, unread count, and a "view results" pointer only.
2. **Spec-unknown resolved (survey-verified)**: a fired one-time reminder is queryable — `mark_reminder_dispatched` sets `enabled=0, next_run_at=NULL, last_status, last_run_at`; the fired-vs-user-disabled predicate is `enabled=0 AND next_run_at IS NULL AND last_run_at IS NOT NULL`. Recorded here for PR-2's Completed chip; nothing in PR-1 consumes it beyond the History group's "last fire" line (existing columns).
3. **ADR citation discipline**: dev carries an ADR-099 NUMBER COLLISION (`099-schedule-editor-shape.md` vs `099-persistent-terminal-session-runtime-boundary.md`). All references in this program cite by FILENAME (`ADR-099-schedule-editor-shape`); the renumbering belongs to the user's in-flight `docs/lesson-adr-number-collisions` branch — do NOT touch either ADR file.
4. **Widget home**: `tldw_chatbook/Widgets/detail_value_row.py` (the redesign intends reuse beyond scheduling); scheduling-specific composition stays in `UI/Screens/scheduling/`.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-redesign-pr1`, branch `feat/schedules-detail-regrammar` off current `origin/dev`. Never the main checkout; NEVER `git stash` (multiple violations last program; absolute); no pkill beyond own PIDs; `git --no-pager`; foreground pytest only; tmp_path DBs.
- NO schema migration. NO behavior changes to actions/sync — this is a rendering regrammar plus two read-only count seams.
- Survey with exact seams: `redesign-pr1-survey.md` in the SDD workspace.
- Diagnostics pin is a SCRIPT (`scripts/check_persistent_diagnostic_inventory.py --write` + commit the JSON) whenever ANY logger statement is added/moved/reworded — three implementers got this wrong last program.
- Census merge bar = COUNT parity with dev's CI Perf Guard run (local headroom masks regressions); new widget modules must not become boot-resident.
- CSS via `css/build_css.py` only (never hand-edit the bundle); style new pieces through `$ds-*` tokens (`css/core/_variables.tcss`) even though the legacy file mostly uses raw tokens — improve only what this PR touches.
- UI change ⇒ update `Docs/User_Guide/` schedules page.
- Commit trailer on every commit:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv
```

---

### Task 1: `DetailValueRow` + `DetailGroup` widgets

**Files:** Create `tldw_chatbook/Widgets/detail_value_row.py`; modify `css/features/_scheduling.tcss` (+ rebuild via `css/build_css.py`). Test: `Tests/UI/test_detail_value_row.py` (new).

**Interfaces (produced; Tasks 3/4 consume; PR-3 will extend):**
- `DetailValueRow(label: str, value: str | Text, *, affordance: bool = False, value_id: str | None = None)` — label left (muted), value right-aligned; `affordance=True` renders a dimmed `▾` (non-interactive in PR-1); `.update_value(value)` refreshes in place; a hidden error-line `Static` slot exists below (id derived from `value_id`), `.show_error(msg)` / `.clear_error()` implemented and tested but unused by PR-1 callers.
- `DetailGroup(title: str, *, collapsed: bool = False)` — titled container composing rows; collapsible via the house `Collapsible` idiom if one exists, else the `ConsoleInspectorSection` toggle pattern (read it first: `Widgets/Console/console_inspector_section.py`).
- Both style exclusively via `$ds-*` tokens; long values truncate with ellipsis, never wrap the row.

- [ ] TDD: rendered (painted) label/value assertions — the last program's lesson: never assert stored attributes a widget might ignore; affordance dimmed; update_value in place; error slot show/clear; group collapse toggle; ellipsis on overlong value. FAIL → implement → PASS → commit `feat(ui): DetailValueRow + DetailGroup widgets`.

### Task 2: Count seams

**Files:** Modify `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py`. Test: `Tests/Scheduling/test_scheduled_tasks_db.py`.

**Interfaces:** extend `count_unread_results(owner_id=None, definition_id: str | None = None)` (or the file's nearest count seam — read first; keep one seam, add the filter) and add `count_automation_runs(definition_id: str) -> int`. House conventions (closing() reads); datetime-safety not implicated (counts).

- [ ] TDD both seams incl. definition_id filtering across owners. FAIL → implement → PASS → commit `feat(scheduling): per-definition count seams`.

### Task 3: Reminder detail regrammar (`TaskDetail`)

**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/task_detail.py` (+`_scheduling.tcss` as needed). Tests: the existing task_detail/workbench test files (locate; update rendering assertions to painted-cell style where touched).

**Behavior:** the field display re-renders through `DetailGroup`s per the spec §5 reminder column — Details: `Runs on` (owner label, transfer badge text preserved as the row's value suffix when in-flight); Frequency: `Repeat` / `At` / `Timezone` / `Notifications` (derive display strings from the reminder's schedule the same way the current field rendering does — reuse its formatting helpers, do not re-derive); History (collapsed): `Last fire` (last_run_at + last_status), link line to run history. The body/prompt card stays. EVERYTHING ELSE IS PRESERVED VERBATIM: action buttons (edit/enable/disable/run-now/transfer/cancel/retry incl. disabled-with-reason), incidents inline, run-history section, footer shortcuts, message classes — the regrammar touches field DISPLAY only. Compact-width behavior unchanged.

- [ ] TDD: grouped rows render the same data the old display did (pin a representative reminder's rendered values); every pre-existing test in the touched files stays green (behavior-preservation is the gate); collapsed History expands. FAIL → implement → PASS → commit `feat(scheduling): reminder detail pane regrammar`.

### Task 4: Definitions detail pane (first one)

**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` (Automations tab: compose the new detail widget beside/below the table — read the tab's current layout and the in-source no-detail comment; feed it on row highlight, off-thread reads per the conflicts precedent), possibly a small `UI/Screens/scheduling/definition_detail.py` module (mirror how `results_tab.py` was split out). Tests: `Tests/UI/test_schedules_automations_tab.py` + new assertions.

**Behavior:** on definition-row highlight render: question text card; Details group — `Runs on` (owner label + transfer badge when in-flight), `Model` (pin or "Provider default" — reuse `automation_execution_target_label`), `Generation`, `Finding policy` (preset), `Sources` (joined plurals); Frequency group — schedule summary fields (kind-appropriate: repeat/at/timezone or interval or cron — reuse existing schedule-summary formatting); History group (collapsed) — last run outcome (from runs), run count (Task 2 seam), unread results count (Task 2 seam), "view results" pointer line. All values through the escape/Text discipline from last program (server-derived strings never hit a markup parser raw). Existing keybindings/actions untouched. Compact width: the detail hides with the existing responsive mechanism.

- [ ] TDD: highlight → painted values match a seeded definition (both owners; a server-mirrored row shows its server owner label); counts render; escape pinned with a bracket-bearing name; off-thread read discipline. FAIL → implement → PASS → commit `feat(scheduling): definitions detail pane`.

### Task 5: Spec commit + docs + gates

**Files:** Commit `backlog/docs/spec-2026-09-02-schedules-screen-redesign.md` (copied into the worktree; add a one-line errata note in its §10 citing the ADR-099 filename-collision discipline per ruling 3). Update `Docs/User_Guide/` schedules page (new detail-pane look, per-family rows). Gates.

- [ ] Full `Tests/Scheduling/ -q`; all schedules UI files + the two new test files; census (COUNT parity vs dev's latest CI Perf Guard number — record both); diagnostics pin; ruff; CSS bundle reproduces. Commit `docs(scheduling): redesign spec + user guide for the detail regrammar`.

---

## After the tasks
Final whole-branch review (opus, behavior-preservation + escape-discipline lenses) → one fix wave → PR `feat(scheduling): detail-pane regrammar — DetailValueRow groups (redesign PR-1)` → paged bot-comment read → adjudicate → sequential rebase-watch-merge (stale-read grace; count-parity check on any census failure).
