# Schedules Redesign — PR-2: Unified list, filter chips, rail chrome

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Queue tab becomes the redesign's unified surface: one list of reminders AND recurring-question definitions across both owners, filtered by the four chips, searched in-memory, ticked every 60s, with the rail chrome (Create ▾, Mark-all-read, bottom status strip) around it. Read-mostly: definition rows are viewable with the PR-1 detail pane; their actions stay on the Automations tab until PR-4 retires it.

**Architecture:** A pure row-adapter module (`unified_rows.py`) turns reminder + definition rows into one `UnifiedRow` shape — status bucket (the chip predicates, including the verified fired-one-time predicate and PR-5's transfer-arming semantics), glyph, schedule summary, relative next-run, owner, unread count (derived in Python from ONE all-owners results listing routed through the dual local/server id-space resolution — never N queries, never a naive join). The Queue `DataTable` renders those rows; highlight routes between the two PR-1 detail panes by toggling the existing `pane-hidden` class (they become siblings in the Queue detail area). The cross-owner reminder seam is one parameter thread (the DB layer already supports `owner_id=None`, unused). The rail relocates the existing Create chooser, adds search + Mark-all-read, and a bottom strip hosting the sync widget (new width-compact path) + a conflicts badge that switches to the Conflicts tab (the overlay is PR-4 polish). The 60s ticker clones the file's own `set_interval` pause/resume idiom with `on_screen_suspend`/`on_screen_resume`.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite, pytest.

**Spec:** `backlog/docs/spec-2026-09-02-schedules-screen-redesign.md` §3/§4 (tracked on dev since PR-1). Planning rulings (binding):
1. **Morph-in-place**: the Queue tab IS the unified surface; Automations/Conflicts/Results tabs coexist untouched until PR-4. Definition rows in the unified list are viewable (highlight → PR-1's `DefinitionDetail`); NO new definition-action wiring in PR-2 — reminder rows keep every existing behavior verbatim.
2. **Chip mapping** (spec §3 + PR-5 semantics): Active = enabled reminders + `configured` definitions, INCLUDING `to_server_pending`/`to_server_failed` rows (reuse the armable-filter semantics — `DORMANT_TRANSFER_STATES` is the authority, do not restate the state list); Paused = disabled-but-not-fired reminders + `paused` definitions; Completed = fired one-time reminders (`enabled=0 AND next_run_at IS NULL AND last_run_at IS NOT NULL` — name it `reminder_has_fired(row)` where the bucket logic lives) + `archived` definitions; All = Active + Paused. `disabled`-lifecycle definitions bucket as Paused with their lock reason preserved for the detail pane.
3. **Unread derivation**: one all-owners `list_automation_results` call per refresh; counts grouped in Python and resolved through the SAME dual id-space logic as `results_tab.py`'s `index_definitions_by_id`/`definition_for_result` — hoist those helpers to a shared module if importing them would cycle (check direction first). A definition row's dot shows when its resolved unread count > 0.
4. **Status strip minimalism**: the strip hosts the EXISTING `SyncStatusWidget` under a new compact styling path + the owner indicator it already contains + a conflicts badge chip (existing count semantics — single-owner reminder conflicts — with the count's scope stated in the tooltip) that SWITCHES to the Conflicts tab. No overlay, no new conflict queries.
5. **Sort**: `next_run_at` ascending (None last) within Active; most-recent-first (`last_run_at`/`updated_at` desc) for Paused/Completed; the All chip keeps Active's order for armed rows then appends Paused by recency. Search = case-insensitive substring over title + question/body, in-memory.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-redesign-pr2`, branch `feat/schedules-unified-list` off current `origin/dev`. Never the main checkout; NEVER `git stash`; no pkill beyond own PIDs; `git --no-pager`; FOREGROUND pytest only; tmp_path DBs.
- NO schema migration; NO sync/action behavior changes — listing, filtering, rendering, routing only (plus the one service parameter thread).
- Survey with exact seams: `redesign-pr2-survey.md` in the SDD workspace.
- Diagnostics pin is a SCRIPT (`scripts/check_persistent_diagnostic_inventory.py --write` + commit JSON) on any logger change. Census merge bar = COUNT parity with dev's CI (watch new imports on the workbench path). CSS via the build flow, `$ds-*` tokens, source+bundle together. Geometry/paint tests need `CSS_PATH = BUNDLED_STYLESHEET` (documented trap). Painted-output assertions only. `DetailGroup(title=...)` is keyword-only.
- UI change ⇒ `Docs/User_Guide/` schedules page update.
- Commit trailer on every commit:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv
```

---

### Task 1: Row adapter + cross-owner seam (pure core)

**Files:** Create `tldw_chatbook/UI/Screens/scheduling/unified_rows.py` (pure — no Textual imports). Modify `tldw_chatbook/Scheduling/services/scheduling_service.py` (`list_tasks` gains `owner_id: str | None | EllipsisType`-style opt-in spans-owners parameter — read the current signature and pick the least-invasive shape; default preserves every caller), and hoist the id-space helpers if ruling 3's cycle check demands it. Tests: `Tests/Scheduling/test_unified_rows.py` (new), `test_scheduling_service.py`.

**Interfaces (produced; Task 2 consumes):**
- `@dataclass UnifiedRow`: `kind` ("reminder"|"definition"), `row_id`, `title`, `schedule_summary`, `next_run_at`, `owner_id`, `owner_label`, `transfer_state`, `bucket` ("active"|"paused"|"completed"), `glyph`, `unread_count`, `search_blob`, `source_row` (the original dict/task).
- `build_unified_rows(reminders: list, definitions: list[dict], results: list[dict]) -> list[UnifiedRow]` — buckets per ruling 2 (`reminder_has_fired` defined here; definition arming defers to `DORMANT_TRANSFER_STATES` + lifecycle), glyphs per spec §4 (`○` recurring, `▶` one-shot/monitor, `⏸` paused, `✓` completed), schedule summaries via the EXISTING formatting helpers (import, don't re-derive), owner labels via `owner_display_label`, unread via ruling 3's Python grouping.
- `filter_rows(rows, *, chip: str, query: str) -> list[UnifiedRow]` and `sort_rows(rows, chip)` per ruling 5.

- [ ] TDD: bucket table (every chip × both primitives × transfer states incl. pending/failed-stay-active and the fired predicate); unread resolution across BOTH id spaces (a server-mirrored definition's results counted — the survey's warning is the test); sort orders; search; spans-owners `list_tasks` (existing callers pinned unchanged). FAIL → implement → PASS → commit `feat(scheduling): unified row adapter + cross-owner listing`.

### Task 2: The unified Queue surface

**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` (Queue tab: chips row above the table; the DataTable renders `UnifiedRow`s — glyph/title/subtitle columns per spec §4 with the existing owner-suffix and transfer-badge rendering preserved; unread dot; chip + search filtering; `load_tasks` becomes the unified loader calling Task 1 with the three listings — reminders spans-owners, both definition halves via the existing merge precedent, one results listing; detail routing: compose `DefinitionDetail` as a sibling of `TaskDetail` in the Queue detail area, toggle via the `pane-hidden` class on highlight by row kind; reminder-row behavior byte-preserved — every existing binding/action/test), CSS as needed. Tests: `Tests/UI/test_schedules_workbench.py` + a new `Tests/UI/test_schedules_unified_list.py`.

- [ ] TDD: mixed listing renders both kinds (painted); chips filter (each chip's bucket, incl. a fired reminder under Completed and a to_server_pending row under Active); search narrows; highlight routes to the right detail pane both directions; reminder actions all still fire (the existing test suite is the preservation gate — zero unrelated assertion changes allowed); definition rows expose NO actions. FAIL → implement → PASS → commit `feat(scheduling): unified queue list with filter chips`.

### Task 3: Rail chrome + status strip

**Files:** Modify `schedules_workbench.py` (rail header: relocate the existing "+ New" chooser as `Create ▾`; search input wiring; `Mark all as read` action visible only when total unread > 0, reusing the Results tab's per-row mutation fan-out; bottom status strip: the existing `SyncStatusWidget` restyled compact for the strip (new width-compact CSS path — the survey says one compact path exists for local-only; add the width-triggered one), conflicts badge chip (existing count, scope in tooltip, click switches to the Conflicts tab)), `sync_status_widget.py` (the compact path), CSS. Tests: workbench + a small sync-widget compact test.

- [ ] TDD: chooser reachable from the rail (both create paths still open their modals); Mark-all-read visibility + action; strip renders compact; badge switches tabs. FAIL → implement → PASS → commit `feat(scheduling): rail chrome and status strip`.

### Task 4: Ticker + docs + gates

**Files:** Modify `schedules_workbench.py` (60s `set_interval` cloning the file's own pause/resume-with-immediate-refresh idiom — the survey's line refs; suspend on `on_screen_suspend`, resume+refresh on `on_screen_resume`; tick updates ONLY the visible rows' relative next-run text, no full reload), `Docs/User_Guide/` schedules page (unified list, chips, rail). Gates.

- [ ] TDD: tick updates relative text without a reload (fake clock or injected now — read how schedule_compute tests inject time); suspend/resume behavior; then FULL gates — `Tests/Scheduling/ -q`, all schedules UI files, census (count parity vs dev CI), pin script, ruff, bundle byte-identical. Commit `feat(scheduling): relative-time ticker + docs`.

---

## After the tasks
Final whole-branch review (opus; preservation + bucket-correctness + id-space lenses) → one fix wave → PR `feat(scheduling): unified schedules list + filter chips + rail (redesign PR-2)` → paged bot read → adjudicate → rebase-watch with `gh pr merge` INSIDE the 30s loop on CLEAN (auto-merge is disabled repo-wide; exit-then-merge loses the race).
