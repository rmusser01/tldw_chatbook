# Home Triage Rail (H1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Home screen as a triage surface — rail sections of selectable rows (Needs Attention / Running / Recent / Details) driving a focus canvas with the existing control actions — per spec §1.

**Architecture:** Pure state first: `Home/dashboard_state.py` gains row/canvas builders alongside the existing (temporarily retained) line-blob builder; a new `Home/home_rail_state.py` holds rail prefs (mirroring `console_rail_state`). New `Widgets/Home/` package renders rail rows and the canvas; `home_screen.py` becomes orchestration only. Reuses shipped Console primitives: `ConsoleRailSectionHeader`, `format_console_relative_age`, the `save_setting_to_cli_config` prefs pattern.

**Tech Stack:** Python 3.11+, Textual, pytest + pytest-asyncio, existing `_build_test_app` UI harness.

**Spec:** `Docs/superpowers/specs/2026-07-04-home-library-redesign-design.md` §1, §3–§5 (H1). Resolved decisions §6 bind: `Open in Console` primary for console-origin items; per-source `detail_route`; limited inline preview allowed; proceed now, inherit Console P3/P4 conventions later.

## Global Constraints

- Run tests with: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q <target> --tb=short`
- The `timeout` shell command is not available.
- `tldw_chatbook/css/tldw_cli_modular.tcss` is GENERATED: edit `tldw_chatbook/css/components/_agentic_terminal.tcss` (or the Home component file if one exists — check `css/components/` first), run `./build_css.sh`, commit both.
- Rail conventions (binding, from the Console): section headers via `ConsoleRailSectionHeader` (existing widget, `-`/`+` glyphs), `▸` marker on the selected row, age labels via `format_console_relative_age`, counters dimmed at zero and bright otherwise, `Details` collapsed by default.
- Section titles exactly: `Needs Attention`, `Running`, `Recent`, `Details`. Header line format exactly: `Home | {Ready|Blocked} · {Local|Server: <label>}`.
- Rail prefs persisted under `home.rail_state` (global scope, one key `sections`), defaults: attention/running/recent open, details collapsed.
- Existing behavior contracts that MUST survive: every `HOME_CONTROL_METHODS` action still reachable (now from the canvas action row) with identical `app_instance` method dispatch and kwargs; `home-primary-action` next-best-action navigation unchanged; `NavigateToScreen` routing unchanged.
- **Stage only files you changed (`git add <specific paths>`). NEVER `git add -A`, `git add .`, or `git commit -a`.** Never touch `.claude/settings.local.json`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Home screen changes require live screenshot QA + explicit user approval before merge (Task 7).
- Work in the `claude/home-library-redesign` branch/worktree.

## File Structure

- Modify `tldw_chatbook/Home/dashboard_state.py` — add row/canvas pure builders (keep existing exports working until Task 4 removes the screen's use of line-blob sections; do not delete `summarize_home_dashboard` in this plan — other callers may exist; verify in Task 4).
- Modify `tldw_chatbook/Home/active_work_adapter.py` — recent-work items.
- Create `tldw_chatbook/Home/home_rail_state.py` — rail preferences (pure).
- Create `tldw_chatbook/Widgets/Home/__init__.py`, `home_rail.py`, `home_canvas.py`.
- Modify `tldw_chatbook/UI/Screens/home_screen.py` — compose rework, selection dispatch, prefs wiring.
- Tests: extend `Tests/Home/test_dashboard_state.py`, `Tests/Home/test_active_work_adapter.py`; create `Tests/Home/test_home_rail_state.py`, `Tests/UI/test_home_triage_rail.py`.

---

### Task 1: Pure row/canvas state builders

**Files:**
- Modify: `tldw_chatbook/Home/dashboard_state.py`
- Test: `Tests/Home/test_dashboard_state.py`

**Interfaces:**
- Consumes: existing `HomeDashboardInput`, `HomeActiveWorkItem`, `categorize_run_status`, `choose_next_best_action`, `build_home_controls`, `_first_item_for_status` internals.
- Produces (Tasks 3–4 rely on exact names):
  - `HomeActiveWorkItem` gains `updated_at: str = ""` (ISO text; blank → blank age label).
  - `@dataclass(frozen=True) HomeRailRow(row_id: str, section_id: str, glyph: str, title: str, age_label: str, source: str = "", status_category: str = "", detail_route: str = "chat")`
  - `@dataclass(frozen=True) HomeRailSectionState(section_id: str, title: str, count: int, rows: tuple[HomeRailRow, ...], empty_copy: str)`
  - `@dataclass(frozen=True) HomeCanvasState(title: str, lines: tuple[str, ...], actions: tuple[HomeControl, ...], next_action: HomeAction, next_action_is_canvas: bool)`
  - `@dataclass(frozen=True) HomeTriageState(header_line: str, sections: tuple[HomeRailSectionState, ...], details_lines: tuple[str, ...], canvas: HomeCanvasState, selected_row_id: str)`
  - `build_home_triage_state(state: HomeDashboardInput, *, selected_row_id: str = "", now: datetime | None = None) -> HomeTriageState`

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Home/test_dashboard_state.py`:

```python
from datetime import datetime, timezone

from tldw_chatbook.Home.dashboard_state import (
    HomeActiveWorkItem,
    HomeDashboardInput,
    build_home_triage_state,
)

_NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


def _items_input(**overrides) -> HomeDashboardInput:
    defaults = dict(
        model_ready=True,
        active_work_items=(
            HomeActiveWorkItem(
                item_id="wf:approve-1",
                title="Approval: publish chatbook",
                source="Workflows",
                status="pending_approval",
                detail_route="workflows",
                console_available=True,
                updated_at="2026-07-04T11:57:00+00:00",
            ),
            HomeActiveWorkItem(
                item_id="watch:run-1",
                title="Watchlist sweep",
                source="Watchlists",
                status="running",
                detail_route="watchlists",
                updated_at="2026-07-04T12:00:00+00:00",
            ),
            HomeActiveWorkItem(
                item_id="sched:fail-1",
                title="Retry: ingest failure",
                source="Schedules",
                status="failed",
                detail_route="schedules",
                updated_at="2026-07-04T11:00:00+00:00",
            ),
        ),
    )
    defaults.update(overrides)
    return HomeDashboardInput(**defaults)


def test_triage_sections_split_by_status_with_ages():
    triage = build_home_triage_state(_items_input(), now=_NOW)
    by_id = {section.section_id: section for section in triage.sections}
    attention = by_id["attention"]
    running = by_id["running"]
    assert attention.title == "Needs Attention"
    assert attention.count == 2  # approval + failed
    titles = [row.title for row in attention.rows]
    assert "Approval: publish chatbook" in titles
    assert "Retry: ingest failure" in titles
    approval_row = next(r for r in attention.rows if r.row_id == "wf:approve-1")
    assert approval_row.age_label == "3m"
    assert approval_row.glyph == "●"
    assert running.count == 1
    assert running.rows[0].age_label == "now"


def test_triage_header_line_formats():
    triage = build_home_triage_state(_items_input(), now=_NOW)
    assert triage.header_line == "Home | Ready · Local"
    blocked = build_home_triage_state(
        _items_input(model_ready=False, runtime_source="server", server_label="lab"),
        now=_NOW,
    )
    assert blocked.header_line == "Home | Blocked · Server: lab"


def test_triage_default_selection_prefers_attention_and_builds_canvas():
    triage = build_home_triage_state(_items_input(), now=_NOW)
    assert triage.selected_row_id == "wf:approve-1"
    assert triage.canvas.title == "Approval: publish chatbook"
    action_ids = [a.control_id for a in triage.canvas.actions]
    assert "home-approve" in action_ids and "home-reject" in action_ids
    assert triage.canvas.next_action_is_canvas is False


def test_triage_explicit_selection_and_missing_age_blank():
    triage = build_home_triage_state(
        _items_input(
            active_work_items=(
                HomeActiveWorkItem(
                    item_id="x:1",
                    title="No timestamp item",
                    source="ACP",
                    status="running",
                ),
            )
        ),
        selected_row_id="x:1",
        now=_NOW,
    )
    assert triage.selected_row_id == "x:1"
    assert triage.sections[1].rows[0].age_label == ""


def test_triage_empty_input_makes_next_action_the_canvas():
    triage = build_home_triage_state(HomeDashboardInput(model_ready=True), now=_NOW)
    assert all(section.count == 0 for section in triage.sections[:2])
    assert triage.canvas.next_action_is_canvas is True
    assert triage.canvas.next_action.label
    by_id = {s.section_id: s for s in triage.sections}
    assert by_id["attention"].empty_copy == "No approvals or failures pending."
    assert triage.details_lines  # system status relocated here
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q Tests/Home/test_dashboard_state.py -k triage --tb=short`

Expected: FAIL with `ImportError: cannot import name 'build_home_triage_state'`.

- [ ] **Step 3: Implement**

In `dashboard_state.py`:

1. Add `from datetime import datetime, timezone` and `from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age` to imports.
2. Add `updated_at: str = ""` to `HomeActiveWorkItem` (after `console_available`).
3. Add the four dataclasses from the Interfaces block verbatim.
4. Add glyph mapping and builder:

```python
_CATEGORY_GLYPHS = {
    APPROVAL_RUN_STATUS: "●",
    FAILED_RUN_STATUS: "●",
    PAUSED_RUN_STATUS: "○",
    RUNNING_RUN_STATUS: "●",
    UNKNOWN_RUN_STATUS: "○",
}
_ATTENTION_CATEGORIES = frozenset({APPROVAL_RUN_STATUS, FAILED_RUN_STATUS})
_RUNNING_CATEGORIES = frozenset({RUNNING_RUN_STATUS, PAUSED_RUN_STATUS})


def _item_row(item: HomeActiveWorkItem, section_id: str, now: datetime) -> HomeRailRow:
    category = categorize_run_status(item.status)
    return HomeRailRow(
        row_id=item.item_id,
        section_id=section_id,
        glyph=_CATEGORY_GLYPHS.get(category, "○"),
        title=item.title,
        age_label=format_console_relative_age(item.updated_at, now=now),
        source=item.source,
        status_category=category,
        detail_route=item.detail_route,
    )


def _header_line(state: HomeDashboardInput) -> str:
    readiness = "Ready" if state.model_ready else "Blocked"
    source = str(state.runtime_source or RUNTIME_SOURCE_LOCAL).strip().lower()
    if source == RUNTIME_SOURCE_SERVER:
        label = str(state.server_label or "").strip()
        runtime = f"Server: {label}" if label else "Server"
    else:
        runtime = "Local"
    return f"Home | {readiness} · {runtime}"


def build_home_triage_state(
    state: HomeDashboardInput,
    *,
    selected_row_id: str = "",
    now: datetime | None = None,
) -> HomeTriageState:
    """Build the Home triage rail + canvas display state.

    Args:
        state: Adapter-provided dashboard input.
        selected_row_id: Explicit row selection; falls back to control
            priority (approval > failed > running > paused > first).
        now: Reference time for age labels (defaults to UTC now).

    Returns:
        Immutable triage state: header line, rail sections, details lines,
        and the canvas for the selected row (or the next best action when
        nothing is selectable).
    """
    reference_now = now or datetime.now(timezone.utc)
    attention_rows: list[HomeRailRow] = []
    running_rows: list[HomeRailRow] = []
    for item in state.active_work_items:
        category = categorize_run_status(item.status)
        if category in _ATTENTION_CATEGORIES:
            attention_rows.append(_item_row(item, "attention", reference_now))
        elif category in _RUNNING_CATEGORIES:
            running_rows.append(_item_row(item, "running", reference_now))
    recent_rows = tuple(
        _item_row(item, "recent", reference_now) for item in state.recent_work_items
    )

    sections = (
        HomeRailSectionState(
            "attention",
            "Needs Attention",
            len(attention_rows),
            tuple(attention_rows),
            "No approvals or failures pending.",
        ),
        HomeRailSectionState(
            "running",
            "Running",
            len(running_rows),
            tuple(running_rows),
            "Nothing running right now.",
        ),
        HomeRailSectionState(
            "recent",
            "Recent",
            len(recent_rows),
            recent_rows,
            "Runs, chatbooks, imports, and schedules will appear here.",
        ),
    )

    all_rows = {row.row_id: row for section in sections for row in section.rows}
    selected = all_rows.get(selected_row_id)
    if selected is None:
        fallback_item = choose_home_selected_item(state)
        selected = all_rows.get(fallback_item.item_id) if fallback_item else None
    next_action = choose_next_best_action(state)
    if selected is not None:
        item = next(
            i
            for i in tuple(state.active_work_items) + tuple(state.recent_work_items)
            if i.item_id == selected.row_id
        )
        canvas = HomeCanvasState(
            title=item.title,
            lines=(
                f"Source: {item.source} · Status: {item.status}",
                f"{selected.glyph} {selected.status_category or 'item'}"
                + (f" since {selected.age_label}" if selected.age_label else ""),
                f"Route: {item.detail_route}",
            ),
            actions=build_home_controls(state),
            next_action=next_action,
            next_action_is_canvas=False,
        )
        selected_id = selected.row_id
    else:
        canvas = HomeCanvasState(
            title=next_action.label,
            lines=(next_action.reason,),
            actions=(),
            next_action=next_action,
            next_action_is_canvas=True,
        )
        selected_id = ""

    return HomeTriageState(
        header_line=_header_line(state),
        sections=sections,
        details_lines=_system_status_lines(state),
        canvas=canvas,
        selected_row_id=selected_id,
    )
```

5. Add `recent_work_items: tuple[HomeActiveWorkItem, ...] = ()` to `HomeDashboardInput` (after `active_work_items`) — Task 2 populates it; default keeps every existing caller/test working.

- [ ] **Step 4: Run tests to verify they pass**

`env HOME=... .venv/bin/python -m pytest -q Tests/Home/test_dashboard_state.py --tb=short` — new tests PASS, all pre-existing tests in the file still PASS (all additions are default-valued).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Home/dashboard_state.py Tests/Home/test_dashboard_state.py
git commit -m "feat(home): pure triage rail/canvas state builders"
```

---

### Task 2: Recent-work items from the adapter

**Files:**
- Modify: `tldw_chatbook/Home/active_work_adapter.py`
- Test: `Tests/Home/test_active_work_adapter.py`

**Interfaces:**
- Consumes: Task 1's `HomeDashboardInput.recent_work_items` and `HomeActiveWorkItem.updated_at`.
- Produces: the concrete adapter(s) in `active_work_adapter.py` populate `recent_work_items` (most recent first, cap 8) and `updated_at` on active items where source data carries timestamps.

- [ ] **Step 1: Read the adapter and write failing tests**

Read `tldw_chatbook/Home/active_work_adapter.py` (211+ lines; `HomeActiveWorkAdapter` Protocol at line 91, `UnavailableHomeActiveWorkAdapter` at 114, and the app-backed implementation below it). Identify where `active_work_items` is assembled and which source records expose timestamps (run records / chatbook artifacts / schedule entries). Then, in `Tests/Home/test_active_work_adapter.py`, extend the existing adapter test pattern (the file already builds fake app inputs) with:
- a test asserting items built from records with timestamps carry `updated_at` (ISO text passthrough — no reformatting);
- a test asserting completed/terminal records appear in `recent_work_items` (recent-first, capped at 8) and NOT in `active_work_items`;
- a test asserting `UnavailableHomeActiveWorkAdapter.build_dashboard_input(...)` still returns `recent_work_items == ()`.
Exact assertions must follow the fixtures already present in that file — mirror its existing test structure (this is the one task where the plan cannot pre-write literal test bodies; the fixtures are file-specific. Everything else about the contract above is exact.)

- [ ] **Step 2: Run to verify failures** — same env pattern, `-k "recent or updated_at"`. Expected: FAIL (fields unpopulated).

- [ ] **Step 3: Implement** — thread timestamps and terminal records through the concrete adapter; do not touch the Protocol beyond documenting the new fields (they're carried by `HomeDashboardInput`, which is already the Protocol's return type).

- [ ] **Step 4: Run the adapter + dashboard-state suites** — both fully green.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Home/active_work_adapter.py Tests/Home/test_active_work_adapter.py
git commit -m "feat(home): adapter supplies recent work items and activity timestamps"
```

---

### Task 3: Rail preferences module

**Files:**
- Create: `tldw_chatbook/Home/home_rail_state.py`
- Test: `Tests/Home/test_home_rail_state.py` (new)

**Interfaces:**
- Produces: `HOME_RAIL_SECTION_IDS = ("attention", "running", "recent", "details")`; `@dataclass(frozen=True) HomeRailPreferences(attention_open: bool = True, running_open: bool = True, recent_open: bool = True, details_open: bool = False)`; `coerce_home_rail_preferences(raw: Any) -> HomeRailPreferences`; `serialize_home_rail_preferences(prefs) -> dict[str, bool]`. Mirrors `console_rail_state`'s `_coerce_bool` semantics exactly (bools/ints/strings; missing → default).

- [ ] **Step 1: Failing tests** — round-trip, missing-key defaults, `"off"`-string falsiness, unknown input → defaults (copy the structure of `Tests/Chat/test_console_rail_state.py`'s section-pref tests, adapted to the four Home fields).
- [ ] **Step 2: Verify RED** (ImportError).
- [ ] **Step 3: Implement** — copy the coerce/serialize pattern from `tldw_chatbook/Chat/console_rail_state.py` (`_coerce_bool`, `_TRUE_STRINGS`/`_FALSE_STRINGS`) into the new module with the four Home fields. Do not import from console_rail_state (screen-independence; the pattern is 30 lines).
- [ ] **Step 4: Verify GREEN.**
- [ ] **Step 5: Commit** `feat(home): rail section preferences`.

---

### Task 4: Widgets + screen rework

**Files:**
- Create: `tldw_chatbook/Widgets/Home/__init__.py`, `tldw_chatbook/Widgets/Home/home_rail.py`, `tldw_chatbook/Widgets/Home/home_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/home_screen.py`
- Test: `Tests/UI/test_home_triage_rail.py` (new)

**Interfaces:**
- Consumes: Tasks 1–3 states; `ConsoleRailSectionHeader`/`CONSOLE_RAIL_SECTION_TOGGLE_PREFIX` from `tldw_chatbook.Widgets.Console.console_rail_section` (reused as-is — it is screen-agnostic chrome).
- Produces:
  - `HomeRail(Vertical)` (`#home-rail`): per section — `ConsoleRailSectionHeader(title-with-count, section_id=f"home-{id}", open=…)`, body `#home-rail-body-{id}` containing row `Button`s id `home-row-{row_id-sanitized}` classes `home-rail-row` (+`home-rail-row-selected`), text `f"{'▸' if selected else ' '} {row.glyph} {row.title}"` + right-aligned age; `sync_state(triage, prefs)` recompose pattern; posts nothing itself (screen dispatches by id prefix `home-row-`).
  - `HomeCanvas(Vertical)` (`#home-canvas`): title Static `#home-canvas-title`, line Statics `#home-canvas-lines`, action-row of `HomeActionButton`s (existing class, ids preserved: `home-approve`, `home-reject`, `home-pause`, `home-resume`, `home-retry`, `home-open-details`, `home-open-in-console`, `home-open-chatbook-details`, `home-open-chatbook-in-console`), next-action callout Static `#home-next-action-callout` + `#home-primary-action` button (id preserved); `sync_state(canvas_state)`.
  - Screen: `compose_content` yields one header Static `#home-header-line`, then `Horizontal(#home-triage-grid)` with `HomeRail` + `HomeCanvas`; instance `self._home_selected_row_id: str = ""`; `on_button_pressed` branches: `home-row-` prefix → set selection, rebuild triage state, `rail.sync_state` + `canvas.sync_state`; `CONSOLE_RAIL_SECTION_TOGGLE_PREFIX + "home-"` prefix → flip pref, persist via a `@work(thread=True)` `save_setting_to_cli_config("home.rail_state", "sections", serialized)` mirror of the Console pattern, toggle body display + `header.sync_open`; existing control ids → the UNCHANGED `_activate_home_control` / `_activate_home_primary_action` paths.
  - Old ids removed: the four stacked header rows, `home-dashboard-grid` panes, `home-followup-row` (their tests updated in this task).

- [ ] **Step 1: Failing pilot tests** in `Tests/UI/test_home_triage_rail.py` (harness: `_build_test_app` + the Home screen the way `Tests/UI/test_destination_shells.py` reaches screens; set `app._home_dashboard_test_input` — the screen already honors this override at `home_screen.py:117` — to a `HomeDashboardInput` with the Task 1 fixture items):
  1. rail renders three open sections with counts in titles + Details collapsed; attention row shows `▸` and age;
  2. clicking a running row switches canvas title and selected class;
  3. canvas action buttons dispatch: press `home-approve` → assert the app-instance method stub was called (the harness app records calls — follow the existing pattern in old Home tests, grep `approve_active_home_item` in Tests/);
  4. empty input → canvas shows next-best-action as canvas + all sections show empty copy;
  5. details toggle flips body display and persists `home.rail_state` in `app.app_config`.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement** widgets then screen rework per Interfaces. Keep `HomeActionButton` and both `_activate_*` methods byte-identical. Delete `compose_content`'s old body wholesale; the state builder call becomes `build_home_triage_state(dashboard_input, selected_row_id=self._home_selected_row_id)`.
- [ ] **Step 4: Run** the new file + ALL existing Home-related UI tests (`grep -rl "home-" Tests/UI/ | xargs`-style discovery; update stale assertions preserving intent — e.g., tests asserting the four old header rows now assert the single header line; tests asserting section text blobs now assert rows). Full green required.
- [ ] **Step 5: Commit** `feat(home): triage rail and focus canvas`.

---

### Task 5: First-run handoff + next-action callout behavior

**Files:**
- Modify: `tldw_chatbook/Home/dashboard_state.py` (canvas variant), `tldw_chatbook/UI/Screens/home_screen.py` (routing)
- Test: `Tests/Home/test_dashboard_state.py`, `Tests/UI/test_home_triage_rail.py`

**Interfaces:**
- Consumes: `state.model_ready` (readiness already threaded by the adapter from the same single source Console uses).
- Produces: when `model_ready` is False AND nothing needs attention, the canvas becomes the setup handoff: title `Set up Console`, single line `Console needs a working model before live AI tasks.`, one action — the existing `#home-primary-action` labeled `Set up Console model` routing exactly as today's `fix_model_setup` action (Settings, PROVIDERS_MODELS context — the `_home_primary_action_context` path is already correct). No checklist duplication on Home.

- [ ] **Step 1: Failing tests** — pure: `build_home_triage_state(HomeDashboardInput(model_ready=False))` canvas has `next_action.action_id == "fix_model_setup"`, `next_action_is_canvas is True`; pilot: blocked empty input renders `Set up Console model` button and pressing it posts `NavigateToScreen("settings")` with the providers category context (assert via the message capture pattern used by the old Home navigation tests).
- [ ] **Step 2: RED.** (The pure case may already pass via Task 1's fallback — if so, note it and keep the test as a regression lock; the pilot case is the real subject.)
- [ ] **Step 3: Implement** any missing routing glue.
- [ ] **Step 4: GREEN**, both suites.
- [ ] **Step 5: Commit** `feat(home): first-run canvas hands off to Console setup`.

---

### Task 6: CSS

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (or the components file that already styles `home-*` selectors — grep first; use that file)
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_home_triage_rail.py` (stylesheet presence test mirroring `test_generated_console_stylesheet_includes_rail_section_rules`)

Selectors: `.home-rail-row` (height 2, like Console conversation rows), `.home-rail-row-selected`, `#home-rail` (width 3fr min 24, mirroring `#console-left-rail`), `#home-canvas`, `#home-header-line`, `#home-next-action-callout`; prune dead rules for removed ids (`home-dashboard-grid`, `home-pane-divider`, `home-followup-row` — grep both CSS files).

- [ ] Steps: failing presence/absence test → CSS + `./build_css.sh` → green → commit `style(home): triage rail styles, prune dashboard grid rules`.

---

### Task 7: Verification, screenshot QA, approval gate

- [ ] **Step 1:** Full affected run: `Tests/Home/`, `Tests/UI/test_home_triage_rail.py`, plus every UI file touched in Task 4. All green (no known baseline failures exist for Home).
- [ ] **Step 2:** Live captures to `Docs/superpowers/qa/home-triage-h1-<date>/` using the proven recipe (fresh isolated HOME; playwright bundled chromium; `wait_until="commit"`; wait for `.intro-dialog` hidden; route-abort non-localhost; kill stale app processes first — see `Docs/superpowers/qa/console-rail-ia-2026-07/README.md` notes): (1) blocked first-run → Set up Console canvas; (2) populated triage (seed via `_home_dashboard_test_input` if live seeding is impractical — but at least one capture must be the real app surface); (3) selection switch; (4) Details expanded.
- [ ] **Step 3:** Present to the user for explicit approval before merge (standing rule).
- [ ] **Step 4:** Commit QA artifacts.
