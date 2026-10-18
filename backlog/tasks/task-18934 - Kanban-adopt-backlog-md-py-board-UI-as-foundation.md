---
id: TASK-18934
title: 'Kanban: adopt the backlog-md-py board UI as the foundation'
status: Done
assignee:
- '@Robert'
created_date: 2026-08-19 09:55
updated_date: 2026-10-18 16:56
labels:
- kanban
- ui
- agents
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A kanban surface exists in the app but is not directly accessible and its UI is believed broken or poor (2026-08-19 hermes-release review follow-up; hermes's Kanban is its founding desktop plugin and a proven multi-agent work-board pattern). Rather than patching the current surface, migrate and improve the kanban Textual UI/app from backlog-md-py as the foundation and rebuild chatbook's board on top of it: keyboard-first column/card navigation, card detail view, and a real entry point from the app shell (nav/palette). Foundation work, explicitly not immediate — coordinate with the surfaces that would drive cards (Scheduling module jobs, agent-run wake completions) so the board has honest data sources. Step zero is an audit of what kanban data/surface actually exists today, since the current state is secondhand knowledge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An audit of the existing kanban surface and its data (what exists, what is broken, whether anything is persisted) is recorded in the task before any code lands
- [x] #2 The backlog-md-py kanban TUI is evaluated and ported/adapted as a self-contained screen or widget: keyboard-first navigation (columns ←/→, cards ↑/↓, Enter opens card detail) with no mouse dependency
- [x] #3 Existing kanban data (if the audit finds any usable) maps or migrates to the new surface honestly — no silent loss, and dead/unusable legacy data is explicitly reported rather than carried
- [x] #4 The board is reachable from the app shell (nav bar and command palette) under normal chrome rules; no dead or hidden entry points
- [x] #5 Keybindings follow ADR-031 (htop-style single-letter screen actions; no terminal-convention keys; footer hints only advertise implemented actions)
- [x] #6 Tests cover navigation, data mapping, and shell accessibility; a user guide page documents the board
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/075-kanban-board-foundation.md (to be drafted before implementation).
Reason: long-lived application-structure decision — replaces an existing (broken) surface with a ported architecture and defines the board's data ownership (local store vs Scheduling/agent-run sources). Not immediate: sequence after current Console/fleet work settles.

1. Audit the existing kanban surface and data; record findings in this task
2. Draft ADR-075: port vs rewrite boundary, data ownership, integration seams (Scheduling, agent runs)
3. Port/adapt the backlog-md-py board UI as a self-contained Textual surface
4. Shell entry points + keyboard conventions per ADR-031
5. Tests + user guide page
<!-- SECTION:PLAN:END -->

## Deferral note (2026-10-11, dev `9f8533686c`)

DEFERRED from the t2-hermes porting slice — untouched, deliberately, and
consistent with the task's own "explicitly not immediate" framing. Sequencing
facts for whenever it starts:

1. The task's step 0 (audit the existing kanban surface) interacts with
   PR #3084 (open at audit time), which reworks the Kanban trio's boot
   wiring (lazy construction, retiring the zero-consumer local Kanban
   store) — the audit must run against that landed state, not today's.
2. ADR-075 (port vs rewrite boundary, data ownership) is a prerequisite
   before any port of the backlog-md-py board.

## Step-zero audit (2026-10-18, dev `b07fc04252`)

Recorded before any code landed (AC#1). Performed by exhaustive
case-insensitive search across the repo plus targeted reads; the premise
corrections below supersede the task description's secondhand "believed
broken" framing.

**1. There is no kanban UI at all — nothing is broken because nothing
exists.** Zero hits for "kanban" in `UI/`, `Widgets/`, `Event_Handlers/`,
or `Constants.py`: no screen, widget, palette command, nav destination, or
`TAB_*` constant. The "surface not directly accessible" was in fact a
service layer with no consumer.

**2. What exists is a fully-built, fully-tested, zero-consumer service
layer** — `Kanban_Interop/` (6 files, ~4.3k lines):
`local_kanban_service.py` (SQLite CRUD/move/copy/archive/labels/
checklists/comments/bulk/search/links/import/export/versioning),
`local_kanban_db.py` (11-table schema incl. FTS5 probe, creation deferred
to first connect), `server_kanban_service.py` (policy-gated REST wrapper
over `tldw_api` kanban endpoints), `kanban_scope_service.py` (local/server
router), `recovery.py` (backup participant). Store:
`tldw_chatbook_kanban.db` in the user data dir. Readers/writers beyond
tests: none.

**3. PR #3084's rework is merged (0ac67671e6, 2026-10-12) and its real
effect was KEEP + lazy, not retire**: module-scope import removed
(TASK-21107), the trio builds lazily on first property access
(`app_service_wiring.py:965-1037` under `_kanban_services_lock`), nothing
constructed at boot. Pinned by `Tests/App/test_boot_no_feature_db_files.py`,
`Tests/Utils/test_tldw_api_schema_deferral.py`,
`Tests/Performance/test_ui_ready_module_census.py`. The audit ran against
this landed state, as the deferral note required.

**4. Honest data sources do not exist yet.** Scheduling (`Scheduling/`,
`UI/Screens/scheduling/`) has zero card/board concepts; agent-run wakes
(`Agents/agent_service.py`, ADR-199) deliver via Console toasts/badges
only. **AC#3 mapping verdict: there is no legacy kanban data to migrate
and no dead surface to report — the local store ships empty and nothing
user-facing ever wrote to it. The board starts honestly empty with manual
card creation; Scheduling/wake feeds are future integration seams named
in ADR-237, not synthetic data wired now.**

**5. backlog-md-py board UI is port-ready.** Sibling checkout at
`/Users/macbook-dev/Documents/GitHub/backlog-md-py` (on `main` @ 39e66a8,
57 commits behind origin/main); also present as a WHEEL in the dev venv
site-packages and NOT declared in pyproject.toml (the ADR-225 undeclared
-dependency trap). Board TUI in `src/backlog_py/tui/` (~3.0k lines):
`widgets.py` (`BoardColumn(VerticalScroll)`, `TaskCard(Static)`,
`BoardHeader`, `FilterBar`, `TaskInspector`), `screens.py`
(`BoardScreen(Container)` — the port-ready piece), `models.py` (pure
view-models `BoardSnapshot`/`TaskView`/`FilterState`/`SelectionState`),
`data.py` (`BoardDataSource` Protocol + markdown-backed local impl +
daemon impl), `dialogs.py` (ModalScreen dialogs), `app.py` (App shell,
bindings, 5s refresh worker). Port frictions: App-level bindings must
move onto a chatbook screen; `q`/`ctrl+q` quit bindings violate ADR-031;
its data layer is markdown-files, not our store.

**6. Registration pattern (current)**: `Constants.py` `TAB_KANBAN` +
label → `UI/Navigation/screen_registry.py` `_SCREEN_ROUTES` entry →
`shell_destinations.py` `SHELL_DESTINATION_ORDER` + `SHELL_DESTINATION_
SHORTCUTS` in lockstep (ADR-031/152); palette entries derive
automatically. Next-free shortcut slot to be taken at implementation.

**7. ADR numbering**: `075-kanban-board-foundation.md` does not exist and
075 is TAKEN (`075-durable-character-emote-metadata.md`); highest existing
number is 236 → **ADR-237** is this task's ADR.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented on branch feat/18934-kanban-board (worktree .worktrees/tasks-18925-28-34, base dev b07fc04252). ADR-237 (backlog/decisions/237-kanban-board-port-and-data-ownership.md) — the task's stale 'ADR-075' path was taken by another decision; 237 is next-free.

Audit (AC#1): recorded in this file before any code — no kanban UI ever existed (nothing broken to patch); a fully-tested zero-consumer service layer (Kanban_Interop local SQLite + server REST + router) was kept lazily by merged PR #3084; NO legacy data exists to migrate (AC#3: store ships empty, nothing user-facing ever wrote to it — honest empty start with user-initiated n-key bootstrap of a Personal/To Do/Doing/Done board); backlog-md-py's board TUI is port-ready in the sibling repo.

Port (AC#2): new tldw_chatbook/UI/Kanban/ package — board_models.py (pure view-models CardView/BoardSnapshot/FilterState/SelectionState + filter_snapshot + upstream's refresh-stable select_after_refresh; snapshot carries list_ids so mutations work off any loader), board_widgets.py (BoardHeader/BoardColumn/CardWidget/CardInspector, markup-free rendering, ASCII-sanitized widget ids), kanban_board_screen.py (BaseAppScreen; arrows+hjkl columns/rows with empty-column skipping, shift+h/l move card via service.move_card, Enter focuses inspector, n = NewCardModal in focused column or first-board bootstrap, / filter, r refresh; exclusive async workers for load and mutations; snapshot_loader injectable for tests). Keyboard per ADR-031 (AC#5): upstream q/ctrl+q dropped; single-letter actions; no terminal-convention keys.

Registration (AC#4): TAB_KANBAN + label, _SCREEN_ROUTES entry, SHELL_DESTINATION_ORDER appended LAST with f8 (next free slot per the left-to-right walk; no existing slot moved) + lockstep test dict updated (the explicit-edit rule).

Data ownership: exclusively app.local_kanban_service (ADR-237); server REST/scope router untouched.

Files: UI/Kanban/{__init__,board_models,board_widgets,kanban_board_screen}.py; Constants.py; UI/Navigation/{screen_registry,shell_destinations}.py; Docs/User_Guide/kanban.md (new page, AC#6); backlog/decisions/237-...md; Tests/Kanban/test_kanban_board.py (12 tests: filter/selection/movement pure logic, registration lockstep incl. route+constant, screen render+navigate, store move mutation).

Verification: 12/12 green; ruff clean; nav/lockstep battery has the identical 58 pre-existing setup errors (RecoveryRequired env condition) verified byte-identical on pristine via stash — no regression.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
