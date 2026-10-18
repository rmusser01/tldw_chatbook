# ADR-237: Kanban board — ported from backlog-md-py, owned by LocalKanbanService

Date: 2026-10-18
Status: Accepted (implemented with TASK-18934)
Supersedes: none (the task originally named an "ADR-075" path; 075 was taken
by durable-character-emote-metadata, so this decision takes the next free
number, 237)

## Context

TASK-18934's step-zero audit (recorded in the task file) established:

1. **No kanban UI exists anywhere in the app** — no screen, widget, palette
   command, or destination. The "believed broken" surface was in fact a
   fully-built, fully-tested, **zero-consumer service layer**
   (`Kanban_Interop/`: local SQLite store, server REST wrapper, scope
   router) that merged PR #3084 deliberately kept, lazily constructed.
2. **backlog-md-py** (the author's sibling project) contains a proven,
   keyboard-first board TUI (`src/backlog_py/tui/`, ~3k lines) whose
   architecture — pure view-models (`BoardSnapshot`/`TaskView`/
   `FilterState`/`SelectionState`), a mountable `BoardScreen` container,
   `BoardColumn`/`TaskCard` widgets, selection preservation across
   refreshes — is exactly the multi-agent work-board pattern the task
   wants. It is installed in the dev venv as a wheel but **undeclared** in
   `pyproject.toml` (the ADR-225 trap).
3. No honest data sources emit cards yet: Scheduling jobs and agent-run
   wakes have no card/board concepts.

## Decision

**Port, don't depend.** The board TUI's architecture and widgets are
vendored into `tldw_chatbook/UI/Kanban/` and adapted — the app does NOT
take a runtime dependency on `backlog_md_py`. Reasons: the venv wheel is
undeclared and stale; the source's data layer is markdown-files with
orchestration/run-history machinery this app does not have; and a UI
foundation we intend to evolve (chatbook card sources, permission-aware
feeds) must be editable in-tree. Attribution and the upstream file map
are recorded in the package docstrings.

**Data ownership: `LocalKanbanService`.** The board reads and writes
exclusively through the existing lazy `app.local_kanban_service`
(`tldw_chatbook_kanban.db`, 11-table schema, versioned rows, activity
log). Columns are kanban **lists** in board order; cards are kanban cards
(title/description/priority). The server REST path and the scope router
stay service-level concerns the board does not touch yet. A thin adapter
(`KanbanBoardScreen`'s snapshot loader) is the only code that knows both
sides — the view-models and widgets stay store-agnostic, so a future
server-backed or mixed board swaps the adapter, not the UI.

**Keyboard: ADR-031 reconciliation.** The upstream `q`/`ctrl+q` quit
bindings are dropped (app-global `ctrl+q` owns quit; the destination's
own hotkey owns navigation). Kept from upstream: arrows (+ hidden
`hjkl`), `shift+h`/`shift+l` move the selected card between adjacent
columns, `enter` focuses the card detail (inspector), `n` creates a card
in the focused column, `/` focuses the filter input, `r` refreshes. All
screen actions are single-letter htop-style; footer hints advertise only
implemented actions.

**Honest empty start.** The board ships with manual card creation over an
empty store — no synthetic feeds, no seeded demo data, and no migration
(audit finding #4: no legacy data exists). Scheduling jobs and agent-run
wake completions are named **future integration seams**: when they grow
card emission, they write through `LocalKanbanService` like any other
client, and the board needs no changes to surface them.

## Alternatives considered

- **Depend on `backlog_md_py` as a package** — rejected: undeclared stale
  wheel (ADR-225 pattern), wrong data layer, would freeze the UI against
  our evolution.
- **Build a board from scratch** — rejected: duplicates a proven
  selection-preservation/navigation design; the port is smaller than a
  rewrite and carries upstream's lessons.
- **Server-backed board first** — deferred: the scope router exists, but
  the local store is the honest default (works with no server) and the
  adapter seam keeps the door open.
- **Wire Scheduling/agent-wake feeds now** — rejected as dishonest data:
  neither surface emits cards today; fabricating a mapping would put
  wrong data on the board. Recorded as future seams instead.

## Consequences

- `UI/Kanban/` is a new UI package with its own tests; registration
  follows the lockstep rules (TAB_KANBAN ↔ `_SCREEN_ROUTES` ↔
  `SHELL_DESTINATION_ORDER` + `SHELL_DESTINATION_SHORTCUTS`, ADR-031/152).
- The vendored code is adapted, not copied verbatim: view-models drop
  markdown-project fields (paths, orchestration queue, run history) in
  favor of kanban card fields; refresh runs as an exclusive async worker
  on the screen (the service is async on the app loop) instead of
  upstream's thread + 5s interval polling.
- Re-syncing with upstream backlog-md-py improvements is a manual diff
  exercise, accepted in exchange for in-tree evolvability.
