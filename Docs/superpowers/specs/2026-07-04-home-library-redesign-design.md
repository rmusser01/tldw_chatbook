# Home & Library Redesign Design

**Date:** 2026-07-04
**Status:** Draft for user review (direction pre-approved: Option A both screens; Library verb model)
**Scope:** Home screen (`UI/Screens/home_screen.py`) and Library screen (placeholder `UI/Screens/library_screen.py` replaced; capabilities integrated from `main`'s Media/Ingestion/RAG-Search/Notes surfaces).
**Parent brainstorm:** `Docs/Design/New_UI/2026-07-04-home-library-redesign-brainstorm.md` (approved decisions + integration inventory).
**Design anchor:** the shipped Console redesign — spec `Docs/superpowers/specs/2026-07-02-console-dual-audience-ux-design.md`, PRs #576/#577.

## Goal

Extend the Console's interaction model (rail sections → selectable rows →
canvas → actions) to Home and Library:

- **Home** becomes the triage surface: everything needing attention is a
  visible, selectable row; the canvas shows the selected item with its
  actions.
- **Library** becomes the landing page for local media/content, organized
  around the user's four verbs: **(re)view · search · ingest · create**
  (create includes flashcards, quizzes, and Notes).

## Shared conventions (inherited from Console, binding)

- Left rail of collapsible sections (`-`/`+` toggles), state persisted via
  `save_setting_to_cli_config` under a per-screen namespace
  (`home.rail_state`, `library.rail_state`) mirroring `console.rail_state`.
- Rows: `▸` active marker, recent-first, relative age labels via
  `format_console_relative_age` (generalized out of
  `Workspaces/conversation_browser_state.py` if needed).
- Glyphs `✓/●/○` for step/run state; counters dimmed at zero, bright
  otherwise; one frame per region; plumbing collapsed under `Details`.
- Pure display-state modules (no Textual imports) + small single-purpose
  widgets (posting-style), screen files orchestration-only.
- Empty/first-run states are one quiet line plus at most one action; hard
  prerequisites route to the Console setup journey rather than duplicating
  it.

## 1. Home — triage rail + focus canvas

```
┌ Rail ────────────────────┐ ┌ Canvas ─────────────────────────────┐
│ Needs Attention (2)   -  │ │ Approval: publish chatbook           │
│ ▸ Approval: publish…  3m │ │ Source: Workflows · Status: waiting  │
│   Retry: ingest fail  1h │ │ ● waiting on you since 3m            │
│ Running (1)           -  │ │                                      │
│   Watchlist sweep     now│ │ [Approve] [Reject] [Open in Console] │
│ Recent                -  │ │ [Open details]                       │
│   Chat: tides expl…   2h │ ├ Next best action ────────────────────┤
│   Note: meeting…      1d │ │ Add an API key → Settings (Enter)    │
│ Details               +  │ └──────────────────────────────────────┘
└──────────────────────────┘
```

- **Rail sections:** `Needs Attention (n)` (approvals, failures — count in
  header), `Running (n)`, `Recent` (cross-module recent work), `Details`
  (collapsed: everything in today's System Status pane — runtime, server,
  ACP, storage).
- **Selection model:** rows are focusable buttons (Console conversation-row
  pattern). Selecting a row renders it in the canvas; the existing
  `HOME_CONTROL_METHODS` actions (approve/reject/pause/resume/retry/open)
  become the canvas action row, enabled per item state. This replaces the
  current invisible `choose_home_selected_item` selection.
- **Header:** collapses to one line `Home | {Ready|Blocked} · {Local|Server: x}`
  (replacing today's four stacked header/status/scope/hint rows). Key hints
  move to the screen footer.
- **Next Best Action:** rendered once — a single-line callout beneath the
  canvas; it IS the canvas content when nothing is selected and no item
  needs attention.
- **First-run:** when Console setup is incomplete (same readiness single
  source + `console.onboarding.first_send_completed`), the canvas shows one
  line and a `Set up Console` button routing to Console (whose blocking
  modal owns the journey). No checklist duplication on Home.
- **State:** `Home/dashboard_state.py` evolves from preformatted line blobs
  to row dataclasses: `HomeRailRow(row_id, section, glyph, title, age_label,
  source, status, detail_route, actions)`. `summarize_home_dashboard`
  becomes `build_home_dashboard_state` returning sections of rows + the
  selected-item canvas state. Existing adapter inputs
  (`HomeDashboardInput`) unchanged.
- **Data flow:** ingestion `ProcessingDashboard` lifecycle events (from the
  Library Ingest section, below) surface as `Running` rows with
  `● running` / `✓ complete` / failure rows promoted to `Needs Attention`.

## 2. Library — catalog rail + browse workbench

```
┌ Rail ────────────────────┐ ┌ Canvas ──────────────────────────────┐
│ [search everything…    ] │ │ Media (17)          [type: All ▾]     │
│ Browse               -   │ │ ▸ intro-lecture.mp4   video      2h  │
│ ▸ Media (17)             │ │   paper-draft.pdf     document   1d  │
│   Conversations (128)    │ │   standup-notes.md    note       3d  │
│   Notes (42)             │ ├ Preview ─────────────────────────────┤
│   Collections (5)        │ │ intro-lecture.mp4 · video · 2h       │
│ Create               -   │ │ transcript excerpt / metadata…       │
│   New note               │ │ [Open] [Run RAG on this] [Export]    │
│   Flashcards  due: 12    │ └──────────────────────────────────────┘
│   Quizzes (2)            │
│   Study decks (3)        │
│ Ingest               -   │
│   Import media           │
│   Import / Export        │
│ Details              +   │
└──────────────────────────┘
```

- **Rail-top search:** a persistent input above the sections (the "search"
  verb is always one focus away). Typing searches across content types;
  results render in the canvas; the advanced RAG surface (modes, top-k,
  source filters, saved searches, history — from `main`'s
  `Views/RAGSearch/` package) opens from the results canvas.
- **Browse** (the "(re)view" verb): content types as rows with live counts —
  Media first, then Conversations, Notes, Collections. Selecting one loads
  the canvas list→preview split: list rows reuse the Console browser row
  grammar (title, secondary line, age); preview is type-appropriate
  (`Widgets/Media/` viewer panel for media; message/preview snippets for
  conversations/notes). Media sub-types (from `MediaWindow_v2`'s navigation
  panel) become a filter select in the canvas header, not rail rows.
- **Create** (the "create" verb): `New note` (routes into the Notes
  workbench), `Flashcards due: n`, `Quizzes (n)`, `Study decks (n)` —
  creation entry points plus practice surfaces; due-counts are the
  attention hook and mirror into Home's `Needs Attention` when nonzero.
- **Ingest** (the "ingest" verb): canvas hosts `main`'s ingestion panels —
  local drop-zone flow (SmartFileDropZone → UnifiedProcessor →
  ProcessingDashboard), remote per-type API forms, web clipper, results
  panel — as sub-entries `Import media` / `Import / Export` (clipper lives
  inside Import media).
- **Details** (collapsed): workspaces, owner boundaries, readiness, storage
  — the placeholder hub's readiness/owner copy relocated.
- **Retired:** the nine mode chips and the hub summary table (rail counts +
  search replace them). The canvas empty state carries the landing
  guidance: `Search, pick a content type, or ingest something new.`
- **Integration source of truth:** `main`'s four surfaces per the
  brainstorm's inventory table. Three are already modular packages
  (`Widgets/Media/`, `Widgets/NewIngest/`, `Views/RAGSearch/`) — the work is
  porting them to dev (divergence check first) and wiring them into the
  canvas, not rebuilding. Notes: Library lists/creates and ROUTES to the
  Notes workbench for editing (recommended; absorb-vs-route is an open
  question below).

## 3. Architecture

| Unit | Role |
|---|---|
| `Home/dashboard_state.py` (rework) | Pure row/canvas state builder (`build_home_dashboard_state`) |
| `Widgets/Home/home_rail.py`, `home_canvas.py` (new) | Rail sections + selected-item canvas widgets |
| `Library/library_state.py` (new pure module) | Rail counts, browse-list rows, search-result rows, due-counts |
| `Widgets/Library/…` (new package) | Rail, browse canvas (list+preview), search canvas, ingest canvas, create canvas — one widget file each; canvases wrap the ported `main` packages |
| `UI/Screens/home_screen.py`, `library_screen.py` | Orchestration only: compose, selection dispatch, prefs persistence, routing |

- Rail prefs: reuse the `ConsoleRailPreferences` coerce/serialize pattern
  under `home.rail_state` / `library.rail_state` (per-screen global scope —
  no per-conversation scoping here).
- Error handling: absent adapters/services degrade to empty sections with
  quiet copy (Console convention); ingestion failures surface as rows, not
  banners; search with no providers shows setup routing, not errors.

## 4. Testing

- House pattern: pure-state unit tests first (`Tests/Home/`,
  `Tests/Library/`), pilot tests per widget/screen behavior, generated-CSS
  presence tests for new selectors, screenshot QA via the proven
  textual-serve capture recipe, and the standing per-screen user screenshot
  approval gate before merge.
- Port verification: each ported `main` package lands with its existing
  tests (or minimal new smoke tests where main had none) before canvas
  wiring builds on it.

## 5. Phasing (each independently mergeable, own plan)

- **H1 — Home triage rail + canvas** (small; no Library dependency).
- **L1 — Library shell:** rail (sections, counts, prefs, search input),
  Browse ▸ Conversations canvas (dev already has the browser state), chips
  and hub retired.
- **L2 — Browse ▸ Media + Notes + Collections:** port `Widgets/Media/`,
  wire viewer preview; Notes listing + routing.
- **L3 — Search + Ingest + Create:** port `Views/RAGSearch/` and
  `Widgets/NewIngest/`; Create section with due-counts; Home `Running`
  integration.

## 6. Open questions (answer before the corresponding plan)

1. **Sequencing** vs Console Phases 3–4 (keyboard layer is in flight;
   Ctrl+K fuzzy-find should extend to Library content once both exist).
2. **Home detail routes:** `Open in Console` stays primary, or inline Home
   detail views later?
3. **Notes absorb vs route** (spec recommends route; absorb would make
   Library the literal single landing page at the cost of duplicating a
   3,278-line workbench).
4. **Dev/main divergence** of the three portable packages — must be audited
   in L2/L3 planning (they may already exist on dev in some form).
