# Home & Library Redesign — Brainstorm

Date: 2026-07-04
Status: **direction approved by user 2026-07-04 — Option A for BOTH screens.**
User's Library identity statement (verbatim intent): *"Library is meant to be
the landing page for local media/content, for (re)viewing or searching it,
ingesting it, or creating it (flashcards + quizzes + Notes)."*
Decisions recorded in the "Approved decisions" section at the end; two
questions remain open (sequencing, Home detail routes). Next step: per-screen
specs via the standard brainstorm→spec→plan flow.
Anchor: the shipped Console redesign (PRs #576/#577) and its spec
`Docs/superpowers/specs/2026-07-02-console-dual-audience-ux-design.md`.

## Design language to inherit (already shipped in Console)

- Left rail of **collapsible sections** with `-`/`+` toggles, state persisted
  per scope via config (`console.rail_state` pattern).
- **Real selectable lists** with auto-titles, recent-first ordering, and
  relative age labels (`now`, `2m`, `1h`, `3d`) — `format_console_relative_age`
  is reusable as-is.
- Glyphs: `✓/●/○` step-state, `▸` active row, `▾/+/-` disclosure.
- **Plumbing demoted** into a collapsed `Details` section.
- **Pure display-state modules** (no Textual imports) feeding thin widgets;
  posting-style one-widget-one-file decomposition.
- First-run gating: live-step guidance; hard prerequisites use the
  Console-scoped **blocking modal** pattern; one-time flags persisted at
  `console.onboarding.*` (a `home.onboarding.*` / `library.onboarding.*`
  namespace follows naturally).
- One frame per region; zero-value counters dimmed, non-zero bright.

## Current state and pain

### Home (`UI/Screens/home_screen.py`, 315 lines)

Three-pane grid (Attention Queue / System Status / Selected-item inspector)
plus a follow-up row (Next Best Action / Recent Work). Every pane body is a
**preformatted text blob** from `summarize_home_dashboard`.

Pain points:
1. There is a "selected item" concept but **no visible selectable list** —
   users can't see or change what's selected; controls act on an invisible
   target.
2. Text-wall panes: no rows, no glyphs, no ages, nothing scannable.
3. Duplicated status chrome (`home-status-row` + `home-status` + scope-filter
   row + key-hints row = four stacked header lines).
4. "Next Best Action" is both a button in pane 1 and a section at the bottom
   — the same fact rendered twice.
5. No first-run story: a fresh install shows "Blocked" text but the setup
   journey lives in Console; Home should hand off to it, not restate it.

### Library (`UI/Screens/library_screen.py`, 4,112 lines)

**Correction (user, 2026-07-04): the dev-branch Library screen is a
placeholder/WIP, not the original.** The functionality Library is meant to
usurp/integrate lives on `main` as four mature surfaces — Media, Ingestion,
RAG/Search, and Notes (inventoried below). The redesign treats main's
surfaces as the capability source of truth and the dev placeholder as
disposable scaffolding.

The placeholder is a mode-chip hub with **nine flat modes**: sources,
conversations, search, import-export, workspaces, collections, study,
flashcards, quizzes. The hub mode is a read-only summary table (counts per
content type) plus readiness copy and next-action cards.

#### Integration inventory — what `main` has that Library must absorb

| main surface | Shape | Capabilities | → Library verb section |
|---|---|---|---|
| **Media** — `UI/MediaWindow_v2.py` (1,754 ln) + `Widgets/Media/` package | Already modular: navigation panel (dynamic media types incl. "All Media", collapsible), search/list/viewer panels, ingestion-source panel | Browse by media type, per-type search, item list, full media viewer | **Browse ▸ Media** — nav types become Browse sub-filters; list panel = canvas list; viewer panel = canvas preview |
| **Ingestion** — `UI/MediaIngestWindowRebuilt.py` (1,864 ln) + `Widgets/NewIngest/` | `LocalIngestionPanel` (SmartFileDropZone, UnifiedProcessor, ProcessingDashboard w/ started·complete·error lifecycle), `RemoteIngestionPanel` (per-media-type forms → tldw API), `WebClipperPanel`, `IngestionResultsPanel` | Local drop-zone ingestion with progress dashboard, remote API ingestion, web clipping, results review | **Ingest** — canvas hosts these panels (local / remote / clipper sub-entries); ProcessingDashboard lifecycle should also feed Home's `Running` rail section |
| **RAG/Search** — `UI/Views/RAGSearch/` package | Window + mixins: search input, mode/scope selects, top-k params, source-filter checkboxes (Media/Conversations/Notes…), saved-searches panel, history dropdown, index/embeddings actions, search handoff | Cross-content RAG search with advanced options, saved searches, history, chat handoff | **Rail-top search input** (quick path) + **Search canvas** (advanced options, saved searches, history); source-filter checkboxes map 1:1 to Browse's content types |
| **Notes** — `UI/Screens/notes_screen.py` on main (3,278 ln) | The already-rebuilt Console-parity Notes workbench (PR #501/#504 lineage) | Notes CRUD, templates, file sync | **Create ▸ New note** + **Browse ▸ Notes** — list/create/route from Library; do NOT duplicate the editor (open question below: absorb vs route) |

The encouraging part: three of the four are already decomposed into
posting-style packages (`Widgets/Media/`, `Widgets/NewIngest/`,
`Views/RAGSearch/`) that can slot into the Library canvas model largely
as-is. The redesign is mostly **composition + rail wiring + porting those
packages to dev**, not rebuilding capabilities. (Verify during spec: how far
dev has diverged from main in these packages, and whether flashcards/quizzes
exist on main at all or are dev-side additions — they appear to be the
latter.)

Pain points (of the dev placeholder):
1. Nine sibling chips hide structure: study/flashcards/quizzes are one
   family; search and import-export are actions, not places; workspaces is
   plumbing.
2. The hub is a **report about the library, not the library** — you read
   counts, then click a chip, then land in a different layout. Two hops to
   see any actual content.
3. No persistent rail — violates the established Console-rail principle for
   new screens (see project memory) and makes every mode a full-canvas swap
   with no cross-mode orientation.
4. One 4,112-line screen file owning nine layouts (162 defs) — the exact
   shape the Console phases deliberately decomposed away from.

## Shared goals

- One interaction model across Console/Home/Library: rail sections → select
  a row → canvas shows it → act on it.
- Everything scannable in rows (glyph + title + age), nothing in prose blobs.
- Plumbing collapsed by default, everywhere.
- Decompose into pure-state modules + small widgets as we touch each area.

---

## HOME

### Option A — Triage rail + focus canvas (recommended)

Home becomes the triage surface: the rail lists everything that wants
attention; the canvas is the selected item with its actions.

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
│  (System status, runtime,│
│   ACP, server, storage)  │
└──────────────────────────┘
```

- Rail sections: **Needs Attention** (approvals, failures — count in header,
  dimmed when 0), **Running**, **Recent** (cross-module recent work rows with
  age labels), **Details** (collapsed: today's whole "System Status" pane).
- Selecting a row drives the canvas; the existing
  `HOME_CONTROL_METHODS` actions become the canvas action row — controls
  finally have a *visible* target. Empty states per section
  (`No approvals pending.`).
- Header collapses to ONE line: `Home | Ready · Local` (counters dimmed at
  zero); key hints move to the footer (Console phase-3 convention).
- "Next Best Action" appears once, as a single-line callout under the canvas
  (or as the canvas itself when nothing is selected).
- First-run: if Console setup is incomplete (`console.onboarding` +
  readiness single-source), the canvas shows one line + a `Set up Console`
  button that routes there — no duplicated checklist, the blocking modal
  already owns that journey.
- Existing pure module `Home/dashboard_state.py` evolves from line-blobs to
  row dataclasses (title/glyph/age/route) — the Console row pattern.

### Option B — Chronological feed

Single merged feed (attention + running + recent interleaved, newest first)
with inline action buttons per row; no rail. Simpler to build, good for small
screens, but loses the rail parity, per-section counts/collapse, and makes
"needs attention" findable only by scrolling. Keep as fallback if Option A
feels heavy for a dashboard.

**Recommendation: A.** It is nearly a transliteration of the shipped Console
rail mechanics onto Home's existing data — low novelty risk, high
consistency payoff, and it fixes the invisible-selection problem outright.

---

## LIBRARY

### Option A — Catalog rail + browse workbench (recommended)

Kill the nine mode chips. The rail *is* the mode switcher; the canvas is a
real browser of the selected section, with a preview pane (posting-style
list → preview split).

```
┌ Rail ───────────────────┐ ┌ Canvas ──────────────────────────────┐
│ Browse               -  │ │ Conversations (128)   [search…]      │
│ ▸ Conversations (128)   │ │ ▸ Draft a haiku abo…   Chats     2h  │
│   Notes (42)            │ │   Explain how tides…   Chats     2h  │
│   Media (17)            │ │   Workspace B prompt   ws-b      3d  │
│   Collections (5)       │ ├ Preview ─────────────────────────────┤
│ Study                -  │ │ Draft a haiku about morning coffee   │
│   Study decks (3)       │ │ 4 messages · workspace Chats · 2h    │
│   Flashcards due: 12    │ │ [Open in Console] [Add to collection]│
│   Quizzes (2)           │ │ [Export]                             │
│ Actions              -  │ └──────────────────────────────────────┘
│   Search / RAG          │
│   Import / Export       │
│ Details              +  │
│  (workspaces, owners,   │
│   readiness, storage)   │
└─────────────────────────┘
```

- **Browse** section: content types as rows with live counts (bright when
  non-zero). Selecting one loads the canvas list — recent-first, age labels,
  same row grammar as Console's conversation browser (the
  `conversation_browser_state` builder generalizes here).
- **Study** section: study/flashcards/quizzes unified as one family; the
  `due: 12` count is the attention hook.
- **Actions** section: Search/RAG and Import/Export are *verbs* — selecting
  them opens their existing flows in the canvas (their current mode layouts
  survive as canvas content, just reachable from the rail).
- **Details**: workspaces, owner boundaries, readiness — today's hub
  readiness/owner copy, collapsed.
- The old hub summary table becomes unnecessary: the rail's live counts ARE
  the summary. The hub's next-action cards become the canvas empty state
  ("Select a content type — or import something to get started"), reusing
  the Console guidance patterns (empty-state one-liners, first-run flag).
- Decomposition requirement (this is the real payoff): each section's canvas
  becomes its own widget module under `Widgets/Library/`, with pure state in
  `Library/…_state.py`; `library_screen.py` shrinks to orchestration —
  mirroring exactly what `chat_screen.py` + `Widgets/Console/` did.

### Option B — Grouped chips + interactive hub (conservative)

Keep the mode strip but group it (`Browse | Study | Tools` with a submenu
row), and upgrade the hub table rows into buttons that jump to modes. Much
smaller diff, no rail; but it keeps the two-hop navigation and the 4k-line
file, and diverges from the Console/Notes/Personas rail direction the
project has already committed to (Notes redesign gate requires Console
parity). Only worth choosing if Library is considered frozen legacy.

**Recommendation: A**, phased:
1. **L1 — rail + Browse/Conversations** (the browser state builder already
   exists), chips removed, hub retired to empty state.
2. **L2 — Notes/Media/Collections canvases** migrated one per task.
3. **L3 — Study family + Actions**, file decomposition completed.

Home is a single small phase (H1) and can ship before or in parallel with
L1 — it touches nothing Library does.

## Approved decisions (user, 2026-07-04)

1. **Home: Option A** (triage rail + focus canvas).
2. **Library: Option A** (catalog rail + browse workbench; the nine mode
   chips are removed).
3. **Library identity**: the landing page for local media/content, organized
   around four verbs — **(re)view · search · ingest · create** — where
   *create* explicitly includes flashcards, quizzes, AND Notes.

### Library rail, refined to the user's verb model

The approved-direction rail from Option A, renamed and re-cut so each
section IS one of the user's verbs (this supersedes the earlier
Browse/Study/Actions cut):

```
┌ Rail ────────────────────┐
│ [search everything…    ] │  ← persistent cross-content search input at the
│ Browse               -   │    top of the rail (the "searching it" verb is
│ ▸ Media (17)             │    always one focus away, not a mode); results
│   Conversations (128)    │    render in the canvas; Search/RAG deep flow
│   Notes (42)             │    opens from the results view.
│   Collections (5)        │
│ Create               -   │  ← the "creating it" verb: New note ·
│   New note               │    New flashcard deck · New quiz, plus the
│   Flashcards  due: 12    │    practice surfaces (due counts are the
│   Quizzes (2)            │    attention hook; reviewing decks satisfies
│   Study decks (3)        │    the "(re)viewing" verb for study content).
│ Ingest               -   │  ← the "ingesting it" verb: media ingestion +
│   Import media           │    import/export flows (today's import-export
│   Import / Export        │    mode content becomes this section's canvas).
│ Details              +   │  ← workspaces, owner boundaries, readiness,
└──────────────────────────┘    storage (collapsed plumbing, per convention).
```

- Media moves to the top of Browse (it is the headline content type for a
  "local media/content" landing page).
- The old hub summary table is fully retired: rail counts + the persistent
  search box replace it; the canvas empty state carries the landing-page
  guidance ("Search, pick a content type, or ingest something new").
- Search placement note: a rail-top input mirrors the Console Session
  section's search and keeps the verb reachable from every state; the
  dedicated Search/RAG mode survives as the canvas that renders results and
  advanced RAG options.

## Still open (small)

1. Sequencing: do Home/Library proceed now and inherit Console Phases 3–4
   (keyboard layer, visual pass) later, or wait? (Ctrl+K fuzzy-find is an
   obvious Library synergy once Phase 3 lands.)
2. Home detail routes: does `Open in Console` remain the primary detail
   route for active work items, or does Home grow inline detail views?
3. Notes integration stance: does Library ABSORB the Notes workbench as a
   canvas, or list/create notes and ROUTE to the existing Notes screen for
   editing? (Routing avoids duplicating a 3,278-line surface; absorbing
   makes Library the single landing page the user described.)
4. Porting logistics: the main-branch packages (`Widgets/Media/`,
   `Widgets/NewIngest/`, `Views/RAGSearch/`) must be brought onto dev (or
   confirmed present/divergence-checked) before L2/L3 planning.

## Next step

Write the two specs (Home H1; Library L1–L3 with the verb-model rail above)
via the standard flow — spec → user approval → plan → subagent-driven
execution with the same review gates, capture recipe, and screenshot
approval rule as Console Phases 1–2.
