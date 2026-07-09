# Library L3a QA evidence — Search promotion + Create counts + Home due-mirror

Plan: `Docs/superpowers/plans/2026-07-09-library-l3a-search-counts-home.md`
Captures: textual-serve + playwright chromium, viewport 2050x1240, device_scale_factor 1,
isolated HOME `/private/tmp/tldw-l3a-qa`, seeded 4 notes / 4 conversations / 3 media.

## Pre-change smoke (Task 1, spec-mandated "smoke first")

| Capture | What it shows |
|---|---|
| `l3a-pre-library-landing.png` | Library shell landing with seeded counts (Media 3, Conversations 4, Notes 4); Search / RAG and Collections rows still legacy `mode` rows; Create rows uncounted. |
| `l3a-pre-search-panel-idle.png` | Current `browse-search` mode canvas, idle. Blocked callout + disabled run state. |
| `l3a-pre-search-panel-submitted.png` | Query `tides` submitted. Panel reaches Ready, dispatches, and the Evidence region renders the **service-unavailable recovery block** — confirming live that no production `library_rag_search_service` exists (L3a Task 3 wires one). |

### Rendering-defect inventory (feeds plan Task 5)

1. **Query-row layout defect:** the `Run Search/RAG` button renders ABOVE and to the right of the query input, hugging the viewport edge (clipped-looking when disabled) instead of sitting inline with the input. The Ready/Blocked callout box overlaps the button's region. Root area: `#library-rag-query-controls` fixed heights (11/14) + `#library-rag-query-row` Horizontal.
2. **Legacy workbench copy throughout:** ASCII dashed section rules (`-- Query ---`), pipe-drawn scope text table (`Scope | Count | Eligibility | Next action`), rows for `Workspace eligible` / `Collections (Read/review WIP)` / `Import/Export recovery`, keyboard hint `Tab: move panes` (panes no longer exist), and double status copy (`Blocked.` + `Unavailable: Run Library Search/RAG.`). Not Console-parity list grammar.
3. **Mode hardcoded:** header always `Mode: RAG Answer | Top 5`; no mode control exists (Task 5 adds the cycling button, default `search`).
4. **Degrade copy verbose:** the unavailable state dumps Why/Next/Recovery/Owner lines raw into the Evidence region — functional, but Task 5's status-line work should present the quiet line + setup routing per spec.
5. No horizontal clipping/wrap failures at 2050x1240 beyond (1); scope table and long lines wrap inside the canvas. No forbidden widgets (panel is Static/Input/Button only).

### Post-change captures (Task 11, at final HEAD unless noted)

Seeded: 4 notes / 4 conversations / 3 media / 3 decks / 7 due flashcards / 2 quizzes.

| Capture | What it shows |
|---|---|
| `l3a-rail-counts-dim.png` | Create rows pre-study-seed: `Study decks (0)`, `Flashcards due: 0` (dimmed), `Quizzes (0)`. |
| `l3a-rail-counts-bright.png` | Seeded counts: `Flashcards due: 7` (bright), `Study decks (3)`, `Quizzes (2)`. |
| `l3a-search-results-final.png` | Rail-submitted `tides` in Search mode: all three sources hit — note, media, conversation (score badge) — each with Select evidence + **Open**; scrollable canvas. |
| `l3a-search-multi-results-fixed.png` | Multi-result regression fix verification (`research`). |
| `l3a-query-region-ready-state.png` | Restacked query region (callout → full-width input → Run row) + `Recent searches` loaded **from disk in a fresh process** (restart persistence). |
| `l3a-search-idle-history.png` | Fresh canvas entry, Search mode default, blocked-empty-query gate, history expanded. |
| `l3a-rag-mode-blocked.png` | Mode toggled to `RAG Answer` (this env has a RAG runtime, so the gate shown is the empty-query one; provider-gate copy is pilot-covered). |
| `l3a-open-note-editor.png` | Result → Open (note) lands straight in the in-Library notes editor. |
| `l3a-open-media-viewer.png` | Result → Open (media) lands straight in the in-Library media viewer. |
| `l3a-home-due-row.png` / `l3a-home-due-selected.png` | Home Needs Attention (1): `Flashcards due: 7 · Library`; selected canvas with `Review flashcards`. |
| `l3a-study-flashcards-landing.png` | `Review flashcards` → one hop onto the Study screen's Flashcards section. |

### QA-wave findings fixed before the gate (commits e308a71f, ec1a207c, 04ceaf7a)

1. History entries containing `[`/`]` crashed the panel via Rich markup (persisted crash-loop) — escaped at render + sanitized at load.
2. History never loaded after app restart (`app_config` from `load_settings()` lacks `[library.search]`) — loader falls back to the CLI config read. NOTE: `library.rail_state` reads share this `app_config` pattern — follow-up audit.
3. Query region rendered broken (callout overlapping input; Run button floating top-right) — restacked to house grammar; fixed heights removed.
4. Multi-result searches rendered only the first row: the search canvas never scrolled (plain `Vertical`), rows past the fold were mounted but clipped — panel is now a `VerticalScroll`; geometry-based pilot added. NOTE: `#library-canvas` (shared ancestor) still does not scroll — follow-up audit for other canvases with unbounded content.
5. Stale `searching…` on canvas re-entry; search launched past a note-conflict abort; mode-toggle mid-flight mislabeling; one severed test assertion; history rerun not repopulating the input — all fixed (see `.superpowers/sdd/final-review-fixes-report.md`).

### Known observations at the gate (deliberate/pre-existing)

- Legacy copy inside the panel (ASCII `-- Query --` rules, pipe-drawn scope table, `Tab: move panes` hint, Workspace/Import-Export scope rows) predates L3a; full Console-parity restyle was not in L3a scope — flag for a follow-up polish pass if wanted.
- RAG-unavailable setup routing: when no RAG runtime exists, the pre-flight provider gate blocks Run with Console-convention copy; the richer service-level recovery copy remains for non-UI callers (deliberate Task-5 decision).
- Study screen itself is unchanged (spec: not absorbed).

### Follow-ups carried (unchanged from spec)

- Service-backed conversations FTS filter (in-canvas filter stays client-side over the snapshot).
- Saved searches (deferred; history only).
- Rail-badge refresh at sync completion; last-synced timestamp (L2b.2 carryovers).
- Conversations open-by-id prepend vs snapshot-refresh race (narrow window).
- `app_config` vs CLI-config read audit (`library.rail_state`); `#library-canvas` scroll audit.
