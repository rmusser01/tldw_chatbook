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

### Post-change captures (Task 11)

_To be added: search canvas idle w/ history; searching status line; results with Open;
open-landed notes editor + media viewer; `Flashcards due: N` bright/dim rail states;
Home Needs Attention due row; Study screen landed on flashcards via `Review flashcards`._

### Follow-ups carried (unchanged from spec)

- Service-backed conversations FTS filter (in-canvas filter stays client-side over the snapshot).
- Saved searches (deferred; history only).
- Rail-badge refresh at sync completion; last-synced timestamp (L2b.2 carryovers).
