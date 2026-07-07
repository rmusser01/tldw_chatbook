# Library Browse ▸ Media full in-canvas viewer — QA evidence (2026-07-06)

Branch: `claude/library-l2` (worktree, off origin/dev with H1+L1 merged).
The viewer **replaces the standalone Media screen** for browsing: selecting a
media row in Browse ▸ Media opens a scrollable in-Library viewer with
capability parity to `Widgets/Media/MediaViewerPanel` — metadata view+edit,
content view + in-content search, delete (with confirm), reading highlights
(list/add/delete), read-it-later, analysis view+edit, and use-in-chat.
Out of scope by design: LLM analysis (re)generation and ingestion.

Captured from textual-serve (real app CSS + worktree code via PYTHONPATH) in
headless bundled chromium, viewport 2050x1240, fresh isolated HOME
`/private/tmp/tldw-l2-qa2`. Seven REAL media seeded via `add_media_with_keywords`;
`release-plan.pdf` additionally seeded with a real analysis version
(`save_analysis_version`) and two reading highlights (`create_highlight`) so the
populated Analysis / Highlights / search states are genuine, not fixtures.

- `viewer-populated-2026-07-06.png` — default viewer for `release-plan.pdf`:
  metadata (Type/Author/URL/Keywords/Ingested), Content section + search box,
  full multi-paragraph content, a real **Analysis** summary (wraps across two
  lines), two **Highlights** (quote + Color + Note + Delete), a collapsed
  `▸ Add highlight`, and the action row (Edit / Delete / Read it later /
  Use in Chat / Open in Media).
- `metadata-edit-2026-07-06.png` — Edit pressed: four prefilled full-width
  inputs (title / author / URL / keywords) + Save / Cancel. Only the local
  allowlist fields are editable; `version` is never sent (was a silent-no-op
  bug, now fixed + regression-tested).
- `content-search-2026-07-06.png` — typed `milestone` into the content search:
  status `Match 1 of 6 matches`, and the `◀ Prev / Next ▶` nav appears **only
  while searching** (hidden when the query is empty).
- `analysis-edit-2026-07-06.png` — Edit analysis pressed: a full-width TextArea
  prefilled with the current analysis + Save / Cancel. Editing an empty
  analysis creates the first version; LLM generation is out of scope.
- `add-highlight-2026-07-06.png` — `Add highlight` expanded: Quote / Note /
  Color inputs + Add highlight, with the existing highlights list still visible
  above.

## Verification

- Final whole-branch review (opus, range `149b15f0..8b1e5e17`): **Ready to
  merge: Yes** — no Critical or Important issues; async offload, metadata
  allowlist, highlight seam, mutation→refetch→recompose, state resets, and
  real-backend test coverage all confirmed. Seven prior minors triaged as
  acceptable follow-ups.
- Post-review fixes (this session, verified live + tested):
  1. `fix(library): wrap long media viewer text bodies` (a33bdbd8) — the viewer
     set its own width to `13fr` while already the sole child of the `13fr`
     canvas host; an `fr` width there breaks `width:100%` child resolution, so
     long unbroken lines (analysis summary, long URL) clipped at the right edge
     instead of wrapping. Root-caused with a minimal served repro. Fix: viewer
     fills the canvas with `1fr`, text Statics get `width:100%`, viewer no
     longer scrolls horizontally.
  2. `fix(library): guard media highlight-add against double-press duplicate`
     (663aa069) — the only fresh review minor; the non-idempotent add write is
     now exclusive in its own worker group.
- Full affected suites green after the fixes: `Tests/Library/` +
  `Tests/UI/test_library_shell.py` + `Tests/UI/test_destination_shells.py`
  = 262 passed, 1 pre-existing skip.

## UX/HCI design pass (2026-07-06, follow-up)

A senior-designer review of the viewer surfaced nine items; all were addressed
(commits `a864ce3c`, `34147154`). New evidence:

- `design-pass-viewer-2026-07-06.png` — bold/bright title, muted metadata, and
  Content/Analysis/Highlights section headers with thin top-rule dividers;
  highlights lead with a color-tinted swatch (amber ● / green ●) instead of the
  word "Color: yellow", each in its own indented card with a compact `✕ Delete`;
  rail search reads `Search Library…` (was `Search conversations…`) while
  browsing Media; the temporal line reads `Updated: 3h` (matches the list).
- `design-pass-search-highlighted-2026-07-06.png` — searching `milestone`
  reverse-highlights every occurrence in the body, so `Match 1 of 6` points at
  hits the reader can actually see.
- `design-pass-metadata-edit-2026-07-06.png` — the edit form heading reads
  `Edit media details` (no title duplication) and every field carries a
  persistent label (Title / Author / URL / Keywords) so a cleared field stays
  identifiable; the destructive Delete is separated to the far end of the
  action row and de-emphasized.

Items and resolutions: (1) edit-field labels + no duplicate title; (2) search
term highlighting in the body; (3) context-aware rail placeholder; (4) Delete
separated from Edit + per-row highlight delete; (5) title/metadata/section
hierarchy + dividers; (6) color swatches; (7) action-row grouping; (8) content
region loosened; (9) `Updated:` label + key priority aligned to the list.
Full affected suites green: 263 passed, 1 pre-existing skip. Follow-up tracked:
make the rail search actually search the active source (today it only searches
conversations); primary action row can fall below the fold in a content-rich
viewer (scroll to reach).

## Notes to weigh at approval

- `[Open in Media]` remains in the action row as an escape hatch to the legacy
  Media screen; it can be retired once the in-Library viewer is accepted as the
  primary surface.
- Header `Media (N≤5)` can disagree with the rail badge total (`Media (7)`) due
  to the shared `LIBRARY_SOURCE_PAGE_SIZE = 5` fetch cap — a tracked L1/L2
  follow-up, not a viewer defect.
- Analysis edit appends a new DocumentVersions row per save (backend has no
  in-place analysis update) — a backend follow-up, out of scope here.
