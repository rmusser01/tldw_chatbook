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

## Notes to weigh at approval

- `[Open in Media]` remains in the action row as an escape hatch to the legacy
  Media screen; it can be retired once the in-Library viewer is accepted as the
  primary surface.
- Header `Media (N≤5)` can disagree with the rail badge total (`Media (7)`) due
  to the shared `LIBRARY_SOURCE_PAGE_SIZE = 5` fetch cap — a tracked L1/L2
  follow-up, not a viewer defect.
- Analysis edit appends a new DocumentVersions row per save (backend has no
  in-place analysis update) — a backend follow-up, out of scope here.
