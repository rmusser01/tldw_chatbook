# Library Browse ▸ Media canvas (L2a) — QA evidence (2026-07-06)

Branch: claude/library-l2 (worktree, off origin/dev with H1+L1 merged).
Captured from textual-serve (real app CSS + worktree code via PYTHONPATH) in
headless bundled chromium, viewport 2050x1240 (device_scale_factor 1), fresh
isolated HOME `/private/tmp/tldw-l2-qa`. Seven REAL media items across three
types (video / document / audio) seeded into the isolated media DB
(`default_user/tldw_chatbook_media_v2.db`, client_id
`tldw_cli_local_instance_v1`) via `add_media_with_keywords` — no fixtures or
`_test_input` overrides.

- library-media-populated-2026-07-06.png — Browse ▸ Media populated: canvas
  header `Media (5)` with the cycling `type: All ▸` filter button on its own
  line; five media rows (`title` + `type` secondary: release-plan.pdf/document,
  interview.mp3/audio, demo-walkthrough.mp4/video, paper-draft.pdf/document,
  morning-briefing.mp3/audio), first row selected (`▸`); preview shows
  title / `Type` / `Updated` and the `Open in Media` action.
- library-media-filter-applied-2026-07-06.png — filter cycled to `document`
  (two presses of the filter button: All → audio → document): button reads
  `type: document ▸`, status line `2 of 5 · type: document`, list narrowed to
  the two documents (release-plan.pdf, paper-draft.pdf).
- library-media-selection-switch-2026-07-06.png — after clicking the video
  row: `▸` on demo-walkthrough.mp4, preview swapped to `Type: video`.

## Verification

Full affected suites green at head `e9337200`: `Tests/Library/` +
`Tests/UI/test_library_shell.py` + `test_destination_shells.py` +
`test_post_release_workspaces_library_depth.py` = 203 passed, 1 pre-existing
skip.

## Live-QA findings and fixes (this session)

1. Header layout bug (fixed): the type-filter Select consumed the whole header
   row and crushed the title into a 1-char vertical column. Root cause traced
   via headless region probes: a `1fr` title placed before fixed-width header
   children lets the title eat the row and overflow the siblings off the
   visible canvas.
2. Invisible filter control (fixed by design change): even after the layout
   fix, the Textual `Select` — and any sibling the header places after the
   title — did not render visibly in the deployed/served TUI, while full-width
   elements (list rows, the `Open in Media` button) render fine. Resolved by
   replacing the `Select` with a **cycling filter Button** (`type: <T> ▸`) on
   its own line, which renders reliably and advances through the media types on
   press. Functionally equivalent; more legible in a terminal.

## Notes to weigh at approval

- Action row is `[Open in Media]` only — per-item `Use in Chat` / `Export` /
  `Run RAG` are deferred (no clean handoff yet); they live in the full Media
  viewer reached via `Open in Media`.
- The canvas header count and the type filter are scoped to the fetched media
  snapshot (`LIBRARY_SOURCE_PAGE_SIZE = 5`): the rail badge shows the true DB
  total (`Media (7)`) while the canvas shows `Media (5)`. Raising the shared
  fetch cap (with L1's conversations cap) is a tracked follow-up; until then a
  large library shows at most 5 rows and the filter only sees those types.
- `Updated: unknown` in these captures reflects the seed's date field not being
  surfaced by the media list response; real ingested media with a
  `last_modified` shows a relative age. Not a defect in the canvas.
