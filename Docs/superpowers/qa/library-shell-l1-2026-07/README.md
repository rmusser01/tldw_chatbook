# Library Shell (L1) — QA evidence (2026-07-05)

Branch: claude/library-shell-l1 (worktree, head `92c31597`). Captured from
textual-serve (real app CSS + worktree code via PYTHONPATH) in headless
bundled chromium, viewport 2050x1240 (device_scale_factor 1), fresh isolated
HOME `/private/tmp/tldw-l1-qa`. Browse ▸ Conversations shows five REAL
conversations seeded into the isolated ChaChaNotes DB
(`default_user/tldw_chatbook_ChaChaNotes.db`, client_id
`tldw_cli_local_instance_v1`) with real message counts and relative ages —
no fixtures or `_test_input` overrides.

- library-shell-landing-2026-07-05.png — fresh/zero-count state: one-line
  header `Library | Local`; rail-top `Search conversations…` input; sections
  Browse / Create / Ingest with counts (Media (0), Conversations (0),
  Notes (0+), etc.), Details collapsed (+); canvas shows the landing copy
  `Search, pick a content type, or ingest something new.`
- library-shell-conversations-2026-07-05.png — Browse ▸ Conversations
  populated: rail count `Conversations (5)` with `▸` on the row; canvas lists
  the five conversations (title truncated with `…`, `N messages - age`
  secondary: `4 messages - 3m`, `6 messages - 2h`, `3 messages - 1d`,
  `5 messages - 3d`, `2 messages - 2w`); first row selected; preview shows
  title / `Messages: 4` / `Updated: 3m` and the `Open in Console` action in a
  horizontal toolbar.
- library-shell-selection-switch-2026-07-05.png — after clicking the second
  conversation row: `▸`/selected style moves, canvas preview swaps to
  `Explain how tides work` / `Messages: 6` / `Updated: 2h`.
- library-shell-flashcards-mode-2026-07-05.png — Create ▸ Flashcards: the
  legacy flashcards mode body renders inside `#library-canvas` (handoff copy,
  source-context list carrying the five conversations, WIP callout, and the
  live `Flashcards` handoff button) — proves legacy-surface reachability
  survives the chip retirement.
- library-shell-details-expanded-2026-07-05.png — Details toggled open:
  runtime/count summary + workspace rules relocated from the retired
  System/hub panes, wrapping cleanly within the rail pane (no clip past the
  border).
- library-shell-details-actions-2026-07-05.png — rail scrolled to the bottom
  of Details: the full workspace depth panel (per-source visibility table,
  `Console/RAG handoff: 0 eligible, 5 blocked`, workspace rules) ending in
  `Workspace actions → Create local workspace` and the Use-in-Console
  affordance — proves the rail now scrolls and every Details row is reachable
  (the `overflow: hidden` rail-reachability concern from review is resolved by
  `92c31597`).

## Verification

Full affected suites green at head: `Tests/Library/` + `Tests/UI/test_library_shell.py`
(72), the seven re-anchored legacy Library UI suites, `test_destination_shells.py`,
`test_master_shell_design_system_contract.py`, `test_non_obscuring_focus_contract.py`,
`test_home_triage_rail.py`. The only red tests in the swept area are 8
base-verified pre-existing failures in `test_destination_visual_parity_correction.py`
(home/console/chat, unrelated to Library).

## Product reductions / notes to weigh at approval

- The conversations preview is three lines (title / messages / updated); the
  legacy inspector's workspace, authority, owner, and handoff-eligibility
  lines are not carried into the L1 preview.
- The Search/RAG canvas is single-pane (the legacy right-pane inspector
  content was dropped); the Import/Export canvas is guidance copy only — its
  routes live on the `Import media` / `Import / Export` rail rows.
- The conversations canvas lists the five most-recent conversations (the
  shared local-source snapshot cap); rail search currently filters only those
  fetched records. The pure module already supports a higher limit for when
  L2 lands the media/notes browsers.
- The workspaces surface is relocated under Details (collapsed by default);
  its actions (Create local workspace, Use in Console) are two interactions
  deep and reachable by scrolling — a plan-mandated placement, not a
  regression.
