# Home Triage Rail (H1) — QA evidence (2026-07-04)

Branch: claude/home-library-redesign (worktree). Captured from textual-serve
(real app CSS, worktree code) in headless bundled chromium, fresh isolated
HOME (/private/tmp/tldw-h1-qa).

- home-triage-fresh-blocked-2026-07-04.png — fresh install: one-line header
  "Home | Ready · Local"; rail sections Needs Attention / Running / Recent
  with quiet empty copy and Details collapsed (+); canvas shows the next
  best action ("Import Library sources" + reason + primary button).
- home-triage-details-expanded-2026-07-04.png — Details toggled open:
  status summary + runtime/readiness/server lines relocated from the old
  System Status pane.

## Populated-triage live captures (2026-07-05, second capture session)

Captured from textual-serve at http://127.0.0.1:9071 (real app CSS, worktree
code) in headless bundled chromium, fresh isolated HOME
(/private/tmp/tldw-h1-qa-populated). Work items seeded through the screen's
`_home_dashboard_test_input` override (the same hook the pilot suite uses)
because live watchlist/chatbook service data is not available in an isolated
QA HOME; everything downstream of the input — state builders, rail rows,
selection, canvas, CSS — is the real app surface.

The first populated capture exposed two Console-parity defects the
empty-state captures could not: row titles wrapped across both label lines
(pushing the source line out of the 2-cell row) and the canvas actions
stacked vertically (ds-toolbar class sat on each button, not a Horizontal
container). Fixed in fix(home): console-parity rail rows and horizontal
canvas toolbar; the captures below are post-fix.

- home-triage-populated-2026-07-05.png — populated triage: Needs Attention
  (2) with `▸` on the selected approval row (● glyph, focus-bg selected
  style), titles truncated at 20 chars with the full title as tooltip,
  source line carries the age ("Workflows - 3m", "Schedules - 1h"),
  Running (1) "now", Recent (2) with ○ glyphs and 42m/3h ages, Details
  collapsed (+); canvas shows the approval item (source/status/age/route
  lines), the control actions in one horizontal toolbar row including
  "Open in Console" (console_available item), and the next-best-action
  callout ("Review pending approvals").
- home-triage-selection-switch-2026-07-05.png — after clicking the Running
  row: `▸` moved to "Watchlist sweep:...", canvas swapped to that item
  ("running since now", Route: watchlists) with the toolbar intact.

Control dispatch (method + kwargs) remains covered by the pilot suite
(Tests/UI/test_home_triage_rail.py + reworked legacy suites) under the real
stylesheet (HomeHarness CSS_PATH).

Verification: Tests/Home (52) + test_home_triage_rail (6) +
test_home_screen (31) + gate1 Home tests + console rail-section suite =
115 passed, 0 failed.
