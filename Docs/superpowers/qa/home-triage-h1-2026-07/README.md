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

Populated-triage states (selectable rows with glyphs/age labels, selection
switching the canvas, control dispatch) are covered by the 95-test pilot
suite (Tests/UI/test_home_triage_rail.py + reworked legacy suites) which
runs under the real stylesheet (HomeHarness CSS_PATH); a live populated
capture needs real watchlist/chatbook service data and is deferred to the
approval conversation.

Verification: Tests/Home (52) + test_home_triage_rail (6) +
test_home_screen (31) + gate1 Home tests + console rail-section suite =
115 passed, 0 failed.
