# Console Rail IA Phase 1 — QA evidence (2026-07-02)

Branch: claude/console-rail-ia-phase1 (base 1156ece1, head c85dba6c + fixes)
Captured from textual-serve (real app CSS) in headless Chrome, isolated HOME
(/private/tmp/tldw-qa-home-rail-ia), fresh first-run state.

- console-rail-ia-fresh-2026-07-02.png — first run: four sections (Session /
  Context / Model / Details) in order, Details collapsed (+), Model shows the
  two compact lines, active conversation row carries the "now" age label.
- console-rail-ia-details-expanded-2026-07-02.png — Details toggled open (-):
  Storage / Sync / File tools / Server handoff / Handoff / ACP rows demoted
  into the disclosure.
- console-rail-ia-two-tabs-age-labels-2026-07-02.png — second native session
  via New tab: rail lists "> Chat 2 … active session - now" and
  "Chat 1 … open session - now".

Persistence evidence (visual relaunch capture blocked by a textual-serve
session-spawn flake after repeated sessions; behavior verified instead by):
1. Pilot test test_console_details_toggle_expands_and_persists (green).
2. The live toggle session wrote the full 6-key preference shape to the
   isolated HOME config:
   [console.rail_state."console_rail_state:workspace-default:<conv-id>"]
   left_open = true / right_open = false / session_open = true /
   context_open = true / model_open = true / details_open = true
3. coerce/serialize round-trip unit tests (Tests/Chat/test_console_rail_state.py).

Test verification (2026-07-02): affected set = 327 passed, 2 failed, both
pre-existing baseline (confirmed failing at branch base 1156ece1):
test_console_session_surface_uses_flex_height_not_full_percent_height,
test_console_browser_selecting_duplicate_membership_row_ignores_other_workspace_open_session.
