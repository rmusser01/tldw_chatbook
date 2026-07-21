# Live verification — Console rail flush-left width-aware row names

Date: 2026-07-21. Branch: `feat/console-rail-row-wrap`. Driver: `verify` skill
(tmux, real `python -m tldw_chatbook.app` from the worktree venv).

## Confirmed live (things the test suite cannot show)

1. **New layout renders correctly with real data** (`wide-235col-real-data.txt`).
   Conversation names render flush left with no `►`/two-space marker prefix;
   the metadata line ("Chats - saved chat - 5w") is each row's final line; the
   star column (`*`/`.`) stays aligned; the "New conversation" alias button is
   present.
2. **The alias button survives the width-driven relabel recompose.** Its
   presence after render is the live proof that the `Relabeled` message →
   `chat_screen` idempotent alias re-mount path works (the branch's main
   architectural risk).
3. **Metadata line truncates at narrow width** (`narrow-110col-real-data.txt`).
   At a tight rail the metadata "Chats - saved chat - 5w" is cut to "Chats -"
   by `truncate_console_row_cells` — the live budget-aware truncation path.
4. **Live resize exercises `_maybe_relabel_for_width` and reaches a stable
   fixed point in both directions with no oscillation.** Resized the running
   session 235 → 120 → 110 → 235 cols. The rail re-rendered each way and was
   byte-identical across multiple seconds at every width (rail-only diffs; the
   only per-frame change anywhere was the composer's blinking text cursor).
   This validates the equality guard and the `scrollbar-gutter: stable`
   decoupling against the flap loop the design warned about.

## Covered by real-app tests, not by a tmux screenshot

A conversation **name physically wrapping to two lines** is not in these
captures. The Console "Chats" browser only surfaces conversations created
through the Console *send* path (which sets the console-session linkage the
`local_chat_conversation_service` scope query requires); raw-seeded DB rows
(even with `scope_type='global'` and messages) are not listed, and driving a
real send needs a working LLM provider. Seeding that linkage or driving the
multi-step rename modal was disproportionate for a screenshot.

The two-line wrap is instead covered by `test_console_rail_titles_wrap_at_
measured_width` and `test_console_rail_wrap_budget_tracks_terminal_width` in
`Tests/UI/test_console_workspace_context_rail.py`, which render in a **real
Textual app** (`app.run_test`) at real geometry and assert each name line is
≤ the measured budget and that the narrow-terminal budget is strictly smaller
than the wide one — real renders, not mocks — plus the final review's manual
trace of the wrap algorithm.

## Captures

- `wide-235col-real-data.txt` — full new layout, real DB.
- `narrow-110col-real-data.txt` — same rail resized narrow (metadata truncation).
