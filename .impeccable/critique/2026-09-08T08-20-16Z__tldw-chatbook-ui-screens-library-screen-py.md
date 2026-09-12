---
target: "Library ▸ Media (critique #7, post fix wave)"
total_score: 31
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
timestamp: 2026-09-08T08-20-16Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design-review sub-agent · B: detector/evidence sub-agent, isolated on separate scratch profiles and tmux sockets; parent synthesized and reconciled the two). Target tldw_chatbook/UI/Screens/library_screen.py at dev 8dc5366dc — after the full critique-#6 fix wave (six findings 31983/31982/31981/31979/31980/31968 + Qodo follow-up 32039). Live at 235x52 and 100x30 under tmux, isolated copies of the real DBs; no analysis provider configured.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | The reader keeps painting an item after a 0-result filter or after the loaded item is deleted (A: stale reader) |
| 2 | Match System / Real World | 4 | Plain, honest language throughout ("Trash is empty. Items you delete from Media land here.") |
| 3 | User Control and Freedom | 3 | Undo/Trash/Restore/Cancel-default are exemplary, but at 235 wide the Escape ladder will not close the reader and a round-trip strands you on a "Get started" card (B #8) |
| 4 | Consistency and Standards | 2 | Conversations select mode is a degraded, truncated copy of Media's; `/` targets the rail search; two checkbox glyph styles |
| 5 | Error Prevention | 3 | Strong confirmations, but 0-selected disables Export/Review/Delete with no inline reason (only Analyze explains) |
| 6 | Recognition Rather Than Recall | 3 | The `/` "focus search" hint points at the wrong field; the reader footer shows review keys with "No items to review" |
| 7 | Flexibility and Efficiency | 3 | Rich keyboard model, but at 235 wide the first arrow-Down off a focused row leaks focus into the reader instead of advancing the cursor (B #8) |
| 8 | Aesthetic and Minimalist Design | 3 | A regional-indicator flag keyword drifts the row frame +2 cells and overpaints the reader header; the select-mode action column is ragged |
| 9 | Recognize/Diagnose/Recover from Errors | 4 | Import queue names file + errno + Retry/Dismiss; the analysis block names the fix; the failed-load Retry now escalates to a reopen step |
| 10 | Help and Documentation | 3 | F1, contextual footers, supported-formats copy, placeholders |
| **Total** | | **31 / 40** | **Good** |

Trend for this surface: 24 → 26 → 28 → 25 → 21 → 25 → **31** (out of 40). The critique-#6 fix wave is the largest single move in the program's history and lands the surface in the "Good" band for the first time. Assessment A alone scored 33; I reconciled to 31 for two deterministic defects B caught that A missed (the 235-wide Escape/arrow focus-routing, dinging H3 and H7).

## The six fixes, independently confirmed by both assessors

- **Keyboard focus is a glyph cursor, not a colour-only near-match** (was critique #5/#6's sharpest P1): a `█` left-edge bar + `▸` caret + `Loaded ·` text, distinct from the selected background; arrows move it. B measured it carries by shape, not colour alone. At 100x30 the bar drops but the caret and text still carry it.
- **Blocked actions explain themselves inline with a next step**: `○ Generate` and select-mode `○ Analyze` both paint an always-visible "No analysis provider is configured · Set one in Settings ▸ Providers & Models." — no hover required.
- **Permanent-delete is danger-marked and separated**: rose ink (rgb 209,126,146) distinct from Cancel, a 3-cell gap, "This cannot be undone.", Cancel defaulted.
- **The wide layout shows the full title**: with no item open the Items pane widens and a 98-char title paints in full (cell_len 98, no ellipsis); opening an item restores the reader.
- **Scroll survives a mode change**: Read → Analysis → Read keeps the same lines painted (verified where the body overflows).
- **The failed-load Retry escalates to a recovery step** after a repeat: "· reopen Chatbook to reconnect to the media database".

## Design Specificity Verdict

Highly specific — purpose-built, not a template (Assessment A). Match-reason suffixes, a graded reversible-then-irreversible delete model, honest type-neutral empty states, a first-class review-set mode, and an explicit nav overflow are details no generic list-detail scaffold carries. The remaining issues are consistency debt across a strong system, not the absence of one. The detector is not applicable (a Textual TUI has no DOM; `.py` is unscannable), so all of B's evidence is live-measured.

## Priority Issues (new — these did not exist to find at critique #6, or were masked by its P0s)

- **[P1] `/` focuses the rail's global "Search Library…", not the Media "Title/keyword…" filter it advertises.** Keyboard-first is the product's core; the footer says "/ focus search" but focus jumps two panes to a different search, so a filter query does nothing until noticed. Worst for keyboard-only users. Fix: bind `/` to the active canvas's filter, or relabel. (A: caps/04, caps/10)
- **[P1] At 235 wide, arrow-Down off a focused list row leaks focus into the reader and the Escape ladder won't close it.** The first Down defocuses the list instead of advancing the cursor; four Escapes neither close the reader nor move a cursor; a Console round-trip strands Library on a "Get started" card needing Home→Library to recover. Passes at 100x30, so it is a width-dependent focus-routing bug. Fix: route arrow keys to the list cursor while the list owns focus at every width; make Escape close the reader. (B #8)
- **[P1] Conversations select mode is a degraded, partly-unlabeled copy of Media's** ("Selec", a bare "○" with no label, wrapped "0 selected"). Fix: share the Media select-toolbar component. (A: caps/32)
- **[P2] The reader desyncs from the list** — it keeps painting an item after a 0-result filter or after the loaded item is deleted. Fix: clear to "No selection" (or "This item was deleted — Undo") when the loaded item leaves the visible set. (A: caps/06, caps/29)
- **[P2] A regional-indicator flag keyword miscounts by two cells** — the row frame drifts to column 237 (vs 235) and the reader header overpaints ("NoNo analysis yet."). This is the stated ceiling of task-31955's cell-aware cut, now shown to have a visible frame consequence. Fix: a UAX #29 flag-pair-aware width, or refuse to paint a half-flag. (B #5)
- **[P2] 0-selected disables Export/Review/Delete with `○` but no inline reason** — Analyze explains its block; the others just dim, against the surface's own "explain why unavailable" rule. Fix: one "Select items to enable" line by the disabled bulk row. (A: caps/25)

## Minor Observations

The reader More strip's "Move to trash" clips to "Move to " at some widths and reads as quiet-danger grey rather than a louder role (B #6 — the quiet-danger ink is by design, the clip is not). Trash stays enabled while Export/Select/Review gate on a failed load — this is by design (task-31982 made Trash the one ungated recovery control). Soft-delete "Delete" renders dimmer than "Cancel" (A). Two checkbox glyph styles (list vs Import-behavior). Phantom review-set footer keys when no set is active.

## What's Working

Honest blocked/failed states, everywhere and consistently. A graded, reversible-then-irreversible delete model that protects exactly the high-stakes moments. Match-reason transparency plus a real review workflow. These are the surface at its best and are what carried it into the Good band.
