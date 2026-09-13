---
target: "Library ▸ Media (critique #6, post rider burn-down)"
total_score: 25
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
timestamp: 2026-09-07T22-29-06Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design-review sub-agent · B: detector/evidence sub-agent, isolated on separate scratch profiles and tmux sockets; parent traced the undo path in source and adjudicated the A/B contradiction before scoring). Target tldw_chatbook/UI/Screens/library_screen.py at dev 5a059e8f5 (all twenty wave-5 riders plus follow-up PR #2488 merged). Live at 235x52 and 100x30 under tmux, isolated copies of the real profile's DBs; no analysis provider configured; one app instance per profile.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Counts, receipts and "List may be out of date" are surfaced; a failed-list Retry, however, gives no feedback and no attempt counter (B check 7) |
| 2 | Match System / Real World | 3 | `Selected media · 1 items`; `Match 1 of 2` counts source lines, not the term's 46 occurrences |
| 3 | User Control and Freedom | 3 | Undo works cleanly when the DB is healthy (B: 31→29→31, Enter undoes); the gap is a zero-item review session the user never started, whose only exit is a footer key |
| 4 | Consistency and Standards | 2 | Media, Conversations and Prompts use three different list grammars in one destination; Notes' select toolbar is absent at 235x52 but present at 100x30 |
| 5 | Error Prevention | 2 | `Delete permanently` carries no danger colour and sits one space from `Cancel`, which gets the full focus treatment |
| 6 | Recognition Rather Than Recall | 2 | The only explanation for a blocked `○ Generate` is a mouse-hover tooltip; the `○`/`·`/`✓` glyphs are never legended |
| 7 | Flexibility and Efficiency | 3 | Keyboard-first with real review shortcuts, but `/` focuses the rail's global search rather than the Media filter in front of you, and sort vanishes in select mode |
| 8 | Aesthetic and Minimalist Design | 3 | Generally uncluttered; the fixed 52-column list pane truncates a 98-char title to 47 while ~120 columns of reader sit empty |
| 9 | Error Recovery | 2 | A faulting read cascades to disable Export/Select/Review/facets/Trash, and its Retry repeats one sentence forever with no "reopen the app" next step |
| 10 | Help and Documentation | 2 | The footer key ladder truncates at 100x30, dropping `R exit review` — the only documented exit from a session the app starts itself |
| **Total** | | **25/40** | **Acceptable** |

Trend context: the wave-5 fixes are visible and both assessors credit them (the load-failure callout with an inline Retry, the honest storage notes, the human retry reasons, the padded select-mode columns). The score recovers from critique #5's 21 to 25; the deterministic gaps that remain — focus invisibility, scroll loss on a mode change, blocked-reason discoverability, the destructive-button affordance, and a 235-column layout that shows less than the 100-column one — hold it in the Acceptable band.

## Design Specificity Verdict

Authored for this product, with two unauthored seams. The review set is a genuine first-class object (`·`/`✓` row markers, a counting banner, `]`/`[`/`m`/`R`), the reader refuses to lie about storage (`Type: markdown (stored as plaintext)`; `No Markdown formatting to render — showing the stored text` where the toggle would be; `No stored content.` for an empty row), and the export gate prints its own reason inline. None of that is category furniture. The two seams that read as a different application: the Import canvas (generic form-builder output, nine unrelated controls in one flat group) and the Conversations canvas one rail row away (no panel frame, a `-` where Media uses `·`, a raw UUID where Media shows a title).

## The undo P0: adjudicated down to a robustness finding

Assessment A reported a P0 — a bulk-delete Undo that restored nothing, reported false counts (`✗ undo failed · 1 of 2` while 0 were restored), and disabled the rest of the surface until restart. Assessment B, on a separate profile, got the opposite on the same code: a clean delete and undo (rail 31→29→31, Enter undoes). The undo handler itself is sound: it counts rows actually restored and increments the rail by exactly that (`library_screen.py:24537-24552`), and the `N of M` receipt counts genuine restore exceptions mapped through `_retry_failure_reason`. A's failure therefore required a real media-DB fault mid-session (A's profile carried a messy two-media-DB scratch and three restarts), which B never hit. This is the same class as critique #5's P0 / task-31220 — a storage fault, not a deterministic undo bug.

What is deterministic and real, and worth fixing: (1) when the media DB faults, the failure cascades to disable independent reads — Export, Select, Review these, the facet counts and the whole Trash view — that did not need that connection; (2) the fault-state Retry repeats one sentence with no attempt counter and no next step ("Reopen Chatbook to reconnect"); (3) a restore that returns a non-`Mapping` is counted as neither success nor failure, so the receipt count can drift from reality. These are the honest residue of A's P0.

## Priority Issues

- **[P1] Keyboard focus on a list row is invisible.** The focused row's background differs from the selected row's by three units in one channel (`rgb(28,70,102)` vs `rgb(25,68,102)`), with bold+underline shared by both and no glyph — so arrow-key navigation reads as a dead key (A journey 8; B check 11). In a keyboard-first product this is the sharpest contradiction of the brand. Fix: give the focused row a distinct glyph or a clearly separated background, not a 3/255 colour delta. Command: /impeccable colorize.
- **[P1] Blocked actions explain themselves only on mouse hover.** Clicking `○ Generate` produces a byte-identical screen; the reason (`No analysis provider is configured.`) appears only on hover, unreachable by keyboard (A P1; B). Export already does this right — `No destination chosen` printed under the blocked button. Fix: propagate the Export pattern; render the reason inline beside every blocked control. Command: /impeccable clarify.
- **[P1] A faulting media DB cascades and never tells the user how to recover.** One failed read disables Export/Select/Review/facets/Trash, its Retry repeats forever with no next step, and the undo receipt can misreport a count when a restore returns a non-`Mapping` (traced above; A P0 adjudicated, B clean-undo counter-evidence, cross-refs task-31220 / task-31942 / rider task-31972). Fix: scope the failure to the failing read; give Retry a real reconnect and a "reopen the app" fallback message; re-read committed rows for the receipt count. Command: /impeccable harden.
- **[P2] Read → Analysis → Read loses the reader's scroll position.** Switching modes and back repaints the top of the document (A persona; B check 9, reproduced at 100x30). Fix: preserve the scroll offset across mode changes (rider task-31968 is the adjacent reading-position bug). Command: /impeccable optimize.
- **[P2] The 235-column layout shows less than the 100-column one.** A 98-char title truncates to 47 in a fixed 52-column list pane while ~120 columns of reader hold one sentence; at 100x30 the same title wraps in full (A; B check 3). Fix: let the list pane widen when the reader is empty, or wrap the title. Command: /impeccable layout.
- **[P2] The destructive action carries no danger affordance.** `Delete permanently` is painted in ordinary body colours one space from a fully-styled `Cancel`; the theme's `blocked-error: $error` role is unused here (A; B). Fix: give it the `$error` role and at least three cells of separation. Command: /impeccable clarify.

## Persona Red Flags

**Alex (power user):** `/` focuses the rail's global search, not the Media filter; sort disappears in select mode; `Match 1 of 2` for a term occurring 46 times ends "next match" after two presses; Read→Analysis→Read resets scroll.
**Sam (keyboard-only):** the blocked-action reason is hover-only; the focused row is invisible (3/255 colour delta); at 100x30 the footer drops `R exit review`, removing the only documented exit from a review session the app started itself; the `esc focus rail` hint points at a rail not rendered at that width.
**Riley (stress tester):** the emoji flag-pair keyword breaks the list pane frame by 8 columns (the stated ceiling of task-31955's cell-based cut); a filter miss offers no action where Prompts offers New/Import; the empty document is a dead end (`No stored content.` with no re-ingest or source link).

## Minor Observations

The destination header never names the section (Library / Media / Import share one header); counts are stated three times on the landing; the keyword reason suffix truncates with room to spare (`quarterly-…` can't be told from `quarterly-review`); `1 result · Enter opens` shows for an already-loaded row; the More strip is inline and pushes the body down; entering select mode discards the open reader; rail labels rename at the breakpoint (`Conversations`→`Chats`); `Analyze after import` ships checked on a machine with no analysis provider.

## Strengths

1. Honest storage states, stated in the user's terms — the anti-reference "hidden recovery states" deliberately refused.
2. The review set is a real, complete mechanic that survives Console round-trips and a restart, carried by glyphs and words, never colour alone.
3. Failure copy in the ingest queue and the export gate is exemplary and is exactly the pattern the rest of the surface should copy.
