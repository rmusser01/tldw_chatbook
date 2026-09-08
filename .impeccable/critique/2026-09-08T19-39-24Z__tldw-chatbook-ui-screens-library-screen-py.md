---
target: "Library ▸ Media (critique #8, post crit7 wave)"
total_score: 31
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 0
timestamp: 2026-09-08T19-39-24Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design-review sub-agent · B: detector/evidence sub-agent, isolated on separate scratch profiles and tmux sockets; parent synthesized). Target tldw_chatbook/UI/Screens/library_screen.py at the current dev tip — after the full critique-#7 fix wave (six findings + Qodo follow-up). Live at 235x52 and 100x30, isolated DB copies, no analysis provider.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | The reader misreports "Reviewing: All media — No items to review" on a plain open, and a 0-result filter leaves the reader painting a gone item |
| 2 | Match System / Real World | 3 | "Reviewing" jargon on a plain read; a raw UUID and "1 messages" in the Conversations reader |
| 3 | User Control and Freedom | 3 | Undo/Restore/Escape/Cancel-default are strong; review mode re-arms by default and accumulates empty sets |
| 4 | Consistency and Standards | 3 | "Export" vs "Export selected"; "Conversations" vs "Chats"; soft-delete "Delete" dimmer than Cancel |
| 5 | Error Prevention | 3 | Confirmations and irreversibility guards are solid; the 0-result reader split-brain is unprevented |
| 6 | Recognition Rather Than Recall | 4 | Match-reason suffixes, `(selected)` markers, per-context footer keymaps, review progress |
| 7 | Flexibility and Efficiency | 3 | Rich keyboard model; arrow eager-loads each row; `s` silently no-ops off the list |
| 8 | Aesthetic and Minimalist Design | 3 | Clean dense grammar; the "Reviewing:" framing and accumulated empty sets are noise |
| 9 | Recognize/Diagnose/Recover from Errors | 3 | Import Retry/Dismiss with cause, provider block names the fix; raw errno in the empty state |
| 10 | Help and Documentation | 3 | Contextual footers, F1, Import/Export copy; no in-surface explanation of "Reviewing"/"Sets" |
| **Total** | | **31 / 40** | **Good (held)** |

Trend for this surface: 24 → 26 → 28 → 25 → 21 → 25 → 31 → **31** (out of 40). The score holds in the Good band. The critique-#7 wave's improvement (25 → 31) is preserved with NO regression: the evidence assessor confirmed all six of its fixes hold. The score did not rise further because critique #8 surfaced new, subtler issues (below) rather than any of the six fixes failing.

## The six critique-#7 fixes — all confirmed holding (Assessment B, measured)

- **Keyboard focus** is a `█` cursor bar (accent rgb(1,120,212)) + `▸` marker, distinct from the selected-row background, moving on Down/Up at both sizes.
- **`/` focuses the Media filter** (`#library-media-filter`), not the rail search; the Prompts route is code-confirmed.
- **Blocked actions** (`Generate`, select-mode `Analyze`) carry an always-visible inline reason with the Settings next step — not hover-only.
- **Zero-selection** shows "Select items to enable." only when rows exist; a successful empty list shows nothing (the Qodo fix holds).
- **Wide layout** paints a full 98-char title with no item open; opening restores the reader.
- **Delete affordance**: `Delete permanently` in the readable error ink (rgb 209,126,146), ~4 cells from a Cancel-defaulted button; and the failed-load Retry escalates to a reopen step. No colour-only states anywhere.

## Two gaps in just-landed fixes (both assessors)

- **The 0-result filter does not clear the reader** (evidence check #6). task-32043 clears the reader when the loaded item is DELETED (confirmed working), but a filter that returns zero results still leaves the reader painting the last-loaded item, with "‹ Back" into an empty list. The fix passed its pin, but the pin's scenario did not match the live filter path — a real gap on 32043's own AC#1.
- **The flag-emoji +2 frame drift persists on the reader's Info "Keywords:" line** (evidence check #10). task-32044 fixed the list-row keyword-reason suffix, but the Info tab renders the raw keyword, and a regional-indicator flag pair there still drifts the frame +2 cells (line 237 vs 235). The fix was scoped to one surface; the same width miscount lives on a second.

## Priority Issues (new)

- **[P2] The reader is always framed as "Reviewing: All media", and opening media mints empty review sets.** A plain read wears review scaffolding it never asked for, and the Sets history accumulates empty "No items to review" sets. Fix: create a set only on "Review these"; default the header to the title (as it already does at 100×30 and after `R`).
- **[P2] The 0-result filter reader split-brain** (the 32043 gap above). Fix: extend the reader-clear to the settled-empty filter path that the live scenario actually takes.
- **[P3] The flag-emoji drift on the Info tab** (the 32044 gap above). Fix: apply the flag-pair-aware width to the Info "Keywords:" line, not only the list suffix.
- **[P3] Toggling Read-later while reading resets scroll to top** (a pure Read→Analysis→Read round-trip preserves it). Fix: update the flag in place rather than reloading the reader.
- **[P3] Label/marker drift and uneven disabled-reason coverage** — "Export" vs "Export selected", "Conversations" vs "Chats", soft-delete "Delete" dimmer than "Cancel", "1 messages", a raw UUID; and the disabled Delete lacks the inline reason Export/Review show.

## Design Specificity Verdict

Highly specific — a bespoke "precision workbench" grammar, not a transplanted template (both assessors). The one place specificity works against the brand is the review-set concept bleeding into plain reading. The detector is not applicable (a Textual TUI has no DOM; `.py` is unscannable), so all of B's evidence is live-measured.

## What's Working

The destruction-safety arc is the peak — soft-delete receipt + Undo, Trash Restore, and an irreversible delete gated by a warning, danger colour, and a Cancel default. Honest blocked/failed states inline with the next step. Recognition-rich list and reader. These carried the surface into the Good band and keep it there.
