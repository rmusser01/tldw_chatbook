---
target: Library ▸ Media surface (post review-sets program)
total_score: 24
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
timestamp: 2026-09-03T12-58-19Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: live design review sub-agent · B: detector/evidence sub-agent)

Target: Library ▸ Browse ▸ Media (list, toolbar, select mode, Reader, Trash, review sets) — live tmux at 235×52 + 100×30, seeded 6-item profile. Session caveat: a second app instance from another worktree shared the user profile during Assessment A; findings that depend on persistence are attributed accordingly.

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | Footer goes stale at the set-completion moment; list view gives no hint a set is active; "Loaded in Reader" marker persisted after the Reader emptied |
| 2 | Match System / Real World | 3 | "Remove later" reads as deferred removal; "Open manager" jargon; "Match 1 of 1 matches" counts blocks, not occurrences |
| 3 | User Control and Freedom | 3 | "Review set dismissed." has no Undo (bulk delete does); no keyboard route back to a paused set |
| 4 | Consistency and Standards | 3 | Same width: Conversations stacks its toolbar readably, Media chops horizontally; `]`/`[` silently change meaning browse↔set |
| 5 | Error Prevention | 3 | Armed confirm + danger placement strong; undermined by `Tr`/`R` mystery-meat buttons two cells apart |
| 6 | Recognition Rather Than Recall | 1 | Resting toolbar renders `t so E Tr R Se`; select-mode actions render as bare `○ ○ ○`; reviewed state of the current item shown nowhere; set name never shown |
| 7 | Flexibility and Efficiency | 3 | Walking is keyboard-first, but no key creates a set or opens the picker; `]` dies at the last item |
| 8 | Aesthetic and Minimalist Design | 2 | "Image preview unavailable" + "Retry preview" failure chrome atop every plain document; verbose state prefixes displace row titles to "Quart"/"SQLit" |
| 9 | Error Recovery | 2 | Delete has receipts + Trash; the review-set subsystem fails silently end-to-end by design (`service is None → return`, silent auto-resume) |
| 10 | Help and Documentation | 2 | Footer teaching + stamped user guide, but the guide promises a completion gesture and auto-resume the product doesn't deliver |
| **Total** | | **24/40** | **Acceptable** |

## Design Specificity Verdict

**LLM assessment (A):** Authored at the model layer, category-generic-to-broken at the presentation layer. The interaction model is unmistakably this product's: a footer that re-teaches itself per mode with honest escape labels, and review-set semantics (advance-marks-what-you-leave, tombstones, live-only progress, title snapshots surviving deletion) that no template produces. But the rendered resting state betrays it: at 235×52 the six-button toolbar renders as `t so E Tr R Se`, select-mode actions as three bare `○`, and delete-confirm safety copy clips mid-word.

**Deterministic scan (B):** The bundled detector targets web markup: the five Python widget files scanned clean as plain text (exit 0, zero findings — "unscannable-by-design", not "clean"). A TCSS fallback scan found 2 advisory findings (`#6f7782` border, `rgb(245,245,245)`) — both in Console styling sources, outside this surface. Mechanical greps agree with A's strengths: 0 hardcoded hex in the media widgets, every unicode state marker (`✓ ○ ▸ ☑ ☐`) is text-paired in code, notice copy inventory is clean and specific, all 14 inline `styles.` assignments are sizing-only. B's independent 100×30 capture reproduced the toolbar fragmentation (`ty so Ex Tr Re Sel`) and a clipped Reader action row — the legibility failure is not width-specific, it is the pane's grammar.

**Visual overlays:** not applicable (terminal UI, no browser target).

## Overall Impression

A genuinely well-modeled surface wearing the wrong clothes at rest. The safety grammar (armed confirms, receipts, Trash) and the review-set data model are better than most shipping products; the footer-as-contract is the brand executed correctly. But the two moments that define the surface — glancing at the toolbar to act, and finishing a review set — are respectively illegible and dead. Fix the finish line and the resting toolbar and this jumps a band.

## What's Working

1. **The footer as a live contract** — keys appear only when they work, escape labels state their true effect per focus context, and set progress rides in the same line. Recognition-over-recall done the terminal-native way.
2. **Delete-safety grammar** — count-naming armed confirm, danger button at the far end, `✓ deleted · N items · in Trash` receipt with Undo, honest Trash empty copy. Layered and recoverable.
3. **The review-set data model** — tombstones, live-only progress, title snapshots for deleted items, cap-with-notice. Deletion-honest design most products get wrong.

## Priority Issues

- **[P1] The documented completion gesture is dead, and the footer lies at the finish.** `check_action` for `library_media_next_item/prev_item` gates `]`/`[` on browse-row adjacency and never consults the active review set (confirmed at `library_screen.py:30772`), so at the last row `]` is disabled while the footer still advertises "] next in set" — reproduced live (footer pinned at "6 of 6 · 5 reviewed"; the final mark never lands). It also misfires whenever set order ≠ browse order (filtered/selection sets). Separately, the walk's clamp branch marks + refreshes completion but skips the viewer sync, so even a working final step would leave the footer stale. **Why:** the single moment the feature builds toward is a silent no-op; the doc promises it. **Fix:** `check_action` returns True for these actions whenever `_review_set_active()`; add the viewer sync on the no-target branch; give completion an actual moment (e.g. "All 6 reviewed" notice + offer to dismiss). **Suggested command:** /impeccable harden
- **[P1] The Items-pane toolbar is illegible in the resting 3-pane layout.** At the pane's 40-col floor all six compact buttons chop to `t so E Tr R Se`; select mode renders `0/selec/ted` + three bare `○`; the armed-confirm safety copy clips to "You can und / restore later from Tr". Tooltips are mouse-only. B reproduced independently at 100×30. **Why:** the actions the surface exists for are unrecognizable in its default state; the accessibility markers become the entire label. **Fix:** below a measured width, switch to the stacked one-action-per-row grammar the Conversations canvas already uses, or overflow into a single "Actions ▸" chooser strip (machinery exists). **Suggested command:** /impeccable adapt
- **[P1] Review-set failures are silent end-to-end.** Every failure path is a deliberate silent return (`service is None → return`; auto-resume "any failure is silent"). During Assessment A the Sets button produced no response across three activations — likely environmental (a second app instance shared the profile and collections writes stopped landing), but that is exactly the point: a wedged storage path is indistinguishable from the feature not existing, and "progress saved between visits" can silently be false. **Why:** the product's own anti-reference is "hidden recovery states". **Fix:** the picker button always responds (a `service is None` press says storage is unavailable); surface a health notice when the collections DB cannot be opened/written. **Suggested command:** /impeccable harden
- **[P2] The current item's reviewed state — and the set's name — are displayed nowhere.** `m`'s only feedback is the aggregate counter changing; landing mid-set you must toggle twice to learn the state. **Fix:** a per-item glyph in the footer segment or a one-line set header ("Reviewing: All media — 3 of 6 · ✓") above the Reader. **Suggested command:** /impeccable polish
- **[P2] Failure chrome on the primary reading surface.** Every plain document's Read tab leads with "Image preview unavailable — showing complete stored text" + a persistent "Retry preview" button above the content, and the wide-mode row prefix "Loaded in Reader" consumes ~28 of ~35 label cells, leaving "Quart"/"SQLit". **Fix:** surface preview status only when an image was expected; use the compact prefix grammar (exists) or a leading glyph in narrow layouts. **Suggested command:** /impeccable distill

## Persona Red Flags

**Alex (impatient power user):** cannot start a review set from the keyboard (footer offers `/ F6 s esc`; "Review these" is a click-only button currently labeled `R`, two cells from `Tr`); his `]` rhythm dies at the last item with the footer still advertising it — he presses it three times, doubts himself, leaves the set uncompleted; after R-exit, resuming means mousing to "Sets".

**Sam (keyboard-only / no color-only meaning):** the non-color vocabulary is strong on paper (`☐/☑`, `▸`, `✓`, focus boxes) — but in the resting layout the `○` disabled markers ARE the labels for the three bulk actions, and the reason tooltips are hover-only, so Sam can never read what a control is or why it's off; the clipped confirm copy means arming a destructive action whose safety sentence reads "You can und / restore later from Tr".

## Minor Observations

"Match 1 of 1 matches" double-counts the word and counts blocks; duplicate "All media" sets can be created back-to-back (picker rows differ only by progress); `R` the key exits review while `R` the chopped button creates one; "Open manager" unexplained; pager disabled-reason itself truncates; Reader keeps showing the prior item while browsing Trash; two advisory TCSS color literals (Console-scoped) are outside DESIGN.md's palette.

## Questions to Consider

1. If the resting 3-pane layout cannot render six labeled buttons, is a toolbar the wrong grammar for this pane — should list actions live in one chooser strip and the pane spend its columns on titles?
2. A review set is the product's first real "workflow object" — why is its entire runtime UI a footer string? Would a one-line set header above the Reader solve name-invisibility, per-item state, and list-level awareness at once?
3. For a persistence feature, is the silent-failure doctrine ever acceptable when the write path is down — what would the brand's own "no hidden recovery states" commitment call this?
