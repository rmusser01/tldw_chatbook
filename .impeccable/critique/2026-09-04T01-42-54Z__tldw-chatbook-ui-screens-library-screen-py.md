---
target: Library ▸ Media (library_screen.py)
total_score: 28
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
timestamp: 2026-09-04T01-42-54Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design review · B: detector/evidence; B re-spawned once mid-run after a stall — mechanical pass only, no cross-contamination)

Target: Library ▸ Browse ▸ Media (Operate mode) at dev tip 60d95fc7d2 (post re-critique fix wave 2: #2358/#2359/#2361). Live at 235×52 and 100×30, plus source.

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Auto-resume re-arms the banner ("Read later — 1 of 2") over a document that is not in the set |
| 2 | Match System / Real World | 3 | "Sets" is a cipher until the feature is already learned; tooltip is mouse-only in a terminal |
| 3 | User Control and Freedom | 3 | Picker Dismiss is a one-click soft-delete with no confirm, no undo, no surfaced reopen path |
| 4 | Consistency and Standards | 3 | "Review these" opens item 1 in the Reader; "Review selected" does not |
| 5 | Error Prevention | 3 | Bulk delete choreography is textbook; Dismiss sits adjacent to every picker row unguarded |
| 6 | Recognition Rather Than Recall | 2 | Sort strip clips "Title A-Z" and renders "Title Z-A" nowhere; no reviewed-marks on list rows mid-walk |
| 7 | Flexibility and Efficiency | 3 | Real keyboard core (] [ m R s space /) but no key for Sets/Review these/select-all; first ] after resume is a wasted sync press |
| 8 | Aesthetic and Minimalist Design | 2 | Reader content capped at 75vh strands ~10 rows at 52-row terminals; default-open find input spends 3 rows on every fresh item |
| 9 | Error Recovery | 3 | Delete: undo receipt + durable Trash. Dismiss: nothing |
| 10 | Help and Documentation | 3 | Contextual footer teaching is exemplary; the user guide's create-promise is false for the Select path |
| **Total** | | **28/40** | **Good (lower edge of the band)** |

## Design Specificity Verdict

**LLM assessment**: Authored for this product, decisively. The house grammar — state-in-text everywhere ("✓ Newest", "Read (selected)"), ○ disabled markers with reasons, honest-footer discipline that retires keys at boundaries and on completion, "typing in field" while focus is in an Input, provenance honesty in the Info tab, the receipt grammar ("✓ deleted · N items · in Trash") — is not a category template. The weakness is fit-and-finish, not identity.

**Deterministic scan**: `detect.mjs --json` over the five media widgets returned `[]` (exit 0) — no-signal, since the engine targets web markup and these are Python TUI files. Mechanical greps: 0 hex colors in all five files (clean); 22 `Button(` constructors in library_media_canvas.py with 12 carrying tooltips (5 kwarg + 7 post-construction incl. the two gate helpers that stamp reasons onto disabled actions); every Unicode state marker (✓ ○ ▸ ☑ ☐ ✕) co-occurs with words in the same label — none is a bare glyph. The one severity oddity B flagged (the same "Review-set storage is unavailable." string is `warning` at one site, `error` at two) is documented intentional in code: ambient severity one notch below gesture severity (task-31225 rider).

**Visual overlays**: Not applicable — terminal UI, no browser surface.

**Where A and B agree**: B's live captures independently confirm A's strengths — the wave-2 fixes hold (Clear filter visible and adjacent on a "zz" miss with an honest "No media matched 'zz'." empty state; the type chooser renders its options; select mode swaps to ☐ rows, ○ disabled bulk actions, and a "space toggle selection" footer). B's verbatim footer-code quote matches the behavior A saw in five distinct states.

## Overall Impression

The instrument-panel discipline this surface has been building for six phases now genuinely lands: footer honesty, receipts, and completion ceremony are brand-defining. What drags the score is that the review-set feature's two entry/re-entry seams break its own core promise — continuity. You invoke "Review selected" and see nothing happen; you come back to a saved place and the banner points at the wrong document. The single biggest opportunity: make the review-set lifecycle as honest as the footer already is.

## What's Working

1. **Honest-footer discipline is a signature.** Keys are removed when a text field would swallow them ("typing in field"), when boundaries make them no-ops, and when completion retires them ("m toggle reviewed | R finish review | All 6 reviewed"). Verified live in five states and in code (`_review_footer_entries`, `_library_footer_shortcuts_for_current_state`).
2. **The walk model is well-designed.** Advance-marks-what-you-leave, [ never marks, final ] as completion gesture, tombstone-skipping with live-only progress; `review_set_state.py` is pure and correct about edge cases.
3. **Destructive-action choreography.** Danger isolated on its own row, ○ disabled forms with reasons, armed confirm naming both recovery paths ("undo right away, or restore later from Trash"), receipt, durable Trash.

## Priority Issues

1. **[P1] "Review selected" doesn't start the review.** The bulk Review action toasts "Reviewing 2 items." then leaves the user in select mode with unchecked boxes and a blank reader pane. Verified in code: both create paths call `_open_library_media_viewer(items[0])` (library_screen.py:39305), but nothing on the selection path ever exits select mode (`_library_media_select_mode` is cleared only by Done/bulk-delete, lines 24249/24526), so the viewer never surfaces. Contradicts the shipped doc's promise ("Creating a set activates it and opens its first item in the Reader"). **Why**: the user's invoked feature is invisible at the moment of invocation. **Fix**: exit select mode before the viewer open on the selection path, exactly as "Review these" behaves.
2. **[P1] Auto-resume restores the frame, not the item.** Leaving Media and returning re-arms banner + footer over an off-set document; the first ] is a silent sync that advances nothing. Verified in code: the once-per-set gate (`_review_set_auto_resumed`, library_screen.py:39576) returns without opening the cursor item on re-entry, while the banner re-arms unconditionally. **Why**: the status line disagrees with the visible document exactly when the feature's promise is "your place is saved." **Fix**: whenever the review banner/footer arms, load the cursor item — or suppress the banner until the viewer matches.
3. **[P1] Sort chooser clips options off-pane.** Renders "✓ Newest  Oldest  Title A-" in the ~38-col items pane; "Title Z-A" is invisible with no overflow cue, and a keyboard user can select an option that is rendered nowhere. Verified in code: the sort chooser is a horizontal `compose_library_choice_strip` (library_media_canvas.py:690-701) while the type chooser is a vertical `OptionList` (line 681). **Fix**: use the type chooser's vertical OptionList, or wrap the strip to two rows like the toolbar's multi-row grammar (task-30043).
4. **[P2] Dismiss is a one-click, adjacent, unconfirmed soft-delete with no surfaced recovery.** Every picker row is [open][Dismiss] side by side; Dismiss fires immediately and closes the dialog; no reopen path exists in the UI. A mid-walk set with many done-marks dies to one mis-click — a hidden recovery state, the product's own stated anti-reference. **Fix**: undo on the toast (the delete-receipt pattern already exists) or an armed confirm on incomplete sets.
5. **[P2] The Reader strands its vertical space.** At 52 rows the content box ends near row 39 (`#library-media-viewer-content { max-height: 75vh }`) leaving ~10 blank rows while long documents scroll inside; the default-open "Search content…" input spends 3 more rows above; single-page pagers still show two dead ○ controls. **Fix**: let the content box fill remaining height; collapse the find input until Find is invoked.

## Persona Red Flags

**Alex (impatient power user)**: three unnecessary steps between "Review selected" and the first document (discover nothing happened → Done → ]); no keybinding for Sets, Review these, or select-all; the resume-sync eats his first ]; 38-col rows truncate half the titles on a 235-col terminal so he identifies items by mouse tooltip.

**Sam (keyboard-only, shape-grammar dependent)**: genuinely well served — ☑/☐/▸/○ markers all carry words (B verified every code-line marker has text co-occurrence). Two real flags: the sort strip lets him arrow to and select an option that is not rendered anywhere on screen; "✓" means *active* on the picker's top row one line above rows where ✓ means *completed* — an easy misread as "Read later is finished."

## Minor Observations

- Pane-join artifacts at the reader pane's top/bottom edges ("┌───┐─────", "└───┘─────").
- Trash header truncates: "Local Trash · 0 i".
- Find counts lines, not occurrences, with tautological copy ("Match 1 of 1 matches" over ~30 visible hits).
- The More menu replaces the entire body while open — disorienting for a disclosure.
- Reader tab is sticky across items: a walk begun after visiting Info opens item 1 on Info, not Read.
- Escape's label roams ("esc focus rail" / "esc focus Library" / "esc focus Items") — precise, but three names for "go left."
- Silent active-set replacement (creating a set deactivates the current one wordlessly) + anonymous auto-names ("2 selected items") make picker rows indistinguishable later. [P3]

## Questions to Consider

1. The core act here is *reading*, yet the reading surface is the only region refusing its space — should an active review walk collapse the list pane into a focus mode?
2. Auto-mark-on-leave equates "paged past" with "reviewed" — ]]]]]]  completes a set without reading a word. Is completion a claim the UI should make on the user's behalf?
3. If review sets are durable workflow objects worth a picker, why can't they be named, and why is their only lifecycle exit an unconfirmed, unrecoverable Dismiss?
