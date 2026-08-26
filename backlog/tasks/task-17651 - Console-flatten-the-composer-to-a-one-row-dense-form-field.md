---
id: TASK-17651
title: 'Console: flatten the composer to a one-row dense-form field'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17'
labels:
  - console
  - ux
dependencies:
  - task-17650
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The composer spends 4 chrome rows on every draft height: `COMPOSER_CHROME_ROWS = 4` (border 2 + padding 2) is hard-coded in ConsoleComposerBar and re-asserted inline on every draft repaint, so the widget renders 5-8 total rows to show 1-4 draft rows. The 2026-08-17 audit also proved the composer's focus-border CSS is dead code: `frame_console_region()` writes the border as an inline style, which outranks every stylesheet rule (the composer renders `#6f7782` even while carrying `console-composer-focused`).

Flatten the composer to the task-1586 dense-form convention: a one-column left edge at rest, thick `$accent` edge plus focus background when focused (three concurrent signals, no dimensional change on focus). The draft keeps its existing 1-4 row auto-grow; total composer height becomes 1-4 rows instead of 5-8. Below the workspace grid's closing border, the control deck (composer, chips, footer) reads as dense-form rows rather than a framed region — the workbench frame grammar closes at the grid.

Owner decisions (2026-08-17): keep ALL send affordances (disabled-reason banner, Send, Dictate); keep the 4-row draft cap (raising it is a separate follow-up); the Composer collapse control is KEPT and repurposed as a same-height content swap to the run-status variant (status + Stop + Expand) — it is no longer a row-saving lever, and its height-economics tests change accordingly.

Also in scope (reallocated from TASK-17650 during implementation review): consolidate the FOUR stacked separator rows between the last transcript line and the composer content — `#console-native-transcript`'s own border, the `#console-transcript-region` inline frame, the `#console-workspace-grid` inline frame, and the composer's top border — down to ONE separator. The transcript's border rows are its keyboard-focus affordance (`:focus` recolors them; pinned by `test_console_transcript_focus_uses_stable_border_geometry`), so removing them requires a replacement focus treatment that is dimensionally stable and non-obscuring, designed together with the composer's dense-form focus edge. Compact mode's existing border-drop (TASK-2154.1) is the precedent for the transcript side.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The default composer renders exactly 1 row with all current children on the content row (Composer collapse control, Menu, draft, disabled-reason banner, Send, Dictate) and grows with the draft to at most 4 rows
- [x] #2 The focus treatment follows the dense-form convention, causes no dimensional change, and demonstrably renders on the running screen (i.e. is not overridden by inline frame styles)
- [x] #3 The composer collapse control still exists and swaps to the run-status variant at the same height; updated tests encode the new economics (collapse no longer promises extra transcript rows)
- [x] #4 Exactly one separator row renders between the last transcript content line and the composer content row (transcript border, region frame, grid frame, and composer top border consolidated), and the transcript keeps a visible, dimensionally-stable keyboard-focus treatment at all sizes
- [x] #5 The transcript gains at least 7 more rows at 150x44 versus the post-TASK-17650 baseline (4 composer chrome + 2 transcript border + 1 duplicate frame row)
- [x] #6 All affected geometry pins are updated to the new contract (composer-collapse suite, internals-decomposition geometry blocks, non-obscuring-focus composer AND transcript stable-border rules), using bundle-loading harnesses wherever geometry is asserted
- [x] #7 The setup-blocked/first-run state renders correctly with the flattened composer (the disabled-reason banner is the widest single-row competitor) and the draft's usable width at 150 columns does not regress
- [x] #8 User Guide Console page updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED pins first: single-row default geometry + single-separator contract (bundle harness), watched fail.
2. COMPOSER_CHROME_ROWS 4 -> 0; init heights from constants; CSS padding 0 1, max-height 4, $ds-console-composer-height 1.
3. frame.py: `bottom` param; composer removed from the frame grammar (CSS dense-form edge owns it); grid children (rails, handles, region — both compose paths incl. the recovery subclass) suppress bottom edges so the grid's border is the single separator.
4. Transcript border none in both states (compact-mode drop generalized); TASK-359 focus painter extended to the region's column lines with per-edge writes (the shorthand would resurrect suppressed edges).
5. Focus tests with PAINTED assertions; mutation-tested; collateral pins updated deliberately (heights, frame contracts, click boundaries, snapshots helper).
6. Live probes (ready/long-draft/collapsed/setup-blocked/compact, both chip placements); docs + DESIGN.md; lessons entry.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The composer is now a dense-form one-row bar: `COMPOSER_CHROME_ROWS = 0`, no border box, `padding: 0 1`, growing 1-4 rows with the draft. It left the workbench frame grammar entirely — `frame_console_region` gained a `bottom` param, the grid keeps its full border as the bottom stack's single separator, and every grid child (rails, handles, transcript region — both compose paths, including the provider-recovery subclass) suppresses its bottom edge. The transcript widget draws no border at any size (the TASK-2154.1 compact drop generalized); its keyboard-focus cue is the region's column lines, recolored by the TASK-359 pane-stop painter (extended with per-edge writes — the `border` shorthand would silently resurrect suppressed edges) plus a scrollbar accent.

Headline catch, recorded in lessons-testing-evidence.md: removing the border box ACTIVATED the global `*:focus` outline on the composer (and latently the transcript) — corner glyphs overpainting the row while every style-level read stayed pristine. Caught only by the painted row-map probe; fixed with `outline: none` opt-outs per the reset's own DataTable pattern, and the focus tests now pin the painted first cells (`│`/`█`, never `┌─`). A second self-inflicted catch: the freshly written pin `"outline:" not in focus` banned the cure along with the disease — the updated pin bans `outline: solid/heavy` and REQUIRES the opt-out.

Contract updates, all deliberate: composer heights 5-8 -> 1-4 across the collapse suite and internals-decomposition; collapse is a same-height content swap (its +4-transcript-rows promise retired); the textual-web click-boundary forgiveness tests inverted — the ±1 rows now belong to the neighboring strips (the production hit-test needed NO change; it always keyed off the composer's real box); frame-contract tests and the snapshots' `_assert_solid_border` helper learned the suppressed bottom edges; the composer's CSS pins moved to the dense-form grammar. Bundleless-harness tests now pin only the inline no-frame fact, with CSS edges pinned in bundle-loaded tests.

Evidence: 838 passed on the 16-file sweep (the screen-size ratchet stays red exactly as on dev — task-3751; this branch adds +13 net lines to chat_screen.py, with the new logic in frame.py/CSS). Live probes at 150x44: transcript region 29 -> 33 (+4 outer, +7 content lines with the interior border rows), draft width 70 -> 71, single `└──` separator, growth capped at 4, collapse/setup-blocked/compact all sane in both chip placements. RED-first on the two core pins; mutation tests on both focus mechanisms.

Files: `console_composer_bar.py`, `UI/Console_Modules/frame.py` + `transcript.py` + `provider_continuation_recovery.py`, `chat_screen.py`, `css/components/_agentic_terminal.tcss` + `core/_variables.tcss` (+ bundle), tests: `test_console_composer_collapse.py`, `test_console_internals_decomposition.py`, `test_non_obscuring_focus_contract.py`, `test_workbench_visual_snapshots.py`; docs: `Docs/User_Guide/console.md`, `DESIGN.md`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
