---
id: TASK-441
title: Fix clipped copy on the first-run Get started card
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-24 06:30'
labels:
  - home
  - ux
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live on a fresh profile: step 3 of the Get started card renders "Send your first message  Composer unlocks after" - the sentence is cut off mid-thought at the card width.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The step-3 explainer renders as a complete sentence at default terminal sizes
- [x] #2 Card copy wraps instead of truncating when space is short
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the render path: CONSOLE_SETUP_STEP_THREE_DETAIL (console_onboarding_state.py) is composed with the label into one Static via ConsoleSetupModal._step_text; the actual clipping is in the CSS driving that Static, not the string.
2. Confirm task-389 already widened .console-setup-modal-card (width:66) and set .console-setup-step to height:1 + text-wrap:nowrap + text-overflow:ellipsis. That satisfies AC1 at default terminal size but violates AC2 (it truncates/ellipsizes narrow terminals instead of wrapping).
3. Change .console-setup-step to height:auto; min-height:1; and drop nowrap/ellipsis so Static's default wrap behavior applies -- the Vertical card (already height:auto) absorbs the extra row.
4. Regenerate the generated bundle (tldw_cli_modular.tcss) via build_css.py; never hand-edit it.
5. Update/extend Tests/UI/test_console_setup_card_fit.py: keep the AC1 width-fit pin, replace the ellipsis-property pin with a wrap-property pin, and add real-stylesheet geometry tests (CSS_PATH=bundled tcss) asserting the full sentence renders on one row at 80x24 and wraps onto >=2 rows with every word preserved at a narrow width.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: task-389 (an earlier fix for the same symptom, filed against an
older review) already widened .console-setup-modal-card to width:66 so the
full 58-cell step-3 sentence fits at default terminal sizes (AC1), but its
overflow strategy for narrow terminals was `.console-setup-step { height: 1;
text-wrap: nowrap; text-overflow: ellipsis; }` -- i.e. it deliberately
truncated with an ellipsis rather than wrapping, which is what this review
(rp-ux-review-2026-07-21, filed before task-389's fix landed) captured as
"Send your first message  Composer unlocks after" and what AC#2 explicitly
asks not to happen.

Fix: changed .console-setup-step (tldw_chatbook/css/components/_agentic_terminal.tcss)
to `height: auto; min-height: 1;` and dropped `text-wrap: nowrap` /
`text-overflow: ellipsis`, restoring Static's default wrap behavior. The
.console-setup-modal-card Vertical container already used height:auto, so it
absorbs the extra row with no further change. Regenerated
tldw_cli_modular.tcss via build_css.py (component .tcss is the source of
truth; never hand-edit the bundle).

Verified geometry directly (Textual pilot with the real bundled stylesheet
loaded via CSS_PATH): at 80x24 the step-3 Static stays height=1 and renders
the complete sentence "3. o Send your first message  Composer unlocks after
setup" on one row (AC1). At 50x30 (card capped below its 66-cell width by
max-width:90%) the Static grows to height=2 and every word -- including
"setup", the exact word the review's evidence showed dropped -- is painted
across the two rows (AC2). Confirmed the pre-fix CSS by temporarily
stashing the change and re-running the same probe: at 50x30 it rendered
"...Composer..." with a literal ellipsis mid-sentence, the truncating
behavior AC2 disallows.

Tests: Tests/UI/test_console_setup_card_fit.py -- kept the existing AC1
width-fit pin, replaced the CSS-property pin that asserted nowrap+ellipsis
with one asserting height:auto and the absence of nowrap/ellipsis, and added
two real-stylesheet rendering tests (App with CSS_PATH=the bundled tcss)
that assert the rendered row count and row text directly: one line at 80x24
containing the exact full sentence, and >=2 wrapped rows at 50x30 containing
every word of the sentence with none dropped. 4/4 new+updated tests pass,
plus the untouched CSS-build-integrity and setup-modal/onboarding suites
(124 tests) all still pass.
<!-- SECTION:NOTES:END -->
