---
id: TASK-22212
title: >-
  Restore the CSS allowlist ratchet to green: bundle-ride ConsolePromptComparisonModal
status: Done
assignee: []
created_date: '2026-08-24'
labels:
  - css
  - console
  - dev-red
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22212).

Dev is red at tip: `Tests/UI/test_widget_css_consolidation.py::
test_class_level_css_stays_within_the_allowlist` FAILS on pristine `a71e62e4b` (run
first-hand during this review). PR #2053 (the tip merge) added
`Widgets/Console/console_prompt_comparison_modal.py:23` with a class-level `DEFAULT_CSS`
that is neither in `_UNCONSOLIDATED_CSS_ALLOWLIST` nor bundle-ridden (0 hits in
`css/*.tcss`; the sibling `ConsoleDispatchRecoveryRegion` from the same era is correctly
bundled). Every PR inherits this red. The modal is imported at Console module scope
(`chat_screen.py:523`), so first use also registers a new live stylesheet source — one
step toward the LRUCache(64) parse-cache cliff the ratchet exists to prevent.

## Acceptance Criteria

- [x] The consolidation suite is green on dev (modal CSS rides the bundle; no new allowlist entry)
- [x] `css/tldw_cli_modular.tcss` regenerated via `build_css.py`, both committed
- [x] First open of the compare-drafts modal registers no new stylesheet source (source-count probe or the suite's own check)

## Implementation Plan

1. Rename the modal's class CSS to the screen-tier bundle attribute (the TraceExportDialog precedent)
2. Regenerate the bundle with build_css.py; run the consolidation suite + preflight

## Implementation Notes

Renamed `ConsolePromptComparisonModal.DEFAULT_CSS` to `BUNDLED_SCREEN_CSS` — the exact
convention its sibling `TraceExportDialog` (also a `SafeModalDismissMixin, ModalScreen`)
already uses. `build_css.py` lifts the block into `screen_css_self.tcss` /
`screen_css_scoped.tcss` (2 + 8 selectors verified present); `tldw_cli_modular.tcss`
changed by its Generated timestamp only, since screen-tier CSS lives in the screen sheets.
With no class-level `DEFAULT_CSS` left, Textual's `_post_register` has no per-class
stylesheet source to add, so first open registers no new source — the mechanism the
ratchet pins, now green: `Tests/UI/test_widget_css_consolidation.py` 33 passed (the
allowlist test FAILED on pristine dev before this change; no allowlist entry added).
preflight.sh all green. CSS content is byte-identical — only its delivery tier changed.
Files: `Widgets/Console/console_prompt_comparison_modal.py`, three generated `css/*.tcss`.
