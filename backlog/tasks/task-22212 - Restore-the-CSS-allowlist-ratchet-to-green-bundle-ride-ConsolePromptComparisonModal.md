---
id: TASK-22212
title: >-
  Restore the CSS allowlist ratchet to green: bundle-ride ConsolePromptComparisonModal
status: To Do
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

- [ ] The consolidation suite is green on dev (modal CSS rides the bundle; no new allowlist entry)
- [ ] `css/tldw_cli_modular.tcss` regenerated via `build_css.py`, both committed
- [ ] First open of the compare-drafts modal registers no new stylesheet source (source-count probe or the suite's own check)
