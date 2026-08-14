---
id: TASK-15993
title: 'Generated screen CSS sheets share $variable scope sheet-wide'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - css
  - hardening
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_chatbook/css/widget_css.py:243-249` sends top-level trivia — including `$var: value;` definitions — to both output streams, so a screen block's local variable fallbacks are in scope for every block emitted after it in the same generated sheet (e.g. `EmojiPickerScreen`'s `$ds-*` fallbacks in `screen_css_scoped.tcss` cover all later blocks). Verified INERT today: the only `$ds-*` uses sit inside EmojiPickerScreen's own block, and the blocks after it use no custom variables. But the next screen consolidated below it silently inherits those fallbacks — the same collision class the separate-sheet design exists to avoid, narrowed from app-wide to sheet-wide rather than eliminated. Fix direction: per-block variable isolation in the builder (emit block-local definitions only into that block's slice, or fail the build on cross-block shadowing). Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A $variable defined in one BUNDLED_SCREEN_CSS block cannot silently apply to a later block's rules (isolated or build-failed)
- [ ] #2 A build-time or test-time check pins the property so regrowth cannot reintroduce it
- [ ] #3 Existing generated sheets re-verified byte-stable (or the delta accounted for) after the change
<!-- AC:END -->
