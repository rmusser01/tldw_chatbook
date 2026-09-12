---
id: TASK-32476
title: Tokenize features/_chat.tcss end-to-end as the migration reference
status: Done
assignee:
  - '@kimi'
created_date: '2026-09-11'
labels:
  - ui
  - css
  - design-system
dependencies:
  - task-32475
priority: medium
---

## Description (the why)

ADR-150 introduced the design-token system with component-level exemplars
(`_buttons.tcss`, `_forms.tcss`). The constitution promises a ratchet where
legacy feature sheets migrate opportunistically, but there is no complete
feature-sheet migration yet for contributors to imitate. `_chat.tcss` is the
canonical first target: a full real screen with spacing, control sizing,
motion, state surfaces, and feature-scoped geometry — exactly the mix that
demonstrates what tokenizes and what deliberately stays literal.

## Acceptance Criteria (the what)

- [x] Every spacing, sizing, motion, opacity, typography, and component-state
  value in `_chat.tcss` that maps 1:1 to an existing token consumes the token
- [x] Feature-specific geometry (fixed gutter widths, textarea expansion
  heights, one-off pane minimums) remains literal with its existing comments
- [x] No visual change: all substitutions are value-identical aliases
- [x] No new tokens invented in the feature sheet; anything the catalog
  lacked was either added to `_variables.tcss` first or left literal
- [x] Bundle + screen-owned sheets regenerated; CSS test suite passes
- [x] Constitution updated to name `_chat.tcss` as the migration reference

## Implementation Plan (the how)

ADR required: no
ADR path: N/A — direct implementation of ADR-150
Reason: the governance decision exists; this is execution of its ratchet.

1. Map every literal in `_chat.tcss` to its value-equal token; keep
   semantically distinct theme vars (`$error`, `$primary`) literal
2. Rebuild bundle, run CSS suite, open PR
3. Update `backlog/docs/design-language.md` §6 to point at the migration

## Implementation Notes

Tokenized `_chat.tcss` end-to-end (PR #2632) as the ADR-150 migration
reference. Approach: every literal mapped to a value-identical token
(spacing scale + semantic aliases, `$ds-control-height` / `-compact`,
`$ds-duration-fast`, `$ds-opacity-dim`, `$ds-text-strong` / `-emphasis`,
`$ds-surface-raised` / `-panel`, `$ds-text-primary` / `-muted`,
`$ds-grid-line`, `$ds-disabled-bg`). Deliberately left literal:
feature-specific geometry (3-cell gutter, `min-width: 40`, textarea
expansion heights, the commented `height: 4` override) and semantically
distinct theme vars (`$error` stop state, `$primary` bubbles,
`color: white`). No tokens invented in the feature sheet. Constitution §6
now names this sheet as the migration reference. Verification: 49 CSS
tests passed (governance, bundle sync, build integrity, token regressions);
bundle regenerated — chat is not screen-split, main bundle only.
