---
id: TASK-32475
title: Design token system and design-language constitution for UI
status: Done
assignee:
  - '@kimi'
created_date: '2026-08-05'
labels:
  - ui
  - css
  - design-system
dependencies: []
priority: high
---

## Renumbering provenance

Originally TASK-2511 (2026-08-05); the backlog was regenerated while the work
waited on verification, and on dev the reissued id collided with an unrelated
task. Renumbered to TASK-32475 when the branch was rebased onto origin/dev
(b277f16193) for PR.

## Description (the why)

New screens and widgets are currently styled by inventing literals: thousands
of hardcoded dimension declarations and hex literals live across the TCSS
sources, and 250+ Python call sites mutate `styles.*` directly. The existing
`$ds-*` vocabulary in `core/_variables.tcss` covers semantic color and status
but nothing else, and the design rules that do exist (non-obscuring focus
contract, control-edge convention) survive only as comments. Every new UI
session therefore re-opens the same architectural debates. The goal is a
governed design language: anything that must stay consistent across screens
exists as a token or a written rule, so agents and contributors compose UI
from the language instead of inventing it.

## Acceptance Criteria (the what)

- [x] `core/_variables.tcss` token catalog covers spacing scale, control
  sizing, motion durations, opacity, typography emphasis, and component-state
  semantics, alongside the existing color/status tokens
- [x] A design-language constitution exists under `backlog/docs/` describing
  the token catalog, naming rules, layout laws, interaction rules, and the
  process for adding tokens, and is linked from AGENTS.md
- [x] A governance pytest fails when a `$ds-*` token is referenced but not
  defined in `core/_variables.tcss`
- [x] Exemplar sheets (`components/_buttons.tcss`, `components/_forms.tcss`)
  consume tokens with no visual change (values map 1:1)
- [x] The committed CSS bundle reproduces from sources (bundle sync guard
  passes) and the existing CSS test suite still passes
- [x] ADR created and linked (long-lived UX structure decision)

## Implementation Plan (the how)

ADR required: yes
ADR path: backlog/decisions/150-design-token-system-and-design-language.md
Reason: long-lived UX/application structure and governance decision per
AGENTS.md ADR criteria.

1. Extend `core/_variables.tcss` with spacing/sizing/motion/opacity/
   typography/state token sections; scalar single-value tokens only (Textual
   variable substitution is per-token)
2. Tokenize exemplar sheets `_buttons.tcss` and `_forms.tcss` plus the
   `.sidebar` layout law with 1:1 value mappings (pure refactor)
3. Add `Tests/UI/test_design_token_governance.py`: every `$ds-*` referenced
   in any TCSS source must be defined in `core/_variables.tcss`; hex ratchet;
   post-ADR sheets must use the spacing scale
4. Write `backlog/docs/design-language.md` constitution; link from AGENTS.md
5. Rebuild the bundle, run CSS tests, open PR against dev

## Implementation Notes

Implemented the design-token system additively (ADR-150), no visual changes;
PR #2628 against dev.

- `core/_variables.tcss`: added token sections — spacing scale
  (`$ds-space-0..3` + semantic aliases `inline/stack/section/inset`),
  control sizing (`$ds-control-height`, `-compact`,
  `$ds-textarea-min-height`), motion (`$ds-duration-fast/medium/slow`),
  opacity (`$ds-opacity-dim`), typography emphasis
  (`$ds-text-strong`, `-emphasis`), component-state surfaces
  (`$ds-hover-*`, `$ds-disabled-bg`), and layout-law sidebar geometry
  (`$ds-sidebar-width/min/max`). All scalar single-value tokens; every value
  taken 1:1 from literals already in production stylesheets.
  Deliberately omitted `$ds-opacity-disabled`/`$ds-disabled-fg` from the
  original design: TASK-25822 established that disabled action controls use
  `$ds-disabled-bg` + `$ds-text-disabled-readable` at full opacity.
- Tokenized exemplars: `components/_buttons.tcss`,
  `components/_forms.tcss`, and the `.sidebar` layout law in
  `layout/_sidebars.tcss` (1:1 value mappings, pure refactor).
- `Tests/UI/test_design_token_governance.py`: fails on any `$ds-*`
  referenced but not defined (comments stripped first), duplicate
  definitions, hex literals outside `_variables.tcss`/`Themes/` (legacy
  sheets ratcheted at pinned counts), and raw numeric padding/margin in
  sheets added after ADR-150. Includes a documented-vocabulary availability
  test so the catalog exists before any consumer.
- `backlog/docs/design-language.md`: the constitution — token catalog,
  interaction rules, the new-screen recipe, token-addition process. Linked
  from a new "Design Language (UI Tokens)" section in AGENTS.md.
- Fixed a pre-existing dev failure: `Tests/QA/test_scheduling_css_tokens.py`
  expected pane rules in the main bundle, but TASK-24459's screen-owned
  split moved them to `screen_feature_scheduling.tcss`; the test now checks
  the screen sheet.
- Regenerated `tldw_cli_modular.tcss` + the six screen-owned sheets via
  `build_css.py` (they embed the token definitions).

Verification: 48 CSS tests passed in an isolated worktree off origin/dev
(b277f16193), including the bundle sync guard and the governance suite.
Known pre-existing failure NOT addressed (fails on pristine origin/dev,
unrelated): `Tests/UI/test_css_class_coverage_contract.py` —
`console-activity-*`/`library-details-*` classes composed without rules.

Recovery note: the original 2026-08-05 working-tree edits were lost when the
repo branch moved; the work was re-applied onto origin/dev in a git worktree,
renumbering ADR-042 → ADR-150 (collision) and the task id (regenerated
backlog collision).
