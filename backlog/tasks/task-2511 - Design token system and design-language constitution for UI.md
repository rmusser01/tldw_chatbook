---
id: TASK-2511
title: Design token system and design-language constitution for UI
status: In Progress
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

## Description (the why)

New screens and widgets are currently styled by inventing literals: ~7,100
hardcoded dimension declarations and ~900 hex literals live across the TCSS
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
  sizing, motion durations, disabled/muted opacity, typography emphasis, and
  component-state semantics, alongside the existing color/status tokens
- [x] A design-language constitution exists under `backlog/docs/` describing
  the token catalog, naming rules, layout laws, interaction rules, and the
  process for adding tokens, and is linked from AGENTS.md
- [x] A governance pytest fails when a `$ds-*` token is referenced but not
  defined in `core/_variables.tcss`
- [x] Exemplar sheets (`components/_buttons.tcss`, `components/_forms.tcss`)
  consume tokens with no visual change (values map 1:1)
- [ ] The committed CSS bundle reproduces from sources (bundle sync guard
  passes) and the existing CSS test suite still passes
- [x] ADR created and linked (long-lived UX structure decision)

## Implementation Plan (the how)

ADR required: yes
ADR path: backlog/decisions/042-design-token-system-and-design-language.md
Reason: long-lived UX/application structure and governance decision per
AGENTS.md ADR criteria.

1. Extend `core/_variables.tcss` with spacing/sizing/motion/opacity/
   typography/state token sections; scalar single-value tokens only (Textual
   variable substitution is per-token)
2. Tokenize exemplar sheets `_buttons.tcss` and `_forms.tcss` with 1:1 value
   mappings (pure refactor, no visual change)
3. Add `Tests/UI/test_design_token_governance.py`: every `$ds-*` referenced
   in any TCSS source must be defined in `core/_variables.tcss`
4. Write `backlog/docs/design-language.md` constitution; link from AGENTS.md
   with hard rules for new UI work
5. Rebuild the bundle, run CSS tests, update task notes

## Implementation Notes

Implemented the design-token system additively (ADR-042), no visual changes:

- `core/_variables.tcss`: added token sections — spacing scale
  (`$ds-space-0..3` + semantic aliases `inline/stack/section/inset`),
  control sizing (`$ds-control-height`, `-compact`,
  `$ds-textarea-min-height`), motion (`$ds-duration-fast/medium/slow`),
  opacity (`$ds-opacity-disabled`, `-dim`), typography emphasis
  (`$ds-text-strong`, `-emphasis`), component-state surfaces
  (`$ds-hover-*`, `$ds-disabled-*`), and layout-law sidebar geometry
  (`$ds-sidebar-width/min/max`). All scalar single-value tokens; every value
  was taken 1:1 from literals already in production stylesheets.
- Tokenized exemplars: `components/_buttons.tcss`,
  `components/_forms.tcss`, and the `.sidebar` layout law in
  `layout/_sidebars.tcss`. Token substitution patterns used (scalar,
  mid-shorthand) match existing production usage
  (`border: round $ds-grid-line;`, `height: $ds-home-dashboard-grid-height;`).
- `Tests/UI/test_design_token_governance.py`: fails on any `$ds-*`
  referenced but not defined, on duplicate definitions, on hex literals
  outside `_variables.tcss`/`Themes/` (legacy sheets ratcheted at pinned
  counts), and on raw numeric padding/margin in post-ADR sheets.
- `backlog/docs/design-language.md`: the constitution — token catalog,
  interaction rules (focus contract, Select-three-shapes, readable-error,
  no stacked disabled opacity), the new-screen recipe, and the token-addition
  process. Linked from a new "Design Language (UI Tokens)" section in
  AGENTS.md.
- `backlog/decisions/042-design-token-system-and-design-language.md`: ADR
  with alternatives considered (full migration rejected — ratchet instead).

**Pending verification (shell approvals unavailable in this session):**
`python3 -m pytest Tests/UI/test_design_token_governance.py Tests/UI/ -q`
has not been executed. The committed bundle was synced surgically (module
blocks for the four edited sources replaced in `tldw_cli_modular.tcss`) and
cross-checked by occurrence counts (space tokens 40+5=45, state/sizing group
31/31 exact); a static audit confirmed every `$ds-*` referenced across
core/layout/components/features/utilities is defined. Run
`python3 tldw_chatbook/css/build_css.py` then the pytest line above to close
the remaining AC and mark Done.
