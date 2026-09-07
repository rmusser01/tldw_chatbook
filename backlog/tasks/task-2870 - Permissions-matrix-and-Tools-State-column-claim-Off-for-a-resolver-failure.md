---
id: TASK-2870
title: Permissions matrix and Tools State column claim "Off" for a resolver failure
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:30'
updated_date: '2026-08-07 01:27'
labels:
  - mcp
  - ui
  - honesty
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #1385's round J made the inspector's permission block stop rendering
"Permission: Off" for a `gate_error` verdict (the synthesized fail-closed
state when the permission RESOLVER raises — not a configured Off). Its
commit deliberately left the other `ui_label` renderers alone and pointed
at task-2270 — a mis-attribution: 2270 covers badge staleness and the
decision-note copy, not these labels. This task is the actual filing.

`EffectiveToolState.ui_label` maps `state="deny"` to "Off" with no origin
awareness, so `format_tool_state_label()` renders "Off ·" for gate_error
rows in the Permissions matrix and the Tools-mode State column — a
confident configuration claim about a state the resolver could not read.
Reachable live: `MCPWorkbench._effective_for_display()` synthesizes
`EffectiveToolState(state="deny", origin="gate_error")` whenever per-tool
resolution raises.

The truthful vocabulary is already ruled: "Unknown" for the label (round
J), error-severity color kept (the fail-closed EFFECT is real; only the
causal label lied).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A gate_error-origin state renders "Unknown", never "Off", everywhere `ui_label` feeds copy (Permissions matrix State cells, Tools-mode State column, inspector permission block).
- [x] #2 A genuine deny (any non-gate_error origin) still renders "Off" on every one of those surfaces.
- [x] #3 The severity color/kind for gate_error rows is unchanged (still the blocked/error treatment).
- [x] #4 Round J's inspector test keeps passing; new coverage pins the label at the `ui_label` seam and at least one view-level render.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`EffectiveToolState.ui_label` now owns the honesty rule: `origin ==
"gate_error"` → "Unknown"; genuine denies from any other origin keep
"Off". The inspector's round-J site-local branch collapsed back to a
plain `ui_label` read (round J's test now pins through the property),
`format_tool_state_label()` renders a bare "Unknown" (no origin marker)
for matrix/State-column rows, and `tool_state_kind()` is deliberately
unchanged (error bucket — the fail-closed effect is real). Consumer
sweep before changing the shared property: `_decision_note()`'s string
branches are proven unreachable for gate_error; `_cycled_ui_label()`
and the default-rung renderers construct their own non-gate_error
states. Mutation: dropping the branch reds both new tests and round J's
inspector test. 539 passed across the five reachable suites; no
existing test relied on the dishonest cell. Commit b94bb0910 on
fix/mcp-inspector-honesty-residue.
<!-- SECTION:NOTES:END -->
