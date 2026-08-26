---
target: Console rail label setting design
total_score: 25
max_score: 40
na_heuristics:
p0_count: 0
p1_count: 5
timestamp: 2026-08-08T14-25-34Z
slug: cs-2026-08-07-console-rail-label-setting-design-md
---
# Console Rail Label Setting Design Review

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of system status | 2 | Draft/runtime separation is clear, but success and failure copy do not name the chosen style or next step. |
| 2 | Match system / real world | 3 | The choice is understandable, though terminal implementation language leaks into user copy. |
| 3 | User control and freedom | 3 | Save/Revert are preserved, but Revert is category-wide rather than setting-only. |
| 4 | Consistency and standards | 2 | The spec omits Settings search/guidance wiring and text-carried toggle state required by the design system. |
| 5 | Error prevention | 3 | Safe horizontal fallback and success-only runtime mutation are strong. |
| 6 | Recognition rather than recall | 2 | Users must remember the selection while navigating from Settings back to Console. |
| 7 | Flexibility and efficiency | 3 | Keyboard Save/Revert remain efficient, but search landing and Space activation are not specified. |
| 8 | Aesthetic and minimalist design | 3 | One compact Rail presentation group is disciplined and appropriately scoped. |
| 9 | Error recognition and recovery | 2 | Generic save failure copy lacks cause, impact, and retry guidance. |
| 10 | Help and documentation | 2 | Nearby help exists, but focused-field guidance, search aliases, and user-guide updates are absent. |
| **Total** | | **25/40** | **Acceptable; interaction-contract improvements required before implementation.** |

## Design Specificity Verdict

The proposal is clearly authored for Chatbook: it names the Context and
Inspector rails, terminal-cell constraints, the staged Settings model, and the
local config source. It is not category-interchangeable. Its weakness is not
visual direction; it is that several incumbent Settings contracts are left for
an implementer to infer.

The deterministic detector returned zero findings for
`tldw_chatbook/UI/Screens/settings_screen.py`. That clean result is
non-probative for a Python/Textual surface: it does not inspect focus behavior,
config allowlists, rail widths, or rendered terminal cells.

## Overall Impression

The core product decision is right: horizontal by default, one opt-in stacked
style, both rails governed together, and no restart. Before planning, the spec
needs exact wiring and feedback requirements so the field cannot render yet be
silently dropped during Save, remain undiscoverable through `/` search, or
restore the wrong horizontal glyph/width contract.

## What's Working

1. The compatibility contract is safe: absent or invalid values resolve to the
   established horizontal presentation.
2. The scope is disciplined: expanded headers and non-Console rail consumers
   remain untouched.
3. Placement under a compact Rail presentation group keeps the preference easy
   to scan without adding a Console-local control.

## Priority Issues

### [P1] Make the staged-save wiring explicit

Settings filters dirty Console values through both
`CONSOLE_BEHAVIOR_SAVE_ORDER` and `CONSOLE_BEHAVIOR_CONSOLE_KEYS`. The new field
can appear and stage successfully yet never reach persistence if either list is
missed.

**Fix:** Require a loaded-value resolver, draft staging handler, sync guard,
both allowlists, success-only in-memory update, and Save/Revert/failure tests.
Pin the payload as
`{"console": {"stack_collapsed_rail_labels": <bool>}}`.

### [P1] Integrate field search, focused guidance, and ownership

Settings field search, focus recognition, focused guidance, and owned config
keys are separate manual registries. The spec currently promises easy
discovery without requiring any of them.

**Fix:** Require a stable checkbox ID, `/` search aliases (`rail`, `handle`,
`stacked`, `vertical`), an Enter-to-focus test, four focused-guide rows
(`Purpose`, `Consequences`, `Saved as`, `Applies`), ownership of
`console.stack_collapsed_rail_labels`, and a category summary that mentions
rail presentation.

### [P1] Carry state in text and preserve readable control names

The design system requires toggles to pair paint with a text state word. The
spec currently says the checkbox itself carries state. In stacked mode, the
visible rail button label is newline-separated characters, so the readable
full action must remain available to keyboard users.

**Fix:** Keep the full checkbox label and add draft-aware text such as
`Current style: Horizontal` / `Current style: Stacked`. Require Tab/Shift+Tab,
Space activation, visible focus, and readable `Open Context rail` /
`Open Inspector rail` help independent of stacked paint.

### [P1] Pin the exact horizontal contract and correct the glyph wording

The spec says horizontal mode retains direction glyphs, but production
intentionally renders `Context ▸` on the left and plain `Inspector` on the
right. Stacked mode omits both direction glyphs.

**Fix:** State exact strings and widths: horizontal left `Context ▸` at 13
columns, horizontal right `Inspector` at 11; stacked `Context` and `Inspector`
one character per row at 3 columns. Change help copy from “glyphs are not
rotated” to “direction glyphs are omitted.”

### [P1] Define feedback and category-wide Save/Revert scope

The effect becomes visible only after returning to Console. Generic “Console
behavior settings saved” feedback does not tell users which style is active.
The spec also implies Revert restores only this checkbox, while Settings
actually discards every unsaved Console Behavior edit together.

**Fix:** Require dynamic staged/saved/reverted copy naming Horizontal or
Stacked, say that Save/Revert apply to all unsaved Console Behavior changes,
and make failure copy state that the draft was kept and the active style did
not change.

### [P2] Complete lifecycle, docs, and malformed-value verification

Immediate application is feasible because destination navigation constructs a
fresh Console screen from the updated in-memory config. The spec does not yet
require an integration test proving that lifecycle, shipped-template coverage,
or user-guide updates.

**Fix:** Add Settings-save → fresh-Console tests for false, true, and failed
save; assert shipped config defaults to false; cover absent/false/true/malformed
loads; update Settings and Console user guides. Optionally surface a non-blocking
warning when an explicitly malformed manual value is ignored.

## Persona Red Flags

**Alex, power user:** `/ rail` or `/ stacked` will not find the field unless
the manual search index is updated. Category-wide Revert is broader than the
current wording implies.

**Jordan, first-time user:** “terminal glyphs” and “collapsed rail” are jargon,
and generic Save feedback does not explain what changed or where to see it.

**Sam, keyboard/accessibility-dependent user:** the spec does not guarantee
text-carried state, Space activation, visible multi-signal focus, or a readable
unstacked action label when the visible button is stacked.

## Minor Observations

- “Stacked” is more accurate user-facing language than “vertical”; retain
  “vertical” as a search alias.
- One preference governing both rails is a strong simplification and should
  remain.
- No component TCSS change should be necessary; if CSS is touched, regenerate
  the modular bundle and run its parity test.
- The existing paint-containment regression remains valuable and should not be
  weakened when horizontal becomes the default.

## Questions to Consider

- Should the Settings row show the concrete width payoff (`13 + 11` columns to
  `3 + 3`) so users understand why they might opt in?
- Should malformed manual config be silently corrected on the next Save or
  surfaced as a non-blocking warning first?
- Is dynamic result copy enough to bridge Settings and Console, or should the
  focused guide include a tiny textual example of both styles?

## Recommended Actions

1. **`$impeccable clarify`**: tighten field labels, state words, exact
   staged/saved/reverted/failure copy, and category-wide Save/Revert wording.
2. **`$impeccable harden`**: add the missing search, focus, ownership, config,
   lifecycle, error, documentation, and regression requirements to the spec.
3. **`$impeccable polish`**: after implementation, render both modes and verify
   focus, badges, widths, and Settings discoverability at supported terminal
   sizes.
