# Console Rail Label Setting Design

## Summary

Let users choose how collapsed Console rail labels render. The existing
horizontal handles remain the default; users who prefer a narrower Console can
opt into the stacked top-to-bottom presentation from the canonical Settings
screen.

## User Experience

Settings > Console Behavior adds a staged checkbox labeled **Stack collapsed
rail labels vertically**. It is unchecked by default. Its nearby help copy
explains that the option narrows the collapsed Context and Inspector handles by
stacking one upright character per row; terminal glyphs are not rotated.

The setting follows the Console Behavior category's existing save contract:
changing it creates an unsaved draft, `s`/Save persists it, and `r`/Revert
restores the saved value. A successful Save updates the in-memory application
configuration as well as the config file, so the next Console render uses the
new style immediately without restarting Chatbook. Settings and Console are
mutually exclusive destination screens, so returning to Console is the first
visible opportunity to observe the saved change.

## Configuration Contract

Persist one boolean under `[console]`:

```toml
stack_collapsed_rail_labels = false
```

`false` is the compatibility default. Values use the existing Console boolean
coercion contract; missing or unrecognized input resolves to `false`. This
preserves the pre-feature horizontal presentation for existing users and
installations while keeping the vertical style an explicit opt-in.

The normalized in-memory `[console]` section is the runtime source of truth for
both Settings and Console. Settings must update that object only after a
successful save; failed saves retain the draft and leave the active runtime
style unchanged.

## Console Presentation

`ChatScreen` resolves the saved preference whenever it composes the two
collapsed handles.

- Horizontal: retain the established Context and Inspector labels, direction
  glyphs, side-specific widths, badges, focus behavior, and tooltips.
- Stacked: use the existing three-cell handles, top-to-bottom label/badge
  stacking, centered one-cell paint contract, and horizontal descriptive
  tooltips.

The preference applies to both collapsed Console rails together. Expanded rail
headers and every non-Console consumer of the shared destination handle remain
unchanged. Per-rail preferences, arbitrary rotation, and a live toggle inside
Console are out of scope.

## Settings Placement and Copy

Place the checkbox near the start of the Console Behavior card under a compact
`Rail presentation` section, before composer paste handling. This makes the
visual preference easy to find without mixing it into model fallbacks or agent
execution limits.

The control uses the incumbent dense-form grammar: readable text carries the
meaning, the checkbox carries state, focus causes no layout shift, and the
category's pinned state bar continues to explain that changes are staged until
Save.

## Failure and Edge Cases

- Missing or invalid config: render horizontal handles.
- Save failure: keep the staged checkbox value visible, report the existing
  Console Behavior save error, and do not change in-memory runtime config.
- Rail open during the preference change: no expanded-header change; the saved
  style appears the next time that rail is collapsed/rendered.
- Narrow terminals and badges: preserve the existing horizontal and stacked
  containment contracts for their respective modes.

## Verification

- Configuration tests cover absent, false, true, and malformed values.
- Settings tests cover default unchecked state, staging, Save, Revert, failed
  Save, persistence payload, and in-memory update after successful Save.
- Console tests cover horizontal default labels/widths and the vertical opt-in
  labels/widths for both rails.
- Mounted paint tests retain the stacked one-cell containment regression.
- Existing destination/Personas tests guard unchanged non-Console behavior.
- A post-implementation Textual render captures both saved styles for visual
  review.

## Architecture Decision Record

ADR required: no

ADR path: N/A

Reason: This is one additive, optional presentation preference with a safe
compatibility default. It introduces no schema migration, service contract,
security boundary, dependency, or new long-lived UI architecture.
