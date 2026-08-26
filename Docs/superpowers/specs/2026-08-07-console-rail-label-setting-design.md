# Console Rail Label Setting Design

## Summary

Let users choose how collapsed Console rail labels render. The existing
horizontal handles remain the default; users who prefer a narrower Console can
opt into the stacked top-to-bottom presentation from the canonical Settings
screen.

## User Experience

Settings > Console Behavior adds a staged checkbox labeled **Stack collapsed
rail labels**. It is unchecked by default. Its nearby help copy says: **Uses
narrower 3-column Context and Inspector handles by stacking upright letters.
Direction glyphs are omitted.** User-facing copy calls the opt-in style
**Stacked**; **vertical** remains a search alias.

The checkbox retains that complete visible label as its readable control name.
An adjacent, draft-aware text status carries the value without relying on
checkbox paint alone:

- Saved value: **Saved style: Horizontal** or **Saved style: Stacked**.
- Unsaved draft: **Selected style: Horizontal — unsaved** or **Selected style:
  Stacked — unsaved**.

The setting follows the Console Behavior category's existing save contract:
changing it creates an unsaved draft, `s`/Save persists every unsaved Console
Behavior change, and `r`/Revert discards every unsaved Console Behavior change
and restores the last loaded values. A successful Save updates the in-memory
application configuration as well as the config file. Destination navigation
constructs a fresh Console screen from that updated configuration, so returning
to Console shows the new style immediately without restarting Chatbook. This
guarantee does not require mutating an already-mounted Console: Settings and
Console are mutually exclusive destination screens.

## Configuration Contract

Persist one boolean under `[console]`:

```toml
stack_collapsed_rail_labels = false
```

`false` is the compatibility default. Values use the existing Console boolean
coercion contract; missing or unrecognized input resolves to `false`. This
preserves the pre-feature horizontal presentation for existing users and
installations while keeping the stacked style an explicit opt-in.

The normalized in-memory `[console]` section is the runtime source of truth for
both Settings and Console. Configuration normalization and the shipped config
template both own the default. Settings must stage the exact persistence payload
`{"console": {"stack_collapsed_rail_labels": <bool>}}` through the existing
Console Behavior save path. The field must participate in the category's Console
key allowlist and deterministic save order so it cannot render and become dirty
while being silently omitted from Save.

Settings updates the runtime object only after persistence succeeds. Failed
saves retain the draft and leave the active runtime style unchanged. Missing or
unrecognized manual values use the existing non-blocking boolean-coercion
contract and resolve to `false`; the next successful Save writes a canonical
boolean.

## Console Presentation

`ChatScreen` resolves the saved preference whenever it composes the two
collapsed handles.

- Horizontal left: render `Context ▸` in a 13-column handle.
- Horizontal right: render plain `Inspector` in an 11-column handle; do not add
  a direction glyph.
- Stacked: render `Context` and `Inspector` one upright character per row in
  their existing three-column handles. Omit direction glyphs, retain
  top-to-bottom badge stacking, and preserve the centered one-cell paint
  contract.
- Both modes retain existing badges, focus behavior, open/collapse behavior,
  and horizontal descriptive tooltips. The readable actions remain **Open
  Context rail** and **Open Inspector rail**, independent of visible stacking.

The preference applies to both collapsed Console rails together. Expanded rail
headers and every non-Console consumer of the shared destination handle remain
unchanged. Per-rail preferences, arbitrary rotation, and a live toggle inside
Console are out of scope.

## Settings Placement and Copy

Place the checkbox near the start of the Console Behavior card under a compact
`Rail presentation` section, before composer paste handling. This makes the
visual preference easy to find without mixing it into model fallbacks or agent
execution limits.

Use the stable field ID
`settings-console-stack-collapsed-rail-labels`. Register its visible label and
the aliases `rail`, `handle`, `stacked`, `vertical`, `context`, and `inspector`
in Settings field search. From a mounted Settings screen, `/` search followed by
Enter must focus the checkbox. The field also participates in the focus
whitelist and Settings config-ownership registry as
`console.stack_collapsed_rail_labels`. The Console Behavior category summary
mentions rail presentation.

Focused guidance uses four concise rows:

- **Purpose:** Choose the collapsed Console rail label style.
- **Consequences:** Stacked uses narrower 3-column handles; Horizontal uses the
  established 13- and 11-column handles.
- **Saved as:** `console.stack_collapsed_rail_labels`.
- **Applies:** After saving, when Console is next opened.

The control uses the incumbent dense-form grammar: readable text carries both
meaning and state, focus causes no layout shift, and the category's pinned state
bar continues to explain that changes are staged until Save. Tab and Shift+Tab
reach the field in logical order, Space toggles it, and focus remains visible
without relying on color alone.

Save/Revert feedback names both the category-wide scope and resulting rail
style. Equivalent copy is acceptable when it preserves these facts:

- Successful Save: **Console Behavior saved. Rail labels: Stacked. Return to
  Console to see the change.** (`Horizontal` when unchecked.)
- Revert: **Console Behavior changes reverted. Rail labels: Horizontal.** Use
  the actual restored style.
- Save failure: **Couldn't save Console Behavior. Your draft is still here;
  the active rail-label style is still Horizontal. Try again.** Use the actual
  active style.

## Failure and Edge Cases

- Missing or invalid config: render horizontal handles.
- Save failure: keep all Console Behavior drafts visible, report that the draft
  was retained and the active style did not change, and do not change in-memory
  runtime config.
- Category-wide Revert: discard all Console Behavior drafts together and report
  the actual restored rail-label style.
- Console lifecycle: a successful Save affects the fresh Console created on
  return; a failed Save leaves that fresh Console on the previously active
  style. No live mutation of an already-mounted Console is required.
- Narrow terminals and badges: preserve the existing horizontal and stacked
  containment contracts for their respective modes.

## Verification

- Configuration tests cover the shipped `false` default and absent, false, true,
  and malformed values.
- Settings tests cover the default unchecked state, text-carried saved/draft
  status, Tab/Shift+Tab and Space behavior, `/` search-to-focus, focused
  guidance, staging, category-wide Save/Revert, failed Save, the exact
  persistence payload, and success-only in-memory updates.
- Console tests cover exact horizontal labels and mounted widths (`Context ▸`
  at 13, `Inspector` at 11), plus stacked labels and three-column widths for both
  rails. Both modes preserve open/collapse behavior and readable tooltips.
- A Settings-to-fresh-Console integration test proves saved `false` renders
  horizontal, saved `true` renders stacked, and failed Save retains the prior
  active style without restarting Chatbook.
- Mounted paint tests retain the stacked one-cell containment regression.
- Existing destination/Personas tests guard unchanged non-Console behavior.
- Update `Docs/User_Guide/settings.md` and
  `Docs/User_Guide/console/chat-basics.md` with the default, both styles, the
  category-wide Save/Revert behavior, and when a saved change becomes visible.
- A post-implementation Textual render captures both saved styles for visual
  review.

## Architecture Decision Record

ADR required: no

ADR path: N/A

Reason: This is one additive, optional presentation preference with a safe
compatibility default. It introduces no schema migration, service contract,
security boundary, dependency, or new long-lived UI architecture.
