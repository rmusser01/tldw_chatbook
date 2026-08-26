# Library Rail Bounded-Width Design

**Date:** 2026-08-25  
**Task:** TASK-22301  
**Status:** Approved direction; revised after architecture review

## Problem

The Library navigation rail does not currently have one visible-width contract.
Ordinary Library canvases use a fluid `3fr` rail beside a `13fr` canvas, while
adaptive Media, Conversations, and Notes readers replace that geometry with an
exact 28-cell width. The ordinary rail also has no upper bound, so it grows well
beyond the useful Collections-reference width on wide terminals. Switching
destinations can therefore move the rail edge even when the terminal does not
change.

The desired default remains fluid rather than fixed: use the existing 3:13
Library-to-canvas proportion, but clamp the visible rail to 24–34 cells around
the approximately 31-cell Collections reference. Existing explicit custom
widths, responsive pane collapse, and compact takeovers remain valid exceptions
to that default.

## Goals

- Give every expanded Library rail one shared default-width policy: 3 parts of
  the ordinary 16-part Library/canvas allocation, clamped to an exact 24-cell
  minimum and 34-cell maximum.
- Prevent destination switches from changing an expanded rail's rendered width
  at the same settled shell width.
- Preserve the existing custom-width opt-in and its 24–48-cell range; when
  enabled, the explicit width overrides the bounded default in every destination.
- Preserve adaptive-reader collapse order, priority behavior, hysteresis,
  five-cell grips, and requested-versus-effective preference semantics.
- Preserve ordinary compact rail-only/canvas-only takeovers and manual collapse.
- Preserve search, disclosure, scrolling, focus, row fitting, and selection.

## Non-Goals

- Making 31 cells a fixed width at every terminal size.
- Adding drag handles, new persistence keys, or resize-time preference writes.
- Redesigning Library information architecture, labels, counts, or destination
  list/work-pane sizing.
- Forcing a hidden rail or a rail-only compact takeover into the 24–34 expanded
  rail range.
- Claiming native iTerm2 or Windows Terminal acceptance from Textual Pilot tests.

## Design

### One shared default-width policy, two geometry adapters

The pure Library layout state module owns the sizing constants and a deterministic
default projection. Given the settled width available to the Library shell, it
returns the nearest whole-cell equivalent of the ordinary `3fr` rail beside the
`13fr` canvas, clamped to 24–34 cells. The representative fallback/default value
is 31 cells. Rounding is defined once in the pure helper so ordinary and adaptive
tests do not encode competing arithmetic.

The two existing geometry mechanisms remain intact:

1. **Ordinary Library layouts** retain `width: 3fr` and `min-width: 24`, and add
   `max-width: 34`. Textual remains responsible for the live fractional
   allocation.
2. **Adaptive reader layouts** continue to apply exact cell widths through the
   pure resolver and `LibraryAdaptiveReaderShell.sync_layout()`. When custom
   widths are disabled, the resolver uses the shared bounded projection instead
   of the historical fixed 28-cell target. The adaptive shell must not restore a
   competing `3fr` declaration after resolution.

This is intentionally one policy with two adapters, not a claim that
`LibraryRail` is the sole geometry owner. The app-tier `#library-rail` stylesheet
continues to own presentation and overflow rather than width.

### Preference and migration behavior

`[library.reader]` remains the shared owner of `library_open`,
`custom_widths_enabled`, and `library_width`.

- With custom widths disabled, `library_width` is dormant and the current shell
  width determines the 24–34-cell default.
- With custom widths enabled, the saved 24–48-cell value is rendered in ordinary
  and adaptive destinations whenever the rail is expanded and the responsive
  policy can fit it.
- Existing saved values from 24 through 48 remain valid. Values from 35 through
  48 are not silently clamped to the new default ceiling.
- The default shown for a new installation or reset becomes 31, aligning the
  disabled Settings value with the visual reference. Existing explicit saved
  values are not rewritten.
- Responsive collapse, compression, and terminal resizing never persist an
  effective width.

The Settings label should describe a shared **Library rail**, not imply that the
width applies only to Media or to the work reader.

### Settled behavior matrix

| State | Effective rail behavior |
| --- | --- |
| Ordinary expanded, custom off | Native `3fr`, clamped to 24–34 |
| Adaptive expanded, custom off | Exact-cell projection of 3:13, clamped to 24–34 |
| Any expanded destination, custom on | Saved 24–48 width when it fits |
| Adaptive rail auto-collapsed | Rail hidden; five-cell Library grip remains |
| Adaptive explicit-open escape at extreme width | Resolver may compress below 24 to prevent overflow |
| Ordinary manual collapse | Existing collapsed handle contract remains authoritative |
| Ordinary compact rail-only takeover | Rail fills the available workbench width; 34 is not a cap |
| Ordinary compact canvas-only takeover | Rail is hidden; canvas fills the workbench |

The last four rows are structural responsive states, not inconsistent defaults.
When a rail is expanded alongside content at the same settled width and no
custom override is active, destination switching must produce the same rendered
rail width, allowing only a one-cell compositor rounding difference between the
native fractional adapter and the pure exact-cell adapter.

### Initial mount, navigation, and resize

The first settled frame, post-navigation frame, and post-recompose frame must all
use the same policy. Adaptive readers may construct a pre-layout sentinel before
Textual assigns a real region, but that sentinel must not be treated as a prior
layout for hysteresis and must not persist. Once a positive shell width exists,
the resolver replaces it in place without rebuilding destination content.

Wide-to-compact-to-wide resizing preserves requested open state and custom
preferences. A responsive collapse returns to the correct bounded default or
custom width after the hysteresis threshold is crossed; it must not return to a
stale 28-cell width.

## Accessibility and HCI

- The 24-cell ordinary floor preserves the current search placeholder and
  width-aware row fitting at standard compact widths.
- The 34-cell default ceiling protects reading and editing space on wide
  terminals without removing intentional 35–48-cell accessibility overrides.
- A stable rail edge across destinations reduces spatial relearning for new
  users and preserves fast pointer/keyboard targeting for power users.
- Hidden and rail-only states remain visibly explained by the existing grips,
  handles, and compact-stage controls; blank reserved rail space is forbidden.
- Focus never remains inside a pane that responsive resolution hides.

## Verification

Implementation follows test-driven development and verifies rendered regions,
not only declared style values.

1. Add pure tests for the shared default projection below the floor, within the
   fractional range, at the ceiling, and above the ceiling, including its
   deterministic rounding rule.
2. Add production-styled geometry tests proving the ordinary rail declares
   `3fr`/24/34 and that the adaptive resolver produces the equivalent 24–34
   rendered width with custom widths disabled.
3. At 235, 170, 120, 100, 80, and 60 columns, cover the applicable matrix state:
   ordinary expanded, adaptive expanded/collapsed, compact rail-only, and
   compact canvas-only. Assert containment, non-intersection, and footer bounds.
4. At a width where 3:13 is strictly between the bounds, switch between
   Collections and each shipped adaptive destination and assert equal rendered
   rail regions, allowing at most one cell for compositor rounding.
5. Verify custom values 24, 34, 35, and 48 across ordinary and adaptive
   destinations, plus invalid-value normalization. A saved value above 34 must
   remain intact and must restore after responsive collapse.
6. Verify initial mount, route switch, scoped recompose, and
   wide-to-compact-to-wide recovery without stale geometry, preference writes,
   focus loss, or hysteresis pinning from the zero-width sentinel.
7. Preserve row-label fitting, search, section disclosure, grip/handle access,
   global focus cycling, and adaptive collapse-order tests.
8. Run Library unit/integration suites, generated-CSS integrity when applicable,
   Ruff, compilation, and diff-hygiene gates.
9. Perform isolated PTY UAT at representative wide, standard, and compact
   terminal sizes. Report native terminal-emulator-specific acceptance only
   when it was actually run.

## Documentation and decision record

ADR-086 is amended because this task changes the durable shared Library geometry
policy documented there: the default Library width becomes bounded fractional
instead of fixed, while explicit custom widths retain their existing range. The
earlier adaptive-reader design spec is updated to match so it cannot continue to
advertise 28 as the fixed default.

## ADR Check

**ADR required:** yes

**ADR path:** `backlog/decisions/086-library-adaptive-reader-shell.md`

**Reason:** This changes the long-lived shared Library geometry contract and the
meaning of its existing persisted custom-width boundary. Amending the canonical
ADR is preferable to creating a competing decision record.
