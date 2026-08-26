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

- Give every Library rail expanded **alongside content** one shared default-width
  policy: 3 parts of the ordinary 16-part Library/canvas allocation, clamped to
  an exact 24-cell minimum and 34-cell maximum.
- Prevent destination switches from changing a co-present rail's rendered width
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
- Forcing a hidden rail or a rail-only compact takeover into the 24–34
  alongside-content rail range.
- Claiming native iTerm2 or Windows Terminal acceptance from Textual Pilot tests.

## Design

### One shared default-width policy, two geometry adapters

The pure Library layout state module owns separate default and custom sizing
constants plus one deterministic default projection. Its positive input `W` is
the `LibraryAdaptiveReaderShell.content_region.width`: the allocation remaining
after the outer `#library-shell-grid` border and padding, but before grips and
optional panes are subtracted. Ordinary Textual allocation uses the same
`#library-shell-grid.content_region.width`.

The projection is:

```text
fractional = floor((3 × W + 8) / 16)  # nearest cell; exact halves round upward
default_library_width = clamp(fractional, 24, 34)
```

`W` must be positive. A zero-width pre-layout state produces the existing
all-zero effective-layout sentinel and is not projected or inherited for
hysteresis. The separate 31-cell reference is used only for new/reset preference
defaults and unresolved requested-state construction; it is never rendered into
a zero-width shell. These definitions give adaptive calculations and
ordinary/adaptive comparison tests one oracle.

The two existing geometry mechanisms remain intact:

1. **Ordinary Library layouts** use a state-driven adapter. Alongside content
   with custom widths off, it applies `width: 3fr`, `min-width: 24`, and
   `max-width: 34`. Alongside content with custom widths on, it applies the exact
   saved width with matching minimum and maximum. In a rail-only takeover it
   clears the ceiling and fills the workbench. In a canvas-only or manually
   collapsed state it hides the rail under the existing contract. Returning wide
   restores the correct bounded or custom declarations.
2. **Adaptive reader layouts** continue to apply exact cell widths through the
   pure resolver and `LibraryAdaptiveReaderShell.sync_layout()`. When custom
   widths are disabled, the resolver uses the shared bounded projection instead
   of the historical fixed 28-cell target. The adaptive shell must not restore a
   competing `3fr` declaration after resolution.

This is intentionally one policy with two adapters, not a claim that
`LibraryRail` is the sole geometry owner. The app-tier `#library-rail` stylesheet
continues to own presentation and overflow rather than width.

`LibraryScreen`, already the responsive-stage and configuration orchestration
owner, loads one normalized shared `[library.reader]` snapshot and decides whether
an ordinary rail is alongside-content, rail-only, or hidden. It passes the
snapshot to every `LibraryRail` and invokes one reversible rail-owned style
transition when the compact stage, manual collapse, route, settings generation,
or shell width changes. `LibraryRail` applies the declarations; it does not read
configuration. Adaptive shells continue to replace those declarations with the
exact effective width after the screen's pure resolver has run.

Within the adaptive resolver, one `requested_library_width` is computed per
positive shell width: the saved value when custom mode is on, otherwise the
bounded projection. Every fit, priority, collapse, hysteresis, and final-width
calculation uses that same value; the historical dormant `library_width` field
must not leak back into default-mode thresholds.

### Preference and migration behavior

`[library.reader]` remains the shared owner of `library_open`,
`custom_widths_enabled`, and `library_width`.

- With custom widths disabled, `library_width` is dormant and the current shell
  width determines the 24–34-cell default.
- With custom widths enabled, the saved 24–48-cell value is rendered in ordinary
  and adaptive destinations whenever the rail is alongside content and the
  responsive policy can fit it.
- Existing saved values from 24 through 48 remain valid. Values from 35 through
  48 are not silently clamped to the new default ceiling.
- New installations and explicit Settings reset use 31, aligning the disabled
  Settings value with the visual reference. There is no migration: an existing
  stored 28 remains dormant while custom mode is off and becomes the intentional
  28-cell override if the user later enables custom widths. Other existing saved
  values are likewise not rewritten.
- Responsive collapse, compression, and terminal resizing never persist an
  effective width.

The Settings label should describe a shared **Library rail**, not imply that the
width applies only to Media or to the work reader.

### Settled behavior matrix

| State | Effective rail behavior |
| --- | --- |
| Ordinary alongside content, custom off | Native `3fr`, clamped to 24–34 |
| Adaptive alongside content, custom off | Exact-cell projection of 3:13, clamped to 24–34 |
| Any alongside-content destination, custom on | Saved 24–48 width when it fits |
| Adaptive rail auto-collapsed | Rail hidden; five-cell Library grip remains |
| Adaptive explicit-open escape at extreme width | Resolver may compress below 24 to prevent overflow |
| Ordinary manual collapse | Existing collapsed handle contract remains authoritative |
| Ordinary compact rail-only takeover | Rail fills the available workbench width; 34 is not a cap |
| Ordinary compact canvas-only takeover | Rail is hidden; canvas fills the workbench |

“Alongside content” means the rail and at least one canvas/list/work region are
simultaneously displayed. Rail-only, canvas-only, manually collapsed, adaptive
auto-collapsed, and compressed-priority states are mutually exclusive structural
exceptions rather than expanded alongside-content rails.

When a rail is alongside content at the same settled width and no
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
3. Use the production-styled outer-width matrix below. The grid content width
   `W` is asserted before applying the formula, so a future border/padding change
   fails honestly instead of silently changing the oracle.

   | Terminal | Expected production `W` | Ordinary default | Adaptive default state |
   | ---: | ---: | --- | --- |
   | 235 | 231 | alongside, 34 | Library 34, Items open, Work open |
   | 170 | 166 | alongside, 31 | Library 31, Items open, Work open |
   | 120 | 116 | alongside, 24 | Library collapsed; Items and Work open |
   | 100 | 96 | rail-only fills `W`, then canvas-only hides rail after activation | Library collapsed; Conversations/Media Items open; Notes Items collapsed; Work open |
   | 80 | 76 | rail-only fills `W`, then canvas-only hides rail after activation | Library and Items collapsed; Work open |
   | 60 | 56 | rail-only fills `W`, then canvas-only hides rail after activation | Library and Items collapsed; Work uses compositor escape if needed |

   Each row asserts containment, non-intersection, and footer bounds. If the
   production box model intentionally changes, the design record and matrix must
   be updated rather than weakening the assertion.
4. At 170 columns, switch through Media, Chats, Notes, Prompts, Skills,
   Collections, Search/RAG, Import, Export, Study handoffs, and the landing
   canvas. Assert a 31-cell co-present rail on every ordinary route and the same
   pre-collapse rail width after each adaptive reader's settled `sync_layout`,
   allowing at most one cell only if Textual's native allocation proves it.
5. Verify custom values 24, 34, 35, and 48 across ordinary and adaptive
   destinations, plus invalid-value normalization. A saved value above 34 must
   remain intact and must restore after responsive collapse.
6. Verify initial mount, route switch, scoped recompose, and
   wide-to-compact-to-wide recovery without stale geometry, width-preference
   writes, focus loss, or hysteresis pinning from the zero-width sentinel. Cover
   rail-only → wide restoration separately for bounded and custom declarations.
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
