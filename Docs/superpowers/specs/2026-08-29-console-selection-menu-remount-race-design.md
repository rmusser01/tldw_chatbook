# Console Selection Menu Remount Race Design

**Date:** 2026-08-29
**Status:** Approved for implementation planning

## Problem

The Console can raise Textual's app-fatal `DuplicateIds` when a new text
selection remounts `#console-selection-menu` while the previous menu is still
attached but already marked `_pruning`.

`ConsoleTranscript._text_selected` correctly promises to await every attached
menu before remounting. It currently calls `_attached_selection_menus`, whose
fire-and-forget semantics deliberately exclude `_pruning` menus. The remount
therefore sees no menu during the exact lifecycle window that requires an
await, then synchronously registers a second widget with the same ID.

The existing consecutive-selection pilot test yields between mouse events.
That gives Textual time to process the prune and does not cover the submitted
live-terminal ordering.

## Goals

- A selection-menu remount waits until every previously attached menu on the
  owning screen is detached, including menus whose removal is already pending.
- Fire-and-forget dismissal continues to avoid issuing duplicate removal
  requests for `_pruning` menus.
- Repeated or fast text selections leave exactly one selection menu mounted and
  never terminate the app with `DuplicateIds`.
- Existing menu placement, focus, selection, feedback, and dismissal behavior
  remains unchanged.

## Non-goals

- Dynamic menu IDs or permitting multiple simultaneous selection menus.
- A new menu manager, lock, queue, or generalized widget-lifecycle abstraction.
- Changes to unrelated transcript rendering or roleplay styling.

## Design

Keep the two lifecycle operations intentionally separate:

1. Ordinary dismissal continues through `_attached_selection_menus`. That
   helper remains the non-pruning view used by synchronous, fire-and-forget
   callers.
2. `_text_selected` uses Textual's public screen query API and awaits
   `self.screen.query(ConsoleSelectionMenu).remove()` before mounting the
   replacement.

The public query includes attached menus even after Textual has marked them
`_pruning`, and the awaited query removal does not complete until those nodes
are detached. This intentionally performs one DOM walk on a completed text
selection, which is a low-frequency boundary where correctness matters more
than preserving the registry helper's fire-and-forget optimization. The hot
dismissal path keeps using the existing registry helper.

Calling `remove()` again for a menu whose prune is already pending issues an
additional idempotent prune request and awaits detachment; it is not treated as
proof that the first request has already completed. No new state, lock, queue,
or registry bookkeeping is introduced.

## Failure Handling

The invalid DOM state is prevented before `mount`; `DuplicateIds` is not caught
or suppressed. Genuine Textual mount failures continue to propagate. Existing
detached-screen behavior remains unchanged because resolving `self.screen`
still occurs at the same boundary.

## Verification

Add a deterministic regression that:

1. mounts a selection menu;
2. schedules its removal without awaiting the prune;
3. retains the old widget object and confirms it is both attached and
   `_pruning`; also confirm the ordinary dismissal helper omits it while the
   unfiltered screen query still finds it;
4. directly awaits `_text_selected` before any `pilot.pause()` or other yield
   that could settle the pending prune; and
5. asserts the old widget is detached, exactly one replacement menu is
   mounted, the replacement is a different object and is not `_pruning`; and
6. after one `pilot.pause()`, asserts that the same replacement remains mounted
   and the app is still running.

Keep the existing slower consecutive-drag pilot test as complementary
settled-interaction coverage, but rename it or update its docstring so it does
not claim to exercise the no-yield race. Run the focused Console
selection-menu, transcript, and dismissal suites that exercise the changed
seam.

## Delivery and ADR Check

This is one atomic corrective task and one PR-sized change.

ADR required: no
ADR path: N/A
Reason: this is a lifecycle race fix within the existing Textual ownership and
screen-scoped menu architecture; it introduces no new boundary or policy.
