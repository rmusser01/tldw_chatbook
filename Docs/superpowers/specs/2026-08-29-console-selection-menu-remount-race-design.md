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
2. `_text_selected` obtains the existing unfiltered screen-scoped candidate
   list from `selection_menus_on_screen(self.screen)` and awaits `remove()` for
   every returned menu before mounting the replacement.

The registry already re-derives attachment from the live Textual DOM and is
screen-scoped. It includes an attached `_pruning` menu, so no new state or
bookkeeping is required.

## Failure Handling

The invalid DOM state is prevented before `mount`; `DuplicateIds` is not caught
or suppressed. Genuine Textual mount failures continue to propagate. Existing
detached-screen behavior remains unchanged because resolving `self.screen`
still occurs at the same boundary.

## Verification

Add a deterministic regression that:

1. mounts a selection menu;
2. schedules its removal without awaiting the prune;
3. confirms the old menu is both attached and `_pruning`;
4. invokes the remount path before yielding to Textual; and
5. asserts the app remains running with exactly one mounted menu.

Keep the existing slower consecutive-drag pilot test as complementary
interaction coverage. Run the focused Console selection-menu, transcript, and
dismissal suites that exercise the changed seam.

## Delivery and ADR Check

This is one atomic corrective task and one PR-sized change.

ADR required: no
ADR path: N/A
Reason: this is a lifecycle race fix within the existing Textual ownership and
screen-scoped menu architecture; it introduces no new boundary or policy.
