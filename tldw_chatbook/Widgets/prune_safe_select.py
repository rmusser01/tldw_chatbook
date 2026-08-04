# prune_safe_select.py
# Description: A `Select` that stays silent while Textual is pruning it.
#
"""Prune-safe `Select` (TASK-1960).

Textual's `Select` crashes with `NoMatches: No nodes match '#label' on
SelectCurrent(...)` when a DOM prune lands inside its own mount cascade. The
crash is not a `Select` bug so much as a consequence of two Textual
behaviours that are individually reasonable and jointly unsound:

1. `Widget.mount` (`textual/widget.py`) begins with

       if self._closing or self._pruning:
           return AwaitMount(self, [])

   — a widget that is being pruned silently mounts *nothing* and hands back
   an already-satisfied awaitable. No exception, no log line.

2. `MessagePump._pre_process` (`textual/message_pump.py`) dispatches
   `Compose` and then `Mount`, and its `finally` block sets
   `_mounted_event` and `_is_mounted = True` **unconditionally** — including
   when the `Compose` dispatch mounted nothing because of (1).

Put together: a `SelectCurrent` that is registered into the DOM and *then*
caught by a prune before its own `Compose` runs ends up reporting
`is_mounted=True` with zero children. Its parent `Select`'s
`AwaitMount` unblocks, `Select._on_mount` runs, and both of the steps it
performs reach into children that were never mounted:

    Select._on_mount
      -> _setup_options_renderables -> self.query_one(SelectOverlay)
      -> _init_selected_option -> self.value = hint -> _watch_value
           -> self.query_one(SelectCurrent)     # guarded upstream
           -> SelectCurrent.update(prompt)
              -> self.query_one("#label", Static)   # NOT guarded: raises

Upstream guards only the outer `query_one(SelectCurrent)` — the case where
the `Select`'s own children are missing. It does not guard the narrower case
where `SelectCurrent` exists but its `#label` child does not.

The fix here is deliberately *not* a catch-and-carry-on. It is gated on
`_pruning`/`_closing`, which mean "Textual has already committed to removing
this widget from the DOM". A pruning widget is never painted again, so
declining to initialise it loses nothing: there is no stale-placeholder
failure mode, because there is no surface left to be stale. Any *other* way
of arriving at a half-composed `Select` is a different bug and still raises
loudly, which is what we want.

Screens whose reactives recompose while background workers are still
running — anything of the "a full-screen `recompose=True` reactive rebuilds
every region" shape — should use this class instead of `Select`.
"""

from __future__ import annotations

from textual.widgets import Select

__all__ = ["PruneSafeSelect"]


class PruneSafeSelect(Select):
    """A `Select` that declines to touch its children once it is pruning.

    Behaviourally identical to `textual.widgets.Select` in every state
    except `_pruning`/`_closing`, where its two mount-time children-touching
    steps become no-ops rather than raising `NoMatches`.

    CSS is unaffected: Textual's type selectors match against every base
    class that inherits CSS (`DOMNode._css_bases`), so an existing
    `Select { ... }` or `.destination-filter-strip Select` rule still
    applies to instances of this class.
    """

    def _setup_options_renderables(self) -> None:
        """Skip the `query_one(SelectOverlay)` rebuild while pruning.

        Reached from `_on_mount` and from `set_options`. The overlay is a
        child mounted by `Select.compose`, so on a pruned-mid-compose
        `Select` it may not exist at all.
        """
        if self._pruning or self._closing:
            return
        super()._setup_options_renderables()

    def _watch_value(self, value) -> None:
        """Skip the `SelectCurrent.update` write-through while pruning.

        This is the exact crash site of TASK-1960: upstream's `_watch_value`
        guards `query_one(SelectCurrent)` but not the `#label` lookup one
        level below it.
        """
        if self._pruning or self._closing:
            return
        super()._watch_value(value)
