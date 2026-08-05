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

__all__ = ["PruneSafeSelect", "PruneSafeSelectCompatibilityError"]

#: The stock `Select` methods this class overrides. If Textual renames either,
#: the overrides below stop being overrides: Python happily defines two unused
#: methods, nothing raises, and the guard is silently bypassed -- the race
#: returns as intermittent flakiness with no signal at all. That is the one
#: failure mode this task refuses (a hidden race), so it is checked at import.
_GUARDED_METHODS = ("_setup_options_renderables", "_watch_value")

#: The private flags the guard reads. Set by `DOMNode`/`MessagePump.__init__`,
#: so they exist on every constructed widget -- until they do not.
_GUARDED_FLAGS = ("_pruning", "_closing")

#: Named in the failure message so whoever hits it knows what to redo, not
#: merely that something moved.
_REVERIFY = (
    "Textual internals PruneSafeSelect guards against have changed; "
    "re-verify the prune-time mount no-op mechanism (TASK-1960) against "
    "this Textual version"
)


class PruneSafeSelectCompatibilityError(RuntimeError):
    """Raised when the Textual internals this class hooks have moved."""


def _textual_version() -> str:
    """Best-effort Textual version, for the failure message only."""
    try:
        from importlib.metadata import version

        return version("textual")
    except Exception:  # pragma: no cover - diagnostics must never mask the real error
        return "unknown"


def _require_internals(owner: object, names: tuple[str, ...], kind: str) -> None:
    """Fail loudly if any of `names` is missing from `owner`.

    Args:
        owner: The class (for methods) or instance (for flags) to check.
        names: Attribute names that must be present.
        kind: Human-readable noun for the message, e.g. `"method"`.

    Raises:
        PruneSafeSelectCompatibilityError: If any name is absent. Deliberately
            an exception, not a warning: a warning in a TUI goes to a log
            nobody reads while the app keeps crashing intermittently.
    """
    missing = [name for name in names if not hasattr(owner, name)]
    if missing:
        owner_name = getattr(owner, "__name__", type(owner).__name__)
        raise PruneSafeSelectCompatibilityError(
            f"{_REVERIFY}. PruneSafeSelect expects {kind}(s) "
            f"{', '.join(missing)} on {owner_name} "
            f"(textual {_textual_version()}); pin textual and re-run "
            f"Tests/Widgets/test_prune_safe_select.py, whose two "
            f"`test_stock_select_still_raises_*` controls say whether the "
            f"underlying Textual defect still exists at all."
        )


# Import-time half: the overrides are only overrides while these names exist.
_require_internals(Select, _GUARDED_METHODS, "method")


class PruneSafeSelect(Select):
    """A `Select` that declines to touch its children once it is pruning.

    Behaviourally identical to `textual.widgets.Select` in every state
    except `_pruning`/`_closing`, where the two mount-time steps that reach
    into children stop doing so rather than raising `NoMatches`. Non-DOM
    state (`_value`) is still kept in sync — see `_watch_value`.

    CSS is unaffected: Textual's type selectors match against every base
    class that inherits CSS (`DOMNode._css_bases`), so an existing
    `Select { ... }` or `.destination-filter-strip Select` rule still
    applies to instances of this class.

    This class couples to Textual private API on purpose — the defect it
    works around is itself in Textual private API, so there is nowhere
    public to hook. `pyproject.toml` allows any `textual>=8.2.8,<9`, and an
    in-range upgrade could rename any of it. Both halves are therefore
    checked (module import for the overridden methods, `__init__` for the
    flags) and failure is an exception, never a default-to-safe: a guard
    that quietly becomes inert would restore the exact intermittent race
    TASK-1960 exists to eliminate, which is strictly worse than a loud stop.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Construct a stock `Select`, then confirm the guard can still work.

        Cheap by design — two `hasattr` calls per widget, after
        `super().__init__` has had its chance to set them.
        """
        super().__init__(*args, **kwargs)
        _require_internals(self, _GUARDED_FLAGS, "attribute")

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
        """Skip only the DOM half of the watcher while pruning.

        This is the exact crash site of TASK-1960. Upstream's `_watch_value`
        guards `query_one(SelectCurrent)` but neither the `#label` lookup one
        level below it (`_select.py:256`) nor the `query_one(SelectOverlay)`
        in its own `else:` branch (`_select.py:613`) — three unguarded child
        lookups behind one guarded one.

        Split exactly along upstream's own line: `Select._watch_value`
        (`_select.py:601-617`) does precisely one thing that is not a child
        lookup — `self._value = value`, its first statement — and everything
        after it queries children. So the shadow is kept in sync here and
        only the DOM work is dropped, rather than dropping both. Skipping the
        assignment as well left `value` and `_value` divergent (measured in
        the TASK-1960 review: `value='all' _value=Select.NULL`), which is a
        worse state to hand to anything that reads the shadow — `_on_mount`
        does, via `_init_selected_option(self._value)`.

        Not painting is safe because `_pruning`/`_closing` are terminal in
        Textual 8.2.8: `App._prune` posts `Prune()` -> `_close_messages` ->
        `_message_loop_exit`, which unregisters the node, and the only reset
        (`App._register`, `app.py:3662`) requires re-registering this exact
        instance — which never happens here, since every Watchlists `Select`
        is constructed fresh inside `compose()`. That is an invariant of
        *Textual*, not of this class: a future Textual that revives pruned
        nodes would resurrect one holding a real `value` whose `#label` still
        shows the placeholder, because the reactive sees no change on the way
        back and never re-fires. If this class ever survives such an upgrade,
        that revival path needs an explicit repaint, not just this sync.
        """
        if self._pruning or self._closing:
            # The one non-DOM statement of `Select._watch_value`.
            self._value = value
            return
        super()._watch_value(value)
