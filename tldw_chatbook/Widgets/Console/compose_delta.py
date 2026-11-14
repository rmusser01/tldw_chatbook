"""Decide a compose delta that moves no box without composing it.

TASK-36012. A widget that rebuilds itself with ``refresh(recompose=True)``
removes and re-mounts every child; each mount asks Textual for a layout, and
Textual answers any layout request with a reflow of the whole screen. When a
new state changes only what some children *say* -- a label, a tooltip, a
colour class -- and not their boxes, that rebuild is pure cost, and the
widget can write the new values into its mounted children instead.

TASK-33620.15.20. Deciding that by composing the widget twice off the DOM
and diffing the two trees cost as much as the rebuild it replaced: every
widget those composes built was constructed and restyled (each
``add_class``/``set_class`` restyles its node at once, mounted or not), and
then dropped. A widget that patches in place now decides from its state:
compose records the values it wrote that a delta may change, and the patch
compares that record with the values the next state gives. The helpers here
are the parts of that comparison that do not depend on the widget:

* ``ComposeReadView`` answers, before anything else, which state fields the
  last compose read. A delta confined to fields it did not read needs neither
  a patch nor a rebuild.
* ``same_visual`` compares two label, content or tooltip values the way they
  render, style included.
* ``box_ignores_content`` says whether a widget's copy can change without a
  layout: only then does a repaint land the geometry a rebuild would.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.content import Content
from textual.widget import Widget


def _visual_key(value: Any) -> Any:
    if isinstance(value, Content):
        return ("content", value.plain, tuple(value.spans))
    if isinstance(value, Text):
        return ("text", value.plain, tuple(value.spans), str(value.style))
    return value


def same_visual(left: Any, right: Any) -> bool:
    """Return whether two labels, contents or tooltips render the same.

    ``Content.__eq__`` compares text only, and ``Text.__eq__`` ignores the
    base style (a colour given as ``Text(icon, style=color)``), so plain
    equality would call a colour change "unchanged".

    Args:
        left: A ``Content``, ``Text``, string or ``None``.
        right: Another such value.

    Returns:
        True when the two show the same text with the same styling.
    """
    return _visual_key(left) == _visual_key(right)


def box_ignores_content(node: Widget) -> bool:
    """True when neither of ``node``'s box sides is sized by its content.

    An unset side fills its container. An ``fr`` width counts as
    content-sized: Textual resolves it as ``auto`` when an auto-width parent
    arranges for its optimal size.

    Args:
        node: A mounted widget whose label or content may change.

    Returns:
        True when new copy repaints inside the same box.
    """
    width = node.styles.width
    height = node.styles.height
    return (width is None or not (width.is_auto or width.is_fraction)) and (
        height is None or not height.is_auto
    )


class ComposeReadView:
    """Attribute-recording view of a widget's state, live only during compose.

    TASK-26836: the Conversations tray's ``_can_skip_recompose`` used
    whole-state value equality, so a delta in a field the tray's ``content``
    never renders (probe-observed: ``conversation_browser`` pushed into the
    ``content="workspace"`` tray) still forced a full recompose -- and its
    paint-blocking App batch. The tray now records which top-level state
    fields ``compose`` actually reads, exactly as it already records the row
    signature of what it builds, and the guard skips when only unread fields
    changed.

    TASK-36012 gave it a second owner: the Details tray rebuilds only when a
    field its last completed compose read changes.

    The view delegates everything to the real state; reads observed while it
    is live can only ADD fields to the recorded set, so interleaved readers
    during a compose window bias toward recomposing more, never less.
    """

    __slots__ = ("_target", "_reads")

    def __init__(self, target: object, reads: set) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_reads", reads)

    def __getattr__(self, name: str):
        if not name.startswith("_"):
            self._reads.add(name)
        return getattr(self._target, name)

    def __eq__(self, other: object) -> bool:
        return self._target == getattr(other, "_target", other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        return self._target is not None

    def __repr__(self) -> str:
        return f"ComposeReadView({self._target!r})"
