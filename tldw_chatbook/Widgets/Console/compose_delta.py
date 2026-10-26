"""Apply a compose delta that moves no box to mounted widgets in place.

TASK-36012. A widget that rebuilds itself with ``refresh(recompose=True)``
removes and re-mounts every child; each mount asks Textual for a layout, and
Textual answers any layout request with a reflow of the whole screen. When a
new state changes only what some children *say* -- a label, a tooltip, a
colour class -- and not their boxes, that rebuild is pure cost.

``plan_in_place_delta`` compares the off-DOM compose output for the state the
widget shows (``before``) with the one for its next state (``after``) and,
only when the two trees have the same shape, ids, inline styles and plain
attributes, returns the writes that turn the mounted tree into ``after``:

* a tooltip, a class set, or a plain (str/int/float/bool/None) attribute;
* a Button label or Static content, only when the mounted widget's width and
  height are both non-``auto`` -- its box cannot depend on its content, so a
  repaint without layout lands the same geometry a rebuild would.

Anything else returns ``None`` and the caller rebuilds as before. Each delta
is also checked against the mounted widget (``mounted`` must still show the
``before`` value), so a tree that drifted from the state it claims to show is
rebuilt rather than patched. A class change that moves a layout rule still
asks for its own layout through Textual's stylesheet, so the patch never
needs to know which classes are visual.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from rich.text import Text
from textual.content import Content
from textual.widget import Widget
from textual.widgets import Button, Input, Static

_PLAIN_TYPES = (str, int, float, bool, type(None))
#: Public attributes Textual sets per instance rather than per state:
#: ``Widget.sort_order`` is a construction counter.
_PER_INSTANCE_ATTRIBUTES = frozenset({"sort_order"})


def _flatten(
    nodes: Sequence[Widget], children: Callable[[Widget], Sequence[Widget]]
) -> list[tuple[int, Widget]]:
    flat: list[tuple[int, Widget]] = []

    def walk(items: Sequence[Widget], depth: int) -> None:
        for node in items:
            flat.append((depth, node))
            walk(children(node), depth + 1)

    walk(nodes, 0)
    return flat


def _composed_children(node: Widget) -> Sequence[Widget]:
    return list(node._pending_children)


def _mounted_children(node: Widget) -> Sequence[Widget]:
    return [child for child in node.children if not child.has_class("-textual-system")]


def _plain_attributes(node: Widget) -> dict[str, Any]:
    return {
        name: value
        for name, value in vars(node).items()
        if not name.startswith("_")
        and name not in _PER_INSTANCE_ATTRIBUTES
        and isinstance(value, _PLAIN_TYPES)
    }


def _copy(node: Widget) -> Any:
    if isinstance(node, Button):
        return node.label
    if isinstance(node, Static):
        return node.content
    return None


def _same(left: Any, right: Any) -> bool:
    """Equality that sees styling: ``Content.__eq__`` compares text only."""
    return _key(left) == _key(right)


def _key(value: Any) -> Any:
    if isinstance(value, Content):
        return ("content", value.plain, tuple(value.spans))
    if isinstance(value, Text):
        return ("text", value.plain, tuple(value.spans), str(value.style))
    return value


def _fixed_inputs(node: Widget) -> tuple[Any, ...]:
    """State a compose sets that this patch never rewrites."""
    fixed: tuple[Any, ...] = (node.disabled,)
    if isinstance(node, Input):
        fixed += (node.value, node.placeholder, node.password)
    return fixed


def _box_ignores_content(node: Widget) -> bool:
    """True when neither of ``node``'s box sides is sized by its content.

    An unset side fills its container. An ``fr`` width counts as
    content-sized: Textual resolves it as ``auto`` when an auto-width parent
    arranges for its optimal size.
    """
    width = node.styles.width
    height = node.styles.height
    return (width is None or not (width.is_auto or width.is_fraction)) and (
        height is None or not height.is_auto
    )


def plan_in_place_delta(
    mounted: Sequence[Widget],
    before: Sequence[Widget],
    after: Sequence[Widget],
) -> list[Callable[[], None]] | None:
    """Return the writes that turn ``mounted`` into ``after``, or ``None``.

    Args:
        mounted: The widget's mounted top-level children, minus any it did
            not compose itself.
        before: Off-DOM compose output for the state ``mounted`` shows.
        after: Off-DOM compose output for the next state.

    Returns:
        Zero-argument writes to apply in order, or ``None`` when the delta
        needs a rebuild (a different shape, a style or input change, a copy
        change in a content-sized box, or a mounted tree that no longer shows
        ``before``).
    """
    live = _flatten(mounted, _mounted_children)
    old = _flatten(before, _composed_children)
    new = _flatten(after, _composed_children)
    if not len(live) == len(old) == len(new):
        return None
    writes: list[Callable[[], None]] = []
    for (live_depth, node), (old_depth, was), (new_depth, will) in zip(live, old, new):
        if not live_depth == old_depth == new_depth:
            return None
        if not type(node) is type(was) is type(will):
            return None
        if not node.id == was.id == will.id:
            return None
        if was.styles.inline.get_rules() != will.styles.inline.get_rules():
            return None
        if _fixed_inputs(was) != _fixed_inputs(will):
            return None
        plan = _node_writes(node, was, will)
        if plan is None:
            return None
        writes.extend(plan)
    return writes


def _node_writes(
    node: Widget, was: Widget, will: Widget
) -> list[Callable[[], None]] | None:
    writes: list[Callable[[], None]] = []
    old_attributes = _plain_attributes(was)
    new_attributes = _plain_attributes(will)
    if old_attributes.keys() != new_attributes.keys():
        return None
    live_attributes = vars(node)
    for name, value in new_attributes.items():
        if value == old_attributes[name]:
            continue
        if live_attributes.get(name) != old_attributes[name]:
            return None
        writes.append(
            lambda node=node, name=name, value=value: setattr(node, name, value)
        )
    if not _same(was.tooltip, will.tooltip):
        if not _same(node.tooltip, was.tooltip):
            return None
        writes.append(lambda node=node, tip=will.tooltip: setattr(node, "tooltip", tip))
    if was.classes != will.classes:
        removed = was.classes - will.classes
        added = will.classes - was.classes
        if not removed <= node.classes or added & node.classes:
            return None
        writes.append(
            lambda node=node, removed=removed, added=added: (
                node.remove_class(*removed),
                node.add_class(*added),
            )
        )
    old_copy, new_copy = _copy(was), _copy(will)
    if not _same(old_copy, new_copy):
        if not _same(_copy(node), old_copy) or not _box_ignores_content(node):
            return None
        if isinstance(node, Button):
            writes.append(
                lambda node=node, label=new_copy: setattr(node, "label", label)
            )
        else:
            writes.append(
                lambda node=node, content=new_copy: node.update(content, layout=False)
            )
    return writes
