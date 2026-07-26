"""Left-rail watchlist tree: roots, watchlists, lazily-loaded sources.

This is the screen's primary navigation surface. Selecting a node sets a
*scope* that the Feeds and Items regions read, which is why the message
carries a structured `TreeScope` rather than a bare id — "watchlist 1" and
"source 10 inside watchlist 1" are different scopes with the same numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Static


ALL_SOURCES_BUCKET = -2
UNASSIGNED_BUCKET = -1


@dataclass(frozen=True)
class TreeScope:
    """What the user has selected, as the panes need to understand it."""

    kind: Literal["all", "unassigned", "watchlist", "source"]
    watchlist_id: int | None = None
    source_id: int | None = None


class TreeScopeChanged(Message):
    """Posted when the tree selection changes."""

    def __init__(self, scope: TreeScope) -> None:
        self.scope = scope
        super().__init__()


class WatchlistTree(Vertical):
    """Roots, watchlists with counts, lazily-expanded sources, tag filters."""

    expanded: reactive[frozenset[int]] = reactive(frozenset(), recompose=True)
    active_tag: reactive[str | None] = reactive(None, recompose=True)

    def __init__(
        self,
        watchlists: Sequence[Mapping[str, Any]],
        counts: Mapping[int, Mapping[str, int]],
        source_rows_loader: Callable[[int], Sequence[Mapping[str, Any]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_class("watchlist-tree")
        self._watchlists = list(watchlists)
        self._counts = dict(counts)
        self._load_source_rows = source_rows_loader
        self._source_cache: dict[int, list[Mapping[str, Any]]] = {}

    # --- rendering ---

    def compose(self) -> ComposeResult:
        yield self._root_node("all", "All sources", ALL_SOURCES_BUCKET)
        yield self._root_node("unassigned", "Unassigned", UNASSIGNED_BUCKET)

        for watchlist in self._visible_watchlists():
            yield from self._watchlist_node(watchlist)

        tags = self._all_tags()
        if tags:
            yield Static("", classes="watchlist-tree-spacer")
            for index, tag in enumerate(tags):
                # Tag text is free-form (spaces, slashes, non-ASCII) and
                # Textual ids are restricted to [a-zA-Z_-][a-zA-Z0-9_-]*, so
                # the id is the tag's position in the ordered tag list, not
                # the tag text itself. The visible label still shows the
                # (escaped) tag text.
                button = Button(f"#{escape_markup(tag)}", id=f"wl-tree-tag-{index}", compact=True)
                button.add_class("watchlist-tree-tag")
                if tag == self.active_tag:
                    button.add_class("is-active")
                yield button

    def _root_node(self, key: str, label: str, bucket: int) -> Button:
        unread = self._counts.get(bucket, {}).get("unread", 0)
        button = Button(f"{label}  {unread}", id=f"wl-tree-node-{key}", compact=True)
        button.add_class("watchlist-tree-root")
        return button

    def _watchlist_node(self, watchlist: Mapping[str, Any]) -> ComposeResult:
        watchlist_id = int(watchlist["id"])
        unread = self._counts.get(watchlist_id, {}).get("unread", 0)
        is_open = watchlist_id in self.expanded
        caret = "▾" if is_open else "▸"

        expander = Button(caret, id=f"wl-tree-expand-{watchlist_id}", compact=True)
        expander.add_class("watchlist-tree-expander")
        yield expander

        node = Button(
            f"{escape_markup(str(watchlist['name']))}  {unread}",
            id=f"wl-tree-node-watchlist-{watchlist_id}",
            compact=True,
        )
        node.add_class("watchlist-tree-watchlist")
        yield node

        if is_open:
            for row in self._source_rows(watchlist_id):
                # A source can belong to more than one watchlist, so the id
                # is qualified by watchlist — otherwise two expanded
                # watchlists sharing a source would mount two buttons with
                # the same id (a MountError) and the scope would be
                # ambiguous besides.
                source = Button(
                    f"  {escape_markup(str(row['name']))}",
                    id=f"wl-tree-node-source-{watchlist_id}-{row['id']}",
                    compact=True,
                )
                source.add_class("watchlist-tree-source")
                yield source

    # --- data ---

    def _visible_watchlists(self) -> list[Mapping[str, Any]]:
        if self.active_tag is None:
            return self._watchlists
        return [w for w in self._watchlists if self.active_tag in (w.get("tags") or [])]

    def _all_tags(self) -> list[str]:
        seen: list[str] = []
        for watchlist in self._watchlists:
            for tag in watchlist.get("tags") or []:
                if tag not in seen:
                    seen.append(tag)
        return seen

    def _source_rows(self, watchlist_id: int) -> list[Mapping[str, Any]]:
        """Fetch a watchlist's sources once, on first expand."""
        if watchlist_id not in self._source_cache:
            self._source_cache[watchlist_id] = list(self._load_source_rows(watchlist_id))
        return self._source_cache[watchlist_id]

    # --- interaction ---

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""

        if button_id.startswith("wl-tree-expand-"):
            event.stop()
            watchlist_id = int(button_id.rsplit("-", 1)[1])
            expanded = set(self.expanded)
            expanded.symmetric_difference_update({watchlist_id})
            self.expanded = frozenset(expanded)
            return

        if button_id.startswith("wl-tree-tag-"):
            event.stop()
            index = int(button_id[len("wl-tree-tag-"):])
            tags = self._all_tags()
            if 0 <= index < len(tags):
                tag = tags[index]
                self.active_tag = None if tag == self.active_tag else tag
            return

        scope: TreeScope | None = None
        if button_id == "wl-tree-node-all":
            scope = TreeScope(kind="all")
        elif button_id == "wl-tree-node-unassigned":
            scope = TreeScope(kind="unassigned")
        elif button_id.startswith("wl-tree-node-watchlist-"):
            scope = TreeScope(kind="watchlist", watchlist_id=int(button_id.rsplit("-", 1)[1]))
        elif button_id.startswith("wl-tree-node-source-"):
            remainder = button_id[len("wl-tree-node-source-"):]
            watchlist_part, _, source_part = remainder.partition("-")
            scope = TreeScope(
                kind="source",
                watchlist_id=int(watchlist_part),
                source_id=int(source_part),
            )

        if scope is not None:
            event.stop()
            self.post_message(TreeScopeChanged(scope))
