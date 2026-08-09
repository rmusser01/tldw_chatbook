"""Left-rail watchlist tree: roots, watchlists, lazily-loaded sources.

This is the screen's primary navigation surface. Selecting a node sets a
*scope* that the centre header's scoped summary and the Items region read,
which is why the message carries a structured `TreeScope` rather than a
bare id — "watchlist 1" and "source 10 inside watchlist 1" are different
scopes with the same numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Static


ALL_SOURCES_BUCKET = -2
UNASSIGNED_BUCKET = -1
# The Starred smart feed (TASK-3072 plan task 6). Its badge rides the same
# `counts` mapping as every other node -- the screen inserts this bucket
# from `get_flagged_items_count()` on every tree-data load, so the badge
# refreshes through the existing debounced counts path.
STARRED_BUCKET = -3
# The Today smart feed (TASK-3791 plan task 4): unread items at/after local
# midnight, from `get_unread_items_count_since`. All Unread needs no bucket
# of its own -- its badge is the same unread count `ALL_SOURCES_BUCKET`
# already carries (one fact, two angles).
TODAY_BUCKET = -4

# App-controlled constant, never user data: the same markup rule as the
# article list's glyphs (`_STAR_GLYPH` there) -- a remote feed title never
# supplies this string.
_STARRED_LABEL = "★ Starred"


@dataclass(frozen=True)
class TreeScope:
    """What the user has selected, as the panes need to understand it."""

    kind: Literal["all", "unassigned", "watchlist", "source", "starred", "unread", "today"]
    watchlist_id: int | None = None
    source_id: int | None = None


class TreeScopeChanged(Message):
    """Posted when the tree selection changes."""

    def __init__(self, scope: TreeScope) -> None:
        self.scope = scope
        super().__init__()


class TreeExpansionChanged(Message):
    """Posted when the user expands or collapses a watchlist node.

    The owning screen mirrors this, for the same reason `SourcesPane` posts
    `CreateFormDraftChanged`: this widget is built by a factory the screen
    calls on every full recompose (a section switch, a tree-data reload, a
    local-snapshot apply), so a brand new `WatchlistTree` is constructed and
    pane-local expansion would silently collapse under the user.
    """

    def __init__(self, expanded: frozenset[int]) -> None:
        self.expanded = expanded
        super().__init__()


class TreeTagFilterChanged(Message):
    """Posted when the user sets or clears the rail's tag filter.

    Screen-mirrored for the identical reason as `TreeExpansionChanged`.
    """

    def __init__(self, tag: str | None) -> None:
        self.tag = tag
        super().__init__()


class CreateWatchlistRequested(Message):
    """Posted when the user asks for a new watchlist (task-895).

    Carries no id: creation has no subject yet. The owning screen prompts
    for a name and calls ``WatchlistBundleService.create``; the tree does
    not touch the service itself, matching how every other action on this
    screen is routed (`CreateSourceRequested`, `SaveRuleRequested`, ...).
    """


class RenameWatchlistRequested(Message):
    """Posted when the user asks to rename the watchlist in scope."""

    def __init__(self, watchlist_id: int) -> None:
        self.watchlist_id = watchlist_id
        super().__init__()


class DeleteWatchlistRequested(Message):
    """Posted when the user asks to delete the watchlist in scope."""

    def __init__(self, watchlist_id: int) -> None:
        self.watchlist_id = watchlist_id
        super().__init__()


class AddSourceToWatchlistRequested(Message):
    """Posted when the user asks to add a source to the watchlist in scope."""

    def __init__(self, watchlist_id: int) -> None:
        self.watchlist_id = watchlist_id
        super().__init__()


class RemoveSourceFromWatchlistRequested(Message):
    """Posted when the user asks to drop the source node in scope.

    Both ids are carried because membership is many-to-many: "source 10"
    alone does not say which watchlist it is being removed from, the same
    reason the source node's own id is watchlist-qualified.
    """

    def __init__(self, watchlist_id: int, source_id: int) -> None:
        self.watchlist_id = watchlist_id
        self.source_id = source_id
        super().__init__()


class WatchlistTree(Vertical):
    """Roots, watchlists with counts, lazily-expanded sources, tag filters."""

    expanded: reactive[frozenset[int]] = reactive(frozenset(), recompose=True)
    active_tag: reactive[str | None] = reactive(None, recompose=True)
    # task-876: the node matching the screen's `tree_scope` -- read-only from
    # this widget's own perspective. Unlike `expanded`/`active_tag`, this
    # value never originates from a click inside the tree itself: a real
    # tree click already knows which scope it just posted (the owning screen
    # reconciles `tree_scope` in `_apply_tree_scope`, since a breadcrumb
    # promotion moves the same reactive without touching this widget at
    # all), so there is no watcher here mirroring it back up -- only the
    # screen writes it, via `_build_tree_pane` (seeding a freshly constructed
    # instance, the same way `expanded`/`active_tag` are seeded) and
    # `watch_tree_scope` (pushing into the still-mounted instance after a
    # real click or a breadcrumb promotion, since neither rebuilds this
    # widget on its own).
    active_scope: reactive["TreeScope | None"] = reactive(None, recompose=True)
    # task-895: why the five write verbs cannot run right now, or `None`
    # when they can. A single string because it is used verbatim in two
    # places -- the disabled buttons' tooltips and the visible note under
    # them -- so there is no way for the hover copy and the on-screen copy
    # to drift apart. Screen-owned like every other reactive here (the
    # screen derives it from `runtime_backend` and service availability),
    # and `recompose=True` because it changes which buttons are disabled.
    write_disabled_reason: reactive[str | None] = reactive(None, recompose=True)

    def __init__(
        self,
        watchlists: Sequence[Mapping[str, Any]],
        counts: Mapping[int, Mapping[str, int]],
        source_rows_loader: Callable[[int], Sequence[Mapping[str, Any]]],
        expanded: frozenset[int] | Sequence[int] = (),
        active_tag: str | None = None,
        active_scope: "TreeScope | None" = None,
        write_disabled_reason: str | None = None,
        source_counts: Mapping[int, Mapping[str, int]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_class("watchlist-tree")
        self._watchlists = list(watchlists)
        self._counts = dict(counts)
        self._load_source_rows = source_rows_loader
        # Per-source {total, unread} for the source-node badges (task-2513
        # Task 8). `counts` above is keyed by watchlist/bucket and cannot
        # answer for a source; a source missing here renders no badge,
        # which is the honest state for "no items yet".
        self._source_counts: dict[int, Mapping[str, int]] = dict(source_counts or {})
        self._source_cache: dict[int, list[Mapping[str, Any]]] = {}
        # `set_reactive`, not plain assignment: both reactives are
        # `recompose=True`, so assigning here would queue a recompose of a
        # widget that has not composed once yet -- rebuilding identical
        # children a tick after mount -- and would fire the watchers below,
        # bouncing the seeded value straight back at the screen as if the
        # user had just expanded something. `set_reactive` sets the value
        # without validators or watchers, and `compose()` reads it on the
        # FIRST render because it runs after this constructor.
        self.set_reactive(WatchlistTree.expanded, frozenset(expanded))
        self.set_reactive(WatchlistTree.active_tag, active_tag)
        self.set_reactive(WatchlistTree.active_scope, active_scope)
        self.set_reactive(WatchlistTree.write_disabled_reason, write_disabled_reason)

    # --- rendering ---

    #: TASK-2304 AC#3. Every node in this tree renders a bare number after
    #: its name, and nothing on screen said what that number counted -- the
    #: 2026-08-04 UAT watched it read 0 while the centre header said
    #: "(1 source)" and reasonably concluded the two disagreed about the same
    #: fact. They never described the same fact: this one is UNREAD ITEMS
    #: (`SubscriptionsDB.get_watchlist_item_counts`), the header's is SOURCES
    #: in the current tree scope.
    #:
    #: A legend row rather than a per-node suffix ("All sources  0 unread"):
    #: the rail's interior is 26 columns (`.watchlists-region-left_rail` is
    #: 28 wide), a suffix costs 7 of them on EVERY node, and any label that
    #: overflows renders with an ellipsis -- which
    #: `test_watchlists_left_rail_is_labelled_when_expanded` rightly fails on.
    #: One row labels every node at once and cannot truncate. The tooltips
    #: say it too, so the answer is available both ways.
    _COUNT_LEGEND = "Counts: unread items"

    def compose(self) -> ComposeResult:
        yield from self._action_bar()
        yield Static(
            self._COUNT_LEGEND,
            id="wl-tree-count-legend",
            classes="watchlist-tree-count-legend",
        )
        yield self._root_node("all", "All sources", ALL_SOURCES_BUCKET)
        yield self._root_node("unassigned", "Unassigned", UNASSIGNED_BUCKET)
        # TASK-3791 plan task 4: the smart-feed cluster. All Unread reads
        # the same bucket as All sources (its badge IS the global unread
        # count); Today's rides TODAY_BUCKET, which the screen inserts from
        # the local-midnight unread count on every tree-data load.
        yield self._root_node("unread", "All Unread", ALL_SOURCES_BUCKET)
        yield self._root_node("today", "Today", TODAY_BUCKET)
        yield self._root_node(
            "starred",
            _STARRED_LABEL,
            STARRED_BUCKET,
            phrase=self._starred_phrase,
        )

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
                button = Button(
                    f"#{escape_markup(tag)}",
                    id=f"wl-tree-tag-{index}",
                    compact=True,
                    tooltip=(
                        f"Clear the \"{escape_markup(tag)}\" tag filter."
                        if tag == self.active_tag
                        else f"Show only watchlists tagged \"{escape_markup(tag)}\"."
                    ),
                )
                button.add_class("watchlist-tree-tag")
                if tag == self.active_tag:
                    button.add_class("is-active")
                yield button

    # Copy for an action that *could* run but has no subject yet. Kept
    # beside the action table rather than inlined so the two "pick
    # something first" messages read as one vocabulary, and so a disabled
    # button always has a reason -- a bare `disabled=True` with no tooltip
    # is the shape of defect this program has already fixed once.
    _NEEDS_WATCHLIST = "Select a watchlist in the tree first."
    _NEEDS_SOURCE = "Select a source inside a watchlist first."

    # Matched exactly, not by prefix: `wl-tree-remove-source` and
    # `wl-tree-node-source-1-10` would both survive a naive
    # `startswith("wl-tree-")` split, and the node ids are parsed positionally
    # further down.
    _ACTION_BUTTON_IDS = frozenset(
        {
            "wl-tree-new",
            "wl-tree-rename",
            "wl-tree-delete",
            "wl-tree-add-source",
            "wl-tree-remove-source",
        }
    )

    def _action_bar(self) -> ComposeResult:
        """The tree's five write verbs, plus why they are off when they are.

        Rendered above the roots so the rail reads "here is what you can do,
        here is what you have". Enablement is derived from `active_scope`
        (rename/delete/add operate on the watchlist in scope; remove
        operates on the source node in scope) and from
        `write_disabled_reason`, which the screen sets when the backend or
        the runtime cannot service a write at all.

        Every disabled button carries the reason as its tooltip, and the
        blocking reason is *also* rendered as a visible line when it is not
        scope-related -- a user should not have to hover to learn that the
        server backend has no wire path for these edits.
        """
        scope = self.active_scope
        reason = self.write_disabled_reason
        on_watchlist = (
            scope is not None
            and scope.kind == "watchlist"
            and scope.watchlist_id is not None
        )
        on_source = (
            scope is not None
            and scope.kind == "source"
            and scope.watchlist_id is not None
            and scope.source_id is not None
        )

        def action(
            label: str, button_id: str, *, allowed: bool, blocked_copy: str, ready_copy: str
        ) -> Button:
            disabled_reason = reason if reason else (None if allowed else blocked_copy)
            button = Button(
                label,
                id=button_id,
                compact=True,
                disabled=disabled_reason is not None,
                tooltip=disabled_reason or ready_copy,
            )
            button.add_class("watchlist-tree-action")
            return button

        yield Horizontal(
            action(
                "New",
                "wl-tree-new",
                allowed=True,
                blocked_copy="",
                ready_copy="Create a new watchlist.",
            ),
            action(
                "Rename",
                "wl-tree-rename",
                allowed=on_watchlist,
                blocked_copy=self._NEEDS_WATCHLIST,
                ready_copy="Rename the selected watchlist.",
            ),
            action(
                "Delete",
                "wl-tree-delete",
                allowed=on_watchlist,
                blocked_copy=self._NEEDS_WATCHLIST,
                ready_copy="Delete the selected watchlist. Its sources are kept.",
            ),
            classes="watchlist-tree-actions",
        )
        yield Horizontal(
            # TASK-2303 AC#1. This button ASSIGNS a source that already
            # exists; it does not make one. Labelled "Add source" it was one
            # of three near-synonyms on the same screen ("Add source",
            # "Create source", "New Source") covering two different
            # operations, and the 2026-08-04 UAT read it as a third way to
            # create a feed. The vocabulary is split by verb now: NEW brings
            # a source into existence, ADD files one that already does, and
            # the tooltip names the other operation so a user who wanted it
            # is not left guessing.
            action(
                "Add existing",
                "wl-tree-add-source",
                allowed=on_watchlist,
                blocked_copy=self._NEEDS_WATCHLIST,
                ready_copy=(
                    "Add a source you already have to the selected watchlist. "
                    "To make a new one, use New source under Sources."
                ),
            ),
            action(
                "Remove",
                "wl-tree-remove-source",
                allowed=on_source,
                blocked_copy=self._NEEDS_SOURCE,
                ready_copy="Remove the selected source from its watchlist.",
            ),
            classes="watchlist-tree-actions",
        )
        if reason:
            yield Static(
                reason,
                id="wl-tree-actions-unavailable",
                classes="watchlist-tree-actions-note",
            )

    @staticmethod
    def _unread_phrase(unread: int) -> str:
        """"3 unread items" / "1 unread item" / "No unread items".

        TASK-2304 AC#3, the hover half of the answer. Says the word the bare
        number cannot, on the node the pointer is actually over, including
        the zero case -- "0" beside a name the user has just assigned a
        source to is the exact reading the UAT could not resolve.
        """
        if unread == 0:
            return "No unread items"
        noun = "item" if unread == 1 else "items"
        return f"{unread} unread {noun}"

    @staticmethod
    def _starred_phrase(starred: int) -> str:
        """The Starred root's hover phrase (TASK-3072).

        The legend row labels the rail's numbers "unread items"; this one
        badge counts STARRED items instead, so the node's own tooltip must
        say the true word -- the same TASK-2304 AC#3 honesty rule, applied
        to the one node the legend does not describe.
        """
        if starred == 0:
            return "No starred items"
        noun = "item" if starred == 1 else "items"
        return f"{starred} starred {noun}"

    def _root_node(
        self,
        key: str,
        label: str,
        bucket: int,
        *,
        phrase: Callable[[int], str] | None = None,
    ) -> Button:
        count = self._counts.get(bucket, {}).get("unread", 0)
        phrase_fn = phrase if phrase is not None else self._unread_phrase
        button = Button(
            f"{label}  {count}",
            id=f"wl-tree-node-{key}",
            compact=True,
            tooltip=f"Show {label.lower()}. {phrase_fn(count)}.",
        )
        button.add_class("watchlist-tree-root")
        # `key` is a root kind -- "all", "unassigned", "unread", "today" or
        # "starred" (the only five callers, in `compose()`) -- each also a
        # valid `TreeScope.kind` value.
        if self.active_scope == TreeScope(kind=key):
            button.add_class("is-active")
        return button

    def _watchlist_node(self, watchlist: Mapping[str, Any]) -> ComposeResult:
        watchlist_id = int(watchlist["id"])
        unread = self._counts.get(watchlist_id, {}).get("unread", 0)
        is_open = watchlist_id in self.expanded
        caret = "▾" if is_open else "▸"
        watchlist_name = escape_markup(str(watchlist["name"]))

        expander = Button(
            caret,
            id=f"wl-tree-expand-{watchlist_id}",
            compact=True,
            tooltip=(
                f"Collapse {watchlist_name}." if is_open else f"Expand {watchlist_name}."
            ),
        )
        expander.add_class("watchlist-tree-expander")

        node = Button(
            f"{watchlist_name}  {unread}",
            id=f"wl-tree-node-watchlist-{watchlist_id}",
            compact=True,
            tooltip=(
                f"Show sources in {watchlist_name}. "
                f"{self._unread_phrase(unread)}."
            ),
        )
        node.add_class("watchlist-tree-watchlist")
        if self.active_scope == TreeScope(kind="watchlist", watchlist_id=watchlist_id):
            node.add_class("is-active")

        # TASK-997: one row, not two. These were yielded as two separate
        # children of the tree's `Vertical`, so the chevron stacked ABOVE the
        # name -- and, inheriting Textual's `min-width: 16`, painted seven
        # columns in from the left of a 26-column rail rather than beside
        # anything. Every watchlist cost two rows of the screen's primary
        # navigation. The row is a container, so both buttons keep their own
        # ids and `Button.Pressed` still bubbles to `on_button_pressed` here.
        yield Horizontal(expander, node, classes="watchlist-tree-row")

        if is_open:
            for row in self._source_rows(watchlist_id):
                # A source can belong to more than one watchlist, so the id
                # is qualified by watchlist — otherwise two expanded
                # watchlists sharing a source would mount two buttons with
                # the same id (a MountError) and the scope would be
                # ambiguous besides.
                source_id = int(row["id"])
                source_name = escape_markup(str(row["name"]))
                # task-2513 Task 8: the per-feed unread badge. Unlike roots
                # and watchlists (which always show their count), a source
                # shows the number only when it is positive — a permanent
                # "0" on every feed is noise in the rail's narrowest indent,
                # and the tooltip still answers the zero case in words.
                source_unread = self._source_counts.get(source_id, {}).get("unread", 0)
                badge = f"  {source_unread}" if source_unread > 0 else ""
                source = Button(
                    # TASK-1091 left-aligns tree labels. Four spaces plus the
                    # Button line pad keep the source name one column past
                    # the parent name after the expander's three columns.
                    f"    {source_name}{badge}",
                    id=f"wl-tree-node-source-{watchlist_id}-{row['id']}",
                    compact=True,
                    tooltip=f"Show items from {source_name}. {self._unread_phrase(source_unread)}.",
                )
                source.add_class("watchlist-tree-source")
                if self.active_scope == TreeScope(
                    kind="source", watchlist_id=watchlist_id, source_id=source_id
                ):
                    source.add_class("is-active")
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

    def watch_expanded(self, expanded: frozenset[int]) -> None:
        """Tell the owning screen what is open, so it survives a recompose."""
        if self.is_mounted:
            self.post_message(TreeExpansionChanged(expanded))

    def watch_active_tag(self, tag: str | None) -> None:
        """Tell the owning screen the tag filter, for the same reason."""
        if self.is_mounted:
            self.post_message(TreeTagFilterChanged(tag))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""

        if button_id in self._ACTION_BUTTON_IDS:
            event.stop()
            self._post_action(button_id)
            return

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
        elif button_id == "wl-tree-node-unread":
            scope = TreeScope(kind="unread")
        elif button_id == "wl-tree-node-today":
            scope = TreeScope(kind="today")
        elif button_id == "wl-tree-node-starred":
            scope = TreeScope(kind="starred")
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

    def _post_action(self, button_id: str) -> None:
        """Turn an action press into the matching request message.

        Re-checks `write_disabled_reason` and the scope rather than trusting
        the `disabled=` flag `compose()` baked in: a disabled Button never
        emits `Pressed`, so in practice this is belt-and-braces -- but the
        scope can also be pushed in from the screen (`watch_tree_scope`)
        between renders, and posting a rename for a scope that names no
        watchlist would hand the screen an id it cannot resolve.
        """
        if self.write_disabled_reason:
            return
        if button_id == "wl-tree-new":
            self.post_message(CreateWatchlistRequested())
            return

        scope = self.active_scope
        if scope is None:
            return
        if button_id == "wl-tree-remove-source":
            if (
                scope.kind == "source"
                and scope.watchlist_id is not None
                and scope.source_id is not None
            ):
                self.post_message(
                    RemoveSourceFromWatchlistRequested(
                        watchlist_id=int(scope.watchlist_id),
                        source_id=int(scope.source_id),
                    )
                )
            return

        if scope.kind != "watchlist" or scope.watchlist_id is None:
            return
        watchlist_id = int(scope.watchlist_id)
        if button_id == "wl-tree-rename":
            self.post_message(RenameWatchlistRequested(watchlist_id))
        elif button_id == "wl-tree-delete":
            self.post_message(DeleteWatchlistRequested(watchlist_id))
        elif button_id == "wl-tree-add-source":
            self.post_message(AddSourceToWatchlistRequested(watchlist_id))
