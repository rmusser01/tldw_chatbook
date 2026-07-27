"""Inspector pane for the watchlists screen.

The Inspector renders a **breadcrumb stack** -- watchlist > source > item --
rather than a single flat level. The deepest current selection is expanded
into full detail plus its action buttons; every shallower level (an ancestor
in the tree the user drilled through to get here) collapses to one clickable
line. Clicking an ancestor crumb posts `BreadcrumbScopeSelected` so whatever
owns the tree selection can promote it.

Actions are derived from the SAME "deepest level" computation that drives the
detail text (see `_resolve_levels`/`_Level`), not from a second, independent
notion of "which action set is showing" -- so it is structurally impossible
for the buttons on screen to belong to a different level than the detail
above them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Static

from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .watchlist_tree import TreeScope


class PreviewRequested(Message):
    """Posted when the user requests a preview of the selected entity."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class CheckNowRequested(Message):
    """Posted when the user requests an immediate check of the selected source."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class StageInConsoleRequested(Message):
    """Posted when the user requests staging the selected entity in Console."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class DeleteRequested(Message):
    """Posted when the user requests deletion of the selected entity."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class MarkReviewedRequested(Message):
    """Posted when the user marks a watchlist item as reviewed."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class IngestRequested(Message):
    """Posted when the user ingests a watchlist item."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class IgnoreRequested(Message):
    """Posted when the user ignores a watchlist item."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class EditRuleRequested(Message):
    """Posted when the user requests editing an alert rule."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class BreadcrumbScopeSelected(Message):
    """Posted when the user clicks a shallower (collapsed) breadcrumb level.

    Carries the `TreeScope` that breadcrumb level corresponds to, so the
    screen can promote it back to the tree's current selection -- swapping
    the Inspector's detail and actions together, not just the label.
    `WatchlistsCollectionsScreen.handle_breadcrumb_scope_selected` consumes
    it and routes straight into `_apply_tree_scope`, the same reconciliation
    a real tree click uses -- promoting a breadcrumb IS navigating the tree
    to that node.
    """

    def __init__(self, scope: TreeScope) -> None:
        self.scope = scope
        super().__init__()


@dataclass(frozen=True)
class _Level:
    """One level of the breadcrumb stack: what to show, and what it means."""

    kind: str
    label: str
    entity: dict[str, Any] | None
    # Only set for scope-derived (ancestor-eligible) levels -- the level
    # appended for `selected_entity` is always the deepest, never clickable.
    target_scope: TreeScope | None


class InspectorPane(RecomposeCaptureGuard, Vertical):
    """Context-aware inspector showing a breadcrumb stack of actions."""

    selected_entity = reactive[dict[str, Any] | None](None, recompose=True)
    scope = reactive[TreeScope | None](None, recompose=True)
    breadcrumb_labels = reactive[list[str]]([], recompose=True)
    #: TASK-998. Whether this profile has nothing to select at all. Screen-
    #: seeded like the three reactives above, for the same reason: the pane
    #: has no service of its own, and "nothing is selected" and "nothing
    #: exists to select" are different states that need different copy.
    first_run = reactive(False, recompose=True)

    def compose(self):
        # No "Inspector" title here. `_build_inspector_pane` already opens the
        # RIGHT_RAIL with `Static("Inspector", classes="destination-section
        # watchlists-column-title")`, and this widget is mounted inside that
        # same region -- so emitting one here rendered the word twice in one
        # box, once left-aligned as the rail's heading and once centred as
        # this pane's. That is the doubled-heading defect task 6 exists to
        # remove; it survived task 6 because that task compared the REGION's
        # title against its content's, and both of these live inside the
        # content. Caught in a live capture of the assembled branch.
        #
        # The outer heading is the one that stays: this region holds the state
        # summaries and Console actions as well as the entity inspector, so
        # the heading belongs to the rail, not to this widget.
        levels = self._resolve_levels()
        if not levels:
            # TASK-998. "Select a source, run, item, rule, or notification"
            # is correct guidance once those things can exist, and a dead end
            # before then: on first run it names five things the user cannot
            # do, in a rail that is a third of the screen. Split on whether
            # there is anything to select rather than softening one string to
            # cover both -- the populated copy is right and stays exactly as
            # it was. The id is shared so callers testing for "the Inspector
            # has nothing selected" keep working across both.
            if self.first_run:
                yield Static(
                    "Nothing to inspect yet.",
                    id="inspector-empty-state",
                )
                yield Static(
                    "Sources, runs, items and rules show their actions here "
                    "once they exist. Start with New in the rail, then "
                    "New Source under Sources.",
                    id="inspector-first-run-hint",
                )
                return
            yield Static(
                "Select a source, run, item, rule, or notification to see actions.",
                id="inspector-empty-state",
            )
            return

        for index, level in enumerate(levels[:-1]):
            button = Button(
                level.label,
                id=f"inspector-breadcrumb-{index}",
                compact=True,
                tooltip=f"Show {level.label}.",
            )
            button.add_class("inspector-breadcrumb")
            yield button

        deepest = levels[-1]
        if deepest.entity is not None:
            entity = deepest.entity
            title = (
                entity.get("name")
                or entity.get("source_title")
                or entity.get("title")
                or "Untitled"
            )
            yield Static(Text(f"Selected: {title}"), id="inspector-entity-title")
            yield Static(Text(f"Type: {deepest.kind}"), id="inspector-entity-type")
        else:
            yield Static(Text(deepest.label), id="inspector-entity-title")
            yield Static(Text(f"Type: {deepest.kind.capitalize()}"), id="inspector-entity-type")

        with Vertical(id="inspector-actions"):
            if deepest.kind == "source":
                yield Button("Preview", id="inspector-preview-button", variant="primary")
                yield Button("Check now", id="inspector-check-now-button", variant="primary")
                yield Button("Stage in Console", id="inspector-stage-console-button")
                yield Button("Delete", id="inspector-delete-button", variant="error")
            elif deepest.kind == "run":
                yield Button("Stage in Console", id="inspector-stage-console-button")
                yield Button("Delete", id="inspector-delete-button", variant="error")
            elif deepest.kind == "item":
                yield Button("Mark reviewed", id="inspector-mark-reviewed-button", variant="primary")
                yield Button("Ingest", id="inspector-ingest-button", variant="primary")
                yield Button("Ignore", id="inspector-ignore-button", variant="error")
            elif deepest.kind == "rule":
                yield Button("Edit", id="inspector-edit-rule-button", variant="primary")
                yield Button("Delete", id="inspector-delete-button", variant="error")
            elif deepest.kind == "notification":
                yield Static("Use the inbox controls to mark read or dismiss.")
            elif deepest.kind == "watchlist":
                # Bulk actions over the watchlist's member sources. Rendered
                # DISABLED, not omitted (fix round 2, Finding 2): the ids are
                # shared with the source-level actions and post the same
                # real messages, but every consumer handler on the screen
                # early-returns silently on `entity is None` -- there is no
                # bulk-check/bulk-delete backend behind a bare watchlist
                # selection, only the per-source one. An enabled button
                # whose click produces no action, no error, and no toast is
                # worse than an absent one. The full spec table also lists
                # "Add source" and "Rename" here; those have no message
                # types at all yet and stay out of scope entirely, same as
                # before.
                yield Button(
                    "Check now",
                    id="inspector-check-now-button",
                    variant="primary",
                    disabled=True,
                    tooltip=(
                        "Unavailable: checking every source in a watchlist at "
                        "once is not implemented yet. Select one of its "
                        "sources to check it."
                    ),
                )
                yield Button(
                    "Delete",
                    id="inspector-delete-button",
                    variant="error",
                    disabled=True,
                    tooltip=(
                        "Unavailable: deleting a whole watchlist from here is "
                        "not implemented yet."
                    ),
                )
            else:
                yield Button("Delete", id="inspector-delete-button", variant="error")

    def _scope_levels(self) -> list[_Level]:
        """Ancestor levels implied by the tree `scope` alone (no entity)."""
        scope = self.scope
        if scope is None or scope.kind not in ("watchlist", "source"):
            return []
        labels = list(self.breadcrumb_labels or [])

        levels = [
            _Level(
                kind="watchlist",
                label=labels[0] if labels else f"Watchlist {scope.watchlist_id}",
                entity=None,
                target_scope=TreeScope(kind="watchlist", watchlist_id=scope.watchlist_id),
            )
        ]
        if scope.kind == "source":
            levels.append(
                _Level(
                    kind="source",
                    label=labels[1] if len(labels) > 1 else f"Source {scope.source_id}",
                    entity=None,
                    target_scope=TreeScope(
                        kind="source",
                        watchlist_id=scope.watchlist_id,
                        source_id=scope.source_id,
                    ),
                )
            )
        return levels

    def _resolve_levels(self) -> list[_Level]:
        """The full breadcrumb stack: scope ancestors, then the deepest level.

        `selected_entity`, when set, is always one level deeper than
        whatever `scope` describes -- a specific row picked within a pane
        (a source, run, item, rule, or notification) is more specific than
        "browsing this watchlist/source". When `selected_entity` is `None`,
        the scope's own tail node (source, or watchlist) is itself the
        deepest level, with its own action set.
        """
        levels = self._scope_levels()
        entity = self.selected_entity
        if entity is not None:
            entity_type = self._entity_type(entity)
            title = (
                entity.get("name")
                or entity.get("source_title")
                or entity.get("title")
                or "Untitled"
            )
            levels.append(
                _Level(kind=entity_type, label=str(title), entity=entity, target_scope=None)
            )
        return levels

    @staticmethod
    def _entity_type(entity: dict[str, Any]) -> str:
        # Alert rules are most specifically identified by their backend kind or
        # rule id; check them first so they are never mistaken for items.
        if entity.get("entity_kind") == "watchlist_alert_rule" or "rule_id" in entity:
            return "rule"
        if entity.get("entity_kind") == "client_notification":
            return "notification"
        if "source_type" in entity or "url" in entity:
            return "source"
        if "status" in entity and ("found_count" in entity or "processed_count" in entity):
            return "run"
        if "condition_type" in entity:
            return "rule"
        if "item_id" in entity or "source_name" in entity:
            return "item"
        return "unknown"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)

        if button_id.startswith("inspector-breadcrumb-"):
            event.stop()
            try:
                index = int(button_id[len("inspector-breadcrumb-"):])
            except ValueError:
                return
            levels = self._resolve_levels()
            # The last level is always the deepest (not a clickable
            # ancestor); only an in-range ancestor index promotes anything.
            if 0 <= index < len(levels) - 1:
                target = levels[index].target_scope
                if target is not None:
                    self.post_message(BreadcrumbScopeSelected(target))
            return

        entity = self.selected_entity
        if button_id == "inspector-preview-button":
            self.post_message(PreviewRequested(entity))
        elif button_id == "inspector-check-now-button":
            self.post_message(CheckNowRequested(entity))
        elif button_id == "inspector-stage-console-button":
            self.post_message(StageInConsoleRequested(entity))
        elif button_id == "inspector-delete-button":
            self.post_message(DeleteRequested(entity))
        elif button_id == "inspector-mark-reviewed-button":
            self.post_message(MarkReviewedRequested(entity))
        elif button_id == "inspector-ingest-button":
            self.post_message(IngestRequested(entity))
        elif button_id == "inspector-ignore-button":
            self.post_message(IgnoreRequested(entity))
        elif button_id == "inspector-edit-rule-button":
            self.post_message(EditRuleRequested(entity))
        event.stop()
