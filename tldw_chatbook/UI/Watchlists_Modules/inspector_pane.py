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

from loguru import logger
from rich.text import Text
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Static, TextArea

from ...Subscriptions.noise_defaults import (
    first_invalid_selector,
    invalid_selector_message,
)
from ...Utils.input_validation import sanitize_string
from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .overview_pane import OverviewPane
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


class ToggleBriefingQueueRequested(Message):
    """Posted when the user queues or unqueues an item for briefing.

    Spec #2 phase 1's queue-for-briefing affordance -- the input side of
    "queued AND NOT already in a briefing" selection; the read path
    (`queued_for_briefing` on the normalized item) was already wired by
    Task 1. Same shape as `SaveNoiseSelectorsRequested`: a verb-first
    message class posted from `on_button_pressed`, consumed by exactly one
    screen `@on` handler.

    `queued` is always the FLIP of what the entity held when the button was
    pressed, not a fixed "queue" action -- the same button both queues and
    unqueues, and its label states the current value.

    Args:
        item_id: The item's raw database row id (`entity["item_id"]`, an
            int for a real item). `SubscriptionsDB.set_item_briefing_queued`
            takes this directly, and the flag is local-only by design (the
            same global shape as read status, ADR-018) -- so unlike
            `SaveNoiseSelectorsRequested.source_id`, there is no
            namespaced/server form for the screen to resolve first.
        queued: The value to set the flag to (not the value it currently
            holds).
    """

    def __init__(self, item_id: Any, queued: bool) -> None:
        self.item_id = item_id
        self.queued = queued
        super().__init__()


class SaveNoiseSelectorsRequested(Message):
    """Posted when the user saves a url-family source's noise selectors.

    TASK-1362 (spec §2). This is the ONLY edit path a source has: before it,
    nothing on the Watchlists screen could change a source at all (only alert
    rules had an Edit affordance), so the loop the spec is built around --
    a noisy item's diff names what churned, the user adds one rule to silence
    it -- required deleting the source and recreating it from scratch, losing
    its history. Deliberately carries only the selector text: this is not a
    general source-edit form, and widening it to one is a separate piece of
    work.

    Args:
        source_id: The source's watchlist item id, namespaced
            (``local:subscription:5``) exactly as `DeleteRequested` and the
            other source actions carry it -- `WatchlistScopeService` resolves
            either form.
        text: The field's contents verbatim (sanitized, outer whitespace
            trimmed). Empty means empty: a user who cleared the field is
            saying "watch everything on this page".
    """

    def __init__(self, source_id: Any, text: str) -> None:
        self.source_id = source_id
        self.text = text
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
    #: TASK-998, widened by TASK-1020. What the profile behind this Inspector
    #: reports: `loading`, `empty` or `populated` (`OverviewPane.LOADING` and
    #: friends). Screen-seeded like the three reactives above, for the same
    #: reason: the pane has no service of its own, and "nothing is selected",
    #: "nothing exists to select" and "nothing has answered yet" are three
    #: different states that need three different lines. It was a bool, and
    #: `False` had to stand for both loading and populated -- so during the
    #: in-flight window the rail told a brand-new user to "Select a source,
    #: run, item, rule, or notification", naming five things that did not
    #: exist. The value is the same one the Overview region keys off, so the
    #: two regions cannot disagree.
    profile_state = reactive(OverviewPane.LOADING, recompose=True)

    #: TASK-1362 (spec §2). The source types whose checks run through
    #: `URLMonitor.check_url` -- the only ones that extract text from HTML and
    #: therefore the only ones `ignore_selectors` means anything for. A feed
    #: source's items come from the feed's own entries; nothing on that path
    #: consults a selector, so offering the control there would be a field
    #: that silently does nothing. `site` is the create form's alias for
    #: `url` (`LocalWatchlistsService._local_type_for_source_type`), accepted
    #: here because a hand-built or server-side entity can still carry it.
    _URL_FAMILY_SOURCE_TYPES = frozenset({"url", "url_list", "sitemap", "site"})

    #: Deliberately SHORTER than `SourcesPane`'s copy of the same field's
    #: label, which is the one thing about the two that cannot match (whole-
    #: branch review, Important 5). Textual's border-label renderer truncates
    #: with an ellipsis and says nothing about it: the create form's field is
    #: ~91 columns wide in the centre column and comfortably holds a 65-char
    #: label, while this one lives in the right rail -- measured at ~26 columns
    #: -- where that same label painted as "Ignore elements (CSS s…". The
    #: truncation ate the syntax note that was the entire reason the label was
    #: long, so the label states only what the field is and the syntax moves to
    #: the tooltip, which has no width budget at all.
    #:
    #: There is no `border_subtitle` here for the same reason. Two rail-width
    #: border labels is two truncations, and the help copy it carried
    #: duplicated the Save button's tooltip one row below it.
    #:
    #: Duplicated as literals rather than imported because `sources_pane`
    #: imports FROM this module (`CheckNowRequested`, `PreviewRequested`) --
    #: importing back would close an import cycle.
    _IGNORE_SELECTORS_LABEL = "Ignore (CSS)"
    #: The guidance the label no longer has room for. A tooltip is the right
    #: home for it here (and the wrong one in the create form, where the field
    #: is the thing the user is filling in): the Inspector's field is prefilled
    #: with rules that already work, so the syntax matters only to someone
    #: reaching for it.
    _IGNORE_SELECTORS_HELP = (
        "One CSS rule per line; a comma within a line groups selectors. "
        "Matching elements are stripped before this page is compared, so "
        "changes inside them are not reported. Too noisy? The item diff names "
        "what churned; add a rule here to silence it."
    )
    _IGNORE_SELECTORS_MAX_LENGTH = 4000

    @classmethod
    def _is_url_family_source(cls, entity: dict[str, Any] | None) -> bool:
        """Whether `ignore_selectors` can affect how this source is checked.

        Args:
            entity: A normalized source entity, or None.

        Returns:
            True only for url/url_list/sitemap sources.
        """
        if not entity:
            return False
        source_type = entity.get("source_type")
        if source_type is None:
            source_type = entity.get("type")
        return str(source_type or "").strip().lower() in cls._URL_FAMILY_SOURCE_TYPES

    @staticmethod
    def _ignore_selectors_text(entity: dict[str, Any]) -> str:
        """The source's current selectors as the newline text a field holds.

        `normalize_local_subscription_row` publishes them under
        `settings["ignore_selectors"]` as a **list** (and omits the key
        entirely when the column is empty), so the stored newline text has to
        be reassembled here. The bare `ignore_selectors` key is the fallback
        shape a hand-built dict uses.

        Args:
            entity: A normalized source entity.

        Returns:
            One rule per line, or "" when the source has none.
        """
        candidates = []
        settings = entity.get("settings")
        if isinstance(settings, dict):
            candidates.append(settings.get("ignore_selectors"))
        candidates.append(entity.get("ignore_selectors"))
        for stored in candidates:
            if isinstance(stored, (list, tuple)):
                joined = "\n".join(str(selector) for selector in stored)
                if joined:
                    return joined
            elif stored:
                return str(stored)
        return ""

    def _noise_selectors_editor(self, entity: dict[str, Any]):
        """The one editable field a source has, plus its Save button.

        Placed with the entity detail rather than inside `#inspector-actions`:
        the four buttons in that block are one-shot actions on the selected
        source, and interleaving a multi-line text field among them would
        either split them or put an edit control below `Delete`.
        """
        field = TextArea(
            self._ignore_selectors_text(entity),
            id="inspector-noise-selectors",
            # Same reasoning as the create form's copy of this field: one rule
            # per line is the stored format, but a rule can be wider than the
            # rail, and a horizontal scrollbar would eat one of the field's
            # two content rows.
            soft_wrap=True,
        )
        field.border_title = self._IGNORE_SELECTORS_LABEL
        # No `border_subtitle`: see `_IGNORE_SELECTORS_LABEL`. The rail is too
        # narrow for a second border label, and this text is the full guidance
        # the shortened title dropped.
        field.tooltip = self._IGNORE_SELECTORS_HELP
        yield field
        yield Button(
            "Save selectors",
            id="inspector-save-selectors-button",
            variant="success",
            compact=True,
            tooltip=(
                "Save these rules. The next check re-baselines this source "
                "instead of reporting the stripped noise as a change."
            ),
        )

    #: Spec #2 phase 1. The queue button's label states the CURRENT value, so
    #: the same control both queues and unqueues -- there is no separate
    #: "Unqueue" button to render or hide.
    _QUEUE_BRIEFING_LABEL = "Queue for briefing"
    _UNQUEUE_BRIEFING_LABEL = "Unqueue from briefing"

    @classmethod
    def _queue_briefing_button(cls, entity: dict[str, Any]) -> Button:
        """The item's queue-for-briefing toggle, labelled by its current state.

        Reads `queued_for_briefing` straight off the normalized item
        (`normalize_watchlist_item`'s read path, Task 1) -- there is no
        separate reactive to seed, so a freshly selected item always shows
        the flag it actually holds, and a screen-side patch that mutates
        this same entity dict in place is picked up the next time the
        Inspector rebuilds for it.
        """
        queued = bool(entity.get("queued_for_briefing"))
        return Button(
            cls._UNQUEUE_BRIEFING_LABEL if queued else cls._QUEUE_BRIEFING_LABEL,
            id="inspector-queue-briefing-button",
            compact=True,
            tooltip=(
                "Remove this item from the pool the next briefing draws from."
                if queued
                else "Add this item to the pool the next briefing draws from, "
                "regardless of the watchlist's coverage window."
            ),
        )

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
            if self.profile_state == OverviewPane.LOADING:
                # TASK-1020: the third state. Says nothing about what exists,
                # because nothing has answered yet.
                yield Static(
                    "Loading...",
                    id="inspector-empty-state",
                    classes="watchlists-loading-state",
                )
                return
            if self.profile_state == OverviewPane.EMPTY:
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
            # TASK-1362 (spec §2). Only for a url-family SOURCE: an item, a
            # run, a rule or a feed source has no extraction settings for
            # these rules to shape.
            if deepest.kind == "source" and self._is_url_family_source(entity):
                yield from self._noise_selectors_editor(entity)
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
                # No "Mark reviewed" button here (Task 5 fix round 1,
                # Important): `selected_entity` is set by the same
                # `ItemSelected` that now marks an item read on open
                # (`WatchlistsCollectionsScreen._mark_item_read_on_open`),
                # so by the time an item's actions could render here it has
                # already been marked "reviewed" -- this button was only
                # ever reachable on an item already at that status, i.e.
                # dead in practice. Ingest/Ignore are unrelated deliberate
                # actions and are unaffected.
                yield Button("Ingest", id="inspector-ingest-button", variant="primary")
                yield Button("Ignore", id="inspector-ignore-button", variant="error")
                yield self._queue_briefing_button(entity)
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

    #: TASK-1120. Every entity the panes hand this Inspector comes out of
    #: `watchlist_normalizers.py`, and every normalizer stamps an
    #: `entity_kind`. That field is the backend's own answer to "what is
    #: this", so it decides -- the shape heuristics below it are a fallback
    #: for dicts assembled by hand (tree scopes, fixtures) that carry no kind.
    _ENTITY_KINDS = {
        "subscription": "source",
        "watchlist_source": "source",
        "watchlist_run": "run",
        "watchlist_item": "item",
        "watchlist_alert_rule": "rule",
        "client_notification": "notification",
    }

    @staticmethod
    def _entity_type(entity: dict[str, Any]) -> str:
        """Which kind of thing this entity is, and so which actions it gets.

        Guessing from shape alone is what produced TASK-1120: the first test
        used to be `"source_type" in entity or "url" in entity`, and a
        normalized watchlist item carries BOTH -- `source_type` is the type of
        the feed it came from and `url` is the article's own link. Every
        fetched item was therefore typed `source`, and the Inspector offered
        `Preview`/`Check now` over a blog post while `Mark reviewed`, `Ingest`
        and `Ignore` were unreachable. A run had the mirror problem: its stats
        live under `stats`, not as `found_count`/`processed_count` keys, so it
        fell through every branch to `unknown`.

        Args:
            entity: A normalized watchlist entity, or a hand-built dict.

        Returns:
            One of source/run/item/rule/notification, or `unknown`.
        """
        kind = InspectorPane._ENTITY_KINDS.get(str(entity.get("entity_kind") or ""))
        if kind is not None:
            return kind
        # Fallbacks, most specific first. `item_id` now outranks the source
        # keys for the same reason the map exists at all.
        if "rule_id" in entity or "condition_type" in entity:
            return "rule"
        if "item_id" in entity or "source_name" in entity:
            return "item"
        if "run_id" in entity or (
            "status" in entity
            and ("found_count" in entity or "processed_count" in entity)
        ):
            return "run"
        if "source_type" in entity or "url" in entity:
            return "source"
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
        elif button_id == "inspector-ingest-button":
            self.post_message(IngestRequested(entity))
        elif button_id == "inspector-ignore-button":
            self.post_message(IgnoreRequested(entity))
        elif button_id == "inspector-queue-briefing-button":
            self._post_toggle_briefing_queue(entity)
        elif button_id == "inspector-edit-rule-button":
            self.post_message(EditRuleRequested(entity))
        elif button_id == "inspector-save-selectors-button":
            self._post_noise_selectors_save(entity)
        event.stop()

    def _post_noise_selectors_save(self, entity: dict[str, Any] | None) -> None:
        """Read the field and ask the screen to persist it (TASK-1362).

        The text is read off the mounted `TextArea` rather than mirrored into
        a reactive on every keystroke: unlike the create form's copy of this
        field, nothing here survives a rebuild that needs re-seeding -- a
        rebuilt Inspector re-reads the stored value from the entity itself.
        """
        if entity is None:
            # Fix round 1 (Minor 4): neither of these two paths may return
            # silently. A button that produces no write, no error and no
            # toast is indistinguishable from a broken one.
            self._report_nothing_to_save(
                "Ignore-rule save pressed with no entity selected."
            )
            return
        try:
            field = self.query_one("#inspector-noise-selectors", TextArea)
        except Exception:
            self._report_nothing_to_save(
                "Ignore-rule save pressed but #inspector-noise-selectors is "
                "not mounted."
            )
            return
        text = sanitize_string(
            field.text, max_length=self._IGNORE_SELECTORS_MAX_LENGTH
        ).strip()
        # Same refusal as the create form, same copy (see
        # `invalid_selector_message`). Blocking the save is the point: writing
        # an unparseable rule would leave the source permanently carrying a
        # line that suppresses nothing, and the extraction guard's log warning
        # is not a place a TUI user looks.
        bad_selector = first_invalid_selector(text)
        if bad_selector is not None:
            self._report_invalid_selector(bad_selector)
            return
        self.post_message(SaveNoiseSelectorsRequested(entity.get("id"), text))

    def _report_invalid_selector(self, selector: str) -> None:
        """Refuse the save and name the line, in the log and on screen."""
        logger.warning(
            f"Ignore-rule save refused: unparseable CSS selector {selector!r}."
        )
        try:
            notify = getattr(self.app, "notify", None)
        except Exception:
            notify = None
        if callable(notify):
            # markup=False -- see the note on the create form's copy of this.
            notify(
                invalid_selector_message(selector),
                severity="error",
                markup=False,
            )

    def _post_toggle_briefing_queue(self, entity: dict[str, Any] | None) -> None:
        """Read the entity's current flag and post its flip.

        Same "no silent no-op" rule as `_post_noise_selectors_save`: an item
        button is only ever rendered when `selected_entity` is an item (see
        `compose`), so `entity is None` here is a state defect rather than
        anything the user did -- but a press that produces no write, no
        error and no toast is indistinguishable from a broken button, so it
        is refused loudly rather than swallowed.
        """
        if entity is None:
            self._report_nothing_to_queue(
                "Queue-for-briefing pressed with no entity selected."
            )
            return
        item_id = entity.get("item_id")
        if item_id is None:
            self._report_nothing_to_queue(
                "Queue-for-briefing pressed for an entity carrying no item id."
            )
            return
        queued = not bool(entity.get("queued_for_briefing"))
        self.post_message(ToggleBriefingQueueRequested(item_id, queued))

    def _report_nothing_to_queue(self, reason: str) -> None:
        """Say so, in the log and on screen, when the queue toggle cannot write."""
        logger.warning(reason)
        try:
            notify = getattr(self.app, "notify", None)
        except Exception:
            notify = None
        if callable(notify):
            notify("Nothing to queue: no item is selected.", severity="warning")

    def _report_nothing_to_save(self, reason: str) -> None:
        """Say so, in the log and on screen, when Save cannot do anything."""
        logger.warning(reason)
        try:
            notify = getattr(self.app, "notify", None)
        except Exception:
            notify = None
        if callable(notify):
            notify("Nothing to save: no ignore rules field is open.", severity="warning")
