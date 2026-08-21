"""Console-native run inspector."""

from __future__ import annotations

from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard
from tldw_chatbook.Widgets.Console.console_inspector_ownership import (
    ACTION_GROUPS,
    ROW_GROUPS as _ROW_GROUPS,
    ROW_IDS as _ROW_IDS,
    InspectorOwnedContent,
    InspectorOwnershipPolicy,
    classify_inspector_content,
)

_ACTION_GROUPS = ACTION_GROUPS


class ConsoleRunInspector(RecomposeCaptureGuard, Vertical):
    """Render Console run readiness, recovery, and action affordances."""

    def __init__(
        self,
        state: ConsoleInspectorState,
        *,
        ownership_policy: InspectorOwnershipPolicy = InspectorOwnershipPolicy.STRICT,
        **kwargs: Any,
    ) -> None:
        ownership = classify_inspector_content(state, ownership_policy)
        super().__init__(**kwargs)
        self.state = state
        self.ownership_policy = ownership_policy
        self._ownership = ownership
        self._reported_unknown_fingerprints: set[tuple[str, ...]] = set()
        self._report_unowned_content(ownership)
        self.styles.height = "auto"
        self.styles.min_height = 0
        #: Count of wholesale recomposes taken by ``sync_state`` (test seam).
        self.recompose_count = 0

    def sync_state(self, state: ConsoleInspectorState) -> None:
        """Refresh the mounted inspector from a new display-state snapshot.

        TASK-259: when only row text/status changed (same rendered row ids,
        same actions, same dictionary section shape), the mounted row
        ``Static`` widgets are updated in place instead of tearing down and
        recomposing the whole inspector. Any structural change (rows added/
        removed/reordered, action or dictionary changes) still recomposes.

        Args:
            state: New inspector display-state snapshot.
        """
        if state == self.state:
            return
        previous = self.state
        previous_ownership = self._ownership
        ownership = classify_inspector_content(state, self.ownership_policy)
        self.state = state
        self._ownership = ownership
        self._report_unowned_content(ownership)
        if (
            not self.is_mounted
            or self._structural_key(previous, previous_ownership)
            != self._structural_key(state, ownership)
            or not self._apply_row_updates(previous, previous_ownership)
        ):
            self.recompose_count += 1
            self.refresh(recompose=True)
            return
        # Deferred to match the recompose path's timing: a wholesale
        # recompose lands on the NEXT refresh cycle, i.e. after any rail
        # cascade the owning screen applies later in the same sync tick.
        self.call_after_refresh(self._restore_rail_cascade_visibility)

    def _report_unowned_content(self, ownership: InspectorOwnedContent) -> None:
        """Log one privacy-safe diagnostic for each unknown fingerprint."""
        fingerprint = ownership.unknown_identifiers
        if not fingerprint or fingerprint in self._reported_unknown_fingerprints:
            return
        self._reported_unknown_fingerprints.add(fingerprint)
        logger.warning("Inspector ownership incomplete: {}", fingerprint)

    def _restore_rail_cascade_visibility(self) -> None:
        """Mirror recompose semantics for the Console rail-collapse cascade.

        A wholesale recompose replaces every child, implicitly dropping the
        forced ``display=False`` (and its ``_console_rail_prior_display``
        marker) that ``ChatScreen._sync_console_rail_descendant_visibility``
        stamps on descendants while the inspector rail is collapsed. The
        in-place update path keeps the original children, so it must restore
        the same state explicitly or rows updated while the rail is hidden
        would stay ``display=False`` after the rail reopens mid-recompose
        (and diverge from the recompose path's observable DOM).
        """
        for child in self.query("*"):
            prior_display = getattr(child, "_console_rail_prior_display", None)
            if prior_display is None:
                continue
            child.display = bool(prior_display)
            child.styles.display = "block" if prior_display else "none"
            delattr(child, "_console_rail_prior_display")

    @classmethod
    def _rendered_row_entries(
        cls,
        state: ConsoleInspectorState,
        ownership: InspectorOwnedContent | None = None,
    ) -> list[tuple[str, str, str]]:
        """Return ``(widget_id, text, status)`` for each row ``compose`` mounts.

        Mirrors the grouped-then-leftover walk in ``compose`` (including its
        duplicate-label semantics) so per-row updates target exactly the
        mounted widgets.

        Args:
            state: Inspector display-state snapshot to project.

        Returns:
            Row entries in compose order, dictionary rows last.
        """
        owned = ownership or classify_inspector_content(
            state, InspectorOwnershipPolicy.STRICT
        )
        entries: list[tuple[str, str, str]] = []
        rows_by_label = {
            row.label: (index, row)
            for owner, _heading_id, _labels in _ROW_GROUPS
            for index, row in owned.rows_for(owner)
        }
        for _owner, _heading_id, labels in _ROW_GROUPS:
            for label in labels:
                if label not in rows_by_label:
                    continue
                index, row = rows_by_label[label]
                entries.append((cls._row_id(row, index), row.text, row.status))
        for index, row in enumerate(getattr(state, "dictionary_rows", ()) or ()):
            entries.append(
                (f"console-inspector-dictionaries-row-{index}", row.text, row.status)
            )
        for index, row in enumerate(getattr(state, "world_book_rows", ()) or ()):
            entries.append(
                (f"console-inspector-worldbooks-row-{index}", row.text, row.status)
            )
        return entries

    @classmethod
    def _structural_key(
        cls,
        state: ConsoleInspectorState,
        ownership: InspectorOwnedContent | None = None,
    ) -> tuple:
        """Return a key identifying the mounted widget structure for a state.

        Two states with equal keys mount the same widget ids in the same
        order with identical action buttons, so they differ at most in row
        text/status -- safe for in-place updates.

        Args:
            state: Inspector display-state snapshot to fingerprint.

        Returns:
            Hashable structure key (row ids + action tuples).
        """

        def _action_key(action: ConsoleInspectorAction) -> tuple:
            return (
                action.widget_id,
                action.label,
                action.enabled,
                getattr(action, "disabled_reason", ""),
                getattr(action, "tooltip", ""),
                getattr(action, "classes", ""),
            )

        owned = ownership or classify_inspector_content(
            state, InspectorOwnershipPolicy.STRICT
        )
        return (
            tuple(entry[0] for entry in cls._rendered_row_entries(state, owned)),
            tuple(_action_key(action) for action in owned.known_actions),
            tuple(
                _action_key(action)
                for action in getattr(state, "dictionary_actions", ()) or ()
            ),
            tuple(
                _action_key(action)
                for action in getattr(state, "world_book_actions", ()) or ()
            ),
        )

    def _apply_row_updates(
        self,
        previous: ConsoleInspectorState,
        previous_ownership: InspectorOwnedContent,
    ) -> bool:
        """Update changed row Statics in place after a non-structural change.

        Args:
            previous: The state snapshot that produced the mounted rows.

        Returns:
            True when all changed rows were updated in place; False when a
            target widget was missing (caller falls back to recompose).
        """
        new_summary = self._status_summary()
        if new_summary != self._status_summary(previous, previous_ownership):
            try:
                summary = self.query_one(
                    "#console-inspector-run-status-summary", Static
                )
            except (NoMatches, QueryError):
                return False
            summary.update(new_summary)
        old_entries = self._rendered_row_entries(previous, previous_ownership)
        for (widget_id, text, status), (_old_id, old_text, old_status) in zip(
            self._rendered_row_entries(self.state, self._ownership), old_entries
        ):
            if text == old_text and status == old_status:
                continue
            try:
                row_widget = self.query_one(f"#{widget_id}", Static)
            except (NoMatches, QueryError):
                return False
            row_widget.update(text)
            if status != old_status:
                row_widget.remove_class(f"console-inspector-row-{old_status}")
                row_widget.add_class(f"console-inspector-row-{status}")
        return True

    @staticmethod
    def _row_id(row: ConsoleDisplayRow, index: int) -> str:
        return _ROW_IDS[row.label]

    @staticmethod
    def _button_for_action(action: ConsoleInspectorAction) -> Button:
        button = Button(
            action.label,
            id=action.widget_id,
            classes=action.classes,
            variant="primary" if action.enabled else "default",
            tooltip=action.tooltip if action.enabled else "",
        )
        button.disabled = not action.enabled
        if action.enabled:
            button.styles.height = 1
            button.styles.min_height = 1
        else:
            button.styles.display = "none"
            button.styles.width = 0
            button.styles.min_width = 0
            button.styles.height = 0
            button.styles.min_height = 0
        return button

    def _compose_action(self, action: ConsoleInspectorAction) -> ComposeResult:
        yield self._button_for_action(action)
        if not action.enabled and action.disabled_reason:
            reason = Static(
                action.disabled_reason,
                id=f"{action.widget_id}-reason",
                classes="console-inspector-disabled-reason console-hidden-control",
            )
            reason.styles.display = "none"
            reason.styles.height = 0
            reason.styles.min_height = 0
            yield reason

    def _status_summary(
        self,
        state: ConsoleInspectorState | None = None,
        ownership: InspectorOwnedContent | None = None,
    ) -> str:
        """Return the primary run-inspector state in one scannable row.

        Args:
            state: Snapshot to summarize; defaults to the current state.
        """
        owned = ownership or self._ownership
        if owned.incomplete:
            return "Status: Inspector data incomplete"
        rows = {row.label: row for row in (state or self.state).rows}
        provider = rows.get("Provider")
        approvals = rows.get("Approvals")
        rag_source = rows.get("Sources") or rows.get("RAG/source")
        if provider is not None and provider.status == "blocked":
            return "Status: Blocked"
        if approvals is not None and approvals.status == "blocked":
            return "Status: Needs approval"
        if rag_source is not None and rag_source.status == "blocked":
            return "Status: Source blocked"
        # TASK-347: a live generation must not read "Ready" — but a mid-run
        # block / pending approval above is still the more important signal.
        if getattr(state or self.state, "run_active", False):
            return "Status: Generating…"
        return "Status: Ready"

    def compose(self) -> ComposeResult:
        yield Static(
            self._status_summary(),
            id="console-inspector-run-status-summary",
            classes="console-inspector-status-summary",
        )
        rows_by_label = {
            row.label: (index, row)
            for owner, _heading_id, _labels in _ROW_GROUPS
            for index, row in self._ownership.rows_for(owner)
        }

        for heading, heading_id, labels in _ROW_GROUPS:
            group_labels = [label for label in labels if label in rows_by_label]
            group_actions = self._ownership.actions_for(heading)
            if not group_labels and not group_actions:
                continue

            if not group_labels and not any(action.enabled for action in group_actions):
                for action in group_actions:
                    yield from self._compose_action(action)
                continue

            yield Static(
                heading,
                id=heading_id,
                classes="console-inspector-group-heading destination-section",
            )
            for label in group_labels:
                row_entry = rows_by_label[label]
                index, row = row_entry
                yield Static(
                    row.text,
                    id=self._row_id(row, index),
                    classes=f"console-inspector-row console-inspector-row-{row.status}",
                    markup=False,
                )

            for action in group_actions:
                yield from self._compose_action(action)

        dict_rows = getattr(self.state, "dictionary_rows", ())
        dict_actions = getattr(self.state, "dictionary_actions", ())
        if dict_rows or dict_actions:
            yield Static(
                "Chat Dictionaries",
                id="console-inspector-dictionaries-heading",
                classes="console-inspector-group-heading destination-section",
            )
            for index, row in enumerate(dict_rows):
                yield Static(
                    row.text,
                    id=f"console-inspector-dictionaries-row-{index}",
                    classes=f"console-inspector-row console-inspector-row-{row.status}",
                    markup=False,
                )
            for action in dict_actions:
                yield from self._compose_action(action)

        world_book_rows = getattr(self.state, "world_book_rows", ())
        world_book_actions = getattr(self.state, "world_book_actions", ())
        if world_book_rows or world_book_actions:
            yield Static(
                "World Books",
                id="console-inspector-worldbooks-heading",
                classes="console-inspector-group-heading destination-section",
            )
            for index, row in enumerate(world_book_rows):
                yield Static(
                    row.text,
                    id=f"console-inspector-worldbooks-row-{index}",
                    classes=f"console-inspector-row console-inspector-row-{row.status}",
                    markup=False,
                )
            for action in world_book_actions:
                yield from self._compose_action(action)
