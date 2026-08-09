"""Screen-owned retained Prompt history disclosure for the Library editor."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Collapsible, Static, TextArea

from ...Library.library_prompts_state import (
    PromptHistoryState,
    history_restore_gate,
    prompt_history_count_label,
)


class _ScopedHistoryAction(Message):
    """One immutable action captured from a specific history view model."""

    __slots__ = ("_prompt_uuid", "_scope_token")

    def __init__(self, *, prompt_uuid: str, scope_token: int) -> None:
        self._prompt_uuid = prompt_uuid
        self._scope_token = scope_token
        super().__init__()

    @property
    def prompt_uuid(self) -> str:
        return self._prompt_uuid

    @property
    def scope_token(self) -> int:
        return self._scope_token


class LibraryPromptHistoryRegion(Vertical):
    """Recompose and translate events inside the retained-history disclosure."""

    class Ready(Message):
        """A freshly mounted region needs the screen's current controller state."""

    class DisclosureOpened(_ScopedHistoryAction):
        """The retained-history disclosure was opened."""

    class DisclosureClosed(_ScopedHistoryAction):
        """The retained-history disclosure was closed."""

    class CountRetryRequested(_ScopedHistoryAction):
        """The scalar history count should be retried."""

    class PageRequested(_ScopedHistoryAction):
        """The first, retry, or next retained-history page was requested."""

    class RowSelected(_ScopedHistoryAction):
        """A retained snapshot row was selected."""

        def __init__(
            self,
            *,
            prompt_uuid: str,
            scope_token: int,
            change_id: int,
            source_version: int,
        ) -> None:
            self.change_id = change_id
            self.source_version = source_version
            super().__init__(prompt_uuid=prompt_uuid, scope_token=scope_token)

    class RestoreRequested(_ScopedHistoryAction):
        """The selected retained snapshot should enter confirmation."""

    class ReloadRequested(_ScopedHistoryAction):
        """The retained-history first page should be reset and reloaded."""

    view_model: reactive[tuple[PromptHistoryState | None, bool, bool]] = reactive(
        (None, False, True), recompose=True
    )

    def __init__(
        self,
        state: PromptHistoryState | None,
        *,
        dirty: bool,
        current_compatible: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.view_model = (state, dirty, current_compatible)

    def sync_state(
        self,
        state: PromptHistoryState | None,
        *,
        dirty: bool,
        current_compatible: bool,
    ) -> None:
        """Apply one immutable view model and recompose this region only."""
        if state is not None:
            try:
                self.query_one(
                    "#library-prompt-history-collapsible", Collapsible
                ).title = prompt_history_count_label(state)
            except NoMatches:
                pass
        self.view_model = (state, dirty, current_compatible)

    def on_mount(self) -> None:
        """Request a live sync after an outer canvas recompose mounts this region."""
        self.post_message(self.Ready())

    @staticmethod
    def _bind_action_scope(
        control: Button | Collapsible, state: PromptHistoryState
    ) -> None:
        """Bind one immutable render-generation identity to an action control."""
        control.history_action_scope = (state.prompt_uuid, state.scope_token)

    @staticmethod
    def _action_scope(control: Button | Collapsible) -> dict[str, str | int] | None:
        """Read immutable identity from the control that emitted the event."""
        scope = getattr(control, "history_action_scope", None)
        if (
            not isinstance(scope, tuple)
            or len(scope) != 2
            or not isinstance(scope[0], str)
            or type(scope[1]) is not int
        ):
            return None
        return {
            "prompt_uuid": scope[0],
            "scope_token": scope[1],
        }

    def _action_button(
        self, state: PromptHistoryState, label: str, **kwargs: Any
    ) -> Button:
        """Build one action button scoped to this compose generation."""
        button = Button(label, **kwargs)
        self._bind_action_scope(button, state)
        return button

    def compose(self) -> ComposeResult:
        state, dirty, current_compatible = self.view_model
        if state is None:
            return
        disclosure = Collapsible(
            *self._body_children(
                state,
                dirty=dirty,
                current_compatible=current_compatible,
            ),
            title=prompt_history_count_label(state),
            collapsed=not state.is_open,
            id="library-prompt-history-collapsible",
        )
        self._bind_action_scope(disclosure, state)
        yield disclosure

    def _body_children(
        self,
        state: PromptHistoryState,
        *,
        dirty: bool,
        current_compatible: bool,
    ) -> list[Static | Button | TextArea]:
        children: list[Static | Button | TextArea] = []
        if state.count_status == "error":
            children.extend(
                (
                    Static(
                        "Retained history count is unavailable.",
                        id="library-prompt-history-count-error",
                        classes="library-prompt-history-error",
                        markup=False,
                    ),
                    self._action_button(
                        state,
                        "Retry count",
                        id="library-prompt-history-retry-count",
                        classes="library-canvas-action",
                        compact=True,
                    ),
                )
            )
        if state.page_status == "loading" and not state.rows:
            children.append(
                Static(
                    "Loading retained history…",
                    id="library-prompt-history-loading",
                    classes="destination-purpose",
                    markup=False,
                )
            )
        elif state.page_status == "error":
            children.extend(
                (
                    Static(
                        state.error or "Couldn't load retained history.",
                        id="library-prompt-history-page-error",
                        classes="library-prompt-history-error",
                        markup=False,
                    ),
                    self._action_button(
                        state,
                        "Retry",
                        id="library-prompt-history-retry-page",
                        classes="library-canvas-action",
                        compact=True,
                    ),
                )
            )
        elif state.page_status == "loaded" and not state.rows:
            children.append(
                Static(
                    "No retained versions are available.",
                    id="library-prompt-history-empty",
                    classes="destination-purpose",
                    markup=False,
                )
            )

        for row in state.rows:
            artifact_label = row.artifact_type.title()
            if row.artifact_type_raw and (
                row.artifact_type_raw.casefold() != row.artifact_type.casefold()
            ):
                artifact_label = (
                    f"{artifact_label} · stored type {row.artifact_type_raw}"
                )
            label = escape_markup(
                f"v{row.version} · change {row.change_id} · "
                f"{row.timestamp} · {artifact_label}\n{row.change_summary}"
            )
            button = Button(
                label,
                id=f"library-prompt-history-row-{row.change_id}",
                classes="library-prompt-history-row",
                compact=True,
            )
            button.change_id = row.change_id
            button.source_version = row.version
            self._bind_action_scope(button, state)
            button.set_class(
                state.selected is not None
                and state.selected.change_id == row.change_id
                and state.selected.source_version == row.version,
                "history-selected",
            )
            children.append(button)

        if state.page_status == "loading" and state.rows:
            children.append(
                Static(
                    "Loading older retained versions…",
                    id="library-prompt-history-loading-older",
                    classes="destination-purpose",
                    markup=False,
                )
            )
        elif state.has_more:
            children.append(
                self._action_button(
                    state,
                    "Load older versions",
                    id="library-prompt-history-load-older",
                    classes="library-canvas-action",
                    compact=True,
                )
            )

        selection = state.selected
        if selection is not None:
            row = selection.row
            metadata = (
                f"Selected v{row.version} · change {row.change_id}\n"
                f"Name: {row.name or '—'}\nAuthor: {row.author or '—'}\n"
                f"Description: {row.details or '—'}\n"
                f"Keywords: {', '.join(row.keywords) if row.keywords else '—'}"
            )
            children.extend(
                (
                    Static(
                        metadata,
                        id="library-prompt-history-metadata",
                        classes="library-prompt-history-metadata",
                        markup=False,
                    ),
                    Static(
                        "Stored System lane",
                        classes="library-prompt-field-label",
                        markup=False,
                    ),
                    TextArea(
                        row.system_preview,
                        read_only=True,
                        id="library-prompt-history-system",
                    ),
                    Static(
                        "Stored User lane",
                        classes="library-prompt-field-label",
                        markup=False,
                    ),
                    TextArea(
                        row.user_preview,
                        read_only=True,
                        id="library-prompt-history-user",
                    ),
                    Static(
                        row.compatibility_reason
                        or "Compatible with current local Prompt capabilities.",
                        id="library-prompt-history-compatibility",
                        classes=(
                            "library-prompt-history-compatibility"
                            if row.restore_eligible
                            else "library-prompt-history-error"
                        ),
                        markup=False,
                    ),
                )
            )
            if not row.keywords_captured:
                children.append(
                    Static(
                        "Current keywords were not captured in this older retained "
                        "version; restoring it keeps the current keywords.",
                        id="library-prompt-history-keywords-disclosure",
                        classes="destination-purpose",
                        markup=False,
                    )
                )

        gate = history_restore_gate(state, dirty=dirty)
        restore_reason = gate.reason
        if not current_compatible:
            restore_reason = (
                "This compatibility-only editor cannot restore retained history."
            )
        if state.restore_request is not None:
            restore_reason = "Restoring retained version…"
        elif (
            state.restore_outcome is not None and state.restore_outcome.reload_required
        ):
            restore_reason = state.restore_outcome.message
        restore_enabled = (
            gate.enabled
            and current_compatible
            and state.restore_request is None
            and not (
                state.restore_outcome is not None
                and state.restore_outcome.reload_required
            )
        )
        if restore_enabled:
            restore_reason = "Confirmation creates a new current version."
        children.extend(
            (
                self._action_button(
                    state,
                    "Restore selected version…",
                    id="library-prompt-history-restore",
                    classes="library-canvas-action console-action-primary",
                    compact=True,
                    disabled=not restore_enabled,
                ),
                Static(
                    restore_reason,
                    id="library-prompt-history-restore-reason",
                    classes="destination-purpose",
                    markup=False,
                ),
            )
        )
        if state.restore_outcome is not None:
            if not state.restore_outcome.reload_required:
                children.append(
                    Static(
                        state.restore_outcome.message,
                        id="library-prompt-history-outcome",
                        classes=(
                            "library-prompt-history-success"
                            if state.restore_outcome.kind in {"restored", "no_change"}
                            else "library-prompt-history-error"
                        ),
                        markup=False,
                    )
                )
            if state.restore_outcome.kind == "snapshot_unavailable":
                children.append(
                    self._action_button(
                        state,
                        "Reload retained history",
                        id="library-prompt-history-reload",
                        classes="library-canvas-action",
                        compact=True,
                    )
                )
            if state.restore_outcome.keyword_disclosure:
                children.append(
                    Static(
                        state.restore_outcome.keyword_disclosure,
                        id="library-prompt-history-outcome-keywords",
                        classes="destination-purpose",
                        markup=False,
                    )
                )
        return children

    @on(Collapsible.Toggled, "#library-prompt-history-collapsible")
    def _on_disclosure_toggled(self, event: Collapsible.Toggled) -> None:
        event.stop()
        scope = self._action_scope(event.collapsible)
        if scope is None:
            return
        if event.collapsible.collapsed:
            self.post_message(self.DisclosureClosed(**scope))
        else:
            self.post_message(self.DisclosureOpened(**scope))

    @on(Button.Pressed, "#library-prompt-history-retry-count")
    def _on_count_retry(self, event: Button.Pressed) -> None:
        event.stop()
        scope = self._action_scope(event.button)
        if scope is not None:
            self.post_message(self.CountRetryRequested(**scope))

    @on(Button.Pressed, "#library-prompt-history-retry-page")
    @on(Button.Pressed, "#library-prompt-history-load-older")
    def _on_page_requested(self, event: Button.Pressed) -> None:
        event.stop()
        scope = self._action_scope(event.button)
        if scope is not None:
            self.post_message(self.PageRequested(**scope))

    @on(Button.Pressed, ".library-prompt-history-row")
    def _on_row_selected(self, event: Button.Pressed) -> None:
        event.stop()
        change_id = getattr(event.button, "change_id", None)
        source_version = getattr(event.button, "source_version", None)
        scope = self._action_scope(event.button)
        if scope is not None and type(change_id) is int and type(source_version) is int:
            self.post_message(
                self.RowSelected(
                    **scope,
                    change_id=change_id,
                    source_version=source_version,
                )
            )

    @on(Button.Pressed, "#library-prompt-history-restore")
    def _on_restore_requested(self, event: Button.Pressed) -> None:
        event.stop()
        scope = self._action_scope(event.button)
        if scope is not None:
            self.post_message(self.RestoreRequested(**scope))

    @on(Button.Pressed, "#library-prompt-history-reload")
    def _on_reload_requested(self, event: Button.Pressed) -> None:
        event.stop()
        scope = self._action_scope(event.button)
        if scope is not None:
            self.post_message(self.ReloadRequested(**scope))
