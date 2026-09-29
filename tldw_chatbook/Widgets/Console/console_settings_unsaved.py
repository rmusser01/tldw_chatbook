"""Ask before a close gesture discards unapplied Chat settings edits.

TASK-33003.5 (ADR-031 task-16211): Esc, a backdrop click and Cancel never
discard an edited Chat settings draft by themselves. The modal's close guard
gains an ``unsaved`` mode, after the memory-reset and compaction modes, that
names the edited fields and offers Apply to this chat (Enter), Discard (d)
and Keep editing (Esc). An unedited draft still closes at once.

A field counts as edited when its effective value differs from its baseline:
the committed value for a field that arrived already changed (quick-surface
transfer), otherwise the value the controls showed once the modal finished its
initial sync. The second half keeps values the controls merely normalize at
mount (a blank committed endpoint shown as its configured default) from
reading as edits. A suspended draft (credential round-trip) carries its edits
only as raw control values, so its unedited values are the ones the controls
showed before those raw values were restored.

The logic lives here, not in ``console_settings_modal.py``, because that
module sits at its ADR-097 size ceiling; the modal keeps only the wiring.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from textual import events
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_context_policy import ContextPolicyError
from tldw_chatbook.Chat.console_provider_support import (
    GENERATION_FIELD_REQUEST_KEYS,
    MODEL_FIELD_LABELS,
)
from tldw_chatbook.Chat.console_roleplay_identity import (
    ChatDisplayNameError,
    normalize_chat_display_name,
)
from tldw_chatbook.Chat.console_settings_apply import ConsoleSettingsAction

#: Session-settings fields Apply commits, with the label each editor shows.
_SETTINGS_FIELDS = {
    "provider": "Provider",
    "model": "Model",
    "base_url": MODEL_FIELD_LABELS["endpoint"],
    **{name: MODEL_FIELD_LABELS[name] for name in GENERATION_FIELD_REQUEST_KEYS},
}
#: Context-policy fields, labelled as the Context and memory view shows them.
_CONTEXT_FIELDS = {
    "budget_mode": MODEL_FIELD_LABELS["conversation_budget_mode"],
    "custom_budget_tokens": "Conversation max tokens",
    "compaction_mode": MODEL_FIELD_LABELS["compaction_mode"],
    "compaction_representation": "Representation",
    "trigger_ratio": "Compact at (%)",
    "target_ratio": MODEL_FIELD_LABELS["compaction_target_ratio"],
    "summary_max_tokens": "Summary response max",
    "failure_behavior": "If compaction fails",
    "carry_forward_mode": MODEL_FIELD_LABELS["compaction_carry_forward_mode"],
}
_NAME_LABEL = "Your name in this chat"
#: Stands in for every context field while the view holds an invalid value.
_CONTEXT_INVALID_LABEL = "Context and memory"
_MISSING = object()
_INVALID = object()
_GUARD_BUTTONS = "#console-settings-close-guard Button"


def chat_settings_values(
    settings: Any, context_overrides: Any | None, display_name: object
) -> dict[str, object]:
    """Flatten what Apply to this chat commits into ``{label: value}``.

    Args:
        settings: A ``ConsoleSessionSettings``.
        context_overrides: A ``ConsoleContextPolicyOverrides``, or ``None``
            when the Context and memory view holds a value that does not
            validate.
        display_name: The chat's user display-name override.

    Returns:
        Effective values keyed by the field label the editors show.
    """
    values: dict[str, object] = {
        label: getattr(settings, name) for name, label in _SETTINGS_FIELDS.items()
    }
    if context_overrides is None:
        values[_CONTEXT_INVALID_LABEL] = _INVALID
    else:
        values.update(
            {
                label: getattr(context_overrides, name)
                for name, label in _CONTEXT_FIELDS.items()
            }
        )
    values[_NAME_LABEL] = display_name
    return values


def unsaved_baseline(
    mounted: Mapping[str, object],
    opened: Mapping[str, object],
    committed: Mapping[str, object],
) -> dict[str, object]:
    """Return the value each field counts as unedited at.

    Args:
        mounted: Values the controls showed after the initial sync.
        opened: Values the modal opened with (a transferred draft, say).
        committed: The chat's committed values.

    Returns:
        The committed value for a field that opened already changed, else the
        mounted value.
    """
    return {
        label: value if opened.get(label) == committed.get(label) else committed.get(label)
        for label, value in mounted.items()
    }


def unsaved_labels(
    current: Mapping[str, object], baseline: Mapping[str, object]
) -> tuple[str, ...]:
    """Return the labels whose current value differs from the baseline."""
    return tuple(
        label
        for label, value in current.items()
        if baseline.get(label, _MISSING) != value
    )


def esc_hint_copy(count: int) -> str:
    """Return the Esc hint for ``count`` unsaved edits (ADR-031 task-16211)."""
    return f"Esc close (asks: {count} unsaved)" if count else "Esc close"


def unsaved_prompt_copy(labels: Iterable[str], *, can_apply: bool = True) -> str:
    """Return the unsaved-edits prompt copy naming ``labels``.

    Args:
        labels: The edited fields' labels.
        can_apply: Whether Apply to this chat is available; the key line
            names only the keys that work (ADR-031 rule 4).

    Returns:
        The prompt's message: the edited fields, then the keys.
    """
    names = tuple(labels)
    noun = "edit" if len(names) == 1 else "edits"
    keys = (
        "Enter apply · d discard · Esc keep editing"
        if can_apply
        else "Apply is unavailable for this draft · d discard · Esc keep editing"
    )
    return f"{len(names)} unsaved {noun} to this chat: {', '.join(names)}.\n{keys}"


class ConsoleSettingsUnsavedGuardMixin:
    """The ``unsaved`` close-guard mode of ``ConsoleSettingsModal``.

    It uses the modal's close guard, draft builders and ``_submit`` directly;
    the modal wires it in at the one call site each gesture already reaches.
    Named ``on_*`` handlers compose with the modal's own (Textual walks the
    MRO); a plain mixin cannot carry ``@on`` handlers.
    """

    _unsaved_committed: tuple[Any, Any, object]
    _unsaved_baseline: dict[str, object] | None = None
    _unsaved_suspended_opened: dict[str, object] | None = None
    _unsaved_labels: tuple[str, ...] = ()
    _unsaved_retry: Callable[[], object] | None = None
    _unsaved_discard_approved = False

    def _current_chat_settings_values(self) -> dict[str, object]:
        try:
            overrides = self._build_context_policy_overrides()
        except (ContextPolicyError, ValueError):
            overrides = None
        name = self.query_one("#console-settings-user-display-name", Input).value
        try:
            name = normalize_chat_display_name(name, blank_means_none=True)
        except ChatDisplayNameError:
            pass
        return chat_settings_values(self._build_draft(), overrides, name)

    def _record_suspended_opened_values(self) -> None:
        """Record the controls' values before a suspended draft is restored.

        A credential round-trip's snapshot keeps the settings the first modal
        opened with and carries the edits only as raw control values, so the
        controls composed from those settings stand in for the first modal's
        unedited values.
        """
        try:
            self._unsaved_suspended_opened = self._current_chat_settings_values()
        except (NoMatches, QueryError):
            pass

    def _capture_unsaved_baseline(self) -> None:
        """Record the unedited values once the initial control sync is done."""
        settings, context_state, name = self._unsaved_committed
        opened_overrides = self._draft.context_policy_overrides
        committed = chat_settings_values(
            settings,
            opened_overrides if context_state is None else context_state.overrides,
            name,
        )
        opened = chat_settings_values(
            self._settings, opened_overrides, self._user_display_name_override
        )
        try:
            mounted = (
                self._unsaved_suspended_opened or self._current_chat_settings_values()
            )
        except (NoMatches, QueryError):
            return
        self._unsaved_baseline = unsaved_baseline(mounted, opened, committed)
        self._sync_unsaved_hint()

    def _unsaved_field_labels(self) -> tuple[str, ...]:
        if self._unsaved_baseline is None:
            return ()
        try:
            return unsaved_labels(
                self._current_chat_settings_values(), self._unsaved_baseline
            )
        except (NoMatches, QueryError):
            return ()

    def _ask_before_discarding(self, retry: Callable[[], object]) -> bool:
        """Show the unsaved prompt instead of closing, if anything is edited.

        Args:
            retry: The close to run again once the user chooses Discard.

        Returns:
            ``True`` when the prompt is shown and the caller must not close.
        """
        if self._unsaved_discard_approved:
            self._unsaved_discard_approved = False
            return False
        labels = self._unsaved_field_labels()
        if not labels:
            return False
        self._unsaved_labels = labels
        self._unsaved_retry = retry
        self._show_settings_close_guard("unsaved")
        return True

    def _sync_unsaved_prompt(self, mode: str | None) -> None:
        """Show the unsaved-mode buttons and copy only in that mode."""
        unsaved = mode == "unsaved"
        apply = self.query_one("#console-settings-close-apply", Button)
        modal_apply = self.query_one("#console-settings-save", Button)
        apply.display = unsaved
        apply.disabled = modal_apply.disabled or not modal_apply.display
        self.query_one("#console-settings-close-discard", Button).display = unsaved
        self.query_one("#console-settings-close-return", Button).label = (
            "Keep editing" if unsaved else "Return"
        )
        if unsaved:
            self.query_one("#console-settings-close-message", Static).update(
                unsaved_prompt_copy(self._unsaved_labels, can_apply=not apply.disabled)
            )

    def _unsaved_prompt_focus_selector(self) -> str:
        apply = self.query_one("#console-settings-close-apply", Button)
        return (
            "#console-settings-close-return"
            if apply.disabled
            else "#console-settings-close-apply"
        )

    def _sync_unsaved_hint(self) -> None:
        try:
            hint = self.query_one("#console-settings-esc-hint", Static)
        except (NoMatches, QueryError):
            return
        hint.update(esc_hint_copy(len(self._unsaved_field_labels())))

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Route Esc, backdrop and Cancel; in the prompt they keep editing."""
        del source
        if self._settings_close_guard_mode == "unsaved":
            self.query_one("#console-settings-close-return", Button).press()
            return
        self._request_settings_close()

    def on_key(self, event: events.Key) -> None:
        """``d`` discards, but only while the unsaved prompt has focus."""
        if (
            event.key == "d"
            and self._settings_close_guard_mode == "unsaved"
            and self.focused in self.query(_GUARD_BUTTONS)
        ):
            event.stop()
            self.query_one("#console-settings-close-discard", Button).press()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Run the prompt's Apply and Discard; re-read the Esc hint."""
        if event.button.id == "console-settings-close-discard":
            event.stop()
            retry, self._unsaved_retry = self._unsaved_retry, None
            self._hide_settings_close_guard()
            if retry is not None:
                self._unsaved_discard_approved = True
                retry()
        elif event.button.id == "console-settings-close-apply":
            event.stop()
            focus = self._settings_close_guard_focus
            self._settings_close_guard_focus = None
            self._hide_settings_close_guard()
            # The modal's own Apply path; an invalid draft keeps the modal
            # open with its validation summary.
            self._submit(ConsoleSettingsAction.APPLY_TO_CHAT)
            if not self._safe_dismiss_committed:
                self.call_after_refresh(self._restore_settings_close_focus, focus)
        self.call_after_refresh(self._sync_unsaved_hint)

    def on_input_changed(self, _event: Input.Changed) -> None:
        self.call_after_refresh(self._sync_unsaved_hint)

    def on_select_changed(self, _event: object) -> None:
        self.call_after_refresh(self._sync_unsaved_hint)
