"""Chat settings' Model view: core first, one field-row grammar (TASK-33006.1).

Spec §6 gives every editor one row: Label (18) | a one-row control sized to
its value type | one Source word | one help line. Labels and help lines come
from the shared field table (``MODEL_CONFIG_FIELDS``). Source words come from
the resolver Switch model uses (``resolve_console_value_layers`` mapped
through ``CONSOLE_VALUE_SOURCE_WORDS``), so the modal keeps no mapping of its
own. A blank field says what a blank sends instead of showing a placeholder.

The view opens core-first: the MODEL row, then Temperature, Max tokens,
Streaming and the reasoning or thinking controls, then the Sampling,
Connection, Request estimate and name disclosures. Focus opens on Temperature,
or on the recovery action while a connection blocker stands; a restored focus
target wins over both (``_restore_suspended_scroll_and_focus``), and a missing
or unavailable one still falls back to Connection (TASK-30012).

The logic lives here, not in ``console_settings_modal.py``, because that
module sits at its ADR-097 size ceiling; the modal keeps only the wiring.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from textual import events
from textual.containers import Horizontal
from textual.css.query import NoMatches, QueryError
from textual.widget import Widget
from textual.widgets import Collapsible, Input, Select, Static

from tldw_chatbook.Chat.console_provider_support import (
    MODEL_CONFIG_FIELDS,
    MODEL_FIELD_LABELS,
    console_generation_control_support,
    supported_generation_fields,
)
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_VALUE_SOURCE_WORDS,
    resolve_console_value_layers,
)
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.Chat.provider_readiness import provider_config_key

from .console_settings_summary import build_console_readiness_presentation

#: The CORE rows, in order (spec §7 mock (b)). Unsupported reasoning or
#: thinking rows are hidden by the modal's support sync.
CORE_FIELDS = (
    "temperature",
    "max_tokens",
    "streaming",
    "reasoning_effort",
    "reasoning_summary",
    "verbosity",
    "thinking_effort",
    "thinking_budget_tokens",
)
#: The rows behind the one-row Sampling disclosure.
SAMPLING_FIELDS = (
    "top_p",
    "min_p",
    "top_k",
    "seed",
    "presence_penalty",
    "frequency_penalty",
)
FIELD_ROW_FIELDS = CORE_FIELDS + SAMPLING_FIELDS
SAMPLING_DISCLOSURE_ID = "console-settings-sampling"
CONNECTION_DISCLOSURE_ID = "console-settings-connection-disclosure"
MODEL_ROW_LABEL = "Model"
#: Streaming at chat scope is a plain On/Off choice (ADR-095:75-81); the
#: Source word, not a third option, says when it is inherited.
STREAMING_OPTIONS = (("On", "on"), ("Off", "off"))
#: What a blank optional field sends: nothing, so the provider's own default.
BLANK_FIELD_HELP = "blank = provider default"
#: A control whose support is unknown stays visible and says so in its help
#: line (TASK-30012 AC#3).
GENERATION_CONTROL_UNKNOWN_COPY = "Support not verified for this model."
#: Ends the Sampling line that names the hidden fields (TASK-33006.2).
HIDDEN_FIELDS_REASON = "(this provider does not accept them)"
SAMPLING_TITLE = "Sampling"
_CHOICE_FIELDS = frozenset(
    {"reasoning_effort", "reasoning_summary", "verbosity", "thinking_effort"}
)
#: Hidden only on an authoritative "unsupported"; "unknown" stays visible.
_SUPPORT_CONTROL_FIELDS = _CHOICE_FIELDS | {"thinking_budget_tokens"}
#: Apply refuses these blank (``_required_sampling_errors``).
_REQUIRED_FIELDS = frozenset({"temperature", "top_p"})
#: Recovery actions that are not a connection blocker: tuning opens first.
_TUNING_RECOVERY_ACTIONS = frozenset({None, "wait_for_active_run"})
#: The focusable control each connection recovery action lands on.
_RECOVERY_FOCUS = {
    "configure_credential": "#console-settings-configure-credential",
    "configure_endpoint": "#console-settings-base-url",
    "save_endpoint": "#console-settings-base-url",
}


def field_control_id(name: str) -> str:
    """Return the widget id of one field's control.

    Args:
        name: A field-table name, e.g. ``"max_tokens"``.

    Returns:
        The id, e.g. ``"console-settings-max-tokens"``.
    """
    return "console-settings-" + name.replace("_", "-")


SAMPLING_FOCUS_IDS = frozenset(field_control_id(name) for name in SAMPLING_FIELDS)
#: Restorable focus targets that live inside the Connection disclosure.
CONNECTION_FOCUS_IDS = frozenset(
    {
        "console-settings-provider",
        "console-settings-provider-picker",
        "console-settings-model-picker",
        "console-settings-model-custom",
        "console-settings-base-url",
    }
)


def hidden_fields_line(provider_name: str, hidden: Iterable[str]) -> str:
    """Return the Sampling title, naming the fields the provider rejects.

    Args:
        provider_name: The provider's display name.
        hidden: Field-table names of the hidden fields, in line order.

    Returns:
        ``"Sampling"`` when nothing is hidden, else for example
        ``"Sampling · hidden for Anthropic: Min P, Seed (this provider does
        not accept them)"``, using the field table's labels.
    """
    names = ", ".join(MODEL_FIELD_LABELS[name] for name in hidden)
    if not names:
        return SAMPLING_TITLE
    return (
        f"{SAMPLING_TITLE} · hidden for {provider_name}: {names} {HIDDEN_FIELDS_REASON}"
    )


def connection_blocked(readiness: Any) -> bool:
    """Whether a connection blocker stands, so Connection opens expanded.

    Args:
        readiness: A ``ConsoleSettingsReadiness``.

    Returns:
        True unless the draft is ready or only waits for an active run.
    """
    return readiness.recovery_action not in _TUNING_RECOVERY_ACTIONS


def _show(widget: Static, text: str) -> None:
    """Update a one-row Static only when its text changes (no relayout)."""
    if str(widget.content) != text:
        widget.update(text)


class ConsoleSettingsFieldRowsMixin:
    """The Model view's field rows, MODEL row and open focus.

    It uses the modal's draft builders and controls directly. Named ``on_*``
    handlers compose with the modal's own (Textual walks the MRO); a plain
    mixin cannot carry ``@on`` handlers.
    """

    _field_source_cache: tuple[tuple[object, ...], dict[str, str]] | None = None
    _unknown_support_fields: frozenset[str] = frozenset()

    def _field_row(self, name: str) -> Horizontal:
        """Build one Model view field row: label, control, Source word, help.

        Args:
            name: A field-table name from ``FIELD_ROW_FIELDS``.

        Returns:
            The row, id ``<control id>-row``.
        """
        from .console_settings_modal import ConsoleSettingsInput

        control_id = field_control_id(name)
        value = getattr(self._settings, name)
        # An obsolete restored choice's recovery copy sits beside its Select;
        # it is hidden while empty, so the row grammar holds.
        validation: list[Widget] = []
        if name == "streaming":
            control: Widget = Select(
                STREAMING_OPTIONS,
                value=self._streaming_select_value(),
                allow_blank=False,
                id=control_id,
                classes="console-settings-control",
            )
        elif name in _CHOICE_FIELDS:
            control = self._generation_choice_select(control_id, value)
            validation.append(self._generation_choice_validation(control_id))
        else:
            control = ConsoleSettingsInput(
                value=self._format_value(value),
                id=control_id,
                classes="console-settings-control",
            )
        return Horizontal(
            Static(MODEL_FIELD_LABELS[name], classes="console-settings-field-label"),
            control,
            *validation,
            Static(
                "",
                id=f"{control_id}-source",
                classes="console-settings-field-source",
                markup=False,
            ),
            Static(
                MODEL_CONFIG_FIELDS[name].help,
                id=f"{control_id}-help",
                classes="console-settings-help-line",
                markup=False,
            ),
            id=f"{control_id}-row",
            classes="console-settings-modal-row console-settings-field-row",
        )

    def _model_row(self) -> Horizontal:
        """Build the MODEL row: the draft's model, provider and readiness.

        Returns:
            The row, refreshed by ``_sync_model_row``.
        """
        return Horizontal(
            Static(MODEL_ROW_LABEL, classes="console-settings-field-label"),
            Static(
                "",
                id="console-settings-model-summary",
                classes="console-settings-control",
                markup=False,
            ),
            id="console-settings-model-row",
            classes="console-settings-modal-row",
        )

    def _sync_model_row(self, readiness: Any) -> None:
        """Show the draft's model, provider and readiness word.

        Args:
            readiness: The draft's ``ConsoleSettingsReadiness``.
        """
        try:
            summary = self.query_one("#console-settings-model-summary", Static)
        except (NoMatches, QueryError):
            return
        model = self._current_model_value() or "no model"
        word = build_console_readiness_presentation(readiness).primary_label
        _show(
            summary,
            f"{model} · {provider_display_name(self._active_provider)} · {word}",
        )

    def _streaming_select_value(self) -> str:
        """Return the Streaming Select value: the effective On or Off."""
        return "on" if self._effective_streaming_value() else "off"

    def _show_streaming_value(self) -> None:
        """Project the streaming draft into its Select without an edit echo."""
        select = self.query_one("#console-settings-streaming", Select)
        with self.prevent(Select.Changed):
            select.value = self._streaming_select_value()

    def _field_help(self, name: str, control: Widget) -> str:
        """Return one row's help line, saying what a blank field sends.

        Args:
            name: The row's field-table name.
            control: The row's control.

        Returns:
            The field table's help, prefixed for a blank optional field; a
            blank required field names its valid range instead. A control
            whose support is unknown leads with the neutral copy.
        """
        field = MODEL_CONFIG_FIELDS[name]
        if isinstance(control, Select):
            blank = control.value is Select.NULL
        else:
            blank = isinstance(control, Input) and not control.value.strip()
        if not blank:
            text = field.help
        elif name in _REQUIRED_FIELDS:
            text = f"Required: {field.valid_range}."
        else:
            text = f"{BLANK_FIELD_HELP} · {field.help}"
        if name in self._unknown_support_fields:
            return f"{GENERATION_CONTROL_UNKNOWN_COPY} {text}"
        return text

    def _sync_generation_control_support(self) -> None:
        """Hide the fields the draft's provider does not accept (spec rule 2).

        Samplers follow the shared ``supported_generation_fields``; the
        reasoning and thinking controls hide only on an authoritative
        "unsupported", and an "unknown" one stays with neutral help copy
        (TASK-30012 AC#3). The Sampling title names every hidden field, so a
        model change re-decides both at once. Hidden values are not rewritten
        here: Apply commits them blank (the controller's rebase) and the
        request never carries them.
        """
        provider = self._active_provider
        model = self._current_model_value()
        supported = supported_generation_fields(provider, model, self._app_config)
        focused = self.app.focused
        hidden: list[str] = []
        unknown: set[str] = set()
        for name in SAMPLING_FIELDS + CORE_FIELDS:
            if name in _SUPPORT_CONTROL_FIELDS:
                support = console_generation_control_support(provider, model, name)
                shown = support != "unsupported"
                if support == "unknown":
                    unknown.add(name)
            else:
                shown = name in supported
            row = self.query_one(f"#{field_control_id(name)}-row")
            if not shown:
                hidden.append(name)
                if focused is not None and row in focused.ancestors_with_self:
                    self.call_after_refresh(self._focus_highest_priority_connection)
            row.display = shown
        self._unknown_support_fields = frozenset(unknown)
        self.query_one(f"#{SAMPLING_DISCLOSURE_ID}", Collapsible).title = (
            hidden_fields_line(provider_display_name(provider, self._app_config), hidden)
        )
        self._sync_unsaved_hint()  # re-reads each row's help line

    def _sync_field_rows(self, edited_labels: Iterable[str]) -> None:
        """Show every field row's Source word and help line (spec §6).

        The resolver runs only when the draft's pair or its edited set
        changes; a keystroke otherwise only re-reads the blank state.

        Args:
            edited_labels: Labels of the fields edited in this open, from the
                unsaved guard (``_unsaved_field_labels``).
        """
        labels = frozenset(edited_labels)
        provider = self._active_provider
        model = self._current_model_value()
        committed = self._unsaved_committed[0]
        chat_pair = (provider_config_key(committed.provider), committed.model) == (
            provider_config_key(provider),
            model,
        )
        edited = frozenset(
            name for name in FIELD_ROW_FIELDS if MODEL_FIELD_LABELS[name] in labels
        )
        key = (provider, model, edited, chat_pair)
        if self._field_source_cache is None or self._field_source_cache[0] != key:
            layers = resolve_console_value_layers(
                self._app_config,
                provider,
                model,
                FIELD_ROW_FIELDS,
                edited=edited,
                chat_settings=committed if chat_pair else None,
            )
            words = {
                name: CONSOLE_VALUE_SOURCE_WORDS[layer] for name, layer in layers.items()
            }
            self._field_source_cache = (key, words)
        words = self._field_source_cache[1]
        for row in self.query(".console-settings-field-row"):
            control_id = str(row.id).removesuffix("-row")
            name = control_id.removeprefix("console-settings-").replace("-", "_")
            _show(row.get_child_by_id(f"{control_id}-source", Static), words[name])
            help_line = row.get_child_by_id(f"{control_id}-help", Static)
            _show(help_line, self._field_help(name, row.get_child_by_id(control_id)))
            # An obsolete restored choice's recovery copy takes the help
            # line's room while it shows (it needs about 84 cells).
            help_line.display = not any(
                error.display for error in row.query(".console-settings-error")
            )

    def _focus_highest_priority_connection(self) -> None:
        """Focus where the shown view opens (R13 of the Phase 6 plan).

        The Context view opens on Budget strategy. The Model view opens on
        Temperature, or, while a connection blocker stands, on its recovery
        action inside the Connection disclosure, which opens first.
        """
        if not self.is_mounted or not self.query("#console-settings-provider"):
            return
        if self._active_view == "context":
            self._focus_context_control()
            return
        # The draft's readiness with its own test evidence, so a known
        # failure (refused, key rejected) opens on the fix too.
        action = self._readiness_for_current_draft(self._build_draft()).recovery_action
        if action in _TUNING_RECOVERY_ACTIONS:
            temperature = self.query_one("#console-settings-temperature", Input)
            if self._is_effectively_focusable(temperature):
                temperature.focus()
            return
        disclosure = self.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible)
        if disclosure.collapsed:
            disclosure.collapsed = False
            self.call_after_refresh(self._focus_highest_priority_connection)
            return
        selector = _RECOVERY_FOCUS.get(action)
        if selector is not None:
            control = self.query_one(selector)
            if self._is_effectively_focusable(control):
                control.focus()
                control.scroll_visible(animate=False)
                return
        if action == "select_model":
            self._focus_model_control()
            return
        self._focus_connection_fallback()

    def _focus_restored_fallback(self) -> None:
        """Focus Connection when a restored focus target is missing or unavailable.

        TASK-30012's fallback, kept as it was (TASK-33006.1 AC#11): the Model
        view shows with Connection open and a live Connection control takes
        focus, in either view. Connection's contents become focusable only
        once its opening paints, hence the deferred focus.
        """
        if self._active_view != "model":
            self._show_settings_view("model")
        disclosure = self.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible)
        if disclosure.collapsed:
            disclosure.collapsed = False
            self.call_after_refresh(self._focus_connection_fallback)
            return
        self._focus_connection_fallback()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Open a closed disclosure when focus lands inside it.

        Tab never enters one (its contents are hidden), but a restored, a
        recovery or a programmatic target can; its own title can too, and
        that keeps the disclosure closed.

        Args:
            event: The focus event bubbling from the focused descendant.
        """
        widget = event.widget
        for ancestor in widget.ancestors:
            if (
                isinstance(ancestor, Collapsible)
                and ancestor.collapsed
                and widget.parent is not ancestor
            ):
                ancestor.collapsed = False

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        """Retain the Sampling and Connection disclosure state.

        Args:
            event: The disclosure that opened.
        """
        self._remember_disclosure(event.collapsible, disclosed=True)

    def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        """Retain the Sampling and Connection disclosure state.

        Args:
            event: The disclosure that closed.
        """
        self._remember_disclosure(event.collapsible, disclosed=False)

    def _remember_disclosure(self, collapsible: Collapsible, *, disclosed: bool) -> None:
        """Record one disclosure's state for the suspended-draft snapshot.

        The snapshot keeps its ``advanced_generation`` key for the Sampling
        disclosure, so suspended drafts and state stores stay compatible.
        """
        if collapsible.id == SAMPLING_DISCLOSURE_ID:
            self._advanced_generation_disclosed = disclosed
        elif collapsible.id == CONNECTION_DISCLOSURE_ID:
            self._connection_details_disclosed = disclosed
        self.call_after_refresh(self._sync_fold_hint)
