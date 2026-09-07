"""Console quick model popover (Alt+M)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Literal, Mapping, Protocol, Sequence
from uuid import uuid4

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Grid, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.Chat.console_context_policy import (
    ContextCompactionMode,
    ContextCompactionRepresentation,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    build_console_model_options,
    build_console_provider_options,
)
from tldw_chatbook.Chat.console_settings_apply import (
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSurface,
    ConsoleSettingsSubmission,
    ConsoleSettingsTransfer,
    remember_model_draft,
)
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from .console_context_controls import (
    ConsoleContextControlState,
    build_console_context_control_state,
    format_context_tokens,
)
from tldw_chatbook.Widgets.model_search_picker import (
    ModelPickerInput,
    ModelSearchPicker,
)

CONSOLE_POPOVER_OPEN_FULL_SETTINGS = "open-full-settings"


@dataclass(frozen=True, slots=True)
class ConsoleModelPopoverResult:
    """Deprecated pre-ADR-095 result retained for downstream import stability.

    The rebuilt popover never emits this value. Callers receive a committed typed
    submission, a full-settings transfer, or ``None``.
    """

    settings: ConsoleSessionSettings
    compaction_mode: ContextCompactionMode


_CONSOLE_POPOVER_TEMPERATURE_MIN = 0.0
_CONSOLE_POPOVER_TEMPERATURE_MAX = 2.0
_FULL_SETTINGS_ACTION = "full_settings"


class DraftRebaser(Protocol):
    """Injected provider/model rebase seam owned by ConsoleChatController."""

    def __call__(
        self,
        state: ConsoleSettingsDraftState,
        *,
        provider: str,
        model: str | None,
        app_config: Mapping[str, object],
        exposed_fields: frozenset[str],
    ) -> ConsoleSettingsDraftState: ...


LiveCommitter = Callable[[ConsoleSettingsSubmission], ConsoleSettingsLiveCommit]
PopoverSubmitAction = ConsoleSettingsAction | Literal["full_settings"]


class DefaultReadinessResolver(Protocol):
    """Injected configuration-owned readiness seam."""

    def __call__(
        self,
        provider: str,
        model: str | None,
    ) -> ConsoleSettingsReadiness: ...


def _temperature_in_range(value: float) -> bool:
    """Return whether a parsed temperature is finite and within modal bounds.

    Args:
        value: Parsed temperature candidate.

    Returns:
        True if ``value`` is within ``[0.0, 2.0]``. NaN and infinite values
        always return False, since any comparison against them is False.
    """
    return (
        isfinite(value)
        and _CONSOLE_POPOVER_TEMPERATURE_MIN
        <= value
        <= _CONSOLE_POPOVER_TEMPERATURE_MAX
    )


def _widget_screen_region(widget: Widget) -> Any:
    """Return one mounted widget's region in screen coordinates."""

    return getattr(widget, "screen_region", None) or widget.region


class ConsolePopoverInput(Input):
    """Input that releases Textual Web mouse capture before action clicks."""

    def on_click(self, event: events.Click | None = None) -> None:
        self.release_mouse()
        if event is None:
            return
        recover = getattr(self.screen, "_recover_redirected_control_click", None)
        if callable(recover):
            recover(event)

    def on_blur(self) -> None:
        self.release_mouse()


class ConsoleModelPopover(
    SafeModalDismissMixin,
    ModalScreen["ConsoleSettingsCommittedSubmission | ConsoleSettingsTransfer | None"],
):
    """Quick exact-conversation settings and explicit default actions."""

    DEFAULT_CSS = """
    ConsoleModelPopover {
        align: center middle;
    }

    #console-model-popover {
        width: 60;
        max-width: 100%;
        height: 100%;
        min-height: 18;
        max-height: 32;
        border: tall $surface-lighten-1;
        background: $panel;
        padding: 1 2;
    }

    #console-model-popover-body {
        height: 1fr;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
    }

    .console-popover-field-label {
        color: $text-muted;
        margin: 1 0 0 0;
    }

    .console-popover-provenance {
        height: auto;
        color: $text-muted;
    }

    #console-popover-scope {
        height: auto;
        text-style: bold;
    }

    #console-popover-durability,
    #console-popover-defaults-target,
    #console-popover-save-model-default-copy,
    #console-popover-make-new-chat-default-copy,
    #console-popover-defaults-compaction-scope,
    #console-popover-new-chat-default-block {
        height: auto;
        color: $text-muted;
    }

    #console-popover-error {
        height: auto;
        background: $error 25%;
        color: $text-error;
        text-style: bold;
        border: round $error;
        padding: 0 1;
        margin: 1 0 0 0;
    }

    #console-popover-defaults-panel {
        height: auto;
        margin: 1 0 0 0;
    }

    .console-popover-context-row {
        height: 1;
        color: $text-muted;
    }

    #console-popover-compaction-help {
        height: auto;
        color: $text-muted;
    }

    #console-popover-footer {
        height: auto;
        background: $panel;
    }

    #console-popover-fold-hint {
        height: 1;
        color: $text-muted;
    }

    #console-popover-main-actions,
    #console-popover-default-actions {
        height: 6;
        min-height: 6;
        margin: 1 0 0 0;
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-rows: 3 3;
        grid-gutter: 0 1;
    }

    #console-popover-main-actions Button,
    #console-popover-default-actions Button {
        width: 1fr;
        min-width: 0;
    }
    """

    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#console-model-popover"

    def __init__(
        self,
        *,
        origin: ConsoleSettingsOrigin,
        app_config: Mapping[str, object],
        initial_draft: ConsoleSettingsDraftState,
        providers_models: Mapping[str, Sequence[str]],
        context_state: ConsoleContextControlState | None = None,
        scope_copy: str,
        durability_copy: str,
        draft_rebaser: DraftRebaser,
        live_committer: LiveCommitter,
        default_readiness_resolver: DefaultReadinessResolver,
        **kwargs: Any,
    ) -> None:
        """Initialize one exact-origin quick settings transaction.

        Args:
            origin: Stable session/conversation binding captured before opening.
            app_config: Configuration snapshot used only by the injected rebaser.
            initial_draft: Complete typed draft shared with full settings.
            providers_models: Mapping of provider key to its available model
                names, used to build the provider and model selects.
            context_state: Current context usage/policy presentation state.
            scope_copy: Exact conversation scope label.
            durability_copy: Exact unsaved or temporary durability label.
            draft_rebaser: Controller-owned provider/model rebase callback.
            live_committer: Synchronous exact-origin live commit callback.
            default_readiness_resolver: Controller-owned readiness for the
                configuration-backed provider/model target.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        if not isinstance(origin, ConsoleSettingsOrigin):
            raise TypeError("origin must be ConsoleSettingsOrigin")
        if not isinstance(initial_draft, ConsoleSettingsDraftState):
            raise TypeError("initial_draft must be ConsoleSettingsDraftState")
        self._origin = origin
        self._app_config = app_config
        self._draft = initial_draft
        self._providers_models = providers_models
        self._context_state = context_state or build_console_context_control_state(
            settings=initial_draft.settings,
            estimate=ConsoleSettingsContextEstimate(
                used_tokens=None,
                token_limit=None,
                label="Context: unavailable",
            ),
            overrides=initial_draft.context_policy_overrides,
        )
        self._initial_compaction_mode = (
            self._context_state.resolved_policy.policy.compaction_mode
        )
        self._initial_compaction_override = (
            initial_draft.context_policy_overrides.compaction_mode
        )
        self._compaction_mode_edited = False
        self._scope_copy = scope_copy
        self._durability_copy = durability_copy
        self._draft_rebaser = draft_rebaser
        self._live_committer = live_committer
        self._default_readiness_resolver = default_readiness_resolver
        self._streaming = bool(initial_draft.settings.streaming)
        self._temperature_mount_value = (
            ""
            if initial_draft.settings.temperature is None
            else str(initial_draft.settings.temperature)
        )
        self._temperature_mount_echo_pending = True
        self._active_view: Literal["main", "defaults"] = "main"
        self._main_scroll_y = 0.0
        self._updating_controls = False
        self._submit_pending = False
        self._carried_from: dict[str, tuple[str, str | None]] = {}
        # TASK-364: the provider Select fires a mount-time Select.Changed for its
        # initial value; without this tracker `_provider_changed` would rebuild
        # the model options with current_model=None and wipe the prefilled model.
        # Only a REAL provider change (value differs from what the model options
        # currently reflect) should reset the model.
        self._model_options_provider = initial_draft.settings.provider

    def _provider_select_options(self) -> list[tuple[str, str]]:
        """Provider options labeled with the shared catalog display names.

        TASK-364: mirror ``ConsoleSettingsModal._provider_select_options`` so the
        quick popover shows the same names as the full modal (``llama.cpp``, not
        the raw ``llama_cpp`` key).
        """
        return [
            (provider_display_name(option.value), option.value)
            for option in build_console_provider_options(self._providers_models)
        ]

    def _field_draft(self, name: str) -> ConsoleSettingsFieldDraft | None:
        return next(
            (field for field in self._draft.field_drafts if field.name == name),
            None,
        )

    def _field_provenance_copy(self, name: str) -> str:
        field = self._field_draft(name)
        if field is None or not field.dirty:
            return "Inherited"
        if field.provenance is ConsoleSettingsFieldProvenance.CARRIED:
            source = self._carried_from.get(name) or self._keyed_carried_source(name)
            if source is not None:
                provider, model = source
                return f"Edited — carried from {provider}/{model or 'No model'}"
        return "Edited"

    def _keyed_carried_source(self, name: str) -> tuple[str, str | None] | None:
        """Return one unambiguous explicit source from remembered keyed drafts."""

        target = (self._draft.settings.provider, self._draft.settings.model)
        sources: list[tuple[str, str | None]] = []
        for remembered in self._draft.model_drafts:
            if (remembered.provider, remembered.model) == target:
                continue
            remembered_field = next(
                (
                    candidate
                    for candidate in remembered.field_drafts
                    if candidate.name == name
                    and candidate.dirty
                    and candidate.provenance is ConsoleSettingsFieldProvenance.EXPLICIT
                ),
                None,
            )
            if remembered_field is not None:
                sources.append((remembered.provider, remembered.model))
        return sources[0] if len(sources) == 1 else None

    def _target_label(self) -> str:
        settings = self._draft.settings
        return f"{settings.provider}/{settings.model or 'No model'}"

    def _default_target_copy(self) -> str:
        return f"Defaults target: {self._target_label()}"

    def _save_model_default_copy(self) -> str:
        return (
            "Remember Temperature + Streaming for "
            f"{self._target_label()}. New-chat provider/model unchanged."
        )

    def _make_new_chat_default_copy(self) -> str:
        return (
            "Save this model profile and start eligible new chats with "
            f"{self._target_label()}."
        )

    def compose(self) -> ComposeResult:
        """Build the provider, model, temperature, and streaming controls."""
        settings = self._draft.settings
        provider_options = self._provider_select_options()
        model_options = [
            (option.label, option.value)
            for option in build_console_model_options(
                settings.provider, self._providers_models, settings.model
            )
        ]
        with Vertical(id="console-model-popover"):
            with VerticalScroll(id="console-model-popover-body"):
                yield Static("Conversation settings", classes="console-modal-header")
                yield Static(self._scope_copy, id="console-popover-scope", markup=False)
                yield Static(
                    self._durability_copy,
                    id="console-popover-durability",
                    markup=False,
                )
                error = Static("", id="console-popover-error", markup=False)
                error.display = False
                yield error
                yield Static("Provider", classes="console-popover-field-label")
                yield Select(
                    provider_options,
                    value=settings.provider,
                    id="console-popover-provider",
                )
                yield Static("Model", classes="console-popover-field-label")
                model_select = Select(
                    model_options,
                    # Select.NULL, not Select.BLANK: on this Textual version
                    # BLANK doesn't exist on Select and silently resolves to
                    # Widget.BLANK (False), an illegal value that crashes the
                    # Select at mount (TASK-16502).
                    value=(settings.model if settings.model else Select.NULL),
                    id="console-popover-model",
                    allow_blank=True,
                )
                model_select.display = False
                yield model_select
                yield ModelSearchPicker(
                    id="console-popover-model-search",
                    provider_select_id="#console-popover-provider",
                    current_model=settings.model,
                    providers_models=self._providers_models,
                )
                yield Static("Temperature", classes="console-popover-field-label")
                yield ConsolePopoverInput(
                    value=(
                        ""
                        if settings.temperature is None
                        else str(settings.temperature)
                    ),
                    placeholder="Temperature",
                    id="console-popover-temperature",
                )
                yield Static(
                    self._field_provenance_copy("temperature"),
                    id="console-popover-temperature-provenance",
                    classes="console-popover-provenance",
                    markup=False,
                )
                yield Button(
                    f"Streaming: {'on' if self._streaming else 'off'}",
                    id="console-popover-streaming",
                    compact=True,
                )
                yield Static(
                    self._field_provenance_copy("streaming"),
                    id="console-popover-streaming-provenance",
                    classes="console-popover-provenance",
                    markup=False,
                )
                yield Static(
                    "Response max  "
                    f"{format_context_tokens(settings.max_tokens)} tokens for the next reply",
                    id="console-popover-response-max",
                    classes="console-popover-context-row",
                    markup=False,
                )
                yield Static(
                    f"Request       {self._context_state.request_row}",
                    id="console-popover-request-usage",
                    classes="console-popover-context-row",
                    markup=False,
                )
                yield Static(
                    f"Conversation  {self._context_state.conversation_row}",
                    id="console-popover-conversation-usage",
                    classes="console-popover-context-row",
                    markup=False,
                )
                # TASK-26019: named category rows from the LAST prepared
                # request's own accounting -- present only after a send has
                # been prepared, and viewing never triggers a model call.
                if self._context_state.breakdown_rows:
                    yield Static(
                        "Last request by category:",
                        id="console-popover-breakdown-header",
                        classes="console-popover-context-row",
                        markup=False,
                    )
                    for row_index, breakdown_row in enumerate(
                        self._context_state.breakdown_rows
                    ):
                        hint = f"  — {breakdown_row.hint}" if breakdown_row.hint else ""
                        yield Static(
                            f"  {breakdown_row.label}: "
                            f"{format_context_tokens(breakdown_row.tokens)}{hint}",
                            id=f"console-popover-breakdown-{row_index}",
                            classes="console-popover-context-row",
                            markup=False,
                        )
                yield Static(
                    "Compaction    at "
                    f"{format_context_tokens(self._context_state.compaction_trigger_tokens)} tokens",
                    id="console-popover-compaction-threshold",
                    classes="console-popover-context-row",
                    markup=False,
                )
                yield Static(
                    self._compaction_help_text(),
                    id="console-popover-compaction-help",
                    markup=False,
                )
                yield Select(
                    [
                        ("Ask", ContextCompactionMode.ASK.value),
                        ("Automatic", ContextCompactionMode.AUTOMATIC.value),
                        ("Off", ContextCompactionMode.OFF.value),
                    ],
                    value=self._context_state.resolved_policy.policy.compaction_mode.value,
                    id="console-popover-compaction-mode",
                    disabled=self._context_state.busy,
                )
                defaults_panel = Vertical(id="console-popover-defaults-panel")
                defaults_panel.display = False
                with defaults_panel:
                    yield Static(
                        self._default_target_copy(),
                        id="console-popover-defaults-target",
                        markup=False,
                    )
                    yield Static(
                        self._save_model_default_copy(),
                        id="console-popover-save-model-default-copy",
                        markup=False,
                    )
                    yield Static(
                        self._make_new_chat_default_copy(),
                        id="console-popover-make-new-chat-default-copy",
                        markup=False,
                    )
                    yield Static(
                        "Compaction stays with this chat.",
                        id="console-popover-defaults-compaction-scope",
                        markup=False,
                    )
                    blocked = Static(
                        "",
                        id="console-popover-new-chat-default-block",
                        markup=False,
                    )
                    blocked.display = False
                    yield blocked
            with Vertical(id="console-popover-footer"):
                fold_hint = Static(
                    "▼ more — scroll for conversation settings",
                    id="console-popover-fold-hint",
                    markup=False,
                )
                fold_hint.display = False
                yield fold_hint
                with Grid(id="console-popover-main-actions"):
                    yield Button(
                        "Cancel",
                        id="console-popover-cancel",
                        compact=True,
                    )
                    yield Button(
                        "Full settings…",
                        id="console-popover-full-settings",
                        compact=True,
                    )
                    yield Button(
                        "Defaults…",
                        id="console-popover-defaults",
                        compact=True,
                    )
                    yield Button(
                        "Apply to this chat",
                        id="console-popover-apply",
                        variant="primary",
                        compact=True,
                    )
                default_actions = Grid(id="console-popover-default-actions")
                default_actions.display = False
                with default_actions:
                    yield Button(
                        "Save as model default",
                        id="console-popover-save-model-default",
                        compact=True,
                    )
                    yield Button(
                        "Make default for new chats",
                        id="console-popover-make-new-chat-default",
                        variant="primary",
                        compact=True,
                    )
                    yield Button(
                        "Back",
                        id="console-popover-defaults-back",
                        compact=True,
                    )

    def _compaction_help_text(self) -> str:
        representation = (
            self._context_state.resolved_policy.policy.compaction_representation
        )
        if representation is ContextCompactionRepresentation.VISUAL_TRANSCRIPT:
            return "Renders older turns on-device; no summary model call."
        if representation is ContextCompactionRepresentation.HYBRID:
            return "Adds local pages to a text summary; Automatic adds one model call."
        return "Summarizes older turns. Automatic may add one extra model call."

    def on_mount(self) -> None:
        """Settle the narrow-height fold affordance after first layout."""
        self._sync_default_content()
        self.call_after_refresh(self._sync_fold_hint)

    def on_resize(self, _event: events.Resize) -> None:
        """Recompute the fold affordance when the terminal size changes."""
        self.call_after_refresh(self._sync_fold_hint)

    def _sync_fold_hint(self) -> None:
        """Expose hidden quick settings while keeping the actions pinned."""
        if not self.is_mounted:
            return
        try:
            body = self.query_one("#console-model-popover-body", VerticalScroll)
            hint = self.query_one("#console-popover-fold-hint", Static)
        except NoMatches:
            return
        hint.display = body.virtual_size.height > body.container_size.height

    def _set_error(self, message: str, *, focus: Widget | None = None) -> None:
        try:
            error = self.query_one("#console-popover-error", Static)
        except NoMatches:
            return
        error.update(message)
        error.display = bool(message)
        if focus is not None:
            focus.focus()
            focus.scroll_visible(animate=False)

    @staticmethod
    def _parse_temperature(raw: str) -> float | None:
        text = raw.strip()
        if not text or not validate_text_input(text, max_length=32):
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if _temperature_in_range(value) else None

    def _replace_quick_field(
        self,
        name: str,
        value: object,
        *,
        direct_edit: bool,
    ) -> None:
        fields = list(self._draft.field_drafts)
        existing_index = next(
            (index for index, field in enumerate(fields) if field.name == name),
            None,
        )
        field = ConsoleSettingsFieldDraft(
            name=name,
            effective_value=value,
            profile_override=value,
            provenance=(
                ConsoleSettingsFieldProvenance.EXPLICIT
                if direct_edit
                else ConsoleSettingsFieldProvenance.INHERITED
            ),
            dirty=direct_edit,
        )
        if existing_index is None:
            fields.append(field)
        else:
            existing = fields[existing_index]
            field = replace(
                existing,
                effective_value=value,
                profile_override=value,
                provenance=(
                    ConsoleSettingsFieldProvenance.EXPLICIT
                    if direct_edit
                    else existing.provenance
                ),
                dirty=direct_edit or existing.dirty,
            )
            fields[existing_index] = field
        if direct_edit:
            self._carried_from.pop(name, None)
        self._draft = replace(
            self._draft,
            settings=replace(self._draft.settings, **{name: value}),
            field_drafts=tuple(fields),
        )

    @on(Input.Changed, "#console-popover-temperature")
    def _temperature_changed(self, event: Input.Changed) -> None:
        if self._updating_controls:
            return
        if self._temperature_mount_echo_pending:
            self._temperature_mount_echo_pending = False
            if event.value == self._temperature_mount_value:
                return
        value = self._parse_temperature(event.value)
        if value is not None:
            self._replace_quick_field("temperature", value, direct_edit=True)
        else:
            field = self._field_draft("temperature")
            if field is not None and not field.dirty:
                fields = tuple(
                    replace(candidate, dirty=True)
                    if candidate.name == "temperature"
                    else candidate
                    for candidate in self._draft.field_drafts
                )
                self._draft = replace(self._draft, field_drafts=fields)
        self._sync_provenance_labels()

    def _remember_current_draft(self) -> ConsoleSettingsDraftState:
        temperature = self._parse_temperature(
            self.query_one("#console-popover-temperature", Input).value
        )
        if temperature is not None:
            self._replace_quick_field("temperature", temperature, direct_edit=False)
        try:
            mode = ContextCompactionMode(
                str(self.query_one("#console-popover-compaction-mode", Select).value)
            )
        except ValueError:
            mode = self._draft.context_policy_overrides.compaction_mode
        self._draft = replace(
            self._draft,
            settings=replace(self._draft.settings, streaming=self._streaming),
            context_policy_overrides=replace(
                self._draft.context_policy_overrides,
                compaction_mode=self._compaction_override_for(mode),
            ),
        )
        self._draft = remember_model_draft(self._draft)
        return self._draft

    def _compaction_override_for(
        self, mode: ContextCompactionMode | None
    ) -> ContextCompactionMode | None:
        """Keep an untouched effective mode sparse without erasing edits."""
        if not self._compaction_mode_edited and mode == self._initial_compaction_mode:
            return self._initial_compaction_override
        return mode

    def _rebase_to(
        self,
        provider: str,
        model: str | None,
        *,
        preserve_custom_model_input: bool = False,
    ) -> None:
        source = (self._draft.settings.provider, self._draft.settings.model)
        if source == (provider, model):
            return
        remembered = self._remember_current_draft()
        previous_carried = dict(self._carried_from)
        rebased = self._draft_rebaser(
            remembered,
            provider=provider,
            model=model,
            app_config=self._app_config,
            exposed_fields=QUICK_MODEL_DEFAULT_FIELDS,
        )
        self._draft = rebased
        self._streaming = bool(rebased.settings.streaming)
        self._carried_from = {
            field.name: previous_carried.get(field.name, source)
            for field in rebased.field_drafts
            if field.provenance is ConsoleSettingsFieldProvenance.CARRIED
        }
        self._sync_controls_from_draft(
            source_provider=source[0],
            preserve_custom_model_input=preserve_custom_model_input,
        )

    def _sync_controls_from_draft(
        self,
        *,
        source_provider: str,
        preserve_custom_model_input: bool = False,
    ) -> None:
        if not self.is_mounted:
            return
        settings = self._draft.settings
        self._updating_controls = True
        try:
            provider_select = self.query_one("#console-popover-provider", Select)
            if provider_select.value != settings.provider:
                with provider_select.prevent(Select.Changed):
                    provider_select.value = settings.provider
            temperature = self.query_one("#console-popover-temperature", Input)
            with temperature.prevent(Input.Changed):
                temperature.value = (
                    "" if settings.temperature is None else str(settings.temperature)
                )
            self.query_one(
                "#console-popover-streaming", Button
            ).label = f"Streaming: {'on' if self._streaming else 'off'}"
            model_select = self.query_one("#console-popover-model", Select)
            options = [
                (option.label, option.value)
                for option in build_console_model_options(
                    settings.provider,
                    self._providers_models,
                    settings.model,
                )
            ]
            model_select.set_options(options)
            model_select.value = settings.model if settings.model else Select.NULL
            picker = self.query_one("#console-popover-model-search", ModelSearchPicker)
            if source_provider != settings.provider:
                picker.refresh_provider(
                    settings.provider,
                    current_model=settings.model,
                )
            elif preserve_custom_model_input and picker.custom_mode:
                pass
            else:
                picker.set_model_value(settings.model)
            self.query_one("#console-popover-response-max", Static).update(
                "Response max  "
                f"{format_context_tokens(settings.max_tokens)} tokens for the next reply"
            )
        finally:
            self._updating_controls = False
        self._model_options_provider = settings.provider
        self._sync_provenance_labels()
        self._sync_default_content()

    def _sync_provenance_labels(self) -> None:
        if not self.is_mounted:
            return
        for name in ("temperature", "streaming"):
            try:
                marker = self.query_one(f"#console-popover-{name}-provenance", Static)
            except NoMatches:
                continue
            marker.update(self._field_provenance_copy(name))

    def _sync_default_content(self) -> None:
        if not self.is_mounted:
            return
        updates = {
            "#console-popover-defaults-target": self._default_target_copy(),
            "#console-popover-save-model-default-copy": (
                self._save_model_default_copy()
            ),
            "#console-popover-make-new-chat-default-copy": (
                self._make_new_chat_default_copy()
            ),
        }
        for selector, copy in updates.items():
            try:
                self.query_one(selector, Static).update(copy)
            except NoMatches:
                pass

        settings = self._draft.settings
        readiness = self._default_readiness_resolver(
            settings.provider,
            settings.model,
        )
        if not settings.model:
            block_copy = "Unavailable: choose a model first."
        elif not readiness.native_send_supported:
            block_copy = f"Unavailable: {readiness.detail}"
        else:
            block_copy = ""
        try:
            button = self.query_one("#console-popover-make-new-chat-default", Button)
            block = self.query_one("#console-popover-new-chat-default-block", Static)
        except NoMatches:
            return
        button.disabled = bool(block_copy)
        block.update(block_copy)
        block.display = bool(block_copy)

    @on(Select.Changed, "#console-popover-provider")
    def _provider_changed(self, event: Select.Changed) -> None:
        """Refresh the model options when the provider select changes.

        Args:
            event: The provider select's change event.
        """
        event.stop()
        if self._updating_controls:
            return
        provider = str(event.value)
        # TASK-364: ignore the mount-time echo and redundant same-provider events
        # so the prefilled model survives; only a genuine provider change resets
        # the model options (a stale model from another provider must not linger).
        if provider == self._model_options_provider:
            return
        self._rebase_to(provider, None)

    @on(Select.Changed, "#console-popover-compaction-mode")
    def _compaction_mode_changed(self, event: Select.Changed) -> None:
        """Remember a deliberate edit even when the initial mode is reselected."""
        event.stop()
        if event.value != self._initial_compaction_mode.value:
            self._compaction_mode_edited = True

    @on(ModelSearchPicker.ModelSelected)
    def _model_search_selected(self, event: ModelSearchPicker.ModelSelected) -> None:
        """Rebase through the controller when a catalog model is committed."""
        event.stop()
        model_id = event.model_id.strip()
        if not model_id:
            return
        provider = str(self.query_one("#console-popover-provider", Select).value)
        self._rebase_to(provider, model_id)

    @on(ModelSearchPicker.ModelValueChanged)
    def _model_value_changed(self, event: ModelSearchPicker.ModelValueChanged) -> None:
        """Rebase custom model IDs through the same controller seam."""
        event.stop()
        if self._updating_controls:
            return
        provider = str(self.query_one("#console-popover-provider", Select).value)
        self._rebase_to(
            provider,
            event.model_id,
            preserve_custom_model_input=event.custom,
        )

    @on(ModelPickerInput.EscapePressed)
    async def _model_picker_escape_unhandled(
        self, event: ModelPickerInput.EscapePressed
    ) -> None:
        """Safely dismiss after the picker declines an unhandled Escape."""
        event.stop()
        await self.request_safe_cancel(source="model-picker")

    @on(Button.Pressed, "#console-popover-streaming")
    def _toggle_streaming(self, event: Button.Pressed) -> None:
        """Flip the local streaming toggle and relabel the button.

        Args:
            event: The streaming toggle button's press event.
        """
        event.stop()
        self._streaming = not self._streaming
        event.button.label = f"Streaming: {'on' if self._streaming else 'off'}"
        self._replace_quick_field(
            "streaming",
            self._streaming,
            direct_edit=True,
        )
        self._sync_provenance_labels()

    @on(Button.Pressed, "#console-popover-full-settings")
    def _full_settings(self, event: Button.Pressed) -> None:
        """Transfer the exact draft without applying it."""
        event.stop()
        self._submit(_FULL_SETTINGS_ACTION)

    @on(Button.Pressed, "#console-popover-defaults")
    def _show_defaults(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_view("defaults")

    @on(Button.Pressed, "#console-popover-defaults-back")
    def _back_from_defaults(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_view("main")

    def _set_view(self, view: Literal["main", "defaults"]) -> None:
        body = self.query_one("#console-model-popover-body", VerticalScroll)
        if view == "defaults":
            self._main_scroll_y = body.scroll_y
        self._active_view = view
        main = self.query_one("#console-popover-main-actions", Grid)
        defaults = self.query_one("#console-popover-default-actions", Grid)
        panel = self.query_one("#console-popover-defaults-panel", Vertical)
        main.display = view == "main"
        defaults.display = view == "defaults"
        panel.display = view == "defaults"
        self._sync_default_content()
        self.call_after_refresh(self._sync_fold_hint)
        if view == "defaults":
            self.query_one("#console-popover-save-model-default", Button).focus()
            self.call_after_refresh(self._reveal_defaults_panel)
        else:
            self.query_one("#console-popover-defaults", Button).focus()
            self.call_after_refresh(self._restore_main_scroll)

    def _reveal_defaults_panel(self) -> None:
        """Show the exact default intent before a pinned action can commit it."""

        if not self.is_mounted or self._active_view != "defaults":
            return
        try:
            panel = self.query_one("#console-popover-defaults-panel", Vertical)
        except NoMatches:
            return
        panel.scroll_visible(
            animate=False,
            immediate=True,
            force=True,
        )

    def _restore_main_scroll(self) -> None:
        """Return to the conversation controls the user left for Defaults."""

        if not self.is_mounted or self._active_view != "main":
            return
        try:
            body = self.query_one("#console-model-popover-body", VerticalScroll)
        except NoMatches:
            return
        body.scroll_to(
            y=self._main_scroll_y,
            animate=False,
            immediate=True,
            force=True,
        )

    @on(Button.Pressed, "#console-popover-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, "#console-popover-apply")
    def _apply(self, event: Button.Pressed) -> None:
        """Apply the validated draft to the exact originating conversation."""
        event.stop()
        self._submit(ConsoleSettingsAction.APPLY_TO_CHAT)

    @on(Button.Pressed, "#console-popover-save-model-default")
    def _save_model_default(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit(ConsoleSettingsAction.SAVE_MODEL_DEFAULT)

    @on(Button.Pressed, "#console-popover-make-new-chat-default")
    def _make_new_chat_default(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit(ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT)

    def _validated_draft(self) -> ConsoleSettingsDraftState | None:
        provider_select = self.query_one("#console-popover-provider", Select)
        provider = str(provider_select.value or "").strip()
        if not provider or provider == str(Select.NULL):
            self._set_error("Choose a provider.", focus=provider_select)
            return None

        picker = self.query_one("#console-popover-model-search", ModelSearchPicker)
        model = picker.value
        if model is None:
            self._set_error("Choose a model.", focus=picker)
            return None

        temperature_input = self.query_one("#console-popover-temperature", Input)
        temperature_text = temperature_input.value.strip()
        temperature = self._parse_temperature(temperature_text)
        if temperature is None and temperature_text:
            self._set_error(
                "Temperature must be a finite number from 0 to 2.",
                focus=temperature_input,
            )
            return None

        if (self._draft.settings.provider, self._draft.settings.model) != (
            provider,
            model,
        ):
            self._rebase_to(provider, model)
        self._replace_quick_field(
            "temperature",
            temperature,
            direct_edit=False,
        )
        self._draft = replace(
            self._draft,
            settings=replace(
                self._draft.settings,
                provider=provider,
                model=model,
                temperature=temperature,
                streaming=self._streaming,
            ),
            context_policy_overrides=replace(
                self._draft.context_policy_overrides,
                compaction_mode=self._compaction_override_for(
                    ContextCompactionMode(
                        str(
                            self.query_one(
                                "#console-popover-compaction-mode", Select
                            ).value
                        )
                    )
                ),
            ),
        )
        self._draft = remember_model_draft(self._draft)
        self._set_error("")
        return self._draft

    def _release_mouse_capture(self) -> None:
        captured = self.app.mouse_captured
        if captured is not None:
            captured.release_mouse()
        if self.app.mouse_captured is not None:
            self.app.capture_mouse(None)

    @staticmethod
    def _without_endpoint_intent(
        draft: ConsoleSettingsDraftState,
    ) -> ConsoleSettingsDraftState:
        """Return a quick-submission draft with no endpoint save intent."""

        return replace(
            draft,
            model_drafts=tuple(
                replace(remembered, endpoint_draft=None)
                for remembered in draft.model_drafts
            ),
            endpoint_draft=None,
        )

    def on_click(self, event: events.Click) -> None:  # type: ignore[override]
        """Recover control clicks redirected through a captured input."""
        self._recover_redirected_control_click(event)

    def _recover_redirected_control_click(self, event: events.Click) -> None:
        captured = self.app.mouse_captured
        click_origin = getattr(event, "widget", None)
        focused = self.app.focused
        screen_routed = click_origin is self and isinstance(
            focused, ConsolePopoverInput
        )
        if (
            not isinstance(captured, ConsolePopoverInput)
            and not isinstance(click_origin, ConsolePopoverInput)
            and not screen_routed
        ):
            return
        if isinstance(captured, ConsolePopoverInput):
            captured.release_mouse()
        if event.button != 1 or event.screen_x is None or event.screen_y is None:
            return
        for control in (*self.query(Select), *self.query(Button)):
            if control.disabled or not control.display:
                continue
            if _widget_screen_region(control).contains(event.screen_x, event.screen_y):
                control.focus()
                if isinstance(control, Select):
                    control.action_show_overlay()
                else:
                    control.press()
                event.stop()
                return

    def _submit(self, action: PopoverSubmitAction) -> None:
        if self._submit_pending or self._safe_dismiss_committed:
            return
        draft = self._validated_draft()
        if draft is None:
            return
        if action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT:
            button = self.query_one("#console-popover-make-new-chat-default", Button)
            if button.disabled:
                block = self.query_one(
                    "#console-popover-new-chat-default-block", Static
                )
                self._set_error(str(block.renderable) or "Default is unavailable.")
                return

        self._release_mouse_capture()
        if action == _FULL_SETTINGS_ACTION:
            self.dismiss_safe_once(ConsoleSettingsTransfer(self._origin, draft))
            return

        draft = self._without_endpoint_intent(draft)
        submission = ConsoleSettingsSubmission(
            submission_id=uuid4().hex,
            action=action,
            surface=ConsoleSettingsSurface.QUICK_POPOVER,
            origin=self._origin,
            draft=draft,
            user_display_name_override=None,
            default_field_mask=(
                frozenset()
                if action is ConsoleSettingsAction.APPLY_TO_CHAT
                else QUICK_MODEL_DEFAULT_FIELDS
            ),
        )
        self._submit_pending = True
        try:
            live_commit = self._live_committer(submission)
        except ValueError as error:
            message = str(error).strip().rstrip(".")
            if message == "Chat closed; nothing applied":
                self.notify("Chat closed; nothing applied", severity="warning")
                self.dismiss_safe_once(None)
                return
            self._set_error(str(error) or "Settings could not be applied.")
            self._submit_pending = False
            return
        except Exception:
            self._set_error("Settings could not be applied; nothing changed.")
            self._submit_pending = False
            return
        if not isinstance(live_commit, ConsoleSettingsLiveCommit):
            self._set_error("Settings could not be applied; nothing changed.")
            self._submit_pending = False
            return
        delivered = False
        try:
            delivered = self.dismiss_safe_once(
                ConsoleSettingsCommittedSubmission(submission, live_commit)
            )
        finally:
            if not delivered and live_commit.durability_admission is not None:
                live_commit.durability_admission.release()

    async def action_request_safe_cancel(self) -> None:
        """Leave Defaults first; cancel the popover from the main view."""
        if self._active_view == "defaults":
            self._set_view("main")
            return
        await super().action_request_safe_cancel()

    async def action_dismiss_popover(self) -> None:
        """Dismiss the popover with no result (Escape)."""
        await self.action_request_safe_cancel()
