"""Shared, presentation-only Prompt-variable collection dialog."""

from __future__ import annotations

from dataclasses import dataclass, field

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import DescendantFocus
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Static

from tldw_chatbook.Prompt_Management.prompt_variables import (
    PromptApplicationDestination,
    PromptVariableApplication,
    PromptVariablePlan,
    compile_prompt_variables,
    validate_prompt_application_guards,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


MODAL_ID = "prompt-variables-dialog"
DESTINATION_COPY_ID = "prompt-variables-destination"
SYSTEM_CHECKBOX_ID = "prompt-variables-apply-system"
SYSTEM_STATE_ID = "prompt-variables-system-state"
SYSTEM_COPY_ID = "prompt-variables-system-copy"
VARIABLES_SCROLL_ID = "prompt-variables-scroll"
STATUS_ID = "prompt-variables-status"
APPLY_BUTTON_ID = "prompt-variables-apply"
ORIGINAL_BUTTON_ID = "prompt-variables-original"
CANCEL_BUTTON_ID = "prompt-variables-cancel"
VARIABLE_ROW_CLASS = "prompt-variable-row"
VARIABLE_INPUT_CLASS = "prompt-variable-input"

SYSTEM_CHECKBOX_COPY = "Replace the current session System prompt with this System lane"


@dataclass(frozen=True, slots=True)
class PromptVariablesDialogRequest:
    """Ephemeral source lanes and guards captured before opening the dialog."""

    system_text: str | None = field(repr=False)
    user_text: str | None = field(repr=False)
    destination: PromptApplicationDestination
    target_session_id: str
    composer_fingerprint: str | None = field(repr=False)
    system_fingerprint: str | None = field(repr=False)

    def __post_init__(self) -> None:
        if self.system_text is None and self.user_text is None:
            raise ValueError("Prompt variable dialog requires at least one source lane")
        if self.system_text is not None and not isinstance(self.system_text, str):
            raise TypeError("Prompt variable dialog System source must be text")
        if self.user_text is not None and not isinstance(self.user_text, str):
            raise TypeError("Prompt variable dialog User source must be text")
        validate_prompt_application_guards(
            destination=self.destination,
            target_session_id=self.target_session_id,
            composer_fingerprint=self.composer_fingerprint,
            system_fingerprint=self.system_fingerprint,
            requires_system_fingerprint=self.system_text is not None,
        )


class PromptVariablesDialog(
    SafeModalDismissMixin, ModalScreen[PromptVariableApplication | None]
):
    """Collect shared values and return a guarded application without mutation."""

    SAFE_MODAL_CONTENT = "#prompt-variables-dialog"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]

    def __init__(self, request: PromptVariablesDialogRequest) -> None:
        super().__init__()
        self.request = request
        self._system_selected = False
        self._values: dict[str, str] = {}
        self._plan = self._compile_active_plan()
        self._last_focused_variable: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id=MODAL_ID):
            yield Static(
                "Prompt variables", classes="console-modal-header", markup=False
            )
            yield Static(
                self._destination_copy,
                id=DESTINATION_COPY_ID,
                markup=False,
            )
            if self.request.system_text is not None:
                with Horizontal(id="prompt-variables-system-authorization"):
                    yield Checkbox(
                        SYSTEM_CHECKBOX_COPY,
                        value=False,
                        id=SYSTEM_CHECKBOX_ID,
                    )
                    yield Static("Off", id=SYSTEM_STATE_ID, markup=False)
                    yield Static(
                        SYSTEM_CHECKBOX_COPY,
                        id=SYSTEM_COPY_ID,
                        markup=False,
                    )
            with VerticalScroll(id=VARIABLES_SCROLL_ID, can_focus=False):
                yield from self._variable_rows()
            yield Static(self._status_copy, id=STATUS_ID, markup=False)
            with Horizontal(id="prompt-variables-actions"):
                yield Button("Cancel", id=CANCEL_BUTTON_ID)
                yield Button(
                    "Use original placeholders",
                    id=ORIGINAL_BUTTON_ID,
                    disabled=not self._can_use_original,
                )
                yield Button(
                    "Apply",
                    id=APPLY_BUTTON_ID,
                    variant="primary",
                    disabled=not self._can_apply,
                )

    @property
    def _destination_copy(self) -> str:
        if self.request.destination == "replace_snapshot":
            return "Replace the current Console draft"
        return "Append to the current Console draft"

    @property
    def _has_active_lane(self) -> bool:
        return self._system_selected or self.request.user_text is not None

    @property
    def _can_apply(self) -> bool:
        return self._has_active_lane and self._plan.is_valid

    @property
    def _can_use_original(self) -> bool:
        return self._has_active_lane

    @property
    def _status_copy(self) -> str:
        if not self._has_active_lane:
            return "Select a lane to apply"
        if not self._plan.is_valid:
            if self._plan.issues[0].code == "name_too_long":
                return "A Prompt variable name exceeds 64 characters."
            return "This Prompt has more than 64 variables."
        return ""

    def _compile_active_plan(self) -> PromptVariablePlan:
        return compile_prompt_variables(
            system_text=(self.request.system_text if self._system_selected else None),
            user_text=self.request.user_text,
        )

    def _variable_rows(self) -> list[Vertical]:
        rows: list[Vertical] = []
        for index, variable in enumerate(self._plan.variables):
            lane_copy = " + ".join(lane.title() for lane in variable.lanes)
            rows.append(
                Vertical(
                    Static(
                        f"{variable.name} — {lane_copy}",
                        classes="prompt-variable-label",
                        markup=False,
                    ),
                    Input(
                        value=self._values.get(variable.name, ""),
                        id=f"prompt-variable-value-{index}",
                        classes=VARIABLE_INPUT_CLASS,
                    ),
                    classes=VARIABLE_ROW_CLASS,
                )
            )
        return rows

    def _capture_visible_values(self) -> None:
        inputs = list(self.query(f".{VARIABLE_INPUT_CLASS}"))
        for variable, widget in zip(self._plan.variables, inputs, strict=True):
            if isinstance(widget, Input):
                self._values[variable.name] = widget.value

    def _focused_variable_name(self) -> str | None:
        focused = self.focused
        if not isinstance(focused, Input):
            return self._last_focused_variable
        for variable, widget in zip(
            self._plan.variables,
            self.query(f".{VARIABLE_INPUT_CLASS}"),
            strict=True,
        ):
            if widget is focused:
                return variable.name
        return self._last_focused_variable

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        widget = event.widget
        if not isinstance(widget, Input) or not widget.has_class(VARIABLE_INPUT_CLASS):
            return
        for variable, value_input in zip(
            self._plan.variables,
            self.query(f".{VARIABLE_INPUT_CLASS}"),
            strict=True,
        ):
            if value_input is widget:
                self._last_focused_variable = variable.name
                return

    def _restore_variable_focus(self, variable_name: str | None) -> None:
        if variable_name is None:
            return
        for variable, widget in zip(
            self._plan.variables,
            self.query(f".{VARIABLE_INPUT_CLASS}"),
            strict=True,
        ):
            if variable.name == variable_name and isinstance(widget, Input):
                widget.focus()
                return

    def _sync_action_state(self) -> None:
        self.query_one(f"#{APPLY_BUTTON_ID}", Button).disabled = not self._can_apply
        self.query_one(
            f"#{ORIGINAL_BUTTON_ID}", Button
        ).disabled = not self._can_use_original
        self.query_one(f"#{STATUS_ID}", Static).update(self._status_copy)

    def _application(self, *, render: bool) -> PromptVariableApplication:
        self._capture_visible_values()
        apply_system = self._system_selected
        apply_user = self.request.user_text is not None
        if render:
            lanes = self._plan.render(self._values)
            system_text = lanes.system_text
            user_text = lanes.user_text
        else:
            system_text = self.request.system_text if apply_system else None
            user_text = self.request.user_text if apply_user else None
        return PromptVariableApplication(
            system_text=system_text,
            user_text=user_text,
            apply_system=apply_system,
            apply_user=apply_user,
            destination=self.request.destination,
            target_session_id=self.request.target_session_id,
            composer_fingerprint=self.request.composer_fingerprint,
            system_fingerprint=(
                self.request.system_fingerprint if apply_system else None
            ),
        )

    @on(Button.Pressed, f"#{CANCEL_BUTTON_ID}")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, f"#{APPLY_BUTTON_ID}")
    def _apply(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(self._application(render=True))

    @on(Button.Pressed, f"#{ORIGINAL_BUTTON_ID}")
    def _use_original(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(self._application(render=False))

    @on(Checkbox.Changed, f"#{SYSTEM_CHECKBOX_ID}")
    async def _system_selection_changed(self, event: Checkbox.Changed) -> None:
        focus_name = self._focused_variable_name()
        self._capture_visible_values()
        self._system_selected = event.value
        self.query_one(f"#{SYSTEM_STATE_ID}", Static).update(
            "On" if event.value else "Off"
        )
        self._plan = self._compile_active_plan()
        scroll = self.query_one(f"#{VARIABLES_SCROLL_ID}", VerticalScroll)
        await scroll.remove_children()
        await scroll.mount_all(self._variable_rows())
        self._sync_action_state()
        self._restore_variable_focus(focus_name)
        self.call_after_refresh(self._restore_variable_focus, focus_name)
