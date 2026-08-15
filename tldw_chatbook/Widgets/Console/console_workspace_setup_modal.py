"""Console new-workspace setup modal.

Task: "New Workspace" in the Console used to create a bare "Workspace N"
record with no folder binding and no hint of what it maps to. This modal
makes creation a two-field setup step: a (prefilled) name and a REQUIRED
folder binding, validated inline against the exact same rules
``LocalWorkspaceRegistryService.add_folder_binding`` enforces, so Create
only fires on a bindable path and the workspace is never created
unlabeled.

Cancel/Escape dismisses with ``None`` and creates nothing.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Checkbox, Input, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

#: Debounce for inline validation -- each check does a ``Path.resolve()``
#: plus a folder-bindings SELECT, so keystrokes must not trigger it
#: directly (mirrors the Console conversation-search debounce).
VALIDATION_DEBOUNCE_SECONDS = 0.3

MODAL_ID = "console-workspace-setup-modal"
NAME_INPUT_ID = "console-workspace-setup-name"
PATH_INPUT_ID = "console-workspace-setup-path"
WRITE_CHECKBOX_ID = "console-workspace-setup-write"
ERROR_STATIC_ID = "console-workspace-setup-error"
HINT_STATIC_ID = "console-workspace-setup-hint"
CANCEL_BTN_ID = "console-workspace-setup-cancel"
CREATE_BTN_ID = "console-workspace-setup-create"


class ConsoleWorkspaceSetupResult(NamedTuple):
    """What a confirmed setup carries back to the opener."""

    name: str
    folder_path: str
    allow_write: bool


#: Returns the human-readable reason a (name, path) pair is invalid, or
#: ``None`` when both are acceptable. Injected so tests can stub the
#: registry without a database; the production seam is built in
#: ``ConsoleWorkspaceController._open_console_workspace_setup_modal``.
SetupFieldValidator = Callable[[str, str], Optional[str]]


class ConsoleWorkspaceSetupModal(
    SafeModalDismissMixin, ModalScreen[Optional[ConsoleWorkspaceSetupResult]]
):
    """Create a named workspace bound to one agent-accessible folder."""

    DEFAULT_CSS = """
    ConsoleWorkspaceSetupModal {
        align: center middle;
    }

    #console-workspace-setup-modal {
        width: 64;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-workspace-setup-modal Input {
        width: 100%;
        margin: 0 0 1 0;
    }

    #console-workspace-setup-modal Label {
        margin: 0;
    }

    #console-workspace-setup-write {
        margin: 0 0 1 0;
    }

    #console-workspace-setup-hint {
        height: auto;
        color: grey;
        margin: 0 0 1 0;
    }

    #console-workspace-setup-error {
        height: auto;
        min-height: 1;
        color: red;
        margin: 0 0 1 0;
    }

    #console-workspace-setup-actions {
        height: 3;
        min-height: 3;
        align-horizontal: right;
    }

    #console-workspace-setup-cancel,
    #console-workspace-setup-create {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }
    """

    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = f"#{MODAL_ID}"

    def __init__(
        self,
        *,
        suggested_name: str,
        validate: SetupFieldValidator,
        debounce_seconds: float = VALIDATION_DEBOUNCE_SECONDS,
    ) -> None:
        """Initialize the setup modal.

        Args:
            suggested_name: Prefilled workspace name (the auto-generated
                ``Workspace N``); the user may replace it.
            validate: Read-only pre-check for (name, folder path).
                Returns an error string or ``None``.
            debounce_seconds: Validation debounce; overridable to a small
                positive value (never 0 -- Textual's ``set_timer`` divides
                by the interval) in tests so it fires within one pause.
        """
        super().__init__()
        self._suggested_name = suggested_name
        self._validate = validate
        self._debounce_seconds = debounce_seconds
        self._validation_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id=MODAL_ID):
            yield Static("New Workspace", classes="console-modal-header")
            yield Static("Name", classes="console-modal-label")
            yield Input(
                value=self._suggested_name,
                id=NAME_INPUT_ID,
                placeholder="Workspace name",
            )
            yield Static("Folder (agent file-tool access)", classes="console-modal-label")
            yield Input(
                value="",
                id=PATH_INPUT_ID,
                placeholder="~/projects/my-project",
            )
            yield Checkbox(
                "Read-write access (agent can modify files)",
                id=WRITE_CHECKBOX_ID,
            )
            yield Static(
                "A folder is required: the workspace is created bound to it, "
                "read-only by default.",
                id=HINT_STATIC_ID,
                markup=False,
            )
            yield Static("", id=ERROR_STATIC_ID, markup=False)
            with Horizontal(id="console-workspace-setup-actions"):
                yield Button("Cancel", id=CANCEL_BTN_ID)
                yield Button(
                    "Create",
                    id=CREATE_BTN_ID,
                    variant="primary",
                    disabled=True,
                )

    def on_mount(self) -> None:
        """Focus the name field (fully selected) after mixin setup."""
        super().on_mount()
        name_input = self.query_one(f"#{NAME_INPUT_ID}", Input)
        name_input.focus()
        name_input.select_all()
        # Validate the initial (empty-path) state so Create is disabled
        # for a reason the user can see, not just disabled-by-default.
        self.call_after_refresh(self._run_validation)

    @on(Input.Changed, f"#{NAME_INPUT_ID}")
    @on(Input.Changed, f"#{PATH_INPUT_ID}")
    def _schedule_validation(self, event: Input.Changed) -> None:
        event.stop()
        self._create_button.disabled = True
        if self._validation_timer is not None:
            self._validation_timer.stop()
        self._validation_timer = self.set_timer(
            self._debounce_seconds, self._run_validation
        )

    @on(Input.Submitted, f"#{PATH_INPUT_ID}")
    def _submit(self, event: Input.Submitted) -> None:
        event.stop()
        self._create()

    @on(Button.Pressed, f"#{CANCEL_BTN_ID}")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, f"#{CREATE_BTN_ID}")
    def _create_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._create()

    @property
    def _create_button(self) -> Button:
        return self.query_one(f"#{CREATE_BTN_ID}", Button)

    def _run_validation(self) -> None:
        self._validation_timer = None
        name = self.query_one(f"#{NAME_INPUT_ID}", Input).value.strip()
        path = self.query_one(f"#{PATH_INPUT_ID}", Input).value.strip()
        error = self._validate(name, path)
        error_static = self.query_one(f"#{ERROR_STATIC_ID}", Static)
        error_static.update(error or "")
        self._create_button.disabled = error is not None

    def _create(self) -> None:
        name = self.query_one(f"#{NAME_INPUT_ID}", Input).value.strip()
        path = self.query_one(f"#{PATH_INPUT_ID}", Input).value.strip()
        error = self._validate(name, path)
        if error is not None:
            # Create can only be pressed while enabled, but Input.Submitted
            # bypasses the disabled check -- validate again here.
            self.query_one(f"#{ERROR_STATIC_ID}", Static).update(error)
            self._create_button.disabled = True
            return
        self.dismiss(
            ConsoleWorkspaceSetupResult(
                name=name,
                folder_path=path,
                allow_write=self.query_one(f"#{WRITE_CHECKBOX_ID}", Checkbox).value,
            )
        )
