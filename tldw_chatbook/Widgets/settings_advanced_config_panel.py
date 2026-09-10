"""Raw TOML editing with persistent drafts and explicit recovery controls."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Grid, Vertical
from textual.widgets import Button, Collapsible, Static, TextArea

from ..UI.Screens.settings_advanced_config import AdvancedConfigSettings
from .confirmation_dialog import ConfirmationDialog


class AdvancedConfigPanel(Vertical):
    """Keep the expert editor usable while the screen owns its draft lifetime."""

    def __init__(self, model: AdvancedConfigSettings, guided_paths: tuple, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.guided_paths = guided_paths
        self._rendered_text = model.state.text
        self._busy_focus: Button | None = None

    def compose(self) -> ComposeResult:
        with Collapsible(
            title="Raw editing guide", collapsed=True, id="settings-advanced-guide"
        ):
            yield Static(
                "Raw TOML bypasses guided validation. Save blocked until the current text validates. Saves are atomic and create a .bak backup."
            )
            yield Static(
                "Drafts survive navigation in this app session. Prefer guided categories when available."
            )
            for category, label in self.guided_paths:
                yield Button(
                    label,
                    id=f"settings-advanced-open-{category.value}",
                    classes="settings-advanced-guided-path-button",
                )
        with Grid(id="settings-advanced-config-actions", classes="settings-action-row"):
            yield Button("Validate Raw TOML", id="settings-advanced-validate-config")
            yield Button(
                "Save Raw TOML",
                id="settings-advanced-save-config",
                disabled=not self.model.can_save,
            )
            yield Button("Load Backup", id="settings-advanced-load-backup")
            yield Button("Revert Raw TOML", id="settings-advanced-revert-config")
        yield Static(
            self.model.validation_status,
            id="settings-advanced-config-validation-status",
            markup=False,
        )
        yield Static(
            self.model.status,
            id="settings-advanced-config-result",
            classes="settings-status-row",
            markup=False,
        )
        yield TextArea(self.model.state.text, id="settings-advanced-config-editor")

    def on_mount(self) -> None:
        self.model.view_changed = self.refresh_state
        self.model.read_editor = self.editor_text
        self.call_after_refresh(self.refresh_state)
        self.app.run_worker(
            self.model.inspect_current(), group="raw-config-inspect", exclusive=False
        )

    def on_unmount(self) -> None:
        if self.model.view_changed == self.refresh_state:
            self.model.view_changed = None
        if self.model.read_editor == self.editor_text:
            self.model.read_editor = None

    def editor_text(self) -> str:
        """Read the document before deferred change events are delivered."""
        return self.query_one(TextArea).text

    def refresh_state(self) -> None:
        if not self.is_mounted:
            return
        editor = self.query_one(TextArea)
        editor.read_only = self.model.busy == "Loading config…"
        if editor.text != self._rendered_text:
            self._rendered_text = editor.text
            self.model.edit(editor.text)
        elif editor.text != self.model.state.text:
            self._rendered_text = self.model.state.text
            with self.prevent(TextArea.Changed):
                editor.text = self.model.state.text
        self.query_one("#settings-advanced-config-validation-status", Static).update(
            self.model.busy or self.model.validation_status
        )
        result = self.query_one("#settings-advanced-config-result", Static)
        result.update(self.model.status)
        result.display = bool(self.model.status)
        for control in self.query("#settings-advanced-config-actions Button"):
            if self.model.busy and control.has_focus:
                self._busy_focus = control
            control.disabled = bool(self.model.busy)
        self.query_one(
            "#settings-advanced-save-config", Button
        ).disabled = not self.model.can_save
        revert = self.query_one("#settings-advanced-revert-config", Button)
        revert.disabled = bool(self.model.busy) or not (
            self.model.state.is_dirty
            or self.model.state.file_changed
            or self.model.state.snapshot is None
        )

        if not self.model.busy and self._busy_focus is not None:
            if self.app.focused is None:
                target = (
                    self._busy_focus
                    if not self._busy_focus.disabled
                    else self.query_one("#settings-advanced-validate-config", Button)
                )
                target.focus()
            self._busy_focus = None

    @on(TextArea.Changed)
    def editor_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is not self.query_one(TextArea):
            return
        event.stop()
        self.model.edit(event.text_area.text)

    def request_replacement(self, source: str) -> None:
        if self.model.busy:
            return
        revision = self.model.state.revision

        async def replace() -> None:
            self.app.run_worker(
                self.model.replace_draft(source, revision),
                group="raw-config-replace",
                exclusive=False,
            )

        if self.model.state.is_dirty:
            self.app.push_screen(
                ConfirmationDialog(
                    title="Replace raw TOML draft",
                    message="Discard unsaved raw TOML and load the backup?"
                    if source == "backup"
                    else "Discard unsaved raw TOML and reload the current config?",
                    confirm_label="Replace draft",
                    cancel_label="Keep editing",
                    confirm_callback=replace,
                )
            )
        else:
            self.app.run_worker(
                self.model.replace_draft(source, revision),
                group="raw-config-replace",
                exclusive=False,
            )

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        action = {
            "settings-advanced-validate-config": "validate",
            "settings-advanced-save-config": "save",
            "settings-advanced-load-backup": "backup",
            "settings-advanced-revert-config": "revert",
        }.get(event.button.id)
        if action is None:
            return
        event.stop()
        if action in ("backup", "revert"):
            self.request_replacement(action)
        else:
            self.app.run_worker(
                getattr(self.model, action)(),
                group="raw-config-action",
                exclusive=False,
            )
