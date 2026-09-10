"""Guided web-search form using the incumbent terminal Settings visual system.

THESIS: make setup and the active default independently understandable.
OWN-WORLD: inherit compact keyboard-first Settings, tokens, and Save/Revert.
STORY: default → backend fields → local setup check → explicit saved search test.
FIRST VIEWPORT: show the default and backend editor before diagnostic detail.
FORM: one scrolling document, native selects and masked inputs, stacked labels.
FINISH: readable status text and useful focus states at wide and compact sizes.
"""

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, Link, Select, Static

from ..UI.Screens.settings_web_search import APPLICATION_DEFAULT, WebSearchSettings
from ..Web_Scraping.search_backend_settings import BACKENDS


class WebSearchSettingsPanel(Vertical):
    """A disposable view over the Settings screen's persistent draft state."""

    def __init__(self, model: WebSearchSettings, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self._refreshing = False
        self._capturing_input = False
        self._rendered_inputs: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("Web Search", classes="destination-section settings-column-title")
        yield Static("Default search backend", classes="settings-detail-row")
        options = [("Application default (DuckDuckGo)", APPLICATION_DEFAULT)] + [
            (spec.label, key) for key, spec in BACKENDS.items()
        ]
        default = self.model.default_backend
        if default not in {value for _, value in options}:
            options.append(("Unsupported saved default — choose a backend", default))
        yield Select(
            options,
            value=default,
            allow_blank=False,
            id="web-search-default",
            classes="settings-compact-select",
        )
        yield Static(
            "Basic and deep search share this default.",
            classes="settings-detail-row",
        )
        yield Static(
            self.model.default_status,
            id="web-search-default-status",
            classes="settings-detail-row",
            markup=False,
        )
        yield Static("Configure backend", classes="settings-detail-row")
        yield Select(
            [(spec.label, key) for key, spec in BACKENDS.items()],
            value=self.model.backend,
            allow_blank=False,
            id="web-search-backend",
            classes="settings-compact-select",
        )
        yield Static(
            "Editing another backend keeps your default unchanged.",
            classes="settings-detail-row",
        )
        with Vertical(id="settings-web-search-fields"):
            yield from self._fields()
        yield Static(
            self.model.setup_status,
            id="web-search-setup",
            classes="settings-detail-row",
            markup=False,
        )
        yield Static(
            self.model.save_status,
            id="web-search-save-status",
            classes="settings-detail-row",
            markup=False,
        )
        with Vertical(id="settings-web-search-compact-actions"):
            yield Static("", id="web-search-save-hint", classes="settings-detail-row")
            yield Button(
                "Save all search settings", id="web-search-save", variant="primary"
            )
            yield Button("Revert search changes", id="web-search-revert")
        yield Button(
            "Test saved settings",
            id="web-search-test",
            disabled=not self.model.can_test,
        )
        yield Static(
            "Sends “tldw chatbook” to this backend; may use API quota. No AI answer is generated.",
            classes="settings-detail-row",
        )
        yield Static("", id="web-search-test-hint", classes="settings-detail-row")
        yield Static(
            self.model.test_status,
            id="web-search-test-status",
            classes="settings-detail-row",
            markup=False,
        )

    def _fields(self) -> ComposeResult:
        spec = BACKENDS[self.model.backend]
        yield Static(spec.description, classes="settings-detail-row")
        if spec.notice:
            yield Static(spec.notice, classes="settings-detail-row")
        yield Link("Open backend setup guide", url=spec.docs_url)
        for field in spec.fields:
            value = self.model.input_value(field)
            self._rendered_inputs[field.key] = value
            yield Static(field.label, classes="settings-detail-row")
            yield Input(
                value=value,
                password=field.secret,
                placeholder="Enter replacement key"
                if field.secret
                else field.placeholder,
                id=f"web-search-{field.key}",
                classes="settings-compact-input",
            )
            yield Static(
                self.model.field_status(field),
                id=f"web-search-source-{field.key}",
                classes="settings-detail-row",
                markup=False,
            )
            yield Button(
                f"Clear local {field.label.lower()}", id=f"web-search-clear-{field.key}"
            )

    def on_mount(self) -> None:
        # A new view invalidates evidence even when it mounts before the old
        # view unmounts and takes over callback ownership.
        self.model.invalidate_test()
        self.model.view_changed = self.refresh_status
        self.model.capture_input = self.capture_pending_input
        self.call_after_refresh(self.refresh_status)

    def on_unmount(self) -> None:
        if self.model.view_changed == self.refresh_status:
            self.model.view_changed = None
            self.model.invalidate_test()
        if self.model.capture_input == self.capture_pending_input:
            self.model.capture_input = None

    def capture_pending_input(self) -> None:
        """Capture the actual inputs before painting status or starting work."""
        if self._refreshing or self._capturing_input:
            return
        self._capturing_input = True
        try:
            for key, rendered in tuple(self._rendered_inputs.items()):
                matches = self.query(f"#web-search-{key}")
                if matches:
                    value = matches.first(Input).value
                    if value != rendered:
                        self._rendered_inputs[key] = value
                        self.model.edit(key, value)
        finally:
            self._capturing_input = False

    def refresh_status(self) -> None:
        if not self.is_mounted:
            return
        self.capture_pending_input()
        for selector, text in (
            ("#web-search-setup", self.model.setup_status),
            ("#web-search-default-status", self.model.default_status),
            ("#web-search-save-status", self.model.save_status),
            ("#web-search-test-status", self.model.test_status),
            (
                "#web-search-test-hint",
                "Save or revert changes before testing."
                if self.model.draft.is_dirty
                else "",
            ),
        ):
            widget = self.query_one(selector, Static)
            widget.update(text)
            widget.display = bool(text)
        for field in BACKENDS[self.model.backend].fields:
            matches = self.query(f"#web-search-source-{field.key}")
            if matches:
                matches.first(Static).update(self.model.field_status(field))
            inputs = self.query(f"#web-search-{field.key}")
            if inputs:
                control = inputs.first(Input)
                value = self.model.input_value(field)
                if control.value != value:
                    self._rendered_inputs[field.key] = value
                    with self.prevent(Input.Changed):
                        control.value = value
        for control in self.query("Input, Select, Button"):
            control.disabled = self.model.saving
        self.query_one("#web-search-test", Button).disabled = not self.model.can_test
        self.query_one("#web-search-save-hint", Static).update(
            "Saving search settings…"
            if self.model.saving
            else "Unsaved changes in Web Search."
            if self.model.draft.is_dirty
            else "No unsaved changes."
        )
        for selector in ("#web-search-save", "#web-search-revert"):
            self.query_one(selector, Button).disabled = (
                self.model.saving or not self.model.draft.is_dirty
            )

    async def reload_fields(self) -> None:
        self._refreshing = True
        try:
            fields = self.query_one("#settings-web-search-fields", Vertical)
            await fields.remove_children()
            self._rendered_inputs.clear()
            await fields.mount(*self._fields())
        finally:
            self._refreshing = False
        self.refresh_status()

    @on(Select.Changed)
    async def select_changed(self, event: Select.Changed) -> None:
        if event.select not in self.query(Select) or not isinstance(event.value, str):
            return
        event.stop()
        self.capture_pending_input()
        if event.select.id == "web-search-default":
            if event.value != self.model.default_backend:
                self.model.set_default(event.value)
                self.model.select_backend(
                    "duckduckgo" if event.value == APPLICATION_DEFAULT else event.value
                )
                with self.prevent(Select.Changed):
                    self.query_one(
                        "#web-search-backend", Select
                    ).value = self.model.backend
                await self.reload_fields()
        elif (
            event.select.id == "web-search-backend"
            and event.value != self.model.backend
        ):
            self.model.select_backend(event.value)
            await self.reload_fields()

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        if (
            self._refreshing
            or event.input not in self.query(Input)
            or event.value != event.input.value
        ):
            return
        event.stop()
        key = (event.input.id or "").removeprefix("web-search-")
        field = next(
            (
                field
                for field in BACKENDS[self.model.backend].fields
                if field.key == key
            ),
            None,
        )
        if field and event.value != self.model.input_value(field):
            self._rendered_inputs[field.key] = event.value
            self.model.edit(key, event.value)

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("web-search-"):
            return
        event.stop()
        if button_id == "web-search-test":
            self.app.run_worker(
                self.model.test_saved(), group="web-search-probe", exclusive=False
            )
        elif button_id == "web-search-save":
            self.screen.action_settings_save_category(allow_text_entry_focus=True)
        elif button_id == "web-search-revert":
            self.screen.action_settings_revert_category(allow_text_entry_focus=True)
        elif button_id.startswith("web-search-clear-"):
            self.model.clear(button_id.removeprefix("web-search-clear-"))
            await self.reload_fields()

    async def revert(self) -> None:
        self.model.revert()
        default = self.query_one("#web-search-default", Select)
        with self.prevent(Select.Changed):
            default.value = self.model.default_backend
        await self.reload_fields()
