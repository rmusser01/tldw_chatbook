"""Staged Buddy and Persona controls shared by Console and floating companions."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Select, Static, Switch

from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

PERSONA_UNCHANGED = "#unchanged"
PERSONA_NONE = "#none"
_NO_ARTWORK = "#none"
_NO_TARGET = "#none"


@dataclass(frozen=True, slots=True)
class BuddyTargetChoice:
    """A named, explicitly resolved follow target with Persona edit availability."""

    key: str
    label: str
    binding: BuddyBinding
    persona_editable: bool = True
    persona_unavailable_reason: str = ""
    current_persona: str = "None"


@dataclass(frozen=True, slots=True)
class BuddyManagementChoice:
    """User intent returned by Apply; constructing this changes no application state."""

    enabled: bool = False
    buddy_id: str | None = None
    import_path: str = ""
    binding: BuddyBinding | None = None
    persona_choice: str = PERSONA_UNCHANGED
    animated: bool = True
    speak_responses: bool = False
    width: int = 28
    height: int = 12


class BuddyManagementModal(
    SafeModalDismissMixin, ModalScreen[BuddyManagementChoice | None]
):
    """Edit a staged choice, returning None on Escape or Cancel.

    Args:
        buddies: Installed artwork (display label, stable Buddy ID) choices.
        targets: Named local conversation and workspace choices.
        personas: Available assistant Persona (display label, profile ID) choices.
        initial: Current preference snapshot; never modified by this form.
        preview: Read-only callback accepting Buddy ID and expression state.
    """

    SAFE_MODAL_CONTENT = "#buddy-management"
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "request_safe_cancel", "Cancel")
    ]

    DEFAULT_CSS = """
    BuddyManagementModal { align: center middle; }
    #buddy-management {
        width: 78; max-width: 96%; height: 90%; max-height: 48;
        border: round $accent; background: $panel; padding: 0 1;
    }
    #buddy-management-title { height: 2; text-style: bold; padding-top: 1; }
    #buddy-management-body { height: 1fr; padding-right: 1; }
    .buddy-section { height: auto; margin-top: 1; text-style: bold; }
    .buddy-help { height: auto; color: $text-muted; }
    #buddy-management-body Select, #buddy-management-body Input { width: 100%; }
    .buddy-toggle-row { height: 3; align-vertical: middle; }
    .buddy-toggle-row Static { width: 1fr; height: auto; }
    .buddy-toggle-row Switch { width: auto; }
    #buddy-size { height: 3; }
    #buddy-management-body #buddy-size Input { width: 1fr; min-width: 4; }
    #buddy-size Static { width: 9; height: 3; content-align: left middle; }
    #buddy-size #buddy-height-label { width: 10; padding-left: 1; }
    #buddy-preview { height: auto; max-height: 12; content-align: center middle; }
    #buddy-preview-actions { height: 3; }
    #buddy-management-body #buddy-preview-state { width: 1fr; min-width: 10; }
    #buddy-preview-button { width: auto; min-width: 11; }
    #buddy-form-error, #buddy-import-error { height: auto; color: $error; }
    #buddy-advanced { height: auto; padding: 0; }
    #buddy-preview-button:focus { text-style: bold reverse; }
    #buddy-management-actions { height: 3; min-height: 3; align-horizontal: right; }
    #buddy-management-actions Button { width: auto; min-width: 10; margin-left: 1; }
    """

    def __init__(
        self,
        *,
        buddies: tuple[tuple[str, str], ...] = (),
        targets: tuple[BuddyTargetChoice, ...] = (),
        personas: tuple[tuple[str, str], ...] = (),
        initial: BuddyManagementChoice | None = None,
        preview: Callable[[str, str], Any] | None = None,
        apply: Callable[[BuddyManagementChoice], Any] | None = None,
    ) -> None:
        super().__init__()
        self._buddies = buddies
        self._targets = targets
        self._personas = personas
        self._initial = initial or BuddyManagementChoice()
        self._preview_callback = preview
        self._preview_generation = 0
        self._apply_callback = apply
        self._applying = False

    def compose(self) -> ComposeResult:
        initial = self._initial
        selected_artwork = initial.buddy_id or _NO_ARTWORK
        artwork_choices = [(Text("Choose artwork"), _NO_ARTWORK)] + [
            (Text(label), key) for label, key in self._buddies
        ]
        if initial.buddy_id and initial.buddy_id not in {
            key for _, key in self._buddies
        }:
            artwork_choices.append(
                (Text("Selected artwork is unavailable"), initial.buddy_id)
            )
        target_choices = [(Text("None — appearance only"), _NO_TARGET)] + [
            (Text(target.label), target.key) for target in self._targets
        ]
        target_key = next(
            (t.key for t in self._targets if t.binding == initial.binding), _NO_TARGET
        )
        if initial.binding is not None and target_key == _NO_TARGET:
            # Preserve the missing choice so Apply cannot silently detach a stale binding.
            target_key = "#missing"
            target_choices.append(
                (Text("Current target is unavailable — choose another"), target_key)
            )
        with Vertical(id="buddy-management"):
            yield Static("Buddy & Persona Management", id="buddy-management-title")
            with VerticalScroll(id="buddy-management-body"):
                yield Static("Buddy", classes="buddy-section")
                with Horizontal(classes="buddy-toggle-row"):
                    yield Static("Show Buddy")
                    yield Switch(initial.enabled, id="buddy-enabled")
                yield Select(
                    artwork_choices,
                    value=selected_artwork,
                    allow_blank=False,
                    id="buddy-artwork",
                )
                if self._preview_callback is not None:
                    with Horizontal(id="buddy-preview-actions"):
                        yield Select(
                            [
                                (s.replace("_", " ").capitalize(), s)
                                for s in (
                                    "idle",
                                    "thinking",
                                    "speaking",
                                    "approval_needed",
                                    "error",
                                )
                            ],
                            value="idle",
                            allow_blank=False,
                            id="buddy-preview-state",
                        )
                        yield Button("Preview", id="buddy-preview-button")
                    yield Static("", id="buddy-preview")
                yield Static("Expressions", classes="buddy-help")
                yield Select(
                    [("Dynamic", "dynamic"), ("Static", "static")],
                    value="dynamic" if initial.animated else "static",
                    allow_blank=False,
                    id="buddy-motion",
                )
                yield Static("Follow", classes="buddy-section")
                yield Select(
                    target_choices,
                    value=target_key,
                    allow_blank=False,
                    id="buddy-follow",
                )
                yield Static(
                    "This target stays fixed when you switch screens or conversations.",
                    classes="buddy-help",
                )
                yield Static("Persona", classes="buddy-section")
                yield Static(
                    "", id="buddy-persona-help", classes="buddy-help", markup=False
                )
                yield Select(
                    [
                        (Text("Keep current assignment"), PERSONA_UNCHANGED),
                        (Text("None"), PERSONA_NONE),
                    ]
                    + [(Text(label), key) for label, key in self._personas],
                    value=initial.persona_choice,
                    allow_blank=False,
                    id="buddy-persona",
                )
                with Collapsible(
                    title="Import pack & size",
                    collapsed=not bool(initial.import_path),
                    id="buddy-advanced",
                ):
                    yield Static(
                        "Import a native Buddy pack (.tldw-persona-vpack or .zip)",
                        classes="buddy-help",
                    )
                    yield Input(
                        value=initial.import_path,
                        placeholder="Path to pack; installed when you Apply",
                        id="buddy-import",
                    )
                    yield Static("", id="buddy-import-error", markup=False)
                    with Horizontal(id="buddy-size"):
                        yield Static("Width")
                        yield Input(
                            str(initial.width), type="integer", id="buddy-width"
                        )
                        yield Static("Height", id="buddy-height-label")
                        yield Input(
                            str(initial.height), type="integer", id="buddy-height"
                        )
                yield Static("Notifications & voice", classes="buddy-section")
                with Horizontal(classes="buddy-toggle-row"):
                    yield Static("Speak responses with conversation names")
                    yield Switch(initial.speak_responses, id="buddy-speech")
                yield Static(
                    "Workspace Buddies accept typed replies; voice input is available only for a conversation Buddy.",
                    classes="buddy-help",
                )
            yield Static("", id="buddy-form-error", markup=False)
            with Horizontal(id="buddy-management-actions"):
                yield Button("Cancel", id="buddy-cancel")
                yield Button("Apply", variant="primary", id="buddy-apply")

    def on_mount(self) -> None:
        super().on_mount()
        self._sync_persona_controls()
        self.query_one("#buddy-enabled").focus()

    def _target(self) -> BuddyTargetChoice | None:
        key = self.query_one("#buddy-follow", Select).value
        return next((target for target in self._targets if target.key == key), None)

    @on(Select.Changed, "#buddy-follow")
    def _follow_changed(self) -> None:
        if self.is_mounted:
            self.query_one("#buddy-persona", Select).value = PERSONA_UNCHANGED
            self._sync_persona_controls()

    def _sync_persona_controls(self) -> None:
        target = self._target()
        control = self.query_one("#buddy-persona", Select)
        control.disabled = target is None or not target.persona_editable
        if target is None:
            message = "Choose a conversation or workspace to manage its Persona."
        elif not target.persona_editable:
            message = (
                target.persona_unavailable_reason
                or "Persona changes are unavailable for this target."
            )
        elif target.binding.kind == "workspace":
            message = "Default Persona for future new conversations in this workspace."
        else:
            message = "Assistant Persona for this conversation. Artwork is independent."
        if target is not None:
            label = (
                "Current default"
                if target.binding.kind == "workspace"
                else "Current Persona"
            )
            message = f"{label}: {target.current_persona}. {message}"
        self.query_one("#buddy-persona-help", Static).update(message)

    @on(Button.Pressed, "#buddy-preview-button")
    async def _preview(self, event: Button.Pressed) -> None:
        event.stop()
        buddy_id = self.query_one("#buddy-artwork", Select).value
        if buddy_id == _NO_ARTWORK or self._preview_callback is None:
            self.query_one("#buddy-preview", Static).update(
                "Choose installed artwork to preview."
            )
            return
        self._preview_generation += 1
        generation = self._preview_generation
        try:
            result = self._preview_callback(
                str(buddy_id), str(self.query_one("#buddy-preview-state", Select).value)
            )
            if inspect.isawaitable(result):
                result = await result
            if self.is_mounted and generation == self._preview_generation:
                self.query_one("#buddy-preview", Static).update(result)
        except Exception:  # noqa: BLE001 - untrusted asset preview must not terminate the modal
            if self.is_mounted and generation == self._preview_generation:
                self.query_one("#buddy-preview", Static).update(
                    "Preview unavailable. Choose another expression or check the pack."
                )

    @on(Button.Pressed, "#buddy-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once(None)

    def dismiss_safe_once(self, result: object) -> bool:
        if self._applying:
            return False
        return super().dismiss_safe_once(result)

    @on(Button.Pressed, "#buddy-apply")
    def _apply(self, event: Button.Pressed) -> None:
        event.stop()
        if self._applying:
            return
        try:
            choice = self._choice()
        except ValueError as exc:
            self.query_one("#buddy-form-error", Static).update(str(exc))
            return
        if self._apply_callback is None:
            self.dismiss_safe_once(choice)
            return
        self._applying = True
        self.query_one("#buddy-apply", Button).disabled = True
        self.query_one("#buddy-apply", Button).label = "Applying…"
        self.query_one("#buddy-cancel", Button).disabled = True
        self.query_one("#buddy-management-body").disabled = True
        self.query_one("#buddy-form-error", Static).update(
            "Applying settings… Keep this dialog open."
        )
        self.run_worker(
            self._commit(choice), group="buddy-management-commit", exclusive=True
        )

    async def _commit(self, choice: BuddyManagementChoice) -> None:
        succeeded = False
        try:
            result = self._apply_callback(choice)
            if inspect.isawaitable(result):
                await result
            succeeded = True
        except Exception as exc:  # noqa: BLE001 - preserve staged input on storage failure
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else "Could not save Buddy settings. Check profile storage and retry."
            )
            self.query_one("#buddy-form-error", Static).update(message)
            if choice.import_path:
                self.query_one("#buddy-import-error", Static).update(message)
                self.query_one("#buddy-advanced", Collapsible).collapsed = False
        finally:
            self._applying = False
            self.query_one("#buddy-management-body").disabled = False
            self.query_one("#buddy-apply", Button).disabled = False
            self.query_one("#buddy-apply", Button).label = "Apply"
            self.query_one("#buddy-cancel", Button).disabled = False
        if succeeded:
            self.dismiss_safe_once(None)

    def _choice(self) -> BuddyManagementChoice:
        enabled = self.query_one("#buddy-enabled", Switch).value
        artwork = str(self.query_one("#buddy-artwork", Select).value)
        archive = self.query_one("#buddy-import", Input).value.strip()
        if enabled and artwork not in {key for _, key in self._buddies} and not archive:
            raise ValueError(
                "Choose artwork or enter a Buddy pack path before enabling it."
            )
        target_key = self.query_one("#buddy-follow", Select).value
        target = self._target()
        if target_key != _NO_TARGET and target is None:
            raise ValueError("Choose an available conversation or workspace.")
        try:
            width = int(self.query_one("#buddy-width", Input).value)
            height = int(self.query_one("#buddy-height", Input).value)
        except ValueError:
            raise ValueError("Enter whole numbers for width and height.") from None
        if not (8 <= width <= 120 and 4 <= height <= 60):
            raise ValueError(
                "Use a width of 8–120 and a height of 4–60 terminal cells."
            )
        persona = str(self.query_one("#buddy-persona", Select).value)
        if persona not in {
            PERSONA_UNCHANGED,
            PERSONA_NONE,
            *[key for _, key in self._personas],
        }:
            raise ValueError("Choose an available Persona or None.")
        if persona != PERSONA_UNCHANGED and (
            target is None or not target.persona_editable
        ):
            raise ValueError("Persona changes are unavailable for this target.")
        return BuddyManagementChoice(
            enabled=enabled,
            buddy_id=None if artwork == _NO_ARTWORK else artwork,
            import_path=archive,
            binding=target.binding if target else None,
            persona_choice=persona,
            animated=self.query_one("#buddy-motion", Select).value == "dynamic",
            speak_responses=self.query_one("#buddy-speech", Switch).value,
            width=width,
            height=height,
        )
