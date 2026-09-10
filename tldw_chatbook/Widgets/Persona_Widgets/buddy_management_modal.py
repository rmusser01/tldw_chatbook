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
    #buddy-artwork-pages { height: 3; }
    #buddy-artwork-pages Button { width: auto; min-width: 10; }
    #buddy-artwork-page { width: 1fr; content-align: center middle; }
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
        artwork_page: Callable[[int], Any] | None = None,
        artwork_page_size: int = 100,
        selected_buddy: tuple[str, str] | None = None,
        import_petdex: Callable[..., Any] | None = None,
        create_character: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__()
        self._buddies = buddies
        self._artwork_page = artwork_page
        self._artwork_page_size = artwork_page_size
        self._artwork_offset = 0
        self._paging = False
        self._selected_buddy = selected_buddy
        self._targets = targets
        self._personas = personas
        self._initial = initial or BuddyManagementChoice()
        self._preview_callback = preview
        self._preview_generation = 0
        self._apply_callback = apply
        self._applying = False
        self._import_petdex = import_petdex
        self._create_character = create_character
        self.staged_review = None
        self._reviewing = False

    def compose(self) -> ComposeResult:
        initial = self._initial
        selected_artwork = initial.buddy_id or _NO_ARTWORK
        artwork_choices = [(Text("Choose artwork"), _NO_ARTWORK)] + [
            (Text(label), key) for label, key in self._buddies
        ]
        if self._selected_buddy and self._selected_buddy[1] not in {
            key for _, key in self._buddies
        }:
            artwork_choices.append(
                (Text(self._selected_buddy[0]), self._selected_buddy[1])
            )
        if (
            initial.buddy_id
            and initial.buddy_id not in {key for _, key in self._buddies}
            and (
                self._selected_buddy is None
                or self._selected_buddy[1] != initial.buddy_id
            )
        ):
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
                if self._import_petdex is not None:
                    yield Button("Import from Petdex", id="buddy-petdex")
                    yield Static(
                        "", id="buddy-staged", classes="buddy-help", markup=False
                    )
                if self._create_character is not None:
                    yield Button(
                        "Create character", id="buddy-character", disabled=True
                    )
                if self._artwork_page is not None:
                    with Horizontal(id="buddy-artwork-pages"):
                        yield Button(
                            "Previous", id="buddy-artwork-previous", disabled=True
                        )
                        yield Static("Page 1", id="buddy-artwork-page")
                        yield Button(
                            "Next",
                            id="buddy-artwork-next",
                            disabled=len(self._buddies) < self._artwork_page_size,
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
                        placeholder="Absolute path or ~/Downloads/pack.tldw-persona-vpack",
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
        self._sync_character_control()
        self.query_one("#buddy-enabled").focus()

    @on(Select.Changed, "#buddy-artwork")
    @on(Input.Changed, "#buddy-import")
    def _artwork_changed(self) -> None:
        if not self.is_mounted:
            return
        if (
            self.query_one("#buddy-artwork", Select).value != _NO_ARTWORK
            or self.query_one("#buddy-import", Input).value
        ):
            self.staged_review = None
            if self._import_petdex is not None:
                self.query_one("#buddy-staged", Static).update("")
        self._sync_character_control()

    def _sync_character_control(self) -> None:
        if self._create_character is not None:
            self.query_one("#buddy-character", Button).disabled = bool(
                self._reviewing
                or self.staged_review is not None
                or self.query_one("#buddy-import", Input).value
                or self.query_one("#buddy-artwork", Select).value == _NO_ARTWORK
            )

    @on(Button.Pressed, "#buddy-petdex")
    @on(Button.Pressed, "#buddy-character")
    def _review_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._reviewing and not self._applying:
            self._reviewing = True
            self.run_worker(
                self._review(event.button.id),
                group="buddy-management-review",
                exclusive=True,
            )

    async def _review(self, action: str) -> None:
        self.query_one("#buddy-apply", Button).disabled = True
        self.query_one("#buddy-management-body").disabled = True
        try:
            if action == "buddy-petdex":
                result = await self._import_petdex(self)
                if result is not None and self.is_mounted:
                    self.query_one("#buddy-artwork", Select).value = _NO_ARTWORK
                    self.query_one("#buddy-import", Input).value = ""
                    self.staged_review = result
                    self.query_one("#buddy-staged", Static).update(
                        f"{result.title} reviewed. Apply to install. Before Apply, "
                        "Cancel discards this staged review without installing it."
                    )
            else:
                await self._create_character(
                    self, str(self.query_one("#buddy-artwork", Select).value)
                )
        except (ValueError, OSError, RuntimeError):
            if self.is_mounted and self.query("#buddy-form-error"):
                self.query_one("#buddy-form-error", Static).update(
                    "Source or profile changed, or review failed. Start a fresh review."
                )
        finally:
            self._reviewing = False
            if self.is_mounted and self.query("#buddy-management-body"):
                self.query_one("#buddy-management-body").disabled = False
                self.query_one("#buddy-apply", Button).disabled = False
                self._sync_character_control()

    def on_unmount(self) -> None:
        super().on_unmount()
        self.staged_review = None

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

    @on(Button.Pressed, "#buddy-artwork-previous")
    @on(Button.Pressed, "#buddy-artwork-next")
    async def _change_artwork_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self._paging or self._applying or self._artwork_page is None:
            return
        delta = (
            -self._artwork_page_size
            if event.button.id == "buddy-artwork-previous"
            else self._artwork_page_size
        )
        offset = max(0, self._artwork_offset + delta)
        self._paging = True
        previous = self.query_one("#buddy-artwork-previous", Button)
        following = self.query_one("#buddy-artwork-next", Button)
        previous.disabled = following.disabled = True
        try:
            page = self._artwork_page(offset)
            if inspect.isawaitable(page):
                page = await page
            if not self.is_mounted:
                return
            control = self.query_one("#buddy-artwork", Select)
            selected = control.value
            known = {key: label for label, key in self._buddies}
            if self._selected_buddy is not None:
                known[self._selected_buddy[1]] = self._selected_buddy[0]
            if selected in known:
                self._selected_buddy = (known[selected], selected)
            self._buddies = tuple(page)
            self._artwork_offset = offset
            options = [(Text("Choose artwork"), _NO_ARTWORK)] + [
                (Text(label), key) for label, key in self._buddies
            ]
            if selected != _NO_ARTWORK and selected not in {
                key for _, key in self._buddies
            }:
                options.append(
                    (
                        Text(known.get(selected, "Selected artwork is unavailable")),
                        selected,
                    )
                )
            control.set_options(options)
            control.value = selected
            self.query_one("#buddy-artwork-page", Static).update(
                f"Page {offset // self._artwork_page_size + 1}"
            )
        except Exception:  # noqa: BLE001 - retain staged selection on failed page reads
            if self.is_mounted:
                self.query_one("#buddy-form-error", Static).update(
                    "Could not load artwork. Retry this page."
                )
        finally:
            self._paging = False
            if self.is_mounted:
                previous.disabled = self._artwork_offset == 0
                following.disabled = len(self._buddies) < self._artwork_page_size

    def _choice(self) -> BuddyManagementChoice:
        # Lazy loading keeps the boot import budget independent of this form.
        from tldw_chatbook.Utils.input_validation import BuddyManagementInput

        try:
            values = BuddyManagementInput(
                enabled=self.query_one("#buddy-enabled", Switch).value,
                artwork=self.query_one("#buddy-artwork", Select).value,
                archive=self.query_one("#buddy-import", Input).value.strip(),
                target=self.query_one("#buddy-follow", Select).value,
                persona=self.query_one("#buddy-persona", Select).value,
                motion=self.query_one("#buddy-motion", Select).value,
                speak_responses=self.query_one("#buddy-speech", Switch).value,
                width=self.query_one("#buddy-width", Input).value,
                height=self.query_one("#buddy-height", Input).value,
            )
        except ValueError:
            raise ValueError(
                "Check the pack path and selections. Use a width of 8–120 and a height of 4–60 whole terminal cells."
            ) from None
        enabled, artwork, archive = values.enabled, values.artwork, values.archive
        available = {key for _, key in self._buddies}
        if self._selected_buddy is not None:
            available.add(self._selected_buddy[1])
        if (
            enabled
            and artwork not in available
            and not archive
            and self.staged_review is None
        ):
            raise ValueError(
                "Choose artwork or enter a Buddy pack path before enabling it."
            )
        target_key = values.target
        target = self._target()
        if target_key != _NO_TARGET and target is None:
            raise ValueError("Choose an available conversation or workspace.")
        persona = values.persona
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
            animated=values.motion == "dynamic",
            speak_responses=values.speak_responses,
            width=values.width,
            height=values.height,
        )
