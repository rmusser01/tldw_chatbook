"""Compact character Voice & Speech controls for the Personas workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Select, Static

from ...TTS import ProfileAvailabilityState
from .personas_pane_messages import CharacterTTSActionRequested

_GLOBAL_PROFILE_VALUE = "__global__"
_CONTEXTS = frozenset({"card", "editor"})


@dataclass(frozen=True, slots=True)
class CharacterTTSProfileOption:
    """One presentation-only profile choice."""

    profile_id: UUID
    display_name: str
    availability: ProfileAvailabilityState

    def __post_init__(self) -> None:
        if (
            type(self.profile_id) is not UUID
            or type(self.display_name) is not str
            or not self.display_name.strip()
            or self.display_name != self.display_name.strip()
            or type(self.availability) is not str
            or self.availability not in {"available", "unavailable", "unverified"}
        ):
            raise ValueError("invalid character TTS profile option")


@dataclass(frozen=True, slots=True)
class CharacterTTSPresentationState:
    """Immutable screen-owned state rendered by both character controls."""

    profiles: tuple[CharacterTTSProfileOption, ...]
    selected_profile_id: UUID | None
    status: str
    controls_enabled: bool
    assignment_count: int | None = None

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if (
            any(type(profile) is not CharacterTTSProfileOption for profile in profiles)
            or len({profile.profile_id for profile in profiles}) != len(profiles)
            or (
                self.selected_profile_id is not None
                and self.selected_profile_id
                not in {profile.profile_id for profile in profiles}
            )
            or type(self.status) is not str
            or not self.status
            or type(self.controls_enabled) is not bool
            or (
                self.assignment_count is not None
                and (
                    type(self.assignment_count) is not int or self.assignment_count < 0
                )
            )
        ):
            raise ValueError("invalid character TTS presentation state")
        object.__setattr__(self, "profiles", profiles)

    @classmethod
    def disabled(
        cls,
        status: str = "Save/reopen before assigning.",
    ) -> "CharacterTTSPresentationState":
        """Return the bounded no-authority/unsaved presentation."""

        return cls(
            profiles=(),
            selected_profile_id=None,
            status=status,
            controls_enabled=False,
        )


class PersonasCharacterTTSWidget(Container):
    """Render character profile state and emit identity-free action intents."""

    DEFAULT_CSS = """
    PersonasCharacterTTSWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 1 0 0 0;
    }

    PersonasCharacterTTSWidget .personas-character-tts-row {
        width: 100%;
        height: auto;
    }

    PersonasCharacterTTSWidget .personas-character-tts-profile {
        width: 1fr;
    }

    PersonasCharacterTTSWidget .personas-character-tts-status {
        width: 100%;
        height: auto;
    }

    PersonasCharacterTTSWidget .personas-character-tts-actions {
        width: 100%;
        height: 1;
    }

    PersonasCharacterTTSWidget .personas-character-tts-actions Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        *,
        context: Literal["card", "editor"],
    ) -> None:
        if context not in _CONTEXTS:
            raise ValueError("invalid character TTS widget context")
        super().__init__(id=f"personas-character-{context}-tts")
        self._control_context = context
        self._state = CharacterTTSPresentationState.disabled()

    @property
    def presentation_state(self) -> CharacterTTSPresentationState:
        """Return the exact immutable state currently rendered."""

        return self._state

    def compose(self) -> ComposeResult:
        with Vertical(classes="personas-character-tts-row"):
            yield Static("Voice & Speech", classes="destination-section")
            yield Select(
                [("Use global default", _GLOBAL_PROFILE_VALUE)],
                value=_GLOBAL_PROFILE_VALUE,
                allow_blank=False,
                compact=True,
                classes="personas-character-tts-profile",
                disabled=True,
            )
            yield Static(
                self._state.status,
                classes="personas-character-tts-status",
                markup=False,
            )
            with Horizontal(classes="personas-character-tts-actions"):
                yield Button(
                    "Preview",
                    classes=("console-action-subdued personas-character-tts-preview"),
                    disabled=True,
                )
                yield Button(
                    "Create",
                    classes="console-action-subdued personas-character-tts-create",
                    disabled=True,
                )
                yield Button(
                    "Edit",
                    classes="console-action-subdued personas-character-tts-edit",
                    disabled=True,
                )
                yield Button(
                    "Remove",
                    classes="console-action-subdued personas-character-tts-remove",
                    disabled=True,
                )

    def apply_state(self, state: CharacterTTSPresentationState) -> None:
        """Render one immutable screen-owned presentation snapshot."""

        if type(state) is not CharacterTTSPresentationState:
            raise TypeError("state must be CharacterTTSPresentationState")
        self._state = state
        selector = self.query_one(".personas-character-tts-profile", Select)
        options = [("Use global default", _GLOBAL_PROFILE_VALUE)]
        options.extend(
            (
                f"{profile.display_name} · {profile.availability}",
                str(profile.profile_id),
            )
            for profile in state.profiles
        )
        with self.prevent(Select.Changed):
            selector.set_options(options)
            selector.value = (
                _GLOBAL_PROFILE_VALUE
                if state.selected_profile_id is None
                else str(state.selected_profile_id)
            )
        selector.disabled = not state.controls_enabled

        self.query_one(
            ".personas-character-tts-status",
            Static,
        ).update(state.status)
        assigned = state.selected_profile_id is not None
        selected = next(
            (
                profile
                for profile in state.profiles
                if profile.profile_id == state.selected_profile_id
            ),
            None,
        )
        available = selected is not None and selected.availability == "available"
        self.query_one(
            ".personas-character-tts-preview",
            Button,
        ).disabled = not (state.controls_enabled and assigned)
        self.query_one(
            ".personas-character-tts-create",
            Button,
        ).disabled = not state.controls_enabled
        edit = self.query_one(".personas-character-tts-edit", Button)
        edit.label = "Edit" if available else "Repair"
        edit.disabled = not (state.controls_enabled and assigned)
        self.query_one(
            ".personas-character-tts-remove",
            Button,
        ).disabled = not (state.controls_enabled and assigned)

    def _restore_selected_value(self) -> None:
        selector = self.query_one(".personas-character-tts-profile", Select)
        with self.prevent(Select.Changed):
            selector.value = (
                _GLOBAL_PROFILE_VALUE
                if self._state.selected_profile_id is None
                else str(self._state.selected_profile_id)
            )

    @on(Select.Changed, ".personas-character-tts-profile")
    def _profile_changed(self, event: Select.Changed) -> None:
        event.stop()
        if not self._state.controls_enabled:
            self._restore_selected_value()
            return
        if event.value == _GLOBAL_PROFILE_VALUE:
            if self._state.selected_profile_id is not None:
                self.post_message(CharacterTTSActionRequested("assign", None))
            return
        try:
            profile_id = UUID(str(event.value))
        except (TypeError, ValueError):
            self._restore_selected_value()
            return
        if profile_id == self._state.selected_profile_id:
            return
        option = next(
            (
                candidate
                for candidate in self._state.profiles
                if candidate.profile_id == profile_id
            ),
            None,
        )
        if option is None or option.availability != "available":
            self._restore_selected_value()
            return
        self.post_message(CharacterTTSActionRequested("assign", profile_id))

    @on(Button.Pressed)
    def _action_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button
        if button.has_class("personas-character-tts-create"):
            self.post_message(CharacterTTSActionRequested("create", None))
            return
        profile_id = self._state.selected_profile_id
        if profile_id is None:
            return
        if button.has_class("personas-character-tts-preview"):
            action = "preview"
        elif button.has_class("personas-character-tts-edit"):
            action = "edit"
        elif button.has_class("personas-character-tts-remove"):
            action = "remove"
        else:
            return
        self.post_message(CharacterTTSActionRequested(action, profile_id))


__all__ = [
    "CharacterTTSProfileOption",
    "CharacterTTSPresentationState",
    "PersonasCharacterTTSWidget",
]
