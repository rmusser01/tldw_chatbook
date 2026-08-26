"""Compact character Voice & Speech controls for the Personas workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Select, Static

from ...TTS import ProfileAvailabilityState
from ...TTS.profile_service import (
    ProfileRecoveryAction,
    TTSProfileDependencyProjection,
)
from ...UI.tts_profile_recovery import dependency_recovery_actions
from .personas_pane_messages import CharacterTTSAction, CharacterTTSActionRequested

_GLOBAL_PROFILE_VALUE = "__global__"
_CONTEXTS = frozenset({"card", "editor"})


@dataclass(frozen=True, slots=True)
class CharacterTTSProfileSuggestion:
    """Bounded non-authoritative identity offered by a Speech Lab save."""

    profile_id: UUID
    repository_generation: int
    profile_revision: int

    def __post_init__(self) -> None:
        if type(self.profile_id) is not UUID:
            raise TypeError("profile_id must be a UUID")
        for name in ("repository_generation", "profile_revision"):
            value = getattr(self, name)
            minimum = 1 if name == "profile_revision" else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} is invalid")


@dataclass(frozen=True, slots=True)
class CharacterTTSProfileOption:
    """One presentation-only profile choice.

    `recovery_action` defaults to "refresh" -- the audio.cpp-transient
    reading -- so any caller that predates this field (or that only ever
    constructs "available"/"unavailable" options, where the value is
    unused) keeps today's behavior unchanged. Only a caller that has the
    real `TTSProfileAvailability` in hand (`personas_screen.py`'s
    presentation builder) threads the actual value, which is what lets an
    "unverified" option distinguish a legacy provider's permanent
    no-catalog-check state from audio.cpp's transient one (slice 2, task 2).
    """

    profile_id: UUID
    display_name: str
    availability: ProfileAvailabilityState
    recovery_action: ProfileRecoveryAction = "refresh"
    dependency: TTSProfileDependencyProjection = field(
        default_factory=TTSProfileDependencyProjection
    )

    def __post_init__(self) -> None:
        if (
            type(self.profile_id) is not UUID
            or type(self.display_name) is not str
            or not self.display_name.strip()
            or self.display_name != self.display_name.strip()
            or type(self.availability) is not str
            or self.availability not in {"available", "unavailable", "unverified"}
            or type(self.recovery_action) is not str
            or self.recovery_action not in {"none", "refresh", "edit"}
            or type(self.dependency) is not TTSProfileDependencyProjection
        ):
            raise ValueError("invalid character TTS profile option")

    @property
    def assignable(self) -> bool:
        """Return the same blocker truth used by the Profile Library."""

        return self.availability != "unavailable" and self.dependency.reason == "none"


@dataclass(frozen=True, slots=True)
class CharacterTTSPresentationState:
    """Immutable screen-owned state rendered by both character controls."""

    profiles: tuple[CharacterTTSProfileOption, ...]
    selected_profile_id: UUID | None
    status: str
    controls_enabled: bool
    assignment_count: int | None = None
    suggested_profile_id: UUID | None = None

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
            or (
                self.suggested_profile_id is not None
                and self.suggested_profile_id
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


def _character_tts_option_suffix(option: CharacterTTSProfileOption) -> str:
    """Return the Select option's availability suffix.

    `recovery_action == "none"` on an "unverified" option means the
    provider has no catalog to preflight, so the state is permanent --
    naming it plain "unverified" would read as an audio.cpp-style
    transient glitch a refresh could resolve. Never branch on
    `provider_id` here; the recovery action already carries the honest
    distinction (ADR-031, `_ALLOWED_RECOVERY_ACTIONS` in
    `TTS/profile_service.py`).
    """

    if option.dependency.display:
        return option.dependency.display
    if option.dependency.advisory_display:
        return f"{option.availability} · {option.dependency.advisory_display}"
    if option.availability == "unverified" and option.recovery_action == "none":
        return "no catalog check"
    return option.availability


class PersonasCharacterTTSWidget(Container):
    """Render character profile state and emit identity-free action intents."""

    BUNDLED_CSS = """
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
        height: auto;
    }

    PersonasCharacterTTSWidget .personas-character-tts-ordinary-actions {
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

    PersonasCharacterTTSWidget .personas-character-tts-recovery-actions {
        width: 100%;
        height: auto;
    }

    PersonasCharacterTTSWidget .personas-character-tts-recovery-actions Button {
        width: 100%;
        margin-right: 0;
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
            with Vertical(classes="personas-character-tts-actions"):
                with Horizontal(classes="personas-character-tts-ordinary-actions"):
                    yield Button(
                        "Preview",
                        classes=(
                            "console-action-subdued personas-character-tts-preview"
                        ),
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
                    yield Button(
                        "Dismiss",
                        classes=(
                            "console-action-subdued "
                            "personas-character-tts-dismiss-suggestion hidden"
                        ),
                        disabled=True,
                        tooltip="Dismiss the saved Voice Profile suggestion.",
                    )
                with Vertical(classes="personas-character-tts-recovery-actions"):
                    yield Button(
                        "Recovery",
                        classes=(
                            "console-action-subdued "
                            "personas-character-tts-dependency-primary hidden"
                        ),
                        disabled=True,
                    )
                    yield Button(
                        "Recovery",
                        classes=(
                            "console-action-subdued "
                            "personas-character-tts-dependency-advisory hidden"
                        ),
                        disabled=True,
                    )

    def apply_state(self, state: CharacterTTSPresentationState) -> None:
        """Render one immutable screen-owned presentation snapshot."""

        if type(state) is not CharacterTTSPresentationState:
            raise TypeError("state must be CharacterTTSPresentationState")
        self._state = state
        has_suggestion = state.suggested_profile_id is not None
        selector = self.query_one(".personas-character-tts-profile", Select)
        options = [("Use global default", _GLOBAL_PROFILE_VALUE)]
        options.extend(
            (
                (
                    f"{profile.display_name} · {_character_tts_option_suffix(profile)}"
                    + (
                        " · Suggested"
                        if profile.profile_id == state.suggested_profile_id
                        else ""
                    )
                ),
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
        broken = selected is not None and selected.availability == "unavailable"
        preview = self.query_one(
            ".personas-character-tts-preview",
            Button,
        )
        preview_visible = assigned and not has_suggestion
        preview.set_class(not preview_visible, "hidden")
        preview.display = preview_visible
        preview.disabled = has_suggestion or not (state.controls_enabled and assigned)
        create = self.query_one(
            ".personas-character-tts-create",
            Button,
        )
        create_visible = not assigned and not has_suggestion
        create.set_class(not create_visible, "hidden")
        create.display = create_visible
        create.disabled = not (create_visible and state.controls_enabled)
        edit = self.query_one(".personas-character-tts-edit", Button)
        edit_visible = assigned and not has_suggestion
        edit.set_class(not edit_visible, "hidden")
        edit.display = edit_visible
        # "Repair" is reserved for a genuinely unavailable profile. An
        # unverified one is not confirmed broken -- it works, this slice just
        # has no catalog check backing that claim yet (task-2450 amendment) --
        # so it keeps the ordinary "Edit" label rather than being presented
        # as needing repair.
        edit.label = "Repair" if broken else "Edit"
        edit.disabled = has_suggestion or not (state.controls_enabled and assigned)
        remove = self.query_one(
            ".personas-character-tts-remove",
            Button,
        )
        remove_visible = assigned and not has_suggestion
        remove.set_class(not remove_visible, "hidden")
        remove.display = remove_visible
        remove.disabled = has_suggestion or not (state.controls_enabled and assigned)
        dismiss = self.query_one(
            ".personas-character-tts-dismiss-suggestion",
            Button,
        )
        dismiss.set_class(not has_suggestion, "hidden")
        dismiss.display = has_suggestion
        dismiss.disabled = not has_suggestion
        recovery_actions = (
            () if selected is None else dependency_recovery_actions(selected.dependency)
        )
        actions_by_role = {action.role: action for action in recovery_actions}
        role_selectors: tuple[tuple[Literal["blocker", "advisory"], str], ...] = (
            ("blocker", ".personas-character-tts-dependency-primary"),
            ("advisory", ".personas-character-tts-dependency-advisory"),
        )
        for role, selector in role_selectors:
            button = self.query_one(selector, Button)
            action = actions_by_role.get(role)
            visible = action is not None and not has_suggestion
            button.set_class(not visible, "hidden")
            button.display = visible
            button.disabled = not (visible and state.controls_enabled)
            button.label = "Recovery" if action is None else action.label
            button.tooltip = None if action is None else action.tooltip

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
        if option is None or not option.assignable:
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
        if button.has_class("personas-character-tts-dismiss-suggestion"):
            self.post_message(CharacterTTSActionRequested("dismiss_suggestion", None))
            return
        profile_id = self._state.selected_profile_id
        if profile_id is None:
            return
        selected = next(
            (
                option
                for option in self._state.profiles
                if option.profile_id == profile_id
            ),
            None,
        )
        if button.has_class("personas-character-tts-dependency-primary"):
            role = "blocker"
        elif button.has_class("personas-character-tts-dependency-advisory"):
            role = "advisory"
        else:
            role = None
        if role is not None:
            if selected is None:
                return
            projected = next(
                (
                    action
                    for action in dependency_recovery_actions(selected.dependency)
                    if action.role == role
                ),
                None,
            )
            if projected is None:
                return
            self.post_message(
                CharacterTTSActionRequested(projected.operation, profile_id)
            )
            return
        if button.has_class("personas-character-tts-preview"):
            action: CharacterTTSAction = "preview"
        elif button.has_class("personas-character-tts-edit"):
            action = "edit"
        elif button.has_class("personas-character-tts-remove"):
            action = "remove"
        else:
            return
        self.post_message(CharacterTTSActionRequested(action, profile_id))


__all__ = [
    "CharacterTTSProfileSuggestion",
    "CharacterTTSProfileOption",
    "CharacterTTSPresentationState",
    "PersonasCharacterTTSWidget",
]
