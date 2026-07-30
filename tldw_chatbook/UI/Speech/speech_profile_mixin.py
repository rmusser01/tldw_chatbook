"""Saved TTS profiles, shared by whatever hosts the Playground controls.

dev built this onto `TTSPlaygroundWidget` while the Console-grammar rebuild
was in flight -- a profile library, and the ability to save the current
result as a profile and preview an exact one. Retiring that widget would
have deleted the UI half of it, so the behaviour moves here instead and the
rebuilt pane inherits it, exactly as the catalog and synthesis paths did.

Ported rather than reimplemented: these are dev's methods verbatim. They
query their controls by id, so the pane mounts the same ids
(`audio-save-profile-btn`, `tts-profile-preview-status`).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from loguru import logger
from rich.text import Text
from textual.widgets import Button, Select, Static

from tldw_chatbook.TTS import (
    ProfileAvailabilityState,
    TTSPlaygroundSelectionPreset,
)
from tldw_chatbook.TTS.adapter_types import TTSRegistryClosedError

from tldw_chatbook.UI.stts_playground_catalog import (
    AUDIO_CPP_PROVIDER_ID,
    CatalogRequestToken,
    SERVER_DEFAULT_VOICE_ID,
    SelectSentinel,
    controls_from_profile_preset,
)
from tldw_chatbook.UI.stts_profile_library import (
    PROFILE_ACTION_FAILED_COPY,
    PROFILE_STORE_UNAVAILABLE_COPY,
    TTSProfileNameModal,
    profile_action_error_copy,
)


#: Copy shown when the generated audio predates the current settings, so
#: saving it as a profile would record something the user did not hear.
#: Kept verbatim from `STTS_Window`, where dev defined it.
_PROFILE_RESULT_STALE_COPY = (
    "TTS settings changed after this audio was generated. Generate a new "
    "result before saving it as a profile."
)


class SpeechProfileMixin:
    """Profile save/preview behaviour, independent of the layout."""

    def init_profile_state(self, profile_preset: Any = None) -> None:
        """Initialise the state the profile path reads.

        Call from the host's ``__init__``. Same contract shape as the other
        Speech mixins: these are read without guards, so a host that skips
        this fails at the first profile action rather than anywhere useful.

        Args:
            profile_preset: The preset this pane was opened with, if any.
                A pane opened on a preset starts in "preview loading".
        """
        #: The preset this pane was opened with, if any.
        self._profile_preset = profile_preset
        #: The preset's availability as last resolved, or None. Seeded from
        #: the preset exactly as dev's `__init__` did.
        self._profile_effective_availability: Any = (
            profile_preset.availability if profile_preset is not None else None
        )
        #: Token for the in-flight voice validation, so a superseded reply
        #: is discarded rather than overwriting a newer one.
        self._profile_voice_validation_token: Any = None
        #: True while an exact-profile preview is still resolving.
        self._profile_preview_loading = profile_preset is not None
        #: Whether `_apply_controls` has run at least once for this preset.
        #: `_end_profile_preset` reads this without a guard; a plain pane
        #: (no preset) starts True since there is nothing to apply, and a
        #: preset pane starts False until priming's own `_apply_controls`
        #: call (during `on_mount`) flips it. Set here rather than left to
        #: `_apply_controls` alone: `TTSPlaygroundWidget.__init__` sets the
        #: same attribute in its constructor, and this pane's preset path
        #: is production-reachable now that `STTS_Window` mounts it.
        self._profile_controls_applied = profile_preset is None
        #: Catalog revision the preview was admitted against.
        self._profile_configuration_revision: int | None = None
        #: Set while saving is deliberately unavailable.
        self._profile_save_suppressed = False
        #: The open name modal, so a second one is not stacked.
        self._active_profile_name_modal: Any = None

    def _clear_profile_voice_validation(
        self,
        request_token: CatalogRequestToken,
    ) -> None:
        """Clear only the pending exact-profile observation owned by a token."""
        if self._profile_voice_validation_token != request_token:
            return
        self._profile_voice_validation_token = None
        if self.is_mounted:
            self._sync_profile_preview_status()
            self._sync_generate_enabled()

    @staticmethod
    def _dismiss_profile_name_modal(modal: TTSProfileNameModal) -> None:
        if modal.is_mounted and modal.is_current:
            modal.dismiss(None)

    def _end_profile_preset(self, *, before_controls: bool = False) -> bool:
        """Detach exact profile semantics after a user selection edit."""
        if self._profile_preset is None:
            return False
        if not before_controls and not self._profile_controls_applied:
            return False
        self._profile_preset = None
        self._profile_effective_availability = None
        self._profile_preview_loading = False
        self._profile_configuration_revision = None
        self._profile_voice_validation_token = None
        self._profile_controls_applied = True
        self._sync_profile_preview_status()
        self._sync_generate_enabled()
        return True

    def _prime_profile_preset_controls(self) -> None:
        """Show one exact preset disabled before service discovery completes."""
        preset = self._profile_preset
        if preset is None:
            return
        provider_id = preset.provider_id
        display_name = (
            "audio.cpp" if provider_id == AUDIO_CPP_PROVIDER_ID else provider_id
        )
        self._selected_provider_id = provider_id
        self._provider_ids = frozenset((provider_id,))
        self._provider_display_names = {provider_id: display_name}
        provider_select = self.query_one("#tts-provider-select", Select)
        provider_select.set_options(
            self._safe_select_options(((display_name, provider_id),))
        )
        self._applying_catalog_controls = True
        try:
            provider_select.value = provider_id
        finally:
            self._applying_catalog_controls = False
        axis_values = getattr(self, "axis_values", None)
        if axis_values is not None:
            # Provider is an axis like any other: this direct write bypasses
            # `_apply_controls` (below), so the model needs telling
            # separately or the row keeps describing whatever `axis_values`
            # held at construction, not the preset that was just primed.
            axis_values["tts-provider-select"] = provider_id
            # `_refresh_axis_markers` is defined only on `SpeechPlaygroundPane`;
            # the `axis_values is not None` check above is what makes calling
            # it safe on a host with no axis row.
            self._refresh_axis_markers()
        provider_select.disabled = True
        self.query_one("#tts-refresh-catalog-btn", Button).disabled = True
        self._show_provider_specific_controls(provider_id)
        self._project_profile_preset_controls(
            provider_id,
            generation_allowed=False,
        )

    def _profile_preview_blocked_presentation(
        self,
        preset: TTSPlaygroundSelectionPreset,
    ) -> tuple[str, ProfileAvailabilityState] | None:
        """Return bounded recovery copy when the exact preset cannot generate."""
        service = self._tts_service
        if service is None:
            return (
                "Profile preview blocked — the TTS service is unavailable.",
                "unavailable",
            )
        catalog = self._catalogs.get(preset.provider_id)
        if catalog is not None and catalog.health.state == "closed":
            return (
                "Profile preview blocked — the TTS service is unavailable.",
                "unavailable",
            )
        try:
            current_revision = service.configuration_revision(preset.provider_id)
        except (KeyError, TTSRegistryClosedError):
            return (
                "Profile preview blocked — the TTS service is unavailable.",
                "unavailable",
            )
        expected_revision = self._profile_configuration_revision
        if expected_revision is None:
            return (
                "Profile preview blocked — refresh or retry from Voice profiles.",
                "unverified",
            )
        if current_revision != expected_revision:
            return (
                "Profile preview blocked — TTS settings changed; refresh models.",
                "unverified",
            )
        if not self._catalog_generation_allowed:
            return (
                "Profile preview blocked — refresh or retry from Voice profiles.",
                "unverified",
            )
        return None

    def _project_profile_preset_controls(
        self,
        provider_id: str,
        *,
        generation_allowed: bool,
    ) -> bool:
        """Project exact preset controls even when no catalog was acquired."""
        preset = self._profile_preset
        if preset is None or preset.provider_id != provider_id:
            return False
        controls = controls_from_profile_preset(
            self._catalogs.get(provider_id),
            preset=preset,
            discovered_voices=self._discovered_voices.get(
                (provider_id, preset.model_id)
            ),
        )
        self._apply_controls(replace(controls, generation_allowed=generation_allowed))
        return True

    async def _save_current_result_as_profile(self) -> None:
        """Save a captured eligible artifact without rereading selectors."""
        artifact = self.current_audio_artifact
        if (
            artifact is None
            or not artifact.profile_save_eligible
            or self._generation_operation_id is not None
            or self._profile_save_suppressed
        ):
            self._sync_save_profile_action()
            return

        modal = TTSProfileNameModal()
        active = self._active_profile_name_modal
        if active is not None:
            self._dismiss_profile_name_modal(active)
        self._active_profile_name_modal = modal
        try:
            display_name = await self.app.push_screen_wait(modal)
        except asyncio.CancelledError:
            self._dismiss_profile_name_modal(modal)
            if self.is_mounted:
                self._sync_save_profile_action()
            raise
        except Exception:  # noqa: BLE001 - isolate modal lifecycle failure
            self._dismiss_profile_name_modal(modal)
            if self.is_mounted:
                self.query_one("#audio-player-status", Static).update(
                    PROFILE_ACTION_FAILED_COPY
                )
                self._sync_save_profile_action()
            return
        finally:
            if self._active_profile_name_modal is modal:
                self._active_profile_name_modal = None
        if not isinstance(display_name, str) or not display_name.strip():
            self._sync_save_profile_action()
            return

        ensure_service = getattr(self.app, "_ensure_tts_profile_service", None)
        if not callable(ensure_service):
            self.query_one("#audio-player-status", Static).update(
                PROFILE_STORE_UNAVAILABLE_COPY
            )
            self._sync_save_profile_action()
            return
        try:
            service = await ensure_service()
            if service is None:
                self.query_one("#audio-player-status", Static).update(
                    PROFILE_STORE_UNAVAILABLE_COPY
                )
                return
            await service.create_from_artifact(display_name, artifact)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - map to bounded UI copy
            copy = (
                _PROFILE_RESULT_STALE_COPY
                if getattr(error, "code", None) == "stale_configuration"
                else profile_action_error_copy(error)
            )
            self.query_one("#audio-player-status", Static).update(copy)
            return
        finally:
            if self.is_mounted:
                self._sync_save_profile_action()
        self.query_one("#audio-player-status", Static).update("Voice profile saved.")

    def _sync_profile_preview_status(self) -> None:
        banner = self.query_one("#tts-profile-preview-status", Static)
        preset = self._profile_preset
        availability = self._profile_effective_availability
        if preset is None or availability is None:
            banner.add_class("hidden")
            banner.update("")
            return
        style_state = availability
        if availability == "unavailable":
            copy = (
                "Profile preview unavailable — return to Voice profiles and "
                "choose Edit."
            )
        elif (
            self._profile_preview_loading
            or self._profile_voice_validation_token is not None
        ):
            copy = "Profile preview loading — checking the exact saved selection."
            style_state = "loading"
        elif (
            blocked := self._profile_preview_blocked_presentation(preset)
        ) is not None:
            copy, style_state = blocked
        elif availability == "unverified":
            copy = (
                "Profile preview unverified — Generate makes one exact attempt "
                "without fallback."
            )
        else:
            copy = "Profile preview — exact saved selection."
        for state in ("loading", "available", "unverified", "unavailable"):
            banner.set_class(
                style_state == state,
                f"profile-preview-{state}",
            )
        banner.update(Text(copy))
        banner.remove_class("hidden")

    def _sync_save_profile_action(self) -> None:
        """Expose save only for an idle artifact with native provenance."""
        button = self.query_one("#audio-save-profile-btn", Button)
        artifact = self.current_audio_artifact
        eligible = bool(
            artifact is not None
            and artifact.profile_save_eligible
            and self._generation_operation_id is None
            and not self._profile_save_suppressed
        )
        button.set_class(not eligible, "hidden")
        button.disabled = not eligible
