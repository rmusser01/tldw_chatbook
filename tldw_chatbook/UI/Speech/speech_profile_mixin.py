"""Saved TTS profiles, shared by whatever hosts the Playground controls.

dev built this onto the legacy playground widget while the Console-grammar
rebuild was in flight -- a profile library, and the ability to save the
current result as a profile and preview an exact one. Retiring that widget
would have deleted the UI half of it, so the behaviour moves here instead
and the rebuilt pane inherits it, exactly as the catalog and synthesis
paths did.

Ported rather than reimplemented: these are dev's methods verbatim. They
query their controls by id, so the pane mounts the same ids
(`audio-save-profile-btn`, `tts-profile-preview-status`).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from uuid import UUID

from rich.text import Text
from textual.message import Message
from textual.widgets import Button, Select, Static

from tldw_chatbook.Constants import TAB_PERSONAS
from tldw_chatbook.TTS import (
    LoadedTTSProfile,
    ProfileAvailabilityState,
    STTSGeneratedAudio,
    STTSPlaygroundResultProjection,
    TTSPlaygroundSelectionPreset,
    TTSProfileAvailabilitySnapshot,
    TTSProfilePageSnapshot,
)
from tldw_chatbook.TTS.adapter_types import TTSRegistryClosedError
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Speech.speech_axis_row import axis_chip_id
from tldw_chatbook.UI.stts_playground_catalog import (
    AUDIO_CPP_PROVIDER_ID,
    CatalogRequestToken,
    controls_from_profile_preset,
    preset_has_no_catalog_check,
)
from tldw_chatbook.UI.stts_profile_library import (
    PROFILE_ACTION_FAILED_COPY,
    PROFILE_STORE_UNAVAILABLE_COPY,
    ProfileTestVerified,
    ProfileVerificationResult,
    TTSCloneProfileSaveReview,
    TTSCloneProfileSaveReviewModal,
    TTSProfileNameModal,
    _consume_profile_test_context,
    _resolve_profile_test_context,
    _retire_profile_test_context,
    profile_action_error_copy,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_tts_widget import (
    CharacterTTSProfileSuggestion,
)

#: Copy shown when the generated audio predates the current settings, so
#: saving it as a profile would record something the user did not hear.
#: Kept verbatim from `STTS_Window`, where dev defined it.
_PROFILE_RESULT_STALE_COPY = (
    "TTS settings changed after this audio was generated. Generate a new "
    "result before saving it as a profile."
)


class AdoptStudioPreferencesRequested(Message):
    """Hand an explicitly adopted preview to the Studio preference editor."""

    def __init__(self, preset: TTSPlaygroundSelectionPreset) -> None:
        super().__init__()
        if type(preset) is not TTSPlaygroundSelectionPreset:
            raise TypeError("Studio adoption requires an exact profile preset")
        self.preset = preset


class SpeechProfileMixin:
    """Profile save/preview behaviour, independent of the layout."""

    if TYPE_CHECKING:
        # This class is composed with the catalog/synthesis mixins and a
        # Textual widget by ``SpeechPlaygroundPane``.  Keep the host contract
        # visible to static analysis without adding runtime members or
        # duplicating the concrete pane's implementation.
        is_mounted: bool
        app: Any
        _tts_service: Any
        _catalogs: dict[Any, Any]
        _catalog_generation_allowed: bool
        _discovered_voices: dict[Any, Any]
        current_audio_artifact: Any
        _generation_operation_id: Any

        def query_one(self, *args: Any, **kwargs: Any) -> Any: ...

        def _sync_generate_enabled(self) -> None: ...

        def _safe_select_options(self, *args: Any, **kwargs: Any) -> Any: ...

        def _refresh_axis_markers(self) -> None: ...

        def _show_provider_specific_controls(self, provider_id: str) -> None: ...

        def _apply_controls(self, controls: Any) -> None: ...

    def init_profile_state(
        self,
        profile_preset: Any = None,
        profile_context_token: UUID | None = None,
    ) -> None:
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
        #: `_apply_controls` alone: the legacy playground widget's
        #: constructor set the same attribute, and this pane's preset path
        #: is production-reachable now that `STTS_Window` mounts it.
        self._profile_controls_applied = profile_preset is None
        #: Catalog revision the preview was admitted against.
        self._profile_configuration_revision: int | None = None
        #: Set while saving is deliberately unavailable.
        self._profile_save_suppressed = False
        #: The open name modal, so a second one is not stacked.
        self._active_profile_name_modal: Any = None
        self._profile_test_context_token = profile_context_token
        self._profile_test_context = (
            _resolve_profile_test_context(profile_context_token, profile_preset)
            if type(profile_preset) is TTSPlaygroundSelectionPreset
            else None
        )
        self._profile_exact_artifacts: dict[str, STTSGeneratedAudio] = {}
        self._profile_matching_artifact_operation: str | None = None
        self._profile_evidence_generation = 0
        self._profile_test_error = False

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
    def _dismiss_profile_name_modal(modal: Any) -> None:
        if modal.is_mounted and modal.is_current:
            modal.dismiss(None)

    def _end_profile_preset(self, *, before_controls: bool = False) -> bool:
        """Detach exact profile semantics after a user selection edit."""
        if self._profile_preset is None:
            return False
        if not before_controls and not self._profile_controls_applied:
            return False
        self._retire_profile_test_authority()
        self._profile_preset = None
        self._profile_effective_availability = None
        self._profile_preview_loading = False
        self._profile_configuration_revision = None
        self._profile_voice_validation_token = None
        self._profile_controls_applied = True
        self._profile_test_context = None
        self._profile_exact_artifacts.clear()
        self._profile_matching_artifact_operation = None
        self._profile_evidence_generation += 1
        self._profile_test_error = False
        self._sync_profile_preview_status()
        self._sync_generate_enabled()
        return True

    def _retire_profile_test_authority(self) -> None:
        """Release this pane's active context without touching replacements."""

        _retire_profile_test_context(self._profile_test_context_token)
        self._profile_test_context_token = None
        self._profile_test_context = None

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
        """Return bounded recovery copy when the exact preset cannot generate.

        TASK-2952 branch trace -- the three "unverified"-styled returns below
        promise a refresh/retry recovery with no check on `preset`'s provider
        class. Each was traced (and confirmed live, `test_speech_playground_
        pane.py`) for whether that promise is a false one on the six
        legacy-bridge providers, the way slice 2 task 3 already fixed the two
        adjacent adoption sites in `speech_catalog_mixin.py`:

        `expected_revision is None` -- requires `_profile_preview_loading`
        False while `_profile_configuration_revision` was never set for the
        preset's own provider. `_load_provider_catalog_worker` sets that
        field synchronously, before any `await`, the moment it targets
        `preset.provider_id` -- and production always targets it on the
        pane's very first load: `initialize`'s provider selection prefers
        `preset_provider` whenever it is registered, and both provider
        classes are registered unconditionally
        (`adapter_bootstrap.build_default_tts_service`,
        `legacy_bridge.legacy_provider_specs` register all six regardless of
        config/deps). `_profile_preset` is also set once at construction and
        never reassigned (`init_profile_state`/`_end_profile_preset` are the
        only writers), so there is no "preset swapped after mount" path
        either. LEGACY-UNREACHABLE -- and audio.cpp-unreachable too, so this
        is not a class distinction to fix. `test_adopted_preset_preview_
        never_shows_a_null_revision_refresh` pins the invariant this relies
        on for both classes.

        `current_revision != expected_revision` -- fires when the
        provider's registry configuration revision changes after the
        preview loaded (a genuine settings edit on that exact provider, or
        `mark_provider_configuration_changed`/`_mark_stale_catalog_result`
        noticing the same drift). This is pure registry bookkeeping
        (`TTSAdapterRegistry.configuration_revision`), not catalog content --
        refresh re-reads the revision and re-syncs it for any provider,
        legacy included. HONEST FOR BOTH CLASSES: confirmed live that
        bumping a legacy provider's revision produces this exact banner, and
        pressing Refresh resolves it into the "no catalog check" copy just
        as it resolves audio.cpp's into its "unverified" copy
        (`test_adopted_preset_preview_revision_mismatch_refresh_recovers_
        for_both_classes`).

        `not self._catalog_generation_allowed` -- the one live path found is
        `_load_provider_voices_worker`'s `TTSRegistryClosedError`
        (non-reconfiguring) branch, which forces this flag False without an
        `_apply_controls` recompute; the generic-exception and reconfiguring
        branches both self-heal through `_apply_catalog`/`_apply_controls`
        before this state could be observed. `TTSRegistryClosedError` comes
        from `TTSAdapterRegistry._closed` -- a registry-wide, one-way seal
        (`adapter_registry.py: close()`) -- identical machinery for
        audio.cpp and every legacy provider. No legacy-specific divergence
        found, so no fix belongs here; pinned as a cross-class symmetry
        check (`test_adopted_preset_preview_registry_closed_during_voice_
        fetch_is_symmetric`).
        """
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
        self._mark_profile_test_axes()
        return True

    def _mark_profile_test_axes(self) -> None:
        """Identify the exact profile-owned axes without exposing stored values."""

        preset = self._profile_preset
        if preset is None or preset.profile_id is None:
            return
        for axis in (
            "tts-provider-select",
            "tts-model-select",
            "tts-voice-select",
            "tts-format-select",
            "tts-speed-input",
        ):
            try:
                label = self.query_one(f"#{axis_chip_id(axis)}", Static)
            except Exception:  # noqa: BLE001 - axis may not be mounted yet
                continue
            label.tooltip = "Profile test selection — session only"
            label.add_class("profile-test-source")

    def _retain_profile_generation_artifact(self, artifact: object) -> None:
        """Retain one exact artifact only for the active profile-test operation."""

        if (
            type(artifact) is not STTSGeneratedAudio
            or self._profile_preset is None
            or self._profile_test_context is None
        ):
            return
        exact = artifact
        self._profile_exact_artifacts = {exact.operation_id: exact}

    def _handle_profile_generation_result(self, artifact: object) -> None:
        """Verify one accepted result against its exact profile test context."""

        preset = self._profile_preset
        if preset is None:
            return
        self._profile_matching_artifact_operation = None
        self._profile_test_error = False
        self._profile_evidence_generation += 1
        generation = self._profile_evidence_generation
        if type(artifact) is not STTSPlaygroundResultProjection:
            self._profile_exact_artifacts.clear()
            self._retire_profile_test_authority()
            self._sync_profile_preview_status()
            self._sync_save_profile_action()
            return
        projection = artifact
        exact = self._profile_exact_artifacts.pop(projection.operation_id, None)
        context = _consume_profile_test_context(
            self._profile_test_context_token,
            preset,
        )
        self._profile_test_context_token = None
        self._profile_test_context = None
        if exact is None or context is None:
            self._profile_exact_artifacts.clear()
            self._sync_profile_preview_status()
            self._sync_save_profile_action()
            return
        self.run_worker(
            self._verify_profile_sample(
                preset,
                context,
                exact,
                projection.operation_id,
                generation,
            ),
            name="verify_voice_profile_sample",
            group="verify_voice_profile_sample",
            exclusive=True,
            exit_on_error=False,
        )

    async def _verify_profile_sample(
        self,
        preset: TTSPlaygroundSelectionPreset,
        context: Any,
        artifact: STTSGeneratedAudio,
        operation_id: str,
        generation: int,
    ) -> None:
        """Record and re-observe evidence with profile/edit/delete race fences."""

        try:
            await asyncio.to_thread(
                context.service.record_sample_evidence,
                context.loaded,
                artifact,
            )
            page = TTSProfilePageSnapshot(
                repository_generation=context.loaded.repository_generation,
                profiles=(context.loaded.profile,),
                total=1,
            )
            snapshot = await context.service.observe_availability(page)
            if type(snapshot) is not TTSProfileAvailabilitySnapshot:
                raise TypeError("invalid profile availability")
            availability = snapshot.profiles[0] if len(snapshot.profiles) == 1 else None
            verified = bool(
                snapshot.repository_generation == context.loaded.repository_generation
                and availability is not None
                and availability.profile_id == context.loaded.profile.profile_id
                and availability.state == "available"
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - provider/profile values stay private
            verified = False
            availability = None

        current = self.current_audio_artifact
        if (
            not self.is_mounted
            or self._profile_evidence_generation != generation
            or self._profile_preset is not preset
            or current is None
            or current.operation_id != operation_id
        ):
            return
        if verified:
            self._profile_effective_availability = "available"
            self._profile_preview_loading = False
            self._profile_voice_validation_token = None
            self._profile_matching_artifact_operation = operation_id
            if availability is not None:
                self.post_message(
                    ProfileTestVerified(
                        ProfileVerificationResult(
                            profile_id=context.loaded.profile.profile_id,
                            repository_generation=(
                                context.loaded.repository_generation
                            ),
                            profile_revision=context.loaded.profile.revision,
                            availability=availability,
                        )
                    )
                )
        else:
            self._profile_test_error = True
        self._sync_profile_preview_status()
        self._sync_save_profile_action()

    async def _save_current_result_as_profile(self) -> None:
        """Save a captured eligible artifact without rereading selectors."""
        artifact = self.current_audio_artifact
        if (
            artifact is None
            or not artifact.profile_save_eligible
            or self._generation_operation_id is not None
            or self._profile_save_suppressed
            or self._profile_preset is not None
            and artifact.operation_id != self._profile_matching_artifact_operation
        ):
            self._sync_save_profile_action()
            return

        clone_save = artifact.clone_profile_save_eligible
        modal = (
            TTSCloneProfileSaveReviewModal()
            if clone_save
            else TTSProfileNameModal()
        )
        active = self._active_profile_name_modal
        if active is not None:
            self._dismiss_profile_name_modal(active)
        self._active_profile_name_modal = modal
        try:
            modal_result = await self.app.push_screen_wait(modal)
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
        if clone_save:
            if type(modal_result) is not TTSCloneProfileSaveReview:
                self._sync_save_profile_action()
                return
            display_name = modal_result.display_name
            choose_character = modal_result.choose_character
        else:
            if not isinstance(modal_result, str) or not modal_result.strip():
                self._sync_save_profile_action()
                return
            display_name = modal_result
            choose_character = False
        if not display_name.strip():
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
            handler = getattr(self.app, "_stts_handler", None)
            save_current = getattr(handler, "save_current_playground_profile", None)
            if not callable(save_current):
                raise RuntimeError("The current speech result is unavailable")
            saved = await save_current(artifact.operation_id, display_name, service)
            if clone_save and type(saved) is not LoadedTTSProfile:
                raise RuntimeError("The saved voice profile result is unavailable")
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
        if clone_save and choose_character:
            assert type(saved) is LoadedTTSProfile
            profile = saved.profile
            self.app.post_message(
                NavigateToScreen(
                    TAB_PERSONAS,
                    {
                        "view": "characters",
                        "voice_profile_suggestion": CharacterTTSProfileSuggestion(
                            profile_id=profile.profile_id,
                            repository_generation=saved.repository_generation,
                            profile_revision=profile.revision,
                        ),
                    },
                )
            )

    def _sync_profile_preview_status(self) -> None:
        banner = self.query_one("#tts-profile-preview-status", Static)
        adopt = self.query_one("#tts-adopt-studio-preferences-btn", Button)
        preset = self._profile_preset
        availability = self._profile_effective_availability
        if preset is None or availability is None:
            banner.add_class("hidden")
            banner.update("")
            adopt.add_class("hidden")
            adopt.disabled = True
            return
        style_state = availability
        is_profile_test = preset.profile_id is not None
        if self._profile_test_error and is_profile_test:
            copy = "Testing voice profile — Needs test. The sample could not be verified."
            style_state = "unverified"
        elif availability == "unavailable":
            copy = (
                "Profile preview unavailable — return to Voice profiles and "
                "choose Edit."
            )
        elif (
            availability == "available"
            and is_profile_test
            and self._profile_matching_artifact_operation is not None
        ):
            copy = "Testing voice profile — Verified by this sample."
        elif availability == "unverified" and is_profile_test:
            copy = (
                "Testing voice profile — Needs test. Generate one exact sample "
                "without fallback."
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
            if preset_has_no_catalog_check(preset):
                copy = (
                    "This provider has no catalog check. Generate makes one "
                    "exact attempt without fallback."
                )
            else:
                copy = (
                    "Profile preview unverified — Generate makes one exact "
                    "attempt without fallback."
                )
        else:
            copy = (
                "Testing voice profile — Verified by this sample."
                if is_profile_test
                else "Profile preview — exact saved selection."
            )
        for state in ("loading", "available", "unverified", "unavailable"):
            banner.set_class(
                style_state == state,
                f"profile-preview-{state}",
            )
        banner.update(Text(copy))
        banner.remove_class("hidden")
        can_adopt = availability != "unavailable"
        adopt.set_class(not can_adopt, "hidden")
        adopt.disabled = not can_adopt

    def _sync_save_profile_action(self) -> None:
        """Expose save only for an idle artifact with native provenance."""
        button = self.query_one("#audio-save-profile-btn", Button)
        artifact = self.current_audio_artifact
        eligible = bool(
            artifact is not None
            and artifact.profile_save_eligible
            and self._generation_operation_id is None
            and not self._profile_save_suppressed
            and (
                self._profile_preset is None
                or artifact.operation_id == self._profile_matching_artifact_operation
            )
        )
        button.set_class(not eligible, "hidden")
        button.disabled = not eligible
