"""TldwCli speech: TTS/STTS event handlers, speech owners and admission.

Moved from ``TldwCli`` in ``app.py`` (TASK-33011 PR-E): the TTS/STTS event
handler bodies, TTS profile/voice-bundle/audio.cpp resource owners and their
shutdown (cluster N), and speech initialization plus delivery admission
(U/U2). Each function takes the app as its first parameter; the bodies are
unchanged apart from ``self`` -> ``app``.

``TldwCli`` keeps a same-named stub for every function here, which imports
this module on first call (``app._speech``). The ``@on`` handlers keep their
decorators on those stubs: Textual dispatches only decorated methods of the
App class. Nothing imports this module at module scope, and every body here
runs after ``_ui_ready``, so it stays out of the ADR-097 UI-ready census.
A test that patches a module-level name one of these bodies reads must patch
it HERE, not on ``tldw_chatbook.app``.

Stay in ``app.py``: ``_bind_tts_service`` (``on_mount`` calls it, before
``_ui_ready``), and ``_ensure_tts_profile_repository``,
``_ensure_tts_profile_service`` and ``_ensure_tts_voice_bundle_service``:
``TTS/profile_source.py`` binds the configured profile source only when its
caller frame's code object IS ``TldwCli.<that method>.__code__``.
"""

# ADR-126: importing ``tldw_chatbook.app`` first runs its recovery fence
# (``admit_startup``) before any runtime import below.
from tldw_chatbook.app import TldwCli  # noqa: I001 -- the fence must import first

import asyncio
import time
from typing import TYPE_CHECKING

from textual.widgets import Markdown

from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Constants import TAB_SETTINGS
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSAudioBookGenerateEvent,
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
    STTSProviderConfigurationChanged,
    STTSSettingsSaveEvent,
)
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSEventHandler,
    TTSGlobalOverrideDecisionEvent,
    TTSMessageSpeechRequestEvent,
    TTSPlaybackEvent,
    TTSProgressEvent,
    TTSRequestEvent,
)
from tldw_chatbook.Metrics.metrics import log_histogram
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactLeaseCoordinator,
    AudioCppArtifactRemovalEvidence,
    AudioCppManagedConsumerIdentity,
    AudioCppModelLibraryObservationSnapshot,
    project_audio_cpp_artifact_removal_evidence,
)
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppSettingsConfig,
    project_audio_cpp_settings_config,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.TTS_Generation import close_tts_resources
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage

# chat_message_enhanced is deliberately NOT imported at module scope
# (TASK-21103): it pulls PIL and the textual_image package at import time.
# The two TTS event handlers that query it import it function-locally.
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

# TASK-21108: the payload class only -- importing it from
# `speech_tts_settings_panel` put that 5,600-line Textual widget module (and
# its fspicker/lab-status/voice-input subtrees) on the app import path for a
# frozen dataclass. `speech_tts_panel_types` re-exports into the panel, so
# this is the same class object the panel and its tests use.
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_panel_types import (
    SpeechTTSPanelDraftSnapshot,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

# Task-4 review round 2: `_offer_tts_global_override`'s confirmation dialog
# must name which configured-voice domain actually failed -- a per-character
# assignment, or the app-wide default voice profile (slice 3, task 4) --
# since the user is consenting to hear a different voice than the one they
# configured. Keyed by `CharacterTTSResolutionError.domain` /
# `TTSEventHandler.peek_global_override_voice_domain`'s bounded return
# value; `None` (unknown/expired token, or no handler bound) falls back to
# the domain-neutral entry below, which stays accurate for both without
# being vaguer than either domain's own precise copy.
_TTS_GLOBAL_OVERRIDE_PROMPT_COPY: dict[str | None, str] = {
    "character": (
        "The assigned character voice could not be resolved. "
        "Use the current global TTS voice for this message?"
    ),
    "default_profile": (
        "Your default voice profile could not be used. "
        "Use the current global TTS voice for this message?"
    ),
    None: (
        "Your configured voice could not be used for this message. "
        "Use the current global TTS voice instead?"
    ),
}


# --- TTS/STTS event handlers and speech resource owners (cluster N) ---


async def handle_tts_request_event(app, event: TTSRequestEvent) -> None:
    """Handle TTS generation request."""
    app.loguru_logger.info(
        f"TTS request received for text: '{event.text[:50]}...'"
    )
    handler = await app._ensure_tts_handler()
    if handler:
        await handler.handle_tts_request(event)
    else:
        app.loguru_logger.error("TTS handler not initialized")
        app.post_message(
            TTSCompleteEvent(
                message_id=event.message_id or "unknown",
                error="TTS service not available",
            )
        )


async def handle_tts_message_speech_request_event(
    app,
    event: TTSMessageSpeechRequestEvent,
) -> None:
    """Route a trusted Console snapshot without logging private content."""
    app.loguru_logger.info("Trusted Console speech request received")
    try:
        handler = await app._ensure_tts_handler()
    except asyncio.CancelledError:
        event.report_outcome(False)
        raise
    except Exception as error:
        app.loguru_logger.error(
            "TTS handler initialization failed "
            "(operation=trusted_console_speech, exception_category={})",
            type(error).__name__,
        )
        event.report_outcome(False)
        return
    if handler:
        try:
            await handler.handle_tts_request(event)
        except asyncio.CancelledError:
            event.report_outcome(False)
            raise
        except Exception as error:
            app.loguru_logger.error(
                "TTS handler request failed "
                "(operation=trusted_console_speech, exception_category={})",
                type(error).__name__,
            )
            event.report_outcome(False)
    else:
        app.loguru_logger.error(
            "TTS handler not initialized "
            "(operation=trusted_console_speech, "
            "outcome_code=handler_unavailable)"
        )
        try:
            app.post_message(
                TTSCompleteEvent(
                    message_id=event.message_id,
                    error="TTS service not available",
                )
            )
        except Exception as error:
            app.loguru_logger.error(
                "TTS unavailable notice failed "
                "(operation=trusted_console_speech, exception_category={})",
                type(error).__name__,
            )
        finally:
            event.report_outcome(False)


async def handle_tts_global_override_decision_event(
    app,
    event: TTSGlobalOverrideDecisionEvent,
) -> None:
    """Route one opaque character-speech fallback decision.

    Args:
        event: The accepted or rejected message-scoped fallback decision.
    """
    handler = await app._ensure_tts_handler()
    if handler:
        await handler.handle_tts_global_override_decision(event)
    else:
        app.loguru_logger.error(
            "TTS handler not initialized "
            "(operation=global_voice_fallback, "
            "outcome_code=handler_unavailable)"
        )


async def _offer_tts_global_override(app, token: str) -> None:
    """Prompt for one message-scoped global-voice fallback.

    The dialog's copy names the actual configured-voice domain that
    refused (a per-character assignment vs. the app-wide default voice
    profile) -- looked up, without consuming the token, from the
    issuing handler's still-pending state
    (`TTSEventHandler.peek_global_override_voice_domain`). Review
    round 2: the completion toast already used `event.error`'s
    domain-accurate copy; this dialog previously did not, and always
    said "character" even for a default-profile refusal on a message
    with no character context at all.
    """
    handler = getattr(app, "_tts_handler", None)
    voice_domain = (
        handler.peek_global_override_voice_domain(token)
        if handler is not None
        else None
    )
    message = _TTS_GLOBAL_OVERRIDE_PROMPT_COPY.get(
        voice_domain,
        _TTS_GLOBAL_OVERRIDE_PROMPT_COPY[None],
    )
    decision = False
    try:
        result = await app.push_screen_wait(
            ConfirmationDialog(
                title="Use global voice?",
                message=message,
                confirm_label="Use global",
                cancel_label="Cancel",
            )
        )
        decision = result is True
    except asyncio.CancelledError:
        app.post_message(TTSGlobalOverrideDecisionEvent(token, accepted=False))
        raise
    except Exception as error:
        app.loguru_logger.warning(
            "TTS global fallback prompt failed (exception_category={})",
            type(error).__name__,
        )
    app.post_message(TTSGlobalOverrideDecisionEvent(token, accepted=decision))


async def handle_tts_complete_event(app, event: TTSCompleteEvent) -> None:
    await TldwCli._settle_speech_delivery(
        app, event, lambda message: TldwCli._deliver_tts_complete_event(app, message)
    )


async def _deliver_tts_complete_event(app, event: TTSCompleteEvent) -> None:
    """Handle TTS generation completion."""
    from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import (  # noqa: PLC0415 - keeps PIL/textual_image off the boot path (TASK-21103)
        ChatMessageEnhanced,
    )

    app.loguru_logger.info(f"TTS complete for message {event.message_id}")
    playback_lifecycle = getattr(event, "playback_lifecycle", None)

    lifecycle_failure_completion = bool(
        event.error
        and playback_lifecycle is not None
        and playback_lifecycle.state == "failed"
    )
    if (
        playback_lifecycle is not None
        and not playback_lifecycle.is_current()
        and not lifecycle_failure_completion
    ):
        handler = getattr(app, "_tts_handler", None)
        discard = getattr(handler, "discard_stale_console_completion", None)
        if callable(discard):
            try:
                await discard(
                    event.message_id,
                    event.audio_file,
                    playback_lifecycle,
                )
            except Exception:
                playback_lifecycle.report_terminal("failed")
        else:
            playback_lifecycle.report_terminal("stopped")
        return

    if event.error:
        if playback_lifecycle is not None:
            playback_lifecycle.report("failed")
        app.notify(f"TTS failed: {event.error}", severity="error")
        # Update widget state back to idle on error
        try:
            if event.message_id:
                # Find the message widget and update state
                for message_widget in list(app.query(ChatMessage)) + list(
                    app.query(ChatMessageEnhanced)
                ):
                    if (
                        getattr(message_widget, "message_id_internal", None)
                        == event.message_id
                    ):
                        # Update TTS state to idle on error
                        if hasattr(message_widget, "update_tts_state"):
                            message_widget.update_tts_state("idle")
                        # Remove TTS generating class
                        text_widget = message_widget.query_one(
                            ".message-text", Markdown
                        )
                        text_widget.remove_class("tts-generating")
                        break
        except Exception as e:
            app.loguru_logger.error(f"Error updating message UI: {e}")
        # The Console transcript's action row renders from the screen's
        # `_console_speaking_message_id`, not from a legacy widget — on
        # failure it must be cleared too, or the row keeps "⏹ Stop
        # speech" with no speech to stop (TASK-15422).
        if playback_lifecycle is None:
            for screen in reversed(tuple(getattr(app, "screen_stack", ()))):
                if (
                    getattr(screen, "_console_speaking_message_id", None)
                    == event.message_id
                ):
                    screen._console_speaking_message_id = None
                    sync = getattr(screen, "_sync_native_console_chat_ui", None)
                    if callable(sync):
                        try:
                            await sync()
                        except Exception:
                            app.loguru_logger.error(
                                "Console speak-state resync failed after a "
                                "TTS error"
                            )
                    break
        if event.global_override_token is not None:
            app.run_worker(
                app._offer_tts_global_override(event.global_override_token),
                name="tts_global_voice_confirmation",
            )
    else:
        # Update widget state to ready with audio file
        if event.audio_file and event.audio_file.exists():
            if (
                playback_lifecycle is not None
                and not playback_lifecycle.is_current()
            ):
                return
            try:
                widget_found = False
                if event.message_id:
                    # Find the message widget and update state
                    for message_widget in list(app.query(ChatMessage)) + list(
                        app.query(ChatMessageEnhanced)
                    ):
                        if (
                            getattr(message_widget, "message_id_internal", None)
                            == event.message_id
                        ):
                            widget_found = True
                            # Update TTS state to ready with audio file
                            if hasattr(message_widget, "update_tts_state"):
                                message_widget.update_tts_state(
                                    "ready", event.audio_file
                                )
                            # Remove TTS generating class
                            try:
                                text_widget = message_widget.query_one(
                                    ".message-text", Markdown
                                )
                                text_widget.remove_class("tts-generating")
                            except Exception:
                                pass
                            break
                if widget_found:
                    # A legacy ChatMessage/ChatMessageEnhanced widget owns
                    # this message and exposes its own play control - let
                    # the user trigger playback explicitly rather than
                    # auto-playing underneath them.
                    app.notify(
                        "TTS audio ready - click play to listen",
                        severity="information",
                    )
                else:
                    # No legacy widget claims this message (e.g. Console,
                    # which has no per-message playback control), so
                    # there is nothing for the user to click - play the
                    # generated audio immediately instead of going silent.
                    accepted = TldwCli._post_speech_delivery(
                        app,
                        TTSPlaybackEvent(
                            action="play",
                            message_id=event.message_id,
                            playback_lifecycle=playback_lifecycle,
                        ),
                    )
                    if accepted is False and playback_lifecycle is not None:
                        playback_lifecycle.report("failed")
            except Exception as e:
                if playback_lifecycle is not None:
                    playback_lifecycle.report("failed")
                app.loguru_logger.error(f"Error playing audio: {e}")
                app.notify("Failed to play audio", severity="error")
        elif (
            playback_lifecycle is not None
            and playback_lifecycle.state == "generating"
        ):
            playback_lifecycle.report("failed")

        # Remove TTS generating class from message
        try:
            if event.message_id:
                for message_widget in list(app.query(ChatMessage)) + list(
                    app.query(ChatMessageEnhanced)
                ):
                    if (
                        getattr(message_widget, "message_id_internal", None)
                        == event.message_id
                    ):
                        text_widget = message_widget.query_one(
                            ".message-text", Markdown
                        )
                        text_widget.remove_class("tts-generating")
                        break
        except Exception as e:
            app.loguru_logger.error(f"Error updating message UI: {e}")


async def handle_tts_progress_event(app, event: TTSProgressEvent) -> None:
    """Handle TTS generation progress updates."""
    from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import (  # noqa: PLC0415 - keeps PIL/textual_image off the boot path (TASK-21103)
        ChatMessageEnhanced,
    )

    app.loguru_logger.debug(
        f"TTS progress for message {event.message_id}: {event.progress:.0%} - {event.status}"
    )

    try:
        if event.message_id:
            # Find the message widget and update progress
            for message_widget in list(app.query(ChatMessage)) + list(
                app.query(ChatMessageEnhanced)
            ):
                if (
                    getattr(message_widget, "message_id_internal", None)
                    == event.message_id
                ):
                    # Update TTS progress
                    if hasattr(message_widget, "update_tts_progress"):
                        message_widget.update_tts_progress(
                            event.progress, event.status
                        )
                    break
    except Exception as e:
        app.loguru_logger.error(f"Error updating TTS progress: {e}")


async def handle_tts_playback_event(app, event: TTSPlaybackEvent) -> None:
    """Handle TTS playback control."""
    if (
        event.action == "play"
        and getattr(app, "_speech_delivery_paused", False)
        and event not in getattr(app, "_speech_delivery_pending", set())
    ):
        event.report_outcome(False)
        return
    await app._settle_speech_delivery(event, app.control_tts_playback)


async def control_tts_playback(app, event: TTSPlaybackEvent) -> None:
    """Run playback control directly and preserve handler callback order."""
    try:
        if event.action == "play" and getattr(app, "_speech_delivery_paused", False):
            if event in getattr(app, "_speech_delivery_pending", set()):
                app._defer_speech_playback(event)
            else:
                event.report_outcome(False)
            return
        if event.action == "stop":
            deferred = getattr(app, "_speech_delivery_deferred", [])
            for pending in tuple(deferred):
                if event.message_id is None or pending.message_id == event.message_id:
                    deferred.remove(pending)
                    pending.report_outcome(False)
                    if pending.playback_lifecycle is not None:
                        pending.playback_lifecycle.report_terminal("stopped")
        handler = (
            getattr(app, "_tts_handler", None)
            if event.action in {"stop", "pause"}
            else None
        )
        if handler is None:
            handler = await app._ensure_tts_handler()
        if handler:
            await handler.handle_tts_playback(event)
        else:
            event.report_outcome(False)
            if event.playback_lifecycle is not None:
                event.playback_lifecycle.report_terminal("failed")
    except asyncio.CancelledError:
        event.report_outcome(False)
        raise
    except Exception:
        event.report_outcome(False)
        if event.playback_lifecycle is not None:
            event.playback_lifecycle.report_terminal("failed")


async def handle_stts_playground_generate_event(
    app, event: STTSPlaygroundGenerateEvent
) -> None:
    """Handle S/TT/S playground generation request."""
    app.loguru_logger.info(
        "S/TT/S generation request accepted for provider={}",
        event.request.provider_id,
    )
    handler = await app._ensure_stts_handler()
    if handler:
        handler.start_playground_generation(event)
    else:
        app.loguru_logger.error("S/TT/S handler not initialized")
        app.notify("S/TT/S service not available", severity="error")


async def handle_stts_settings_save_event(
    app, event: STTSSettingsSaveEvent
) -> None:
    """Handle S/TT/S settings save."""
    handler = getattr(app, "_stts_handler", None)
    if handler is None:
        handler = await app._ensure_stts_handler()
    if handler:
        await handler.handle_settings_save(event)


def handle_stts_provider_configuration_changed(
    app,
    event: STTSProviderConfigurationChanged,
) -> None:
    """Forward provider invalidation to the retained STTS handler."""
    try:
        TldwCli._deliver_stts_provider_configuration_changed(app, event)
    finally:
        getattr(app, "_speech_delivery_pending", set()).discard(event)


def _deliver_stts_provider_configuration_changed(
    app, event: STTSProviderConfigurationChanged
) -> None:
    handler = getattr(app, "_stts_handler", None)
    if handler is not None:
        handler.on_stts_provider_configuration_changed(event)
    if event.provider_id == "audio_cpp":
        from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

        current_screen = getattr(app, "screen", None)
        if current_screen is not None:
            for window in current_screen.query(LLMManagementWindow):
                window.refresh_model_library_observations()


async def handle_stts_audiobook_generate_event(
    app, event: STTSAudioBookGenerateEvent
) -> None:
    """Handle audiobook generation request."""
    handler = await app._ensure_stts_handler()
    if handler:
        await handler.handle_audiobook_generate(event)


async def _close_tts_service(app) -> None:
    """Close and unbind the application-owned TTS service once."""
    if not app._tts_binding_active:
        return
    try:
        await close_tts_resources()
    finally:
        app._tts_binding_active = False


def _saved_audio_cpp_managed_consumers(
    app,
) -> tuple[AudioCppManagedConsumerIdentity, ...]:
    """Project only exact managed identities from immutable saved Settings."""

    try:
        config = project_audio_cpp_settings_config(app.app_config)
    except (TypeError, ValueError):
        config = AudioCppSettingsConfig()
    return tuple(
        AudioCppManagedConsumerIdentity(
            recipe_id=package.recipe_id,
            recipe_revision=package.recipe_revision,
            model_id=package.public_model_id,
            managed_artifact=package.managed_artifact,
        )
        for package in config.guided_packages
        if package.managed_artifact is not None
    )


def _ensure_audio_cpp_artifact_lease_coordinator(
    app,
) -> AudioCppArtifactLeaseCoordinator:
    """Return the one app-owned coordinator over the shared artifact owner."""

    coordinator = app._audio_cpp_artifact_lease_coordinator
    if coordinator is None:
        coordinator = AudioCppArtifactLeaseCoordinator(
            managed_service(),
            saved_settings_snapshot=app._saved_audio_cpp_managed_consumers,
        )
        app._audio_cpp_artifact_lease_coordinator = coordinator
    return coordinator


def _audio_cpp_removal_settings_inputs(
    app,
) -> tuple[
    AudioCppSettingsConfig,
    AudioCppSettingsConfig | None,
    TTSPreferencesSnapshot,
    TTSPreferencesSnapshot | None,
]:
    """Read saved plus exact detached-or-mounted Speech/TTS draft state."""

    try:
        saved = project_audio_cpp_settings_config(app.app_config)
        saved_preferences = TTSPreferencesSnapshot.from_settings(app.app_config)
    except (TypeError, ValueError):
        raise ProfileRepositoryError("unavailable") from None

    draft_snapshot: SpeechTTSPanelDraftSnapshot | None = None
    stored_settings_state = False
    store = getattr(app, "screen_state_store", None)
    if store is not None:
        try:
            stored = store.restore(TAB_SETTINGS, app._current_runtime_identity())
        except Exception:
            raise ProfileRepositoryError("unavailable") from None
        if stored is not None:
            stored_settings_state = True
            if "speech_tts_panel_draft" in stored:
                candidate = stored["speech_tts_panel_draft"]
                if type(candidate) is not SpeechTTSPanelDraftSnapshot:
                    raise ProfileRepositoryError("unavailable")
                draft_snapshot = candidate
    if draft_snapshot is None and not stored_settings_state:
        current_screen = getattr(app, "screen", None)
        candidate = getattr(current_screen, "_speech_tts_draft_snapshot", None)
        if type(candidate) is SpeechTTSPanelDraftSnapshot:
            draft_snapshot = candidate

    if draft_snapshot is None:
        return saved, None, saved_preferences, None
    try:
        provider = draft_snapshot.state.providers.get("audio_cpp")
        if not isinstance(provider, dict):
            raise ValueError
        draft = AudioCppSettingsConfig.from_mapping(provider)
        draft_preferences = draft_snapshot.state.defaults.snapshot()
    except (TypeError, ValueError):
        raise ProfileRepositoryError("unavailable") from None
    return saved, draft, saved_preferences, draft_preferences


async def _audio_cpp_model_library_observation_snapshot(
    app,
    references: tuple["ArtifactRef", ...],
) -> AudioCppModelLibraryObservationSnapshot:
    """Collect shared evidence once, then project every exact package ref."""

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    if type(references) is not tuple or any(
        type(reference) is not ArtifactRef for reference in references
    ):
        raise TypeError("references must be a tuple of ArtifactRef values")
    if len(set(references)) != len(references):
        raise ValueError("references must be unique")
    if not references:
        return AudioCppModelLibraryObservationSnapshot(())

    saved, draft, saved_preferences, draft_preferences = (
        app._audio_cpp_removal_settings_inputs()
    )

    profile_service = await app._ensure_tts_profile_service()
    if profile_service is None:
        raise ProfileRepositoryError("unavailable")
    profiles_with_counts = (
        await profile_service.bounded_profile_assignment_snapshot()
    )

    try:
        configuration = (
            await app.tts_service.registry.provider_configuration_snapshot(
                "audio_cpp"
            )
        )
        staged_config = (
            None
            if configuration.staged_config is None
            else AudioCppSettingsConfig.from_mapping(configuration.staged_config)
        )
        applied_config = AudioCppSettingsConfig.from_mapping(
            configuration.applied_config
        )
        supervisor = getattr(app.tts_service, "_audio_cpp_supervisor", None)
        admission = None if supervisor is None else supervisor.admission_snapshot()
    except Exception:
        # Runtime evidence is safety-relevant; fail closed without exposing
        # collaborator details through the removal review.
        raise ProfileRepositoryError("unavailable") from None

    def contains(
        config: AudioCppSettingsConfig | None,
        reference: ArtifactRef,
    ) -> bool:
        return config is not None and any(
            package.managed_artifact is not None
            and (
                package.managed_artifact.artifact_id,
                package.managed_artifact.revision,
                package.managed_artifact.variant,
            )
            == (reference.artifact_id, reference.revision, reference.variant)
            for package in config.guided_packages
        )

    live = admission is not None and admission.state in {
        "starting",
        "running",
        "draining",
        "stopping",
    }
    return AudioCppModelLibraryObservationSnapshot(
        tuple(
            project_audio_cpp_artifact_removal_evidence(
                reference,
                saved_settings=saved,
                draft_settings=draft,
                saved_preferences=saved_preferences,
                draft_preferences=draft_preferences,
                profiles=profiles_with_counts,
                staged_runtime_ids=(
                    (f"settings-generation-{configuration.staged_generation}",)
                    if contains(staged_config, reference)
                    else ()
                ),
                live_runtime_ids=(
                    (f"process-generation-{admission.process_generation}",)
                    if live and contains(applied_config, reference)
                    else ()
                ),
            )
            for reference in references
        )
    )


async def _audio_cpp_artifact_removal_evidence(
    app,
    reference: "ArtifactRef",
) -> AudioCppArtifactRemovalEvidence:
    """Collect Task 9 removal evidence through the shared bulk snapshot."""

    snapshot = await TldwCli._audio_cpp_model_library_observation_snapshot(
        app,
        (reference,),
    )
    return snapshot.observations[0]


async def _close_tts_voice_bundle_service(app) -> None:
    """Close and join portability before repository authority is released."""

    service = getattr(app, "_tts_voice_bundle_service", None)
    if service is None:
        return
    close_task = getattr(app, "_tts_voice_bundle_service_close_task", None)
    if close_task is None:

        async def close_portability() -> None:
            await service.close()
            await service.wait_closed()

        close_task = asyncio.create_task(
            close_portability(),
            name="close_tts_voice_bundle_service",
        )
        app._tts_voice_bundle_service_close_task = close_task
    await join_retained_task(close_task)


async def _close_tts_profile_repository(app) -> None:
    """Definitively close the app-owned profile repository once."""

    app._tts_profile_repository_close_requested = True
    repository = getattr(app, "_tts_profile_repository", None)
    if repository is None:
        return

    close_task = getattr(app, "_tts_profile_repository_close_task", None)
    if close_task is None:

        async def close_repository() -> None:
            await repository.close()

        close_task = asyncio.create_task(
            close_repository(),
            name="close_tts_profile_repository",
        )
        app._tts_profile_repository_close_task = close_task

    def record_failure_after_cancellation(
        cancellation: BaseException,
        cleanup_error: BaseException,
    ) -> None:
        cancellation.add_note(
            "TTS profile repository cleanup also failed while preserving "
            "shutdown cancellation"
        )
        app.loguru_logger.warning(
            "TTS profile repository phase=close failed while preserving "
            f"cancellation type={type(cleanup_error).__name__} "
            "code=operation_failed"
        )

    await join_retained_task(
        close_task,
        on_failure_after_cancellation=record_failure_after_cancellation,
    )


async def _close_owned_tts_resources(app) -> None:
    """Close app-owned TTS resources without masking cancellation."""

    failures: list[tuple[str, BaseException]] = []
    buddy_speech = getattr(app, "buddy_speech_coordinator", None)
    if buddy_speech is not None:
        try:
            close_task = getattr(app, "_buddy_speech_close_task", None)
            if close_task is None:
                close_task = asyncio.create_task(
                    buddy_speech.aclose(), name="close_buddy_speech"
                )
                app._buddy_speech_close_task = close_task
            await join_retained_task(close_task)
        except BaseException as buddy_close_error:  # noqa: BLE001 - drain all owners before preserving cancellation
            failures.append(("buddy_speech", buddy_close_error))
    if hasattr(app, "_close_tts_voice_bundle_service"):
        try:
            await app._close_tts_voice_bundle_service()
        except BaseException as portability_close_error:
            failures.append(("voice_bundle_service", portability_close_error))

    try:
        await app._close_tts_profile_repository()
    except BaseException as profile_close_error:
        failures.append(("profile_repository", profile_close_error))

    try:
        await app._close_tts_service()
    except BaseException as service_close_error:
        failures.append(("tts_service", service_close_error))

    if not failures:
        return

    control_flow_failures = [
        failure for failure in failures if not isinstance(failure[1], Exception)
    ]
    primary_phase, primary_error = (
        control_flow_failures[0] if control_flow_failures else failures[0]
    )
    for phase, failure_error in failures:
        if failure_error is primary_error:
            continue
        primary_error.add_note(
            "TTS owner cleanup also failed while preserving the primary error"
        )
        app.loguru_logger.warning(
            f"TTS owner cleanup phase={phase} failed while preserving "
            f"phase={primary_phase} type={type(failure_error).__name__} "
            "code=operation_failed"
        )
    raise primary_error


# --- Speech initialization and delivery admission (U/U2) ---


def _start_deferred_audio_service_initialization(app) -> None:
    """Kick off TTS/STTS initialization after startup readiness."""

    app._schedule_tts_initialization()
    app._schedule_stts_initialization()


def _schedule_tts_initialization(app) -> None:
    if not app._speech_initialization_allowed("tts"):
        return
    if app._tts_handler is not None:
        return
    if app._tts_initialization_task and not app._tts_initialization_task.done():
        return
    app._tts_initialization_task = app._create_deferred_startup_task(
        app._initialize_tts_service(),
        name="deferred_tts_initialization",
    )


def _schedule_stts_initialization(app) -> None:
    if not app._speech_initialization_allowed("stts"):
        return
    if app._stts_handler is not None:
        return
    if app._stts_initialization_task and not app._stts_initialization_task.done():
        return
    app._stts_initialization_task = app._create_deferred_startup_task(
        app._initialize_stts_service(),
        name="deferred_stts_initialization",
    )


def _speech_initialization_allowed(app, kind: str) -> bool:
    if getattr(app, "_speech_initialization_closed", False):
        return False
    if not getattr(app, "_speech_initialization_paused", False):
        return True
    deferred = getattr(app, "_speech_initialization_deferred", None)
    if deferred is None:
        deferred = app._speech_initialization_deferred = set()
    deferred.add(kind)
    return False


def _speech_delivery_close_admission(app) -> None:
    """Keep accepted notifications; defer playback until producer resume."""
    app._speech_delivery_paused = True


async def _speech_delivery_drain(app, deadline: float) -> bool:
    """Settle queued publication delivery; retained autoplay is transient."""
    if not getattr(app, "_speech_delivery_paused", False):
        raise RuntimeError("speech_delivery_not_paused")
    while getattr(app, "_speech_delivery_pending", set()):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(remaining, 0.01))
    return True


def _speech_delivery_resume(app) -> None:
    """Replay accepted playback after handler and service admission reopen."""
    app._speech_delivery_paused = False
    deferred = getattr(app, "_speech_delivery_deferred", [])
    while deferred:
        event = deferred[0]
        accepted = app._post_speech_delivery(event)
        deferred.pop(0)
        if accepted is False:
            event.report_outcome(False)
            if event.playback_lifecycle is not None:
                event.playback_lifecycle.report_terminal("failed")


def _defer_speech_playback(app, event: TTSPlaybackEvent) -> None:
    deferred = getattr(app, "_speech_delivery_deferred", None)
    if deferred is None:
        deferred = app._speech_delivery_deferred = []
    if event not in deferred:
        deferred.append(event)


def _post_speech_delivery(app, event) -> bool:
    """Retain only the installed speech completion/notification routes."""
    if type(event) not in {
        TTSCompleteEvent, TTSPlaybackEvent, STTSProviderConfigurationChanged
    }:
        raise TypeError("unsupported_speech_delivery")
    if (
        type(event) is TTSPlaybackEvent
        and event.action == "play"
        and getattr(app, "_speech_delivery_paused", False)
    ):
        app._defer_speech_playback(event)
        return True
    pending = getattr(app, "_speech_delivery_pending", None)
    if pending is None:
        pending = app._speech_delivery_pending = set()
    pending.add(event)
    try:
        accepted = app.post_message(event)
    except BaseException:
        pending.discard(event)
        raise
    if accepted is False:
        pending.discard(event)
    return accepted


async def _settle_speech_delivery(app, event, deliver) -> None:
    pending = getattr(app, "_speech_delivery_pending", None)
    if pending is None:
        pending = app._speech_delivery_pending = set()
    pending.add(event)
    completion = asyncio.create_task(deliver(event))
    cancellation = None
    try:
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        completion.result()
    finally:
        pending.discard(event)
    if cancellation is not None:
        raise cancellation


def _speech_initialization_close_admission(app) -> None:
    """Defer new service construction while admitted initialization settles."""
    app._speech_initialization_paused = True


async def _speech_initialization_drain(app, deadline: float) -> bool:
    if not getattr(app, "_speech_initialization_paused", False):
        raise RuntimeError("speech_initialization_not_paused")
    while getattr(app, "_speech_initialization_children", {}):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(remaining, 0.01))
    return True


def _speech_initialization_resume(app) -> None:
    """Replay deferred construction only after ordinary storage resumes."""
    app._speech_initialization_paused = False
    deferred = getattr(app, "_speech_initialization_deferred", set())
    app._speech_initialization_deferred = set()
    if "tts" in deferred:
        app._schedule_tts_initialization()
    if "stts" in deferred:
        app._schedule_stts_initialization()


async def _settle_speech_initialization(app) -> asyncio.CancelledError | None:
    """Finish admitted initialization before shutdown cleans its handlers.

    Return waiter cancellation so the existing cleanup phase can preserve
    it until handler retirement; cancelling a wrapper never detaches its
    native initializer. Intake must already be terminally closed.
    """
    if not getattr(app, "_speech_initialization_closed", False):
        raise RuntimeError("speech_initialization_not_closed")
    children = tuple(getattr(app, "_speech_initialization_children", {}).values())
    if not children:
        return None
    completion = asyncio.gather(*children, return_exceptions=True)
    cancellation = None
    while not completion.done():
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    completion.result()
    return cancellation


async def _run_speech_initialization(app, kind: str, initialize):
    if not app._speech_initialization_allowed(kind):
        return None
    children = getattr(app, "_speech_initialization_children", None)
    if children is None:
        children = app._speech_initialization_children = {}
    task = children.get(kind)
    if task is None:

        async def admitted_initialize():
            from tldw_chatbook.Backup_Recovery.activation import execution_scope

            owners = (
                "config",
                "models.artifacts",
                "tts.profile_store",
                "tts.voices",
            )
            with execution_scope(owners) as allowed:
                if not allowed:
                    return None
                return await initialize()

        task = asyncio.create_task(admitted_initialize())
        children[kind] = task

        def settled(completed):
            if children.get(kind) is completed:
                children.pop(kind)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(settled)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # App shutdown may cancel the deferred wrapper. Its existing cleanup
        # must see the published handler before it retires handler resources.
        completion = asyncio.gather(task, return_exceptions=True)
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError:
                continue
        completion.result()
        raise


async def _initialize_tts_service(app):
    return await app._run_speech_initialization(
        "tts", app._initialize_tts_service_owned
    )


async def _initialize_tts_service_owned(app):
    """Initialize the TTS handler outside the startup critical path."""

    phase_start = time.perf_counter()
    try:
        app.loguru_logger.info("Initializing TTS service...")
        handler = TTSEventHandler(
            profile_service_loader=app._ensure_tts_profile_service,
            default_profile_id_reader=(
                lambda: get_cli_setting("app_tts", "default_profile_id", None)
            ),
        )
        handler.app = app
        await handler.initialize_tts()
        app._tts_handler = handler
        app.loguru_logger.info("TTS service initialized successfully")
    except Exception as e:
        app.loguru_logger.error(f"Failed to initialize TTS service: {e}")
        app._tts_handler = None
    finally:
        log_histogram(
            "app_post_mount_phase_duration_seconds",
            time.perf_counter() - phase_start,
            labels={"phase": "tts_init_deferred"},
            documentation="Duration of post-mount phase in seconds",
        )
    return app._tts_handler


async def _initialize_stts_service(app):
    return await app._run_speech_initialization(
        "stts", app._initialize_stts_service_owned
    )


async def _initialize_stts_service_owned(app):
    """Initialize the S/TT/S handler outside the startup critical path."""

    phase_start = time.perf_counter()
    try:
        app.loguru_logger.info("Initializing S/TT/S service...")
        handler = STTSEventHandler(app=app)
        await handler.initialize_stts()
        app._stts_handler = handler
        app.loguru_logger.info("S/TT/S service initialized successfully")
    except Exception as e:
        app.loguru_logger.error(f"Failed to initialize S/TT/S service: {e}")
        app._stts_handler = None
    finally:
        log_histogram(
            "app_post_mount_phase_duration_seconds",
            time.perf_counter() - phase_start,
            labels={"phase": "stts_init_deferred"},
            documentation="Duration of post-mount phase in seconds",
        )
    return app._stts_handler


async def _ensure_tts_handler(app):
    """Return an initialized TTS handler, initializing on first use if needed."""

    if not app._speech_initialization_allowed("tts"):
        return None
    if app._tts_handler is not None:
        return app._tts_handler
    if app._tts_initialization_task and not app._tts_initialization_task.done():
        await app._tts_initialization_task
        return app._tts_handler
    return await app._initialize_tts_service()


async def _ensure_stts_handler(app):
    """Return an initialized S/TT/S handler, initializing on first use if needed."""

    if not app._speech_initialization_allowed("stts"):
        return None
    if app._stts_handler is not None:
        return app._stts_handler
    if app._stts_initialization_task and not app._stts_initialization_task.done():
        await app._stts_initialization_task
        return app._stts_handler
    return await app._initialize_stts_service()
