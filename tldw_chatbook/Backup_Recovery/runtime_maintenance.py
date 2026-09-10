"""Reversible settlement of the app's installed live-capture producers."""

import asyncio
import inspect
import logging
import sys
import time
from dataclasses import dataclass

from .bootstrap import RecoveryRequired
from .rag_definition_participant import participant as definition_participant
from .rag_definition_participant import retained_issues


@dataclass(frozen=True)
class _Hook:
    owner: object
    close: object
    drain: object
    resume: object


def _bind(owner, module, name, prefix="_maintenance"):
    """Bind original methods of an existing installed owner, without creating it."""
    if owner is None:
        return None
    loaded = sys.modules.get("tldw_chatbook." + module)
    expected = None if loaded is None else vars(loaded).get(name)
    if expected is None or type(owner) is not expected:
        raise RecoveryRequired("runtime_owner_unqualified")
    try:
        return _Hook(
            owner,
            getattr(expected, prefix + "_close_admission"),
            getattr(expected, prefix + "_drain"),
            getattr(expected, prefix + "_resume"),
        )
    except AttributeError:
        raise RecoveryRequired("runtime_owner_maintenance_unavailable") from None


async def _settle_stage(hooks, closed, deadline):
    """Fence a dependency level together, then wait for accepted work to finish."""
    for hook in hooks:
        if hook is None or hook in closed:
            continue
        # Retain before invoking: a failing close may already have fenced intake.
        closed.append(hook)
        hook.close(hook.owner)
    for hook in hooks:
        if hook is not None and not await hook.drain(hook.owner, deadline):
            raise RecoveryRequired("runtime_work_not_settled")


async def _resume_hooks(closed):
    """Resume in dependency order, preserving failed owners for explicit recovery."""
    cancellation = None
    while closed:
        hook = closed[-1]
        result = hook.resume(hook.owner)
        if inspect.isawaitable(result):
            completion = asyncio.ensure_future(result)
            while not completion.done():
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
            completion.result()
        closed.pop()
    if cancellation is not None:
        raise cancellation


def _app_hook(app, prefix):
    return _bind(app, "app", "TldwCli", prefix)


def _console_hooks(app, screens):
    from .unsaved_editors import _exact

    runtime = app.console_runtime
    if runtime is not None and not _exact(
        runtime, "Chat.console_runtime", "ConsoleRuntime"
    ):
        raise RecoveryRequired("runtime_owner_unqualified")
    controller = None if runtime is None else runtime.chat_controller
    producers = [
        _bind(
            controller,
            "Chat.console_chat_controller",
            "ConsoleChatController",
            "maintenance",
        ),
        _bind(
            app.console_image_edit_operations,
            "Chat.console_image_edit_operations",
            "ImageEditOperationRegistry",
        ),
    ]
    views = []
    for screen in screens:
        if not _exact(screen, "UI.Screens.chat_screen", "ChatScreen"):
            continue
        producers.append(
            _bind(
                screen._dictation,
                "UI.Console_Modules.dictation",
                "ConsoleDictationController",
                "maintenance",
            )
        )
        views.extend(
            (
                _bind(
                    screen,
                    "UI.Screens.chat_screen",
                    "ChatScreen",
                    "_console_sync_maintenance",
                ),
                _bind(
                    screen._image,
                    "UI.Console_Modules.image",
                    "ConsoleImageController",
                    "_recovered_images",
                ),
            )
        )
    return producers, views


def _authoring_owners(screens):
    """Enumerate existing authoring screens and controls without mounting any."""
    declarations = (
        ("UI.Screens.chat_screen", "ChatScreen"),
        ("UI.Screens.library_screen", "LibraryScreen"),
        ("UI.Screens.settings_screen", "SettingsScreen"),
        ("UI.Screens.personas_screen", "PersonasScreen"),
        ("UI.Screens.stts_screen", "STTSScreen"),
        ("UI.STTS_Window", "STTSWindow"),
        ("UI.stts_profile_library", "TTSProfileEditorModal"),
        ("Widgets.Library.library_notes_canvas", "LibraryNotesCanvas"),
        ("Widgets.Library.library_file_notes_workspace", "LibraryFileNotesWorkspace"),
        ("Widgets.Library.library_prompts_canvas", "LibraryPromptsListCanvas"),
        ("UI.Evals.bench_editor", "BenchEditor"),
        ("UI.Evals.character_bench_editor", "CharacterBenchEditor"),
    )
    owners = {}
    for module, name in declarations:
        loaded = sys.modules.get("tldw_chatbook." + module)
        expected = None if loaded is None else vars(loaded).get(name)
        if expected is None:
            continue
        for screen in screens:
            if isinstance(screen, expected):
                owners[id(screen)] = screen
            for child in screen.query(expected):
                owners[id(child)] = child
    return tuple(owners.values())


class RuntimeMaintenance:
    """Own one app settlement attempt on its event-loop task.

    Producer settlement alone is not native capture authority. The storage gate
    still checks actual outstanding native resources and installed coverage.
    """

    def __init__(self, app):
        _app_hook(app, "_speech_initialization")
        self.app = app
        self.task = asyncio.current_task()
        self.closed = []
        self.screens = ()
        self.logging_handlers = []
        self.pause = None
        self._settled = False

    def _check(self):
        if self.task is not asyncio.current_task():
            raise RecoveryRequired("runtime_maintenance_wrong_task")

    def _owner_snapshot(self):
        app = self.app
        runtime = app.console_runtime
        service = app.tts_service
        return tuple(
            id(owner)
            for owner in (
                runtime,
                None if runtime is None else runtime.chat_controller,
                app.console_image_edit_operations,
                app.scheduler_loop,
                app.evaluation_orchestrator,
                app.local_audio_services_service,
                app.file_notes_session_owner,
                app._tts_handler,
                app._stts_handler,
                app._tts_voice_bundle_service,
                app._tts_profile_service,
                app._tts_profile_repository,
                app._local_stt_dispatch_coordinator,
                app.audio_cpp_model_install_owner,
                app._audio_cpp_artifact_lease_coordinator,
                service,
                service.registry,
                service._clone_materializer,
            )
        )

    def unsaved_editors(self):
        """Read both mounted editors and memory retained across navigation."""
        from .unsaved_editors import probe_unsaved_editors

        self._check()
        return retained_issues(self.app) + probe_unsaved_editors(
            console_runtime=self.app.console_runtime,
            editors=_authoring_owners(tuple(self.app.screen_stack)),
            screen_state_store=self.app.screen_state_store,
        )

    async def settle_producers(self, deadline):
        """Settle upstream work before closing the services it still needs."""
        self._check()
        if self.closed or self._settled:
            raise RecoveryRequired("runtime_maintenance_already_started")
        app = self.app
        await _settle_stage(
            [_app_hook(app, "_screen_navigation")], self.closed, deadline
        )
        await _settle_stage(
            [
                _bind(
                    definition_participant,
                    "Backup_Recovery.rag_definition_participant",
                    "DefinitionParticipant",
                )
            ],
            self.closed,
            deadline,
        )
        if self.unsaved_editors():
            raise RecoveryRequired("needs_user_save_discard")
        # Fence delivery before producers settle: accepted completions may
        # publish autoplay while draining. Resume it after service/handler gates.
        delivery = _app_hook(app, "_speech_delivery")
        self.closed.append(delivery)
        delivery.close(delivery.owner)
        await _settle_stage(
            [_app_hook(app, "_speech_initialization")], self.closed, deadline
        )
        self._owners = self._owner_snapshot()
        self.screens = tuple(app.screen_stack)
        producers, views = _console_hooks(app, self.screens)
        producers.extend(
            (
                _app_hook(app, "_ingest_maintenance"),
                _bind(app.scheduler_loop, "Scheduling.scheduler.loop", "SchedulerLoop"),
                _bind(
                    app.evaluation_orchestrator,
                    "Evals.eval_orchestrator",
                    "EvaluationOrchestrator",
                ),
                _bind(
                    app.local_audio_services_service,
                    "Audio_Services_Interop.local_audio_services_service",
                    "LocalAudioServicesService",
                ),
                _bind(
                    app.file_notes_session_owner,
                    "Notes.file_notes_session_owner",
                    "FileNotesSessionOwner",
                ),
                _bind(
                    app._tts_handler,
                    "Event_Handlers.TTS_Events.tts_events",
                    "TTSEventHandler",
                    "maintenance",
                ),
                _bind(
                    app._stts_handler,
                    "Event_Handlers.STTS_Events.stts_events",
                    "STTSEventHandler",
                    "maintenance",
                ),
            )
        )
        library = sys.modules.get("tldw_chatbook.UI.stts_profile_library")
        if library is not None:
            for screen in self.screens:
                producers.extend(
                    _bind(owner, "UI.stts_profile_library", "STTSProfileLibrary")
                    for owner in screen.query(library.STTSProfileLibrary)
                )
        await _settle_stage(producers, self.closed, deadline)
        if not await delivery.drain(delivery.owner, deadline):
            raise RecoveryRequired("runtime_work_not_settled")
        await _settle_stage(views, self.closed, deadline)
        # Bundle import/export calls the profile service, which can in turn
        # acquire TTS and managed-artifact leases. Settle each caller first.
        for hooks in (
            [
                _bind(
                    app._tts_voice_bundle_service,
                    "TTS.voice_bundle_service",
                    "TTSVoiceBundlePortabilityService",
                )
            ],
            [
                _bind(
                    app._tts_profile_service, "TTS.profile_service", "TTSProfileService"
                )
            ],
            [_bind(app.tts_service, "TTS.TTS_Generation", "TTSService", "maintenance")],
            [
                _bind(
                    app.tts_service.registry,
                    "TTS.adapter_registry",
                    "TTSAdapterRegistry",
                    "maintenance",
                ),
                _bind(
                    app.tts_service._clone_materializer,
                    "TTS.profile_reference_materialization",
                    "TTSCloneReferenceMaterializer",
                ),
                _bind(
                    app._local_stt_dispatch_coordinator,
                    "STT.dispatch_coordinator",
                    "LocalSTTDispatchCoordinator",
                    "maintenance",
                ),
            ],
            [
                _bind(
                    app._tts_profile_repository,
                    "TTS.profile_repository",
                    "TTSProfileRepository",
                )
            ],
            [
                _bind(
                    app.audio_cpp_model_install_owner,
                    "UI.Navigation.audio_cpp_model_handoff",
                    "AudioCppModelInstallOwner",
                    "maintenance",
                )
            ],
            [
                _bind(
                    app._audio_cpp_artifact_lease_coordinator,
                    "TTS.audio_cpp_artifact_dependencies",
                    "AudioCppArtifactLeaseCoordinator",
                    "maintenance",
                )
            ],
        ):
            await _settle_stage(hooks, self.closed, deadline)
        if tuple(app.screen_stack) != self.screens:
            raise RecoveryRequired("runtime_screens_changed")
        if self._owner_snapshot() != self._owners:
            raise RecoveryRequired("runtime_owners_changed")
        for handler, name in (
            (app._tts_handler, "_tts_service"),
            (app._stts_handler, "_stts_service"),
        ):
            service = None if handler is None else getattr(handler, name)
            if service is not None and service is not app.tts_service:
                raise RecoveryRequired("runtime_owner_unqualified")
        if self.unsaved_editors():
            raise RecoveryRequired("needs_user_save_discard")
        self._settled = True

    async def resume(self):
        """Reopen producers only after ordinary storage admission is restored."""
        self._check()
        cancellation = None
        if self.pause is not None:
            try:
                await self.pause.reacquire_startup()
            except asyncio.CancelledError as error:
                cancellation = error
            self.pause.resume()
            self.pause = None
        while self.logging_handlers:
            handler = self.logging_handlers[-1]
            type(handler)._maintenance_resume(handler)
            self.logging_handlers.pop()
        try:
            await _resume_hooks(self.closed)
        finally:
            if not self.closed:
                self._settled = False
        if cancellation is not None:
            raise cancellation

    def retire_local_caches(self):
        """Fence ordinary storage, then release only this thread's owned caches."""
        from . import storage_admission as storage
        from .participants import _retire_current_thread_caches
        from ..Logging_Config import PrivateRotatingFileHandler

        self._check()
        if not self._settled:
            raise RecoveryRequired("runtime_producers_not_settled")
        if self.pause is not None:
            raise RecoveryRequired("local_pause_already_active")
        self.pause = storage._begin_local_pause()
        for handler in tuple(logging.getLogger().handlers):
            if type(handler) is PrivateRotatingFileHandler:
                self.logging_handlers.append(handler)
                PrivateRotatingFileHandler._maintenance_close_admission(handler)
        _retire_current_thread_caches(self.pause)

    def _require_storage_coverage(self, pause):
        """Accept only this app's live, settled coordinator and exact local gate."""
        self._check()
        if (
            not self._settled
            or self.pause is not pause
            or getattr(self.app, "_backup_runtime_maintenance", None) is not self
            or tuple(self.app.screen_stack) != self.screens
            or self._owner_snapshot() != self._owners
            or self.unsaved_editors()
        ):
            raise RecoveryRequired("participant_runtime_coverage_incomplete")
        if not pause.drain(time.monotonic()):
            raise RecoveryRequired("runtime_native_resources_not_settled")


async def monitor_app(app):
    """Yield this live app to native maintenance intent, then restore admission.

    This retained app task owns the whole pause. Native startup readmission waits
    off-loop until the exclusive capturer releases its gate; cancellation cannot
    abandon that handoff. A refused attempt waits for its intent to end before
    retrying, preserving user work and avoiding repeated pause/resume cycles.
    """
    from . import storage_admission as storage

    refused = False
    while True:
        await asyncio.sleep(0.1)
        try:
            requested = storage._local_pause_requested()
        except (OSError, ValueError, RuntimeError):
            app._backup_maintenance_error = "admission_state_unavailable"
            continue
        if not requested:
            refused = False
            continue
        if refused:
            continue
        runtime = RuntimeMaintenance(app)
        app._backup_runtime_maintenance = runtime
        app._backup_maintenance_error = None
        try:
            deadline = time.monotonic() + 30
            await runtime.settle_producers(deadline)
            runtime.retire_local_caches()
            while not runtime.pause.drain(time.monotonic()):
                if time.monotonic() >= deadline:
                    raise RecoveryRequired("runtime_native_resources_not_settled")
                await asyncio.sleep(0.01)
            runtime.pause.retire_startup(runtime)
        except (OSError, ValueError, RuntimeError) as error:
            # Surface actionable local refusals without propagating arbitrary
            # owner exception text into logs or the backup UI.
            app._backup_maintenance_error = (
                "needs_user_save_discard"
                if type(error) is RecoveryRequired
                and error.args == ("needs_user_save_discard",)
                else "runtime_work_not_settled"
            )
            refused = True
        finally:
            # The same task must retain local-pause authority throughout native
            # reacquisition; moving resume into a separate task invalidates it.
            try:
                await runtime.resume()
            finally:
                if runtime.pause is None and not runtime.closed:
                    app._backup_runtime_maintenance = None
