# stts_events.py
# Description: Event handlers for S/TT/S (Speech/Text-to-Speech) functionality
#
# Imports
import asyncio
from collections.abc import Coroutine, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

from loguru import logger
from rich.markup import escape

#
# Third-party imports
from textual.message import Message
from textual.widgets import Button, ProgressBar, RichLog, Static

#
# Local imports
from tldw_chatbook.TTS import (
    OpenAISpeechRequest,
    STTSGeneratedAudio,
    STTSPlaygroundRequest,
    TTSRequest,
    get_tts_service,
)
from tldw_chatbook.TTS.adapter_registry import ReconfigureResult
from tldw_chatbook.TTS.adapter_types import (
    ProgressSink,
    TTSOperationError,
    TTSProgress,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.audio_cpp_config import project_audio_cpp_config
from tldw_chatbook.TTS.legacy_bridge import legacy_provider_config
from tldw_chatbook.TTS.TTS_Generation import _join_retained_task
from tldw_chatbook.Utils.secure_temp_files import (
    create_secure_temp_file,
    secure_delete_file,
)

#
#######################################################################################################################
#
# Event Messages


class _SettingBinding(NamedTuple):
    destinations: tuple[tuple[str, str], ...]
    provider_id: str | None = None


def _app_tts_binding(
    key: str,
    provider_id: str | None = None,
) -> _SettingBinding:
    return _SettingBinding((("app_tts", key),), provider_id)


_TTS_SETTING_BINDINGS = {
    "audio_cpp": _SettingBinding(
        (("app_tts", "audio_cpp"),),
        "audio_cpp",
    ),
    "default_provider": _SettingBinding(
        (
            ("app_tts", "default_provider"),
            ("tts_settings", "default_tts_provider"),
        )
    ),
    "default_voice": _SettingBinding(
        (
            ("app_tts", "default_voice"),
            ("tts_settings", "default_tts_voice"),
        )
    ),
    "default_model": _SettingBinding(
        (
            ("app_tts", "default_model"),
            ("tts_settings", "default_openai_tts_model"),
        )
    ),
    "default_format": _SettingBinding(
        (
            ("app_tts", "default_format"),
            ("tts_settings", "default_openai_tts_output_format"),
        )
    ),
    "default_speed": _SettingBinding(
        (
            ("app_tts", "default_speed"),
            ("tts_settings", "default_openai_tts_speed"),
        )
    ),
    "openai_api_key": _SettingBinding(
        (("API", "openai_api_key"),),
        "openai",
    ),
    "OPENAI_BASE_URL": _app_tts_binding("OPENAI_BASE_URL", "openai"),
    "OPENAI_ORG_ID": _app_tts_binding("OPENAI_ORG_ID", "openai"),
    "elevenlabs_api_key": _SettingBinding(
        (("API", "elevenlabs_api_key"),),
        "elevenlabs",
    ),
    "ELEVENLABS_DEFAULT_MODEL": _app_tts_binding(
        "ELEVENLABS_DEFAULT_MODEL", "elevenlabs"
    ),
    "ELEVENLABS_OUTPUT_FORMAT": _app_tts_binding(
        "ELEVENLABS_OUTPUT_FORMAT", "elevenlabs"
    ),
    "ELEVENLABS_VOICE_STABILITY": _app_tts_binding(
        "ELEVENLABS_VOICE_STABILITY", "elevenlabs"
    ),
    "ELEVENLABS_SIMILARITY_BOOST": _app_tts_binding(
        "ELEVENLABS_SIMILARITY_BOOST", "elevenlabs"
    ),
    "ELEVENLABS_STYLE": _app_tts_binding("ELEVENLABS_STYLE", "elevenlabs"),
    "ELEVENLABS_USE_SPEAKER_BOOST": _app_tts_binding(
        "ELEVENLABS_USE_SPEAKER_BOOST", "elevenlabs"
    ),
    "KOKORO_DEVICE_DEFAULT": _app_tts_binding("KOKORO_DEVICE_DEFAULT", "kokoro"),
    "KOKORO_USE_ONNX": _app_tts_binding("KOKORO_USE_ONNX"),
    "KOKORO_ONNX_MODEL_PATH_DEFAULT": _app_tts_binding(
        "KOKORO_ONNX_MODEL_PATH_DEFAULT", "kokoro"
    ),
    "KOKORO_ONNX_VOICES_JSON_DEFAULT": _app_tts_binding(
        "KOKORO_ONNX_VOICES_JSON_DEFAULT", "kokoro"
    ),
    "KOKORO_MAX_TOKENS": _app_tts_binding("KOKORO_MAX_TOKENS", "kokoro"),
    "KOKORO_ENABLE_VOICE_MIXING": _app_tts_binding(
        "KOKORO_ENABLE_VOICE_MIXING", "kokoro"
    ),
    "KOKORO_TRACK_PERFORMANCE": _app_tts_binding("KOKORO_TRACK_PERFORMANCE", "kokoro"),
    "CHATTERBOX_DEVICE": _app_tts_binding("CHATTERBOX_DEVICE", "chatterbox"),
    "CHATTERBOX_VOICE_DIR": _app_tts_binding("CHATTERBOX_VOICE_DIR", "chatterbox"),
    "CHATTERBOX_EXAGGERATION": _app_tts_binding(
        "CHATTERBOX_EXAGGERATION", "chatterbox"
    ),
    "CHATTERBOX_CFG_WEIGHT": _app_tts_binding("CHATTERBOX_CFG_WEIGHT", "chatterbox"),
    "CHATTERBOX_TEMPERATURE": _app_tts_binding("CHATTERBOX_TEMPERATURE", "chatterbox"),
    "CHATTERBOX_CHUNK_SIZE": _app_tts_binding("CHATTERBOX_CHUNK_SIZE", "chatterbox"),
    "CHATTERBOX_RANDOM_SEED": _app_tts_binding("CHATTERBOX_RANDOM_SEED", "chatterbox"),
    "CHATTERBOX_NUM_CANDIDATES": _app_tts_binding(
        "CHATTERBOX_NUM_CANDIDATES", "chatterbox"
    ),
    "CHATTERBOX_VALIDATE_WHISPER": _app_tts_binding(
        "CHATTERBOX_VALIDATE_WHISPER", "chatterbox"
    ),
    "CHATTERBOX_PREPROCESS_TEXT": _app_tts_binding(
        "CHATTERBOX_PREPROCESS_TEXT", "chatterbox"
    ),
    "CHATTERBOX_NORMALIZE_AUDIO": _app_tts_binding(
        "CHATTERBOX_NORMALIZE_AUDIO", "chatterbox"
    ),
    "CHATTERBOX_TARGET_DB": _app_tts_binding("CHATTERBOX_TARGET_DB", "chatterbox"),
    "CHATTERBOX_MAX_CHUNK_SIZE": _app_tts_binding(
        "CHATTERBOX_MAX_CHUNK_SIZE", "chatterbox"
    ),
    "CHATTERBOX_STREAMING": _app_tts_binding("CHATTERBOX_STREAMING", "chatterbox"),
    "CHATTERBOX_STREAM_CHUNK_SIZE": _app_tts_binding(
        "CHATTERBOX_STREAM_CHUNK_SIZE", "chatterbox"
    ),
    "CHATTERBOX_ENABLE_CROSSFADE": _app_tts_binding(
        "CHATTERBOX_ENABLE_CROSSFADE", "chatterbox"
    ),
    "CHATTERBOX_CROSSFADE_MS": _app_tts_binding(
        "CHATTERBOX_CROSSFADE_MS", "chatterbox"
    ),
    "HIGGS_MODEL_PATH": _SettingBinding(
        (("HiggsSettings", "model_path"),),
        "higgs",
    ),
    "HIGGS_VOICE_SAMPLES_DIR": _SettingBinding(
        (("HiggsSettings", "voice_samples_dir"),),
        "higgs",
    ),
    "HIGGS_DEVICE": _SettingBinding(
        (("HiggsSettings", "device"),),
        "higgs",
    ),
    "HIGGS_ENABLE_FLASH_ATTN": _SettingBinding(
        (("HiggsSettings", "enable_flash_attn"),),
        "higgs",
    ),
    "HIGGS_DTYPE": _SettingBinding(
        (("HiggsSettings", "dtype"),),
        "higgs",
    ),
    "HIGGS_MAX_REFERENCE_DURATION": _SettingBinding(
        (("HiggsSettings", "max_reference_duration"),),
        "higgs",
    ),
    "HIGGS_DEFAULT_LANGUAGE": _SettingBinding(
        (("HiggsSettings", "default_language"),),
        "higgs",
    ),
    "HIGGS_ENABLE_VOICE_CLONING": _SettingBinding(
        (("HiggsSettings", "enable_voice_cloning"),),
        "higgs",
    ),
    "HIGGS_ENABLE_MULTI_SPEAKER": _SettingBinding(
        (("HiggsSettings", "enable_multi_speaker"),),
        "higgs",
    ),
    "HIGGS_SPEAKER_DELIMITER": _SettingBinding(
        (("HiggsSettings", "speaker_delimiter"),),
        "higgs",
    ),
    "HIGGS_TRACK_PERFORMANCE": _SettingBinding(
        (("HiggsSettings", "track_performance"),),
        "higgs",
    ),
    "HIGGS_MAX_NEW_TOKENS": _SettingBinding(
        (("HiggsSettings", "max_new_tokens"),),
        "higgs",
    ),
    "HIGGS_TEMPERATURE": _SettingBinding(
        (("HiggsSettings", "temperature"),),
        "higgs",
    ),
    "HIGGS_TOP_P": _SettingBinding(
        (("HiggsSettings", "top_p"),),
        "higgs",
    ),
    "HIGGS_REPETITION_PENALTY": _SettingBinding(
        (("HiggsSettings", "repetition_penalty"),),
        "higgs",
    ),
    "ALLTALK_TTS_URL_DEFAULT": _app_tts_binding("ALLTALK_TTS_URL_DEFAULT", "alltalk"),
    "ALLTALK_TTS_VOICE_DEFAULT": _app_tts_binding(
        "ALLTALK_TTS_VOICE_DEFAULT", "alltalk"
    ),
    "ALLTALK_TTS_LANGUAGE_DEFAULT": _app_tts_binding(
        "ALLTALK_TTS_LANGUAGE_DEFAULT", "alltalk"
    ),
    "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT": _app_tts_binding(
        "ALLTALK_TTS_OUTPUT_FORMAT_DEFAULT", "alltalk"
    ),
}
_TTS_PROVIDER_ORDER = (
    "audio_cpp",
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)

_RECOVERY_ACTION_COPY = {
    "check_server": "Check the configured audio.cpp server and retry.",
    "configure_server": "Open STTS Settings and configure the audio.cpp server.",
    "edit_request": "Adjust the text or selected options and retry.",
    "refresh_models": "Refresh models in the STTS Playground and retry.",
    "retry": "Retry from the STTS Playground.",
}


def _effective_provider_config(
    provider_id: str,
    effective_settings: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project one provider's effective registry configuration."""
    if provider_id == "audio_cpp":
        return project_audio_cpp_config(effective_settings).to_mapping()
    return legacy_provider_config(provider_id, effective_settings)


class STTSPlaygroundGenerateEvent(Message):
    """Event carrying one immutable Playground generation snapshot."""

    def __init__(self, request: STTSPlaygroundRequest) -> None:
        super().__init__()
        if not isinstance(request, STTSPlaygroundRequest):
            raise TypeError("request must be an STTSPlaygroundRequest")
        self.request = request


class STTSSettingsSaveEvent(Message):
    """Event when TTS settings are saved"""

    def __init__(self, settings: Dict[str, Any]):
        super().__init__()
        self.settings = deepcopy(settings)


class STTSProviderConfigurationChanged(Message):
    """Signal that one provider's effective configuration revision changed."""

    def __init__(self, provider_id: str, configuration_revision: int) -> None:
        super().__init__()
        self.provider_id = provider_id
        self.configuration_revision = configuration_revision


@dataclass(frozen=True, slots=True)
class _STTSPlaygroundState:
    """Read-only handler-owned Playground lifecycle snapshot."""

    active_operation_id: str | None
    artifact: STTSGeneratedAudio | None
    generation_active: bool


class STTSAudioBookGenerateEvent(Message):
    """Event when audiobook generation is requested"""

    def __init__(
        self,
        content: str,
        chapters: list,
        narrator_voice: str,
        output_format: str,
        options: Dict[str, Any],
    ):
        super().__init__()
        self.content = content
        self.chapters = chapters
        self.narrator_voice = narrator_voice
        self.output_format = output_format
        self.options = options


#######################################################################################################################
#
# Event Handler Mixin


class STTSEventHandler:
    """Event handler for S/TT/S functionality"""

    def __init__(self, app=None):
        self.app = app  # Reference to the main app
        self._stts_service = None
        self._current_audio_file = None
        self._current_playground_artifact: STTSGeneratedAudio | None = None
        self._is_generating = False
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._generation_task: asyncio.Task[None] | None = None
        self._active_playground_operation_id: str | None = None
        self._playground_audio_files: set[Path] = set()
        self._playground_operation_files: dict[str, set[Path]] = {}
        self._playground_file_leases: dict[Path, int] = {}
        self._cleanup_task: asyncio.Task[None] | None = None
        self._settings_save_lock = asyncio.Lock()

    async def initialize_stts(self) -> None:
        """Initialize S/TT/S service"""
        try:
            self._stts_service = await get_tts_service()
            logger.info("S/TT/S service initialized successfully")
        except Exception:
            logger.error("Failed to initialize S/TT/S service")
            self._stts_service = None

    def playground_state(self) -> _STTSPlaygroundState:
        """Return immutable handler-owned generation and artifact state."""
        return _STTSPlaygroundState(
            active_operation_id=self._active_playground_operation_id,
            artifact=self._current_playground_artifact,
            generation_active=self._is_generating,
        )

    def start_playground_generation(
        self,
        event: STTSPlaygroundGenerateEvent,
    ) -> None:
        """Start and retain exactly one handler-owned Playground task."""
        if self._cleanup_task is not None:
            logger.debug("Ignoring TTS generation after STTS cleanup started")
            return
        if self._generation_task is not None and not self._generation_task.done():
            self.app.notify("TTS generation already in progress", severity="warning")
            return
        if self._is_generating:
            self.app.notify("TTS generation already in progress", severity="warning")
            return

        task = asyncio.create_task(
            self.handle_playground_generate(event),
            name=f"stts_playground_{event.request.operation_id}",
        )
        self._generation_task = task
        self._active_tasks.add(task)
        task.add_done_callback(self._playground_generation_done)

    def _playground_generation_done(self, task: asyncio.Task[None]) -> None:
        self._active_tasks.discard(task)
        if self._generation_task is task:
            self._generation_task = None
        try:
            task.exception()
        except BaseException:
            pass

    def _track_operation_file(self, operation_id: str, path: Path) -> None:
        path = Path(path)
        self._playground_audio_files.add(path)
        self._playground_operation_files.setdefault(operation_id, set()).add(path)

    def _forget_operation_file(self, operation_id: str, path: Path) -> None:
        path = Path(path)
        self._playground_audio_files.discard(path)
        operation_files = self._playground_operation_files.get(operation_id)
        if operation_files is None:
            return
        operation_files.discard(path)
        if not operation_files:
            self._playground_operation_files.pop(operation_id, None)

    def _delete_operation_files(
        self,
        operation_id: str,
        *,
        keep: frozenset[Path] = frozenset(),
    ) -> None:
        for path in tuple(self._playground_operation_files.get(operation_id, ())):
            if path in keep:
                continue
            if self._playground_file_leases.get(path, 0) > 0:
                continue
            if secure_delete_file(path) or not path.exists():
                self._forget_operation_file(operation_id, path)

    def lease_playground_artifact(self, artifact: STTSGeneratedAudio) -> bool:
        """Pin a handler-owned artifact across a deferred UI action."""
        path = Path(artifact.path)
        operation_files = self._playground_operation_files.get(
            artifact.operation_id,
            set(),
        )
        if (
            path not in self._playground_audio_files
            or path not in operation_files
            or not path.exists()
        ):
            return False
        self._playground_file_leases[path] = (
            self._playground_file_leases.get(path, 0) + 1
        )
        return True

    def release_playground_artifact(self, artifact: STTSGeneratedAudio) -> None:
        """Release one lease and retire the artifact when it is no longer current."""
        path = Path(artifact.path)
        count = self._playground_file_leases.get(path, 0)
        if count <= 0:
            return
        if count == 1:
            self._playground_file_leases.pop(path, None)
        else:
            self._playground_file_leases[path] = count - 1
            return

        current_path = (
            self._current_playground_artifact.path
            if self._current_playground_artifact is not None
            else None
        )
        if current_path == path:
            return
        for operation_id, operation_files in tuple(
            self._playground_operation_files.items()
        ):
            if path in operation_files:
                self._delete_operation_files(operation_id)

    def _accept_playground_artifact(self, artifact: STTSGeneratedAudio) -> None:
        """Store the new artifact before securely retiring older files."""
        self._current_playground_artifact = artifact
        self._current_audio_file = artifact.path
        self._track_operation_file(artifact.operation_id, artifact.path)
        for operation_id in tuple(self._playground_operation_files):
            self._delete_operation_files(
                operation_id,
                keep=(
                    frozenset({artifact.path})
                    if operation_id == artifact.operation_id
                    else frozenset()
                ),
            )

    async def _generate_audio_cpp(
        self,
        snapshot: STTSPlaygroundRequest,
        progress_sink: ProgressSink | None,
    ) -> STTSGeneratedAudio:
        """Generate one complete native audio.cpp WAV response."""
        if self._stts_service is None:
            raise RuntimeError("TTS service is not initialized")

        request = TTSRequest(
            provider_id="audio_cpp",
            model_id=snapshot.model_id,
            text=snapshot.text,
            voice=snapshot.voice_id,
            response_format="wav",
            speed=1.0,
            options={},
        )
        response = None
        primary_error: BaseException | None = None
        try:
            response = await self._stts_service.synthesize(request, progress_sink)
            chunks = [chunk async for chunk in response.byte_stream]
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if response is not None:
                try:
                    await response.aclose()
                except BaseException:
                    if primary_error is None:
                        raise
                    logger.warning(
                        "Failed to close audio.cpp response after {}",
                        type(primary_error).__name__,
                    )

        path = Path(
            create_secure_temp_file(
                b"".join(chunks),
                suffix=f".{response.audio_format.removeprefix('.')}",
                prefix="stts_playground_",
            )
        )
        self._track_operation_file(snapshot.operation_id, path)
        try:
            return STTSGeneratedAudio(
                path=path,
                provider_id=response.provider_id,
                model_id=response.model_id,
                voice_id=snapshot.voice_id,
                source_text=snapshot.text,
                operation_id=snapshot.operation_id,
                audio_format=response.audio_format,
                content_type=response.content_type,
                metadata=response.metadata,
            )
        except BaseException:
            if secure_delete_file(path) or not path.exists():
                self._forget_operation_file(snapshot.operation_id, path)
            raise

    async def _generate_legacy(
        self,
        snapshot: STTSPlaygroundRequest,
        progress_sink: ProgressSink | None,
    ) -> STTSGeneratedAudio:
        """Retain the existing stream-and-convert path for legacy providers."""
        if self._stts_service is None:
            raise RuntimeError("TTS service is not initialized")

        requested_format = snapshot.response_format.lower()
        if requested_format not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
            requested_format = "mp3"
        request = OpenAISpeechRequest(
            model=snapshot.model_id,
            input=snapshot.text,
            voice=snapshot.voice_id or "default",
            response_format="wav",
            speed=snapshot.speed,
        )
        options = dict(snapshot.options)
        if snapshot.provider_id in {"chatterbox", "higgs"} and options:
            request.extra_params = options

        internal_model_id = self._legacy_internal_model_id(snapshot, options)
        created_paths: set[Path] = set()
        try:
            chunks = [
                chunk
                async for chunk in self._stts_service.generate_audio_stream(
                    request,
                    internal_model_id,
                    progress_sink=progress_sink,
                )
            ]
            wav_file = Path(
                create_secure_temp_file(
                    b"".join(chunks),
                    suffix=".wav",
                    prefix="stts_playground_",
                )
            )
            created_paths.add(wav_file)
            self._track_operation_file(snapshot.operation_id, wav_file)
            output_file = wav_file
            audio_format = "wav"

            if requested_format != "wav":
                conversion_destination = wav_file.with_suffix(f".{requested_format}")
                created_paths.add(conversion_destination)
                self._track_operation_file(
                    snapshot.operation_id,
                    conversion_destination,
                )
                converted_file = await self._convert_audio_format(
                    wav_file,
                    requested_format,
                )
                if converted_file is not None:
                    output_file = Path(converted_file)
                    created_paths.add(output_file)
                    self._track_operation_file(snapshot.operation_id, output_file)
                    audio_format = requested_format
                    if secure_delete_file(wav_file) or not wav_file.exists():
                        self._forget_operation_file(
                            snapshot.operation_id,
                            wav_file,
                        )
                        created_paths.discard(wav_file)
                elif (
                    secure_delete_file(conversion_destination)
                    or not conversion_destination.exists()
                ):
                    self._forget_operation_file(
                        snapshot.operation_id,
                        conversion_destination,
                    )
                    created_paths.discard(conversion_destination)

            return STTSGeneratedAudio(
                path=output_file,
                provider_id=snapshot.provider_id,
                model_id=snapshot.model_id,
                voice_id=snapshot.voice_id,
                source_text=snapshot.text,
                operation_id=snapshot.operation_id,
                audio_format=audio_format,
                content_type=self._audio_content_type(audio_format),
                metadata={},
            )
        except BaseException:
            for path in created_paths:
                if secure_delete_file(path) or not path.exists():
                    self._forget_operation_file(snapshot.operation_id, path)
            raise

    @staticmethod
    def _legacy_internal_model_id(
        snapshot: STTSPlaygroundRequest,
        options: Mapping[str, Any],
    ) -> str:
        provider_id = snapshot.provider_id
        if provider_id == "openai":
            model_id = snapshot.model_id.lower().replace("-", "")
            return f"openai_official_{model_id}"
        if provider_id == "elevenlabs":
            return f"elevenlabs_{snapshot.model_id}"
        if provider_id == "kokoro":
            engine = "onnx" if options.get("use_onnx", True) else "pytorch"
            return f"local_kokoro_default_{engine}"
        if provider_id == "chatterbox":
            return "local_chatterbox_default"
        if provider_id == "higgs":
            return "local_higgs_v2"
        if provider_id == "alltalk":
            return f"alltalk_{snapshot.model_id}"
        return snapshot.model_id

    @staticmethod
    def _audio_content_type(audio_format: str) -> str:
        return {
            "aac": "audio/aac",
            "flac": "audio/flac",
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "pcm": "audio/L16",
            "wav": "audio/wav",
        }.get(audio_format, "application/octet-stream")

    async def handle_playground_generate(
        self, event: STTSPlaygroundGenerateEvent
    ) -> None:
        """Run playground TTS inside the handler's retained event task."""
        if self._cleanup_task is not None:
            logger.debug("Ignoring TTS generation after STTS cleanup started")
            return
        if self._is_generating:
            self.app.notify("TTS generation already in progress", severity="warning")
            return

        if not self._stts_service:
            operation_id = event.request.operation_id
            self._is_generating = True
            self._active_playground_operation_id = operation_id
            self._deliver_generation_failure(
                operation_id,
                "The TTS service is unavailable",
            )
            self.app.notify("TTS service not initialized", severity="error")
            self._is_generating = False
            self._finish_generation_ui(operation_id)
            self._active_playground_operation_id = None
            return

        self._is_generating = True
        self._active_playground_operation_id = event.request.operation_id
        await self._generate_tts_worker(event)

    async def _generate_tts_worker(
        self,
        event: STTSPlaygroundGenerateEvent,
    ) -> None:
        """Generate from one immutable request and deliver one artifact."""
        snapshot = event.request
        self._show_generation_progress(snapshot.operation_id)

        async def progress_callback(info: TTSProgress) -> None:
            self._update_generation_progress(snapshot.operation_id, info)

        try:
            if snapshot.provider_id == "audio_cpp":
                artifact = await self._generate_audio_cpp(
                    snapshot,
                    progress_callback,
                )
            else:
                artifact = await self._generate_legacy(
                    snapshot,
                    progress_callback,
                )
            self._accept_playground_artifact(artifact)
            self._deliver_generation_success(
                snapshot.operation_id,
                artifact,
            )
            self.app.notify("TTS generation complete!", severity="information")
        except asyncio.CancelledError:
            self._delete_operation_files(snapshot.operation_id)
            raise
        except Exception as error:
            self._delete_operation_files(snapshot.operation_id)
            message = self._generation_error_copy(error)
            if isinstance(error, TTSOperationError):
                logger.error(
                    "TTS generation failed (code={}, retryable={})",
                    error.code,
                    error.retryable,
                )
            else:
                logger.error(
                    "TTS generation failed ({})",
                    type(error).__name__,
                )
            self._deliver_generation_failure(snapshot.operation_id, message)
            self.app.notify(
                f"TTS generation failed: {escape(message)}",
                severity="error",
            )
        finally:
            if self._active_playground_operation_id == snapshot.operation_id:
                self._is_generating = False
                self._finish_generation_ui(snapshot.operation_id)
                self._active_playground_operation_id = None

    def _show_generation_progress(self, operation_id: str) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def show() -> None:
            playground.query_one("#generation-status-container").remove_class("hidden")
            playground.query_one("#generation-progress").update(
                total=100,
                progress=0,
            )

        self._invoke_playground(playground, show)

    def _update_generation_progress(
        self,
        operation_id: str,
        info: TTSProgress,
    ) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def update() -> None:
            playground.query_one(
                "#generation-status-text",
                Static,
            ).update(info.status or "Generating...")
            if info.fraction is not None:
                playground.query_one(
                    "#generation-progress",
                    ProgressBar,
                ).update(progress=info.fraction * 100)
            log = playground.query_one("#tts-generation-log", RichLog)
            audio_duration = info.metrics.get("audio_duration")
            if isinstance(audio_duration, (int, float)):
                log.write(f"[dim]Generated {audio_duration:.1f}s of audio[/dim]")
            elif info.processed is not None:
                if info.total is None:
                    log.write(f"[dim]Processed {info.processed} item(s)[/dim]")
                else:
                    log.write(
                        f"[dim]Processed {info.processed}/{info.total} item(s)[/dim]"
                    )

        self._invoke_playground(playground, update)

    def _deliver_generation_success(
        self,
        operation_id: str,
        artifact: STTSGeneratedAudio,
    ) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def deliver() -> None:
            playground.query_one("#tts-generation-log", RichLog).write(
                "[bold green]Generation complete[/bold green]"
            )
            callback = getattr(playground, "_generation_complete", None)
            if callable(callback):
                callback(artifact)
                return
            playground.query_one("#audio-play-btn", Button).disabled = False
            playground.query_one("#audio-export-btn", Button).disabled = False
            playground.query_one("#audio-player-status", Static).update(
                "Audio ready to play"
            )

        self._invoke_playground(playground, deliver)

    def _deliver_generation_failure(
        self,
        operation_id: str,
        message: str,
    ) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def deliver() -> None:
            playground.query_one("#tts-generation-log", RichLog).write(
                f"[bold red]Generation failed: {escape(message)}[/bold red]"
            )
            callback = getattr(playground, "_generation_complete", None)
            if callable(callback):
                callback(None)

        self._invoke_playground(playground, deliver)

    def _finish_generation_ui(self, operation_id: str) -> None:
        playground = self._mounted_playground(operation_id)
        if playground is None:
            return

        def finish() -> None:
            sync_generate_enabled = getattr(
                playground,
                "_sync_generate_enabled",
                None,
            )
            if callable(sync_generate_enabled):
                sync_generate_enabled()
            else:
                playground.query_one(
                    "#tts-generate-btn",
                    Button,
                ).disabled = False
            playground.query_one("#generation-status-container").add_class("hidden")

        self._invoke_playground(playground, finish)

    def _mounted_playground(self, operation_id: str) -> Any | None:
        if operation_id != self._active_playground_operation_id:
            return None
        try:
            from tldw_chatbook.UI.STTS_Window import TTSPlaygroundWidget

            return self.app.query_one(TTSPlaygroundWidget)
        except Exception as error:
            logger.debug(
                "TTS Playground is not mounted ({})",
                type(error).__name__,
            )
            return None

    @staticmethod
    def _invoke_playground(
        playground: Any,
        callback: object,
        *args: object,
    ) -> None:
        try:
            call_from_thread = getattr(playground, "call_from_thread", None)
            if callable(call_from_thread):
                call_from_thread(callback, *args)
            elif callable(callback):
                callback(*args)
        except Exception as error:
            logger.debug(
                "Playground generation display update failed ({})",
                type(error).__name__,
            )

    @staticmethod
    def _generation_error_copy(error: Exception) -> str:
        if isinstance(error, TTSOperationError):
            parts = [str(error)]
            recovery_copy = _RECOVERY_ACTION_COPY.get(error.recovery_action or "")
            if recovery_copy is not None:
                parts.append(recovery_copy)
            elif error.retryable:
                parts.append(_RECOVERY_ACTION_COPY["retry"])
            return " ".join(parts)
        if isinstance(error, TTSProviderReconfiguringError):
            return "TTS settings are being applied; retry shortly"
        if isinstance(error, TTSRegistryClosedError):
            return "The TTS service is unavailable"
        if isinstance(error, ValueError):
            return "TTS is not configured; open STTS Settings"
        return "Unexpected TTS generation failure; retry"

    async def handle_settings_save(self, event: STTSSettingsSaveEvent) -> None:
        """Handle settings save"""
        if self._cleanup_task is not None:
            logger.debug("Ignoring STTS settings after cleanup started")
            return
        async with self._settings_save_lock:
            if self._cleanup_task is not None:
                logger.debug("Ignoring STTS settings after cleanup started")
                return
            await self._persist_settings(event)

    async def _persist_settings(self, event: STTSSettingsSaveEvent) -> None:
        """Persist one settings event and refresh its affected providers."""
        try:
            from tldw_chatbook.config import (
                load_settings,
                save_settings_to_cli_config,
            )

            section_values: dict[str, dict[str, Any]] = {}
            saved_destinations: list[tuple[str, str, str]] = []
            candidate_provider_ids: set[str] = set()
            for key, value in event.settings.items():
                binding = _TTS_SETTING_BINDINGS.get(key)
                if binding is None:
                    continue
                for section, setting_name in binding.destinations:
                    section_values.setdefault(section, {})[setting_name] = deepcopy(
                        value
                    )
                    saved_destinations.append((key, section, setting_name))
                if binding.provider_id is not None:
                    candidate_provider_ids.add(binding.provider_id)

            if section_values and save_settings_to_cli_config(section_values) is False:
                raise RuntimeError("STTS settings batch save failed")
            for key, section, setting_name in saved_destinations:
                logger.info(f"Saved {key} to [{section}].{setting_name}")

            effective_settings = load_settings()
            candidate_providers = [
                provider_id
                for provider_id in _TTS_PROVIDER_ORDER
                if provider_id in candidate_provider_ids
            ]
            if candidate_providers:
                service = self._stts_service
                if service is None:
                    service = await get_tts_service()
                    self._stts_service = service
                results = await asyncio.gather(
                    *(
                        service.reconfigure_provider(
                            provider_id,
                            _effective_provider_config(
                                provider_id,
                                effective_settings,
                            ),
                        )
                        for provider_id in candidate_providers
                    ),
                    return_exceptions=True,
                )
                failed_providers = [
                    provider_id
                    for provider_id, result in zip(candidate_providers, results)
                    if isinstance(result, BaseException)
                ]
                for provider_id, result in zip(candidate_providers, results):
                    if result is ReconfigureResult.CHANGED:
                        self.app.post_message(
                            STTSProviderConfigurationChanged(
                                provider_id,
                                service.configuration_revision(provider_id),
                            )
                        )
                if failed_providers:
                    logger.error(
                        "Failed to reconfigure TTS providers: {}",
                        ", ".join(failed_providers),
                    )
                    self.app.notify(
                        "Settings saved, but some TTS providers could not be updated",
                        severity="error",
                    )
                    return

            self.app.notify("Settings saved successfully!", severity="information")
        except Exception:
            message = "Failed to save settings"
            logger.error(message)
            self.app.notify(message, severity="error")

    async def _convert_audio_format(
        self, input_file: Path, output_format: str
    ) -> Optional[Path]:
        """Convert audio file to a different format using ffmpeg"""
        process: asyncio.subprocess.Process | None = None
        try:
            # Create output file with requested format
            output_file = input_file.with_suffix(f".{output_format}")

            # Use ffmpeg for conversion
            cmd = [
                "ffmpeg",
                "-i",
                str(input_file),
                "-y",  # Overwrite output files
                "-loglevel",
                "error",  # Suppress verbose output
            ]

            # Add format-specific options
            if output_format == "mp3":
                # High quality MP3 encoding
                cmd.extend(["-codec:a", "libmp3lame", "-b:a", "192k"])
            elif output_format == "opus":
                cmd.extend(["-codec:a", "libopus", "-b:a", "128k"])
            elif output_format == "aac":
                cmd.extend(["-codec:a", "aac", "-b:a", "192k"])
            elif output_format == "flac":
                cmd.extend(["-codec:a", "flac"])

            cmd.append(str(output_file))

            # Run conversion asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            # Wait for the process to complete
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"Successfully converted audio to {output_format}")
                return output_file
            else:
                stderr_text = stderr.decode("utf-8") if stderr else "Unknown error"
                logger.error(f"ffmpeg conversion failed: {stderr_text}")
                return None

        except asyncio.CancelledError:
            if process is not None:
                terminate_task = asyncio.create_task(
                    self._terminate_conversion_process(process),
                    name="stts_ffmpeg_terminate",
                )
                await _join_retained_task(terminate_task)
            raise
        except FileNotFoundError:
            logger.error(
                "ffmpeg not found. Please install ffmpeg for audio format conversion."
            )
            return None
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None

    @staticmethod
    async def _terminate_conversion_process(
        process: asyncio.subprocess.Process,
    ) -> None:
        """Terminate ffmpeg, escalating to kill if it does not exit promptly."""
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.wait()

    async def handle_audiobook_generate(
        self, event: STTSAudioBookGenerateEvent
    ) -> None:
        """Handle audiobook generation"""
        if self._cleanup_task is not None:
            logger.debug("Ignoring audiobook generation after STTS cleanup started")
            return
        try:
            from tldw_chatbook.TTS.audiobook_generator import (
                AudioBookGenerator,
                AudioBookProgress,
                AudioBookRequest,
            )

            logger.info("AudioBook generation requested")

            # Initialize audiobook generator
            generator = AudioBookGenerator(self._stts_service)
            await generator.initialize()

            # Create audiobook request from event data
            audiobook_request = AudioBookRequest(
                content=event.content,
                title=event.options.get("title", "Untitled Book"),
                author=event.options.get("author", "Unknown"),
                narrator_voice=event.narrator_voice,
                provider=event.options.get("provider", "openai"),
                model=event.options.get("model", "tts-1"),
                output_format=event.output_format,
                chapter_detection=event.options.get("chapter_detection", True),
                multi_voice=event.options.get("multi_voice", False),
                character_voices=event.options.get("character_voices", {}),
                voice_settings=event.options.get("voice_settings", {}),
                background_music=event.options.get("background_music"),
                music_volume=event.options.get("music_volume", 0.1),
                chapter_pause_duration=event.options.get("chapter_pause_duration", 2.0),
                paragraph_pause_duration=event.options.get(
                    "paragraph_pause_duration", 0.5
                ),
                sentence_pause_duration=event.options.get(
                    "sentence_pause_duration", 0.3
                ),
                max_chunk_size=event.options.get("max_chunk_size", 4000),
                enable_ssml=event.options.get("enable_ssml", False),
                normalize_audio=event.options.get("normalize_audio", True),
                target_db=event.options.get("target_db", -20.0),
            )

            # Get cost estimate
            estimated_cost = generator.get_cost_estimate(audiobook_request)

            # Update UI with initial status
            if hasattr(self.app, "query_one"):
                try:
                    from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget

                    audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                    if audiobook_widget:
                        log = audiobook_widget.query_one(
                            "#audiobook-generation-log", RichLog
                        )
                        log.write(
                            "[bold yellow]Starting audiobook generation...[/bold yellow]"
                        )
                        log.write(f"Estimated cost: ${estimated_cost:.2f}")
                except Exception:
                    pass  # UI element not found, continue without UI updates

            # Define progress callback
            async def progress_callback(progress: AudioBookProgress):
                """Update UI with generation progress"""
                if hasattr(self.app, "query_one"):
                    try:
                        from tldw_chatbook.UI.STTS_Window import (
                            AudioBookGenerationWidget,
                        )

                        audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                        if audiobook_widget:
                            log = audiobook_widget.query_one(
                                "#audiobook-generation-log", RichLog
                            )

                            # Update progress message
                            if progress.current_chapter:
                                log.write(
                                    f"[cyan]Processing: {progress.current_chapter}[/cyan]"
                                )

                            # Update progress bar if available
                            if progress.total_chapters > 0:
                                percent_complete = (
                                    progress.completed_chapters
                                    / progress.total_chapters
                                ) * 100
                                log.write(
                                    f"Progress: {progress.completed_chapters}/{progress.total_chapters} chapters ({percent_complete:.1f}%)"
                                )

                            # Show time estimates
                            if progress.estimated_completion:
                                remaining_time = (
                                    progress.estimated_completion - datetime.now()
                                )
                                if remaining_time.total_seconds() > 0:
                                    minutes_remaining = int(
                                        remaining_time.total_seconds() / 60
                                    )
                                    log.write(
                                        f"Estimated time remaining: {minutes_remaining} minutes"
                                    )

                            # Show errors if any
                            for error in progress.errors:
                                log.write(f"[bold red]Error: {error}[/bold red]")
                    except Exception:
                        pass  # UI element not found, continue without UI updates

            # Generate the audiobook
            output_path = await generator.generate_audiobook(
                audiobook_request, progress_callback=progress_callback
            )

            # Update UI with completion
            if hasattr(self.app, "query_one"):
                try:
                    from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget

                    audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                    if audiobook_widget:
                        log = audiobook_widget.query_one(
                            "#audiobook-generation-log", RichLog
                        )
                        log.write(
                            "[bold green]✓ AudioBook generation complete![/bold green]"
                        )
                        log.write(f"Output file: {output_path}")
                        log.write(
                            f"Total duration: {generator.progress.actual_duration / 60:.1f} minutes"
                        )

                        # Enable export button if available
                        export_btn = audiobook_widget.query_one(
                            "#audiobook-export-btn", Button
                        )
                        if export_btn:
                            export_btn.disabled = False
                            # Store the output path for export
                            audiobook_widget.generated_audiobook_path = output_path
                except Exception:
                    pass  # UI element not found

            # Store the generated audiobook path for playback
            self._current_audio_file = output_path

            # Notify the UI widget
            if audiobook_widget and hasattr(
                audiobook_widget, "audiobook_generation_complete"
            ):
                audiobook_widget.audiobook_generation_complete(True, output_path)

            self.app.notify(
                f"AudioBook generated successfully: {output_path.name}",
                severity="information",
            )

        except ImportError as e:
            logger.error(f"Failed to import audiobook generator: {e}")
            self.app.notify(
                "AudioBook generation module not available", severity="error"
            )
        except Exception as e:
            logger.error(f"AudioBook generation failed: {e}")
            self.app.notify(f"AudioBook generation failed: {e}", severity="error")

            # Notify the UI widget of failure
            try:
                from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget

                audiobook_widget = self.app.query_one(AudioBookGenerationWidget)
                if audiobook_widget and hasattr(
                    audiobook_widget, "audiobook_generation_complete"
                ):
                    audiobook_widget.audiobook_generation_complete(False)
            except Exception:
                pass

    async def play_current_audio(self) -> None:
        """Play the current audio file"""
        audio_path = self._current_playground_audio_path()
        if audio_path is None or not audio_path.exists():
            self.app.notify("No audio file to play", severity="warning")
            return

        try:
            # Use the existing play_audio_file function
            from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
                play_audio_file,
            )

            play_audio_file(audio_path)
            self.app.notify("Playing audio...", severity="information")
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            self.app.notify(f"Failed to play audio: {e}", severity="error")

    async def export_current_audio(self, target_path: Path) -> None:
        """Export the current audio file"""
        audio_path = self._current_playground_audio_path()
        if audio_path is None or not audio_path.exists():
            self.app.notify("No audio file to export", severity="warning")
            return

        try:
            import shutil

            from tldw_chatbook.Utils.path_validation import (
                validate_filename,
                validate_path_simple,
            )

            target_path = Path(target_path)
            validate_path_simple(target_path, require_exists=False)
            validated_parent = validate_path_simple(
                target_path.parent, require_exists=True
            ).resolve()
            validated_filename = validate_filename(target_path.name)
            validated_target_path = validated_parent / validated_filename

            shutil.copy2(audio_path, validated_target_path)
            self.app.notify(
                f"Audio exported to {validated_target_path}", severity="information"
            )
        except Exception as e:
            logger.error(f"Failed to export audio: {e}")
            self.app.notify(f"Failed to export audio: {e}", severity="error")

    def _current_playground_audio_path(self) -> Path | None:
        """Return artifact provenance before the compatibility path field."""
        if self._current_playground_artifact is not None:
            return self._current_playground_artifact.path
        if self._current_audio_file is None:
            return None
        return Path(self._current_audio_file)

    def _start_event_task(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """Start and retain an event task until it finishes."""
        if self._cleanup_task is not None:
            coroutine.close()
            logger.debug("Ignoring STTS event after cleanup started")
            return
        task = asyncio.create_task(coroutine)
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def cleanup_tts_resources(self) -> None:
        """Join retained cleanup before propagating caller cancellation."""
        if self._cleanup_task is None:
            caller = asyncio.current_task()
            self._cleanup_task = asyncio.create_task(
                self._cleanup_owned_resources(caller),
                name="stts_handler_cleanup",
            )
        await _join_retained_task(self._cleanup_task)

    async def _cleanup_owned_resources(
        self,
        caller: asyncio.Task[Any] | None,
    ) -> None:
        """Cancel handler work and delete only playground-owned temporary audio."""
        tasks = tuple(task for task in self._active_tasks if task is not caller)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.difference_update(tasks)

        owned_paths = tuple(self._playground_audio_files)
        self._playground_file_leases.clear()
        for path in owned_paths:
            if secure_delete_file(path) or not path.exists():
                self._playground_audio_files.discard(path)
                for operation_id in tuple(self._playground_operation_files):
                    self._forget_operation_file(operation_id, path)

        if (
            self._current_audio_file in owned_paths
            and self._current_audio_file not in self._playground_audio_files
        ):
            self._current_audio_file = None
        if (
            self._current_playground_artifact is not None
            and self._current_playground_artifact.path
            not in self._playground_audio_files
        ):
            self._current_playground_artifact = None
        if self._generation_task is not None and self._generation_task.done():
            self._generation_task = None
        self._active_playground_operation_id = None
        self._is_generating = False

    def on_stts_playground_generate_event(
        self, event: STTSPlaygroundGenerateEvent
    ) -> None:
        """Start a retained async task for playground generation."""
        self.start_playground_generation(event)

    def on_stts_settings_save_event(self, event: STTSSettingsSaveEvent) -> None:
        """Handle settings save event"""
        self._start_event_task(self.handle_settings_save(event))

    def on_stts_provider_configuration_changed(
        self,
        event: STTSProviderConfigurationChanged,
    ) -> None:
        """Invalidate any mounted Playground for the changed provider."""
        for widget in self.app.query("TTSPlaygroundWidget"):
            callback = getattr(widget, "mark_provider_configuration_changed", None)
            if callable(callback):
                callback(event.provider_id, event.configuration_revision)

    def on_stts_audiobook_generate_event(
        self, event: STTSAudioBookGenerateEvent
    ) -> None:
        """Handle audiobook generate event"""
        self._start_event_task(self.handle_audiobook_generate(event))


#
# End of stts_events.py
#######################################################################################################################
