from typing import TYPE_CHECKING

from tldw_chatbook.TTS.adapter_types import (
    CapabilitySnapshotState,
    ProgressSink,
    ProviderHealth,
    TTSAudioResponse,
    TTSModelInfo,
    TTSNativeCapabilitySnapshot,
    TTSOperationCode,
    TTSOperationError,
    TTSProgress,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSRequest,
    TTSStructuredVoiceAdapter,
    TTSVoiceDiscoveryResult,
    VoiceDiscoveryState,
)
from tldw_chatbook.TTS.audio_schemas import NormalizationOptions, OpenAISpeechRequest
from tldw_chatbook.TTS.audio_cpp_supervisor import (
    AudioCppDiagnosticLine,
    AudioCppProcessAdmissionSnapshot,
    AudioCppProcessFailure,
    AudioCppProcessSnapshot,
    AudioCppProcessState,
    AudioCppReadyEndpoint,
    AudioCppTTSCapability,
)
from tldw_chatbook.TTS.character_request_resolver import (
    CharacterTTSRequestResolution,
    CharacterTTSRequestResolver,
    CharacterTTSResolutionError,
    CharacterTTSResolutionSource,
)
from tldw_chatbook.TTS.playground_types import (
    STTSGeneratedAudio,
    STTSPlaygroundResultProjection,
    STTSPlaygroundCloneSnapshot,
    STTSPlaygroundProfilePreview,
    STTSPlaygroundRequest,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.preferences import TTSConfigMutation, TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_errors import (
    ProfileRepositoryError,
    ProfileServiceError,
    ProfileValidationError,
)
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneReference,
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.profile_service import (
    LoadedCharacterTTSAssignment,
    LoadedTTSProfile,
    PortableProfileAvailabilityObservation,
    PortableProfileImportPlan,
    PortableProfileImportResult,
    ProfileAvailabilityState,
    TTSPlaygroundSelectionPreset,
    TTSProfileAvailability,
    TTSProfileAvailabilitySnapshot,
    TTSProfilePageSnapshot,
    TTSProfileService,
)
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    ProfileBackupReceipt,
    ProfileRepositoryState,
    ProfileRestoreReceipt,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileDraft,
    TTSProfilePage,
    canonical_json_options,
)
from tldw_chatbook.TTS.TTS_Generation import (
    AudioCppCloneSetupProjection,
    AudioCppRuntimeObservation,
    TTSService,
    bind_tts_service,
    close_tts_resources,
    get_tts_service,
    reset_tts_service_binding,
)
from tldw_chatbook.TTS.voice_bundle_codec import (
    TTSCloneVoiceBundle,
    TTSVoiceBundleError,
    TTSVoiceBundleSinks,
    encode_clone_voice_bundle,
    inspect_clone_voice_bundle,
)
# TASK-21108: `voice_bundle_service` (1,857 lines) is the one member of this
# package nothing needs before first paint -- `app.py` builds the portability
# service on first use and `UI/stts_profile_library` is the only other
# consumer -- yet this eager package init put it on the
# `import tldw_chatbook.app` path, because `from tldw_chatbook.TTS import
# TTSProfileService` executes the whole file. The five names below are served
# by the PEP 562 `__getattr__` at the bottom instead, so the public package
# API is unchanged and the module loads on first attribute access.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.TTS.voice_bundle_service import (
        TTSVoiceBundleHandle,
        TTSVoiceBundleImportChoice,
        TTSVoiceBundleImportResult,
        TTSVoiceBundlePortabilityService,
        TTSVoiceBundleReview,
    )

_LAZY_VOICE_BUNDLE_SERVICE_NAMES = frozenset(
    {
        "TTSVoiceBundleHandle",
        "TTSVoiceBundleImportChoice",
        "TTSVoiceBundleImportResult",
        "TTSVoiceBundlePortabilityService",
        "TTSVoiceBundleReview",
    }
)


def __getattr__(name: str) -> object:
    """Resolve the deferred voice-bundle-service exports on first access.

    Args:
        name: The attribute requested from this package.

    Returns:
        object: The attribute, imported from ``voice_bundle_service`` and
        cached in the module globals so later reads skip this hook.

    Raises:
        AttributeError: For any other name, so ``from tldw_chatbook.TTS
            import <submodule>`` still falls through to the normal submodule
            import machinery.
    """
    if name in _LAZY_VOICE_BUNDLE_SERVICE_NAMES:
        from tldw_chatbook.TTS import voice_bundle_service

        value = getattr(voice_bundle_service, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the eager and deferred names this package serves."""
    return sorted(set(globals()) | _LAZY_VOICE_BUNDLE_SERVICE_NAMES)


__all__ = [
    "AssignedTTSProfileSnapshot",
    "AudioCppDiagnosticLine",
    "AudioCppCloneSetupProjection",
    "AudioCppProcessAdmissionSnapshot",
    "AudioCppProcessFailure",
    "AudioCppProcessSnapshot",
    "AudioCppProcessState",
    "AudioCppReadyEndpoint",
    "AudioCppRuntimeObservation",
    "AudioCppTTSCapability",
    "CapabilitySnapshotState",
    "CanonicalTTSCloneReference",
    "CharacterRef",
    "CharacterTTSRequestResolution",
    "CharacterTTSRequestResolver",
    "CharacterTTSResolutionError",
    "CharacterTTSResolutionSource",
    "CharacterTTSAssignment",
    "LoadedCharacterTTSAssignment",
    "LoadedTTSProfile",
    "NormalizationOptions",
    "OpenAISpeechRequest",
    "ProfileAvailabilityState",
    "ProfileBackupReceipt",
    "ProfileRepositoryError",
    "ProfileRepositoryState",
    "ProfileRestoreReceipt",
    "ProfileServiceError",
    "ProfileStoreResult",
    "ProfileValidationError",
    "PortableProfileAvailabilityObservation",
    "PortableProfileImportPlan",
    "PortableProfileImportResult",
    "ProgressSink",
    "ProviderHealth",
    "STTSGeneratedAudio",
    "STTSPlaygroundResultProjection",
    "STTSPlaygroundCloneSnapshot",
    "STTSPlaygroundProfilePreview",
    "STTSPlaygroundRequest",
    "TTSAudioResponse",
    "TTSConfigMutation",
    "TTSGenerationProfile",
    "TTSCloneReference",
    "TTSCloneRecipeRequirement",
    "TTSCloneReferenceSummary",
    "TTSCloneVoiceBundle",
    "TTSModelInfo",
    "TTSNativeCapabilitySnapshot",
    "TTSOperationCode",
    "TTSOperationError",
    "TTSPlaygroundSelectionPreset",
    "TTSPreferencesSnapshot",
    "TTSProfileAvailability",
    "TTSProfileAvailabilitySnapshot",
    "TTSProfileDraft",
    "TTSProfilePage",
    "TTSProfilePageSnapshot",
    "TTSProfileRepository",
    "TTSProfileService",
    "TTSProgress",
    "TTSProviderCatalog",
    "TTSProviderDescriptor",
    "TTSRequest",
    "TTSRequestedSelectionSnapshot",
    "TTSService",
    "TTSStructuredVoiceAdapter",
    "TTSVoiceDiscoveryResult",
    "TTSVoiceBundleError",
    "TTSVoiceBundleSinks",
    "TTSVoiceBundleHandle",
    "TTSVoiceBundleImportChoice",
    "TTSVoiceBundleImportResult",
    "TTSVoiceBundlePortabilityService",
    "TTSVoiceBundleReview",
    "VoiceDiscoveryState",
    "bind_tts_service",
    "canonical_json_options",
    "close_tts_resources",
    "get_tts_service",
    "encode_clone_voice_bundle",
    "inspect_clone_voice_bundle",
    "reset_tts_service_binding",
]
