"""Exact lazy public exports; recovery declarations do not bootstrap runtime."""

from importlib import import_module

_EXPORTS = {
    "CapabilitySnapshotState": (
        "tldw_chatbook.TTS.adapter_types",
        "CapabilitySnapshotState",
    ),
    "ProgressSink": ("tldw_chatbook.TTS.adapter_types", "ProgressSink"),
    "ProviderHealth": ("tldw_chatbook.TTS.adapter_types", "ProviderHealth"),
    "TTSAudioResponse": ("tldw_chatbook.TTS.adapter_types", "TTSAudioResponse"),
    "TTSModelInfo": ("tldw_chatbook.TTS.adapter_types", "TTSModelInfo"),
    "TTSNativeCapabilitySnapshot": (
        "tldw_chatbook.TTS.adapter_types",
        "TTSNativeCapabilitySnapshot",
    ),
    "TTSOperationCode": ("tldw_chatbook.TTS.adapter_types", "TTSOperationCode"),
    "TTSOperationError": ("tldw_chatbook.TTS.adapter_types", "TTSOperationError"),
    "TTSProgress": ("tldw_chatbook.TTS.adapter_types", "TTSProgress"),
    "TTSProviderCatalog": ("tldw_chatbook.TTS.adapter_types", "TTSProviderCatalog"),
    "TTSProviderDescriptor": (
        "tldw_chatbook.TTS.adapter_types",
        "TTSProviderDescriptor",
    ),
    "TTSRequest": ("tldw_chatbook.TTS.adapter_types", "TTSRequest"),
    "TTSStructuredVoiceAdapter": (
        "tldw_chatbook.TTS.adapter_types",
        "TTSStructuredVoiceAdapter",
    ),
    "TTSVoiceDiscoveryResult": (
        "tldw_chatbook.TTS.adapter_types",
        "TTSVoiceDiscoveryResult",
    ),
    "VoiceDiscoveryState": ("tldw_chatbook.TTS.adapter_types", "VoiceDiscoveryState"),
    "NormalizationOptions": ("tldw_chatbook.TTS.audio_schemas", "NormalizationOptions"),
    "OpenAISpeechRequest": ("tldw_chatbook.TTS.audio_schemas", "OpenAISpeechRequest"),
    "AudioCppDiagnosticLine": (
        "tldw_chatbook.TTS.audio_cpp_supervisor",
        "AudioCppDiagnosticLine",
    ),
    "AudioCppProcessAdmissionSnapshot": (
        "tldw_chatbook.TTS.audio_cpp_supervisor",
        "AudioCppProcessAdmissionSnapshot",
    ),
    "AudioCppProcessFailure": (
        "tldw_chatbook.TTS.audio_cpp_supervisor",
        "AudioCppProcessFailure",
    ),
    "AudioCppProcessSnapshot": (
        "tldw_chatbook.TTS.audio_cpp_supervisor",
        "AudioCppProcessSnapshot",
    ),
    "AudioCppProcessState": (
        "tldw_chatbook.TTS.audio_cpp_supervisor",
        "AudioCppProcessState",
    ),
    "AudioCppReadyEndpoint": (
        "tldw_chatbook.TTS.audio_cpp_supervisor",
        "AudioCppReadyEndpoint",
    ),
    "AudioCppTTSCapability": (
        "tldw_chatbook.TTS.audio_cpp_supervisor",
        "AudioCppTTSCapability",
    ),
    "CharacterTTSRequestResolution": (
        "tldw_chatbook.TTS.character_request_resolver",
        "CharacterTTSRequestResolution",
    ),
    "CharacterTTSRequestResolver": (
        "tldw_chatbook.TTS.character_request_resolver",
        "CharacterTTSRequestResolver",
    ),
    "CharacterTTSResolutionError": (
        "tldw_chatbook.TTS.character_request_resolver",
        "CharacterTTSResolutionError",
    ),
    "CharacterTTSResolutionSource": (
        "tldw_chatbook.TTS.character_request_resolver",
        "CharacterTTSResolutionSource",
    ),
    "STTSGeneratedAudio": ("tldw_chatbook.TTS.playground_types", "STTSGeneratedAudio"),
    "STTSPlaygroundResultProjection": (
        "tldw_chatbook.TTS.playground_types",
        "STTSPlaygroundResultProjection",
    ),
    "STTSPlaygroundCloneSnapshot": (
        "tldw_chatbook.TTS.playground_types",
        "STTSPlaygroundCloneSnapshot",
    ),
    "STTSPlaygroundProfilePreview": (
        "tldw_chatbook.TTS.playground_types",
        "STTSPlaygroundProfilePreview",
    ),
    "STTSPlaygroundRequest": (
        "tldw_chatbook.TTS.playground_types",
        "STTSPlaygroundRequest",
    ),
    "TTSRequestedSelectionSnapshot": (
        "tldw_chatbook.TTS.playground_types",
        "TTSRequestedSelectionSnapshot",
    ),
    "TTSConfigMutation": ("tldw_chatbook.TTS.preferences", "TTSConfigMutation"),
    "TTSPreferencesSnapshot": (
        "tldw_chatbook.TTS.preferences",
        "TTSPreferencesSnapshot",
    ),
    "ProfileRepositoryError": (
        "tldw_chatbook.TTS.profile_errors",
        "ProfileRepositoryError",
    ),
    "ProfileServiceError": ("tldw_chatbook.TTS.profile_errors", "ProfileServiceError"),
    "ProfileValidationError": (
        "tldw_chatbook.TTS.profile_errors",
        "ProfileValidationError",
    ),
    "TTSProfileRepository": (
        "tldw_chatbook.TTS.profile_repository",
        "TTSProfileRepository",
    ),
    "CanonicalTTSCloneReference": (
        "tldw_chatbook.TTS.profile_reference_types",
        "CanonicalTTSCloneReference",
    ),
    "TTSCloneReference": (
        "tldw_chatbook.TTS.profile_reference_types",
        "TTSCloneReference",
    ),
    "TTSCloneRecipeRequirement": (
        "tldw_chatbook.TTS.profile_reference_types",
        "TTSCloneRecipeRequirement",
    ),
    "TTSCloneReferenceSummary": (
        "tldw_chatbook.TTS.profile_reference_types",
        "TTSCloneReferenceSummary",
    ),
    "LoadedCharacterTTSAssignment": (
        "tldw_chatbook.TTS.profile_service",
        "LoadedCharacterTTSAssignment",
    ),
    "LoadedTTSProfile": ("tldw_chatbook.TTS.profile_service", "LoadedTTSProfile"),
    "PortableProfileAvailabilityObservation": (
        "tldw_chatbook.TTS.profile_service",
        "PortableProfileAvailabilityObservation",
    ),
    "PortableProfileImportPlan": (
        "tldw_chatbook.TTS.profile_service",
        "PortableProfileImportPlan",
    ),
    "PortableProfileImportResult": (
        "tldw_chatbook.TTS.profile_service",
        "PortableProfileImportResult",
    ),
    "ProfileAvailabilityState": (
        "tldw_chatbook.TTS.profile_service",
        "ProfileAvailabilityState",
    ),
    "TTSPlaygroundSelectionPreset": (
        "tldw_chatbook.TTS.profile_service",
        "TTSPlaygroundSelectionPreset",
    ),
    "TTSProfileAvailability": (
        "tldw_chatbook.TTS.profile_service",
        "TTSProfileAvailability",
    ),
    "TTSProfileAvailabilitySnapshot": (
        "tldw_chatbook.TTS.profile_service",
        "TTSProfileAvailabilitySnapshot",
    ),
    "TTSProfilePageSnapshot": (
        "tldw_chatbook.TTS.profile_service",
        "TTSProfilePageSnapshot",
    ),
    "TTSProfileService": ("tldw_chatbook.TTS.profile_service", "TTSProfileService"),
    "AssignedTTSProfileSnapshot": (
        "tldw_chatbook.TTS.profile_types",
        "AssignedTTSProfileSnapshot",
    ),
    "CharacterRef": ("tldw_chatbook.TTS.profile_types", "CharacterRef"),
    "CharacterTTSAssignment": (
        "tldw_chatbook.TTS.profile_types",
        "CharacterTTSAssignment",
    ),
    "ProfileBackupReceipt": ("tldw_chatbook.TTS.profile_types", "ProfileBackupReceipt"),
    "ProfileRepositoryState": (
        "tldw_chatbook.TTS.profile_types",
        "ProfileRepositoryState",
    ),
    "ProfileRestoreReceipt": (
        "tldw_chatbook.TTS.profile_types",
        "ProfileRestoreReceipt",
    ),
    "ProfileStoreResult": ("tldw_chatbook.TTS.profile_types", "ProfileStoreResult"),
    "TTSGenerationProfile": ("tldw_chatbook.TTS.profile_types", "TTSGenerationProfile"),
    "TTSProfileDraft": ("tldw_chatbook.TTS.profile_types", "TTSProfileDraft"),
    "TTSProfilePage": ("tldw_chatbook.TTS.profile_types", "TTSProfilePage"),
    "canonical_json_options": (
        "tldw_chatbook.TTS.profile_types",
        "canonical_json_options",
    ),
    "AudioCppCloneSetupProjection": (
        "tldw_chatbook.TTS.TTS_Generation",
        "AudioCppCloneSetupProjection",
    ),
    "AudioCppRuntimeObservation": (
        "tldw_chatbook.TTS.TTS_Generation",
        "AudioCppRuntimeObservation",
    ),
    "TTSService": ("tldw_chatbook.TTS.TTS_Generation", "TTSService"),
    "bind_tts_service": ("tldw_chatbook.TTS.TTS_Generation", "bind_tts_service"),
    "close_tts_resources": ("tldw_chatbook.TTS.TTS_Generation", "close_tts_resources"),
    "get_tts_service": ("tldw_chatbook.TTS.TTS_Generation", "get_tts_service"),
    "reset_tts_service_binding": (
        "tldw_chatbook.TTS.TTS_Generation",
        "reset_tts_service_binding",
    ),
    "TTSCloneVoiceBundle": (
        "tldw_chatbook.TTS.voice_bundle_codec",
        "TTSCloneVoiceBundle",
    ),
    "TTSVoiceBundleError": (
        "tldw_chatbook.TTS.voice_bundle_codec",
        "TTSVoiceBundleError",
    ),
    "TTSVoiceBundleSinks": (
        "tldw_chatbook.TTS.voice_bundle_codec",
        "TTSVoiceBundleSinks",
    ),
    "encode_clone_voice_bundle": (
        "tldw_chatbook.TTS.voice_bundle_codec",
        "encode_clone_voice_bundle",
    ),
    "inspect_clone_voice_bundle": (
        "tldw_chatbook.TTS.voice_bundle_codec",
        "inspect_clone_voice_bundle",
    ),
    "TTSVoiceBundleHandle": (
        "tldw_chatbook.TTS.voice_bundle_service",
        "TTSVoiceBundleHandle",
    ),
    "TTSVoiceBundleImportChoice": (
        "tldw_chatbook.TTS.voice_bundle_service",
        "TTSVoiceBundleImportChoice",
    ),
    "TTSVoiceBundleImportResult": (
        "tldw_chatbook.TTS.voice_bundle_service",
        "TTSVoiceBundleImportResult",
    ),
    "TTSVoiceBundlePortabilityService": (
        "tldw_chatbook.TTS.voice_bundle_service",
        "TTSVoiceBundlePortabilityService",
    ),
    "TTSVoiceBundleReview": (
        "tldw_chatbook.TTS.voice_bundle_service",
        "TTSVoiceBundleReview",
    ),
}

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


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from tldw_chatbook.Backup_Recovery.storage_admission import admit_startup

    admit_startup()
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
