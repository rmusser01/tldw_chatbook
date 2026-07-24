from __future__ import annotations

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSProviderCatalog,
)

ELEVENLABS_MODELS = (
    "eleven_monolingual_v1",
    "eleven_multilingual_v1",
    "eleven_multilingual_v2",
    "eleven_turbo_v2",
    "eleven_turbo_v2_5",
    "eleven_flash_v2",
    "eleven_flash_v2_5",
    "english_v1",
    "elevenlabs",
)
LEGACY_MODELS = {
    "openai": ("tts-1", "tts-1-hd"),
    "elevenlabs": ELEVENLABS_MODELS[:7],
    "kokoro": ("kokoro",),
    "chatterbox": ("chatterbox",),
    "higgs": ("higgs-audio-v2",),
    "alltalk": ("alltalk",),
}
OPENAI_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
)
ELEVENLABS_VOICES = (
    "21m00Tcm4TlvDq8ikWAM",
    "AZnzlk1XvdvUeBnXmlld",
    "EXAVITQu4vr4xnSDxMaL",
    "ErXwobaYiN019PkySvjV",
    "MF3mGyEYCl7XYWbV9V6O",
    "TxGEqnHWrfWFTfGW9XjX",
    "VR6AewLTigWG4xSOukaG",
    "pNInz6obpgDQGcFmaJgB",
    "yoZ06aMxZJJ28mfd3POQ",
)
KOKORO_VOICES = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
)

_ALL_VISIBLE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
_VOICES = {
    "openai": OPENAI_VOICES,
    "elevenlabs": ELEVENLABS_VOICES,
    "kokoro": KOKORO_VOICES,
    "chatterbox": ("default",),
    "higgs": ("default",),
    "alltalk": ("female_01.wav", "male_01.wav"),
}
_OPTIONS = {
    "openai": (),
    "elevenlabs": (
        "stability",
        "similarity_boost",
        "style",
        "use_speaker_boost",
    ),
    "kokoro": ("language", "use_onnx"),
    "chatterbox": (
        "exaggeration",
        "cfg_weight",
        "temperature",
        "num_candidates",
        "validate_with_whisper",
    ),
    "higgs": (
        "temperature",
        "top_p",
        "repetition_penalty",
        "language",
    ),
    "alltalk": ("language",),
}


def legacy_catalog(provider_id: str) -> TTSProviderCatalog:
    models = LEGACY_MODELS.get(provider_id)
    if models is None:
        raise KeyError(f"Unknown legacy provider: {provider_id}")
    return TTSProviderCatalog(
        provider_id=provider_id,
        revision=1,
        health=ProviderHealth(state="available", fresh=True),
        models=tuple(
            TTSModelInfo(
                model_id=model_id,
                display_name=model_id.replace("_", " ").title(),
                family=provider_id,
                upstream_mode="legacy",
                formats=_ALL_VISIBLE_FORMATS,
                voices=_VOICES[provider_id],
                supports_speed=True,
                supports_options=_OPTIONS[provider_id],
            )
            for model_id in models
        ),
        approximate=True,
    )
