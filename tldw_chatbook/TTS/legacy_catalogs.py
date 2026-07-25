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
LEGACY_DEFAULT_MODELS = {
    "openai": "tts-1",
    "elevenlabs": "eleven_multilingual_v2",
    "kokoro": "kokoro",
    "chatterbox": "chatterbox",
    "higgs": "higgs-audio-v2",
    "alltalk": "alltalk",
}
LEGACY_MODEL_LABELS = {
    "openai": {
        "tts-1": "TTS-1 (Standard)",
        "tts-1-hd": "TTS-1-HD (High Quality)",
    },
    "elevenlabs": {
        "eleven_monolingual_v1": "Eleven Monolingual v1",
        "eleven_multilingual_v1": "Eleven Multilingual v1",
        "eleven_multilingual_v2": "Eleven Multilingual v2 (Default)",
        "eleven_turbo_v2": "Eleven Turbo v2",
        "eleven_turbo_v2_5": "Eleven Turbo v2.5",
        "eleven_flash_v2": "Eleven Flash v2 (Low Latency)",
        "eleven_flash_v2_5": "Eleven Flash v2.5 (Ultra Low Latency)",
    },
    "kokoro": {"kokoro": "Kokoro 82M"},
    "chatterbox": {"chatterbox": "Chatterbox 0.5B"},
    "higgs": {"higgs-audio-v2": "Higgs Audio V2 3B"},
    "alltalk": {"alltalk": "AllTalk TTS"},
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
LEGACY_DEFAULT_VOICES = {
    "openai": "alloy",
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",
    "kokoro": "af_alloy",
    "chatterbox": "default",
    "higgs": "professional_female",
    "alltalk": "female_01.wav",
}
LEGACY_VOICE_OPTIONS = {
    "openai": tuple((voice.title(), voice) for voice in OPENAI_VOICES),
    "elevenlabs": (
        ("Rachel", "21m00Tcm4TlvDq8ikWAM"),
        ("Domi", "AZnzlk1XvdvUeBnXmlld"),
        ("Bella", "EXAVITQu4vr4xnSDxMaL"),
        ("Antoni", "ErXwobaYiN019PkySvjV"),
        ("Elli", "MF3mGyEYCl7XYWbV9V6O"),
        ("Josh", "TxGEqnHWrfWFTfGW9XjX"),
        ("Arnold", "VR6AewLTigWG4xSOukaG"),
        ("Adam", "pNInz6obpgDQGcFmaJgB"),
        ("Sam", "yoZ06aMxZJJ28mfd3POQ"),
    ),
    "kokoro": (
        ("Alloy (US Female)", "af_alloy"),
        ("Aoede (US Female)", "af_aoede"),
        ("Bella (US Female)", "af_bella"),
        ("Heart (US Female)", "af_heart"),
        ("Jessica (US Female)", "af_jessica"),
        ("Kore (US Female)", "af_kore"),
        ("Nicole (US Female)", "af_nicole"),
        ("Nova (US Female)", "af_nova"),
        ("River (US Female)", "af_river"),
        ("Sarah (US Female)", "af_sarah"),
        ("Sky (US Female)", "af_sky"),
        ("Adam (US Male)", "am_adam"),
        ("Michael (US Male)", "am_michael"),
        ("Emma (UK Female)", "bf_emma"),
        ("Isabella (UK Female)", "bf_isabella"),
        ("George (UK Male)", "bm_george"),
        ("Lewis (UK Male)", "bm_lewis"),
    ),
    "chatterbox": (
        ("Default Voice", "default"),
        ("Upload Reference Audio", "custom"),
    ),
    "higgs": (
        ("Professional Female", "professional_female"),
        ("Warm Female", "warm_female"),
        ("Storyteller Male", "storyteller_male"),
        ("Deep Male", "deep_male"),
        ("Energetic Female", "energetic_female"),
        ("Soft Female", "soft_female"),
        ("Upload Reference Audio", "custom"),
    ),
    "alltalk": (
        ("Female 01", "female_01.wav"),
        ("Female 02", "female_02.wav"),
        ("Female 03", "female_03.wav"),
        ("Female 04", "female_04.wav"),
        ("Male 01", "male_01.wav"),
        ("Male 02", "male_02.wav"),
        ("Male 03", "male_03.wav"),
        ("Male 04", "male_04.wav"),
    ),
}

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
                display_name=LEGACY_MODEL_LABELS[provider_id][model_id],
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
