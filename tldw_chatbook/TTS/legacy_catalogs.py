from __future__ import annotations

from types import MappingProxyType

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
# Official Kokoro v1.0 voice packs (hexgrad/Kokoro-82M).
KOKORO_VOICE_OPTIONS = (
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
    ("Echo (US Male)", "am_echo"),
    ("Eric (US Male)", "am_eric"),
    ("Fenrir (US Male)", "am_fenrir"),
    ("Liam (US Male)", "am_liam"),
    ("Michael (US Male)", "am_michael"),
    ("Onyx (US Male)", "am_onyx"),
    ("Puck (US Male)", "am_puck"),
    ("Santa (US Male)", "am_santa"),
    ("Alice (UK Female)", "bf_alice"),
    ("Emma (UK Female)", "bf_emma"),
    ("Isabella (UK Female)", "bf_isabella"),
    ("Lily (UK Female)", "bf_lily"),
    ("Daniel (UK Male)", "bm_daniel"),
    ("Fable (UK Male)", "bm_fable"),
    ("George (UK Male)", "bm_george"),
    ("Lewis (UK Male)", "bm_lewis"),
    ("Dora (Spanish Female)", "ef_dora"),
    ("Alex (Spanish Male)", "em_alex"),
    ("Santa (Spanish Male)", "em_santa"),
    ("Siwis (French Female)", "ff_siwis"),
    ("Alpha (Hindi Female)", "hf_alpha"),
    ("Beta (Hindi Female)", "hf_beta"),
    ("Omega (Hindi Male)", "hm_omega"),
    ("Psi (Hindi Male)", "hm_psi"),
    ("Sara (Italian Female)", "if_sara"),
    ("Nicola (Italian Male)", "im_nicola"),
    ("Alpha (Japanese Female)", "jf_alpha"),
    ("Gongitsune (Japanese Female)", "jf_gongitsune"),
    ("Nezumi (Japanese Female)", "jf_nezumi"),
    ("Tebukuro (Japanese Female)", "jf_tebukuro"),
    ("Kumo (Japanese Male)", "jm_kumo"),
    ("Dora (Brazilian Portuguese Female)", "pf_dora"),
    ("Alex (Brazilian Portuguese Male)", "pm_alex"),
    ("Santa (Brazilian Portuguese Male)", "pm_santa"),
    ("Xiaobei (Mandarin Female)", "zf_xiaobei"),
    ("Xiaoni (Mandarin Female)", "zf_xiaoni"),
    ("Xiaoxiao (Mandarin Female)", "zf_xiaoxiao"),
    ("Xiaoyi (Mandarin Female)", "zf_xiaoyi"),
    ("Yunjian (Mandarin Male)", "zm_yunjian"),
    ("Yunxi (Mandarin Male)", "zm_yunxi"),
    ("Yunxia (Mandarin Male)", "zm_yunxia"),
    ("Yunyang (Mandarin Male)", "zm_yunyang"),
)
KOKORO_VOICES = tuple(voice for _label, voice in KOKORO_VOICE_OPTIONS)
ALLTALK_VOICES = ("alloy", "echo", "fable", "nova", "onyx", "shimmer")

LEGACY_DEFAULT_VOICES = {
    "openai": "alloy",
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",
    "kokoro": "af_alloy",
    "chatterbox": "default",
    "higgs": "professional_female",
    "alltalk": "alloy",
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
    "kokoro": KOKORO_VOICE_OPTIONS,
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
    "alltalk": tuple((voice.title(), voice) for voice in ALLTALK_VOICES),
}

_ALL_VISIBLE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
_VOICES = {
    "openai": OPENAI_VOICES,
    "elevenlabs": ELEVENLABS_VOICES,
    "kokoro": KOKORO_VOICES,
    "chatterbox": ("default",),
    "higgs": ("default",),
    "alltalk": ALLTALK_VOICES,
}
LEGACY_REQUEST_OPTION_KEYS = MappingProxyType(
    {
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
)


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
                supports_options=LEGACY_REQUEST_OPTION_KEYS[provider_id],
            )
            for model_id in models
        ),
        approximate=True,
    )
