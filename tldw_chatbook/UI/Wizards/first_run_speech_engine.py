"""The Speech step's engine choice (TASK-34100.13, voice-speech-15).

The step offered only a 633 MiB Parakeet download (or a model folder, or a
GGUF): no cloud Whisper for a user with an OpenAI key, and no word that an
engine already installed here (parakeet-mlx on a Mac, faster-whisper) would
work without a download. The step now opens with an engine choice built from
what is detected:

* "No dictation for now" -- preselected on a first run, writes nothing;
* "Already available -- <engine>" when a local engine imports;
* "On this computer -- Parakeet (633 MiB download)", the managed install;
* "Cloud -- OpenAI Whisper (audio is sent to OpenAI)" when an OpenAI key exists.

Only a choice made in this run writes ``[transcription]``, on Next. Pure
apart from ``detect_engine_facts`` (``find_spec`` probes, no imports).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

ENGINE_NONE = "none"
ENGINE_INSTALLED = "installed"
ENGINE_PARAKEET = "parakeet"
ENGINE_CLOUD = "cloud"

#: The managed Parakeet path's provider id; it is the "On this computer"
#: choice, never "Already available".
PARAKEET_ONNX_PROVIDER = "parakeet-onnx"
REMOTE_WHISPER_PROVIDER = "remote-whisper"
OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_WHISPER_MODEL = "whisper-1"

#: Display names for local engines.
ENGINE_NAMES: Mapping[str, str] = {
    "parakeet-mlx": "Parakeet MLX",
    "lightning-whisper-mlx": "Lightning Whisper MLX",
    "faster-whisper": "faster-whisper",
    "qwen2audio": "Qwen2-Audio",
    "parakeet": "NVIDIA Parakeet (NeMo)",
    "canary": "NVIDIA Canary (NeMo)",
}

#: Engines that take Whisper model ids. Any other engine handed one of the
#: shipped templates' Whisper models (``base`` now, ``distil-large-v3``
#: before TASK-32755.4) cannot load it: parakeet-mlx asked for ``base`` 404s.
WHISPER_ID_ENGINES = frozenset({"faster-whisper", "lightning-whisper-mlx"})
TEMPLATE_WHISPER_MODELS = frozenset({"base", "distil-large-v3"})

#: Each local engine's own default model, mirroring
#: ``TranscriptionService``'s per-provider defaults. ``[transcription]
#: default_model`` is not provider-scoped, so an engine change writes its own
#: model too: parakeet-mlx asked for a Whisper model id 404s. faster-whisper
#: takes ``base``, the shipped template's model since TASK-32755.4 and the
#: service's own fallback, so a fresh profile's pick writes nothing.
ENGINE_DEFAULT_MODELS: Mapping[str, str] = {
    "parakeet-mlx": "mlx-community/parakeet-tdt-0.6b-v2",
    "lightning-whisper-mlx": "distil-large-v3",
    "faster-whisper": "base",
    "qwen2audio": "Qwen2-Audio-7B-Instruct",
    "parakeet": "nvidia/parakeet-tdt-0.6b-v2",
    "canary": "nvidia/canary-1b-flash",
}

NO_DICTATION_DETAIL = "Nothing is changed. Set dictation up later by running setup again."
#: Fields of ``[transcription.remote_whisper]`` that name a server's own key.
REMOTE_KEY_FIELDS = ("api_key", "api_key_env_var")
CLOUD_DETAIL = (
    "Used when you transcribe audio and video files; it uses your OpenAI key. "
    "Live dictation in Console stays on this computer and needs an engine "
    "installed here."
)


@dataclass(frozen=True)
class EngineFacts:
    """What the step can offer on this machine.

    Attributes:
        installed_engine: A local engine whose runtime imports (never the
            managed Parakeet path), or "" for none.
        openai_key: Whether an OpenAI key is configured (the value is never read
            into this object).
        parakeet_mib: The managed Parakeet download, model plus VAD, in MiB.
        kept_remote: The host of a Whisper server (not OpenAI's) that
            ``[transcription]`` already sends audio files to, or "". "No
            dictation for now" leaves it alone and says so (review R1).
    """

    installed_engine: str = ""
    openai_key: bool = False
    parakeet_mib: int = 0
    kept_remote: str = ""


@dataclass(frozen=True)
class EngineChoice:
    """One row of the engine radio."""

    key: str
    label: str
    detail: str


def engine_display_name(provider_id: str) -> str:
    return ENGINE_NAMES.get(provider_id, provider_id)


def pick_installed_engine(configured: str, installed: tuple[str, ...]) -> str:
    """The local engine Console dictation would use today, if one imports.

    Mirrors ``Chat.console_voice_input``'s resolver: the configured provider
    when it is installed, else the first installed one -- excluding the
    managed Parakeet path, which has its own choice.
    """
    candidates = [name for name in installed if name != PARAKEET_ONNX_PROVIDER]
    if configured in candidates:
        return configured
    return candidates[0] if candidates else ""


def is_openai_endpoint(url: object) -> bool:
    """Whether a remote-whisper URL is OpenAI's own, over https."""
    try:
        parts = urlsplit(str(url or ""))
    except ValueError:
        return False
    return parts.scheme.lower() == "https" and (
        (parts.hostname or "").lower() == "api.openai.com"
    )


def remote_whisper_host(remote: Mapping[str, Any]) -> str:
    """The host a ``[transcription.remote_whisper]`` table sends audio to.

    The table's endpoint, or the transcription service's own default when
    none is set; only the host name, never a path or credentials in the URL.
    """
    endpoint = remote.get("api_endpoint") or "http://localhost:8000/v1/audio/transcriptions"
    try:
        return urlsplit(str(endpoint)).hostname or "an unnamed server"
    except ValueError:
        return "an unnamed server"


def remote_whisper_summary(remote: Mapping[str, Any]) -> str:
    """The Summary's words for an enabled remote-whisper: OpenAI only for OpenAI."""
    if is_openai_endpoint(remote.get("api_endpoint")):
        return "OpenAI Whisper (cloud; for audio and video files)"
    return f"Whisper server at {remote_whisper_host(remote)} (for audio and video files)"


def _remote_in_use(transcription: Mapping[str, Any]) -> Mapping[str, Any]:
    """The remote-whisper table when it is the enabled default provider, else {}."""
    if str(transcription.get("default_provider") or "") != REMOTE_WHISPER_PROVIDER:
        return {}
    remote = transcription.get("remote_whisper")
    if isinstance(remote, Mapping) and bool(remote.get("enabled")):
        return remote
    return {}


def openai_key_configured(
    app_config: Mapping[str, Any], environ: Mapping[str, str]
) -> bool:
    """Whether the OpenAI provider resolves a key (bool only).

    Review R2 (TASK-34100.13): resolved by dev's own
    ``provider_readiness.resolve_provider_credential`` -- its ADR-012
    ``credential_source`` and a named ``api_key_env_var`` included -- from
    the loaded ``[api_settings.openai]`` (which ``load_settings`` has already
    folded a legacy ``[API] openai_api_key`` into). The transcription path
    resolves the key it sends the same way, so Cloud is never offered on a
    key remote-whisper cannot send, nor on one the provider config excludes.
    """
    from tldw_chatbook.Chat.provider_readiness import resolve_provider_credential
    from tldw_chatbook.config import ProviderSettingsError, provider_settings_for_key
    from tldw_chatbook.UI.Wizards.first_run_setup_state import full_config_sections

    api_settings = full_config_sections(app_config, "api_settings")["api_settings"]
    try:
        table = provider_settings_for_key(api_settings, "openai")
    except ProviderSettingsError:
        return False
    return resolve_provider_credential("openai", table, environ=environ)[0] is not None


def detect_engine_facts(
    app_config: Mapping[str, Any],
    environ: Optional[Mapping[str, str]] = None,
    *,
    parakeet_bytes: int = 0,
) -> EngineFacts:
    """Probe this machine (``find_spec`` only) for the engine radio."""
    from tldw_chatbook.UI.Wizards.first_run_setup_state import full_config_sections
    from tldw_chatbook.Utils.local_stt_providers import installed_local_providers

    env = os.environ if environ is None else environ
    transcription = full_config_sections(app_config, "transcription")["transcription"]
    configured = str(transcription.get("default_provider") or "")
    remote = _remote_in_use(transcription)
    return EngineFacts(
        installed_engine=pick_installed_engine(configured, installed_local_providers()),
        openai_key=openai_key_configured(app_config, env),
        parakeet_mib=round(parakeet_bytes / 1048576) if parakeet_bytes else 0,
        kept_remote=(
            remote_whisper_host(remote)
            if remote and not is_openai_endpoint(remote.get("api_endpoint"))
            else ""
        ),
    )


def refresh_engine_facts(step: Any) -> None:
    """Re-detect the engine facts when the Speech step is shown.

    Live (TASK-34100.13): the step is built when the wizard opens, before
    Connect saves a key, so an OpenAI key typed in this run never offered the
    Cloud row. Facts a caller injected are left alone.

    Args:
        step: The ``SpeechSetupStep``.
    """
    if getattr(step, "_facts_injected", False):
        return
    app_config = getattr(getattr(step.wizard, "app_instance", None), "app_config", {})
    fresh = detect_engine_facts(
        app_config if isinstance(app_config, Mapping) else {},
        parakeet_bytes=step._parakeet_bytes(),
    )
    if fresh == step._engine_facts:
        return
    step._engine_facts = fresh
    if (step._engine == ENGINE_CLOUD and not fresh.openai_key) or (
        step._engine == ENGINE_INSTALLED and not fresh.installed_engine
    ):
        step._engine = ENGINE_NONE
    step.refresh(recompose=True)


def engine_choices(facts: EngineFacts) -> list[EngineChoice]:
    """The engine radio's rows, in order, built from what was detected."""
    size = f"{facts.parakeet_mib} MiB download" if facts.parakeet_mib else "download"
    none_detail = (
        "Nothing is changed: audio and video files keep going to your Whisper "
        f"server at {facts.kept_remote}."
        if facts.kept_remote
        else NO_DICTATION_DETAIL
    )
    choices = [EngineChoice(ENGINE_NONE, "No dictation for now", none_detail)]
    if facts.installed_engine:
        name = engine_display_name(facts.installed_engine)
        choices.append(
            EngineChoice(
                ENGINE_INSTALLED,
                f"Already available — {name} (installed)",
                f"{name} is installed on this computer, so there is nothing to "
                "download now; it fetches its speech model the first time you "
                "dictate.",
            )
        )
    choices.append(
        EngineChoice(
            ENGINE_PARAKEET,
            f"On this computer — Parakeet ({size})",
            "Speech-to-text that runs on this computer after a one-time download.",
        )
    )
    if facts.openai_key:
        choices.append(
            EngineChoice(
                ENGINE_CLOUD,
                "Cloud — OpenAI Whisper (audio is sent to OpenAI)",
                CLOUD_DETAIL,
            )
        )
    return choices


def saved_engine(transcription: Mapping[str, Any], facts: EngineFacts) -> str:
    """The choice a saved ``[transcription]`` amounts to; NONE when untouched.

    Only providers setup itself writes count as a saved choice: the template
    default (faster-whisper or a platform MLX engine) is not a user's pick,
    and neither is a remote-whisper pointed anywhere but OpenAI (review R1:
    a self-hosted server read as "Cloud -- OpenAI Whisper").
    """
    provider = str(transcription.get("default_provider") or "")
    if provider == PARAKEET_ONNX_PROVIDER:
        return ENGINE_PARAKEET
    remote = _remote_in_use(transcription)
    if remote and facts.openai_key and is_openai_endpoint(remote.get("api_endpoint")):
        return ENGINE_CLOUD
    return ENGINE_NONE


def _model_fits(engine: str, model: str) -> bool:
    """Whether ``model`` can be ``engine``'s own (review R4): anything but
    empty, or a shipped template's Whisper id on an engine that takes none."""
    if not model:
        return False
    return engine in WHISPER_ID_ENGINES or model not in TEMPLATE_WHISPER_MODELS


def build_engine_commit(
    engine: str, *, facts: EngineFacts, transcription: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """The write for the "installed" and "cloud" choices; ``{}`` when unchanged.

    "none" writes nothing; "parakeet" is the managed install's own commit.
    The engine's own model is written with it, except that a model the user
    already gave that same engine stays (review R4).
    """
    provider = str(transcription.get("default_provider") or "")
    model = str(transcription.get("default_model") or "")
    if engine == ENGINE_INSTALLED and facts.installed_engine:
        target = facts.installed_engine
        target_model = ENGINE_DEFAULT_MODELS.get(target, model)
        if provider != target:
            return {
                "transcription": {"default_provider": target, "default_model": target_model}
            }
        if _model_fits(target, model) or not target_model:
            return {}
        return {"transcription": {"default_model": target_model}}
    if engine == ENGINE_CLOUD and facts.openai_key:
        remote = transcription.get("remote_whisper")
        remote = remote if isinstance(remote, Mapping) else {}
        if (
            provider == REMOTE_WHISPER_PROVIDER
            and model == OPENAI_WHISPER_MODEL
            and bool(remote.get("enabled"))
            and remote.get("api_endpoint") == OPENAI_TRANSCRIPTIONS_URL
        ):
            return {}
        repointed = {
            "enabled": True,
            "api_endpoint": OPENAI_TRANSCRIPTIONS_URL,
            "model": OPENAI_WHISPER_MODEL,
        }
        if not is_openai_endpoint(remote.get("api_endpoint")):
            # Review R1: a key issued for the server being replaced (or the
            # variable naming one) would otherwise go to api.openai.com.
            repointed.update(
                {field: "" for field in REMOTE_KEY_FIELDS if str(remote.get(field) or "").strip()}
            )
        return {
            "transcription": {
                "default_provider": REMOTE_WHISPER_PROVIDER,
                "default_model": OPENAI_WHISPER_MODEL,
            },
            "transcription.remote_whisper": repointed,
        }
    return {}


#: voice-speech-12: the runtime-missing line names the extras package (no
#: pinned version, no assumed installer) and the step offers to copy it.
PARAKEET_EXTRAS_PACKAGE = "tldw_chatbook[transcription_parakeet_onnx]"
RUNTIME_MISSING_COPY = (
    "Dictation on this computer needs an optional package that isn't "
    "installed. Install the extras package below with the tool you installed "
    "chatbook with, then reopen setup. Skipping is fine — set this up later "
    "from Lab ▸ Models."
)


def recommendation_line(model_label: str, mib: int, *, chosen: bool) -> str:
    """The Parakeet card: "Recommended:" until the user picks something else."""
    prefix = "Selected" if chosen else "Recommended"
    size = f" One-time {mib} MiB download." if mib else ""
    return f"{prefix}: {model_label} — speech-to-text that runs on this computer.{size}"
