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
    """

    installed_engine: str = ""
    openai_key: bool = False
    parakeet_mib: int = 0


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


def openai_key_configured(
    app_config: Mapping[str, Any], environ: Mapping[str, str]
) -> bool:
    """Whether an OpenAI key exists in config or the environment (bool only).

    Resolved exactly as the transcription path resolves it
    (``config.resolve_openai_provider_key``), so Cloud is never offered on a
    key remote-whisper cannot send (review F6).
    """
    from tldw_chatbook.config import resolve_openai_provider_key
    from tldw_chatbook.UI.Wizards.first_run_setup_state import full_config_sections

    sections = full_config_sections(app_config, "api_settings", "API")
    key = resolve_openai_provider_key(
        sections["api_settings"].get("openai"),
        sections["API"].get("openai_api_key"),
        environ,
    )
    return key is not None


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
    return EngineFacts(
        installed_engine=pick_installed_engine(configured, installed_local_providers()),
        openai_key=openai_key_configured(app_config, env),
        parakeet_mib=round(parakeet_bytes / 1048576) if parakeet_bytes else 0,
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
    choices = [EngineChoice(ENGINE_NONE, "No dictation for now", NO_DICTATION_DETAIL)]
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
    default (faster-whisper or a platform MLX engine) is not a user's pick.
    """
    provider = str(transcription.get("default_provider") or "")
    if provider == PARAKEET_ONNX_PROVIDER:
        return ENGINE_PARAKEET
    remote = transcription.get("remote_whisper")
    enabled = isinstance(remote, Mapping) and bool(remote.get("enabled"))
    if provider == REMOTE_WHISPER_PROVIDER and enabled and facts.openai_key:
        return ENGINE_CLOUD
    return ENGINE_NONE


def build_engine_commit(
    engine: str, *, facts: EngineFacts, transcription: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """The write for the "installed" and "cloud" choices; ``{}`` when unchanged.

    "none" writes nothing; "parakeet" is the managed install's own commit.
    """
    provider = str(transcription.get("default_provider") or "")
    model = str(transcription.get("default_model") or "")
    if engine == ENGINE_INSTALLED and facts.installed_engine:
        target = facts.installed_engine
        target_model = ENGINE_DEFAULT_MODELS.get(target, model)
        if provider == target and model == target_model:
            return {}
        return {"transcription": {"default_provider": target, "default_model": target_model}}
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
        return {
            "transcription": {
                "default_provider": REMOTE_WHISPER_PROVIDER,
                "default_model": OPENAI_WHISPER_MODEL,
            },
            "transcription.remote_whisper": {
                "enabled": True,
                "api_endpoint": OPENAI_TRANSCRIPTIONS_URL,
                "model": OPENAI_WHISPER_MODEL,
            },
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
