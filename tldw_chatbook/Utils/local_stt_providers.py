"""The one list of local speech-to-text providers, for every layer that needs it.

Two layers used to keep their own answer to "which providers are local?" and
they drifted, twice:

* `LazyLiveDictationService._initialize_streaming_transcriber()`'s privacy
  allowlist once spelled a provider `"lightning-whisper"` while the rest of the
  app dispatched on `"lightning-whisper-mlx"`, so privacy mode silently
  rewrote that user's provider to `parakeet-mlx`.
* That same allowlist then held three providers while the Console's resolver
  had grown to seven. The Console would warm (and announce) `parakeet-onnx`,
  and `start_dictation()` would afterwards rewrite the provider to
  `parakeet-mlx` -- so a first press downloaded model A for minutes and then
  transcribed with model B, and on Linux with only `onnx_asr` installed every
  chunk failed outright.

Both are the same defect: two copies of one fact. This module is that fact.
Everything here is deliberately free of heavy imports -- detection is
`importlib.util.find_spec` only, never a real import -- so the Console can
consult it at app start without dragging in faster-whisper, NeMo or torch.

`remote-whisper` is absent on purpose. It needs only `requests`, so it would
always resolve as "installed", and it sends audio off the machine, which is
exactly what privacy mode exists to prevent. Do not "complete the set".
"""

from __future__ import annotations

import importlib.util
import sys

# Provider id -> required import name(s), ALL of which must resolve for the
# provider to count as installed. Mirrors `get_available_providers()` in
# `Local_Ingestion/transcription_service.py` -- same providers, same
# declaration order (which is also the Console resolver's fallback preference),
# same detection rule per provider:
#   - parakeet-onnx           find_spec("onnx_asr")
#   - parakeet-mlx            find_spec("parakeet_mlx"), darwin only
#   - lightning-whisper-mlx   find_spec("lightning_whisper_mlx"), darwin only
#   - faster-whisper          find_spec("faster_whisper")
#   - qwen2audio              find_spec("torch") AND find_spec("transformers")
#   - parakeet / canary       find_spec("nemo") (NVIDIA NeMo)
LOCAL_PROVIDER_MODULES: dict[str, tuple[str, ...]] = {
    "parakeet-onnx": ("onnx_asr",),
    "parakeet-mlx": ("parakeet_mlx",),
    "lightning-whisper-mlx": ("lightning_whisper_mlx",),
    "faster-whisper": ("faster_whisper",),
    "qwen2audio": ("torch", "transformers"),
    "parakeet": ("nemo",),
    "canary": ("nemo",),
}

#: Every provider that runs entirely on this machine, in preference order.
#: This is what privacy mode (`dictation.privacy.local_only`) means by "local",
#: and what the Console resolver may choose from. One tuple, both consumers.
LOCAL_STT_PROVIDERS: tuple[str, ...] = tuple(LOCAL_PROVIDER_MODULES)

#: Providers usable only on Apple Silicon. Mirrors
#: `transcription_service._optional_module_available()`'s
#: `sys.platform == "darwin"` gate: a force-installed package on Linux must not
#: be reported as usable, or the mic button lights up and then fails at capture.
DARWIN_ONLY_PROVIDERS: frozenset[str] = frozenset(
    {"parakeet-mlx", "lightning-whisper-mlx"}
)


def module_installed(module_name: str) -> bool:
    """Return True when `module_name` is importable, without importing it.

    `find_spec` is required here rather than `optional_deps.check_dependency`,
    which really imports the module and would drag torch/NeMo into app start.

    Args:
        module_name: Dotted module name to look for.

    Returns:
        True when a spec resolves for it.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        # A namespace package with a broken parent raises rather than
        # returning None; treat that as "not usable".
        return False


def provider_is_local(provider: str) -> bool:
    """Return True when `provider` runs entirely on this machine.

    Args:
        provider: A transcription provider id.

    Returns:
        True when the provider never sends audio off the machine.
    """
    return provider in LOCAL_PROVIDER_MODULES


def provider_installed(provider: str) -> bool:
    """Return True when `provider`'s required module(s) all resolve.

    Darwin-only providers additionally require `sys.platform == "darwin"`,
    checked before touching `find_spec` at all so a non-darwin platform never
    even looks at whether the module happens to be importable.

    Args:
        provider: A transcription provider id.

    Returns:
        True when the provider is usable on this machine right now.
    """
    module_names = LOCAL_PROVIDER_MODULES.get(provider)
    if module_names is None:
        return False
    if provider in DARWIN_ONLY_PROVIDERS and sys.platform != "darwin":
        return False
    return all(module_installed(name) for name in module_names)


def installed_local_providers() -> tuple[str, ...]:
    """Return the local transcription providers that are actually installed.

    Returns:
        Installed provider ids in `LOCAL_PROVIDER_MODULES` declaration order.
    """
    return tuple(
        provider for provider in LOCAL_PROVIDER_MODULES if provider_installed(provider)
    )


#: Provider ids persisted by the retired and retained dictation
#: implementations before being corrected to their real dispatch id.
#: `"lightning-whisper"` was a misspelling of `"lightning-whisper-mlx"`.
#: Read-side code must still translate an already-saved value without
#: rewriting the user's configuration as a side effect of loading it.
LEGACY_PROVIDER_IDS: dict[str, str] = {
    "lightning-whisper": "lightning-whisper-mlx",
}


def normalize_provider_id(provider: str | None) -> str | None:
    """Translate a persisted legacy provider id to its current dispatch id.

    Intended for read-side use only (e.g. loading `dictation.provider` from
    config) -- callers must not write the normalized value back to config as
    a side effect of merely reading it.

    Args:
        provider: A transcription provider id as read from config, or None.

    Returns:
        The current provider id `provider` maps to via `LEGACY_PROVIDER_IDS`,
        or `provider` unchanged when it is not a known legacy alias
        (including when it is None).
    """
    if provider is None:
        return None
    return LEGACY_PROVIDER_IDS.get(provider, provider)
