"""Resolve the Analyze-after-import provider for Library ingest (task-3301).

The ingest canvas advertises "Analyze after import", but the local pipeline
never carried a provider or credential: ``_ingest_job_options`` forwarded
only the boolean, and every processor gates analysis on ``api_name`` -- so
the option was a silent no-op. This module is the single seam that turns
the app's *existing* configured analysis default into something the
pipeline can act on:

* the provider comes from ``[analysis_defaults] provider`` -- the same
  config the Media viewer's analysis panel already reads as its default;
* readiness + credential resolution go through
  ``Chat/provider_readiness.get_provider_readiness`` -- the one definition
  of "this provider can actually be called" shared with Console, covering
  ``api_settings.<provider>.api_key``, the conventional environment
  variable, and the keyless local providers.

Both the ingest panel (pre-Start hint), the job-option builder (skip
reason), and the queue's done-row note consume the same resolution, so the
promise made before Start and the record left after it can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from tldw_chatbook.Chat.provider_readiness import (
    get_provider_readiness,
    provider_config_key,
)


#: Short reason recorded on a job when ``[analysis_defaults]`` names no
#: provider at all. Also the substring the panel hint builds on.
NO_ANALYSIS_PROVIDER_REASON = "no analysis provider is configured"

# ---------------------------------------------------------------------------
# (task-3301 xhigh review round, F10) The full [analysis_defaults] call
# shape. Defaults mirror the Media viewer's analysis panel
# (`Widgets/Media/media_viewer_panel.py::populate_providers`), which seeds
# its inputs from the same config section -- so an ingest analysis and a
# viewer analysis run with identical settings under identical config.
# ---------------------------------------------------------------------------
ANALYSIS_DEFAULT_TEMPERATURE = 0.7
ANALYSIS_DEFAULT_TOP_P = 0.95
ANALYSIS_DEFAULT_MIN_P = 0.05
ANALYSIS_DEFAULT_MAX_TOKENS = 4096
ANALYSIS_DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant specialized in analyzing media content."
)


def chat_dispatch_name(provider: Optional[str]) -> Optional[str]:
    """Map a configured provider spelling onto a chat-dispatchable name.

    (task-3301 xhigh review round, F5) ``get_provider_readiness`` accepts
    display spellings (``"MistralAI"``, ``"KoboldCpp"``, ``"MLX-LM"``, ...)
    and can mark ANY provider ready once a credential is configured -- but
    ``chat_api_call`` only dispatches names in its ``API_CALL_HANDLERS``
    table. This is the one place those two universes are reconciled: a
    resolution is only ever "ready" with a name this function vouches for.

    Args:
        provider: The configured provider name in any spelling.

    Returns:
        The exact ``API_CALL_HANDLERS`` key to dispatch with, or ``None``
        when the provider has no chat handler at all.
    """
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    candidate = (provider or "").strip().lower()
    if not candidate:
        return None
    if candidate in API_CALL_HANDLERS:
        return candidate
    # Normalize both sides the way the readiness layer does (spaces and
    # hyphens to underscores) so display spellings land on handler keys
    # regardless of which separator each uses ("MLX-LM" -> "mlx_lm",
    # "Local-LLM" -> "local-llm").
    normalized_handlers = {
        provider_config_key(handler): handler for handler in API_CALL_HANDLERS
    }
    return normalized_handlers.get(provider_config_key(provider))


@dataclass(frozen=True)
class IngestAnalysisResolution:
    """Outcome of resolving the configured analysis provider.

    Attributes:
        provider: The configured provider display name (e.g. ``"OpenAI"``),
            or ``""`` when none is configured.
        api_key: The resolved credential when one exists. ``None`` for a
            keyless local provider that is nonetheless ready, and for every
            not-ready outcome.
        ready: Whether an analysis call can actually be made.
        short_reason: One-line reason for a job record when not ready
            (e.g. ``"OpenAI is not ready (Missing API key)"``); ``""`` when
            ready.
        hint: Full user-facing sentence for the ingest panel when not
            ready; ``""`` when ready.
        dispatch_name: The exact ``API_CALL_HANDLERS`` key an analysis call
            dispatches with (F5). ``""`` for every not-ready outcome --
            ``ready`` is only ever True with a vouched-for dispatch name.
        keyless: True when the provider is ready WITHOUT a credential (F8).
            The job-option builder turns this into the explicit
            ``analysis_keyless_ok`` opt-in the processors' credential gates
            require for keyless dispatch.
        model: ``[analysis_defaults] model``, or ``None`` to let the
            provider handler pick its configured default (F10).
        temperature: Sampling temperature for the analysis call (F10).
        top_p: Top-P for the analysis call (F10).
        min_p: Min-P for the analysis call (F10).
        max_tokens: Response token cap for the analysis call (F10).
        system_prompt: System prompt for the analysis call (F10).
    """

    provider: str
    api_key: Optional[str]
    ready: bool
    short_reason: str
    hint: str
    dispatch_name: str = ""
    keyless: bool = False
    model: Optional[str] = None
    temperature: float = ANALYSIS_DEFAULT_TEMPERATURE
    top_p: float = ANALYSIS_DEFAULT_TOP_P
    min_p: float = ANALYSIS_DEFAULT_MIN_P
    max_tokens: int = ANALYSIS_DEFAULT_MAX_TOKENS
    system_prompt: str = ANALYSIS_DEFAULT_SYSTEM_PROMPT


def resolve_ingest_analysis_provider(
    app_config: object,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> IngestAnalysisResolution:
    """Resolve whether Analyze-after-import has a callable provider.

    Args:
        app_config: The loaded app configuration mapping. Anything that is
            not a mapping degrades to "unconfigured" rather than raising --
            this runs on submission paths that must never crash.
        environ: Environment mapping, injectable for deterministic tests;
            defaults to ``os.environ`` inside ``get_provider_readiness``.

    Returns:
        The resolution -- see :class:`IngestAnalysisResolution`.
    """
    config: Mapping = app_config if isinstance(app_config, Mapping) else {}
    defaults = config.get("analysis_defaults")
    defaults = defaults if isinstance(defaults, Mapping) else {}
    provider = str(defaults.get("provider") or "").strip()

    if not provider:
        return IngestAnalysisResolution(
            provider="",
            api_key=None,
            ready=False,
            short_reason=NO_ANALYSIS_PROVIDER_REASON,
            hint=(
                "Analyze after import is on, but no analysis provider is "
                "configured — imports will run without analysis. Set "
                "provider under [analysis_defaults] in your config."
            ),
        )

    readiness = get_provider_readiness(provider, config, environ=environ)
    if readiness.ready:
        # (F5) Readiness alone is not dispatchability: the readiness gate
        # accepts names (custom, local_onnx, any provider with a configured
        # credential) that `chat_api_call` has no handler for -- which used
        # to surface only at analysis time as an in-band
        # "Error: Invalid API Name". Constrain here so ready == callable.
        dispatch = chat_dispatch_name(provider)
        if not dispatch:
            return IngestAnalysisResolution(
                provider=provider,
                api_key=None,
                ready=False,
                short_reason=(
                    f"provider '{provider}' is not supported for ingest analysis"
                ),
                hint=(
                    f"Analyze after import is on, but provider '{provider}' "
                    "is not supported for ingest analysis — imports will run "
                    "without analysis. Set a chat-capable provider under "
                    "[analysis_defaults] in your config."
                ),
            )
        return IngestAnalysisResolution(
            provider=provider,
            api_key=readiness.api_key,
            ready=True,
            short_reason="",
            hint="",
            dispatch_name=dispatch,
            keyless=readiness.api_key is None,
            model=_optional_str(defaults.get("model")),
            temperature=_as_float(
                defaults.get("temperature"), ANALYSIS_DEFAULT_TEMPERATURE
            ),
            top_p=_as_float(defaults.get("top_p"), ANALYSIS_DEFAULT_TOP_P),
            min_p=_as_float(defaults.get("min_p"), ANALYSIS_DEFAULT_MIN_P),
            max_tokens=_as_int(
                defaults.get("max_tokens"), ANALYSIS_DEFAULT_MAX_TOKENS
            ),
            system_prompt=_optional_str(defaults.get("system_prompt"))
            or ANALYSIS_DEFAULT_SYSTEM_PROMPT,
        )

    return IngestAnalysisResolution(
        provider=provider,
        api_key=None,
        ready=False,
        short_reason=f"{provider} is not ready ({readiness.reason})",
        hint=(
            f"Analyze after import is on, but {readiness.user_message} "
            "Imports will run without analysis."
        ),
    )


def analysis_unavailable_reason(resolution: IngestAnalysisResolution) -> str:
    """One-sentence reason an analysis call cannot be made now, or ``""``.

    task-28007 AC#5: the Reader's Generate action learned this only AFTER
    the click, as a toast. Both the disabled control's tooltip and the
    handler's post-click guard read it from here, so the label and the
    refusal can never say different things. ``short_reason`` (not
    ``hint``) is the source: the hint is written for the ingest panel and
    talks about imports.

    Args:
        resolution: The outcome of :func:`resolve_ingest_analysis_provider`.

    Returns:
        A capitalised, full-stopped sentence, or ``""`` when ready.
    """
    if resolution.ready:
        return ""
    # Strip BEFORE the fallback: a whitespace-only short_reason is truthy,
    # and stripping it afterwards left "" for `reason[0]` to raise on. No
    # resolution the resolver builds is blank today, but this is a public
    # seam other gates feed resolutions into.
    reason = (resolution.short_reason or "").strip() or NO_ANALYSIS_PROVIDER_REASON
    sentence = reason[0].upper() + reason[1:]
    return sentence if sentence.endswith(".") else f"{sentence}."


def _optional_str(value: Any) -> Optional[str]:
    """Return a stripped non-empty string, else None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_float(value: Any, fallback: float) -> float:
    """Coerce a possibly-display-string number, falling back."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def _as_int(value: Any, fallback: int) -> int:
    """Coerce a possibly-display-string number, falling back."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback
