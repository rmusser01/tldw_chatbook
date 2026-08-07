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
from typing import Mapping, Optional

from tldw_chatbook.Chat.provider_readiness import get_provider_readiness


#: Short reason recorded on a job when ``[analysis_defaults]`` names no
#: provider at all. Also the substring the panel hint builds on.
NO_ANALYSIS_PROVIDER_REASON = "no analysis provider is configured"


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
    """

    provider: str
    api_key: Optional[str]
    ready: bool
    short_reason: str
    hint: str


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
        return IngestAnalysisResolution(
            provider=provider,
            api_key=readiness.api_key,
            ready=True,
            short_reason="",
            hint="",
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
