"""Pure, bounded Speech runtime-status and navigation projections.

The module does not contact a provider or persist state.  It translates an
already accepted capability observation into ADR-039's configuration/runtime
vocabularies and keeps local dependency facts independent from provider
reachability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from tldw_chatbook.TTS.adapter_types import (
    TTSNativeCapabilityObservation,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConfigurationState,
    SpeechTTSDiagnosticCategory,
    SpeechTTSNavigationIntent,
    SpeechTTSNavigationTarget,
    SpeechTTSRuntimeState,
    SpeechTTSRuntimeStatus,
    SpeechTTSStatusFreshness,
    speech_tts_model_scope,
)


@dataclass(frozen=True, slots=True)
class SpeechLocalDependencyAvailability:
    """Independent import availability for Speech's local capabilities."""

    stt: bool
    kokoro: bool
    chatterbox: bool
    higgs: bool

    def __post_init__(self) -> None:
        for value in (self.stt, self.kokoro, self.chatterbox, self.higgs):
            if type(value) is not bool:
                raise TypeError("Speech dependency availability must be boolean")

    @classmethod
    def all_available(cls) -> SpeechLocalDependencyAvailability:
        """Return a dependency snapshot where every local capability is ready."""

        return cls(stt=True, kokoro=True, chatterbox=True, higgs=True)


@dataclass(frozen=True, slots=True)
class SpeechTTSStatusRow:
    """One display-ready row whose state comes from an approved vocabulary."""

    row_id: str
    label: str
    state: SpeechTTSConfigurationState | SpeechTTSRuntimeState
    detail: str = ""

    @property
    def copy(self) -> str:
        """Return bounded plain-text copy for Textual ``markup=False`` widgets."""

        suffix = f" — {self.detail}" if self.detail else ""
        return f"{self.label}: {self.state.value}{suffix}"


@dataclass(frozen=True, slots=True)
class SpeechTTSStatusProjection:
    """Independent configuration, runtime, catalog, and dependency facts."""

    provider_id: str
    configuration_state: SpeechTTSConfigurationState
    runtime_status: SpeechTTSRuntimeStatus | None
    catalog_status: SpeechTTSRuntimeStatus | None
    catalog_state: SpeechTTSRuntimeState
    local_dependencies: SpeechLocalDependencyAvailability

    @property
    def runtime_state(self) -> SpeechTTSRuntimeState:
        """Return Not checked when no accepted observation exists."""

        if self.runtime_status is None:
            return SpeechTTSRuntimeState.NOT_CHECKED
        return self.runtime_status.runtime_state

    @property
    def runtime_ready(self) -> bool:
        """Return remote provider readiness without consulting local packages."""

        return self.runtime_state is SpeechTTSRuntimeState.READY

    @property
    def summary(self) -> str:
        """Return a bounded provider summary independent of local capabilities."""

        provider_label = (
            "OpenAI-compatible" if self.provider_id == "openai" else self.provider_id
        )
        return f"{provider_label} speech: {self.runtime_state.value}"

    def rows(self, *, dirty_draft: bool = False) -> tuple[SpeechTTSStatusRow, ...]:
        """Return stable rows without provider payloads or arbitrary exceptions."""

        runtime_detail = _runtime_detail(self.runtime_status, dirty_draft=dirty_draft)
        catalog_detail = _catalog_detail(
            self.catalog_status,
            dirty_draft=dirty_draft,
        )
        rows = [
            SpeechTTSStatusRow(
                "provider-configuration",
                "Selected provider configuration",
                self.configuration_state,
            ),
            SpeechTTSStatusRow(
                "provider-runtime",
                "Selected provider runtime",
                self.runtime_state,
                runtime_detail,
            ),
            SpeechTTSStatusRow(
                "catalog-freshness",
                "Catalog and voices",
                self.catalog_state,
                catalog_detail,
            ),
        ]
        for row_id, label, extra, available in (
            (
                "stt-dependency",
                "Local transcription",
                "transcription_faster_whisper",
                self.local_dependencies.stt,
            ),
            (
                "kokoro-dependency",
                "Local Kokoro",
                "local_tts",
                self.local_dependencies.kokoro,
            ),
            (
                "chatterbox-dependency",
                "Local Chatterbox",
                "chatterbox",
                self.local_dependencies.chatterbox,
            ),
            (
                "higgs-dependency",
                "Local Higgs",
                "higgs_tts",
                self.local_dependencies.higgs,
            ),
        ):
            rows.append(
                SpeechTTSStatusRow(
                    row_id,
                    label,
                    (
                        SpeechTTSRuntimeState.READY
                        if available
                        else SpeechTTSRuntimeState.UNAVAILABLE
                    ),
                    "" if available else f"tldw_chatbook[{extra}]",
                )
            )
        return tuple(rows)


def speech_tts_status_is_newer(
    candidate: SpeechTTSRuntimeStatus,
    current: SpeechTTSRuntimeStatus,
    *,
    catalog_axis: bool,
) -> bool:
    """Compare evidence without treating an absent optional revision as old."""

    if (
        type(candidate) is not SpeechTTSRuntimeStatus
        or type(current) is not SpeechTTSRuntimeStatus
    ):
        raise TypeError("Speech runtime status is invalid")
    if type(catalog_axis) is not bool:
        raise TypeError("Speech status axis is invalid")
    if candidate.saved_configuration_revision != current.saved_configuration_revision:
        return (
            candidate.saved_configuration_revision
            > current.saved_configuration_revision
        )
    candidate_runtime = candidate.runtime_revision
    current_runtime = current.runtime_revision
    if candidate_runtime is not None and current_runtime is not None:
        if candidate_runtime != current_runtime:
            return candidate_runtime > current_runtime
    elif candidate_runtime is not current_runtime:
        return _has_newer_status_observation(candidate, current)
    if catalog_axis:
        candidate_catalog = candidate.catalog_revision
        current_catalog = current.catalog_revision
        if candidate_catalog is not None and current_catalog is not None:
            if candidate_catalog != current_catalog:
                return candidate_catalog > current_catalog
        elif candidate_catalog is not current_catalog:
            return _has_newer_status_observation(candidate, current)
    return _has_newer_status_observation(candidate, current)


def _has_newer_status_observation(
    candidate: SpeechTTSRuntimeStatus,
    current: SpeechTTSRuntimeStatus,
) -> bool:
    """Order equal-revision evidence, letting terminal results close transitions."""

    if candidate.observed_at != current.observed_at:
        return candidate.observed_at > current.observed_at
    transitional = {
        SpeechTTSRuntimeState.CHECKING,
        SpeechTTSRuntimeState.RECONFIGURING,
    }
    return (
        current.runtime_state in transitional
        and candidate.runtime_state not in transitional
    )


def newest_speech_tts_status(
    first: SpeechTTSRuntimeStatus | None,
    second: SpeechTTSRuntimeStatus | None,
    *,
    catalog_axis: bool,
) -> SpeechTTSRuntimeStatus | None:
    """Return the newer of two statuses under one independent axis."""

    if first is None:
        return second
    if second is None:
        return first
    return (
        second
        if speech_tts_status_is_newer(
            second,
            first,
            catalog_axis=catalog_axis,
        )
        else first
    )


class SpeechTTSRuntimeStatusStore:
    """App-scoped latest safe provider and catalog observations."""

    def __init__(self) -> None:
        self._runtime: dict[str, SpeechTTSRuntimeStatus] = {}
        self._catalog: dict[tuple[str, str | None], SpeechTTSRuntimeStatus] = {}

    def publish_runtime(self, status: SpeechTTSRuntimeStatus) -> None:
        """Publish a provider-runtime status unless a newer one is retained."""

        if type(status) is not SpeechTTSRuntimeStatus:
            raise TypeError("Speech runtime status is invalid")
        current = self._runtime.get(status.provider_id)
        if current is None or speech_tts_status_is_newer(
            status,
            current,
            catalog_axis=False,
        ):
            self._runtime[status.provider_id] = status

    def publish_catalog(self, status: SpeechTTSRuntimeStatus) -> None:
        """Publish a model-scoped catalog/voice status."""

        if type(status) is not SpeechTTSRuntimeStatus:
            raise TypeError("Speech catalog status is invalid")
        key = (status.provider_id, status.model_scope)
        current = self._catalog.get(key)
        if current is None or speech_tts_status_is_newer(
            status,
            current,
            catalog_axis=True,
        ):
            self._catalog[key] = status

    def runtime_status(self, provider_id: str) -> SpeechTTSRuntimeStatus | None:
        """Return the latest safe provider-runtime status."""

        SpeechTTSNavigationTarget(provider_id)
        return self._runtime.get(provider_id)

    def catalog_status(
        self,
        provider_id: str,
        model_id: str | None,
    ) -> SpeechTTSRuntimeStatus | None:
        """Return exact-model catalog status, then provider-wide status."""

        SpeechTTSNavigationTarget(provider_id)
        model_scope = speech_tts_model_scope(model_id)
        exact = self._catalog.get((provider_id, model_scope))
        provider_wide = self._catalog.get((provider_id, None))
        if exact is None:
            return provider_wide
        if provider_wide is None:
            return exact
        return newest_speech_tts_status(
            exact,
            provider_wide,
            catalog_axis=True,
        )


def speech_tts_runtime_status_store(owner: object) -> SpeechTTSRuntimeStatusStore:
    """Return the one process-local Speech status owner attached to an app."""

    store = getattr(owner, "_speech_tts_runtime_status_store", None)
    if store is None:
        store = SpeechTTSRuntimeStatusStore()
        setattr(owner, "_speech_tts_runtime_status_store", store)
    if type(store) is not SpeechTTSRuntimeStatusStore:
        raise TypeError("Speech runtime status owner is invalid")
    return store


def _runtime_detail(
    status: SpeechTTSRuntimeStatus | None,
    *,
    dirty_draft: bool,
) -> str:
    if status is None:
        return "No observation applies to the saved configuration."
    parts = [f"saved revision {status.saved_configuration_revision}"]
    if status.runtime_revision is not None:
        parts.append(f"runtime revision {status.runtime_revision}")
    if status.catalog_revision is not None:
        parts.append(f"catalog revision {status.catalog_revision}")
    if status.model_scope is not None:
        parts.append("selected model scope")
    parts.append(
        status.observed_at.astimezone(timezone.utc).strftime(
            "observed %Y-%m-%d %H:%M UTC"
        )
    )
    if status.diagnostic_category is not None:
        parts.append(status.diagnostic_category.value)
    if status.recovery_action is not None:
        parts.append(f"recovery {status.recovery_action.value}")
    if dirty_draft:
        parts.append("saved evidence does not validate this unsaved draft")
    return "; ".join(parts)


def _catalog_detail(
    status: SpeechTTSRuntimeStatus | None,
    *,
    dirty_draft: bool,
) -> str:
    if status is None:
        return "No accepted catalog or voice observation."
    parts = [f"saved revision {status.saved_configuration_revision}"]
    if status.runtime_revision is not None:
        parts.append(f"runtime revision {status.runtime_revision}")
    if status.catalog_revision is not None:
        parts.append(f"catalog revision {status.catalog_revision}")
    if status.model_scope is not None:
        parts.append("selected model scope")
    parts.append(
        status.observed_at.astimezone(timezone.utc).strftime(
            "observed %Y-%m-%d %H:%M UTC"
        )
    )
    parts.append(f"evidence {status.freshness.value}")
    if status.diagnostic_category is not None:
        parts.append(status.diagnostic_category.value)
    if status.recovery_action is not None:
        parts.append(f"recovery {status.recovery_action.value}")
    if dirty_draft:
        parts.append("applies to the saved connection only")
    return "; ".join(parts)


def _runtime_state_for_catalog(
    catalog: TTSProviderCatalog | None,
    *,
    saved_configuration_revision: int,
    applied_configuration_revision: int,
    observed_runtime_revision: int,
    current_runtime_revision: int,
    catalog_axis: bool,
) -> tuple[
    SpeechTTSRuntimeState,
    SpeechTTSStatusFreshness,
    SpeechTTSDiagnosticCategory | None,
    SpeechTTSNavigationIntent | None,
]:
    if (
        applied_configuration_revision != saved_configuration_revision
        or observed_runtime_revision != current_runtime_revision
        or catalog is None
    ):
        return (
            SpeechTTSRuntimeState.STALE,
            SpeechTTSStatusFreshness.STALE,
            (
                SpeechTTSDiagnosticCategory.CATALOG
                if catalog_axis
                else SpeechTTSDiagnosticCategory.CONFIGURATION
            ),
            (
                SpeechTTSNavigationIntent.REFRESH_MODELS
                if catalog_axis
                else SpeechTTSNavigationIntent.TEST
            ),
        )
    health = catalog.health
    if catalog_axis and not health.fresh:
        return (
            SpeechTTSRuntimeState.STALE,
            SpeechTTSStatusFreshness.STALE,
            SpeechTTSDiagnosticCategory.CATALOG,
            SpeechTTSNavigationIntent.REFRESH_MODELS,
        )
    if health.state == "available":
        return (
            SpeechTTSRuntimeState.READY,
            SpeechTTSStatusFreshness.FRESH,
            None,
            None,
        )
    if health.state == "reconfiguring":
        return (
            SpeechTTSRuntimeState.RECONFIGURING,
            SpeechTTSStatusFreshness.FRESH,
            SpeechTTSDiagnosticCategory.CONFIGURATION,
            SpeechTTSNavigationIntent.TEST,
        )
    if health.state == "not_configured":
        return (
            SpeechTTSRuntimeState.UNAVAILABLE,
            SpeechTTSStatusFreshness.FRESH,
            SpeechTTSDiagnosticCategory.CONFIGURATION,
            SpeechTTSNavigationIntent.CONFIGURE,
        )
    return (
        SpeechTTSRuntimeState.UNAVAILABLE,
        SpeechTTSStatusFreshness.FRESH,
        SpeechTTSDiagnosticCategory.CONNECTION,
        SpeechTTSNavigationIntent.TEST,
    )


def speech_tts_runtime_status_from_catalog(
    *,
    provider_id: str,
    saved_configuration_revision: int,
    applied_configuration_revision: int,
    observed_runtime_revision: int,
    current_runtime_revision: int,
    catalog: TTSProviderCatalog | None,
    model_id: str | None,
    observed_at: datetime,
    catalog_axis: bool = False,
) -> SpeechTTSRuntimeStatus:
    """Build one safe runtime observation from an already accepted catalog."""

    SpeechTTSNavigationTarget(provider_id)
    if catalog is not None and catalog.provider_id != provider_id:
        raise ValueError("Speech catalog provider does not match")
    state, freshness, category, recovery = _runtime_state_for_catalog(
        catalog,
        saved_configuration_revision=saved_configuration_revision,
        applied_configuration_revision=applied_configuration_revision,
        observed_runtime_revision=observed_runtime_revision,
        current_runtime_revision=current_runtime_revision,
        catalog_axis=catalog_axis,
    )
    return SpeechTTSRuntimeStatus(
        provider_id=provider_id,
        saved_configuration_revision=saved_configuration_revision,
        runtime_revision=observed_runtime_revision,
        catalog_revision=catalog.revision if catalog is not None else None,
        model_scope=speech_tts_model_scope(model_id),
        runtime_state=state,
        observed_at=observed_at,
        freshness=freshness,
        diagnostic_category=category,
        recovery_action=recovery,
    )


def _apply_observed_model_freshness(
    status: SpeechTTSRuntimeStatus,
    snapshot: object,
    model_id: str | None,
) -> SpeechTTSRuntimeStatus:
    """Refine a ready catalog with model-scoped native voice evidence."""

    if (
        model_id is None
        or status.runtime_state is not SpeechTTSRuntimeState.READY
        or type(snapshot) is not TTSNativeCapabilitySnapshot
        or snapshot.catalog is None
    ):
        return status
    catalog = snapshot.catalog
    model_present = any(model.model_id == model_id for model in catalog.models)
    voice_result = snapshot.voice_results.get(model_id)
    if voice_result is not None and voice_result.state == "model_missing":
        return replace(
            status,
            runtime_state=SpeechTTSRuntimeState.UNAVAILABLE,
            freshness=SpeechTTSStatusFreshness.FRESH,
            diagnostic_category=SpeechTTSDiagnosticCategory.CATALOG,
            recovery_action=SpeechTTSNavigationIntent.REFRESH_MODELS,
        )
    if not model_present:
        if catalog.approximate:
            return replace(
                status,
                runtime_state=SpeechTTSRuntimeState.STALE,
                freshness=SpeechTTSStatusFreshness.STALE,
                diagnostic_category=SpeechTTSDiagnosticCategory.CATALOG,
                recovery_action=SpeechTTSNavigationIntent.REFRESH_MODELS,
            )
        return replace(
            status,
            runtime_state=SpeechTTSRuntimeState.UNAVAILABLE,
            freshness=SpeechTTSStatusFreshness.FRESH,
            diagnostic_category=SpeechTTSDiagnosticCategory.CATALOG,
            recovery_action=SpeechTTSNavigationIntent.REFRESH_MODELS,
        )
    if voice_result is not None and (
        voice_result.state == "complete"
        and voice_result.catalog_revision == catalog.revision
    ):
        return status
    if snapshot.state == "complete" and voice_result is None:
        return status
    return replace(
        status,
        runtime_state=SpeechTTSRuntimeState.STALE,
        freshness=SpeechTTSStatusFreshness.STALE,
        diagnostic_category=SpeechTTSDiagnosticCategory.CATALOG,
        recovery_action=SpeechTTSNavigationIntent.REFRESH_VOICES,
    )


def project_speech_tts_status(
    *,
    provider_id: str,
    configuration_state: SpeechTTSConfigurationState,
    current_configuration_revision: int | None,
    model_id: str | None,
    observation: TTSNativeCapabilityObservation | None,
    local_dependencies: SpeechLocalDependencyAvailability,
    current_runtime_revision: int | None = None,
    applied_configuration_revision: int | None = None,
    runtime_status: SpeechTTSRuntimeStatus | None = None,
    catalog_status: SpeechTTSRuntimeStatus | None = None,
) -> SpeechTTSStatusProjection:
    """Project one selected provider without performing provider work."""

    SpeechTTSNavigationTarget(provider_id)
    if type(configuration_state) is not SpeechTTSConfigurationState:
        raise TypeError("Speech configuration state is invalid")
    if type(local_dependencies) is not SpeechLocalDependencyAvailability:
        raise TypeError("Speech local dependency snapshot is invalid")
    if current_configuration_revision is not None and (
        type(current_configuration_revision) is not int
        or current_configuration_revision < 0
    ):
        raise ValueError("Speech configuration revision is invalid")
    if current_runtime_revision is not None and (
        type(current_runtime_revision) is not int or current_runtime_revision < 0
    ):
        raise ValueError("Speech runtime revision is invalid")
    if applied_configuration_revision is not None and (
        type(applied_configuration_revision) is not int
        or applied_configuration_revision < 0
    ):
        raise ValueError("Applied Speech configuration revision is invalid")

    selected_model_scope = speech_tts_model_scope(model_id)

    def status_for_current_revisions(
        candidate: SpeechTTSRuntimeStatus,
        *,
        catalog_axis: bool,
    ) -> SpeechTTSRuntimeStatus:
        stale = (
            current_configuration_revision is not None
            and candidate.saved_configuration_revision != current_configuration_revision
        ) or (
            current_runtime_revision is not None
            and candidate.runtime_revision is not None
            and candidate.runtime_revision != current_runtime_revision
        )
        if stale and candidate.runtime_state is not SpeechTTSRuntimeState.STALE:
            return replace(
                candidate,
                runtime_state=SpeechTTSRuntimeState.STALE,
                freshness=SpeechTTSStatusFreshness.STALE,
                diagnostic_category=(
                    SpeechTTSDiagnosticCategory.CATALOG
                    if catalog_axis
                    else SpeechTTSDiagnosticCategory.CONFIGURATION
                ),
                recovery_action=(
                    SpeechTTSNavigationIntent.REFRESH_MODELS
                    if catalog_axis
                    else SpeechTTSNavigationIntent.TEST
                ),
            )
        return candidate

    status: SpeechTTSRuntimeStatus | None = None
    observation_status: SpeechTTSRuntimeStatus | None = None
    observation_catalog_status: SpeechTTSRuntimeStatus | None = None
    accepted_catalog_status: SpeechTTSRuntimeStatus | None = None
    if runtime_status is not None:
        if type(runtime_status) is not SpeechTTSRuntimeStatus:
            raise TypeError("Speech runtime status is invalid")
        if (
            runtime_status.provider_id == provider_id
            and runtime_status.model_scope is None
        ):
            status = status_for_current_revisions(
                runtime_status,
                catalog_axis=False,
            )
    if catalog_status is not None:
        if type(catalog_status) is not SpeechTTSRuntimeStatus:
            raise TypeError("Speech catalog status is invalid")
        if catalog_status.provider_id == provider_id and catalog_status.model_scope in {
            None,
            selected_model_scope,
        }:
            accepted_catalog_status = status_for_current_revisions(
                catalog_status,
                catalog_axis=True,
            )
    if observation is not None:
        if type(observation) is not TTSNativeCapabilityObservation:
            raise TypeError("Speech capability observation is invalid")
        snapshot = observation.snapshot
        if (
            snapshot.provider_id == provider_id
            and current_configuration_revision is not None
        ):
            effective_runtime_revision = (
                snapshot.configuration_revision
                if current_runtime_revision is None
                else current_runtime_revision
            )
            effective_applied_revision = (
                current_configuration_revision
                if applied_configuration_revision is None
                else applied_configuration_revision
            )
            observation_status = speech_tts_runtime_status_from_catalog(
                provider_id=provider_id,
                saved_configuration_revision=current_configuration_revision,
                applied_configuration_revision=effective_applied_revision,
                observed_runtime_revision=snapshot.configuration_revision,
                current_runtime_revision=effective_runtime_revision,
                catalog=snapshot.catalog,
                model_id=None,
                observed_at=observation.observed_at,
            )
            observation_catalog_status = speech_tts_runtime_status_from_catalog(
                provider_id=provider_id,
                saved_configuration_revision=current_configuration_revision,
                applied_configuration_revision=effective_applied_revision,
                observed_runtime_revision=snapshot.configuration_revision,
                current_runtime_revision=effective_runtime_revision,
                catalog=snapshot.catalog,
                model_id=model_id,
                observed_at=observation.observed_at,
                catalog_axis=True,
            )
            observation_catalog_status = _apply_observed_model_freshness(
                observation_catalog_status,
                snapshot,
                model_id,
            )
            status = newest_speech_tts_status(
                status,
                observation_status,
                catalog_axis=False,
            )

    catalog_status = newest_speech_tts_status(
        observation_catalog_status,
        accepted_catalog_status,
        catalog_axis=True,
    )
    catalog_state = (
        SpeechTTSRuntimeState.NOT_CHECKED
        if catalog_status is None
        else catalog_status.runtime_state
    )
    return SpeechTTSStatusProjection(
        provider_id=provider_id,
        configuration_state=configuration_state,
        runtime_status=status,
        catalog_status=catalog_status,
        catalog_state=catalog_state,
        local_dependencies=local_dependencies,
    )


def speech_tts_navigation_context(
    target: SpeechTTSNavigationTarget,
) -> dict[str, str]:
    """Serialize only the canonical provider and optional bounded intent."""

    if type(target) is not SpeechTTSNavigationTarget:
        raise TypeError("Speech navigation target is invalid")
    context = {"provider": target.provider_id}
    if target.intent is not None:
        context["intent"] = target.intent.value
    return context


def speech_tts_navigation_target_from_context(
    context: Mapping[str, object],
) -> SpeechTTSNavigationTarget | None:
    """Parse an exact provider/intent context and reject every extra field."""

    if not isinstance(context, Mapping) or not set(context).issubset(
        {"provider", "intent"}
    ):
        return None
    provider_id = context.get("provider")
    if type(provider_id) is not str:
        return None
    raw_intent = context.get("intent")
    try:
        intent = (
            None
            if raw_intent is None
            else SpeechTTSNavigationIntent(raw_intent)
            if type(raw_intent) is str
            else None
        )
        if raw_intent is not None and intent is None:
            return None
        return SpeechTTSNavigationTarget(provider_id, intent)
    except (TypeError, ValueError):
        return None


__all__ = [
    "SpeechLocalDependencyAvailability",
    "SpeechTTSStatusProjection",
    "SpeechTTSStatusRow",
    "SpeechTTSRuntimeStatusStore",
    "newest_speech_tts_status",
    "project_speech_tts_status",
    "speech_tts_navigation_context",
    "speech_tts_navigation_target_from_context",
    "speech_tts_runtime_status_from_catalog",
    "speech_tts_runtime_status_store",
    "speech_tts_status_is_newer",
]
