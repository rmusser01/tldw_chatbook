"""Provider catalog loading, shared by the legacy Playground and the rebuild.

The catalog fills the comparison axes: which providers are offered, which
models each has, which voices each model has, and whether the provider is
reachable at all. Without it the rebuilt pane renders correct, empty selects
-- controls with nothing to choose.

Moved here whole rather than reimplemented. Of these 717 lines, exactly one
method was coupled to the legacy layout: `_show_provider_specific_controls`,
which toggled the per-provider container boxes. That is the seam. The legacy
widget keeps the toggling body; `SpeechPlaygroundPane` overrides it to
re-scope its parameter group instead. Same call site, same responsibility,
two implementations.

`on_tts_provider_select_changed` is decorated `@on(Select.Changed)` with no
selector, so it sees every Select in the host and filters by id itself. A
host must therefore NOT add its own `#tts-provider-select` handler: both
would fire, and a handler that recomposes would destroy the widgets this one
is midway through populating.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from rich.text import Text
from textual import work
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Select, Static, Switch, TextArea

from tldw_chatbook.config import get_cli_setting, get_runtime_config_snapshot
from tldw_chatbook.TTS import STTSPlaygroundResultProjection, get_tts_service
from tldw_chatbook.TTS.adapter_types import (
    TTSOperationError,
    TTSProviderCatalog,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.legacy_catalogs import (
    LEGACY_DEFAULT_MODELS,
    LEGACY_DEFAULT_VOICES,
    LEGACY_VOICE_OPTIONS,
)
from tldw_chatbook.TTS.voice_blend_paths import kokoro_ui_blend_file
from tldw_chatbook.TTS.provider_ids import BUILT_IN_TTS_PROVIDER_IDS
from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
    SettingsEndpointProbePurpose,
    probe_settings_endpoint,
)
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    build_provider_test_fingerprint,
    load_global_speech_tts_state,
    process_provider_test_evidence_store,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    ProviderTestFingerprint,
    SpeechTTSConnectionState,
)
from tldw_chatbook.UI.stts_playground_catalog import (
    AUDIO_CPP_PROVIDER_ID,
    CatalogRequestToken,
    LOADING_SELECT_VALUE,
    PlaygroundControls,
    SERVER_DEFAULT_VOICE_ID,
    SERVER_DEFAULT_VOICE_LABEL,
    UNAVAILABLE_SELECT_VALUE,
    SelectSentinel,
    SelectValue,
    controls_from_catalog,
    controls_from_profile_preset,
    pinned_no_catalog_check_option,
    preset_has_no_catalog_check,
    profile_availability_from_catalog,
    provider_options,
)


class SpeechCatalogMixin:
    """Catalog loading and provider selection, independent of the layout."""

    def _tts_service_factory(self):
        """Return the TTS service, awaitable.

        A hook, not a direct call, so a host can redirect where
        `get_tts_service` resolves from without this mixin needing to know
        about it. Moving catalog loading into a shared mixin once silently
        detached twelve patch sites that expected to patch `get_tts_service`
        on the `STTS_Window` module -- the catalog quietly hit the real
        service instead and every select sat on LOADING forever, which is
        exactly how this was found.

        Returns:
            An awaitable yielding the TTS service.
        """
        return get_tts_service()

    @staticmethod
    def _catalog_test_fingerprint(
        service: object,
        provider_id: str,
        configuration_revision: int,
    ) -> tuple[ProviderTestFingerprint, str] | None:
        """Capture one saved catalog identity before provider work starts."""

        saved_revision = getattr(service, "saved_configuration_revision", None)
        if not callable(saved_revision):
            return None
        try:
            revision = saved_revision(provider_id)
            if revision != configuration_revision:
                return None
            values = get_runtime_config_snapshot().values
            state = load_global_speech_tts_state(
                values if isinstance(values, Mapping) else {}
            )
            fingerprint = build_provider_test_fingerprint(
                state,
                provider_id=provider_id,
                saved_revision=revision,
            )
            base_url = state.providers[provider_id].get("base_url", "")
            if type(base_url) is not str:
                return None
            return fingerprint, base_url
        except Exception:  # noqa: BLE001 - catalog evidence is best effort
            return None

    async def _openai_catalog_test_state(
        self,
        captured: tuple[ProviderTestFingerprint, str] | None,
    ) -> SpeechTTSConnectionState | None:
        """Run the explicit OpenAI-compatible TTS catalog operation."""

        if captured is None:
            return None
        _fingerprint, base_url = captured
        outcome = await probe_settings_endpoint(
            base_url,
            provider="openai",
            purpose=SettingsEndpointProbePurpose.TTS_CATALOG,
        )
        return SpeechTTSConnectionState(outcome.state)

    def _record_catalog_test_state(
        self,
        service: object,
        captured: tuple[ProviderTestFingerprint, str] | None,
        state: SpeechTTSConnectionState | None,
    ) -> None:
        """Publish catalog evidence only while its saved identity is current."""

        if captured is None or state is None:
            return
        fingerprint, base_url = captured
        saved_revision = getattr(service, "saved_configuration_revision", None)
        configuration_revision = getattr(service, "configuration_revision", None)
        try:
            if (
                not callable(saved_revision)
                or not callable(configuration_revision)
                or saved_revision(fingerprint.provider_id)
                != fingerprint.saved_revision
                or configuration_revision(fingerprint.provider_id)
                != fingerprint.saved_revision
            ):
                return
            current = self._catalog_test_fingerprint(
                service,
                fingerprint.provider_id,
                fingerprint.saved_revision,
            )
            if current != (fingerprint, base_url):
                return
            process_provider_test_evidence_store(self.app).record_catalog(
                fingerprint,
                state,
            )
        except Exception:  # noqa: BLE001 - evidence cannot fail catalog loading
            return

    def _cli_setting(self, *args: Any, **kwargs: Any) -> Any:
        """Read a config setting.

        Overridable for the same reason as `_tts_service_factory`.

        Args:
            args: Forwarded to ``get_cli_setting``.
            kwargs: Forwarded to ``get_cli_setting``.

        Returns:
            The configured value, or the supplied default.
        """
        return get_cli_setting(*args, **kwargs)

    def init_catalog_state(self) -> None:
        """Initialise the state catalog loading reads and writes.

        Call from the host's ``__init__``. These seventeen attributes are
        the whole host contract; the methods below read them without
        guards, so a host that skips this fails at the first provider
        change with ``AttributeError`` rather than anywhere informative.
        """
        #: Lazily created TTS service handle.
        self._tts_service: Any = None
        #: Provider id -> human-readable name.
        self._provider_display_names: dict[str, str] = {}
        #: The provider whose controls are currently on screen, versus the
        #: one the user has selected. They differ while a catalog loads.
        self._displayed_provider_id: str | None = None
        self._selected_provider_id: str | None = None
        #: Provider id -> its fetched catalog.
        self._catalogs: dict[str, Any] = {}
        #: Staleness bookkeeping: a config change bumps the revision, and a
        #: reply carrying an older generation is discarded rather than
        #: overwriting fresher data.
        self._catalog_configuration_revisions: dict[str, int] = {}
        self._catalog_request_generations: dict[str, int] = {}
        self._catalog_observed_at: dict[str, datetime] = {}
        self._catalog_runtime_observed_at: dict[str, datetime] = {}
        self._catalog_checking_providers: set[str] = set()
        self._catalog_unavailable_providers: set[str] = set()
        self._voice_request_generations: dict[tuple[str, str], int] = {}
        self._voice_checking_scopes: set[tuple[str, str]] = set()
        self._voice_unavailable_scopes: set[tuple[str, str]] = set()
        self._voice_runtime_observed_at: dict[tuple[str, str], datetime] = {}
        #: (provider, model) -> discovered voice ids.
        self._discovered_voices: dict[tuple[str, str], tuple[str, ...]] = {}
        #: Provider id -> the voice to re-select once its catalog arrives.
        self._pending_voice_selections: dict[str, str] = {}
        #: Provider id -> the control values to restore on return to it.
        self._provider_control_snapshots: dict[str, dict[str, Any]] = {}
        #: Providers whose cached catalog is known to be out of date.
        self._stale_providers: set[str] = set()
        #: Guards against a generation being launched mid-catalog-load, and
        #: against the loader's own writes re-entering the change handlers.
        self._catalog_generation_allowed = False
        self._applying_catalog_controls = False
        #: What the loader last wrote, so a user edit can be told from a
        #: programmatic one.
        self._applied_model_id: str | None = None
        self._applied_voice_id: Any = None
        self._applied_format: str | None = None

    def _load_provider_catalog(
        self,
        provider_id: str | None = None,
        *,
        refresh: bool = False,
        initialize: bool = False,
    ) -> None:
        """Reserve request identity before starting exclusive catalog work."""
        target = provider_id or self._selected_provider_id
        if target is None and initialize:
            target = self._configured_initial_provider_id()
        request_generation = (
            self._reserve_catalog_request(target) if isinstance(target, str) else None
        )
        self._load_provider_catalog_worker(
            provider_id,
            refresh=refresh,
            initialize=initialize,
            request_generation=request_generation,
        )

    def _configured_initial_provider_id(self) -> str:
        """Return one bounded provider for pre-descriptor failure reporting."""

        configured = self._cli_setting(
            "app_tts",
            "default_provider",
            AUDIO_CPP_PROVIDER_ID,
        )
        return (
            configured
            if isinstance(configured, str) and configured in BUILT_IN_TTS_PROVIDER_IDS
            else AUDIO_CPP_PROVIDER_ID
        )

    def _reserve_catalog_request(self, provider_id: str) -> int:
        """Reserve and return the next catalog request generation."""
        generation = self._catalog_request_generations.get(provider_id, 0) + 1
        self._catalog_request_generations[provider_id] = generation
        return generation

    @work(
        exclusive=True,
        group="stts-catalog-discovery",
        exit_on_error=False,
    )
    async def _load_provider_catalog_worker(
        self,
        provider_id: str | None = None,
        *,
        refresh: bool = False,
        initialize: bool = False,
        request_generation: int | None = None,
    ) -> None:
        """Load descriptors and one selected provider catalog."""
        token: CatalogRequestToken | None = None
        profile_voice_token: CatalogRequestToken | None = None
        initial_failure_provider = (
            self._configured_initial_provider_id() if initialize else None
        )
        try:
            if self._tts_service is None:
                self._tts_service = await self._tts_service_factory()

            service = self._tts_service
            if initialize:
                descriptors = service.provider_descriptors()
                options = provider_options(descriptors)
                if not options:
                    self._profile_preview_loading = False
                    self._set_provider_status("No TTS providers are registered")
                    return
                self._provider_ids = frozenset(value for _label, value in options)
                self._provider_display_names = {
                    value: label for label, value in options
                }
                provider_select = self.query_one("#tts-provider-select", Select)
                provider_select.set_options(self._safe_select_options(options))
                provider_select.disabled = False
                configured_default = self._cli_setting(
                    "app_tts",
                    "default_provider",
                    options[0][1],
                )
                preset_provider = (
                    self._profile_preset.provider_id
                    if self._profile_preset is not None
                    else None
                )
                selected = options[0][1]
                if configured_default in self._provider_ids:
                    selected = configured_default
                if preset_provider in self._provider_ids:
                    selected = preset_provider
                self._selected_provider_id = selected
                self._applying_catalog_controls = True
                try:
                    provider_select.value = selected
                finally:
                    self._applying_catalog_controls = False
                self.query_one("#tts-refresh-catalog-btn", Button).disabled = False
                self._show_provider_specific_controls(selected)
                provider_id = selected

            if provider_id is None:
                provider_id = self._selected_provider_id
            if provider_id is None or provider_id not in getattr(
                self, "_provider_ids", ()
            ):
                self._profile_preview_loading = False
                self._sync_profile_preview_status()
                return

            configuration_revision = service.configuration_revision(provider_id)
            self._catalog_checking_providers.add(provider_id)
            self._catalog_unavailable_providers.discard(provider_id)
            self._catalog_runtime_observed_at[provider_id] = datetime.now(timezone.utc)
            self._sync_truthful_status_rows_if_supported()
            if request_generation is None:
                request_generation = self._reserve_catalog_request(provider_id)
            token = CatalogRequestToken(
                provider_id=provider_id,
                configuration_revision=configuration_revision,
                request_generation=request_generation,
            )
            preset = self._profile_preset
            if preset is not None and preset.provider_id == provider_id:
                self._profile_configuration_revision = configuration_revision
            self._set_provider_status("Loading selected provider models…")
            catalog_test = self._catalog_test_fingerprint(
                service,
                provider_id,
                configuration_revision,
            )
            catalog_test_state = (
                await self._openai_catalog_test_state(catalog_test)
                if provider_id == "openai"
                else None
            )
            if catalog_test_state is not None:
                if not self._catalog_token_is_current(token):
                    if self._catalog_request_is_latest(token):
                        self._mark_stale_catalog_result(token)
                    return
                self._record_catalog_test_state(
                    service,
                    catalog_test,
                    catalog_test_state,
                )
            catalog = await service.get_catalog(provider_id, refresh=refresh)
            if not self._catalog_token_is_current(token):
                if self._catalog_request_is_latest(token):
                    self._mark_stale_catalog_result(token)
                return
            if catalog.provider_id != provider_id:
                self._catalog_failure(
                    provider_id,
                    "The selected provider returned an incompatible catalog",
                )
                return
            self._profile_preview_loading = False
            previous_catalog = self._catalogs.get(provider_id)
            if (
                previous_catalog is not None
                and previous_catalog.revision != catalog.revision
            ):
                self._discovered_voices = {
                    key: value
                    for key, value in self._discovered_voices.items()
                    if key[0] != provider_id
                }
                self._voice_checking_scopes = {
                    key for key in self._voice_checking_scopes if key[0] != provider_id
                }
                self._voice_unavailable_scopes = {
                    key
                    for key in self._voice_unavailable_scopes
                    if key[0] != provider_id
                }
                self._voice_runtime_observed_at = {
                    key: value
                    for key, value in self._voice_runtime_observed_at.items()
                    if key[0] != provider_id
                }
            previous_configuration_revision = self._catalog_configuration_revisions.get(
                provider_id
            )
            self._catalogs[provider_id] = catalog
            self._catalog_configuration_revisions[provider_id] = configuration_revision
            self._catalog_observed_at[provider_id] = datetime.now(timezone.utc)
            self._catalog_runtime_observed_at[provider_id] = self._catalog_observed_at[
                provider_id
            ]
            self._catalog_checking_providers.discard(provider_id)
            if catalog.health.state in {"available", "reconfiguring"}:
                self._catalog_unavailable_providers.discard(provider_id)
            else:
                self._catalog_unavailable_providers.add(provider_id)
            if catalog.health.fresh:
                self._stale_providers.discard(provider_id)
            elif (
                previous_configuration_revision is not None
                and previous_configuration_revision != configuration_revision
            ):
                # TASK-2970: a genuine supersession -- this provider's
                # configuration actually changed since our last successful
                # catalog load (tracked via `_catalog_configuration_
                # revisions`), and this fresh fetch itself hasn't caught up
                # to that change yet (`health.fresh` is False). Mirror that
                # as "stale" so the status line's settings-changed copy
                # stays accurate. A load that merely reports non-fresh
                # health with no revision history behind it -- most
                # notably the very first load ever for this provider,
                # where `previous_configuration_revision` is None -- is
                # NOT a genuine supersession: nothing has changed, this is
                # simply how the first read came back. The retired
                # widget's equivalent success path never marked stale here
                # at all (unconditional discard, no `health.fresh` branch);
                # this narrower condition is the one case task-2970's own
                # AC#3 calls out as still warranting it.
                self._stale_providers.add(provider_id)
            else:
                self._stale_providers.discard(provider_id)
            preset = self._profile_preset
            if preset is not None and preset.provider_id == provider_id:
                self._profile_effective_availability = (
                    profile_availability_from_catalog(preset, catalog)
                )
            if (
                preset is not None
                and preset.provider_id == provider_id
                and preset.voice_id is not None
                and self._profile_effective_availability != "unavailable"
            ):
                profile_voice_token = self._reserve_voice_request_token(
                    provider_id,
                    preset.model_id,
                    catalog.revision,
                )
                self._profile_voice_validation_token = profile_voice_token
            self._apply_catalog(provider_id, catalog)
            if (
                preset is not None
                and preset.provider_id == provider_id
                and catalog.health.state == "closed"
            ):
                self._stale_providers.add(provider_id)
                self._catalog_generation_allowed = False
                if self._profile_effective_availability != "unavailable":
                    self._set_provider_status("The TTS service is unavailable")
                self._sync_generate_enabled()
                if profile_voice_token is not None:
                    self._clear_profile_voice_validation(profile_voice_token)
                return
            if (
                preset is not None
                and preset.provider_id == provider_id
                and (
                    self._profile_effective_availability == "unavailable"
                    or preset.voice_id is None
                )
            ):
                return

            model_id = self._current_select_value("#tts-model-select")
            if isinstance(model_id, str):
                self._load_provider_voices(
                    provider_id,
                    model_id,
                    catalog.revision,
                    refresh=refresh,
                    request_token=profile_voice_token,
                )
            elif profile_voice_token is not None:
                self._clear_profile_voice_validation(profile_voice_token)
        except asyncio.CancelledError:
            latest_request = (
                self._catalog_request_is_latest(token)
                if token is not None
                else isinstance(provider_id, str)
                and request_generation
                == self._catalog_request_generations.get(provider_id)
            )
            if isinstance(provider_id, str) and latest_request:
                self._catalog_checking_providers.discard(provider_id)
                self._sync_truthful_status_rows_if_supported()
            if profile_voice_token is not None:
                self._clear_profile_voice_validation(profile_voice_token)
            raise
        except Exception as error:
            if profile_voice_token is not None:
                self._clear_profile_voice_validation(profile_voice_token)
            target = (
                provider_id or self._selected_provider_id or initial_failure_provider
            )
            if token is not None and not self._catalog_token_is_current(token):
                if self._catalog_request_is_latest(token):
                    self._mark_stale_catalog_result(token)
                return
            if target is not None:
                if self._selected_provider_id is None:
                    self._selected_provider_id = target
                self._catalog_configuration_revisions.setdefault(target, 0)
                exact_attempt_allowed = (
                    self._tts_service is not None
                    and not isinstance(error, TTSRegistryClosedError)
                    and not (
                        isinstance(error, TTSOperationError)
                        and error.code in {"configuration_invalid", "not_configured"}
                    )
                )
                self._catalog_failure(
                    target,
                    self._catalog_error_copy(error, target),
                    exact_attempt_allowed=exact_attempt_allowed,
                )

    def _load_provider_voices(
        self,
        provider_id: str,
        model_id: str,
        catalog_revision: int,
        *,
        refresh: bool = False,
        request_token: CatalogRequestToken | None = None,
    ) -> None:
        """Reserve request identity before starting exclusive voice work."""
        token = request_token or self._reserve_voice_request_token(
            provider_id,
            model_id,
            catalog_revision,
        )
        scope = (provider_id, model_id)
        self._voice_checking_scopes.add(scope)
        self._voice_unavailable_scopes.discard(scope)
        self._voice_runtime_observed_at[scope] = datetime.now(timezone.utc)
        self._sync_truthful_status_rows_if_supported()
        preset = self._profile_preset
        if (
            preset is not None
            and preset.provider_id == provider_id
            and preset.model_id == model_id
            and preset.voice_id is not None
        ):
            self._profile_voice_validation_token = token
            self._sync_profile_preview_status()
            self._sync_generate_enabled()
        self._load_provider_voices_worker(
            provider_id,
            model_id,
            catalog_revision,
            refresh=refresh,
            request_token=token,
        )

    def _reserve_voice_request_token(
        self,
        provider_id: str,
        model_id: str,
        catalog_revision: int,
    ) -> CatalogRequestToken:
        """Reserve one voice request and capture its catalog authority."""

        request_key = (provider_id, model_id)
        request_generation = self._voice_request_generations.get(request_key, 0) + 1
        self._voice_request_generations[request_key] = request_generation
        configuration_revision = self._catalog_configuration_revisions.get(provider_id)
        if configuration_revision is None:
            service = self._tts_service
            if service is None:
                raise TTSRegistryClosedError("The TTS service is unavailable")
            configuration_revision = service.configuration_revision(provider_id)
        return CatalogRequestToken(
            provider_id=provider_id,
            configuration_revision=configuration_revision,
            catalog_revision=catalog_revision,
            model_id=model_id,
            request_generation=request_generation,
        )

    @work(
        exclusive=True,
        group="stts-voice-discovery",
        exit_on_error=False,
    )
    async def _load_provider_voices_worker(
        self,
        provider_id: str,
        model_id: str,
        catalog_revision: int,
        *,
        refresh: bool = False,
        request_token: CatalogRequestToken,
    ) -> None:
        """Load voices for only the selected provider model."""
        try:
            service = self._tts_service
            if service is None:
                self._finish_voice_status(request_token, unavailable=True)
                self._clear_profile_voice_validation(request_token)
                return
            observation: TTSVoiceDiscoveryResult | None = None
            observe_voices = getattr(service, "observe_voices", None)
            if provider_id == AUDIO_CPP_PROVIDER_ID and callable(observe_voices):
                observation = await observe_voices(
                    provider_id,
                    model_id,
                    refresh=refresh,
                )
                if (
                    type(observation) is not TTSVoiceDiscoveryResult
                    or observation.provider_id != provider_id
                    or observation.model_id != model_id
                    or observation.catalog_revision != catalog_revision
                ):
                    raise ValueError(
                        "The selected provider returned incompatible voice metadata"
                    )
                voices = observation.voices if observation.state == "complete" else ()
            else:
                voices = await service.get_voices(
                    provider_id,
                    model_id,
                    refresh=refresh,
                )
        except asyncio.CancelledError:
            self._finish_voice_status(request_token, unavailable=False)
            self._clear_profile_voice_validation(request_token)
            raise
        except Exception as error:
            self._clear_profile_voice_validation(request_token)
            if not self._voice_token_is_current(request_token):
                self._finish_voice_status(request_token, unavailable=False)
                return
            self._finish_voice_status(request_token, unavailable=True)
            if isinstance(
                error,
                (TTSProviderReconfiguringError, TTSRegistryClosedError),
            ):
                if provider_id == self._selected_provider_id:
                    preset = self._profile_preset
                    if preset is not None and preset.provider_id == provider_id:
                        if self._profile_effective_availability != "unavailable":
                            self._profile_effective_availability = "unverified"
                        if (
                            isinstance(error, TTSProviderReconfiguringError)
                            and self._profile_effective_availability != "unavailable"
                        ):
                            self._stale_providers.add(provider_id)
                            self._catalog_generation_allowed = True
                            if preset_has_no_catalog_check(preset):
                                self._set_provider_status(
                                    "This provider has no catalog check. Generate "
                                    "makes one exact attempt without fallback and "
                                    "shows a warning."
                                )
                            else:
                                self._set_provider_status(
                                    "Profile availability is unverified. Generate "
                                    "makes one exact attempt without fallback and "
                                    "shows a warning."
                                )
                            self._sync_generate_enabled()
                            return
                        self._stale_providers.add(provider_id)
                        self._catalog_generation_allowed = False
                        if self._profile_effective_availability == "unavailable":
                            self._set_provider_status(
                                "The exact profile selection is unavailable. Return "
                                "to Voice profiles and choose Edit."
                            )
                        else:
                            self._set_provider_status(
                                self._catalog_error_copy(error, provider_id)
                            )
                        self._sync_generate_enabled()
                        return
                    self._stale_providers.add(provider_id)
                    self._catalog_generation_allowed = False
                    self._set_provider_status(
                        self._catalog_error_copy(error, provider_id)
                    )
                    self._sync_generate_enabled()
                return
            logger.warning(
                "TTS voice discovery failed ({})",
                type(error).__name__,
            )
            failed_snapshot = self._control_snapshot_for(provider_id)
            failed_voice = failed_snapshot.get("voice_id")
            self._discovered_voices.pop((provider_id, model_id), None)
            if isinstance(failed_voice, str):
                self._pending_voice_selections[provider_id] = failed_voice
            else:
                self._pending_voice_selections.pop(provider_id, None)
            catalog = self._catalogs.get(provider_id)
            preset = self._profile_preset
            if (
                preset is not None
                and preset.provider_id == provider_id
                and self._profile_effective_availability != "unavailable"
            ):
                self._profile_effective_availability = "unverified"
            if catalog is not None:
                self._apply_catalog(provider_id, catalog)
            if preset is not None and preset.provider_id == provider_id:
                if self._profile_effective_availability != "unavailable":
                    if preset_has_no_catalog_check(preset):
                        self._set_provider_status(
                            "This provider has no catalog check. The exact "
                            "selection remains selected without fallback."
                        )
                    else:
                        self._set_provider_status(
                            "Exact profile voice discovery is unverified; "
                            "the exact selection remains selected without fallback."
                        )
            else:
                # Investigated (slice 2 task 3): this "exact selection" is a
                # manually chosen voice for `provider_id`, not an adopted
                # profile -- this branch only runs when there is no matching
                # preset (`preset is None` or `preset.provider_id !=
                # provider_id`). It carries no legacy-vs-audio.cpp adoption
                # story to tell, so it is left as-is.
                self._set_provider_status(
                    "Voices are unavailable; the exact selection remains unverified"
                    if isinstance(failed_voice, str)
                    else (
                        "Voices are unavailable; the provider default remains available"
                    )
                )
            return

        if not self._voice_token_is_current(request_token):
            self._finish_voice_status(request_token, unavailable=False)
            self._clear_profile_voice_validation(request_token)
            return
        voice_unverified = bool(
            observation is not None and observation.state != "complete"
        )
        if voice_unverified:
            self._discovered_voices.pop((provider_id, model_id), None)
        else:
            self._discovered_voices[(provider_id, model_id)] = tuple(voices)
        catalog = self._catalogs.get(provider_id)
        preset = self._profile_preset
        if preset is not None and preset.provider_id == provider_id:
            if observation is not None:
                if observation.state == "unverified":
                    if self._profile_effective_availability != "unavailable":
                        self._profile_effective_availability = "unverified"
                elif observation.state == "model_missing":
                    self._profile_effective_availability = "unavailable"
                elif (
                    preset.voice_id is not None
                    and preset.voice_id not in observation.voices
                ):
                    self._profile_effective_availability = "unavailable"
                elif catalog is not None:
                    self._profile_effective_availability = (
                        profile_availability_from_catalog(preset, catalog)
                    )
            elif catalog is not None:
                self._profile_effective_availability = (
                    profile_availability_from_catalog(preset, catalog)
                )
        if catalog is not None:
            self._apply_catalog(provider_id, catalog)
        self._finish_voice_status(
            request_token,
            unavailable=voice_unverified,
        )
        if voice_unverified and preset is None:
            # Investigated (slice 2 task 3): `voice_unverified` can only be
            # True when `observation` was populated above, which only
            # happens `if provider_id == AUDIO_CPP_PROVIDER_ID`. This branch
            # is audio.cpp-only reachable AND requires `preset is None` (no
            # adopted profile in play), so it has no legacy-adoption story
            # to tell -- left as-is.
            selected_voice = self._current_select_value("#tts-voice-select")
            self._set_provider_status(
                "Voices are unavailable; the exact selection remains unverified"
                if isinstance(selected_voice, str)
                else "Voices are unavailable; the provider default remains available"
            )
        self._clear_profile_voice_validation(request_token)

    def _finish_voice_status(
        self,
        token: CatalogRequestToken,
        *,
        unavailable: bool,
    ) -> None:
        """Finish only the newest voice request for one provider/model scope."""

        key = (token.provider_id, token.model_id or "")
        if token.request_generation != self._voice_request_generations.get(key):
            return
        self._voice_checking_scopes.discard(key)
        self._voice_runtime_observed_at[key] = datetime.now(timezone.utc)
        if unavailable:
            self._voice_unavailable_scopes.add(key)
        else:
            self._voice_unavailable_scopes.discard(key)
        self._sync_truthful_status_rows_if_supported()

    def _voice_token_is_current(self, token: CatalogRequestToken) -> bool:
        """Return whether a voice result still targets the displayed model."""
        service = self._tts_service
        if service is None or not self.is_mounted:
            return False
        catalog = self._catalogs.get(token.provider_id)
        current_revision = catalog.revision if catalog is not None else None
        selected_model = self._current_select_value("#tts-model-select")
        current_model = selected_model if isinstance(selected_model, str) else None
        try:
            configuration_revision = service.configuration_revision(token.provider_id)
        except (KeyError, TTSRegistryClosedError):
            return False
        return token.matches(
            provider_id=self._selected_provider_id or "",
            configuration_revision=configuration_revision,
            catalog_revision=current_revision,
            model_id=current_model,
            request_generation=self._voice_request_generations.get(
                (token.provider_id, token.model_id or "")
            ),
        )

    def _catalog_token_is_current(self, token: CatalogRequestToken) -> bool:
        service = self._tts_service
        if service is None:
            return False
        try:
            configuration_revision = service.configuration_revision(token.provider_id)
        except (KeyError, TTSRegistryClosedError):
            return False
        return token.matches(
            provider_id=self._selected_provider_id or "",
            configuration_revision=configuration_revision,
            catalog_revision=None,
            model_id=None,
            request_generation=self._catalog_request_generations.get(token.provider_id),
        )

    def _catalog_request_is_latest(self, token: CatalogRequestToken) -> bool:
        """Return whether a catalog token is still its provider's newest request."""
        return token.request_generation == self._catalog_request_generations.get(
            token.provider_id
        )

    def _mark_stale_catalog_result(self, token: CatalogRequestToken) -> None:
        if token.provider_id != self._selected_provider_id:
            return
        self._profile_preview_loading = False
        self._stale_providers.add(token.provider_id)
        self._catalog_checking_providers.discard(token.provider_id)
        self._catalog_generation_allowed = False
        preset = self._profile_preset
        if preset is not None and preset.provider_id == token.provider_id:
            if preset.availability != "unavailable":
                self._profile_effective_availability = "unverified"
            self._project_profile_preset_controls(
                token.provider_id,
                generation_allowed=False,
            )
        display_name = self._provider_display_name(token.provider_id)
        self._set_provider_status(f"{display_name} settings changed; refresh models")
        self._sync_generate_enabled()

    def _seeded_axis_value(self, axis: str, provider_id: str) -> str | None:
        """Return the pane's seeded value for one axis, provider-guarded.

        The host pane is seeded with the user's saved global/Studio axis
        selection (`axis_values` for the session, `axis_defaults` for the
        persisted default — see `_seed_axis_defaults`, which already applies
        exact-mode-only and absence discipline). Legacy providers must
        consult that seed before substituting the hardcoded catalog default,
        or a saved exact custom selection (a custom OpenAI-compatible
        endpoint's model/voice, TASK-15421) is silently replaced with
        `tts-1`/`alloy`.

        The seed is honoured only when it was saved for this provider —
        axis values are scoped to the provider they were saved with, and
        applying an OpenAI model name to another provider would mislabel
        it as that provider's default.

        Deliberately reads ``axis_defaults`` ONLY, never ``axis_values``:
        the defaults are written once at construction from the saved
        configuration, while the session values are mirrored from every
        Select change — after a provider switch they transiently pair the
        new provider id with the previous provider's still-displayed model
        and voice, which would let stale carryover masquerade as a saved
        custom selection. In-session explicit choices are already carried
        by the per-provider control snapshots, not by this seed.

        Args:
            axis: The axis control id, e.g. ``"tts-model-select"``.
            provider_id: Provider whose catalog is being applied.

        Returns:
            The saved default, or ``None`` when absent, blank, or saved
            for a different provider. Hosts without axis defaults return
            ``None``.
        """
        source = getattr(self, "axis_defaults", None)
        if not isinstance(source, dict):
            return None
        if source.get("tts-provider-select") != provider_id:
            return None
        seeded = source.get(axis)
        if isinstance(seeded, str) and seeded:
            return seeded
        return None

    def _apply_catalog(
        self,
        provider_id: str,
        catalog: TTSProviderCatalog,
    ) -> None:
        if provider_id != self._selected_provider_id:
            return
        snapshot = self._control_snapshot_for(provider_id)
        preset = self._profile_preset
        if preset is not None and preset.provider_id != provider_id:
            preset = None
        if preset is not None:
            selected_model: object = preset.model_id
            selected_voice: object = preset.voice_id
            selected_format: object = preset.response_format
            speed = preset.speed
        else:
            selected_model = snapshot.get("model_id")
            if selected_model is None:
                if provider_id == AUDIO_CPP_PROVIDER_ID:
                    configured_model = self._cli_setting(
                        "app_tts",
                        "default_model",
                        None,
                    )
                    selected_model = (
                        configured_model
                        if isinstance(configured_model, str) and configured_model
                        else None
                    )
                else:
                    selected_model = (
                        self._seeded_axis_value("tts-model-select", provider_id)
                        or LEGACY_DEFAULT_MODELS.get(provider_id)
                    )
            selected_voice = snapshot.get("voice_id")
            if selected_voice is None:
                if provider_id == AUDIO_CPP_PROVIDER_ID:
                    configured_voice = self._cli_setting(
                        "app_tts",
                        "default_voice",
                        None,
                    )
                    selected_voice = (
                        configured_voice
                        if isinstance(configured_voice, str) and configured_voice
                        else None
                    )
                else:
                    selected_voice = (
                        self._seeded_axis_value("tts-voice-select", provider_id)
                        or LEGACY_DEFAULT_VOICES.get(provider_id)
                    )
            selected_format = snapshot.get("response_format")
            if selected_format is None:
                selected_format = self._cli_setting("app_tts", "default_format", None)
            speed = self._snapshot_speed(snapshot)
            if provider_id == "openai":
                # A custom (non-catalog) OpenAI id is pinned downstream
                # (TASK-15421), so stale cross-provider carryover must be
                # laundered HERE: on a provider switch the transient control
                # snapshot can record the previous provider's still-displayed
                # values under this provider's key, and the first-catalog
                # fallback that used to clean those up no longer runs for
                # openai. The only legitimate custom id is the provider-
                # guarded seed — the playground offers no free-text entry.
                catalog_model_ids = {model.model_id for model in catalog.models}
                if (
                    isinstance(selected_model, str)
                    and selected_model not in catalog_model_ids
                    and selected_model
                    != self._seeded_axis_value("tts-model-select", provider_id)
                ):
                    selected_model = self._seeded_axis_value(
                        "tts-model-select", provider_id
                    ) or LEGACY_DEFAULT_MODELS.get(provider_id)
                catalog_voice_ids = {
                    voice
                    for model in catalog.models
                    for voice in model.voices
                }
                if (
                    isinstance(selected_voice, str)
                    and selected_voice not in catalog_voice_ids
                    and selected_voice
                    != self._seeded_axis_value("tts-voice-select", provider_id)
                ):
                    selected_voice = self._seeded_axis_value(
                        "tts-voice-select", provider_id
                    ) or LEGACY_DEFAULT_VOICES.get(provider_id)
        pending_voice = self._pending_voice_selections.get(provider_id)
        if pending_voice is not None and preset is None:
            selected_voice = pending_voice

        voice_choices: tuple[tuple[str, SelectValue], ...] | None = None
        discovered_voices: tuple[str, ...] | None
        if provider_id == AUDIO_CPP_PROVIDER_ID:
            model_for_voices = (
                preset.model_id
                if preset is not None
                else self._catalog_model_id(catalog, selected_model)
            )
            discovered_voices = (
                self._discovered_voices.get((provider_id, model_for_voices))
                if model_for_voices is not None
                else None
            )
            voice_discovery_pending = discovered_voices is None
            if (
                voice_discovery_pending
                and preset is None
                and isinstance(selected_voice, str)
                and selected_voice
            ):
                pending_voice = selected_voice
                self._pending_voice_selections[provider_id] = selected_voice
        else:
            model_for_voices = self._catalog_model_id(catalog, selected_model)
            base_voices = self._catalog_model_voices(catalog, model_for_voices)
            voice_choices = self._legacy_voice_choices(provider_id, base_voices)
            discovered_voices = tuple(value for _label, value in voice_choices)
            voice_discovery_pending = False

        if preset is not None:
            controls = controls_from_profile_preset(
                catalog,
                preset=preset,
                discovered_voices=discovered_voices,
            )
        else:
            controls = controls_from_catalog(
                catalog,
                selected_model_id=(
                    selected_model if isinstance(selected_model, str) else None
                ),
                selected_voice_id=(
                    selected_voice
                    if isinstance(selected_voice, (str, SelectSentinel))
                    else None
                ),
                discovered_voices=discovered_voices,
                selected_format=(
                    selected_format if isinstance(selected_format, str) else None
                ),
                speed=speed,
            )
        if voice_choices is not None:
            # This display-label projection must not clobber a custom voice
            # `controls_from_catalog` pinned for a custom OpenAI-compatible
            # endpoint (TASK-15421) -- carry the pin across the swap.
            if (
                provider_id == "openai"
                and isinstance(controls.selected_voice_id, str)
                and controls.selected_voice_id
                not in {value for _label, value in voice_choices}
            ):
                voice_choices = (
                    *voice_choices,
                    pinned_no_catalog_check_option(controls.selected_voice_id),
                )
            controls = replace(controls, voice_options=voice_choices)
        if preset is None and voice_discovery_pending and pending_voice is not None:
            model_changed = (
                selected_model is not None
                and selected_model != controls.selected_model_id
            )
            controls = replace(controls, selection_changed=model_changed)
        self._apply_controls(controls)
        if preset is None and voice_discovery_pending and pending_voice is not None:
            self._provider_control_snapshots.setdefault(provider_id, {})["voice_id"] = (
                pending_voice
            )
            self._catalog_generation_allowed = False
            self._sync_generate_enabled()
        elif provider_id == AUDIO_CPP_PROVIDER_ID and discovered_voices is not None:
            self._pending_voice_selections.pop(provider_id, None)
            # `_apply_controls` above already ran its own `_sync_generate_
            # enabled()`, but with this entry still IN `_pending_voice_
            # selections` -- the pop happens after it returns. Harmless
            # while nothing but this dict's own removal read it; TASK-2951
            # surfaced it because the unified readiness gate now blocks on
            # `provider_id in self._pending_voice_selections` too, so a
            # stale pre-pop "voices are still loading" verdict would
            # otherwise survive on the button after discovery actually
            # finished, with nothing left to correct it.
            self._sync_generate_enabled()
        consume_navigation = getattr(
            self,
            "_consume_navigation_target_after_catalog",
            None,
        )
        if callable(consume_navigation):
            consume_navigation(provider_id)

    def _apply_controls(self, controls: PlaygroundControls) -> None:
        model_select = self.query_one("#tts-model-select", Select)
        voice_select = self.query_one("#tts-voice-select", Select)
        format_select = self.query_one("#tts-format-select", Select)
        speed_input = self.query_one("#tts-speed-input", Input)
        self._applied_model_id = controls.selected_model_id
        self._applied_voice_id = controls.selected_voice_id
        self._applied_format = controls.selected_format
        self._applying_catalog_controls = True
        try:
            self._set_select_state(
                model_select,
                controls.model_options,
                controls.selected_model_id,
                "No models available",
            )
            self._set_select_state(
                voice_select,
                controls.voice_options,
                controls.selected_voice_id,
                "No voices available",
            )
            format_options = tuple(
                (audio_format.upper(), audio_format)
                for audio_format in controls.format_options
            )
            self._set_select_state(
                format_select,
                format_options,
                controls.selected_format,
                "No formats available",
            )
            format_select.disabled = controls.format_locked
            speed_input.value = str(controls.speed)
            speed_input.disabled = controls.speed_locked
        finally:
            self._applying_catalog_controls = False

        self._update_axis_model_from_controls(controls)

        restriction = self.query_one("#tts-audio-cpp-restrictions", Static)
        if controls.provider_id == AUDIO_CPP_PROVIDER_ID:
            restriction.remove_class("hidden")
            format_select.tooltip = "audio.cpp returns one complete WAV response"
            speed_input.tooltip = "audio.cpp currently supports speed 1.0"
        else:
            restriction.add_class("hidden")
            format_select.tooltip = None
            speed_input.tooltip = None

        catalog = self._catalogs.get(controls.provider_id)
        self._displayed_provider_id = controls.provider_id
        preset = self._profile_preset
        if preset is not None and preset.provider_id != controls.provider_id:
            preset = None
        if preset is not None:
            model_select.add_class("profile-exact-select")
            voice_select.add_class("profile-exact-select")
            model_select.tooltip = Text(preset.model_id)
            voice_select.tooltip = Text(
                preset.voice_id
                if preset.voice_id is not None
                else SERVER_DEFAULT_VOICE_LABEL
            )
            availability = self._profile_effective_availability
            self._catalog_generation_allowed = bool(
                controls.generation_allowed and availability != "unavailable"
            )
            if availability == "unavailable":
                self._set_provider_status(
                    "The exact profile selection is unavailable. Return to Voice "
                    "profiles and choose Edit."
                )
            elif availability == "unverified":
                if preset_has_no_catalog_check(preset):
                    self._set_provider_status(
                        "This provider has no catalog check. Generate makes one "
                        "exact attempt without fallback and shows a warning."
                    )
                else:
                    self._set_provider_status(
                        "Profile availability is unverified. Generate makes one "
                        "exact attempt without fallback and shows a warning."
                    )
            else:
                self._set_provider_status(
                    "Profile preview loaded with its exact persisted selection."
                )
        else:
            model_select.remove_class("profile-exact-select")
            voice_select.remove_class("profile-exact-select")
            model_select.tooltip = None
            voice_select.tooltip = None
            service = self._tts_service
            self._catalog_generation_allowed = (
                controls.generation_allowed
                and service is not None
                and catalog is not None
                and controls.provider_id not in self._stale_providers
                and self._catalog_configuration_revisions.get(controls.provider_id)
                == service.configuration_revision(controls.provider_id)
            )
            if catalog is not None:
                self._set_provider_status(self._catalog_health_copy(catalog))
        self._remember_current_controls(controls.provider_id)
        if preset is not None:
            self._profile_controls_applied = True
        self._sync_generate_enabled()
        if controls.selection_changed:
            self.app.notify(
                "Available models or voices changed; a valid selection was chosen",
                severity="warning",
            )

    def _update_axis_model_from_controls(self, controls: PlaygroundControls) -> None:
        """Mirror an applied catalog/preset projection into the axis model.

        `SpeechPlaygroundPane.axis_values`/`axis_defaults` are the model of
        record for the axis row's override markers
        (`Docs/superpowers/specs/2026-07-30-speech-preset-axis-ownership.md`).
        `_apply_controls` is one of three writers of that model -- this
        keeps model/voice/format/speed and provider all in step with what
        was just written to the Selects, then repaints the row.

        A no-op for hosts with no `axis_values` -- `SpeechCatalogMixin` makes
        no assumption that every host carries an axis row and an axis model.

        Model, voice and format are each POPPED, not merely left alone, when
        the projection resolves them to nothing (e.g. a provider whose
        catalog has no models) or to a non-`str` sentinel (e.g. the
        server-default voice): a stale key would otherwise keep describing
        the previous provider's value, or leak an internal `SelectSentinel`
        repr into the model of record.

        Args:
            controls: The projection just applied to the widgets.
        """
        axis_values = getattr(self, "axis_values", None)
        if axis_values is None:
            return
        axis_values["tts-provider-select"] = controls.provider_id
        if isinstance(controls.selected_model_id, str):
            axis_values["tts-model-select"] = controls.selected_model_id
        else:
            axis_values.pop("tts-model-select", None)
        if isinstance(controls.selected_voice_id, str):
            axis_values["tts-voice-select"] = controls.selected_voice_id
        else:
            axis_values.pop("tts-voice-select", None)
        if isinstance(controls.selected_format, str):
            axis_values["tts-format-select"] = controls.selected_format
        else:
            axis_values.pop("tts-format-select", None)
        axis_values["tts-speed-input"] = str(controls.speed)
        # `_refresh_axis_markers` is defined only on `SpeechPlaygroundPane`;
        # reaching this line at all already required the `axis_values`
        # getattr gate above, which is what makes calling it safe on any
        # host with no axis row.
        self._refresh_axis_markers()

    @staticmethod
    def _safe_select_options(
        options: tuple[tuple[str, SelectValue], ...],
    ) -> list[tuple[Text, SelectValue]]:
        return [(Text(label, no_wrap=True), value) for label, value in options]

    def _set_select_state(
        self,
        select: Select,
        options: tuple[tuple[str, SelectValue], ...],
        selected: SelectValue | None,
        empty_label: str,
    ) -> None:
        if not options:
            select.set_options([(empty_label, UNAVAILABLE_SELECT_VALUE)])
            select.value = UNAVAILABLE_SELECT_VALUE
            # ``set_options`` may keep the same value, in which case Textual
            # does not rerun Select's watcher and its closed prompt keeps the
            # previous label. Force that repaint without emitting a synthetic
            # user selection event.
            with select.prevent(Select.Changed):
                select.mutate_reactive(Select.value)
            select.disabled = True
            return
        select.set_options(self._safe_select_options(options))
        select.disabled = False
        select.value = selected or options[0][1]
        with select.prevent(Select.Changed):
            select.mutate_reactive(Select.value)

    def _control_snapshot_for(self, provider_id: str) -> dict[str, Any]:
        if getattr(self, "_displayed_provider_id", None) == provider_id:
            self._remember_current_controls(provider_id)
        return dict(self._provider_control_snapshots.get(provider_id, {}))

    def _remember_current_controls(self, provider_id: str) -> None:
        if getattr(self, "_displayed_provider_id", None) != provider_id:
            return
        speed_value = self.query_one("#tts-speed-input", Input).value
        try:
            speed = float(speed_value)
        except ValueError:
            speed = 1.0
        self._provider_control_snapshots[provider_id] = {
            "model_id": self._current_select_value("#tts-model-select"),
            "voice_id": self._current_select_value("#tts-voice-select"),
            "response_format": self._current_select_value("#tts-format-select"),
            "speed": speed,
        }

    @staticmethod
    def _snapshot_speed(snapshot: Mapping[str, Any]) -> float:
        speed = snapshot.get("speed", 1.0)
        try:
            return float(speed)
        except (TypeError, ValueError):
            return 1.0

    def _current_select_value(self, selector: str) -> SelectValue | None:
        value = self.query_one(selector, Select).value
        if value is LOADING_SELECT_VALUE or value is UNAVAILABLE_SELECT_VALUE:
            return None
        return value if isinstance(value, (str, SelectSentinel)) else None

    @staticmethod
    def _catalog_model_id(
        catalog: TTSProviderCatalog,
        selected_model_id: object,
    ) -> str | None:
        if isinstance(selected_model_id, str) and any(
            model.model_id == selected_model_id for model in catalog.models
        ):
            return selected_model_id
        return catalog.models[0].model_id if catalog.models else None

    @staticmethod
    def _catalog_model_voices(
        catalog: TTSProviderCatalog,
        model_id: str | None,
    ) -> tuple[str, ...]:
        for model in catalog.models:
            if model.model_id == model_id:
                return model.voices
        return ()

    def _legacy_voice_choices(
        self,
        provider_id: str,
        base_voices: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        configured_choices = LEGACY_VOICE_OPTIONS.get(provider_id)
        choices = (
            list(configured_choices)
            if configured_choices is not None
            else [(voice.replace("_", " ").title(), voice) for voice in base_voices]
        )
        if provider_id == "chatterbox":
            choices.extend(self._chatterbox_profile_choices())
        elif provider_id == "higgs":
            choices.extend(self._higgs_profile_choices())
        elif provider_id == "kokoro":
            choices.extend(self._kokoro_blend_choices())
        return tuple(choices)

    @staticmethod
    def _kokoro_blend_choices() -> list[tuple[str, str]]:
        blend_file = kokoro_ui_blend_file()
        if not blend_file.is_file():
            return []
        try:
            payload = json.loads(blend_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            logger.warning("Saved Kokoro voice blends could not be loaded")
            return []
        if not isinstance(payload, Mapping):
            return []
        return [
            (f"Voice blend: {name}", f"blend:{name}")
            for name in payload
            if isinstance(name, str) and name
        ]

    @staticmethod
    def _chatterbox_profile_choices() -> list[tuple[str, str]]:
        try:
            from tldw_chatbook.TTS.backends.chatterbox_voice_manager import (
                ChatterboxVoiceManager,
            )

            voice_dir = Path.home() / ".config" / "tldw_cli" / "chatterbox_voices"
            if not voice_dir.is_dir():
                return []
            profiles = ChatterboxVoiceManager(voice_dir).list_profiles()
            return [
                (str(profile.get("display_name") or profile["name"]), profile["name"])
                for profile in profiles
                if isinstance(profile, Mapping)
                and isinstance(profile.get("name"), str)
                and profile["name"]
            ]
        except Exception:
            logger.warning("Saved Chatterbox voice profiles could not be loaded")
            return []

    @staticmethod
    def _higgs_profile_choices() -> list[tuple[str, str]]:
        try:
            from tldw_chatbook.TTS.backends.higgs_voice_manager import (
                HiggsVoiceProfileManager,
            )

            voice_dir = Path.home() / ".config" / "tldw_cli" / "higgs_voices"
            if not voice_dir.is_dir():
                return []
            profiles = HiggsVoiceProfileManager(voice_dir).list_profiles()
            return [
                (
                    str(profile.get("display_name") or profile["name"]),
                    f"profile:{profile['name']}",
                )
                for profile in profiles
                if isinstance(profile, Mapping)
                and isinstance(profile.get("name"), str)
                and profile["name"]
            ]
        except Exception:
            logger.warning("Saved Higgs voice profiles could not be loaded")
            return []

    def _catalog_health_copy(self, catalog: TTSProviderCatalog) -> str:
        display_name = self._provider_display_name(catalog.provider_id)
        if catalog.provider_id in self._stale_providers:
            return f"{display_name} settings changed; refresh models"
        health = catalog.health
        if health.state == "available" and health.fresh:
            return f"{display_name} is ready"
        if health.state == "available":
            return f"{display_name} catalog is stale; refresh models"
        if health.state == "not_configured":
            return f"{display_name} is not configured; open STTS Settings"
        if health.state == "reconfiguring":
            return f"{display_name} settings are being applied; retry shortly"
        if health.state == "closed":
            return "The TTS service is unavailable"
        return f"{display_name} is unavailable; check STTS Settings"

    def _provider_display_name(self, provider_id: str) -> str:
        return self._provider_display_names.get(provider_id, "TTS provider")

    def _catalog_error_copy(self, error: Exception, provider_id: str) -> str:
        display_name = self._provider_display_name(provider_id)
        if isinstance(error, TTSProviderReconfiguringError):
            return f"{display_name} settings are being applied; retry shortly"
        if isinstance(error, TTSRegistryClosedError):
            return "The TTS service is unavailable"
        if isinstance(error, TTSOperationError):
            if error.code in {"configuration_invalid", "not_configured"}:
                return f"{display_name} is not configured; open STTS Settings"
            if error.code == "contract_incompatible":
                return f"The configured {display_name} service is incompatible"
            return f"{display_name} is unavailable; check STTS Settings"
        if isinstance(error, ValueError):
            return f"{display_name} is not configured; open STTS Settings"
        return f"{display_name} is unavailable; check STTS Settings"

    def _catalog_failure(
        self,
        provider_id: str,
        copy: str,
        *,
        exact_attempt_allowed: bool = False,
    ) -> None:
        logger.warning("TTS catalog discovery failed for {}", provider_id)
        if provider_id != self._selected_provider_id:
            return
        self._catalog_checking_providers.discard(provider_id)
        self._catalog_unavailable_providers.add(provider_id)
        self._catalog_runtime_observed_at[provider_id] = datetime.now(timezone.utc)
        self._profile_preview_loading = False
        preset = self._profile_preset
        if preset is not None and preset.provider_id == provider_id:
            if preset.availability != "unavailable":
                self._profile_effective_availability = "unverified"
            generation_allowed = bool(
                exact_attempt_allowed
                and self._profile_effective_availability != "unavailable"
            )
            self._stale_providers.add(provider_id)
            self._project_profile_preset_controls(
                provider_id,
                generation_allowed=generation_allowed,
            )
            if (
                not generation_allowed
                and self._profile_effective_availability != "unavailable"
            ):
                self._set_provider_status(copy)
            self._sync_generate_enabled()
            self._sync_truthful_status_rows_if_supported()
            return
        self._stale_providers.add(provider_id)
        self._catalog_generation_allowed = False
        self._set_provider_status(copy)
        self._sync_generate_enabled()

    def _set_provider_status(self, copy: str) -> None:
        self.query_one("#tts-provider-status", Static).update(Text(copy))
        self._sync_profile_preview_status()
        self._sync_truthful_status_rows_if_supported()

    def _sync_truthful_status_rows_if_supported(self) -> None:
        """Notify redesigned hosts after bounded catalog state changes."""

        sync = getattr(self, "_sync_truthful_status_rows", None)
        if callable(sync):
            sync()

    def _show_provider_specific_controls(self, provider_id: str) -> None:
        language_row = self.query_one("#kokoro-language-row", Horizontal)
        kokoro_settings = self.query_one("#kokoro-settings", Vertical)
        elevenlabs_settings = self.query_one("#elevenlabs-settings", Vertical)
        chatterbox_settings = self.query_one("#chatterbox-settings", Vertical)
        higgs_settings = self.query_one("#higgs-settings", Vertical)
        language_row.set_class(provider_id == "kokoro", "visible")
        kokoro_settings.set_class(provider_id == "kokoro", "visible")
        elevenlabs_settings.set_class(provider_id == "elevenlabs", "visible")
        chatterbox_settings.set_class(provider_id == "chatterbox", "visible")
        higgs_settings.set_class(provider_id == "higgs", "visible")
        if provider_id == "higgs":
            self._check_higgs_installation()

    def handle_provider_select_changed(self, event: Select.Changed) -> None:
        """Respond to any Select change, filtering by id internally.

        Deliberately NOT decorated with `@on` here. Textual registers
        decorated handlers in its metaclass, scanning only each class's own
        namespace -- a plain mixin never passes through that metaclass, so an
        `@on` method defined here is silently never dispatched. No error, no
        warning: provider switching simply stops working. Each host declares
        the thin decorated handler and delegates to this.
        """
        """Handle canonical provider/model/voice/format selections."""
        if self._applying_catalog_controls:
            return
        if event.value != event.select.value:
            return
        if event.select.id == "tts-provider-select":
            if not isinstance(event.value, str) or event.value not in getattr(
                self, "_provider_ids", ()
            ):
                return
            if event.value == self._selected_provider_id:
                return
            consume_navigation = getattr(
                self,
                "_consume_navigation_target_on_provider_change",
                None,
            )
            if callable(consume_navigation):
                consume_navigation(event.value)
            self._end_profile_preset(before_controls=True)
            if self._selected_provider_id is not None:
                self._remember_current_controls(self._selected_provider_id)
            self._selected_provider_id = event.value
            self._show_provider_specific_controls(event.value)
            self._catalog_generation_allowed = False
            self._sync_generate_enabled()
            self._load_provider_catalog(event.value)
            return
        if event.select.id == "tts-model-select":
            provider_id = self._selected_provider_id
            if provider_id is None or not isinstance(event.value, str):
                return
            if event.value == self._applied_model_id:
                return
            self._end_profile_preset()
            self._remember_current_controls(provider_id)
            catalog = self._catalogs.get(provider_id)
            if catalog is not None:
                self._apply_catalog(provider_id, catalog)
                model_id = self._current_select_value("#tts-model-select")
                if isinstance(model_id, str):
                    self._load_provider_voices(
                        provider_id,
                        model_id,
                        catalog.revision,
                    )
            return
        if event.select.id in {"tts-voice-select", "tts-format-select"}:
            if (
                event.select.id == "tts-voice-select"
                and event.value == self._applied_voice_id
            ) or (
                event.select.id == "tts-format-select"
                and event.value == self._applied_format
            ):
                return
            preset_ended = self._end_profile_preset()
            if event.select.id == "tts-voice-select":
                self._applied_voice_id = (
                    event.value
                    if isinstance(event.value, (str, SelectSentinel))
                    else None
                )
                if (
                    event.value is SERVER_DEFAULT_VOICE_ID
                    and self._selected_provider_id is not None
                ):
                    self._pending_voice_selections.pop(
                        self._selected_provider_id,
                        None,
                    )
            else:
                self._applied_format = (
                    event.value if isinstance(event.value, str) else None
                )
            if self._selected_provider_id is not None:
                self._remember_current_controls(self._selected_provider_id)
            if preset_ended or (
                event.select.id == "tts-voice-select"
                and self._selected_provider_id == AUDIO_CPP_PROVIDER_ID
            ):
                self._reproject_current_catalog()
            else:
                self._sync_generate_enabled()
            return
        if event.select.has_focus and self._end_profile_preset():
            self._reproject_current_catalog()

    def _check_higgs_installation(self) -> None:
        """Check if Higgs Audio is properly installed"""
        try:
            import boson_multimodal  # noqa: F401

            logger.info("Higgs Audio is installed and available")
        except ImportError:
            self.app.notify(
                "⚠️ Higgs Audio not installed! Run: ./scripts/install_higgs.sh",
                severity="warning",
                timeout=10,
            )
            logger.warning("Higgs Audio (boson_multimodal) is not installed")

    def mark_provider_configuration_changed(
        self,
        provider_id: str,
        configuration_revision: int,
    ) -> None:
        """Invalidate cached controls after a changed provider configuration."""
        del configuration_revision
        self._stale_providers.add(provider_id)
        self._discovered_voices = {
            key: value
            for key, value in self._discovered_voices.items()
            if key[0] != provider_id
        }
        self._voice_checking_scopes = {
            key for key in self._voice_checking_scopes if key[0] != provider_id
        }
        self._voice_unavailable_scopes = {
            key for key in self._voice_unavailable_scopes if key[0] != provider_id
        }
        self._voice_runtime_observed_at = {
            key: value
            for key, value in self._voice_runtime_observed_at.items()
            if key[0] != provider_id
        }
        # TASK-3000: an exact profile's voice validation can be in flight
        # on a request that ignores cancellation and keeps running past
        # the `cancel_group` calls below -- ported verbatim from the
        # retired widget's own `mark_provider_configuration_changed`
        # (`git show f560217fb~1:tldw_chatbook/UI/STTS_Window.py`), which
        # detached the token right here, unconditionally, the moment a
        # config change targeted its own provider. Without this, nothing
        # downstream (this method's own worker cancellation, a failed
        # catalog reload's exception path, `_catalog_failure`) ever clears
        # it, `_generation_readiness_error`'s preset branch blocks
        # unconditionally on it being non-None, and Generate stays
        # disabled with no way to recover short of leaving and
        # re-entering the Playground.
        pending_voice_token = self._profile_voice_validation_token
        if (
            pending_voice_token is not None
            and pending_voice_token.provider_id == provider_id
        ):
            self._profile_voice_validation_token = None
        if provider_id != self._selected_provider_id:
            return
        self.app.workers.cancel_group(self, "stts-catalog-discovery")
        self.app.workers.cancel_group(self, "stts-voice-discovery")
        self._catalog_generation_allowed = False
        display_name = self._provider_display_name(provider_id)
        self._set_provider_status(f"{display_name} settings changed; refresh models")
        self._sync_generate_enabled()

    def _rehydrate_handler_state(self) -> None:
        handler = getattr(self.app, "_stts_handler", None)
        snapshot_getter = getattr(handler, "playground_state", None)
        if not callable(snapshot_getter):
            return
        try:
            state = snapshot_getter()
        except Exception as error:
            logger.debug(
                "Could not rehydrate TTS Playground state ({})",
                type(error).__name__,
            )
            return
        artifact = getattr(state, "artifact", None)
        if (
            type(artifact) is STTSPlaygroundResultProjection
            and artifact.path.exists()
        ):
            self._store_delivered_artifact(artifact, announce=False)
        active_operation_id = getattr(state, "active_operation_id", None)
        if getattr(state, "generation_active", False) and isinstance(
            active_operation_id,
            str,
        ):
            self._generation_operation_id = active_operation_id
            self.query_one("#generation-status-container").remove_class("hidden")
            self.query_one("#generation-status-text", Static).update(
                "Generation in progress…"
            )
            self.query_one("#tts-generate-btn", Button).disabled = True

    def _reproject_current_catalog(self) -> None:
        provider_id = self._selected_provider_id
        if provider_id is None:
            return
        catalog = self._catalogs.get(provider_id)
        if catalog is not None:
            self._apply_catalog(provider_id, catalog)

    def _reserve_voice_request_token(
        self,
        provider_id: str,
        model_id: str,
        catalog_revision: int,
    ) -> CatalogRequestToken:
        """Reserve one voice request and capture its catalog authority."""
        request_key = (provider_id, model_id)
        request_generation = self._voice_request_generations.get(request_key, 0) + 1
        self._voice_request_generations[request_key] = request_generation
        configuration_revision = self._catalog_configuration_revisions.get(provider_id)
        if configuration_revision is None:
            service = self._tts_service
            if service is None:
                raise TTSRegistryClosedError("The TTS service is unavailable")
            configuration_revision = service.configuration_revision(provider_id)
        return CatalogRequestToken(
            provider_id=provider_id,
            configuration_revision=configuration_revision,
            catalog_revision=catalog_revision,
            model_id=model_id,
            request_generation=request_generation,
        )

    # Decorated on the host, not here: Textual registers `@on`
    # handlers per-class in its metaclass, so one declared in a
    # mixin is never dispatched.
    def handle_option_switch_changed(self, event: Switch.Changed) -> None:
        if (
            not self._applying_catalog_controls
            and event.switch.has_focus
            and self._end_profile_preset()
        ):
            self._reproject_current_catalog()

    # Decorated on the host, not here: Textual registers `@on`
    # handlers per-class in its metaclass, so one declared in a
    # mixin is never dispatched.
    def handle_speed_changed(self, event: Input.Changed) -> None:
        if self._applying_catalog_controls:
            return
        if event.value != event.input.value:
            return
        if self._selected_provider_id is not None:
            if event.input.id == "tts-speed-input":
                self._remember_current_controls(self._selected_provider_id)
                preset = self._profile_preset
                try:
                    unchanged = (
                        preset is not None and float(event.value) == preset.speed
                    )
                except ValueError:
                    unchanged = False
                if unchanged:
                    return
            elif not event.input.has_focus:
                return
            if self._end_profile_preset():
                self._reproject_current_catalog()

    # Decorated on the host, not here: Textual registers `@on`
    # handlers per-class in its metaclass, so one declared in a
    # mixin is never dispatched.
    def handle_text_changed(self, _event: TextArea.Changed) -> None:
        self._sync_generate_enabled()
