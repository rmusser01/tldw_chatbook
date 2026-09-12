"""Console provider selection, readiness, model discovery and intent admission."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from collections.abc import Mapping
from dataclasses import replace
import asyncio
import os
from loguru import logger
from textual.message_pump import NoActiveAppError
from ..Navigation.pending_handoff_store import (
    ConsoleProviderIntent,
    HandoffChannel,
    PendingHandoffStore,
)
from ..Navigation.vllm_handoff import VllmConsoleIntent, owner_has_current_intent
from ..Screens.provider_model_resolution import (
    ResolvedProviderModelOption,
    resolve_effective_provider_model,
    resolve_provider_model_options,
)
from .session import _has_selected_text
from ...Chat.console_chat_models import (
    ConsoleProviderSelection,
    ConsoleWorkspaceContext,
)
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
    build_default_console_session_settings,
    build_console_settings_readiness,
    build_target_default_console_session_settings,
    validate_console_session_settings,
)
from ...Chat.console_session_endpoint_policy import (
    ConsoleEndpointRollbackOutcome,
    ConsoleEphemeralEndpointPolicy,
)
from ...Chat.console_endpoint_provenance import ConsoleEndpointProvenance
from ...Chat.console_chat_store import ConsoleChatStore
from ...Chat.console_provider_endpoints import first_configured_endpoint
from ...Chat.provider_readiness import provider_config_key
from ...config import get_cli_providers_and_models, load_settings


logger = logger.bind(module="ChatScreen")


class ConsoleProviderSelectionController:
    """Own console provider selection, readiness, model discovery and intent admission.

    App identity is stable for this controller lifetime. All other dependencies
    are explicit callables resolved by wiring at use time. No DOM is owned here.
    """

    def __init__(
        self,
        *,
        app_instance_accessor: Callable[[], Any],
        _active_session_settings: Callable[..., Any],
        _apply_console_settings_summary_state: Callable[..., Any],
        _build_console_settings_summary_state: Callable[..., Any],
        _config_section: Callable[..., Any],
        _console_config_snapshot_is_disk_loaded: Callable[..., Any],
        _console_run_active: Callable[..., Any],
        _current_session_settings: Callable[..., Any],
        _ensure_console_chat_store: Callable[..., Any],
        _normalize_llamacpp_base_url: Callable[..., Any],
        _runtime_app_config: Callable[..., Any],
        _set_control_selection: Callable[..., Any],
        _sync_console_chat_core_state: Callable[..., Any],
        _sync_console_control_bar: Callable[..., Any],
        _sync_console_settings_summary: Callable[..., Any],
        _workspace_context: Callable[..., Any],
        _console_control_provider_accessor: Callable[[], Any],
        _console_control_model_accessor: Callable[[], Any],
        _console_chat_controller_accessor: Callable[[], Any],
        _console_derivation_memo_accessor: Callable[[], Any],
        is_attached_accessor: Callable[[], Any],
    ) -> None:
        self._app_instance_accessor = app_instance_accessor
        self._active_session_settings = _active_session_settings
        self._apply_console_settings_summary_state = (
            _apply_console_settings_summary_state
        )
        self._build_console_settings_summary_state = (
            _build_console_settings_summary_state
        )
        self._config_section = _config_section
        self._console_config_snapshot_is_disk_loaded = (
            _console_config_snapshot_is_disk_loaded
        )
        self._console_run_active = _console_run_active
        self._current_session_settings = _current_session_settings
        self._ensure_console_chat_store = _ensure_console_chat_store
        self._normalize_llamacpp_base_url = _normalize_llamacpp_base_url
        self._runtime_app_config = _runtime_app_config
        self._set_control_selection = _set_control_selection
        self._sync_console_chat_core_state = _sync_console_chat_core_state
        self._sync_console_control_bar = _sync_console_control_bar
        self._sync_console_settings_summary = _sync_console_settings_summary
        self._workspace_context = _workspace_context
        self._console_control_provider_accessor = _console_control_provider_accessor
        self._console_control_model_accessor = _console_control_model_accessor
        self._console_chat_controller_accessor = _console_chat_controller_accessor
        self._console_derivation_memo_accessor = _console_derivation_memo_accessor
        self.is_attached_accessor = is_attached_accessor
        self._console_model_option_warnings = {}

    @property
    def _console_control_provider(self) -> Any:
        return self._console_control_provider_accessor()

    @property
    def _console_control_model(self) -> Any:
        return self._console_control_model_accessor()

    @property
    def _console_chat_controller(self) -> Any:
        return self._console_chat_controller_accessor()

    @property
    def _console_derivation_memo(self) -> Any:
        return self._console_derivation_memo_accessor()

    @property
    def is_attached(self) -> Any:
        return self.is_attached_accessor()

    @property
    def app_instance(self) -> Any:
        return self._app_instance_accessor()

    def _provider_readiness_app_config(self) -> Any:
        """Return the freshest app config for provider-readiness checks.

        ``app.app_config`` is a boot-time snapshot: Settings saves invalidate
        the config module cache but never refresh the snapshot, so readiness
        built from it stays blocked until restart (core-loop UAT 2026-07,
        task-177). When the snapshot looks disk-loaded, re-source it from
        ``load_settings()`` - cheap (cached) except right after a save, which
        is exactly when the fresh read matters.

        Served from the per-pass memo inside a `_console_derivation_scope`
        (task-15452): one draft-edit sync called this 63 times.
        """
        memo = self._console_derivation_memo
        if memo is not None and "app_config" in memo:
            return memo["app_config"]
        try:
            app_config = self._runtime_app_config()
        except (AttributeError, NoActiveAppError):
            app_config = getattr(self.app_instance, "app_config", {}) or {}
        app_config = app_config or {}
        resolved = app_config
        if self._console_config_snapshot_is_disk_loaded(app_config):
            try:
                fresh = load_settings()
            except Exception:
                logger.debug(
                    "Console readiness refresh via load_settings() failed; "
                    "using snapshot"
                )
            else:
                if isinstance(fresh, Mapping) and fresh:
                    resolved = fresh
        if memo is not None:
            memo["app_config"] = resolved
        return resolved

    def _effective_console_provider_model(self) -> tuple[Any, Any]:
        """Return the canonical Console provider/model selection.

        Returns:
            A `(provider, model)` tuple using the same precedence for Console
            control labels and run-inspector readiness.
        """
        effective = resolve_effective_provider_model(
            self._persisted_chat_defaults(),
            console_provider=self._console_control_provider,
            console_model=self._console_control_model,
        )
        return effective.provider, effective.model

    def _persisted_chat_defaults(self) -> Mapping[str, Any]:
        """Return the freshest persisted provider/model defaults."""
        config = self._provider_readiness_app_config()
        if not isinstance(config, Mapping):
            return {}
        defaults = config.get("chat_defaults", {})
        return defaults if isinstance(defaults, Mapping) else {}

    def _providers_models(self) -> dict[str, list[str]]:
        """Return configured provider/model options for Console settings."""
        providers_models = getattr(self.app_instance, "providers_models", None)
        if isinstance(providers_models, dict):
            return {
                str(provider): [str(model) for model in models]
                for provider, models in providers_models.items()
                if isinstance(models, (list, tuple))
            }
        try:
            return get_cli_providers_and_models()
        except Exception:
            logger.debug(
                "Unable to load CLI provider/model registry for Console settings"
            )
            return {}

    async def _providers_models_for_console_settings(
        self,
        provider: str,
        *,
        current_model: str | None = None,
    ) -> dict[str, list[str]]:
        """Return provider/model options including runtime-discovered models."""
        providers_models = self._providers_models()
        provider_key = provider_config_key(provider)
        if not provider_key:
            return providers_models
        try:
            model_options = await resolve_provider_model_options(
                providers_models,
                getattr(
                    self.app_instance,
                    "llm_provider_catalog_scope_service",
                    None,
                ),
                provider=provider_key,
                current_model=current_model,
            )
        except Exception:
            logger.exception(
                "Unable to resolve Console runtime-discovered models for provider=%s model=%s",
                provider_key,
                current_model,
            )
            return providers_models
        merged = {
            provider_name: list(model_ids)
            for provider_name, model_ids in providers_models.items()
        }
        merged[provider_key] = [option.model_id for option in model_options]
        self._remember_console_model_options(provider_key, model_options)
        return merged

    def _remember_console_model_options(
        self,
        provider: str,
        options: list[ResolvedProviderModelOption],
    ) -> None:
        provider_key = provider_config_key(provider)
        self._console_model_option_warnings = {
            key: value
            for key, value in self._console_model_option_warnings.items()
            if key[0] != provider_key
        }
        for option in options:
            model_id = str(option.model_id or "").strip()
            if not model_id or not option.warning:
                continue
            self._console_model_option_warnings[(provider_key, model_id)] = (
                option.warning
            )

    def _console_model_capability_warning(
        self,
        provider: str,
        model: str | None,
    ) -> str:
        model_id = str(model or "").strip()
        if not model_id:
            return ""
        return self._console_model_option_warnings.get(
            (provider_config_key(provider), model_id),
            "",
        )

    def _configured_console_provider(
        self,
        provider: str,
    ) -> tuple[str, list[str]] | None:
        """Resolve a normalized intent against configured provider identities."""
        requested_key = provider_config_key(provider)
        for configured_provider, configured_models in self._providers_models().items():
            if provider_config_key(configured_provider) != requested_key:
                continue
            models = [
                str(model).strip()
                for model in configured_models
                if str(model or "").strip()
                and str(model).strip().lower() not in {"none", "null"}
            ]
            return requested_key, models
        return None

    def _configured_console_provider_default_model(
        self,
        provider: str,
        models: list[str],
    ) -> str | None:
        """Return a valid configured default model for one provider."""
        config = self._provider_readiness_app_config()
        api_settings = (
            config.get("api_settings", {}) if isinstance(config, Mapping) else {}
        )
        provider_settings: Mapping[str, Any] = {}
        if isinstance(api_settings, Mapping):
            for configured_provider, configured_settings in api_settings.items():
                if provider_config_key(str(configured_provider)) != provider:
                    continue
                if isinstance(configured_settings, Mapping):
                    provider_settings = configured_settings
                break
        candidates = (
            provider_settings.get("model"),
            provider_settings.get("api_model"),
            provider_settings.get("default_model"),
        )
        for candidate in candidates:
            model = str(candidate or "").strip()
            if model and model in models:
                return model

        defaults = self._persisted_chat_defaults()
        if provider_config_key(str(defaults.get("provider") or "")) == provider:
            default_model = str(defaults.get("model") or "").strip()
            if default_model and default_model in models:
                return default_model
        return models[0] if models else None

    def _apply_console_provider_intent(
        self,
        intent: ConsoleProviderIntent,
        *,
        store: ConsoleChatStore,
        session_id: str,
        settings: ConsoleSessionSettings,
    ) -> bool:
        """Apply one validated intent to the session captured by its consumer."""
        configured = self._configured_console_provider(intent.provider)
        if configured is None:
            self.app_instance.notify(
                "That provider is unavailable. Choose a configured provider in Settings.",
                severity="warning",
            )
            return False

        provider, models = configured
        model = self._configured_console_provider_default_model(provider, models)
        derived = build_default_console_session_settings(
            self._provider_readiness_app_config(),
            provider,
            model,
        )
        next_settings = replace(
            settings,
            provider=provider,
            model=model,
            base_url=derived.base_url,
            source="user",
        )
        store.replace_session_settings(session_id, next_settings)
        if store.active_session_id == session_id:
            self._set_control_selection(next_settings.provider, next_settings.model)
            self._sync_console_chat_core_state()
            self._sync_console_settings_summary()
            self._sync_console_control_bar()
        self.app_instance.notify(
            f"Console provider set to {provider} for this session.",
            severity="information",
        )
        return True

    def consume_pending_console_provider_intent(self) -> bool:
        """Consume one typed provider intent after the Console session is ready."""
        try:
            store = self._ensure_console_chat_store()
            settings = self._active_session_settings()
            session_id = store.active_session_id
            if session_id is None:
                return False
        except Exception as exc:
            logger.warning(
                "Console provider handoff is not ready (exception_category={})",
                type(exc).__name__,
            )
            return False

        claim = self.app_instance.pending_handoffs.claim(
            HandoffChannel.CONSOLE_PROVIDER
        )
        if claim is None:
            return False
        try:
            if not isinstance(claim.value, ConsoleProviderIntent):
                raise TypeError("Console provider handoff was not typed")
            self._apply_console_provider_intent(
                claim.value,
                store=store,
                session_id=session_id,
                settings=settings,
            )
        except Exception as exc:
            self.app_instance.pending_handoffs.release(claim)
            logger.warning(
                "Console provider handoff will retry "
                "(channel={}, revision={}, exception_category={})",
                claim.channel.value,
                claim.revision,
                type(exc).__name__,
            )
            self.app_instance.notify(
                "Console provider selection could not be applied yet; it will retry.",
                severity="warning",
            )
            return False
        self.app_instance.pending_handoffs.acknowledge(claim)
        return True

    def consume_pending_vllm_console_intent(self) -> bool:
        """Apply one current verified vLLM target to the active session only."""

        store = getattr(self.app_instance, "pending_handoffs", None)
        if type(store) is not PendingHandoffStore:
            return False
        if store.release_recovery(HandoffChannel.VLLM_CONSOLE) is not None:
            recovery_result = store.retry_release_recovery(
                HandoffChannel.VLLM_CONSOLE,
                automatic=False,
            )
            if recovery_result != "released":
                self.app_instance.notify(
                    "vLLM session handoff cleanup is still pending. It will "
                    "retry on the next Console activation.",
                    severity="warning",
                )
                return False
        claim = store.claim(HandoffChannel.VLLM_CONSOLE)
        if claim is None:
            return False
        session_store = None
        session_id = None
        current = None
        next_settings = None
        current_has_user_work = None
        current_controller = None
        current_provider_selection = None
        current_summary_state = None
        current_endpoint_policy = None
        adoption_receipt = None
        replacement_started = False
        try:
            intent = claim.value
            if type(intent) is not VllmConsoleIntent:
                raise TypeError("vLLM Console handoff was not exact")
            owner = getattr(self.app_instance, "_vllm_connection_owner", None)
            if not owner_has_current_intent(owner, intent):
                raise ValueError("vLLM Console handoff is stale")
            if not self.is_attached:
                raise RuntimeError("Console is detached")
            session_store = self._ensure_console_chat_store()
            current = self._active_session_settings()
            session_id = session_store.active_session_id
            if session_id is None:
                raise RuntimeError("Console active session is unavailable")
            active_session = session_store.ensure_session()
            if active_session.id != session_id:
                raise RuntimeError("Console active session changed before adoption")
            current_has_user_work = active_session.has_user_work
            current_endpoint_policy = session_store.session_ephemeral_endpoint_policy(
                session_id
            )
            current_controller = self._console_chat_controller
            current_provider_selection = self._build_console_provider_selection()
            current_summary_state = self._build_console_settings_summary_state()
            configured_vllm = build_target_default_console_session_settings(
                self._provider_readiness_app_config(),
                "vllm",
                intent.model_id,
            )
            next_settings = replace(
                current,
                provider="vllm",
                model=intent.model_id,
                base_url=configured_vllm.base_url,
                source="user",
            )
            endpoint_policy = ConsoleEphemeralEndpointPolicy(
                provider="vllm",
                model=intent.model_id,
                base_url=intent.api_url,
            )
            errors = validate_console_session_settings(
                endpoint_policy.effective_settings(next_settings),
                app_config=self._provider_readiness_app_config(),
            )
            if errors:
                raise ValueError("vLLM Console session settings are invalid")
            adoption_receipt = session_store.adopt_session_ephemeral_endpoint(
                session_id,
                settings=next_settings,
                policy=endpoint_policy,
            )
            replacement_started = True
            self._sync_console_chat_core_state()
            self._sync_console_settings_summary()
            if (
                not self.is_attached
                or session_store.active_session_id != session_id
                or not owner_has_current_intent(owner, intent)
                or not store.acknowledge_current(claim)
            ):
                raise RuntimeError("vLLM Console handoff changed during adoption")
        except BaseException as error:
            if (
                replacement_started
                and session_store is not None
                and session_id is not None
                and next_settings is not None
                and current is not None
                and current_has_user_work is not None
            ):
                try:
                    outcome = (
                        session_store.rollback_session_ephemeral_endpoint_adoption(
                            session_id,
                            expected_settings=next_settings,
                            expected_policy=endpoint_policy,
                            prior_settings=current,
                            prior_policy=current_endpoint_policy,
                            prior_has_user_work=current_has_user_work,
                            receipt=adoption_receipt,
                        )
                    )
                    if outcome is ConsoleEndpointRollbackOutcome.LOST_SESSION_FENCE:
                        raise RuntimeError(
                            "vLLM Console rollback lost its session fence"
                        )
                    if (
                        outcome is ConsoleEndpointRollbackOutcome.RESTORED
                        and session_store.active_session_id == session_id
                    ):
                        try:
                            self._sync_console_chat_core_state()
                        except BaseException:
                            if current_controller is None:
                                if self._console_chat_controller is not None:
                                    raise
                            elif current_provider_selection is None:
                                raise
                            else:
                                current_controller.update_provider_selection(
                                    current_provider_selection
                                )
                        try:
                            self._sync_console_settings_summary()
                        except BaseException:
                            if current_summary_state is None:
                                raise
                            self._apply_console_settings_summary_state(
                                current_summary_state
                            )
                    elif (
                        outcome
                        is ConsoleEndpointRollbackOutcome.BLOCKED_DURABLE_RESTORE
                        and session_store.active_session_id == session_id
                    ):
                        self.app_instance.notify(
                            "vLLM session endpoint blocked because the prior "
                            "conversation metadata could not be restored. Retry "
                            "the handoff or choose a provider before sending.",
                            severity="error",
                        )
                        self._sync_console_chat_core_state()
                        self._sync_console_settings_summary()
                except BaseException as rollback_error:
                    logger.warning(
                        "vLLM Console handoff rollback failed "
                        "(revision={}, exception_category={})",
                        claim.revision,
                        type(rollback_error).__name__,
                    )
                    self.app_instance.notify(
                        "vLLM session handoff could not restore its exact prior "
                        "state. Review the current provider before sending.",
                        severity="error",
                    )
            release_failure = "false"
            try:
                released = store.release(claim) is True
            except BaseException as release_error:
                released = False
                release_failure = "exception"
                logger.warning(
                    "vLLM Console handoff claim release failed "
                    "(revision={}, exception_category={})",
                    claim.revision,
                    type(release_error).__name__,
                )
            if not released:
                try:
                    store.retain_release_recovery(
                        claim,
                        failed_attempts=1,
                        automatic_retry_limit=3,
                        last_failure=release_failure,
                    )
                except BaseException as retention_error:
                    logger.warning(
                        "vLLM Console handoff cleanup ownership transfer failed "
                        "(revision={}, exception_category={})",
                        claim.revision,
                        type(retention_error).__name__,
                    )
                self.app_instance.notify(
                    "vLLM session handoff could not be re-queued yet. Console "
                    "retained cleanup ownership and will retry before adoption.",
                    severity="error",
                )
            if isinstance(
                error,
                (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt, SystemExit),
            ):
                raise
            logger.warning(
                "vLLM Console handoff will retry "
                "(channel={}, revision={}, exception_category={})",
                claim.channel.value,
                claim.revision,
                type(error).__name__,
            )
            return False
        self.app_instance.notify(
            "Using the verified vLLM target for this Console session only.",
            severity="information",
        )
        return True

    def current_console_provider_for_command(self) -> str | None:
        """Return the active session provider without creating a session."""
        settings = self._current_session_settings()
        if settings is None:
            return None
        return str(settings.provider or "").strip() or None

    def _build_console_provider_selection(
        self, session_id: str | None = None
    ) -> ConsoleProviderSelection:
        """Return an owning-session provider selection without switching tabs.

        Served from the per-pass memo inside a `_console_derivation_scope`
        (task-15452): one draft-edit sync built this 7 times for the same
        session.
        """
        memo = self._console_derivation_memo
        memo_key = ("provider_selection", session_id)
        if memo is not None and memo_key in memo:
            return memo[memo_key]
        selection = self._build_console_provider_selection_uncached(session_id)
        if memo is not None:
            memo[memo_key] = selection
        return selection

    def _build_console_provider_selection_uncached(
        self, session_id: str | None = None
    ) -> ConsoleProviderSelection:
        """Derive the provider selection with no memo in front of it."""
        app_config = self._provider_readiness_app_config()
        store = self._ensure_console_chat_store()
        if session_id is None:
            raw_settings = self._active_session_settings()
            target_session_id = store.active_session_id
            selection_settings = (
                store.effective_session_settings(target_session_id)
                if target_session_id is not None
                else raw_settings
            )
            if selection_settings is None:
                selection_settings = raw_settings
        else:
            selection_settings = store.effective_session_settings(session_id)
            if selection_settings is None:
                raise KeyError(f"Unknown Console session: {session_id}")
            target_session_id = session_id
        legacy_model = None
        if session_id is None:
            _legacy_provider, legacy_model = self._effective_console_provider_model()
        elif getattr(selection_settings, "source", "derived") == "user":
            legacy_model = selection_settings.model
        else:
            chat_defaults = self._config_section(app_config, "chat_defaults")
            legacy_model = chat_defaults.get("model")
        return self._build_console_provider_selection_from_settings(
            target_session_id,
            selection_settings,
            legacy_model=legacy_model,
        )

    def _build_console_provider_selection_for_settings(
        self,
        session_id: str,
        settings: ConsoleSessionSettings,
    ) -> ConsoleProviderSelection:
        """Project a validated modal draft without writing session state."""
        app_config = self._provider_readiness_app_config()
        store = self._ensure_console_chat_store()
        store.session_settings_revision(session_id)
        validation_errors = validate_console_session_settings(
            settings,
            app_config=app_config,
        )
        if validation_errors:
            raise ValueError("Console generation test settings are invalid.")
        return self._build_console_provider_selection_from_settings(
            session_id,
            settings,
            legacy_model=settings.model,
        )

    def _build_console_provider_selection_from_settings(
        self,
        target_session_id: str | None,
        selection_settings: ConsoleSessionSettings,
        *,
        legacy_model: object,
    ) -> ConsoleProviderSelection:
        """Build a provider selection from one immutable settings snapshot."""
        app_config = self._provider_readiness_app_config()
        store = self._ensure_console_chat_store()
        provider = provider_config_key(selection_settings.provider) or "llama_cpp"
        explicit_model = (
            str(selection_settings.model).strip()
            if _has_selected_text(selection_settings.model)
            else None
        )
        api_settings = self._config_section(app_config, "api_settings")
        provider_config = self._config_section(api_settings, provider)
        console_config = self._config_section(app_config, "console")
        configured_model_value = (
            provider_config.get("model")
            or provider_config.get("api_model")
            or provider_config.get("default_model")
        )
        configured_model = (
            str(configured_model_value).strip()
            if _has_selected_text(configured_model_value)
            else None
        )
        if not _has_selected_text(legacy_model) and explicit_model == configured_model:
            explicit_model = None

        base_url: str | None = None
        if provider in {"llama_cpp", "local_llamacpp"}:
            fallback_url = (
                os.environ.get("TLDW_CONSOLE_LLAMA_CPP_BASE_URL")
                or console_config.get("llama_cpp_base_url_override")
                or first_configured_endpoint(provider_config)
            )
            override_url = (
                selection_settings.base_url
                if _has_selected_text(selection_settings.base_url)
                else fallback_url
            )
            base_url = self._normalize_llamacpp_base_url(
                str(override_url) if override_url is not None else None
            )
        elif _has_selected_text(selection_settings.base_url):
            base_url = str(selection_settings.base_url).strip()

        current_workspace_context = self._workspace_context()
        if target_session_id is None:
            workspace_context = current_workspace_context
        else:
            workspace_id = store.session_workspace_id(target_session_id)
            workspace_context = (
                current_workspace_context
                if current_workspace_context.active_workspace_id == workspace_id
                else ConsoleWorkspaceContext(active_workspace_id=workspace_id)
            )

        endpoint_policy = (
            store.session_ephemeral_endpoint_policy(target_session_id)
            if target_session_id is not None
            else None
        )
        endpoint_policy_owns_selection = (
            endpoint_policy is not None
            and endpoint_policy.provider == selection_settings.provider
            and endpoint_policy.model == selection_settings.model
        )
        return ConsoleProviderSelection(
            provider=provider,
            base_url=base_url,
            configured_endpoint_fallback_allowed=(not endpoint_policy_owns_selection),
            endpoint_provenance=(
                ConsoleEndpointProvenance.EPHEMERAL_SESSION
                if endpoint_policy_owns_selection
                else ConsoleEndpointProvenance.DURABLE_CONFIGURATION
            ),
            explicit_model=explicit_model,
            configured_model=configured_model,
            temperature=selection_settings.temperature,
            top_p=selection_settings.top_p,
            min_p=selection_settings.min_p,
            top_k=selection_settings.top_k,
            max_tokens=selection_settings.max_tokens,
            seed=selection_settings.seed,
            presence_penalty=selection_settings.presence_penalty,
            frequency_penalty=selection_settings.frequency_penalty,
            reasoning_effort=selection_settings.reasoning_effort,
            reasoning_summary=selection_settings.reasoning_summary,
            verbosity=selection_settings.verbosity,
            thinking_effort=selection_settings.thinking_effort,
            thinking_budget_tokens=selection_settings.thinking_budget_tokens,
            streaming=selection_settings.streaming,
            system_prompt=selection_settings.system_prompt,
            workspace_context=workspace_context,
        )

    def _active_console_provider_model_display(
        self,
    ) -> tuple[str, str | None, ConsoleSessionSettings]:
        """Return provider/model labels backed by active session settings.

        Served from the per-pass memo inside a `_console_derivation_scope`
        (task-15452): the control state and the Workbench state built off it
        each re-derive this leg.
        """
        memo = self._console_derivation_memo
        if memo is not None and "provider_model_display" in memo:
            return memo["provider_model_display"]
        display = self._active_console_provider_model_display_uncached()
        if memo is not None:
            memo["provider_model_display"] = display
        return display

    def _active_console_provider_model_display_uncached(
        self,
    ) -> tuple[str, str | None, ConsoleSessionSettings]:
        """Derive provider/model labels with no memo in front of them."""
        settings = self._active_session_settings()
        selection = self._build_console_provider_selection()
        legacy_provider, _legacy_model = self._effective_console_provider_model()
        provider_display = selection.provider
        is_matching_provider = (
            provider_config_key(str(legacy_provider or "")) == selection.provider
        )
        if is_matching_provider and _has_selected_text(legacy_provider):
            provider_display = str(legacy_provider).strip()
        selected_model = selection.explicit_model or selection.configured_model
        return provider_display, selected_model, settings

    def _active_console_settings_readiness(
        self,
    ) -> tuple[ConsoleSessionSettings, ConsoleSettingsReadiness]:
        """Return effective settings plus Console-native readiness for display/send surfaces.

        Served from the per-pass memo inside a `_console_derivation_scope`
        (task-15452): the blocker copy, the recovery action and the setup
        blocker each re-derive it, and `build_console_settings_readiness`
        is the single most expensive leg of a draft-edit sync.
        """
        memo = self._console_derivation_memo
        if memo is not None and "settings_readiness" in memo:
            return memo["settings_readiness"]
        readiness = self._active_console_settings_readiness_uncached()
        if memo is not None:
            memo["settings_readiness"] = readiness
        return readiness

    def _active_console_settings_readiness_uncached(
        self,
    ) -> tuple[ConsoleSessionSettings, ConsoleSettingsReadiness]:
        """Derive effective settings + readiness with no memo in front."""
        settings = self._active_session_settings()
        selection = self._build_console_provider_selection()
        selected_model = selection.explicit_model or selection.configured_model
        effective_settings = replace(
            settings,
            model=selected_model,
            base_url=selection.base_url,
        )
        readiness = build_console_settings_readiness(
            effective_settings,
            app_config=self._provider_readiness_app_config(),
            active_run=self._console_run_active(),
        )
        model_warning = self._console_model_capability_warning(
            effective_settings.provider,
            selected_model,
        )
        if model_warning and readiness.native_send_supported:
            return effective_settings, replace(
                readiness,
                label="Capabilities unknown",
                detail=f"{readiness.detail}\n{model_warning}",
                native_send_supported=True,
            )
        return effective_settings, readiness
