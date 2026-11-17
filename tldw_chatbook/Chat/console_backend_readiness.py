"""The selected backend's fixed, checked readiness presentation read."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import replace
from inspect import getattr_static
from types import FunctionType, MethodType
from typing import TYPE_CHECKING

from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import config_participants

from . import console_chat_controller as controller_source
from . import console_session_settings as settings_source
from . import console_provider_endpoints as endpoint_source
from .console_display_state import ConsoleControlState
from .console_control_dispatch import require_control_source
from .console_domain_services import _definition, _definition_current
from .console_preparation_reads import preparation_read_for, run_preparation_read
from .provider_test_evidence import ProviderConnectionEvidence


if TYPE_CHECKING:
    from .console_control_dispatch import ControlIdentity
    from .console_execution_backend import ConsoleExecutionBackend
    from .console_session_settings import (
        ConsoleSessionSettings,
        ConsoleSettingsReadiness,
    )


class ConsoleBackendReadiness:
    """Retain one completion-aged read, with actual backend source ownership.

    The one-second age is the existing ConsoleReadinessConfigProjection default.
    Neither the retained value nor its age grants a config/native operation.
    """

    def __init__(self, backend: ConsoleExecutionBackend) -> None:
        self.backend = backend
        self.connection_evidence = ProviderConnectionEvidence()
        self._evidence_owner = self.connection_evidence
        self.reads = set()
        self.pending = False
        self.key = None
        self.value = None
        self.at = 0.0
        self.context_summary = None
        self.context_key = None
        self.agent_history = None
        self.agent_log_available = False
        self.agent_key = None
        self.prompt_history_payload = None
        self.history_key = None

    def _current(self, session, settings, revision, source) -> None:
        backend = self.backend
        backend.require_sources()
        backend.presentation.require_association(backend.presentation.association)
        if _require_context_readers is not _CONTEXT_GUARD[0] or not _definition_current(_CONTEXT_GUARD):
            raise RuntimeError("Console context guard changed")
        _require_context_readers(backend)
        if (
            type(self) is not ConsoleBackendReadiness
            or backend.presentation.readiness is not self
            or not all(
                getattr_static(self, name, None) is record[0]
                and _definition_current(record)
                for name, record in _DEFINITIONS
            )
            or not all(
                getattr_static(owner, name, None) is record[0]
                and _definition_current(record)
                for owner, name, record in _READ_SOURCES
            )
            or type(self.connection_evidence) is not ProviderConnectionEvidence
            or self.connection_evidence is not self._evidence_owner
            or getattr_static(ConsoleControlState, "from_values") is not _CONTROL_LABEL_DESCRIPTOR
            or not _definition_current(_CONTROL_LABEL_BODY)
            or backend.controls.source_revision != revision
            or source.sources is not backend.domain.sources
            or backend.store._sessions.get(session.id) is not session
            or session.settings is not settings
            or session.incarnation_id != source.identity.incarnation_id
            or session.id != source.identity.session_id
        ):
            raise RuntimeError("Console readiness source changed")

    def _key(self, session, settings, revision) -> tuple:
        return (
            config.current_config_identity(),
            id(session),
            id(settings),
            session.settings_revision,
            revision,
            self.connection_evidence.version,
            self.backend.store.session_ephemeral_endpoint_policy(session.id),
        )

    def current_value(
        self,
        identity: ControlIdentity,
    ) -> tuple[ConsoleSessionSettings, ConsoleSettingsReadiness, str, str | None, str]:
        """Consume the completed read at this command's synchronous capture."""
        backend = self.backend
        backend._require_owner_loop()
        source = require_control_source(backend.runtime)
        session = backend.store._sessions.get(identity.session_id)
        if source.identity is not identity or session is None:
            raise RuntimeError("Console readiness command changed")
        settings = session.settings
        self._current(session, settings, identity.source_revision, source)
        if (
            self.pending
            or self.value is None
            or self.key != self._key(session, settings, identity.source_revision)
            or time.monotonic() - self.at >= 1.0
            or self.context_key != self._context_key(session)
            or self.agent_key != self._agent_key(session)
            or self.history_key != self._history_key()
        ):
            raise RuntimeError("Console readiness presentation expired")
        return self.value

    def _history_key(self) -> tuple:
        history = self.backend.domain.sources.prompt_history
        return history.revision, history._loaded, history.persistence_error

    def _agent_key(self, session) -> tuple:
        bridge = self.backend.runtime.agent_bridge
        conversation_id = str(session.persisted_conversation_id or "")
        return (
            bridge, conversation_id, bridge.run_log_target_token(conversation_id),
            bridge._setup_started_at.get(conversation_id),
        )

    def agent_projection(self, session):
        """Build original resident rail values beside the settled history read."""
        from .console_execution_agent_rail_values import (
            ConsoleAgentRailInputs, build_agent_rail_projection,
        )
        from ..DB.AgentRuns_DB import AgentRunsDB

        backend = self.backend
        backend._require_owner_loop()
        if self.agent_key != self._agent_key(session):
            raise RuntimeError("Console Agent history selection changed")
        bridge, conversation_id, _token, _setup = self.agent_key
        live = bridge.live_snapshot(conversation_id)
        handles = tuple(bridge.fleet_snapshot(conversation_id)) if conversation_id else ()
        standard_database = type(bridge._db) is AgentRunsDB and not bridge._db.is_memory_db
        running, pending = backend.controller.fleet_summary_counts()
        wake = backend.controller.fleet_wake
        return build_agent_rail_projection(ConsoleAgentRailInputs(
            conversation_id=conversation_id, live_snapshot=live,
            historical_snapshot=self.agent_history, handles=handles,
            live_run_snapshots=tuple(
                bridge.live_run_snapshot(conversation_id, handle.run_id)
                if handle.run_id else None for handle in handles
            ),
            setup_established=standard_database and _setup is not None,
            primary_run_bound=standard_database and bool(bridge.live_primary_run_id(conversation_id)),
            fleet_running=running, fleet_pending=pending,
            pause_reason=wake.pause_reason(conversation_id) if conversation_id else None,
            wake_has_pending=wake.has_pending(conversation_id) if conversation_id else False,
            has_history_picker=bool(conversation_id),
            full_log_available=self.agent_log_available, drill_run_id=None,
            observed_at=time.monotonic(),
        ))

    def _context_key(self, session) -> tuple:
        backend = self.backend
        store, controller = backend.store, backend.controller
        snapshot = backend.runtime._execution_service._display_owner.context
        return (
            store.active_session_id, id(store.workspace_context),
            id(store.dispatch_recovery_for_session(session.id)),
            id(store.preparation_for_session(session.id)),
            store.session_thinking_history_policy(session.id),
            store.payload_revision(session.id),
            store.display_projection_revision(session.id), session.draft_revision,
            session.attachment_revision, session.one_shot_prefill_revision,
            controller.run_state_for(session.id).status,
            bool(controller._submit_tasks_for_session(session.id)),
            snapshot._cache_key(controller, session.id), id(snapshot.value),
            backend.runtime.snapshot_console_staged_evidence(session.id)[1],
        )

    def _capture_context(self, session) -> dict:
        """Detach original resident inputs before the finite preparation await."""
        from .console_display_state import ConsoleStagedContextState
        from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

        backend = self.backend
        store, controller = backend.store, backend.controller
        launch, _revision, _notice = backend.runtime.snapshot_console_staged_evidence(session.id)
        if launch is not None:
            raise RuntimeError("Console staged evidence requires its process presentation")
        messages = tuple(store.read_only_messages_for_session(session.id))
        history = spend.build_console_spend_history_projection(
            messages, store.dispatch_recovery_for_session(session.id),
            store.preparation_for_session(session.id),
            controller.run_state_for(session.id).status,
            bool(controller._submit_tasks_for_session(session.id)),
        )
        snapshot = backend.runtime._execution_service._display_owner.context
        from .console_context_policy import ConsoleContextPolicyOverrides

        try:
            overrides, global_overrides, memory = snapshot.inputs(controller, session.id)
        except (KeyError, ValueError):
            overrides, global_overrides, memory = ConsoleContextPolicyOverrides(), None, None
        try:
            accounting = controller.context_breakdown_accounting(session.id)
        except Exception:
            accounting = None  # Original optional accounting presentation policy.
        staged = ConsoleStagedContextState.empty()
        return dict(
            settings=session.settings, messages=messages,
            request_ids=frozenset(history.request_ids) if history.request_ids is not None else None,
            active_session=store.active_session_id == session.id,
            seeded_greeting=controller._seeded_greeting_text(session.id, list(messages)),
            draft_text=session.draft,
            staged_source_count=len(store.workspace_context.staged_sources),
            staged_context_summary=staged.summary, staged_text="",
            overrides=overrides, global_overrides=global_overrides, memory=memory,
            accounting=accounting,
            thinking_history_policy=store.session_thinking_history_policy(session.id),
            thinking_history_effective_policy=None,
            agent_runtime_enabled=controller._agent_runtime_enabled,
            agent_bridge_present=controller._agent_bridge is not None,
            has_submit_prefill=bool(controller._resolve_submit_prefill(session.id)[0]),
            has_pending_attachments=bool(store.pending_attachments(session.id)),
        )

    async def warm(
        self,
        identity: ControlIdentity,
    ) -> tuple[ConsoleSessionSettings, ConsoleSettingsReadiness, str, str | None, str]:
        """Return actual settings/readiness after the exact read has retired."""
        backend = self.backend
        backend._require_owner_loop()
        source = require_control_source(backend.runtime)
        if source.identity is not identity or self.pending:
            raise RuntimeError("Console readiness command changed")
        with backend.store._preparation_lock:
            session = backend.store._sessions.get(identity.session_id)
            if session is None or session.settings is None:
                raise RuntimeError("Console readiness session is unavailable")
            settings = session.settings
            key = self._key(session, settings, identity.source_revision)
        self._current(session, settings, identity.source_revision, source)
        if (
            self.key == key
            and self.value is not None
            and time.monotonic() - self.at < 1.0
            and self.context_key == self._context_key(session)
            and self.agent_key == self._agent_key(session)
            and self.history_key == self._history_key()
            and backend.domain.sources.prompt_history._loaded
        ):
            return self.value
        with backend.store._preparation_lock:
            agent_key = self._agent_key(session)
            current_context_key = self._context_key(session)
            context_inputs = self._capture_context(session)
            _context_owners = (
                backend.store.workspace_context,
                backend.store.dispatch_recovery_for_session(session.id),
                backend.store.preparation_for_session(session.id),
                backend.runtime._execution_service._display_owner.context.value,
            )
            if current_context_key != self._context_key(session):
                raise RuntimeError("Console context capture changed")
        effect = backend.controls._begin_effect("display_readiness", self, source)
        observations = []
        self.pending = True

        def current() -> None:
            self._current(session, settings, identity.source_revision, source)
            if self._context_key(session) != current_context_key:
                raise RuntimeError("Console context inputs changed")
            if self._agent_key(session) != agent_key:
                raise RuntimeError("Console Agent selection changed")

        def read():
            from .console_execution_retirement import _ReadResources

            handoff = preparation_read_for(self)
            if (
                handoff is None
                or handoff.creator is not self
                or handoff.task is not task
            ):
                raise RuntimeError("Console readiness handoff changed")
            observation = _ReadResources(None, handoff)
            observations.append(observation)
            from tldw_chatbook.UI.Console_Modules import tokenizer_display
            from tldw_chatbook.model_capabilities import get_model_capabilities
            from .console_prepared_request import positive_token_cap

            with observation.observe():
                bridge, conversation_id, _agent_token, _agent_setup = agent_key
                # The stock historical reader owns its actual AgentRunsDB
                # connection. Never substitute an idle snapshot for this read.
                current()
                history = bridge.historical_snapshot(conversation_id)
                log_available = False
                if conversation_id:
                    from ..UI.Console_Modules.run_log_probe_scope import finite_stock_probe

                    def log_probe():
                        run_id = bridge.resolve_run_log_target(conversation_id, None)
                        return bool(run_id and bridge.run_log_available(run_id))

                    try:
                        log_available = finite_stock_probe(bridge, log_probe)
                    except Exception:
                        # Original optional availability probe fails closed;
                        # actual resource settlement is still checked below.
                        log_available = False
                # The real metadata/tokenizer operations precede the final
                # checked snapshot; its generation must still be the captured one.
                current()
                preparation = (
                    tokenizer_display.capture_tokenizer_preparation(settings.model, settings.provider)
                    if settings.model else None
                )
                tokenizer_ready = preparation is None or preparation()
                window = (
                    backend.runtime.provider_gateway.cached_context_window(settings)
                    if tokenizer_ready and settings.model else None
                )
                caps = None
                if not settings.provider.startswith("custom-ep:"):
                    try:
                        capabilities = get_model_capabilities().get_model_capabilities(
                            settings.provider, settings.model or ""
                        )
                        caps = (
                            positive_token_cap(capabilities, "max_input_tokens", "input_token_limit", "provider_input_cap"),
                            positive_token_cap(capabilities, "max_output_tokens", "output_token_limit", "provider_output_cap"),
                        )
                    except Exception:
                        pass  # Original optional local-capability display policy.
                with config_participants.operation(config) as active:
                    before = config_participants.checked_config_identity(config, active)
                    loaded = copy.deepcopy(config.load_settings())
                    after = config_participants.checked_config_identity(config, active)
                    if before != key[0] or after != before:
                        raise RuntimeError("Console readiness config changed")
                current()
                if preparation is not None and not tokenizer_display.preparation_current(preparation):
                    raise RuntimeError("Console tokenizer preparation changed")
                return loaded, tokenizer_ready, window, caps, history, log_available

        task = asyncio.current_task()
        history_retired = False
        try:
            from ..Backup_Recovery.async_file_participants import FileJob

            history_owner = backend.domain.sources.prompt_history
            history_error = None
            try:
                await history_owner.load()
            except BaseException as error:
                history_error = error
            job = history_owner._load_job
            producer = job._producer if type(job) is FileJob else None
            if producer is not None:
                while not producer.done():
                    try:
                        await asyncio.shield(producer)
                    except asyncio.CancelledError as error:
                        if history_error is None:
                            history_error = error
                    except BaseException as error:
                        if history_error is None:
                            history_error = error
                        break
            if _require_context_readers is not _CONTEXT_GUARD[0] or not _definition_current(_CONTEXT_GUARD):
                raise RuntimeError("Console history receipt guard changed")
            _require_context_readers(backend)
            history_retired = (
                type(job) is FileJob and job._source is history_owner
                and job._route == "prompt_history" and job.native_retired
                and (
                    not job._entered
                    or producer is not None and producer.done() and not producer.cancelled()
                )
            )
            if history_error is not None:
                raise history_error
            current()
            if not history_retired:
                raise RuntimeError("Console prompt history retirement is unproven")
            import json

            history_payload = {
                "revision": history_owner.revision,
                "loaded": history_owner._loaded,
                "error": history_owner.persistence_error,
                "entries": [
                    {"input": item["input"], "timestamp": json.dumps(item["timestamp"])}
                    for item in history_owner._entries
                ],
            }
            history_key = self._history_key()
            loaded, tokenizer_ready, window, caps, history, log_available = await run_preparation_read(
                read,
                creator=self,
                session_id=session.id,
                reads=self.reads,
                observers=(backend.runtime._preparation_reads,),
                require_current=current,
                submission_slot=backend.runtime._display_submission_slot,
            )
            at = time.monotonic()
            current()
            if self._key(session, settings, identity.source_revision) != key:
                raise RuntimeError("Console readiness publication changed")
            if not all(item.confirmed() for item in observations):
                raise RuntimeError("Console readiness native retirement is unproven")
            # Original history, metadata and checked configuration reads have
            # retired. Derive presentation from captured and resident values.
            with backend.store._preparation_lock:
                current()
                if self._key(session, settings, identity.source_revision) != key:
                    raise RuntimeError("Console readiness derivation changed")
                selection = controller_source.resolve_console_selection_core(
                    settings,
                    app_config=loaded,
                )
                model = selection.explicit_model or selection.configured_model
                effective = replace(settings, model=model, base_url=selection.base_url)
                endpoint = key[6]
                owns_endpoint = (
                    endpoint is not None
                    and endpoint.provider == settings.provider
                    and endpoint.model == settings.model
                )
                if owns_endpoint and selection.base_url:
                    loaded = endpoint_source.with_provider_endpoint(
                        loaded,
                        config.normalize_provider_config_key(settings.provider),
                        selection.base_url,
                    )
                readiness = settings_source.build_console_settings_readiness(
                    effective,
                    app_config=loaded,
                    connection_evidence=self.connection_evidence,
                )
                current()
                if self._key(session, settings, identity.source_revision) != key:
                    raise RuntimeError("Console readiness derivation changed")
                provider_chip = ConsoleControlState.from_values(
                    provider=selection.provider, app_config=loaded,
                ).provider_label
                from .console_backend_context_summary import (
                    ConsoleBackendContextInputs, build_backend_context_summary,
                )

                summary = build_backend_context_summary(ConsoleBackendContextInputs(
                    **context_inputs, summary_settings=effective, readiness=readiness,
                    context_window=window, tokenizer_ready=tokenizer_ready,
                    provider_token_caps=caps,
                ))
                current()
                value = effective, readiness, selection.provider, model, provider_chip
                if history_key != self._history_key():
                    raise RuntimeError("Console prompt history publication changed")
                self.context_key, self.context_summary = current_context_key, summary
                self.agent_key, self.agent_history = agent_key, history
                self.agent_log_available = log_available
                self.history_key, self.prompt_history_payload = history_key, history_payload
                self.key, self.value, self.at = key, value, at
            return value
        finally:
            self.pending = False
            # Keep captured owners alive through all reads, then release them.
            _context_owners = None
            # No callback entry is a separate, affirmative no-resource outcome.
            # An entered callback must prove all actually observed acquisitions.
            backend.controls._finish_effect(
                effect, retired=history_retired and all(item.confirmed() for item in observations)
            )


_DEFINITIONS = tuple(
    (name, _definition(value))
    for name, value in vars(ConsoleBackendReadiness).items()
    if type(value) is FunctionType
)
_READ_SOURCES = tuple(
    (owner, name, _definition(getattr(owner, name)))
    for owner, names in (
        (config, ("load_settings", "current_config_identity")),
        (config_participants, ("operation", "checked_config_identity")),
        (controller_source, ("resolve_console_selection_core",)),
        (settings_source, ("build_console_settings_readiness",)),
        (endpoint_source, ("with_provider_endpoint",)),
        (ProviderConnectionEvidence, ("evidence_for", "publish")),
    )
    for name in names
)

_CONTROL_LABEL_DESCRIPTOR = getattr_static(ConsoleControlState, "from_values")
_CONTROL_LABEL_BODY = _definition(_CONTROL_LABEL_DESCRIPTOR.__func__)


from .console_chat_controller import ConsoleChatController
from .console_chat_store import ConsoleChatStore
from .console_provider_gateway import ConsoleProviderGateway
from .console_domain_services import ConsoleDomainServices
from .console_agent_bridge import ConsoleAgentBridge
from .console_fleet_wake import ConsoleFleetWakeCoordinator
from .prompt_history import PromptHistory
from ..Backup_Recovery.async_file_participants import FileJob
from ..Backup_Recovery import raw_participants as _history_raw
from tldw_chatbook.UI.Console_Modules import console_spend_projection as _context_spend
from tldw_chatbook.UI.Console_Modules import tokenizer_display as _tokenizer_display
from tldw_chatbook.UI.Console_Modules import run_log_probe_scope as _run_log_probe
from . import console_backend_context_summary as _context_summary
from . import console_execution_agent_rail_values as _agent_values

_CONTEXT_METHODS = tuple(
    (owner, name, _definition(getattr(owner, name)))
    for owner, names in (
        (ConsoleChatController, ("run_state_for", "_submit_tasks_for_session", "_seeded_greeting_text", "_context_content_for", "context_breakdown_accounting", "_resolve_submit_prefill", "_pinned_prefill_for_session", "fleet_summary_counts")),
        (ConsoleAgentBridge, ("run_log_target_token", "historical_snapshot", "_derive_historical_snapshot", "resolve_run_log_target", "run_log_available", "live_snapshot", "fleet_snapshot", "live_run_snapshot", "live_primary_run_id", "subagent_history_page")),
        (ConsoleFleetWakeCoordinator, ("pause_reason", "has_pending")),
        (PromptHistory, ("load", "_load_locked", "_file_job", "_history_io", "entry_texts")),
        (ConsoleChatStore, ("payload_revision", "display_projection_revision", "read_only_messages_for_session", "dispatch_recovery_for_session", "preparation_for_session", "session_thinking_history_policy", "pending_attachments")),
        (ConsoleProviderGateway, ("cached_context_window",)),
        (ConsoleDomainServices, ("provider_config",)),
        (_context_spend.ConsoleContextReadSnapshot, ("inputs", "_cache_key", "_key", "_closed")),
    ) for name in names
)
_CONTEXT_FUNCTIONS = tuple(
    (owner, name, _definition(getattr(owner, name)))
    for owner, names in (
        (_tokenizer_display, ("capture_tokenizer_preparation", "preparation_current")),
        (_context_spend, ("build_console_spend_history_projection",)),
        (_context_summary, ("build_backend_context_summary", "_settings_estimate", "_context_state")),
        (_agent_values, ("build_agent_rail_projection", "_fleet_rows", "_historical")),
        (_run_log_probe, ("finite_stock_probe", "_stock_default_probe")),
        (_history_raw, ("scope", "_retire")),
    ) for name in names
)


def _require_context_readers(backend) -> None:
    gateway = backend.runtime.provider_gateway
    snapshot = backend.runtime._execution_service._display_owner.context
    receivers = {
        ConsoleChatController: backend.controller, ConsoleChatStore: backend.store,
        ConsoleProviderGateway: gateway, ConsoleDomainServices: backend.domain,
        ConsoleAgentBridge: backend.runtime.agent_bridge,
        ConsoleFleetWakeCoordinator: backend.controller.fleet_wake,
        PromptHistory: backend.domain.sources.prompt_history,
        _context_spend.ConsoleContextReadSnapshot: snapshot,
    }
    from ..DB.AgentRuns_DB import AgentRunsDB

    if (
        backend.controller._agent_bridge is not backend.runtime.agent_bridge
        or backend.runtime.agent_bridge._db is not backend.runtime._agent_runs_db
        or type(backend.runtime._agent_runs_db) is not AgentRunsDB
        or backend.runtime._agent_runs_db.is_memory_db
    ):
        raise RuntimeError("Console Agent source changed")
    history = backend.domain.sources.prompt_history
    if backend.runtime._prompt_history is not history or backend.controller.prompt_history is not history:
        raise RuntimeError("Console prompt history source changed")
    if (
        getattr_static(FileJob, "native_retired", None) is not _HISTORY_RETIREMENT_DESCRIPTOR
        or not _definition_current(_HISTORY_RETIREMENT_BODY)
        or _history_raw._State is not _HISTORY_STATE_TYPE
        or getattr_static(PromptHistory, "__init__") is not _HISTORY_INIT[0]
        or not _definition_current(_HISTORY_INIT)
        or any(
            getattr_static(FileJob, name, None) is not record[0]
            or not _definition_current(record)
            for name, record in _HISTORY_FILE_METHODS
        )
    ):
        raise RuntimeError("Console prompt history retirement source changed")
    for owner, name, record in _CONTEXT_METHODS:
        receiver = receivers[owner]
        method = getattr(receiver, name, None)
        if (
            type(receiver) is not owner or type(method) is not MethodType
            or method.__self__ is not receiver or method.__func__ is not record[0]
            or getattr_static(owner, name, None) is not record[0]
            or not _definition_current(record)
        ):
            raise RuntimeError("Console context reader changed")
    provider = gateway._config_provider
    if type(provider) is not MethodType or provider.__self__ is not backend.domain or provider.__func__ is not ConsoleDomainServices.provider_config:
        raise RuntimeError("Console context config owner changed")
    for owner, name, record in _CONTEXT_FUNCTIONS:
        if getattr_static(owner, name, None) is not record[0] or not _definition_current(record):
            raise RuntimeError("Console context calculation source changed")

_HISTORY_FILE_METHODS = tuple(
    (name, _definition(getattr(FileJob, name)))
    for name in ("__init__", "__exit__", "run", "_work", "_check_creator", "_note_history_scope", "_complete", "_executor_finished")
)
_HISTORY_STATE_TYPE = _history_raw._State
_HISTORY_INIT = _definition(PromptHistory.__init__)
_HISTORY_RETIREMENT_DESCRIPTOR = getattr_static(FileJob, "native_retired")
_HISTORY_RETIREMENT_BODY = _definition(_HISTORY_RETIREMENT_DESCRIPTOR.fget)
_CONTEXT_GUARD = _definition(_require_context_readers)
