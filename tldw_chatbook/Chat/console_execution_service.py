"""In-process construction of the real Console runtime's object graph.

The runtime keeps the authoritative objects and lifecycle. This concrete owner
only assembles its store, bridge and controller with the original receivers.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .console_domain_display import ConsoleDomainDisplayOwner
    from .console_chat_controller import ConsoleChatController
    from .console_chat_store import ConsoleChatStore
    from .console_runtime import ConsoleRuntime


class ConsoleExecutionService:
    """Assemble one real runtime without owning a second graph or App facade."""

    def __init__(
        self,
        runtime: ConsoleRuntime,
        *,
        runtime_source: ModuleType,
        domain_services: Any | None = None,
    ) -> None:
        """Retain the actual runtime whose authoritative fields are populated."""
        self._runtime = runtime
        self._runtime_source = runtime_source
        self.domain_services = domain_services
        self._permission_operations = None
        self._local_controls = None
        self._display_owner = None

    def open_local_controls(self, *, profile_epoch: str, service_epoch: str):
        """Select local controls only after the original controller/host exist."""
        from .console_control_dispatch import ConsoleLocalControls

        if self._local_controls is None:
            self._local_controls = ConsoleLocalControls(
                self, profile_epoch=profile_epoch, service_epoch=service_epoch
            )
        elif (
            self._local_controls.profile_epoch != profile_epoch
            or self._local_controls.service_epoch != service_epoch
        ):
            raise RuntimeError(
                "A Console service cannot adopt successor control epochs"
            )
        self._local_controls._require_graph()
        self.prepare_display_owner().bind_original_graph()
        return self._local_controls

    def ensure_permission_operations(self):
        """Expose complete actions on the explicit domain's existing singleton."""
        domain = self.domain_services
        if domain is None:
            raise TypeError(
                "Legacy Console construction retains its original permission route"
            )
        domain.require_current()
        if self._permission_operations is None:
            from .console_permission_operations import ConsolePermissionOperations

            self._permission_operations = ConsolePermissionOperations(
                domain.sources.control_plane,
                runtime=self._runtime,
                tool_packs=domain.sources.tool_pack_service,
            )
        self._permission_operations._require_current()
        return self._permission_operations

    def ensure_chat_store(
        self,
        *,
        workspace_context: Any | None = None,
        on_scope_flushed: Callable[..., Any] | None = None,
    ) -> "ConsoleChatStore":
        """Return the Console chat store, creating it lazily.

        Moved verbatim from `ChatScreen._ensure_console_chat_store`: the
        durable `ChatPersistenceService` is attached only when the app has
        a ChaChaNotes DB, and the citation repository is dropped when it
        belongs to a different DB than the one being persisted to.

        Args:
            workspace_context: The view's current
                `ConsoleWorkspaceContext`. Read at construction only.
            on_scope_flushed: The view's session-scope-flushed callback.
                Read at construction only.

        Returns:
            ConsoleChatStore: The runtime's store.
        """
        runtime = self._runtime
        if runtime._chat_store is not None or runtime._disposed:
            return runtime._chat_store
        runtime_module = self._runtime_source

        from tldw_chatbook.Chat.chat_persistence_service import (
            ChatPersistenceService,
        )
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
        from tldw_chatbook.Chat.console_library_policy_coordinator import (
            ConsoleLibraryPolicyCoordinator,
        )
        from tldw_chatbook.Chat.console_trace_projection import ConsoleTraceProjection

        persistence = None
        domain = self.domain_services
        if domain is not None:
            domain.require_current()
        db = (
            domain.sources.chat_database
            if domain is not None
            else getattr(runtime._app, "chachanotes_db", None)
        )
        if db is not None:
            try:
                runtime_module.recover_console_trace_calls(db)
            except Exception as exc:
                runtime_module.logger.warning(  # noqa: PLE1205 -- original source owns a Loguru logger
                    "console_trace_recovery_failed: {}", type(exc).__name__
                )
            citation_repository = (
                domain.sources.citation_repository
                if domain is not None
                else getattr(runtime._app, "citation_trace_repository", None)
            )
            if (
                citation_repository is not None
                and getattr(citation_repository, "db", None) is not db
            ):
                citation_repository = None
            persistence = ChatPersistenceService(
                db,
                workspace_registry=(
                    domain.sources.workspace_registry
                    if domain is not None
                    else getattr(runtime._app, "workspace_registry_service", None)
                ),
                citation_repository=citation_repository,
            )
            persistence.retry_recovered_media_references()
            legacy_normalization_enabled = callable(getattr(db, "transaction", None))
            legacy_normalizer: Any | None = None
            native_reader: Any | None = None

            def get_legacy_normalizer() -> Any:
                """Build the legacy adapter only after first paint or first use."""

                nonlocal legacy_normalizer
                if legacy_normalizer is None:
                    from tldw_chatbook.Chat.console_trace_legacy import (
                        LegacyTraceNormalizer,
                    )

                    legacy_normalizer = LegacyTraceNormalizer(db)
                return legacy_normalizer

            def get_native_reader() -> Any:
                """Build the native ledger reader only on first trace inspection."""

                nonlocal native_reader
                if native_reader is None:
                    from tldw_chatbook.Chat.console_trace_native_reader import (
                        ConsoleTraceNativeReader,
                    )

                    native_reader = ConsoleTraceNativeReader(
                        db,
                        repository=persistence.console_trace_repository,
                    )
                return native_reader

            def read_normalized_calls(message_id: str) -> Any:
                """Read native calls first, followed by migrated legacy snapshots."""

                return (
                    *get_native_reader().read_calls(message_id),
                    *get_legacy_normalizer().read_calls(message_id),
                )
        else:
            legacy_normalization_enabled = False
        from tldw_chatbook.Chat.console_canvas_controller import (
            ConsoleCanvasController,
        )

        snapshot = runtime._ensure_canvas_profile_snapshot()
        durable_canvas_service = None
        if db is not None:
            from tldw_chatbook.Canvas.service import CanvasService
            from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

            if isinstance(db, CharactersRAGDB):
                durable_canvas_service = CanvasService(db, profile_snapshot=snapshot)
        runtime._canvas_controller = ConsoleCanvasController(
            durable_service=durable_canvas_service, profile_snapshot=snapshot
        )
        runtime.set_chat_store(
            ConsoleChatStore(
                persistence=persistence,
                assistant_defaults_provider=runtime._resolve_new_console_assistant,
                on_assistant_default_notice=(
                    domain.presentation.notice
                    if domain is not None
                    else lambda notice: runtime._app.notify(notice, severity="warning")
                ),
                settle_provider_traces_off_thread=True,
                trace_projection=(
                    ConsoleTraceProjection(
                        legacy_reader=db.get_message_exchanges,
                        normalized_reader=(
                            read_normalized_calls
                            if legacy_normalization_enabled
                            else None
                        ),
                        normalized_reads_enabled=lambda: (
                            runtime_module.runtime_capture_policy().normalized_reads_enabled
                        ),
                        normalized_writes_enabled=lambda: (
                            runtime_module.runtime_capture_policy().normalized_writes_enabled
                        ),
                        compatibility_metrics=runtime.trace_compatibility_metrics,
                    )
                    if db is not None
                    else None
                ),
                workspace_context=workspace_context,
                on_scope_flushed=on_scope_flushed,
                library_policy_coordinator=(
                    ConsoleLibraryPolicyCoordinator(
                        persistence.console_library_policy_repository
                    )
                    if persistence is not None
                    else None
                ),
                library_policy_defaults_provider=(
                    domain.library_policy_defaults
                    if domain is not None
                    else lambda: runtime_module._current_library_policy_defaults(
                        runtime._app
                    )
                ),
                thinking_history_policy_default_provider=(
                    domain.thinking_history_policy_default
                    if domain is not None
                    else lambda: runtime_module._current_thinking_history_policy_default(
                        runtime._app
                    )
                ),
                canvas_promotion_participant=runtime._canvas_controller,
                canvas_turn_controller=runtime._canvas_controller,
                on_canvas_context_changed=runtime._sync_canvas_native_context,
            )
        )
        if db is not None and legacy_normalization_enabled:
            runtime._schedule_legacy_trace_maintenance(db, get_legacy_normalizer)
        runtime._bind_view_hooks()
        return runtime._chat_store

    def ensure_agent_bridge(
        self,
        *,
        store_factory: Callable[[], Any],
        provider_gateway_factory: Callable[[], Any],
        skills_service: Any | None = None,
        native_tools_enabled_factory: Callable[[], Any] | None = None,
    ) -> Any:
        """Return the Console agent bridge, creating it lazily.

        Moved verbatim from
        `ConsoleAgentController._ensure_console_agent_bridge`, ordering
        included: the durable-DB probe runs FIRST and returns `None` (no
        agent runtime) before the store or the gateway is touched, so an
        in-memory harness still builds neither.

        Factories remain in this compatibility signature to preserve the
        durable-DB probe ordering. The bridge's stored native-tools gate is
        always app-owned and never retains a view callback.

        Args:
            store_factory: Returns the chat store the bridge should use.
                Called only past the durable-DB probe.
            provider_gateway_factory: Returns the provider gateway. Same.
            skills_service: The app's skills scope service, or `None`. A
                plain value: it is a `getattr` on the APP, which the probe
                has already touched.
            native_tools_enabled_factory: Compatibility-only legacy seam;
                the bridge stores the app-owned native-tools gate.

        Returns:
            Any: The `ConsoleAgentBridge`, or `None` when there is no
            durable ChaChaNotes DB to key the sibling `AgentRunsDB` off.
        """
        runtime = self._runtime
        if runtime._agent_bridge is not None or runtime._disposed:
            return runtime._agent_bridge
        if runtime.ensure_activity_receipt_service() is None or runtime._disposed:
            return None
        runtime_module = self._runtime_source

        from tldw_chatbook.Agents.plain_chat_tool_surface import (
            AppToolKillSwitch,
            ControlPlaneToolKillSwitch,
            light_surface_configured,
        )
        from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

        domain = self.domain_services
        if domain is not None:
            domain.require_current()
        runs_db = runtime._agent_runs_db
        # TASK-1971 (Agent Change Review): the tracker is None when git is
        # absent -- the bridge then skips tracking entirely, and runs behave
        # exactly as before the feature existed (spec gating decision).
        from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

        change_tracker = ChangeTurnTracker()
        change_coordinator = None
        change_native = None
        if change_tracker.available:
            from tldw_chatbook.Workspaces.change_review_finalization import (
                ChangeReviewFinalizationCoordinator,
            )
            from tldw_chatbook.Workspaces.change_tracking import (
                NativeUndoIssuer,
                _NativeSnapshotIssuer,
            )

            from tldw_chatbook.Utils.platform_files import (
                supports_native_process_groups,
            )

            if supports_native_process_groups():
                change_native = _NativeSnapshotIssuer(runtime)
                runtime._change_revert_native = NativeUndoIssuer(runtime)

            def _publish_change_review(item: Any) -> None:
                from tldw_chatbook.Workspaces.change_review_ownership import (
                    native_snapshot_ticket,
                )

                ticket = native_snapshot_ticket()
                wave = (
                    getattr(ticket.source, "workflow_wave", None)
                    if ticket is not None
                    else None
                )
                if wave is not None:
                    wave.publish(item)
                    return
                runs_db.record_change_snapshots_batch(
                    run_id=item.run_id,
                    records=[record.__dict__ for record in item.records],
                    kind=item.kind,
                )

            change_coordinator = ChangeReviewFinalizationCoordinator(
                tracker=change_tracker,
                publish=_publish_change_review,
                close_publisher=runs_db.close,
                native_owner=change_native,
            )
            runtime._change_review_coordinator = change_coordinator
        runtime._agent_bridge = ConsoleAgentBridge(
            agent_runs_db=runs_db,
            runtime_capacity_factory=runtime._get_execution_capacity,
            store=store_factory(),
            provider_gateway=provider_gateway_factory(),
            skills_service=skills_service,
            native_tools_enabled=(
                domain.native_tools_enabled
                if domain is not None
                else runtime_module.functools.partial(
                    runtime_module._native_tools_enabled_for_app, runtime._app
                )
            ),
            change_tracker=change_tracker if change_tracker.available else None,
            buddy_sink=runtime.persona_buddy_sink,
            change_finalization_coordinator=change_coordinator,
            change_review_native=change_native,
            change_revert_native=runtime._change_revert_native,
            # run-hooks (Task 5): hand the bridge this runtime's engine
            # accessor so per-turn fire sites (PostToolUse today; PreToolUse
            # and the settle events in later tasks) resolve the app-owned
            # singleton through the same runtime a mounted Console uses --
            # never a view, so headless wake runs reach it identically.
            ensure_run_hooks=runtime.ensure_run_hooks,
            get_hooks_v2=runtime.get_hooks_v2,
            # TASK-34100.5.3: plain chat's light tool surface unless the user
            # turned on Full agent tools; read per send, so a switch flipped
            # in setup or the MCP hub applies on the next send.
            light_tool_surface=light_surface_configured,
            tool_kill_switch=(
                ControlPlaneToolKillSwitch(domain.sources.control_plane)
                if domain is not None
                else AppToolKillSwitch(runtime._app)
            ),
            storage_provider=(
                domain.storage
                if domain is not None
                else lambda: runtime._app.services.storage
            ),  # ADR-250 s6
        )
        # PR3a-2 Task 4: the survivor-completion attention consumer (durable
        # unseen mark + app-wide toast + deep link), registered NEXT TO
        # bridge construction per `FleetDrainFanout.register`'s contract.
        # Captures the APP object only -- never a screen -- because the
        # bridge (and its registered consumers) outlives the screen whenever
        # a survivor is still running at teardown.
        from tldw_chatbook.Chat.console_fleet_attention import (
            register_fleet_attention,
        )

        if domain is not None:
            domain.presentation.register_fleet_attention(
                runtime._agent_bridge, runtime._activity_receipts
            )
        else:
            register_fleet_attention(
                runtime._agent_bridge,
                runtime._app,
                receipt_service=runtime._activity_receipts,
            )
        return runtime._agent_bridge

    def ensure_chat_controller(self, **kwargs: Any) -> "ConsoleChatController":
        """Return the Console chat controller, creating it lazily.

        Selection values are passed through from the custody-producing view.
        Every callable needed after custody is replaced here by an app-owned
        service or runtime-owned frozen-input accessor; only disposable UI
        projections are rebound on attach.

        Returns:
            ConsoleChatController: The runtime's controller.
        """
        runtime = self._runtime
        if runtime._chat_controller is not None or runtime._disposed:
            return runtime._chat_controller
        runtime_module = self._runtime_source

        from tldw_chatbook.Chat.console_chat_controller import (
            ConsoleChatController,
        )

        # Run hooks (spec 2026-09-11, Task 7): hand the controller this
        # runtime's engine accessor exactly as `ensure_agent_bridge` hands
        # it to the bridge (Task 5) -- the production path always wires it,
        # while the optional param's `None` default keeps every direct
        # (controller-only) construction, tests included, unchanged.
        kwargs.setdefault("ensure_run_hooks", runtime.ensure_run_hooks)
        kwargs.setdefault("hook_permissions_accessor", runtime.ensure_hook_permissions)
        domain = self.domain_services
        if domain is not None:
            domain.require_current()
            kwargs["domain_services"] = domain
        runtime_owned = {
            "chat_dictionary_applier": domain.apply_chat_dictionaries
            if domain is not None
            else runtime_module.functools.partial(
                runtime_module._apply_chat_dictionaries_for_app, runtime._app
            ),
            "world_info_applier": domain.apply_world_info
            if domain is not None
            else runtime_module.functools.partial(
                runtime_module._apply_world_info_for_app, runtime._app
            ),
            # The LIVE seam (two arguments). Every turn this runtime admits
            # carries its frozen capture and release explicitly
            # (``staged_evidence_capture`` / ``staged_evidence_release``);
            # the controller no longer looks for them by name (TASK-34352).
            "rag_capture_provider": runtime._capture_console_staged_rag,
            # TASK-33620.8 (AC#2): Automatic evidence, captured the same way.
            "automatic_evidence_capture": runtime._capture_console_prepared_rag,
            "staged_evidence_provider": runtime._has_staged_evidence,
            "default_session_settings": domain.default_session_settings
            if domain is not None
            else runtime_module.functools.partial(
                runtime_module._default_session_settings_for_app,
                runtime._app,
            ),
            "library_provider_factory": runtime._library_provider_factory,
            "global_user_display_name": domain.global_user_display_name
            if domain is not None
            else runtime_module.functools.partial(
                runtime_module._global_user_display_name_for_app, runtime._app
            ),
            "turn_context_provider": None,
            "provider_config": domain.provider_config
            if domain is not None
            else runtime_module.functools.partial(
                runtime_module._provider_config_for_app, runtime._app
            ),
            "confirm_project_instruction_dispatch": (
                runtime._confirm_project_instruction_dispatch
            ),
            "select_project_instruction_binding": (
                runtime._select_project_instruction_binding
            ),
        }
        # TASK-34667: everything in ``runtime_owned`` overwrites whatever a
        # caller passed. Surface that at debug level instead of swallowing
        # it -- an always-overwritten caller copy is the
        # ``_library_provider_for_app`` drift precedent, where a stale copy
        # silently shadowed (and differed from) the live binding and cost
        # 24 Library tools behind one swallowed warning.
        ignored = runtime_owned.keys() & kwargs.keys()
        if ignored:
            runtime_module.logger.debug(
                "ensure_chat_controller: ignoring runtime-owned kwargs "
                f"supplied by caller: {sorted(ignored)}"
            )
        kwargs.update(runtime_owned)
        kwargs.setdefault("buddy_sink", runtime.persona_buddy_sink)
        kwargs.setdefault("scratch_spaces", runtime._scratch_spaces)
        kwargs.setdefault("activity_receipts", runtime._activity_receipts)
        if "canvas_enabled_reader" not in kwargs:
            kwargs["canvas_enabled_reader"] = runtime._canvas_enabled
            kwargs.setdefault("canvas_disabled_reader", runtime.canvas_disabled)
        raw_cli_runtime = (
            domain.sources.raw_cli_runtime
            if domain is not None
            else getattr(runtime._app, "raw_cli_runtime", None)
        )
        kwargs.setdefault(
            "cancel_raw_cli_session",
            getattr(raw_cli_runtime, "cancel_session", None),
        )
        runtime.set_chat_controller(ConsoleChatController(**kwargs))
        runtime._chat_controller._archive_recovery_runtime = runtime._archive_recovery
        runtime._chat_controller.prompt_history = runtime.ensure_prompt_history()
        if runtime.view is None:
            # task-15860 Task 4: a runtime can be VIEWLESS FROM BIRTH, not
            # only after a detach — nothing about the caller that supplies
            # these constructor parameters makes it an attached view, and
            # the wake-at-launch case (Console never opened) has no view at
            # all. Without this the fresh controller would keep the
            # constructor's own `None`s, and `wake_conversation_in_view`'s
            # read site reads that as IN VIEW: the ◈ mark cleared for a
            # delivery nobody could have seen. `only="controller"` because
            # the STORE's one slot (`on_scope_flushed`) is a constructor
            # parameter its own caller just supplied, and nulling it in the
            # restore-before-attach window would drop scope flushes.
            # Production attaches one line later
            # (`ChatScreen._ensure_console_chat_controller`), so this costs
            # a mounted Console nothing.
            runtime._clear_view_hooks(only="controller")
            runtime._clear_view_hooks(only="wake")
        else:
            runtime._bind_view_hooks()
        # ADR-135: the native controller's birth owns recovery. View remounts
        # and repeated ensure/read calls return above without auditing owners.
        wake = runtime._chat_controller.fleet_wake
        if domain is not None:
            domain.presentation.wire_wake(wake, runtime._project_workflow_recovery)
        else:
            wake.wire(
                app=runtime._app,
                startup_ready=lambda: bool(getattr(runtime._app, "_ui_ready", True)),
            )
            wake._recovery_projection = runtime._project_workflow_recovery
        wake.start_recovery()
        return runtime._chat_controller

    def prepare_display_owner(self) -> ConsoleDomainDisplayOwner:
        """Construct inert producers only for the already-selected actual domain."""
        from .console_domain_display import ConsoleDomainDisplayOwner
        from .console_display_phases import ConsoleDisplayPhaseRefused

        domain = self.domain_services
        if domain is None or domain.runtime is not self._runtime:
            raise ConsoleDisplayPhaseRefused("No selected display domain")
        domain.require_current()
        if self._display_owner is None:
            owner = ConsoleDomainDisplayOwner(domain)
            owner.initialize()
            self._display_owner = owner
        owner = self._display_owner
        if owner.domain is not domain or owner.runtime is not self._runtime:
            raise ConsoleDisplayPhaseRefused("Display construction identity changed")
        return owner
