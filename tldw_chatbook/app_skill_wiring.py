from typing import (  # noqa: UP035 - preserve original public annotation objects
    Any,
    Callable,
    NamedTuple,
)

from . import app_service_wiring as _source_wiring

from .Skills_Interop.skill_trust_models import (
    _SkillTrustPublisher,
    _skill_publisher_identity,
    _skill_trust_publisher as _skill_trust_publisher,
)


def record_native_skill_trust_candidate(app, candidate):
    """Record the service a first-use build just published, for the active publication.

    Internal: called only by the two first-use builders (the
    ``local_skill_trust_service`` getter and ``ensure_local_skill_trust_service``)
    right after they fill the empty slot; the setter never calls it. The
    publication registered for this app on this thread and task, and the slot
    still holding ``candidate``, decide; the caller's identity does not. Any
    caller on that thread and task could pass the slot's current value and have
    it recorded; ADR-356 rule 4 accepts that, since in-process code is not a
    boundary this check could enforce.
    """
    marker = _skill_trust_publisher.get()
    if type(marker) is not _SkillTrustPublisher or not marker.active:
        return
    if _skill_publisher_identity() != (marker.thread_id, marker.task):
        return
    source = marker.preparation.trust_source
    if source is None or source.app is not app or candidate is None:
        return
    if vars(app).get("_local_skill_trust_service") is not candidate:
        return
    marker.candidate = candidate


class _ConsoleSkillTrustSource(NamedTuple):
    """One stock service proof; no UI owner or permission verdict is captured."""

    app: Any
    scope: Any
    local: Any
    source_current: Callable[[], bool]
    current: Callable[[], bool]
    loop: Any
    thread: Any
    runtime_reads: set


def _disabled_builtin_skills(config: Any) -> frozenset[str]:
    """Built-in skills disabled in config; imported lazily (off boot path, ADR-097)."""
    from tldw_chatbook.Skills_Interop.builtin_skills import (
        disabled_builtins_from_config,
    )

    return disabled_builtins_from_config(config)


def _read_app_disabled_builtin_skills(app: object) -> frozenset[str]:
    """Read the current stock app config without invoking a UI callback."""
    return _source_wiring._disabled_builtin_skills(getattr(app, "app_config", None))


def _capture_console_skill_trust_source(app, service):
    """Capture resident service sources without a UI scheduling dependency.

    ADR-356 rule 4: the app's own registrations and current state decide (its
    type, slots, runtime, the callbacks it registered for this skills stack,
    config identity and environment), not function or code identity; a
    wrapped or decorated skill collaborator qualifies like the stock one.
    Per-instance overrides of the skill methods still decline.
    """
    _source_wiring = globals()["_source_wiring"]
    import sys
    from types import ModuleType

    metadata = sys.modules.get("tldw_chatbook.Widgets.compact_model_bar")
    if type(metadata) is not ModuleType:
        return None
    app_module = sys.modules.get("tldw_chatbook.app")
    scope_module = sys.modules.get("tldw_chatbook.Skills_Interop.skills_scope_service")
    local_module = sys.modules.get("tldw_chatbook.Skills_Interop.local_skills_service")
    config_module = sys.modules.get("tldw_chatbook.config")
    if any(
        type(module) is not ModuleType
        for module in (app_module, scope_module, local_module, config_module)
    ):
        return None
    app_namespace = vars(app_module)
    app_record = app_namespace.get("_CONSOLE_SKILL_APP_SOURCE")
    if type(app_record) is not tuple or len(app_record) != 5:
        return None
    app_type, defining, _factory, _factory_code, runtime_type = app_record
    if (
        defining is not app_namespace
        or type(app) is not app_type
        or app_namespace.get("TldwCli") is not app_type
    ):
        return None
    app_slots = (
        "_build_local_skill_trust_service",
        "_build_local_skills_stack",
        "ensure_local_skill_trust_service",
        "local_skill_trust_service",
        "local_skills_service",
        "skills_scope_service",
    )
    scope_type = vars(scope_module).get("SkillsScopeService")
    local_type = vars(local_module).get("LocalSkillsService")
    app_fields = metadata._plain_fields(
        app,
        app_type,
        (
            "_local_skill_trust_service",
            "_local_skill_trust_service_build_lock",
            "_local_skills_service",
            "_skills_scope_service",
            "_local_skills_stack_inputs",
            "_local_skills_callbacks",
            "app_config",
            "_shutting_down",
            "_exit",
            "_workers",
            "_thread_id",
            "console_runtime",
            "_console_runtime_shutdown_task",
        ),
    )
    scope_fields = metadata._plain_fields(
        service, scope_type, ("local_service", "server_service", "policy_enforcer")
    )
    if app_fields is None or scope_fields is None:
        return None
    runtime = app_fields.get("console_runtime")
    if (
        runtime is None
        or app_fields.get("_console_runtime_shutdown_task") is not None
        or app_fields.get("_shutting_down") is not False
        or app_fields.get("_exit") is not False
    ):
        raise RuntimeError("console_skill_trust_runtime_closed")
    runtime_fields = metadata._plain_fields(
        runtime, runtime_type, ("_disposed", "_preparation_reads", "_app")
    )
    if (
        runtime_fields is None
        or type(runtime_fields.get("_preparation_reads")) is not set  # exact source metadata
    ):  # exact observer ownership
        return None
    runtime_reads = runtime_fields["_preparation_reads"]

    def runtime_current():
        return (
            app_namespace.get("ConsoleRuntime") is runtime_type
            and metadata._plain_fields(
                runtime, runtime_type, ("_disposed", "_preparation_reads", "_app")
            )
            is runtime_fields
            and app_fields.get("console_runtime") is runtime
            and app_fields.get("_console_runtime_shutdown_task") is None
            and app_fields.get("_shutting_down") is False
            and app_fields.get("_exit") is False
            and runtime_fields.get("_app") is app
            and runtime_fields.get("_disposed") is False
            and runtime_fields.get("_preparation_reads") is runtime_reads
        )

    # An original closing Runtime is a refusal, not eligibility for direct IO.
    if not runtime_current():
        raise RuntimeError("console_skill_trust_runtime_closed")
    local = scope_fields.get("local_service")
    local_fields = metadata._plain_fields(
        local,
        local_type,
        (
            "_trust_service",
            "_trust_service_factory",
            "store_dir",
            "skills_dir",
            "policy_enforcer",
        ),
    )
    if local_fields is None:
        return None
    if (
        any(name in app_fields for name in app_slots)
        or any(name in scope_fields for name in _source_wiring._CONSOLE_SKILL_SCOPE_SLOTS)
        or any(name in local_fields for name in _source_wiring._CONSOLE_SKILL_LOCAL_SLOTS)
    ):
        return None
    if (
        app_fields.get("_skills_scope_service") is not service
        or app_fields.get("_local_skills_service") is not local
        or app_fields.get("_shutting_down") is not False
        or app_fields.get("_exit") is not False
        or type(app_fields.get("app_config")) is not dict  # exact stock metadata; no custom dispatch
    ):
        return None
    # The trust factory must be the one this app registered for this stack;
    # an injected factory declines here, before anything calls it.
    callbacks = app_fields.get("_local_skills_callbacks")
    lazy = local_fields.get("_trust_service_factory")
    if type(callbacks) is not tuple or len(callbacks) != 3 or lazy is not callbacks[0]:
        return None
    # Preserve the original captured policy collaborators; no verdict is reused.
    inputs = app_fields.get("_local_skills_stack_inputs")
    if type(inputs) is not tuple or len(inputs) != 2:
        return None
    if (
        scope_fields.get("policy_enforcer") is not inputs[0]
        or local_fields.get("policy_enforcer") is not inputs[0]
        or scope_fields.get("server_service") is not inputs[1]
    ):
        return None
    mapping = app_fields["app_config"]
    lock = app_fields["_local_skill_trust_service_build_lock"]
    if type(lock) is not _source_wiring.asyncio.Lock:
        return None
    store_dir, skills_dir = (
        local_fields.get("store_dir"),
        local_fields.get("skills_dir"),
    )
    identity = config_module.current_config_identity()
    cache = vars(config_module).get("_SETTINGS_CACHE")
    posture = vars(config_module).get("_SETTINGS_CACHE_POSTURE")
    selector = tuple(
        _source_wiring.os.environ.get(key)
        for key in (
            "TLDW_CONFIG_PATH",
            "HOME",
            "USERPROFILE",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        )
    )
    loop, thread = _source_wiring.asyncio.get_running_loop(), _source_wiring.threading.current_thread()
    if (
        type(app_fields.get("_thread_id")) is not int  # exact stock metadata; no custom dispatch
        or app_fields.get("_thread_id") != thread.ident
    ):
        return None

    def source_current():
        # Captured plain fields only: no Textual accessor or cached authority.
        if not (
            runtime_current()
            and sys.modules.get("tldw_chatbook.app") is app_module
            and app_namespace.get("_CONSOLE_SKILL_APP_SOURCE") is app_record
            and app_namespace.get("TldwCli") is app_type
            and config_module.current_config_identity() == identity
            and vars(config_module).get("_SETTINGS_CACHE") is cache
            and vars(config_module).get("_SETTINGS_CACHE_POSTURE") is posture
            and tuple(
                _source_wiring.os.environ.get(key)
                for key in (
                    "TLDW_CONFIG_PATH",
                    "HOME",
                    "USERPROFILE",
                    "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME",
                )
            )
            == selector
        ):
            return False
        return (
            metadata._plain_fields(
                app,
                app_type,
                (
                    "_local_skill_trust_service",
                    "_local_skill_trust_service_build_lock",
                    "_local_skills_service",
                    "_skills_scope_service",
                    "_local_skills_stack_inputs",
                    "_local_skills_callbacks",
                    "app_config",
                    "_shutting_down",
                    "_exit",
                    "_workers",
                    "_thread_id",
                    "console_runtime",
                    "_console_runtime_shutdown_task",
                ),
            )
            is app_fields
            and metadata._plain_fields(
                service,
                scope_type,
                ("local_service", "server_service", "policy_enforcer"),
            )
            is scope_fields
            and metadata._plain_fields(
                local,
                local_type,
                (
                    "_trust_service",
                    "_trust_service_factory",
                    "store_dir",
                    "skills_dir",
                    "policy_enforcer",
                ),
            )
            is local_fields
            and app_fields.get("_skills_scope_service") is service
            and app_fields.get("_local_skills_service") is local
            and app_fields.get("_local_skill_trust_service_build_lock") is lock
            and app_fields.get("app_config") is mapping
            and type(app_fields.get("_thread_id")) is int  # original exact source field
            and app_fields.get("_thread_id") == thread.ident
            and app_fields.get("_shutting_down") is False
            and app_fields.get("_exit") is False
            and not any(name in app_fields for name in app_slots)
            and app_fields.get("_local_skills_callbacks") is callbacks
            and local_fields.get("_trust_service_factory") is lazy
            and local_fields.get("store_dir") is store_dir
            and local_fields.get("skills_dir") is skills_dir
            and not any(name in scope_fields for name in _source_wiring._CONSOLE_SKILL_SCOPE_SLOTS)
            and not any(name in local_fields for name in _source_wiring._CONSOLE_SKILL_LOCAL_SLOTS)
            and scope_fields.get("server_service") is inputs[1]
            and app_fields.get("_local_skills_stack_inputs") is inputs
            and scope_fields.get("local_service") is local
            and scope_fields.get("policy_enforcer") is inputs[0]
            and local_fields.get("policy_enforcer") is inputs[0]
        )

    def current():
        try:
            return (
                _source_wiring.asyncio.get_running_loop() is loop
                and _source_wiring.threading.current_thread() is thread
                and source_current()
            )
        except (AttributeError, TypeError, ValueError):
            return False

    if not current():
        return None
    return _source_wiring._ConsoleSkillTrustSource(
        app, service, local, source_current, current, loop, thread, runtime_reads
    )


def _capture_console_skill_trust_preparation(app, service, owner_current):
    """Add the original skill-discovery App-worker scheduling contract.

    ADR-356 rule 4: the trust source (the app's registrations and slots) and
    the app's worker-manager slots decide; ``workers``, ``app`` and
    ``run_worker`` must not be overridden on this app object, and their class
    methods are not compared, so a wrapped one qualifies like the stock one.
    """
    _source_wiring = globals()["_source_wiring"]
    import sys

    source = _source_wiring._capture_console_skill_trust_source(app, service)
    if source is None:
        return None
    app_fields = vars(app)
    if (
        app_fields.get("_local_skill_trust_service") is not None
        or vars(source.local).get("_trust_service") is not None
    ):
        return None
    metadata = sys.modules.get("tldw_chatbook.Widgets.compact_model_bar")
    manager = app_fields.get("_workers")
    manager_fields = metadata._plain_fields(
        manager, _source_wiring._SkillWorkerManager, ("_app", "_workers")
    )
    if manager_fields is None:
        return None
    manager_workers = manager_fields.get("_workers")

    def current(expected_trust=None):
        try:
            return (
                source.current()
                and owner_current() is True
                # TASK-38948: App.run_async installs the stock eager factory,
                # so requiring None refused every real run, and the cold trust
                # build and skill read then ran on the UI loop at mount.
                and (
                    source.loop.get_task_factory() is None
                    or source.loop.get_task_factory()
                    is _source_wiring.asyncio.eager_task_factory
                )
                and app_fields.get("_local_skill_trust_service") is expected_trust
                and app_fields.get("_workers") is manager
                and metadata._plain_fields(
                    manager, _source_wiring._SkillWorkerManager, ("_app", "_workers")
                )
                is manager_fields
                and manager_fields.get("_app") is app
                and type(manager_workers) is set  # original exact worker owner
                and manager_fields.get("_workers") is manager_workers
                and not any(
                    name in manager_fields for name in ("_new_worker", "add_worker")
                )
                and not any(name in app_fields for name in ("workers", "app", "run_worker"))
                and app.app is app
            )
        except (AttributeError, TypeError, ValueError):
            return False

    if not current():
        return None
    return (
        app,
        source.source_current,
        current,
        (manager, manager_fields, source.loop, source.runtime_reads),
    )


async def _prepare_console_skill_trust_service(captured):
    """Retain a stock preparation in the App's selected worker drain scope."""
    _source_wiring = globals()["_source_wiring"]
    app, source_current, current, custody = captured
    if not current():
        raise RuntimeError("console_skill_trust_owner_changed")
    # Keep the original scope gate ahead of optional lazy trust construction.
    vars(app)["_skills_scope_service"]._enforce_policy("skills.context.list.local")
    if not current():
        raise RuntimeError("console_skill_trust_owner_changed")
    preparation = _source_wiring.ServiceWiringMixin.ensure_local_skill_trust_service(
        app,
        _source_current=source_current,
        _owner_current=current,
        _read_observers=(custody[3],),
    )
    try:
        worker = app.run_worker(
            preparation,
            group="console-skill-trust-setup",
            exclusive=False,
            exit_on_error=False,
        )
    except BaseException:
        preparation.close()
        raise
    from textual.worker import WorkerCancelled, WorkerFailed

    if type(worker) is not _source_wiring.Worker:
        raise RuntimeError("console_skill_trust_worker_changed")
    manager, manager_fields, loop, _runtime_reads = custody
    values = vars(worker)
    task = values.get("_task")
    if (
        vars(app).get("_workers") is not manager
        or vars(manager) is not manager_fields
        or manager_fields.get("_app") is not app
        or values.get("_node") is not app
        or values.get("_work") is not preparation
        or type(task) is not _source_wiring.asyncio.Task
        or task.get_loop() is not loop
        or not any(item is worker for item in manager_fields["_workers"])
    ):
        raise RuntimeError("console_skill_trust_worker_changed")
    try:
        value = await worker.wait()
    except WorkerCancelled:
        error = vars(worker).get("_error")
        if isinstance(error, _source_wiring.asyncio.CancelledError):
            raise error
        raise _source_wiring.asyncio.CancelledError
    except WorkerFailed:
        error = vars(worker).get("_error")
        if isinstance(error, BaseException):
            raise error
        raise
    if not current(value):
        raise RuntimeError("console_skill_trust_owner_changed")










def _install_skill_source(_source_wiring):
    from textual.worker_manager import WorkerManager as _SkillWorkerManager

    vars(_source_wiring).update({
        "_SkillWorkerManager": _SkillWorkerManager,
        # Skill methods the trust source refuses to see overridden per instance.
        "_CONSOLE_SKILL_SCOPE_SLOTS": (
            "get_context",
            "_call",
            "_enforce_policy",
            "_require_service",
            "_normalize_mode",
            "_normalize_response",
            "_maybe_await",
            "_source_action_id",
            "_normalize_item",
            "_with_record_id",
        ),
        "_CONSOLE_SKILL_LOCAL_SLOTS": ("get_context", "trust_service"),
    })


def _wire_skill_inputs(self):
    try:
        self.server_skills_service = _source_wiring.ServerSkillsService.from_config(
            self.app_config,
            policy_enforcer=self.service_policy_enforcer,
        )
    except ValueError:
        self.server_skills_service = _source_wiring.ServerSkillsService(
            client=None,
            policy_enforcer=self.service_policy_enforcer,
        )
    # The local skills stack (trust service -> local service -> scope
    # facade) is built on first access, not here: constructing the trust
    # service performs OS keyring backend discovery TWICE (marker store +
    # key cache) for a feature most boots never touch (TASK-21111(b)).
    # Every consumer reads these through `getattr(app_instance, ...)` at
    # UI time, so a property is a drop-in.
    self._local_skill_trust_service: _source_wiring.Any | None = None
    # task-33081: serializes the off-loop first build above.
    self._local_skill_trust_service_build_lock = _source_wiring.asyncio.Lock()
    self._local_skills_service: _source_wiring.Any | None = None
    self._skills_scope_service: _source_wiring.Any | None = None
    # Captured NOW, at the timing the eager build had: `_build_local_
    # skills_stack` must not re-read collaborators that a test (or a
    # later boot step) may reassign between construction and first use
    # (the TASK-21108 deferral trap).
    self._local_skills_stack_inputs = (
        self.service_policy_enforcer,
        self.server_skills_service,
    )


class SkillServiceWiringMixin:
    def _build_local_skill_trust_service(self) -> Any:
        """Build the skill trust service. Performs OS keyring discovery.

        Split out of the eager wiring (TASK-21111(b)): it is the only part
        of the local skills stack that touches the keyring -- twice, once
        for the rollback marker store's secure-backend probe and once for
        the trust key cache -- and nothing at startup asks a trust question.
        Deferring the whole SERVICE was not enough on its own: the Console's
        agent bridge takes the skills scope facade during Chat screen mount,
        which merely relocated the discovery from ``__init__`` to mount.
        ``LocalSkillsService`` therefore takes this as a FACTORY and calls it
        on the first trust decision.
        """
        local_skills_store_dir = _source_wiring.default_local_skills_store_dir(_source_wiring.get_user_data_dir())
        trust_store_dir = _source_wiring.default_trust_store_dir(local_skills_store_dir)
        from .Skills_Interop.recovery_activation import is_recovered
        recovered_skills = is_recovered(
            local_skills_store_dir / "skills", trust_store_dir
        )
        if recovered_skills:
            from .Skills_Interop.skill_trust_store import (
                FileSkillTrustGenerationMarkerStore,
            )
            skill_trust_marker_store = FileSkillTrustGenerationMarkerStore(
                trust_store_dir / _source_wiring._SKILL_TRUST_MARKER_FILENAME, store_dir=trust_store_dir
            )
            reduced_rollback_protection = True
            skill_key_cache = None
        else:
            trust_account_scope = _source_wiring.skill_trust_account_scope(trust_store_dir)
            skill_trust_marker_store, reduced_rollback_protection = (
                _source_wiring.build_skill_trust_marker_store_with_fallback(
                    fallback_marker_path=trust_store_dir / _source_wiring._SKILL_TRUST_MARKER_FILENAME,
                    store_dir=trust_store_dir,
                    account_scope=trust_account_scope,
                )
            )
            skill_key_cache = _source_wiring.build_default_skill_trust_key_cache(
                account_scope=trust_account_scope
            )
        return _source_wiring.SkillTrustService(
            skills_dir=local_skills_store_dir / "skills",
            trust_store=_source_wiring.SkillTrustStore(
                store_dir=trust_store_dir,
                marker_store=skill_trust_marker_store,
            ),
            key_cache=skill_key_cache,
            keyring_convenience_enabled=False,
            reduced_rollback_protection=reduced_rollback_protection,
        )

    @property
    def plugin_operations(self) -> Any:
        """Retain one lazy presentation state over the existing plugin owner."""
        from tldw_chatbook.UI.Plugins_Modules.operations import app_operations

        return app_operations(self, self._build_plugin_service())

    @property
    def plugin_source_service(self) -> Any:
        """One lazy source facade, with no construction-time storage or fetching."""
        service = getattr(self, "_plugin_source_service", None)
        if service is None:
            from tldw_chatbook.Plugins.source_service import SourceService

            service = SourceService(self._build_plugin_service())
            self._plugin_source_service = service
        return service

    def _build_plugin_service(self) -> Any:
        """Build one IO-free facade; its worker starts on explicit plugin use."""
        service = getattr(self, "_plugin_service", None)
        if service is None:
            from tldw_chatbook.Plugins.service import PluginService
            service = PluginService(
                _source_wiring.get_user_data_dir(),
                workspace_lookup=self.workspace_registry_service.get_workspace,
                mcp_mapping_owner=self.local_mcp_control_service,
                model_mapping_owner=self.local_llm_provider_catalog_service,
            )
            self._plugin_service = service
        return service

    def _build_local_skills_stack(self) -> None:
        """Build the local skills service + scope facade. Idempotent.

        Body moved out of ``_wire_watchlists_and_notifications_services``
        (TASK-21111(b)). Keyring-free: the trust service is handed over as a
        factory. The collaborators it reads were captured at construction
        time, not re-read now, so deferring changes WHEN it runs and not
        WHAT it binds (the TASK-21108 trap).

        Each slot is filled only if still unset, so an injected double (a
        test assigning one of them between construction and first read) is
        never clobbered by a later sibling access.
        """
        if None not in (self._local_skills_service, self._skills_scope_service):
            return
        policy_enforcer, server_skills_service = self._local_skills_stack_inputs
        if self._local_skills_service is None:
            callbacks = (
                lambda: self.local_skill_trust_service,
                self._build_plugin_service,
                # In-memory config read: this runs on every skills read,
                # including the Console's per-send capture.
                _source_wiring.partial(_source_wiring._read_app_disabled_builtin_skills, self),
            )
            service = _source_wiring.LocalSkillsService(
                store_dir=_source_wiring.default_local_skills_store_dir(_source_wiring.get_user_data_dir()),
                policy_enforcer=policy_enforcer,
                trust_service_factory=callbacks[0],
                plugin_service_factory=callbacks[1],
                builtin_disabled_loader=callbacks[2],
            )
            # Console Send preparation trusts these slots only while they hold
            # what the app registered (ADR-356 rule 4). Publish the record and
            # the service together, with no call between them, so a racing
            # first build cannot leave a record that names a discarded service.
            if self._local_skills_service is None:
                self._local_skills_callbacks = callbacks
                self._local_skills_service = service
        if self._skills_scope_service is None:
            self._skills_scope_service = _source_wiring.SkillsScopeService(
                local_service=self._local_skills_service,
                server_service=server_skills_service,
                policy_enforcer=policy_enforcer,
            )

    async def ensure_local_skill_trust_service(
        self,
        *,
        _source_current: Callable[[], bool] | None = None,
        _owner_current: Callable[[Any], bool] | None = None,
        _read_observers: tuple[set, ...] = (),
    ) -> Any:
        """First-use trust service build, OFF the UI event loop (task-33081).

        The build performs OS keyring backend discovery (SecretService/D-Bus
        on Linux -- the deferred-wiring notes' timing figure never measured
        that platform). The sync property keeps its contract for callers
        already on worker threads; UI-loop callers must come here so the
        discovery never blocks the loop.

        Returns:
            The shared local skill trust service, built once; concurrent
            first callers await the same build under the build lock.
        """
        if _owner_current is not None and _owner_current() is not True:
            raise RuntimeError("console_skill_trust_owner_changed")
        if self._local_skill_trust_service is not None:
            return self._local_skill_trust_service
        published_candidate = None
        async with self._local_skill_trust_service_build_lock:
            if (
                _owner_current is not None
                and _owner_current(self._local_skill_trust_service) is not True
            ):
                raise RuntimeError("console_skill_trust_owner_changed")
            if self._local_skill_trust_service is None:
                from .Chat.console_preparation_reads import run_preparation_read
                def require_current():
                    if _source_current is not None and _source_current() is not True:
                        raise RuntimeError("console_skill_trust_source_changed")
                service = await run_preparation_read(
                    self._build_local_skill_trust_service,
                    creator=self,
                    session_id=None,
                    reads=set(),
                    observers=_read_observers,
                    require_current=require_current,
                )
                if (
                    _owner_current is not None
                    and _owner_current(self._local_skill_trust_service) is not True
                ):
                    raise RuntimeError("console_skill_trust_owner_changed")
                # Another owner may publish while the native builder is held.
                if self._local_skill_trust_service is None:
                    self._local_skill_trust_service = service
                    published_candidate = service
            winner = self._local_skill_trust_service
        if published_candidate is not None and _skill_trust_publisher.get() is not None:
            record_native_skill_trust_candidate(self, published_candidate)
        return winner
