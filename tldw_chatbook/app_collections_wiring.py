from typing import (  # noqa: UP035 - preserve original public annotation objects
    TYPE_CHECKING,
    Any,
    Mapping,
)

from . import app_service_wiring as _source_wiring

if TYPE_CHECKING:  # pragma: no cover - original typing-only references
    from tldw_chatbook.Library.collections_capture_models import (
        ExternalMediaReference,
        ExternalNoteReference,
        ExternalReferenceAvailability,
    )


class _CollectionsSetupObsolete(RuntimeError):
    """The original deferred initializer no longer owns publication."""


class _DeferredCollectionsCaptureScope:
    """Stable app seam that composes the real capture scope on first use."""

    def __init__(self, owner: Any) -> None:
        object.__setattr__(self, "_owner", owner)

    def _resolve(self) -> Any:
        owner = object.__getattribute__(self, "_owner")
        scope = _source_wiring.TldwCli.ensure_collections_capture_services(owner)
        if scope is None or scope is self:
            raise RuntimeError("collections_capture_scope_unavailable")
        return scope

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._resolve(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._resolve(), name)


def _external_reference_payload_id(payload: Any, *keys: str) -> str | None:
    """Read one owner identifier without trusting a response's concrete type."""
    if payload is None:
        return None
    if isinstance(payload, _source_wiring.Mapping):
        values = payload
    else:
        model_dump = getattr(payload, "model_dump", None)
        values = model_dump(mode="json") if callable(model_dump) else None
    for key in keys:
        value = values.get(key) if isinstance(values, _source_wiring.Mapping) else getattr(payload, key, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _external_reference_failure_reason(error: Exception, owner: str) -> str:
    """Map owner failures to bounded provenance reasons."""
    status_code = getattr(error, "status_code", None)
    if status_code in {401, 403} or isinstance(error, PermissionError):
        return f"{owner}_reference_unauthorized"
    if status_code == 404 or isinstance(error, (KeyError, LookupError)):
        return f"{owner}_reference_missing"
    return "reference_resolution_retryable"


def _collections_reference_authority_kind(app: Any, authority_key: str) -> str | None:
    """Return the active owner kind for an exact opaque capture authority."""
    for kind, attribute in (
        ("local", "local_collections_capture_authority"),
        ("server", "server_collections_capture_authority"),
    ):
        authority = getattr(app, attribute, None)
        if authority is not None and getattr(authority, "key", None) == authority_key:
            return kind
    return None


async def _resolve_collections_media_reference(
    app: Any,
    reference: "ExternalMediaReference",
) -> "ExternalReferenceAvailability":
    """Resolve a capture's stored backing-Media identity through its owner."""
    from tldw_chatbook.Library.collections_capture_models import (
        ExternalReferenceAvailability,
    )

    kind = _source_wiring._collections_reference_authority_kind(app, reference.authority_key)
    if kind is None:
        return ExternalReferenceAvailability(
            "unavailable", "media_reference_authority_mismatch"
        )
    mode = (
        _source_wiring.MediaReadingBackend.LOCAL
        if kind == "local"
        else _source_wiring.MediaReadingBackend.SERVER
    )
    try:
        payload = await app.media_reading_scope_service.get_backing_media_item(
            mode=mode,
            media_id=reference.item_id,
            include_content=False,
            include_versions=False,
        )
    except Exception as exc:
        return ExternalReferenceAvailability(
            "unavailable", _source_wiring._external_reference_failure_reason(exc, "media")
        )
    resolved_id = _source_wiring._external_reference_payload_id(payload, "id", "media_id")
    if resolved_id is None:
        return ExternalReferenceAvailability("unavailable", "media_reference_missing")
    if resolved_id != reference.item_id:
        return ExternalReferenceAvailability(
            "unavailable", "media_reference_identity_mismatch"
        )
    return ExternalReferenceAvailability("available")


async def _resolve_collections_note_reference(
    app: Any,
    reference: "ExternalNoteReference",
) -> "ExternalReferenceAvailability":
    """Resolve a capture's note link through Local or Server Notes only."""
    from tldw_chatbook.Library.collections_capture_models import (
        ExternalReferenceAvailability,
    )

    kind = _source_wiring._collections_reference_authority_kind(app, reference.authority_key)
    if kind is None:
        return ExternalReferenceAvailability(
            "unavailable", "note_reference_authority_mismatch"
        )
    scope = _source_wiring.ScopeType.LOCAL_NOTE if kind == "local" else _source_wiring.ScopeType.SERVER_NOTE
    try:
        payload = await app.notes_scope_service.get_note_detail(
            scope=scope,
            note_id=reference.note_id,
            user_id=getattr(app, "notes_user_id", None) if kind == "local" else None,
        )
    except Exception as exc:
        return ExternalReferenceAvailability(
            "unavailable", _source_wiring._external_reference_failure_reason(exc, "note")
        )
    resolved_id = _source_wiring._external_reference_payload_id(payload, "id", "note_id")
    if resolved_id is None:
        return ExternalReferenceAvailability("unavailable", "note_reference_missing")
    if resolved_id != reference.note_id:
        return ExternalReferenceAvailability(
            "unavailable", "note_reference_identity_mismatch"
        )
    return ExternalReferenceAvailability("available")


def _extract_collections_article(
    url: str, *, url_provenance: Any = None
) -> Mapping[str, Any]:
    """Fetch one capture through the existing guarded article extractor."""
    from tldw_chatbook.Local_Ingestion.web_article_ingestion import (
        extract_article_for_ingest,
    )
    from tldw_chatbook.Utils.egress import UrlProvenance

    return extract_article_for_ingest(
        url, {}, url_provenance=url_provenance or UrlProvenance.UNKNOWN
    )


def _capture_deferred_collections_setup(app):
    _source_wiring = globals()["_source_wiring"]
    import inspect
    import sys
    from types import SimpleNamespace

    from tldw_chatbook import config
    from tldw_chatbook.DB import Library_Collections_DB
    from tldw_chatbook.Library import (
        collections_capture_repository,
        collections_capture_service,
        collections_legacy_recovery,
        collections_offline_store,
    )

    app_module = sys.modules["tldw_chatbook.app"]
    # Read like the two skill readers: no stock record means the stock path does
    # not apply (TASK-35302: an attribute read here crashed every app start).
    app_record = vars(app_module).get("_CONSOLE_SKILL_APP_SOURCE")
    if type(app_record) is not tuple or len(app_record) != 5:
        return None
    app_type, defining, factory, factory_code, _runtime_type = app_record
    fields = vars(app)
    database = fields.get("local_library_collections_db")
    scope = fields.get("collections_capture_scope_service")
    if (
        defining is not vars(app_module)
        or app_module.TldwCli is not app_type
        or type(app) is not app_type
        or factory.__code__ is not factory_code
        or inspect.getattr_static(app, "_create_deferred_startup_task") is not factory
        or type(database) is not _source_wiring.LibraryCollectionsDB
        or database.is_memory_db
        or type(scope) is not _source_wiring._DeferredCollectionsCaptureScope
        or fields.get("_collections_capture_initializer_closed", False)
        or fields.get("_shutting_down", False)
        or fields.get("_exit", False)
        or type(fields.get("_deferred_startup_tasks")) is not set  # exact stock compatibility boundary
        or any(
            fields.get(name) is not None for name in _source_wiring._COLLECTIONS_CAPTURE_RESULT_FIELDS
        )
    ):
        return None
    sources = (
        _source_wiring._COLLECTIONS_SETUP_SOURCE,
        config._COLLECTIONS_SETUP_SOURCE,
        Library_Collections_DB._COLLECTIONS_SETUP_SOURCE,
        collections_capture_repository._COLLECTIONS_SETUP_SOURCE,
        collections_capture_service._COLLECTIONS_SETUP_SOURCE,
        collections_legacy_recovery._COLLECTIONS_SETUP_SOURCE,
        collections_offline_store._COLLECTIONS_SETUP_SOURCE,
    )
    checker, checker_code = _source_wiring._COLLECTIONS_SOURCE_CHECKER
    if (
        _source_wiring._collections_setup_sources_current is not checker
        or checker.__code__ is not checker_code
        or not checker(sources)
    ):
        return None
    receivers = tuple(
        (app, name, original) for name, original in _source_wiring._COLLECTIONS_APP_METHODS
    )
    receivers += ((app, "_create_deferred_startup_task", factory),)
    receivers += tuple(
        (database, name, original)
        for owner, name, original, _records in Library_Collections_DB._COLLECTIONS_SETUP_SOURCE[
            1
        ]
        if owner is _source_wiring.LibraryCollectionsDB
    )
    if any(
        inspect.getattr_static(owner, name, None) is not original
        for owner, name, original in receivers
    ):
        return None
    policy = app.runtime_policy
    if policy is None:
        return None
    request = SimpleNamespace(
        app=app,
        scope=scope,
        database=database,
        local=database._thread_local,
        path=database.db_path,
        participant=database._maintenance_participant,
        policy=policy,
        state=policy.state,
        identity=config.current_config_identity(),
        sources=sources,
        receivers=receivers,
        factory=factory,
        factory_code=factory_code,
        reads=set(),
        initializer=None,
        loop=_source_wiring.asyncio.get_running_loop(),
        thread=_source_wiring.threading.current_thread(),
    )
    require = _source_wiring._require_collections_setup_current
    checker = _source_wiring._collections_setup_sources_current
    require_code, checker_code = require.__code__, checker.__code__

    pointer = next(
        row for row in sources[0][1]
        if row[0] is globals() and row[1] == "_source_wiring"
    )

    def current():
        if (
            pointer[0].get(pointer[1]) is not pointer[2]
            or vars(_source_wiring).get("_require_collections_setup_current") is not require
            or require.__code__ is not require_code
            or vars(_source_wiring).get("_collections_setup_sources_current") is not checker
            or checker.__code__ is not checker_code
        ):
            raise _source_wiring._CollectionsSetupObsolete("collections_capture_setup_source_changed")
        require(request)

    request.require_current = current
    return request


def _require_collections_setup_current(request):
    pointer = next(
        row for row in request.sources[0][1]
        if row[0] is globals() and row[1] == "_source_wiring"
    )
    _source_wiring = pointer[2]
    if pointer[0].get(pointer[1]) is not _source_wiring:
        raise _source_wiring._CollectionsSetupObsolete(
            "collections_capture_setup_source_changed"
        )
    import inspect

    from tldw_chatbook import config

    app, database = request.app, request.database
    if (
        vars(app).get("_collections_capture_setup") is not request
        or vars(app).get("_collections_capture_initializer_task")
        is not request.initializer
        or type(request.initializer) is not _source_wiring._COLLECTIONS_TASK
        or request.initializer.get_loop() is not request.loop
        or request.factory.__code__ is not request.factory_code
        or vars(app).get("_collections_capture_initializer_closed", False)
        or vars(app).get("_shutting_down", False)
        or vars(app).get("_exit", False)
        or any(
            inspect.getattr_static(owner, name, None) is not original
            for owner, name, original in request.receivers
        )
        or app.collections_capture_scope_service is not request.scope
        or app.local_library_collections_db is not database
        or database._thread_local is not request.local
        or database.db_path != request.path
        or database._maintenance_participant is not request.participant
        or app.runtime_policy is not request.policy
        or request.policy.state is not request.state
        or not _source_wiring._collections_setup_sources_current(request.sources)
        or config.current_config_identity() != request.identity
        or any(
            vars(app).get(name) is not None
            for name in _source_wiring._COLLECTIONS_CAPTURE_RESULT_FIELDS
        )
    ):
        raise _source_wiring._CollectionsSetupObsolete("collections_capture_setup_changed")
    if _source_wiring.threading.current_thread() is request.thread and (
        _source_wiring.asyncio.get_running_loop() is not request.loop
        or _source_wiring.asyncio.current_task() is not request.initializer
    ):
        raise _source_wiring._CollectionsSetupObsolete("collections_capture_initializer_changed")


def _build_deferred_collections_capture_parts(request):
    from tldw_chatbook.Backup_Recovery.participants import (
        _core_cached_connection,
        _core_closing,
    )
    from tldw_chatbook.Utils.private_paths import lexical_path

    request.require_current()
    database_path = _source_wiring.get_library_collections_db_path()
    request.require_current()
    if lexical_path(database_path) != request.path:
        raise _source_wiring._CollectionsSetupObsolete(
            "collections_capture_database_selection_changed"
        )
    data_root = _source_wiring.get_user_data_dir()
    request.require_current()
    database, local = request.database, request.local
    previous = _core_cached_connection(database, getattr(local, "conn", None))
    connection = None
    try:
        request.require_current()
        connection = database._held_connection()
        request.require_current()
        # Each original repository/filesystem call keeps its own admission.
        # A database-only outer scope would reject the separate archive root.
        return _source_wiring._collections_capture_parts(
            database,
            database_path,
            data_root,
            require_current=request.require_current,
        )
    finally:
        if connection is not None and connection is not previous:
            with _core_closing(database, connection) as allowed:
                if not allowed:
                    raise RuntimeError("collections_capture_connection_not_retired")
                connection.close()
                if getattr(local, "conn", None) is connection:
                    local.conn = None


async def _initialize_deferred_collections_capture(request):
    _source_wiring = next(
        row[2] for row in request.sources[0][1]
        if row[0] is globals() and row[1] == "_source_wiring"
    )
    app = request.app
    try:
        parts = await _source_wiring._COLLECTIONS_RUN_PREPARATION(
            lambda: _source_wiring._build_deferred_collections_capture_parts(request),
            creator=app,
            session_id=None,
            reads=request.reads,
            require_current=request.require_current,
        )
        request.require_current()
        scope = _source_wiring._collections_capture_scope(app)
        service = _source_wiring._local_collections_capture_service(parts)
        request.require_current()
    except _source_wiring._CollectionsSetupObsolete:
        return
    except _source_wiring.asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - preserve the optional local-service unavailable outcome.
        try:
            request.require_current()
        except _source_wiring._CollectionsSetupObsolete:
            return
        app.collections_capture_scope_service = _source_wiring._collections_capture_scope(app)
        _source_wiring.logger.opt(exception=True).warning(
            "Local Collections capture service unavailable during app wiring"
        )
        return
    else:
        app.collections_capture_scope_service = scope
        _source_wiring._publish_collections_capture_parts(app, parts, service)
        app._create_deferred_startup_task(
            app._reconcile_collections_capture_startup(),
            name="deferred_collections_capture_reconciliation",
        )
    finally:
        if vars(app).get("_collections_capture_setup") is request:
            app._collections_capture_setup = None


async def _retire_deferred_collections_capture(app):
    app._collections_capture_initializer_closed = True
    task = vars(app).get("_collections_capture_initializer_task")
    cancellation = None
    if task is None:
        return None
    if not task.done():
        task.cancel()
    while not task.done():
        try:
            # wait does not propagate child cancellation or cancel the child.
            await _source_wiring.asyncio.wait({task})
        except _source_wiring.asyncio.CancelledError as error:
            cancellation = cancellation or error
    if not task.cancelled():
        task.result()
    return cancellation


def _collections_capture_scope(app):
    from tldw_chatbook.Library.collections_capture_service import (
        CollectionsCaptureScopeService,
    )

    return CollectionsCaptureScopeService(
        resolve_media_reference=_source_wiring.functools.partial(
            _source_wiring._resolve_collections_media_reference, app
        ),
        resolve_note_reference=_source_wiring.functools.partial(
            _source_wiring._resolve_collections_note_reference, app
        ),
    )


def _collections_capture_parts(
    database, database_path, data_root, *, require_current=None
):
    from tldw_chatbook.Library.collections_capture_repository import (
        CollectionsCaptureRepository,
    )
    from tldw_chatbook.Library.collections_capture_service import (
        build_local_capture_authority,
    )
    from tldw_chatbook.Library.collections_legacy_recovery import (
        LegacyCollectionsRecovery,
        LegacyCollectionsRecoveryError,
    )
    from tldw_chatbook.Library.collections_offline_store import CollectionsOfflineStore

    if require_current is not None:
        require_current()
    authority = build_local_capture_authority(
        profile_id=str(data_root.resolve()),
        database_identity=str(database_path.resolve()),
    )
    if require_current is not None:
        require_current()
    repository = CollectionsCaptureRepository(database, authority_key=authority.key)
    if require_current is not None:
        require_current()
    offline_store = CollectionsOfflineStore(
        repository,
        data_root=data_root,
        authority_fingerprint=authority.fingerprint,
    )
    if require_current is not None:
        require_current()
    legacy_recovery = LegacyCollectionsRecovery(database)
    try:
        legacy_recovery.list_collections(page=1, size=1)
        legacy_recovery_available = True
    except LegacyCollectionsRecoveryError:
        legacy_recovery_available = False
    if require_current is not None:
        require_current()
    return (
        authority,
        repository,
        offline_store,
        legacy_recovery,
        legacy_recovery_available,
    )


def _local_collections_capture_service(parts):
    from tldw_chatbook.Library.collections_capture_service import (
        LocalCollectionsCaptureService,
    )

    authority, repository, offline_store, _unused_legacy_recovery, legacy_available = parts
    service = LocalCollectionsCaptureService(
        authority,
        repository,
        offline_store=offline_store,
        extractor=_source_wiring._extract_collections_article,
        legacy_recovery_available=legacy_available,
    )
    return service


def _publish_collections_capture_parts(app, parts, service):
    authority, repository, offline_store, legacy_recovery, _legacy_available = parts
    app.collections_capture_repository = repository
    app.collections_offline_store = offline_store
    app.collections_legacy_recovery_service = legacy_recovery
    app.local_collections_capture_authority = authority
    app.local_collections_capture_service = service
    _source_wiring.TldwCli._activate_collections_capture_authority(app)


















def _install_collections_source(_source_wiring):



    # Definition-time compatibility boundary for the finite Collections initializer.
    from tldw_chatbook.Chat.console_preparation_reads import (
        run_preparation_read as _COLLECTIONS_RUN_PREPARATION,
    )
    vars(_source_wiring).update({'_COLLECTIONS_RUN_PREPARATION': _COLLECTIONS_RUN_PREPARATION})

    _COLLECTIONS_TASK = _source_wiring.asyncio.Task
    vars(_source_wiring).update({'_COLLECTIONS_TASK': _COLLECTIONS_TASK})
    _COLLECTIONS_SOURCE_CHECKER = (
        _source_wiring._collections_setup_sources_current,
        _source_wiring._collections_setup_sources_current.__code__,
    )
    vars(_source_wiring).update({'_COLLECTIONS_SOURCE_CHECKER': _COLLECTIONS_SOURCE_CHECKER})
    _COLLECTIONS_CAPTURE_RESULT_FIELDS = (
        "collections_capture_repository",
        "collections_offline_store",
        "collections_legacy_recovery_service",
        "local_collections_capture_authority",
        "local_collections_capture_service",
    )
    vars(_source_wiring).update({'_COLLECTIONS_CAPTURE_RESULT_FIELDS': _COLLECTIONS_CAPTURE_RESULT_FIELDS})
    _COLLECTIONS_APP_METHODS = tuple(
        (name, _source_wiring._collections_wiring.CollectionsServiceWiringMixin.__dict__[name])
        for name in (
            "_wire_collections_capture_services",
            "_reset_collections_capture_services",
            "ensure_collections_capture_services",
            "_deferred_wire_collections_capture_services",
            "_activate_collections_capture_authority",
            "_shutdown_collections_capture_runtime",
            "_reconcile_collections_capture_startup",
        )
    )
    vars(_source_wiring).update({'_COLLECTIONS_APP_METHODS': _COLLECTIONS_APP_METHODS})


    # Original callbacks eligible for finite deferred Collections setup only.
    def _record_collections_setup_source(entries):
        from types import FunctionType

        rows = []
        for owner, name in entries:
            original = owner[name] if type(owner) is dict else getattr(owner, name)  # exact stock compatibility boundary
            function = getattr(original, "__func__", original)
            records = []
            while type(function) is FunctionType:
                records.append(
                    (
                        function,
                        function.__code__,
                        function.__globals__,
                        function.__defaults__,
                        function.__kwdefaults__,
                        tuple((function.__kwdefaults__ or {}).items()),
                        function.__closure__,
                        tuple(
                            (cell, cell.cell_contents)
                            for cell in function.__closure__ or ()
                        ),
                        vars(function).get("__wrapped__"),
                    )
                )
                function = vars(function).get("__wrapped__")
            rows.append((owner, name, original, tuple(records)))
        return vars(_source_wiring), tuple(rows)


    _COLLECTIONS_SETUP_SOURCE = _record_collections_setup_source(
        (
            (vars(_source_wiring), "_collections_setup_sources_current"),
            (vars(_source_wiring), "_capture_deferred_collections_setup"),
            (vars(_source_wiring), "_require_collections_setup_current"),
            (vars(_source_wiring), "_build_deferred_collections_capture_parts"),
            (vars(_source_wiring), "_initialize_deferred_collections_capture"),
            (vars(_source_wiring), "_retire_deferred_collections_capture"),
            (vars(_source_wiring), "_collections_capture_scope"),
            (vars(_source_wiring), "_collections_capture_parts"),
            (vars(_source_wiring), "_local_collections_capture_service"),
            (vars(_source_wiring), "_publish_collections_capture_parts"),
            (vars(_source_wiring), "get_library_collections_db_path"),
            (vars(_source_wiring), "get_user_data_dir"),
            (vars(_source_wiring), "LibraryCollectionsDB"),
            (vars(_source_wiring), "_COLLECTIONS_RUN_PREPARATION"),
            (vars(_source_wiring), "_COLLECTIONS_TASK"),
            (vars(_source_wiring), "_resolve_collections_media_reference"),
            (vars(_source_wiring), "_resolve_collections_note_reference"),
            (vars(_source_wiring), "_extract_collections_article"),
            (_source_wiring.ServiceWiringMixin, "_wire_collections_capture_services"),
            (_source_wiring.ServiceWiringMixin, "_reset_collections_capture_services"),
            (_source_wiring.ServiceWiringMixin, "ensure_collections_capture_services"),
            (_source_wiring.ServiceWiringMixin, "_deferred_wire_collections_capture_services"),
            (_source_wiring.ServiceWiringMixin, "_activate_collections_capture_authority"),
            (_source_wiring.ServiceWiringMixin, "_shutdown_collections_capture_runtime"),
            (_source_wiring.ServiceWiringMixin, "_reconcile_collections_capture_startup"),
        )
    )
    vars(_source_wiring).update({'_COLLECTIONS_SETUP_SOURCE': _COLLECTIONS_SETUP_SOURCE})

    _helper_namespace = globals()
    _helper_entries = tuple((_helper_namespace, name) for name in ('_resolve_collections_media_reference', '_resolve_collections_note_reference', '_extract_collections_article', '_capture_deferred_collections_setup', '_require_collections_setup_current', '_build_deferred_collections_capture_parts', '_initialize_deferred_collections_capture', '_retire_deferred_collections_capture', '_collections_capture_scope', '_collections_capture_parts', '_local_collections_capture_service', '_publish_collections_capture_parts')) + ((_helper_namespace, "_source_wiring"), (_helper_namespace, "__package__"), (vars(_source_wiring), "_collections_wiring"))
    _helper_entries += tuple((CollectionsServiceWiringMixin, name) for name in ('_wire_collections_capture_services', '_reset_collections_capture_services', 'ensure_collections_capture_services', '_deferred_wire_collections_capture_services', '_activate_collections_capture_authority', '_reconcile_collections_capture_startup', '_shutdown_collections_capture_runtime'))
    _helper_rows = _record_collections_setup_source(_helper_entries)[1]
    _COLLECTIONS_SETUP_SOURCE = (_COLLECTIONS_SETUP_SOURCE[0], _COLLECTIONS_SETUP_SOURCE[1] + _helper_rows)
    vars(_source_wiring).update({'_COLLECTIONS_SETUP_SOURCE': _COLLECTIONS_SETUP_SOURCE})
    vars(_source_wiring).update({
        '_COLLECTIONS_APP_METHODS': _COLLECTIONS_APP_METHODS,
        '_COLLECTIONS_CAPTURE_RESULT_FIELDS': _COLLECTIONS_CAPTURE_RESULT_FIELDS,
        '_COLLECTIONS_RUN_PREPARATION': _COLLECTIONS_RUN_PREPARATION,
        '_COLLECTIONS_SETUP_SOURCE': _COLLECTIONS_SETUP_SOURCE,
        '_COLLECTIONS_SOURCE_CHECKER': _COLLECTIONS_SOURCE_CHECKER,
        '_COLLECTIONS_TASK': _COLLECTIONS_TASK,
    })


def _wire_collections_inputs(self):
    try:
        self.server_collections_feeds_service = (
            _source_wiring.ServerCollectionsFeedsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        )
    except ValueError:
        self.server_collections_feeds_service = _source_wiring.ServerCollectionsFeedsService(
            client=None,
            policy_enforcer=self.service_policy_enforcer,
        )
    self.collections_feeds_scope_service = _source_wiring.CollectionsFeedsScopeService(
        local_service=self.local_watchlists_service,
        server_service=self.server_collections_feeds_service,
        policy_enforcer=self.service_policy_enforcer,
    )


class CollectionsServiceWiringMixin:
    def _wire_library_collections_services(self) -> None:
        try:
            self.local_library_collections_db = _source_wiring.LibraryCollectionsDB(
                _source_wiring.get_library_collections_db_path(),
                _source_wiring.CLI_APP_CLIENT_ID,
            )
            self.local_library_collections_service = _source_wiring.LocalLibraryCollectionsService(
                self.local_library_collections_db,
            )
            self.library_collections_service = self.local_library_collections_service
        except Exception:
            _source_wiring.logger.opt(exception=True).warning(
                "Local Library Collections service unavailable during app wiring",
            )
            self.local_library_collections_db = None
            self.local_library_collections_service = None
            self.library_collections_service = None
        # dreams phase 1: the Dreams DB itself is NOT constructed here --
        # every Dreams import must stay off the `_ui_ready` module census
        # (ADR-097 boot ratchet; the budget is at its limit), and a disabled
        # Dreams must not create storage, so `get_dreams_db` builds it on
        # first use while `[dreams] enabled` is set.
        self.dreams_db = None
        self._dreams_db_lock = _source_wiring.threading.Lock()
        # guardian task 1: the same ruling for Guardian (ADR-204 contract
        # 1) -- no import here, no storage while `[guardian] enabled` is
        # off; `get_guardian_db` builds lazily and nothing calls it until
        # the Task 3 settings toggle.
        self.guardian_db = None
        self._guardian_db_lock = _source_wiring.threading.Lock()

    def _wire_collections_capture_services(self) -> None:
        """Compose the profile-owned Local capture authority and scope seam."""
        from tldw_chatbook.Library.collections_capture_repository import (
            CollectionsCaptureRepository,
        )
        from tldw_chatbook.Library.collections_capture_service import (
            CollectionsCaptureScopeService,
            LocalCollectionsCaptureService,
            build_local_capture_authority,
        )
        from tldw_chatbook.Library.collections_legacy_recovery import (
            LegacyCollectionsRecovery,
            LegacyCollectionsRecoveryError,
        )
        from tldw_chatbook.Library.collections_offline_store import (
            CollectionsOfflineStore,
        )
        _source_wiring.TldwCli._reset_collections_capture_services(self)
        self.collections_capture_scope_service = CollectionsCaptureScopeService(
            resolve_media_reference=_source_wiring.functools.partial(
                _source_wiring._resolve_collections_media_reference, self
            ),
            resolve_note_reference=_source_wiring.functools.partial(
                _source_wiring._resolve_collections_note_reference, self
            ),
        )
        try:
            database_path = _source_wiring.get_library_collections_db_path()
            database = getattr(self, "local_library_collections_db", None)
            if not isinstance(database, _source_wiring.LibraryCollectionsDB):
                database = _source_wiring.LibraryCollectionsDB(database_path, _source_wiring.CLI_APP_CLIENT_ID)
                self.local_library_collections_db = database
            data_root = _source_wiring.get_user_data_dir()
            authority = build_local_capture_authority(
                profile_id=str(data_root.resolve()),
                database_identity=str(database_path.resolve()),
            )
            repository = CollectionsCaptureRepository(
                database,
                authority_key=authority.key,
            )
            offline_store = CollectionsOfflineStore(
                repository,
                data_root=data_root,
                authority_fingerprint=authority.fingerprint,
            )
            legacy_recovery = LegacyCollectionsRecovery(database)
            try:
                legacy_recovery.list_collections(page=1, size=1)
                legacy_recovery_available = True
            except LegacyCollectionsRecoveryError:
                legacy_recovery_available = False
            service = LocalCollectionsCaptureService(
                authority,
                repository,
                offline_store=offline_store,
                extractor=_source_wiring._extract_collections_article,
                legacy_recovery_available=legacy_recovery_available,
            )
        except Exception:
            _source_wiring.logger.opt(exception=True).warning(
                "Local Collections capture service unavailable during app wiring"
            )
            return
        self.collections_capture_repository = repository
        self.collections_offline_store = offline_store
        self.collections_legacy_recovery_service = legacy_recovery
        self.local_collections_capture_authority = authority
        self.local_collections_capture_service = service
        _source_wiring.TldwCli._activate_collections_capture_authority(self)

    def _reset_collections_capture_services(self) -> None:
        """Install inert capture seams without importing their implementations."""
        self.collections_capture_repository = None
        self.collections_offline_store = None
        self.collections_legacy_recovery_service = None
        self.local_collections_capture_authority = None
        self.local_collections_capture_service = None
        self.server_collections_capture_authority = None
        self.server_collections_capture_service = None
        self.collections_capture_scope_service = _source_wiring._DeferredCollectionsCaptureScope(self)

    def ensure_collections_capture_services(self) -> Any | None:
        """Compose capture services once, after readiness or on first use."""
        scope = getattr(self, "collections_capture_scope_service", None)
        if scope is None or isinstance(scope, _source_wiring._DeferredCollectionsCaptureScope):
            _source_wiring.TldwCli._wire_collections_capture_services(self)
            scope = getattr(self, "collections_capture_scope_service", None)
            initializer = getattr(self, "_collections_capture_initializer_task", None)
            if (
                initializer is not None
                and not initializer.done()
                and not any(
                    getattr(self, flag, False)
                    for flag in (
                        "_collections_capture_initializer_closed", "_shutting_down", "_exit"
                    )
                )
                and getattr(self, "collections_capture_repository", None) is not None
            ):
                self._create_deferred_startup_task(
                    self._reconcile_collections_capture_startup(),
                    name="deferred_collections_capture_reconciliation",
                )
        return scope

    def _deferred_wire_collections_capture_services(self) -> None:
        """Prepare stock file-backed capture services away from the UI loop."""
        if any(vars(self).get(flag, False) for flag in (
            "_collections_capture_initializer_closed", "_shutting_down", "_exit",
        )):
            return
        task = vars(self).get("_collections_capture_initializer_task")
        if task is not None and not task.done():
            return
        request = _source_wiring._capture_deferred_collections_setup(self)
        if request is not None:
            self._collections_capture_setup = request
            task = _source_wiring._COLLECTIONS_TASK(
                _source_wiring._initialize_deferred_collections_capture(request),
                name="deferred_collections_capture_setup",
            )
            request.initializer = self._collections_capture_initializer_task = task
            self._deferred_startup_tasks.add(task)
            task.add_done_callback(self._deferred_startup_tasks.discard)
            return
        self.ensure_collections_capture_services()
        if getattr(self, "collections_capture_repository", None) is not None:
            self._create_deferred_startup_task(
                self._reconcile_collections_capture_startup(),
                name="deferred_collections_capture_reconciliation",
            )

    def _activate_collections_capture_authority(self) -> None:
        """Activate the capture owner selected by the committed runtime source."""
        scope = getattr(self, "collections_capture_scope_service", None)
        if scope is None:
            return
        runtime_policy = getattr(self, "runtime_policy", None)
        runtime_state = getattr(runtime_policy, "state", None)
        source = str(getattr(runtime_state, "active_source", "local") or "local")
        if source.strip().lower() != "server":
            authority = getattr(self, "local_collections_capture_authority", None)
            service = getattr(self, "local_collections_capture_service", None)
            if authority is not None and service is not None:
                scope.activate(authority, service)
            else:
                deactivate = getattr(scope, "deactivate", None)
                if callable(deactivate):
                    deactivate()
            return
        from tldw_chatbook.Library.collections_capture_service import (
            build_server_capture_authority,
        )
        from tldw_chatbook.Library.server_collections_capture_service import (
            ServerCollectionsCaptureService,
        )
        provider = getattr(self, "server_context_provider", None)
        try:
            context = provider.get_active_context()
            profile_id = str(getattr(context, "active_server_id", "") or "").strip()
            principal_id = _source_wiring.event_principal_id_from_active_context(context) or ""
            if not profile_id or not principal_id:
                raise ValueError("server_capture_identity_unavailable")
            authority = build_server_capture_authority(profile_id, principal_id)
            client = provider.build_client()
            token = str(getattr(context, "auth_token", "") or "")
            credential_fingerprint = _source_wiring.hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest()[:24]
            service = getattr(self, "server_collections_capture_service", None)
            prior_fingerprint = getattr(
                self, "_server_collections_credential_fingerprint", None
            )
            if not (
                isinstance(service, ServerCollectionsCaptureService)
                and service.authority == authority
                and service.client is client
                and prior_fingerprint == credential_fingerprint
            ):
                async def docs_info_provider() -> Any:
                    get_docs_info = getattr(client, "get_server_docs_info", None)
                    if not callable(get_docs_info):
                        raise RuntimeError("server_docs_info_unavailable")
                    return await get_docs_info()
                service = ServerCollectionsCaptureService(
                    authority,
                    client,
                    docs_info_provider=docs_info_provider,
                    credential_fingerprint=credential_fingerprint,
                )
                self.server_collections_capture_authority = authority
                self.server_collections_capture_service = service
                self._server_collections_credential_fingerprint = (
                    credential_fingerprint
                )
            scope.activate(authority, service)
        except Exception:
            _source_wiring.logger.opt(exception=True).warning(
                "Server Collections capture authority unavailable during activation"
            )
            deactivate = getattr(scope, "deactivate", None)
            if callable(deactivate):
                deactivate()

    async def _reconcile_collections_capture_startup(self) -> None:
        """Repair interrupted capture state with retained physical callbacks."""
        from .Chat.console_preparation_reads import run_preparation_read
        from .DB.base_db import operation_owned_connection
        def require_open() -> None:
            if any(getattr(self, flag, False) for flag in (
                "_collections_capture_initializer_closed", "_shutting_down", "_exit",
            )):
                raise _source_wiring.asyncio.CancelledError
        require_open()
        reads = getattr(self, "_collections_capture_reconciliation_reads", None)
        if reads is None:
            reads = self._collections_capture_reconciliation_reads = set()
        repository = getattr(self, "collections_capture_repository", None)
        if repository is not None:
            def interrupt_in_worker():
                with operation_owned_connection(repository.db):
                    return repository.interrupt_stale_extractions()
            await run_preparation_read(
                interrupt_in_worker, creator=self, session_id=None,
                reads=reads, require_current=require_open,
            )
        offline_store = getattr(self, "collections_offline_store", None)
        if offline_store is not None:
            def reconcile_in_worker():
                with operation_owned_connection(offline_store.repository.db):
                    return offline_store.reconcile_batch(limit=25)
            await run_preparation_read(
                reconcile_in_worker, creator=self, session_id=None,
                reads=reads, require_current=require_open,
            )

    async def _shutdown_collections_capture_runtime(self) -> None:
        """Fence capture authority and retire finite work before disposal."""
        self._collections_capture_initializer_closed = True
        failure: BaseException | None = None
        cancellation = None
        try:
            scope = getattr(self, "collections_capture_scope_service", None)
            if scope is not None and not isinstance(
                scope, _source_wiring._DeferredCollectionsCaptureScope
            ):
                deactivate = getattr(scope, "deactivate", None)
                if callable(deactivate):
                    deactivate()
        except BaseException as error:  # noqa: BLE001 - attempt all owned cleanup before exposing the original failure. Retire owned work before exposing callback failure.
            failure = error
        try:
            cancellation = await _source_wiring._retire_deferred_collections_capture(self)
        except BaseException as error:  # noqa: BLE001 - attempt all owned cleanup before exposing the original failure. initializer retirement
            if failure is None:
                failure = error
            elif error is not failure:
                BaseException.add_note(
                    failure, "Collections initializer retirement also failed: " + type(error).__name__
                )
        try:
            reads = getattr(self, "_collections_capture_reconciliation_reads", None)
            if reads:
                from .Chat.console_preparation_reads import drain_preparation_reads
                if await drain_preparation_reads(reads):
                    cancellation = cancellation or _source_wiring.asyncio.CancelledError()
        except BaseException as error:  # noqa: BLE001 - attempt all owned cleanup before exposing the original failure. reconciliation retirement
            if failure is None:
                failure = error
            elif error is not failure:
                BaseException.add_note(
                    failure, "Collections reconciliation retirement also failed: " + type(error).__name__
                )
        try:
            local_service = getattr(self, "local_collections_capture_service", None)
            if local_service is not None:
                await local_service.cancel_extractions()
        except BaseException as error:  # noqa: BLE001 - attempt all owned cleanup before exposing the original failure. extraction shutdown
            if failure is None:
                failure = error
            elif error is not failure:
                BaseException.add_note(
                    failure, "Collections extraction shutdown also failed: " + type(error).__name__
                )
        if failure is not None:
            raise failure
        if cancellation is not None:
            raise cancellation
