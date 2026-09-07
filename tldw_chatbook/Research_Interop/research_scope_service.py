"""Mode-aware routing for the research session/run parity seam."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any

from .research_normalizers import normalize_research_record


_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "research.sessions.server_crud",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_run_centric",
        "user_message": (
            "The current server API exposes deep research through run-centric /research/runs "
            "operations and does not support separate research session CRUD."
        ),
        "affected_action_ids": [
            "research.sessions.create.server",
            "research.sessions.list.server",
            "research.sessions.detail.server",
            "research.sessions.update.server",
            "research.sessions.delete.server",
        ],
    },
    {
        "operation_id": "research.runs.filtered_list.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": (
            "The current server API only supports limit-based research run listing; "
            "offset, session, and status filters are local-only."
        ),
        "affected_action_ids": ["research.runs.list.server"],
    },
    {
        "operation_id": "research.runs.delete.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server API does not support research run deletion.",
        "affected_action_ids": ["research.runs.delete.server"],
    },
]


class ResearchBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


def _is_async_callable(candidate: Any) -> bool:
    """True when calling ``candidate`` returns an awaitable OR an async generator.

    The async-generator arm is not decoration: ``LocalResearchService`` defines
    ``stream_run_events`` as ``async def ... yield``, and
    ``inspect.iscoroutinefunction`` is **False** for such a function. Routing it
    through a thread would hand ``stream_run_events``/``observe_run_events`` a
    coroutine wrapping the generator object, defeating their
    ``inspect.isasyncgen`` branch (TASK-21127).
    """

    def _async(target: Any) -> bool:
        return inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target)

    if _async(candidate):
        return True
    call = getattr(candidate, "__call__", None)
    return call is not None and _async(call)


_BACKEND_EXECUTOR: ThreadPoolExecutor | None = None
_BACKEND_EXECUTOR_LOCK = threading.Lock()


def _backend_executor() -> ThreadPoolExecutor:
    """The single thread every synchronous research-backend call runs on.

    ONE worker, deliberately (the TASK-21125 review's MAJOR-1 finding). Before
    the offload every scope call ran inline on the event loop, which silently
    serialised them against each other; a default-pool dispatch would hand that
    guarantee back and turn each read-check-write in the local service into a
    live lost update. Serialising on one thread restores the loop's ordering and
    keeps the whole latency win -- the point was getting OFF the loop, not going
    wide.
    """
    global _BACKEND_EXECUTOR
    if _BACKEND_EXECUTOR is None:
        with _BACKEND_EXECUTOR_LOCK:
            if _BACKEND_EXECUTOR is None:
                _BACKEND_EXECUTOR = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="research-backend",
                )
    return _BACKEND_EXECUTOR


async def _run_on_backend_thread(call: Any) -> Any:
    """Await ``call()`` on the shared single backend thread.

    Mirrors ``asyncio.to_thread``'s contextvar propagation, which
    ``run_in_executor`` does not do on its own.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    return await loop.run_in_executor(
        _backend_executor(), functools.partial(context.run, call)
    )


class _ThreadOffloadedBackend:
    """Runs a synchronous research backend's calls on the backend thread.

    TASK-21127: ``LocalResearchService`` is plain blocking SQLite and every
    scope method invoked it inline, so the window's 2 s auto-refresh poll and a
    bundle load both read and decoded on the Textual event loop (measured: a
    5.5 MB bundle costs ~15 ms of loop time even once the connection is held).
    Wrapping the backend here rather than at each of the ~25 call sites keeps
    every scope method's ``_maybe_await`` seam working unchanged: the wrapper
    returns a coroutine.

    Backends that are already asynchronous -- including async *generators* --
    pass straight through, so the server backend never pays a thread hop and
    ``stream_run_events`` keeps yielding.
    """

    __slots__ = ("_backend",)

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def offloads(self, name: str) -> bool:
        """Report whether ``name`` will be dispatched to the backend thread.

        A pass-through (already-async) attribute runs its body wherever the
        caller awaits it -- for an async *generator* over a synchronous
        backend, that is the event loop. Callers that can choose between two
        equivalent backend readers use this to prefer the offloaded one.

        Args:
            name: Backend attribute name to classify.

        Returns:
            True when calling ``name`` runs on the shared backend thread.
        """
        attribute = getattr(self._backend, name, None)
        return callable(attribute) and not _is_async_callable(attribute)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._backend, name)
        if not callable(attribute) or _is_async_callable(attribute):
            return attribute

        @functools.wraps(attribute)
        def _offloaded(*args: Any, **kwargs: Any) -> Any:
            return _run_on_backend_thread(functools.partial(attribute, *args, **kwargs))

        return _offloaded


class ResearchScopeService:
    """Route research operations to local or server backends with policy enforcement."""

    def __init__(
        self,
        *,
        local_service: Any,
        server_service: Any,
        policy_enforcer: Any = None,
        sync_scope_service: Any = None,
    ):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer
        self.sync_scope_service = sync_scope_service

    def _normalize_mode(self, mode: ResearchBackend | str | None) -> ResearchBackend:
        if mode is None:
            return ResearchBackend.LOCAL
        if isinstance(mode, ResearchBackend):
            return mode
        try:
            return ResearchBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid research backend: {mode}") from exc

    def _service_for_mode(self, mode: ResearchBackend) -> Any:
        """Return the backend for ``mode``, offloading synchronous calls.

        ``self.local_service`` / ``self.server_service`` keep their identity --
        callers (and the app-wiring tests) still see the objects that were
        passed in; only the dispatch path is wrapped (TASK-21127).
        """
        if mode == ResearchBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local research backend is unavailable.")
            return _ThreadOffloadedBackend(self.local_service)
        if self.server_service is None:
            raise ValueError("Server research backend is unavailable.")
        return _ThreadOffloadedBackend(self.server_service)

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    def _require_sync_scope_service(self) -> Any:
        if self.sync_scope_service is None:
            raise ValueError("Sync scope service is unavailable.")
        return self.sync_scope_service

    @staticmethod
    def _raise_server_sessions_unsupported() -> None:
        raise NotImplementedError(
            "The current server API does not expose separate research session CRUD; "
            "use server research run operations instead."
        )

    @staticmethod
    def _raise_server_run_list_filters_unsupported() -> None:
        raise NotImplementedError(
            "The current server API does not support filtered research run lists; "
            "only limit-based server run listing is available."
        )

    @staticmethod
    def _raise_server_run_delete_unsupported() -> None:
        raise NotImplementedError(
            "The current server API does not support research run deletion."
        )

    def _server_supports_method(self, method_name: str) -> bool:
        return callable(getattr(self.server_service, method_name, None))

    def _server_supports_sessions(self) -> bool:
        return all(
            self._server_supports_method(method_name)
            for method_name in (
                "create_session",
                "list_sessions",
                "get_session",
                "update_session",
                "delete_session",
            )
        )

    def _server_supports_run_delete(self) -> bool:
        return self._server_supports_method("delete_run") and bool(
            getattr(self.server_service, "supports_run_delete", True)
        )

    def _server_supports_filtered_run_list(self) -> bool:
        method = getattr(self.server_service, "list_runs", None)
        if not callable(method):
            return False
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return False
        parameters = signature.parameters
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return True
        return {"offset", "session_id", "status"}.issubset(parameters)

    @staticmethod
    def _action_id(resource: str, action: str, mode: ResearchBackend) -> str:
        return f"research.{resource}.{action}.{mode.value}"

    def list_unsupported_capabilities(
        self,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ResearchBackend.LOCAL:
            return []
        reports = [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]
        if self._server_supports_sessions():
            reports = [
                item
                for item in reports
                if item.get("operation_id") != "research.sessions.server_crud"
            ]
        if self._server_supports_filtered_run_list():
            reports = [
                item
                for item in reports
                if item.get("operation_id") != "research.runs.filtered_list.server"
            ]
        if self._server_supports_run_delete():
            reports = [
                item
                for item in reports
                if item.get("operation_id") != "research.runs.delete.server"
            ]
        return reports

    def record_sync_mirror_report(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        local_records: list[dict[str, Any]] | None = None,
        remote_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ResearchBackend.LOCAL:
            raise ValueError("Research mirror reports require server mode.")
        return self._require_sync_scope_service().record_dry_run_mirror_report(
            mode="server",
            domain="research",
            entity_type="research_run",
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            local_records=local_records or [],
            remote_records=remote_records or [],
        )

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _accepted_parameters(
        owner: type, method_name: str
    ) -> tuple[frozenset[str], bool]:
        """Parameter names of ``owner.method_name``, and whether it takes ``**kwargs``.

        Keyed on the CLASS, not the bound method: a bound method is a fresh
        object per attribute access, so caching on it would never hit and would
        pin every instance alive. TASK-21127: this ran an uncached
        ``inspect.signature`` on every scope call, including each 2 s
        auto-refresh tick.
        """
        method = getattr(owner, method_name)
        parameters = inspect.signature(method).parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        return frozenset(parameters), accepts_kwargs

    async def _call_service(
        self, service: Any, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        method = getattr(service, method_name)
        try:
            accepted, accepts_kwargs = self._accepted_parameters(
                type(getattr(service, "_backend", service)), method_name
            )
        except (AttributeError, TypeError, ValueError):
            # Callables the class lookup cannot describe (test doubles built
            # from instance attributes, C-implemented callables) fall back to
            # the uncached bound-method signature, as before.
            parameters = inspect.signature(method).parameters
            accepted = frozenset(parameters)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in accepted}
        return await self._maybe_await(method(*args, **kwargs))

    @staticmethod
    def _normalize_result(mode: ResearchBackend, kind: str, value: Any) -> Any:
        if isinstance(value, list):
            return [normalize_research_record(mode.value, kind, item) for item in value]
        if isinstance(value, dict):
            return normalize_research_record(mode.value, kind, value)
        return value

    def _normalize_bundle(
        self, mode: ResearchBackend, value: Any, *, run_id: str
    ) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.setdefault("backend", mode.value)
        if isinstance(payload.get("run"), dict):
            payload["run"] = normalize_research_record(
                mode.value, "run", payload["run"]
            )
        if isinstance(payload.get("artifacts"), list):
            artifacts = []
            for item in payload["artifacts"]:
                if not isinstance(item, dict):
                    artifacts.append(item)
                    continue
                artifact = dict(item)
                artifact.setdefault("run_id", run_id)
                artifacts.append(
                    normalize_research_record(mode.value, "artifact", artifact)
                )
            payload["artifacts"] = artifacts
        return payload

    async def list_sessions(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        action_id = self._action_id("sessions", "list", normalized_mode)
        self._enforce_policy(action_id)
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == ResearchBackend.SERVER and not callable(
            getattr(service, "list_sessions", None)
        ):
            self._raise_server_sessions_unsupported()
        result = await self._call_service(
            service,
            "list_sessions",
            limit=limit,
            offset=offset,
            status=status,
        )
        return self._normalize_result(normalized_mode, "session", result)

    async def create_session(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        title: str,
        query: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        action_id = self._action_id("sessions", "create", normalized_mode)
        self._enforce_policy(action_id)
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == ResearchBackend.SERVER and not callable(
            getattr(service, "create_session", None)
        ):
            self._raise_server_sessions_unsupported()
        result = await self._call_service(
            service,
            "create_session",
            title=title,
            query=query,
            **kwargs,
        )
        return self._normalize_result(normalized_mode, "session", result)

    async def get_session(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        session_id: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        action_id = self._action_id("sessions", "detail", normalized_mode)
        self._enforce_policy(action_id)
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == ResearchBackend.SERVER and not callable(
            getattr(service, "get_session", None)
        ):
            self._raise_server_sessions_unsupported()
        result = await self._call_service(service, "get_session", session_id)
        return self._normalize_result(normalized_mode, "session", result)

    async def update_session(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        session_id: str,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        action_id = self._action_id("sessions", "update", normalized_mode)
        self._enforce_policy(action_id)
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == ResearchBackend.SERVER and not callable(
            getattr(service, "update_session", None)
        ):
            self._raise_server_sessions_unsupported()
        result = await self._call_service(
            service,
            "update_session",
            session_id,
            expected_version=expected_version,
            **kwargs,
        )
        return self._normalize_result(normalized_mode, "session", result)

    async def delete_session(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        session_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        action_id = self._action_id("sessions", "delete", normalized_mode)
        self._enforce_policy(action_id)
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == ResearchBackend.SERVER and not callable(
            getattr(service, "delete_session", None)
        ):
            self._raise_server_sessions_unsupported()
        return bool(
            await self._call_service(
                service,
                "delete_session",
                session_id,
                expected_version=expected_version,
            )
        )

    async def launch_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        query: str | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ResearchBackend.SERVER and query is None:
            raise ValueError("query is required for server research runs")
        self._enforce_policy(self._action_id("runs", "launch", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode),
            "launch_run",
            session_id=session_id,
            query=query,
            **kwargs,
        )
        return self._normalize_result(normalized_mode, "run", result)

    async def create_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        query: str,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        action = "create" if normalized_mode == ResearchBackend.LOCAL else "launch"
        self._enforce_policy(self._action_id("runs", action, normalized_mode))
        service = self._service_for_mode(normalized_mode)
        method_name = "create_run" if hasattr(service, "create_run") else "launch_run"
        result = await self._call_service(service, method_name, query=query, **kwargs)
        return self._normalize_result(normalized_mode, "run", result)

    async def list_runs(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
        session_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        action_id = self._action_id("runs", "list", normalized_mode)
        self._enforce_policy(action_id)
        if (
            normalized_mode == ResearchBackend.SERVER
            and (
                offset not in (0, None) or session_id is not None or status is not None
            )
            and not self._server_supports_filtered_run_list()
        ):
            self._raise_server_run_list_filters_unsupported()
        result = await self._call_service(
            self._service_for_mode(normalized_mode),
            "list_runs",
            limit=limit,
            offset=offset,
            session_id=session_id,
            status=status,
        )
        return self._normalize_result(normalized_mode, "run", result)

    async def get_run(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "detail", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode), "get_run", run_id
        )
        return self._normalize_result(normalized_mode, "run", result)

    async def pause_run(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "update", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode), "pause_run", run_id
        )
        return self._normalize_result(normalized_mode, "run", result)

    async def resume_run(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        action = "launch" if normalized_mode == ResearchBackend.SERVER else "update"
        self._enforce_policy(self._action_id("runs", action, normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode), "resume_run", run_id
        )
        return self._normalize_result(normalized_mode, "run", result)

    async def cancel_run(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "update", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode), "cancel_run", run_id
        )
        return self._normalize_result(normalized_mode, "run", result)

    async def delete_run(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
        expected_version: int | None = None,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        action_id = self._action_id("runs", "delete", normalized_mode)
        self._enforce_policy(action_id)
        if (
            normalized_mode == ResearchBackend.SERVER
            and not self._server_supports_run_delete()
        ):
            self._raise_server_run_delete_unsupported()
        return bool(
            await self._call_service(
                self._service_for_mode(normalized_mode),
                "delete_run",
                run_id,
                expected_version=expected_version,
            )
        )

    async def observe_run_events(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
        after_id: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "observe", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        method_name = (
            "observe_run_events"
            if hasattr(service, "observe_run_events")
            else "list_run_events"
        )
        result = getattr(service, method_name)(run_id, after_id=after_id)
        if inspect.isasyncgen(result):
            items = [item async for item in result]
        else:
            items = list(await self._maybe_await(result))
        items = [
            {**item, "run_id": item.get("run_id") or run_id}
            if isinstance(item, dict)
            else item
            for item in items
        ]
        return self._normalize_result(normalized_mode, "event", items)

    async def stream_run_events(
        self,
        run_id: str,
        *,
        mode: ResearchBackend | str | None = None,
        after_id: int = 0,
    ):
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "observe", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        # TASK-21127 left this one path on the loop. `_ThreadOffloadedBackend`
        # passes async generators through unwrapped -- correct for the server
        # backend, but `LocalResearchService.stream_run_events` is an
        # `async def ... yield` whose whole body is
        # `for event in self.list_run_events(...)`, a blocking SQLite read. So
        # iterating the LOCAL stream executed the read on the event loop
        # (measured: `research-backend_0` for `observe_run_events` vs
        # `MainThread` here), on the Research window's 2 s refresh poll.
        #
        # A backend that exposes a SYNCHRONOUS `list_run_events` has no real
        # streaming to offer -- its generator is a snapshot loop over exactly
        # that call -- so take the offloaded reader instead and yield the same
        # items. This adds no thread and no transaction: it selects a method
        # the wrapper already dispatches to the one backend thread, which is
        # what `observe_run_events` above has always done for this backend.
        # A backend whose reader is async (the server) is untouched.
        if isinstance(service, _ThreadOffloadedBackend) and service.offloads(
            "list_run_events"
        ):
            method = getattr(service, "list_run_events")
        else:
            method = getattr(service, "stream_run_events", None)
            if method is None:
                method = getattr(service, "observe_run_events", None)
            if method is None:
                method = getattr(service, "list_run_events")
        result = method(run_id, after_id=after_id)
        if inspect.isasyncgen(result):
            async for item in result:
                yield item
            return
        for item in list(await self._maybe_await(result)):
            yield item

    async def get_bundle(
        self,
        *args: str,
        mode: ResearchBackend | str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        used_positional_run_id = bool(args)
        if args:
            if len(args) != 1:
                raise TypeError(
                    "get_bundle accepts a single run_id positional argument"
                )
            run_id = args[0]
        if run_id is None:
            raise TypeError("get_bundle requires run_id")
        normalized_mode = self._normalize_mode(mode)
        action = (
            "observe"
            if normalized_mode == ResearchBackend.SERVER and used_positional_run_id
            else "detail"
        )
        self._enforce_policy(self._action_id("runs", action, normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode), "get_bundle", run_id
        )
        return self._normalize_bundle(normalized_mode, result, run_id=run_id)

    async def get_artifact(
        self,
        *,
        mode: ResearchBackend | str | None = None,
        run_id: str,
        artifact_name: str,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "detail", normalized_mode))
        result = await self._call_service(
            self._service_for_mode(normalized_mode),
            "get_artifact",
            run_id,
            artifact_name,
        )
        if isinstance(result, dict):
            result = {**result, "run_id": result.get("run_id") or run_id}
        return (
            self._normalize_result(normalized_mode, "artifact", result)
            if result
            else result
        )

    async def patch_and_approve_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        mode: ResearchBackend | str | None = None,
        patch_payload: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("runs", "update", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "patch_and_approve_checkpoint", None)
        if method is None:
            raise NotImplementedError(
                "Research checkpoint approval is only available on supported server backends."
            )
        return await self._maybe_await(
            method(
                run_id,
                checkpoint_id,
                patch_payload=patch_payload,
            )
        )
