"""Scoped leases over the existing MCP client and plugin lifecycle owner."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

from tldw_chatbook.Plugins.admission import RunPluginSnapshot
from tldw_chatbook.Plugins.data_cleanup import root_ref
from tldw_chatbook.Plugins.service import PluginRunOwnership

from .activation import _guard


@dataclass(frozen=True)
class ConnectionAuthorityKey:
    installation_id: str
    revision_digest: str
    executable: str
    arguments: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    cwd: str | None
    endpoint: str
    configuration_digest: str
    credential_bindings: str
    session_isolation: str


@dataclass(frozen=True)
class OwnedMCPInvocation:
    ownership: ConnectionOwnership
    snapshot: RunPluginSnapshot
    component_id: str
    tool_mapping: dict | None = None
    check_permission: Callable[[], None] | None = None


owned_invocation: ContextVar[OwnedMCPInvocation | None] = ContextVar(
    "owned_mcp_invocation", default=None
)


@dataclass
class Request:
    snapshot: RunPluginSnapshot
    owner_id: str
    request_id: str
    accepting: bool = True
    dispatched: bool = False
    token: str | None = None
    task: asyncio.Task | None = None
    outcome: str = "pending"


@dataclass
class Connection:
    key: ConnectionAuthorityKey
    connection_id: str
    owners: set[str] = field(default_factory=set)
    leases: dict[str, str] = field(default_factory=dict)
    requests: dict[str, Request] = field(default_factory=dict)
    launch: asyncio.Task | None = None
    entered_client: bool = False
    close: asyncio.Task | None = None
    session: object | None = None
    profile: object | None = None
    leases_settled: bool = False
    prepare_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ConnectionOwnership:
    """Retain exact host custody; a detached waiter never proves remote completion.

    All async methods run on the client loop. Synchronous revocation callbacks
    only close acceptance and submit cleanup to that loop.
    """

    def __init__(self, *, plugin_service, local_service, loop=None):
        self.plugins = plugin_service
        self.local = local_service
        self.loop = loop or asyncio.get_running_loop()
        self.connections: dict[str, Connection] = {}
        self._lock = threading.RLock()
        self.local.connection_ownership = self

    def attach(self, key: ConnectionAuthorityKey, owner_id: str) -> str:
        """Attach one identity, sharing only explicitly qualified equal authority."""
        if not owner_id or key.session_isolation not in {
            "separate",
            "request_independent",
        }:
            raise ValueError("mcp_connection_authority_invalid")
        with self._lock:
            for connection in self.connections.values():
                if (
                    connection.key == key
                    and connection.close is None
                    and not connection.leases_settled
                    and (
                        key.session_isolation == "request_independent"
                        or owner_id in connection.owners
                    )
                ):
                    connection.owners.add(owner_id)
                    return connection.connection_id
            identity = "owned-" + uuid4().hex
            self.connections[identity] = Connection(key, identity, {owner_id})
            return identity

    def is_connected(self, profile_id: str, owner_id: str | None = None) -> bool:
        """Report an actual ready session; this metadata grants no authority.

        owner_id is ``run_id + ':' + (workspace_id or '')``. When supplied,
        the exact scope must already be attached; a compatible other owner's
        connection alone returns False. Callers must still check the snapshot.
        """
        sessions = getattr(self.local.client, "sessions", {})
        return any(
            connection.profile is not None
            and connection.profile.profile_id == profile_id
            and connection.owners
            and (owner_id is None or owner_id in connection.owners)
            and connection.session is not None
            and sessions.get(connection.connection_id) is connection.session
            and not self._session_unavailable(connection.session)
            and connection.close is None
            and not connection.leases_settled
            for connection in tuple(self.connections.values())
        )

    @staticmethod
    def _session_unavailable(session) -> bool:
        return bool(
            getattr(session, "_closed", False)
            or getattr(session, "_reader_unavailable", False)
            or getattr(session, "_cleanup_complete", False)
        )

    def bind_request(
        self, connection_id: str, request_id: str, snapshot: RunPluginSnapshot
    ) -> None:
        """Bind request identity before dispatch, preserving its original scope."""
        self.plugins.fences.check_snapshot(snapshot)
        with self._lock:
            connection = self.connections[connection_id]
            owner_id = self._owner_id(snapshot)
            if (
                (snapshot.installation_id, snapshot.revision_digest)
                != (connection.key.installation_id, connection.key.revision_digest)
                or owner_id not in connection.owners
                or request_id in connection.requests
            ):
                raise PermissionError("mcp_request_owner_changed")
            connection.requests[request_id] = Request(snapshot, owner_id, request_id)

    @staticmethod
    def _owner_id(snapshot):
        return snapshot.run_id + ":" + (snapshot.workspace_id or "")

    def _schedule(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    async def _reserve(self, connection, snapshot, *, request=None):
        component = connection.profile.plugin_owner["component_id"]
        root = connection.profile.plugin_owner["data_root"]
        roots = ()
        if root is not None:
            rows = json.loads(snapshot.data_roots_json)
            matching = [
                row
                for row in rows
                if all(row.get(key) == value for key, value in root.items())
            ]
            if len(matching) != 1:
                raise PermissionError("plugin_data_binding_changed")
            roots = (root_ref(matching[0]),)
        identity = (
            request.request_id if request is not None else self._owner_id(snapshot)
        )

        def cancel():
            if request is not None:
                request.accepting = False
            return asyncio.wrap_future(
                self._schedule(
                    self.detach(connection.connection_id, self._owner_id(snapshot))
                )
            )

        def reserve():
            coordinator = self.plugins._coordinator
            self.plugins._admission.check(snapshot, component)
            token = coordinator.owner.reserve_launch(
                "mcp:" + identity,
                snapshot.installation_id,
                snapshot.workspace_id,
                snapshot.revision_digest,
                roots=roots,
                root_coverage="known" if roots else "qualified_none",
            )
            record = PluginRunOwnership(
                snapshot.installation_id,
                snapshot.workspace_id,
                snapshot.revision_digest,
                "mcp:" + identity,
                snapshot.run_id,
                None,
                token,
                cancel,
                threading.Event(),
                generations=snapshot.generations,
                component_ceiling=snapshot.selection,
            )
            try:
                if roots:
                    coordinator.root_usage.retain_cancel(token, cancel)
                with self.plugins.fences.live_lock:
                    self.plugins.fences.check_snapshot(snapshot)
                    self.plugins.fences.runs["mcp:" + token] = record
            except BaseException:
                coordinator.owner.settle_process(token, True)
                raise
            return token

        reservation = asyncio.create_task(self.plugins._call(reserve))
        try:
            return await asyncio.shield(reservation)
        except asyncio.CancelledError:
            # The worker may already have reserved custody. Join that exact
            # operation and settle its known pre-launch/pre-request grant.
            token = await reservation
            await self._settle(token)
            raise

    async def _publish(self, token, provenance, *, idle=False):
        def publish():
            owner = self.plugins._coordinator.owner
            owner.publish_process(token, provenance)
            if idle:
                owner.set_process_kind(token, "idle_connection")

        await self.plugins._call(publish)

    async def _settle(self, token):
        def settle():
            self.plugins._coordinator.owner.settle_process(token, True)
            with self.plugins.fences.live_lock:
                record = self.plugins.fences.runs.pop("mcp:" + token, None)
                if record is not None:
                    record.completed.set()

        await self.plugins._call(settle, terminal=True)

    async def connect(
        self, snapshot: RunPluginSnapshot, component_id: str, profile_id: str
    ) -> str:
        """Explicit reviewed connect reserves all root/process grants before launch."""
        await self.plugins.check_mcp_snapshot(snapshot, component_id)
        profile = self.local.store.get_profile(profile_id)
        if profile is None or profile.plugin_owner is None:
            raise PermissionError("plugin_owned_profile_required")
        mappings = [
            row
            for row in json.loads(snapshot.mappings_json)
            if row["kind"] == "connection"
            and row["component_id"] == component_id
            and row["target_reference"] == "local:" + profile_id
        ]
        if len(mappings) != 1:
            raise PermissionError("plugin_connection_not_reviewed")
        mapping = mappings[0]
        key = ConnectionAuthorityKey(
            snapshot.installation_id,
            snapshot.revision_digest,
            profile.command,
            profile.args,
            tuple(sorted(profile.plugin_owner["environment"].items())),
            profile.cwd,
            profile.url,
            mapping["configuration_digest"],
            json.dumps(mapping["credential_bindings"], sort_keys=True),
            profile.plugin_owner["session_isolation"],
        )
        owner_id = self._owner_id(snapshot)
        from tldw_chatbook.Agents.mcp_tool_provider import (
            check_mcp_invocation_policies,
            current_mcp_invocation_policies,
        )

        check_mcp_invocation_policies()
        if current_mcp_invocation_policies():
            # Hook invocation may reuse only this exact scope's existing lease.
            # It may not attach a sibling, reserve idle custody or launch.
            for connection in tuple(self.connections.values()):
                if (
                    connection.key == key
                    and owner_id in connection.owners
                    and connection.session is not None
                    and self.local.client.sessions.get(connection.connection_id)
                    is connection.session
                    and not self._session_unavailable(connection.session)
                    and connection.close is None
                    and not connection.leases_settled
                ):
                    return connection.connection_id
            raise PermissionError("hook_mcp_connection_required")
        for prior in tuple(self.connections.values()):
            if (
                prior.key == key
                and prior.session is not None
                and self._session_unavailable(prior.session)
            ):
                await self._close(prior)
                if not prior.leases_settled:
                    raise PermissionError("mcp_connection_cleanup_pending")
        identity = self.attach(key, owner_id)
        connection = self.connections[identity]
        connection.profile = profile
        try:
            async with connection.prepare_lock:
                if owner_id not in connection.leases:
                    token = await self._reserve(connection, snapshot)
                    connection.leases[owner_id] = token
                if connection.launch is None:
                    with _guard.worker_isolation():
                        connection.launch = asyncio.create_task(
                            self._launch(connection, snapshot)
                        )
            await asyncio.shield(connection.launch)
            async with connection.prepare_lock:
                token = connection.leases[owner_id]
                await self._publish_attached(connection, token)
            await self.plugins.check_mcp_snapshot(snapshot, component_id)
            return identity
        except BaseException:
            await self.detach(identity, owner_id)
            raise

    async def _publish_attached(self, connection, token):
        def is_pending():
            return (
                self.plugins._coordinator.registry._connection.execute(
                    "SELECT state FROM processes WHERE token=?", (token,)
                ).fetchone()[0]
                == "pending"
            )

        if await self.plugins._call(is_pending):
            process = getattr(connection.session, "process", None)
            await self._publish(
                token,
                {
                    "mcp_connection_id": connection.connection_id,
                    "pid": getattr(process, "pid", None),
                },
                idle=True,
            )

    async def _launch(self, connection, snapshot):
        from .local_store import TransportProfile

        profile = connection.profile
        self.local._require_allowed("mcp.external_profiles.launch.local")
        client = self.local._get_client()
        if profile.credential_reference is not None:
            client.credential_service = self.local._credentials()
        launch = TransportProfile(
            profile_id=connection.connection_id,
            transport=profile.transport,
            protocol_version=profile.protocol_version,
            command=profile.command,
            args=profile.args,
            cwd=profile.cwd,
            package_headers=profile.plugin_owner["literal_headers"],
            env=(
                profile.plugin_owner["environment"]
                if profile.transport == "stdio"
                else None
            ),
            url=profile.url,
            development_loopback=profile.development_loopback,
            credential_reference=profile.credential_reference,
            credential_generation=profile.credential_generation,
        )
        try:
            self.plugins.fences.check_snapshot(snapshot)
            # This host-owned transition distinguishes a rejected setup from
            # any operation that reached MCPClient, including cancelled awaits.
            connection.entered_client = True
            if (
                await client.connect_profile(
                    launch,
                    observe_connection=lambda session: setattr(
                        connection, "session", session
                    ),
                )
                is not True
            ):
                raise RuntimeError(
                    client.connection_diagnostics.get(
                        connection.connection_id, "mcp_owned_connect_failed"
                    )
                )
            connection.session = client.sessions[connection.connection_id]
            snapshot = await client.describe_server(connection.connection_id)
            self.local.store.save_discovery_snapshot(profile.profile_id, snapshot)
        except BaseException:
            # No inference from missing rows: cleanup reports through the actual
            # client, including its pending-launch and retained-close owner.
            await self._close(connection)
            raise

    async def invoke(
        self, context, profile_id, tool_name, arguments, call, *, automatic_work=None
    ):
        if context.check_permission is not None:
            context.check_permission()
        await self.plugins.check_mcp_snapshot(context.snapshot, context.component_id)
        mapping = context.tool_mapping
        if (
            mapping is None
            or mapping["target_reference"] != "local:" + profile_id + "::" + tool_name
        ):
            raise PermissionError("plugin_tool_not_reviewed")
        identity = await self.connect(
            context.snapshot, context.component_id, profile_id
        )
        connection = self.connections[identity]
        request_id = uuid4().hex
        self.bind_request(identity, request_id, context.snapshot)
        request = connection.requests[request_id]
        try:
            request.token = await self._reserve(
                connection, context.snapshot, request=request
            )
            from tldw_chatbook.Agents.mcp_tool_provider import (
                check_mcp_invocation_policies,
                current_mcp_invocation_policies,
            )

            policies = current_mcp_invocation_policies()
            if policies and policies[-1].on_owned_request is not None:
                policies[-1].on_owned_request(self.plugins, request.token)
            check_mcp_invocation_policies()
            await self._publish(
                request.token,
                {
                    "mcp_connection_id": identity,
                    "mcp_request_id": request_id,
                    "outcome": "pending",
                },
            )
            await self.plugins.check_mcp_snapshot(
                context.snapshot, context.component_id
            )
            if not request.accepting:
                raise PermissionError("plugin_request_revoked")
            if context.check_permission is not None:
                context.check_permission()
        except BaseException:
            request.outcome = "not_started"
            if request.token is not None:
                await self._settle(request.token)
            connection.requests.pop(request_id, None)
            raise

        async def perform():
            try:
                if not request.accepting:
                    raise PermissionError("plugin_request_revoked")
                if context.check_permission is not None:
                    context.check_permission()
                if automatic_work is not None:
                    automatic_work.check()
                check_mcp_invocation_policies()
                if current_mcp_invocation_policies() and (
                    connection.close is not None
                    or connection.leases_settled
                    or self.local.client.sessions.get(identity)
                    is not connection.session
                    or self._session_unavailable(connection.session)
                ):
                    raise PermissionError("hook_mcp_connection_required")
                request.dispatched = True
                result = await call(identity, tool_name, arguments)
                request.outcome = result.dispatch_state
                if request.outcome in {"settled", "not_started"}:
                    await self._settle(request.token)
                else:
                    await self._uncertain(request)
                return result
            except BaseException:
                if request.dispatched:
                    await self._uncertain(request)
                else:
                    request.outcome = "not_started"
                    await self._settle(request.token)
                raise
            finally:
                if request.outcome in {"settled", "not_started"}:
                    connection.requests.pop(request_id, None)

        with _guard.worker_isolation():
            request.task = asyncio.create_task(perform())
        try:
            result = await asyncio.shield(request.task)
            await self.plugins.check_mcp_snapshot(
                context.snapshot, context.component_id
            )
            if context.check_permission is not None:
                context.check_permission()
            if not request.accepting:
                raise PermissionError("plugin_result_revoked")
            return result
        except asyncio.CancelledError:
            request.accepting = False
            await self._uncertain(request)
            raise

    async def _uncertain(self, request):
        if not request.dispatched or request.outcome in {"settled", "not_started"}:
            return
        request.outcome = "uncertain"
        if request.token is not None:
            await self.plugins._call(
                lambda: self.plugins._coordinator.owner.mark_request_uncertain(
                    request.token, request.request_id
                ),
                terminal=True,
            )

    async def detach(self, connection_id: str, owner_id: str) -> None:
        connection = self.connections.get(connection_id)
        if connection is None:
            return
        connection.owners.discard(owner_id)
        for request in tuple(connection.requests.values()):
            if request.owner_id == owner_id and request.outcome not in {
                "settled",
                "not_started",
            }:
                request.accepting = False
                await self._uncertain(request)
        if not connection.owners:
            if connection.close is None or connection.close.done():
                connection.close = asyncio.create_task(self._close(connection))
            await asyncio.shield(connection.close)

    async def _close(self, connection):
        if (
            connection.launch is not None
            and connection.launch is not asyncio.current_task()
            and not connection.launch.done()
        ):
            try:
                await asyncio.shield(connection.launch)
            except Exception:  # noqa: BLE001, S110
                pass
        async with connection.cleanup_lock:
            await self._close_locked(connection)

    async def _close_locked(self, connection):
        if connection.leases_settled:
            return
        closed = not connection.entered_client or bool(
            connection.session is not None
            and getattr(connection.session, "_cleanup_complete", False)
        )
        if not closed:
            closed = await self.local._get_client().disconnect_from_server(
                connection.connection_id
            )
        if not closed:
            return
        connection.owners.clear()
        for request in tuple(connection.requests.values()):
            if (
                request.outcome not in {"settled", "not_started"}
                and connection.profile.transport == "stdio"
            ):
                if request.task is not None:
                    await asyncio.gather(request.task, return_exceptions=True)
                if request.outcome not in {"settled", "not_started"}:
                    await self._settle(request.token)
                    request.outcome = "settled"
                    connection.requests.pop(request.request_id, None)
        for token in connection.leases.values():
            await self._settle(token)
        connection.leases_settled = True
        if not connection.requests:
            self.connections.pop(connection.connection_id, None)

    async def close_idle_revision(
        self, installation_id: str, revision_digest: str
    ) -> None:
        for connection in tuple(self.connections.values()):
            if (connection.key.installation_id, connection.key.revision_digest) != (
                installation_id,
                revision_digest,
            ):
                continue
            if any(
                request.outcome not in {"settled", "not_started"}
                for request in connection.requests.values()
            ):
                continue
            for owner_id in tuple(connection.owners):
                await self.detach(connection.connection_id, owner_id)
