"""Private originating-turn custody for ordinary native Console execution."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from inspect import getattr_static
from threading import Lock
from types import FunctionType

from tldw_chatbook.Agents.execution_capacity import CapacityRefused, WorkOrigin
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint, WorktreeScope

_current = ContextVar("ordinary_console_tree", default=None)


@dataclass
class _Member:
    settlement: object
    run_id: str | None = None
    row_started: bool = False
    steps: dict = field(default_factory=dict)
    terminal: tuple | None = None
    capture_failed: bool = False
    host_returned: bool = False
    durable: bool = False
    drained: bool = False


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def current_ordinary_tree(controller=None):
    tree = _current.get()
    if tree is not None and (controller is None or tree.controller is controller):
        return tree
    return None


def _bootstrap_personal_service(app):
    from tldw_chatbook.Personal_Context.bootstrap import (
        bootstrap_personal_context_service,
    )

    service = bootstrap_personal_context_service()
    tree = current_ordinary_tree()
    if tree is not None:
        tree.publish_personal_service(app, service)
    return service


def finish_ordinary_submission(controller, task):
    tree = getattr(controller, "_ordinary_submissions", {}).pop(task, None)
    if tree is not None:
        try:
            tree.finish()
        finally:
            if _current.get() is tree:
                _current.set(None)


@contextmanager
def bind_ordinary_tree(tree):
    token = _current.set(tree)
    try:
        if tree is None:
            yield
        else:
            with tree.owner.activate():
                yield
    finally:
        _current.reset(token)


class _OrdinaryTurn:
    def __init__(
        self, controller, session, configuration, origin, source, agent_eligible
    ):
        self.controller, self.session = controller, session
        self.runtime, self.store, self.app = (
            controller._hooks_v2_runtime,
            controller.store,
            controller.app,
        )
        self.source, self.configuration = source, configuration
        self.original_configuration = configuration
        self.agent_eligible = agent_eligible
        self.persistence = self.store.persistence
        self.incarnation = session.incarnation_id
        self.binding_revision = session.conversation_binding_revision
        self.workspace = session.workspace_id
        self.conversation = session.persisted_conversation_id
        self.launch_key = self.conversation or session.id
        self.origin = WorkOrigin.AUTOMATIC if origin.automatic else WorkOrigin.MANUAL
        self.capacity, self.activity = (
            self.runtime.execution_capacity,
            self.runtime.worktree_activity,
        )
        self.owner = self.claim = self.footprint = None
        self.reference = None
        self.snapshot_sources = []
        self.ssh_sources = {}
        self.local_maximum = {}
        self.local_source_values = ()
        self.raw_projection = None
        self.local_providers = {}
        self.other_maximum = {}
        self.catalogs = {}
        self.mcp_entries = {}
        self.mcp_source_maximum = None
        self.goal_catalog_permits = lambda _source, _name: True
        self.virtual_commands = frozenset()
        self.virtual_providers = {}
        self.builtin_maximum = {}
        self.builtin_classes = {}
        self.builtin_providers = {}
        self.catalog_permits = lambda _source, _name: False
        self.library_maximum = frozenset()
        self.library_factory = None
        self.library_services = {}
        self.library_providers = {}
        self.personal_getter = None
        self.personal_service = None
        self.personal_native = False
        self.personal_source_values = ()
        self.members = {}
        self.runs = {}
        self.future_events = {}
        self.stop_notification = None
        self.scratch_lease = None
        self.source_scopes = 0
        self.settled = False
        self.retired = Future()
        self.lock = Lock()
        self.sealed = False
        self.primary_hosts = set()
        self.primary_finishing = False

    def check(self):
        if (
            self.controller._hooks_v2_runtime is not self.runtime
            or self.runtime._chat_controller is not self.controller
            or self.runtime._chat_store is not self.store
            or self.controller.store is not self.store
            or self.store.persistence is not self.persistence
            or self.store._sessions.get(self.session.id) is not self.session
            or self.session.incarnation_id != self.incarnation
            or self.session.conversation_binding_revision != self.binding_revision
            or self.session.workspace_id != self.workspace
            or self.session.persisted_conversation_id != self.conversation
            or self.runtime._execution_capacity is not self.capacity
            or self.runtime._worktree_activity is not self.activity
        ):
            raise RuntimeError("original ordinary turn source changed")
        if self.source is not None:
            self.source.check_memory()

    def check_local_sources(self):
        self.check()
        if self.raw_projection is not None:
            original, was_reachable, values = self.raw_projection
            current, reachable, sources = _raw_shell_metadata(self)
            if (
                current is not original
                or reachable
                and not was_reachable
                or values != sources
            ):
                raise RuntimeError("ordinary raw shell maximum changed")
        for obj, name, value, _code in self.local_source_values:
            current = getattr_static(obj, name, None)
            if current is not value:
                raise RuntimeError("ordinary local resource maximum changed")

    def check_personal_source(self, app=None):
        self.check()
        if app is not None and app is not self.app:
            raise RuntimeError("ordinary personal context app changed")
        if (
            getattr_static(self.app, "_personal_context_service", None)
            is not self.personal_service
        ):
            raise RuntimeError("ordinary personal context cached source changed")
        if any(
            getattr_static(obj, name, None) is not value
            for obj, name, value in self.personal_source_values
        ):
            raise RuntimeError("ordinary personal context source changed")

    def publish_personal_service(self, app, service):
        self.check_personal_source(app)
        native = _capture_personal_service(self, service)
        if not native and not self.footprint.opaque:
            raise RuntimeError("ordinary personal context publication widened")
        self.personal_service = service

    def bind_local_provider(self, provider, names):
        """Validate the one real composition against the original native maximum."""
        from tldw_chatbook.Agents import local_tool_provider as local
        from tldw_chatbook.MCP.permission_store import resolve_effective_state_by_key
        from tldw_chatbook.Tools.remote_root_types import is_remote, local_root_path

        self.check_local_sources()
        if type(provider) is not local._ORDINARY_NATIVE_PROVIDER:
            if not self.footprint.opaque:
                raise RuntimeError("unknown ordinary local provider")
            return
        payload = (
            self.source.permission.load()
            if self.source.permission is not None
            else None
        )
        if payload is None and not self.footprint.opaque:
            raise RuntimeError("ordinary local permission source unavailable")
        reachable = tuple(
            name
            for name in names
            if self.catalog_permits("local", name)
            and (
                name == "ask_user"
                or payload is None
                or resolve_effective_state_by_key(
                    payload,
                    local.LOCAL_SERVER_KEY,
                    name,
                    profile_id=self.configuration.tool_policy_profile_id,
                ).state
                != "deny"
            )
        )
        specs = provider._specs
        for name in reachable:
            spec = specs[name]
            if (
                name not in self.local_maximum
                or spec.worktree_effect != self.local_maximum[name]
            ):
                raise RuntimeError("ordinary local catalog widened")
        if not self.footprint.opaque:
            for authority in (provider._admitted_roots or {}).values():
                if is_remote(authority.root):
                    raise RuntimeError("ordinary local root widened to SSH")
                root = local_root_path(authority.root, site="ordinary maximum")
                self._require_covered_root(root, write=authority.allow_write)
            self._require_covered_root(provider._root, write=False)
            if provider._spill_dir is not None:
                self._require_covered_root(provider._spill_dir.parent, write=True)
        self.local_providers[provider] = _capture_local_provider_sources(
            provider, reachable
        )

    def bind_builtin_provider(self, provider, names):
        from tldw_chatbook.Agents import tool_catalog as catalog
        from tldw_chatbook.MCP.permission_store import (
            BUILTIN_TOOL_SERVER_KEY,
            resolve_effective_state_by_key,
        )

        self.check_local_sources()
        if type(provider) is not catalog._NATIVE_ORDINARY_BUILTIN_PROVIDER:
            if not self.footprint.opaque:
                raise RuntimeError("unknown ordinary builtin provider")
            return
        payload = (
            self.source.permission.load()
            if self.source is not None and self.source.permission is not None
            else None
        )
        retained = {}
        for name in names:
            if not self.catalog_permits("builtin", name):
                continue
            tool = provider._tools[name]
            if name not in catalog.ALWAYS_ON_BUILTIN_NAMES and (
                payload is not None
                and (
                    payload.get("kill_switch", False)
                    or resolve_effective_state_by_key(
                        payload,
                        BUILTIN_TOOL_SERVER_KEY,
                        name,
                        profile_id=self.configuration.tool_policy_profile_id,
                    ).state
                    == "deny"
                )
                or self.session.ephemeral
                and name in {"create_note", "update_note", "write_file"}
            ):
                continue
            if (
                name not in self.builtin_maximum
                or type(tool) is not self.builtin_classes.get(name)
            ) and (self.builtin_classes or not self.footprint.opaque):
                raise RuntimeError("ordinary builtin catalog widened")
            retained[name] = tool
        if not self.footprint.opaque:
            self._require_covered_root(provider._sandbox_root, write=True)
        self.builtin_providers[provider] = retained

    def check_builtin_call(self, provider, name):
        self.check_local_sources()
        tool = self.builtin_providers[provider].get(name)
        if tool is None or provider._tools.get(name) is not tool:
            raise RuntimeError("ordinary builtin call outside original maximum")

    def _require_covered_root(self, root, *, write):
        chain = capture_directory_chain(root)
        scopes = (
            self.footprint.writes
            if write
            else self.footprint.reads + self.footprint.writes
        )
        if not any(
            scope.root == chain and scope.path == chain.canonical_root
            for scope in scopes
        ):
            raise RuntimeError("ordinary local root outside original maximum")

    def check_local_call(self, provider, name):
        self.check_local_sources()
        retained = self.local_providers.get(provider)
        if retained is None:
            if self.footprint.opaque:
                return
            raise RuntimeError("ordinary local provider was not composed")
        _check_local_provider_sources(provider, retained)
        original = dict(retained[0]).get(name)
        if original is None or provider._specs.get(name) is not original:
            raise RuntimeError("ordinary local call outside original maximum")
        if not self.footprint.opaque and provider._agent_roots:
            raise RuntimeError("ordinary local child root outside maximum")

    def bind_virtual_provider(self, provider):
        from tldw_chatbook.Agents.virtual_cli_provider import _NATIVE_VIRTUAL_CLASS
        from tldw_chatbook.Tools.remote_root_types import is_remote, local_root_path
        from tldw_chatbook.Tools.workspace_tool_executor import (
            _NATIVE_ORDINARY_EXECUTOR_CLASS,
        )

        if type(provider) is not _NATIVE_VIRTUAL_CLASS:
            if not self.footprint.opaque:
                raise RuntimeError("ordinary virtual source widened")
            return
        roots = tuple((provider._admitted_roots or {}).items())
        registries = (
            (None, provider._registry),
            *provider._registries_by_alias.items(),
        )
        held = []
        for alias, registry in registries:
            executor = getattr_static(registry, "_workspace_executor", None)
            root = getattr_static(registry, "_root", None)
            remote = getattr_static(registry, "_remote_root", None)
            held.append((alias, registry, executor, root, remote))
            if remote is None and executor is not None:
                if (
                    type(executor) is not _NATIVE_ORDINARY_EXECUTOR_CLASS
                    and not self.footprint.opaque
                ):
                    raise RuntimeError("ordinary virtual local executor widened")
                for name in (
                    "_workspace_root",
                    "_authority_chain",
                    "_user_exclusion_paths",
                ):
                    value = getattr_static(executor, name, None)
                    self.local_source_values += ((executor, name, value, None),)
                if not self.footprint.opaque:
                    self._require_covered_root(
                        getattr_static(executor, "_workspace_root", None), write=False
                    )
            if not self.footprint.opaque:
                if remote is not None:
                    raise RuntimeError("ordinary virtual root widened to SSH")
                self._require_covered_root(root, write=False)
        for _alias, authority in roots:
            if not self.footprint.opaque:
                if is_remote(authority.root):
                    raise RuntimeError("ordinary virtual authority widened to SSH")
                self._require_covered_root(
                    local_root_path(authority.root, site="ordinary virtual maximum"),
                    write=False,
                )
        self.virtual_providers[provider] = roots, tuple(held)

    def check_virtual_call(self, provider, args=None):
        self.check_local_sources()
        if args is not None and args.get("command") not in self.virtual_commands:
            raise RuntimeError("ordinary virtual command outside original maximum")
        retained = self.virtual_providers.get(provider)
        if retained is None:
            if self.footprint.opaque:
                return
            raise RuntimeError("ordinary virtual provider was not composed")
        roots, registries = retained
        current = getattr_static(provider, "_admitted_roots", None) or {}
        aliases = getattr_static(provider, "_registries_by_alias", None)
        if (
            type(current) is not dict
            or type(aliases) is not dict
            or len(current) != len(roots)
            or any(current.get(alias) is not authority for alias, authority in roots)
        ):
            raise RuntimeError("ordinary virtual root source changed")
        for alias, registry, executor, root, remote in registries:
            selected = (
                getattr_static(provider, "_registry", None)
                if alias is None
                else aliases.get(alias)
            )
            if selected is not registry or any(
                getattr_static(registry, name, None) is not original
                for name, original in (
                    ("_workspace_executor", executor),
                    ("_root", root),
                    ("_remote_root", remote),
                )
            ):
                raise RuntimeError("ordinary virtual executor source changed")

    def check_library_factory(self):
        self.check_local_sources()
        if not self.footprint.opaque and (
            self.controller._library_provider_factory is not self.library_factory
            or self.runtime._library_provider_factory is not self.library_factory
            or self.library_factory.runtime is not self.runtime
            or self.runtime._app is not self.app
        ):
            raise RuntimeError("ordinary Library construction resource changed")

    def bind_library_provider(self, provider, names):
        self.check_library_factory()
        if any(
            self.catalog_permits("library", name) and name not in self.library_maximum
            for name in names
        ):
            raise RuntimeError("ordinary Library catalog widened")
        service = getattr_static(provider, "_service", None)
        if not self.footprint.opaque:
            from tldw_chatbook.Agents import library_tool_provider
            from tldw_chatbook.Library import local_library_tool_service

            if (
                type(provider)
                is not library_tool_provider._NATIVE_ORDINARY_LIBRARY_CLASS
                or type(service)
                is not local_library_tool_service._NATIVE_ORDINARY_LIBRARY_CLASS
            ):
                raise RuntimeError("ordinary Library source outside bounded maximum")
            for name, original in self.library_services.items():
                if getattr_static(service, name, None) is not original:
                    raise RuntimeError("ordinary Library backend changed")
        self.library_providers[provider] = (service, frozenset(names))

    def check_library_call(self, provider, name):
        self.check_library_factory()
        service, names = self.library_providers[provider]
        if (
            name not in names
            or getattr_static(provider, "_service", None) is not service
        ):
            raise RuntimeError("ordinary Library dispatch source changed")
        if not self.footprint.opaque and any(
            getattr_static(service, name, None) is not original
            for name, original in self.library_services.items()
        ):
            raise RuntimeError("ordinary Library backend changed")

    def bind_catalog(self, registry, names):
        """Retain the original final dispatch records, after normal narrowing."""
        self.check_local_sources()
        records = {}
        for name in names:
            resolved = registry.resolve_owner_for_name(name)
            if resolved is None:
                # Runtime tools such as spawn are owned by AgentService, not
                # supplied as a provider callback by this registry.
                continue
            tool_id, provider = resolved
            source = registry._source_for(tool_id)
            if not self.goal_catalog_permits(source, name):
                # Goal.narrow_config consumes this same registry after normal
                # composition. Unreachable entries grant no receiving record.
                continue
            if provider in self.local_providers:
                self.check_local_call(provider, name)
            elif provider in self.builtin_providers:
                self.check_builtin_call(provider, name)
            elif provider in self.library_providers:
                self.check_library_call(provider, name)
            elif source == "virtual_cli":
                if name not in self.other_maximum.get(source, ()):
                    raise RuntimeError("ordinary virtual maximum widened")
                self.bind_virtual_provider(provider)
            elif source == "mcp":
                from tldw_chatbook.Agents.mcp_tool_provider import (
                    _CONTROLLER_COMPOSE_FACTORY,
                )
                from tldw_chatbook.MCP.permission_store import definition_hash

                from . import console_agent_bridge as bridge

                if type(provider) is not bridge._NATIVE_ORDINARY_COLLISION_PROVIDER:
                    raise RuntimeError("ordinary MCP collision view changed")
                underlying = getattr_static(provider, "_provider", None)
                allowed = getattr_static(provider, "_allowed_names", None)
                table = getattr_static(underlying, "_entry_by_llm_name", None)
                entry = (
                    table.get(tool_id)
                    if type(underlying) is _CONTROLLER_COMPOSE_FACTORY
                    and type(table) is dict
                    else None
                )
                if (
                    entry is None
                    or self.session.ephemeral
                    or name not in allowed
                    or self.configuration.mcp_definition_maximum.get(entry[0].tool_id)
                    != definition_hash(entry[0].description, entry[0].input_schema)
                ):
                    raise RuntimeError("ordinary MCP maximum widened")
                self.check_mcp_definition(entry[0], name)
                self.mcp_entries[provider, tool_id] = (
                    underlying,
                    allowed,
                    table,
                    entry,
                )
            elif source in {"skill", "plugin"}:
                if self.session.ephemeral or name not in {
                    row.get("name")
                    for row in self.configuration.skill_context_maximum.get(
                        "available_skills", ()
                    )
                }:
                    raise RuntimeError("ordinary skill maximum widened")
            elif name not in self.other_maximum.get(source, ()):
                raise RuntimeError("ordinary provider maximum widened")
            records[name] = (tool_id, provider, source)
        self.catalogs[registry] = records

    def check_catalog_call(self, registry, name, provider):
        self.check_local_sources()
        records = self.catalogs.get(registry)
        if records is None:
            raise RuntimeError("ordinary catalog was not originally composed")
        record = records.get(name)
        if record is None or record[1] is not provider:
            raise RuntimeError("ordinary dispatch outside original catalog")
        tool_id, _provider, source = record
        current = registry.resolve_owner_for_name(name)
        if current is None or current[0] != tool_id or current[1] is not provider:
            raise RuntimeError("ordinary catalog owner changed")
        if source == "mcp":
            from tldw_chatbook.MCP.permission_store import definition_hash

            underlying, allowed, table, entry = self.mcp_entries[provider, tool_id]
            if (
                getattr_static(provider, "_provider", None) is not underlying
                or getattr_static(provider, "_allowed_names", None) is not allowed
                or getattr_static(underlying, "_entry_by_llm_name", None) is not table
                or table.get(tool_id) is not entry
                or self.configuration.mcp_definition_maximum.get(entry[0].tool_id)
                != definition_hash(entry[0].description, entry[0].input_schema)
            ):
                raise RuntimeError("ordinary MCP definition source changed")
            self.check_mcp_definition(entry[0], name)
        if (
            source == "raw_shell"
            and getattr_static(provider, "_runtime", None) is not self.raw_projection[0]
        ):
            raise RuntimeError("ordinary raw runtime changed")
        if source == "personal-context":
            self.check_personal_source()
        if source == "virtual_cli":
            self.check_virtual_call(provider)

    def check_mcp_definition(self, tool, name):
        if self.mcp_source_maximum is None:
            if not self.footprint.opaque:
                raise RuntimeError("ordinary MCP source maximum unavailable")
            return
        original = self.mcp_source_maximum.get(tool.tool_id)
        if original != (tool.tool_id, tool.server_key, tool.name, name):
            raise RuntimeError("ordinary MCP name source changed")

    def has_admitted_members(self):
        with self.lock:
            owners = tuple(self.members)
        return any(owner.is_admitted for owner in owners)

    def persisted(self, store, session, identity):
        """Consume only the original store's committed publication callback."""
        if store is not self.store or session is not self.session:
            return
        if (
            self.conversation is not None
            and self.conversation != identity.conversation_id
        ):
            raise RuntimeError("ordinary turn conversation rebound")
        if session.persisted_conversation_id != identity.conversation_id:
            raise RuntimeError("ordinary committed publication unavailable")
        previous = self.conversation
        self.conversation = identity.conversation_id
        try:
            self.check()
        except BaseException:
            self.conversation = previous
            raise

    def admit(self, footprint):
        self.check()
        if self.runtime._disposed or self.controller._disposed:
            raise CapacityRefused("runtime_closed")
        with self.activity.prelaunch(footprint=footprint) as claim:
            if claim is None:
                raise CapacityRefused("worktree_activity_busy")
            owner = self.capacity.begin_execution(
                origin=self.origin, conversation_id=self.launch_key
            )
            try:
                self.check()
                claim.bind_owner(owner)
            except BaseException:
                owner.finish_root()
                raise
            self.owner, self.claim, self.footprint = owner, claim, footprint
            self.members[owner] = _Member(claim)
        if self.configuration.scratch_space is not None:
            self.scratch_lease = self.controller._scratch_spaces.lease(
                self.configuration.scratch_space
            )
            try:
                self.scratch_lease.__enter__()
            except BaseException:
                self.scratch_lease = None
                owner.finish_root()
                claim.settle(durable=True)
                raise
        owner.on_drained(lambda proven: self._drained(owner, proven))
        with self.runtime._execution_capacity_lock:
            closing = self.runtime._disposed
            if not closing:
                self.runtime._ordinary_turns.add(self)
        if closing:
            # No effect producer can see this unpublished admission. Retire its
            # original private lease/claim outside runtime bookkeeping.
            self.sealed = True
            self.host_returned(owner)
            owner.finish_root()
            raise CapacityRefused("runtime_closed")

    @contextmanager
    def retain_source_scope(self, *, _cleanup_state=None):
        """Reuse the original live scratch lease, including after tombstoning."""
        with self.lock:
            if (
                self.scratch_lease is None
                and self.configuration.scratch_space is not None
            ):
                raise RuntimeError("original ordinary scratch lease retired")
            self.source_scopes += 1
        try:
            yield
        finally:
            if _cleanup_state is None or _cleanup_state[0]:
                with self.lock:
                    self.source_scopes -= 1
                self._retire_scratch()

    def _drained(self, owner, proven):
        with self.lock:
            self.members[owner].drained = proven is True
        self._retire_scratch()

    def _retire_scratch(self):
        lease = None
        settlements = ()
        elected = False
        with self.lock:
            if (
                not self.settled
                and self.sealed
                and not self.source_scopes
                and all(
                    member.host_returned and member.durable and member.drained
                    for member in self.members.values()
                )
            ):
                self.settled = True
                elected = True
                lease, self.scratch_lease = self.scratch_lease, None
                settlements = tuple(
                    member.settlement for member in self.members.values()
                )
        if lease is not None:
            lease.__exit__(None, None, None)
        for settlement in settlements:
            settlement.settle(durable=True)
        if elected:
            # Election is not proof: lease cleanup and every original
            # settlement callback must have actually returned first.
            self.retired.set_result(None)
            with self.runtime._execution_capacity_lock:
                self.runtime._ordinary_turns.discard(self)
            self.runtime._wake_first_send()

    def borrow(self, runtime, *, opaque=True):
        self.check()
        if (
            runtime is not self.runtime
            or self.sealed
            or runtime._disposed
            or self.controller._disposed
        ):
            raise CapacityRefused("ordinary_turn_closed")
        if opaque and not self.footprint.opaque:
            raise RuntimeError("ordinary maximum does not cover this effect")
        self.activity.validate_claim(
            self.claim, owner=self.owner, footprint=self.footprint
        )
        return self.owner

    def attach(self, owner):
        self.check()
        if self.sealed or self.runtime._disposed or self.controller._disposed:
            raise CapacityRefused("ordinary_turn_closed")
        for source in self.snapshot_sources:
            try:
                source.prepare_survivor()
            except CapacityRefused:
                # Capture remains best effort. Its original finalizer will
                # disclose the unavailable survivor admission if needed.
                pass
        settlement = self.claim.attach_child(owner)
        with self.lock:
            self.members[owner] = _Member(settlement)
        owner.on_drained(lambda proven: self._drained(owner, proven))

    def row_started(self, owner, run_id):
        with self.lock:
            member = self.members[owner]
            member.row_started = True
            member.run_id = run_id
            self.runs[run_id] = member

    def required_steps(self, run_id, records):
        """Retain the actual serializer output, before the original write attempt."""
        with self.lock:
            member = self.runs.get(run_id)
            if member is None:
                return
            if records is None:
                member.capture_failed = True
            else:
                member.steps.update((index, _json(record)) for index, record in records)

    def required_terminal(self, run_id, status, result, budget_tokens):
        with self.lock:
            member = self.runs.get(run_id)
            if member is not None:
                member.terminal = (status, result, budget_tokens)

    def retain_event(self, scope):
        from tldw_chatbook.Agents.execution_capacity import current_execution_owner

        owner = current_execution_owner()
        with self.lock:
            owner = owner if owner in self.members else self.owner
            self.future_events[scope] = owner

    def consume_event(self, scope):
        with self.lock:
            self.future_events.pop(scope, None)

    def close_unused_events(self, owner):
        with self.lock:
            scopes = tuple(
                scope for scope, member in self.future_events.items() if member is owner
            )
            for scope in scopes:
                self.future_events.pop(scope)
        for scope in scopes:
            scope.close()

    def host_returned(self, owner):
        with self.lock:
            member = self.members.get(owner)
            if member is None or member.host_returned:
                return
            member.host_returned = True
        proven = not member.row_started
        if (
            member.row_started
            and not member.capture_failed
            and member.terminal is not None
        ):
            # Use the original database and its canonical decoded records. Never
            # infer durability from a swallowed exception or a UI run outcome.
            try:
                with self.source.activate():
                    row = self.source.databases[0].get_run(member.run_id)
                status, result, tokens = member.terminal
                stored = {step["index"]: _json(step) for step in row["steps"]}
                proven = (
                    row["status"] == status
                    and _json(row.get("result")) == _json(result)
                    and (
                        tokens is None
                        or _json(row.get("budget_tokens")) == _json(tokens)
                    )
                    and bool(member.steps)
                    and all(
                        stored.get(index) == value
                        for index, value in member.steps.items()
                    )
                )
            except Exception:  # noqa: BLE001 - uncertainty retains the original claim
                proven = False
        with self.lock:
            member.durable = proven
        self._retire_scratch()

    def finish(self):
        with self.lock:
            self.sealed = True
        self._finish_primary_if_returned()

    def retain_primary_host(self):
        """Retain the original native primary callback before its existing queue."""
        self.check()
        token = object()
        with self.lock:
            if self.sealed:
                raise CapacityRefused("ordinary_turn_closed")
            self.primary_hosts.add(token)
        return token

    def primary_host_returned(self, token):
        with self.lock:
            if token not in self.primary_hosts:
                return
            self.primary_hosts.remove(token)
        self._finish_primary_if_returned()

    def _finish_primary_if_returned(self):
        with self.lock:
            if not self.sealed or self.primary_hosts or self.primary_finishing:
                return
            self.primary_finishing = True
        if self.owner is not None:
            with (
                self.source.activate()
                if self.source is not None
                else self.retain_source_scope()
            ):
                if self.stop_notification is not None:
                    _engine, custody = self.stop_notification
                    if custody is not None and not custody.notification_claimed:
                        custody.abandon_unstarted()
                self.close_unused_events(self.owner)
                self.host_returned(self.owner)
                self.owner.finish_root()


async def admit_ordinary_turn(
    controller, session, configuration, origin, text, candidates, *, agent_eligible
):
    """Select original sources before acquiring the complete ordinary footprint."""
    from tldw_chatbook.Agents.activation import worker_guard

    from .console_agent_sources import current_agent_sources
    from .console_preparation_reads import run_preparation_read
    from .console_reference_preparation import _ReferenceRead
    from .console_workflow_runs import native_workflow_qualified

    if controller._hooks_v2_runtime is None or not native_workflow_qualified():
        return None, configuration
    existing = current_ordinary_tree(controller)
    if existing is not None:
        existing.check()
        return existing, configuration
    source = current_agent_sources(controller)
    tree = _OrdinaryTurn(
        controller, session, configuration, origin, source, agent_eligible
    )
    from .console_configuration_preparation import (
        original_mcp_capture_matches,
        original_skill_capture_preparation,
    )

    tree.mcp_metadata_witnessed = (
        source is not None
        and source.mcp is None
        and getattr_static(tree.app, "unified_mcp_service", None) is None
    ) or original_mcp_capture_matches(controller, session, configuration)
    pending_mcp = configuration.mcp_capture_pending
    pending_review = configuration.change_review_capture_pending
    pending_skill = configuration.skill_capture_pending
    original_configuration = configuration
    if (
        pending_mcp or pending_review or pending_skill
    ) and not original_mcp_capture_matches(
        controller, session, configuration, pending=pending_mcp
    ):
        raise RuntimeError("Original MCP capture source is unavailable; send again.")
    tree.skill_preparation = (
        original_skill_capture_preparation()
        if original_mcp_capture_matches(
            controller,
            session,
            configuration,
            pending=pending_mcp,
            require_complete=False,
        )
        else None
    )

    def capture():
        tree.check()
        if candidates:
            tree.reference = _ReferenceRead(
                controller, session, origin, text, candidates
            )
            tree.reference.capture_authority()
            tree.reference.ordinary_tree = tree
        footprint = _capture_maximum(tree)
        tree.check_local_sources()
        tree.admit(footprint)
        return tree

    try:
        await run_preparation_read(
            worker_guard(controller._agent_bridge)(capture),
            creator=controller,
            session_id=session.id,
            reads=controller._preparation_reads,
            observers=(tree.runtime._preparation_reads,),
            require_current=tree.check,
        )
    finally:
        # Cancellation is delivered after the real capture returns. Its genuine
        # admission still belongs to this submission's original outer finally.
        with tree.runtime._execution_capacity_lock:
            admitted = tree in tree.runtime._ordinary_turns
        if admitted:
            if source is not None:
                source.ordinary_tree = tree
            pending = getattr(controller, "_ordinary_submissions", None)
            if pending is None:
                pending = controller._ordinary_submissions = {}
            pending[asyncio.current_task()] = tree
    _current.set(tree)
    if pending_skill:
        from contextlib import nullcontext
        from dataclasses import replace

        from .console_configuration_preparation import (
            capture_console_skill_catalog_owned,
            finish_console_skill_catalog_owned,
            pending_skill_capture,
        )

        captured = pending_skill_capture(configuration)
        if captured is None:
            raise RuntimeError(
                "Original skill preparation source is unavailable; send again."
            )
        deferred, original_request_configuration = captured
        task = asyncio.current_task()
        records = [
            record
            for record in tree.runtime._turn_custody.values()
            if record.task is task and record.request is not None
        ]
        if (
            len(records) != 1
            or records[0].request.configuration is not original_request_configuration
        ):
            raise RuntimeError(
                "Original skill preparation custody changed; send again."
            )
        record = records[0]

        def current_skill_source():
            tree.check()
            if (
                tree.runtime._turn_custody.get(record.turn_id) is not record
                or record.task is not task
                or record.request is None
                or record.request.configuration is not original_request_configuration
                or not original_mcp_capture_matches(
                    controller,
                    session,
                    original_configuration,
                    pending=pending_mcp,
                    require_complete=False,
                )
            ):
                raise RuntimeError(
                    "Original skill preparation custody changed; send again."
                )

        operation = tree.borrow(tree.runtime).reserve_tool()
        try:
            with (
                tree.source.activate()
                if tree.source is not None
                else bind_ordinary_tree(tree)
            ):
                current_skill_source()
                catalog = await capture_console_skill_catalog_owned(
                    tree.app,
                    tree.store,
                    controller,
                    session_id=session.id,
                    turn_id=record.turn_id,
                    skill_workspace_id=deferred.selection.skill_workspace_id,
                    preparation=deferred.preparation,
                    reads=controller._preparation_reads,
                    observers=(tree.runtime._preparation_reads,),
                    require_current=current_skill_source,
                )
                prepared = await finish_console_skill_catalog_owned(
                    catalog,
                    reads=controller._preparation_reads,
                    observers=(tree.runtime._preparation_reads,),
                    require_current=current_skill_source,
                )
                current_skill_source()
                maximum = dict(prepared.maximum)
                if "plugin_turn_id" in configuration.skill_context_maximum:
                    maximum["plugin_turn_id"] = configuration.skill_context_maximum[
                        "plugin_turn_id"
                    ]
                configuration = replace(
                    configuration,
                    skill_context_maximum=maximum,
                    skill_capture_pending=False,
                )
                tree.configuration = configuration
                if tree.source is not None:
                    tree.source.configuration = configuration
        finally:
            with tree.source.activate() if tree.source is not None else nullcontext():
                operation.finish()
    if pending_mcp:
        from contextlib import nullcontext
        from dataclasses import replace

        from tldw_chatbook.MCP.console_snapshot import (
            _capture_console_definition_selection,
            standard_console_sources,
        )

        from .console_configuration_capture import (
            CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
            _capture_mcp_definition_selection,
        )

        operation = tree.borrow(tree.runtime).reserve_tool()
        try:
            with (
                tree.source.activate()
                if tree.source is not None
                else bind_ordinary_tree(tree)
            ):
                service = getattr(tree.app, "unified_mcp_service", None)
                if not original_mcp_capture_matches(
                    controller, session, original_configuration, pending=True
                ):
                    raise RuntimeError("Original MCP capture source changed.")
                if standard_console_sources(service):
                    selected = await _capture_console_definition_selection(
                        service,
                        CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
                        profile_id=configuration.tool_policy_profile_id,
                        persona_policy_rules=configuration.persona_policy_rules,
                        _admitted=True,
                    )
                else:
                    selected = _capture_mcp_definition_selection(
                        tree.app,
                        profile_id=configuration.tool_policy_profile_id,
                        persona_policy_rules=configuration.persona_policy_rules,
                        _admitted=True,
                    )
                if tree.source is not None:
                    from .console_configuration_preparation import (
                        consume_native_mcp_permission_publication,
                    )

                    consume_native_mcp_permission_publication(tree.source)
                if not original_mcp_capture_matches(
                    controller, session, original_configuration, pending=True
                ):
                    raise RuntimeError("Original MCP capture source changed.")
                tree.check_local_sources()
                configuration = replace(
                    configuration,
                    mcp_tool_maximum=frozenset(selected.definitions),
                    mcp_definition_maximum=selected.definitions,
                    mcp_definition_sources=selected.sources,
                    mcp_definition_complete=selected.complete,
                    mcp_capture_pending=False,
                )
                tree.configuration = configuration
                if tree.source is not None:
                    tree.source.configuration = configuration
                rows = {row[0]: row for row in selected.sources}
                if selected.complete and set(rows) == set(selected.definitions):
                    tree.mcp_source_maximum = rows
                await run_preparation_read(
                    worker_guard(controller._agent_bridge)(
                        lambda: _capture_maximum(tree)
                    ),
                    creator=controller,
                    session_id=session.id,
                    reads=controller._preparation_reads,
                    observers=(tree.runtime._preparation_reads,),
                    require_current=tree.check,
                )
        finally:
            with tree.source.activate() if tree.source is not None else nullcontext():
                operation.finish()
    from .console_turn_context import capture_change_review_admission

    review = await run_preparation_read(
        worker_guard(controller._agent_bridge)(
            lambda: capture_change_review_admission(tree.app, tree.workspace)
        ),
        creator=controller,
        session_id=session.id,
        reads=controller._preparation_reads,
        observers=(tree.runtime._preparation_reads,),
        require_current=tree.check,
    )
    if pending_review:
        from dataclasses import replace

        if not tree.footprint.opaque or not original_mcp_capture_matches(
            controller, session, original_configuration, pending=pending_mcp
        ):
            raise RuntimeError("Original Change Review capture source changed.")
        tree.check_local_sources()
        roots, aliases, skipped = review
        configuration = replace(
            configuration,
            workspace_roots=roots,
            change_review_root_aliases=aliases,
            change_review_skipped_roots=skipped,
            change_review_capture_pending=False,
        )
        tree.configuration = configuration
        if tree.source is not None:
            tree.source.configuration = configuration
    return tree, configuration


async def run_ordinary_preparation(controller, callback, *, ssh=None, opaque=False):
    """Retain one actual original preparation worker before its existing enqueue."""
    from threading import Event

    from tldw_chatbook.Agents.activation import worker_guard

    from .console_preparation_reads import run_preparation_read

    tree = current_ordinary_tree(controller)
    if tree is None:
        return await asyncio.to_thread(callback)
    tree.check_local_sources()
    operation = tree.borrow(tree.runtime, opaque=opaque).reserve_tool()
    entered = Event()

    def owned():
        entered.set()
        with tree.source.activate():
            if ssh is not None:
                with ssh.activate(operation):
                    return callback()
            try:
                return callback()
            finally:
                operation.finish()

    try:
        return await run_preparation_read(
            worker_guard(controller._agent_bridge)(owned),
            creator=controller,
            session_id=tree.session.id,
            reads=controller._preparation_reads,
            observers=(tree.runtime._preparation_reads,),
            require_current=tree.check,
        )
    finally:
        if not entered.is_set():
            with tree.source.activate():
                operation.finish()


def _permission_payload(tree):
    """Read only witnessed native metadata before admission; unknown waits."""
    permission = tree.source.permission if tree.source is not None else None
    if permission is None or tree.owner is None and not tree.mcp_metadata_witnessed:
        return None
    return permission.load()


def _capture_maximum(tree):
    """Project stock declared effects; live composition may only narrow them."""
    from tldw_chatbook.Agents.persona_policy import (
        evaluate_tool_policy,
        parse_persona_policy_from_rules,
    )
    from tldw_chatbook.Workspaces.change_review_consent import ChangeReviewState

    configuration = tree.configuration
    policy = parse_persona_policy_from_rules(configuration.persona_policy_rules)
    advertised = lambda name: (
        evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name=name).advertised
    )
    reads, writes = [], []
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work

    automatic = current_automatic_work()
    goal = automatic.goal if automatic is not None else None
    goal_scope = goal.goal.request.tool_scope if goal is not None else None
    catalog_allowed = lambda source, name: (
        goal_scope is None or f"{source}:{name}" in goal_scope.catalog_tools
    )
    runtime_allowed = lambda name: goal is None or goal.permits_runtime(name)
    tree.catalog_permits = lambda source, name: (
        advertised(name) and catalog_allowed(source, name)
    )
    tree.goal_catalog_permits = catalog_allowed
    sources = {row[0]: row for row in configuration.mcp_definition_sources}
    if tree.mcp_metadata_witnessed and set(sources) == set(
        configuration.mcp_definition_maximum
    ):
        tree.mcp_source_maximum = sources
    opaque = bool(
        configuration.mcp_capture_pending
        or configuration.skill_capture_pending
        or (
            tree.agent_eligible
            and tree.source is not None
            and tree.source.permission is not None
            and not tree.mcp_metadata_witnessed
        )
        or (
            tree.agent_eligible
            and not tree.session.ephemeral
            and (
                (
                    goal_scope is None
                    or any(name.startswith("mcp:") for name in goal_scope.catalog_tools)
                )
                if tree.mcp_source_maximum is None
                else any(
                    tree.catalog_permits("mcp", row[3]) for row in sources.values()
                )
            )
        )
    )
    scratch = configuration.scratch_space
    if scratch is not None:
        scope = WorktreeScope(scratch.root, True, capture_directory_chain(scratch.root))
        reads.append(scope)
        writes.append(scope)
    if tree.source is not None and tree.source.root is not None:
        writes.append(WorktreeScope(tree.source.root, True, tree.source.root_identity))
    if tree.reference is not None:
        opaque |= tree.reference.opaque
        reads.extend(
            WorktreeScope(chain.canonical_root, True, chain)
            for chain in tree.reference.authority.chains
        )
    service = getattr_static(tree.app, "change_review_consent_service", None)
    tree.local_source_values += (
        (tree.app, "change_review_consent_service", service, None),
    )
    if service is not None:
        from tldw_chatbook.Workspaces.change_review_consent import (
            ChangeReviewConsentService,
        )

        if (
            type(service) is ChangeReviewConsentService
            and not service._custom_capability_reader
        ):
            status = service.status(tree.workspace)
            opaque |= (
                status.capability.state is ChangeReviewState.ENABLED
                and status.consent.state is ChangeReviewState.ENABLED
            )
        else:
            opaque = True
    engine = tree.runtime.get_hooks_v2(tree.session.id)
    opaque |= bool(engine is not None and engine.enabled and engine.definitions)
    permissions = tree.runtime._hook_permissions
    if permissions is not None:
        from tldw_chatbook.Agents.run_hooks import load_hooks_config

        review, _targets = permissions.v2_configuration()
        hooks = load_hooks_config(
            {"hooks": review.config.section} if review.config.section_present else {}
        )
        opaque |= bool(hooks.enabled and (hooks.hooks or hooks.v2_handlers))
    opaque |= any(
        item.get("plugin_owned")
        for item in configuration.skill_context_maximum.get("available_skills", ())
    )
    if tree.agent_eligible:
        opaque |= _capture_personal_source(tree)
        opaque |= _capture_other_providers(tree)
        budget = configuration.tool_configuration.get("agent_run_budget_maximum")
        opaque |= bool(
            budget is not None
            and budget.max_subagents
            and advertised("spawn_subagent")
            and runtime_allowed("spawn_subagent")
        )
        bridge = tree.controller._agent_bridge
        if bridge._skills_service is not None:
            import os

            opaque |= bool(os.name != "nt" and runtime_allowed("run_skill_script"))
            opaque |= bool(runtime_allowed("install_skill"))
        if configuration.tool_configuration.get("local_tools_enabled"):
            from tldw_chatbook.Agents import local_tool_provider as local
            from tldw_chatbook.MCP.permission_store import (
                resolve_effective_state_by_key,
            )

            mcp = getattr(tree.app, "unified_mcp_service", None)
            if mcp is not None and not configuration.mcp_capture_pending:
                payload = _permission_payload(tree)
                if payload is None or not payload.get("kill_switch", False):
                    tree.local_maximum = {
                        name: effect
                        for name, effect in local._NATIVE_WORKTREE_EFFECTS.items()
                        if advertised(name)
                        and catalog_allowed("local", name)
                        and (
                            name == "ask_user"
                            or payload is None
                            or resolve_effective_state_by_key(
                                payload,
                                local.LOCAL_SERVER_KEY,
                                name,
                                profile_id=configuration.tool_policy_profile_id,
                            ).state
                            != "deny"
                        )
                    }
                    opaque |= "opaque" in tree.local_maximum.values()
                    raw = getattr_static(tree.app, "raw_cli_runtime", None)
                    if (
                        raw is not None
                        and advertised("shell_exec")
                        and catalog_allowed("raw_shell", "shell_exec")
                    ):
                        original, raw_reachable, raw_values = _raw_shell_metadata(tree)
                        tree.raw_projection = (original, raw_reachable, raw_values)
                        opaque |= raw_values[:1] == ("unknown",)
                        if raw_reachable and (
                            payload is None
                            or resolve_effective_state_by_key(
                                payload,
                                local.LOCAL_SERVER_KEY,
                                "shell_exec",
                                profile_id=configuration.tool_policy_profile_id,
                            ).state
                            != "deny"
                        ):
                            opaque = True
    if tree.agent_eligible:
        opaque |= _capture_library_maximum(tree)
        builtin, builtin_unknown = _capture_builtin_maximum(
            tree, advertised, catalog_allowed
        )
        opaque |= builtin_unknown
        # Root enumeration is a preparation producer even without local tools.
        # Goal selects its finite bindings before any remote recovery scheduling.
        authority = configuration.project_authority
        goal_bindings = (
            None
            if goal is None
            else {
                reference.binding_id: (index == 0 and reference.access == "rw")
                for index, reference in enumerate(
                    (goal.goal.request.binding, *goal.goal.request.source_bindings)
                )
            }
        )
        writable = "write" in tree.local_maximum.values() or "write" in builtin.values()
        for binding in authority.options if authority is not None else ():
            if goal_bindings is not None and binding.binding_id not in goal_bindings:
                continue
            from tldw_chatbook.Tools.remote_root_types import is_remote

            from .console_chat_controller import _project_selection_from_snapshot

            root = _project_selection_from_snapshot(binding).root
            if is_remote(root):
                opaque = True
                continue
            chain = capture_directory_chain(root)
            scope = WorktreeScope(chain.canonical_root, True, chain)
            if scope not in reads:
                reads.append(scope)
            if (
                writable
                and binding.allow_write
                and (goal_bindings is None or goal_bindings[binding.binding_id])
            ):
                writes.append(scope)
    return WorktreeFootprint(tuple(reads), tuple(writes), opaque)


def _capture_local_provider_sources(provider, names):
    """Retain the actual composed receiving maps, not detached catalog labels."""
    specs = tuple((name, provider._specs[name]) for name in names)
    roots = tuple((provider._admitted_roots or {}).items())
    aliases = tuple(
        (alias, tuple(mapping.items()))
        for alias, mapping in provider._path_specs_by_alias.items()
    )
    return specs, roots, aliases


def _check_local_provider_sources(provider, retained):
    """Reuse the original ordinary receiving-instance checks before effects."""
    specs, roots, aliases = retained
    if any(provider._specs.get(name) is not spec for name, spec in specs):
        raise RuntimeError("ordinary local call outside original maximum")
    current_roots = provider._admitted_roots or {}
    if len(current_roots) != len(roots) or any(
        current_roots.get(alias) is not authority for alias, authority in roots
    ):
        raise RuntimeError("ordinary local root source changed")
    current_aliases = provider._path_specs_by_alias
    if len(current_aliases) != len(aliases) or any(
        alias not in current_aliases
        or len(current_aliases[alias]) != len(entries)
        or any(current_aliases[alias].get(tool) is not spec for tool, spec in entries)
        for alias, entries in aliases
    ):
        raise RuntimeError("ordinary local handler source changed")


def _raw_shell_metadata(tree):
    """Read the actual runtime and primitive app policy without invoking callbacks."""
    from .console_raw_cli import RawCliRuntime

    raw = getattr_static(tree.app, "raw_cli_runtime", None)
    if raw is None:
        return None, False, ()
    if not isinstance(raw, RawCliRuntime):
        return raw, True, ("unknown",)
    app_config = getattr_static(tree.app, "app_config", None)
    console = app_config.get("console", {}) if type(app_config) is dict else {}
    if type(console) is not dict:
        return raw, True, ("unknown",)
    permitted = console.get("raw_cli_permitted") is True
    armed = getattr_static(raw, "_armed") is True
    return raw, permitted and armed, (id(app_config), id(console))


def _function_record(function):
    return (
        function,
        function.__code__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
        function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
    )


def _function_unchanged(record):
    function, code, defaults, kwdefaults, entries, closure, cells = record
    return (
        type(function) is FunctionType
        and function.__code__ is code
        and function.__defaults__ is defaults
        and function.__kwdefaults__ is kwdefaults
        and (
            kwdefaults is None
            or len(kwdefaults) == len(entries)
            and all(
                key in kwdefaults and kwdefaults[key] is value for key, value in entries
            )
        )
        and function.__closure__ is closure
        and all(cell.cell_contents is value for cell, value in cells)
    )


def _capture_builtin_maximum(tree, advertised, catalog_allowed):
    """Project declared effects and retain the actual tool classes for composition."""
    import importlib

    from tldw_chatbook import config
    from tldw_chatbook.Agents import tool_catalog as catalog
    from tldw_chatbook.MCP.permission_store import (
        BUILTIN_TOOL_SERVER_KEY,
        resolve_effective_state_by_key,
    )

    effects = {}
    for name, cls in zip(
        catalog.ALWAYS_ON_BUILTIN_NAMES,
        (catalog.CalculatorTool, catalog.DateTimeTool),
        strict=True,
    ):
        if advertised(name) and catalog_allowed("builtin", name):
            effects[name] = "none"
            tree.builtin_classes[name] = cls
    payload = _permission_payload(tree)
    for entry in catalog._GATEABLE_BUILTINS:
        if not config.coerce_bool_setting(
            config.get_cli_setting("tools", entry.gate_key, False), False
        ):
            continue
        if (
            not advertised(entry.tool_name)
            or not catalog_allowed("builtin", entry.tool_name)
            or payload is not None
            and (
                payload.get("kill_switch", False)
                or resolve_effective_state_by_key(
                    payload,
                    BUILTIN_TOOL_SERVER_KEY,
                    entry.tool_name,
                    profile_id=tree.configuration.tool_policy_profile_id,
                ).state
                == "deny"
            )
            or tree.session.ephemeral
            and entry.tool_name in {"create_note", "update_note", "write_file"}
        ):
            continue
        module = importlib.import_module("tldw_chatbook.Tools." + entry.module_name)
        tree.builtin_classes[entry.tool_name] = getattr(module, entry.factory_name)
        effects[entry.tool_name] = entry.worktree_effect
    tree.builtin_maximum = effects
    return effects, "opaque" in effects.values()


def _capture_personal_source(tree):
    tree.personal_getter = getattr_static(
        tree.app, "get_personal_context_service", None
    )
    tree.personal_service = getattr_static(tree.app, "_personal_context_service", None)
    if tree.personal_getter is None:
        return False
    tree.personal_native = _capture_personal_service(tree, tree.personal_service)
    return not tree.personal_native


def _capture_library_maximum(tree):
    """Read original Library declarations and selected backends, never compose."""
    import sys

    from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS

    from .console_library_policy import ConsoleAssistantLibraryAccess
    from .console_runtime import _NativeLibraryProviderFactory

    factory = getattr_static(tree.controller, "_library_provider_factory", None)
    policy = tree.configuration.library_policy_maximum
    if (
        factory is None
        or policy is None
        or policy.assistant_access is not ConsoleAssistantLibraryAccess.ALLOWED
    ):
        return False
    direct = tree.configuration.tool_configuration.get("direct_library_tools", True)
    names = tuple(LIBRARY_TOOL_DESCRIPTORS) if direct else ("search_rag",)
    tree.library_maximum = frozenset(
        name for name in names if tree.catalog_permits("library", name)
    )
    if not (
        isinstance(factory, _NativeLibraryProviderFactory)
        and tree.runtime._library_provider_factory is factory
        and factory.runtime is tree.runtime
        and tree.runtime._app is tree.app
    ):
        return True
    tree.library_factory = factory
    # Native RAG can initialize the selected shared backend; it has no closed
    # original artifact/index destination contract at this boundary.
    if not direct:
        service = getattr_static(tree.app, "library_rag_search_service", None)
        tree.local_source_values += (
            (tree.app, "library_rag_search_service", service, None),
        )
        return service is not None
    values = list(tree.local_source_values)
    opaque = False
    services = (
        (
            "_media",
            "local_media_reading_service",
            "tldw_chatbook.Media.local_media_reading_service",
            "media",
        ),
        ("_notes", "notes_service", "tldw_chatbook.Notes.Notes_Library", "note"),
        (
            "_prompts",
            "local_prompt_service",
            "tldw_chatbook.Prompt_Management.local_prompt_service",
            "prompt",
        ),
        ("_skills", "local_skills_service", None, "skill"),
        (
            "_conversations",
            "local_chat_conversation_service",
            "tldw_chatbook.Chat.chat_conversation_service",
            "conversation",
        ),
    )
    # Only already loaded native source modules qualify. Classification never
    # imports a caller-selected backend or executes a service constructor.
    source_modules = [
        (sys.modules.get("tldw_chatbook.Agents.library_tool_provider"), None),
        (sys.modules.get("tldw_chatbook.Library.local_library_tool_service"), None),
    ]
    for service_field, attribute, module_name, item_type in services:
        service = getattr_static(tree.app, attribute, None)
        tree.library_services[service_field] = service
        values.append((tree.app, attribute, service, None))
        reachable = [
            LIBRARY_TOOL_DESCRIPTORS[name]
            for name in tree.library_maximum
            if LIBRARY_TOOL_DESCRIPTORS[name].item_type == item_type
        ]
        if service is None or not reachable:
            continue
        if any(descriptor.operation in {"save", "rechunk"} for descriptor in reachable):
            opaque = True
        module = sys.modules.get(module_name) if module_name is not None else None
        if module is None:
            opaque = True
            continue
        source_modules.append((module, service))
        # These Library methods use the concrete native DB interfaces. The
        # prompt facade's injected module and skill facades are not such proof.
        if service_field == "_prompts":
            opaque = True
        elif service_field in {"_media", "_notes", "_conversations"}:
            db_field = {
                "_media": "media_db",
                "_notes": "unified_db_template",
                "_conversations": "db",
            }[service_field]
            database = getattr_static(service, db_field, None)
            values.append((service, db_field, database, None))
            if database is not None:
                for path in ("db_path", "db_path_str"):
                    value = getattr_static(database, path, None)
                    values.append((database, path, value, None))
    media = tree.library_services["_media"]
    media_db = getattr_static(tree.app, "media_db", None)
    values.append((tree.app, "media_db", media_db, None))
    if media is not None or media_db is not None:
        if any(
            LIBRARY_TOOL_DESCRIPTORS[name].operation == "rechunk"
            for name in tree.library_maximum
        ):
            opaque = True
        else:
            # Its injected chunk/template/policy interfaces are only known at
            # actual composition; retain OPAQUE until their original source is closed.
            opaque |= any(
                LIBRARY_TOOL_DESCRIPTORS[name].operation
                in {"structure", "chunk", "spec_list", "spec_save"}
                for name in tree.library_maximum
            )
    policy_source = getattr_static(tree.app, "service_policy_enforcer", None)
    values.append((tree.app, "service_policy_enforcer", policy_source, None))
    for module, instance in source_modules:
        if module is None or not hasattr(module, "_NATIVE_ORDINARY_LIBRARY_CLASS"):
            opaque = True
            continue
        cls = module._NATIVE_ORDINARY_LIBRARY_CLASS
        if instance is not None and type(instance) is not cls:
            opaque = True
        values.append(
            (module, cls.__name__, getattr_static(module, cls.__name__), None)
        )
        opaque |= getattr_static(module, cls.__name__) is not cls
    tree.local_source_values = tuple(values)
    return opaque


def _capture_other_providers(tree):
    """Project actual configured provider capabilities without executing factories."""
    from tldw_chatbook import config
    from tldw_chatbook.Agents import tool_catalog

    opaque = False
    canvas = getattr_static(tree.store, "canvas_turn_controller", None)
    tree.local_source_values += ((tree.store, "canvas_turn_controller", canvas, None),)
    if canvas is not None and config.get_canvas_execution_enabled():
        tree.other_maximum["canvas"] = frozenset(
            name
            for name in tool_catalog.CANVAS_RESERVED_TOOL_NAMES
            if tree.catalog_permits("canvas", name)
        )
        opaque = True
    if tree.personal_getter is not None and not tree.session.ephemeral:
        tree.other_maximum["personal-context"] = frozenset(
            name
            for name in tool_catalog.PROFILE_RESERVED_TOOL_NAMES
            if tree.catalog_permits("personal-context", name)
        )
    if (
        tree.configuration.tool_configuration.get("local_tools_enabled")
        and not tree.configuration.mcp_capture_pending
    ):
        payload = _permission_payload(tree)
        if payload is None or not payload.get("kill_switch", False):
            authority = tree.configuration.project_authority
            if (
                authority is not None
                and authority.options
                and tree.catalog_permits("virtual_cli", "virtual_cli")
            ):
                from tldw_chatbook.Agents.virtual_cli_provider import (
                    VIRTUAL_CLI_COMMANDS,
                    VIRTUAL_CLI_SERVER_KEY,
                )
                from tldw_chatbook.MCP.permission_store import (
                    resolve_effective_state_by_key,
                )

                tree.other_maximum["virtual_cli"] = frozenset({"virtual_cli"})
                tree.virtual_commands = frozenset(
                    command
                    for command in VIRTUAL_CLI_COMMANDS
                    if payload is None
                    or resolve_effective_state_by_key(
                        payload,
                        VIRTUAL_CLI_SERVER_KEY,
                        command,
                        profile_id=tree.configuration.tool_policy_profile_id,
                    ).state
                    != "deny"
                )
            if tree.catalog_permits("raw_shell", "shell_exec"):
                tree.other_maximum["raw_shell"] = frozenset({"shell_exec"})
    return opaque


def _capture_personal_service(tree, instance):
    """Retain the actual private repository and encryption resources."""
    from tldw_chatbook.Personal_Context.key_protector import (
        InMemoryProfileKeyProtector,
        KeyringProfileKeyProtector,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
    from tldw_chatbook.Personal_Context.service import PersonalContextService

    if type(instance) is not PersonalContextService:
        return False
    repository = getattr_static(instance, "_repository", None)
    tree.personal_source_values += ((instance, "_repository", repository),)
    if repository is None:
        return True
    path = getattr_static(repository, "db_path", None)
    protector = getattr_static(repository, "_protector", None)
    tree.personal_source_values += (
        (repository, "db_path", path),
        (repository, "_protector", protector),
    )
    return type(repository) is PersonalContextRepository and type(protector) in (
        KeyringProfileKeyProtector,
        InMemoryProfileKeyProtector,
    )
