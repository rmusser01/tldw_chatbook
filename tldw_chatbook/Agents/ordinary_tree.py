"""Custody for one admitted ordinary execution tree and its tool sources."""

from __future__ import annotations

import json
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from inspect import getattr_static
from threading import Lock

from tldw_chatbook.console_app_services import read_mcp_service
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

from .execution_capacity import CapacityRefused, WorkOrigin


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


def encode_ordinary_receipt(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require_original_ordinary_turn(tree: object) -> None:
    """Require the Console tree that owns the original storage lifetime."""
    if not isinstance(tree, OrdinaryTurn):
        raise TypeError("original source retention changed")


class OrdinaryTurn:
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
        self.permission = self.permission_path = None
        self.native_permission_publication = None
        self.reference = None
        self.snapshot_sources = []
        self.ssh_sources = {}
        self.local_maximum = {}
        self.local_source_values = ()
        self.raw_projection = None
        self.local_providers = {}
        self.other_maximum = {}
        self.archive_recovery = None
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

    def retain_mcp_permission_publication(self, publication):
        """Keep the actual publisher's resource receipt under this opaque claim."""
        self.borrow(self.runtime)
        service, store, candidate, _binding, path = publication
        if (
            self.native_permission_publication is not None
            or self.permission is not None
            or read_mcp_service(self.app) is not service
            or service.local_service.store is not store
            or service._permission_store is not candidate
        ):
            raise RuntimeError("Ordinary MCP permission publication changed.")
        self.permission, self.permission_path = candidate, path
        self.native_permission_publication = publication

    def check_local_sources(self):
        self.check()
        if self.raw_projection is not None:
            original, was_reachable, values = self.raw_projection
            from tldw_chatbook.Tools.raw_cli_executor import ordinary_raw_shell_metadata

            current, reachable, sources = ordinary_raw_shell_metadata(self)
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
        native = capture_ordinary_personal_service(self, service)
        if not native and not self.footprint.opaque:
            raise RuntimeError("ordinary personal context publication widened")
        self.personal_service = service

    def bind_local_provider(self, provider, names):
        """Validate the one real composition against the original native maximum."""
        from tldw_chatbook.Agents import local_tool_provider as local
        from tldw_chatbook.MCP.permission_store import resolve_effective_state_by_key
        from tldw_chatbook.Tools.remote_root_types import is_remote, local_root_path

        self.check_local_sources()
        if not local.is_native_local_provider(provider):
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
        self.local_providers[provider] = capture_ordinary_local_provider_sources(
            provider, reachable
        )

    def bind_builtin_provider(self, provider, names):
        from tldw_chatbook.Agents import tool_catalog as catalog
        from tldw_chatbook.MCP.permission_store import (
            BUILTIN_TOOL_SERVER_KEY,
            resolve_effective_state_by_key,
        )

        self.check_local_sources()
        if not catalog.is_native_builtin_provider(provider):
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
        check_ordinary_local_provider_sources(provider, retained)
        original = dict(retained[0]).get(name)
        if original is None or provider._specs.get(name) is not original:
            raise RuntimeError("ordinary local call outside original maximum")
        if not self.footprint.opaque and provider._agent_roots:
            raise RuntimeError("ordinary local child root outside maximum")

    def bind_virtual_provider(self, provider):
        from tldw_chatbook.Agents.virtual_cli_provider import (
            NATIVE_VIRTUAL_CLASS,
            observe_virtual_callbacks,
        )
        from tldw_chatbook.Tools.remote_root_types import is_remote, local_root_path
        from tldw_chatbook.Tools.workspace_tool_executor import (
            _NATIVE_ORDINARY_EXECUTOR_CLASS,
        )

        if type(provider) is not NATIVE_VIRTUAL_CLASS:
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
        callbacks = observe_virtual_callbacks(provider)
        self.virtual_providers[provider] = roots, tuple(held), callbacks

    def check_virtual_call(self, provider, args=None):
        self.check_local_sources()
        if args is not None and args.get("command") not in self.virtual_commands:
            raise RuntimeError("ordinary virtual command outside original maximum")
        retained = self.virtual_providers.get(provider)
        if retained is None:
            if self.footprint.opaque:
                return
            raise RuntimeError("ordinary virtual provider was not composed")
        roots, registries, callbacks = retained
        if any(
            getattr_static(provider, name, None) is not original
            for name, original in callbacks
        ):
            raise RuntimeError("original virtual callback source changed")
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
                not library_tool_provider.is_native_library_provider(provider)
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

    def check_archive_provider(self, provider):
        from tldw_chatbook.Agents.archive_recovery_tool_provider import (
            ArchiveRecoveryToolProvider,
        )

        if self.archive_recovery is None:
            raise RuntimeError("ordinary archive recovery custody unavailable")
        owner, turn_id = self.archive_recovery
        if (
            self.runtime._archive_recovery is not owner
            or self.controller._archive_recovery_runtime is not owner
            or type(provider) is not ArchiveRecoveryToolProvider
            or not owner.authenticates(provider.authority)
            or provider.authority.turn_id != turn_id
            or provider.authority.session_id != self.session.id
        ):
            raise RuntimeError("ordinary archive recovery custody changed")

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
                    NATIVE_COLLISION_FILTERED_MCP_PROVIDER,
                    observe_controller_catalog_entry,
                )
                from tldw_chatbook.MCP.permission_store import definition_hash

                if type(provider) is not NATIVE_COLLISION_FILTERED_MCP_PROVIDER:
                    raise RuntimeError("ordinary MCP collision view changed")
                underlying = getattr_static(provider, "_provider", None)
                allowed = getattr_static(provider, "_allowed_names", None)
                table, entry = observe_controller_catalog_entry(underlying, tool_id)
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
            if source == "archive_recovery":
                self.check_archive_provider(provider)
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
        if source == "archive_recovery":
            self.check_archive_provider(provider)
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
                member.steps.update((index, encode_ordinary_receipt(record)) for index, record in records)

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
                stored = {step["index"]: encode_ordinary_receipt(step) for step in row["steps"]}
                proven = (
                    row["status"] == status
                    and encode_ordinary_receipt(row.get("result")) == encode_ordinary_receipt(result)
                    and (
                        tokens is None
                        or encode_ordinary_receipt(row.get("budget_tokens")) == encode_ordinary_receipt(tokens)
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


def capture_ordinary_local_provider_sources(provider, names):
    """Retain the actual composed receiving maps, not detached catalog labels."""
    specs = tuple((name, provider._specs[name]) for name in names)
    roots = tuple((provider._admitted_roots or {}).items())
    aliases = tuple(
        (alias, tuple(mapping.items()))
        for alias, mapping in provider._path_specs_by_alias.items()
    )
    return specs, roots, aliases


def check_ordinary_local_provider_sources(provider, retained):
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


def capture_ordinary_personal_service(tree, instance):
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
