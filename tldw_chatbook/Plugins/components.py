"""Native runtime projections from the existing immutable admission ceiling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_chatbook.Agents.hooks_v2.models import HookEvent, HookHandler
    from tldw_chatbook.Plugins.coordinator import PluginCoordinator
    from tldw_chatbook.Plugins.models import PackageInspection
    from tldw_chatbook.Plugins.service import PluginService

from .admission import PluginUnavailable, RunPluginSnapshot
from .models import ComponentRecord
from .package_files import capture_package


class NativeComponent(ComponentRecord):
    """Runtime provenance; never persisted as part of package inspection."""

    snapshot: RunPluginSnapshot


def qualified_id(snapshot: RunPluginSnapshot, component_id: str) -> str:
    """Use installation identity, never a display label, for runtime ownership."""
    return f"plugin:{snapshot.installation_id}:{component_id}"


def component_body(component: NativeComponent) -> str:
    """Read complete current retained bytes before exposing instruction content."""
    inspection = component.snapshot.inspection
    capture = capture_package(Path(inspection.materialized_identity))
    if capture.errors or capture.digest != inspection.content_digest:
        raise PluginUnavailable("plugin_material_changed")
    lines = capture.read(component.path).decode("utf-8").splitlines()
    end = next(i for i, line in enumerate(lines[1:], 1) if line == "---")
    return "\n".join(lines[end + 1 :]).strip()


class PluginComponents:
    """Project only the originally admitted components, in stable identity order."""

    @staticmethod
    def project(snapshot: RunPluginSnapshot) -> tuple[ComponentRecord, ...]:
        return tuple(
            NativeComponent(
                **snapshot.inspection.inventory[key]
                .model_copy(update={"availability": "ready", "selection": "selected"})
                .model_dump(),
                snapshot=snapshot,
            )
            for key in sorted(snapshot.selection)
        )

    @staticmethod
    def required_hooks(
        snapshot: RunPluginSnapshot, component_id: str
    ) -> tuple[str, ...]:
        """Resolve the original complete dependency closure to exact hook IDs."""
        if component_id not in snapshot.selection:
            raise PluginUnavailable("plugin_component_not_admitted")
        graph = dict(snapshot.dependencies)
        pending, visited, hooks = [component_id], set(), set()
        while pending:
            key = pending.pop()
            if key in visited:
                continue
            visited.add(key)
            if key not in graph or key not in snapshot.selection:
                raise PluginUnavailable("plugin_dependency_unavailable")
            for dependency in graph[key]:
                if dependency.startswith("hook:"):
                    hooks.add(qualified_id(snapshot, dependency))
                pending.append(dependency)
        return tuple(sorted(hooks))


def component_summary(
    installation_id: str, inspection: PackageInspection, component_id: str, alias: str
) -> dict[str, Any]:
    """Publish only metadata; commands/rules/agents never become skill tools."""
    from .skill_provider import skill_summary

    component = inspection.inventory[component_id]
    if component.kind == "skill":
        return skill_summary(installation_id, inspection, component_id, alias)
    definition = json.loads(component.definition_json)
    return {
        "name": f"{alias}:{component.local_id}",
        "record_id": f"plugin:{installation_id}:{component_id}",
        "tool_name": "plugin_"
        + hashlib.sha256(f"{installation_id}:{component_id}".encode()).hexdigest()[:56],
        "plugin_owned": True,
        "plugin_kind": component.kind,
        "plugin_installation_id": installation_id,
        "plugin_component_id": component_id,
        "plugin_revision": inspection.effective_digest,
        "description": definition.get("description", ""),
        "mode": definition.get("mode"),
        "arguments": definition.get("arguments", []),
        "definition_digest": inspection.effective_digest,
        "user_invocable": False,
        "disable_model_invocation": True,
        "trust_blocked": True,
        "backend": "local",
    }


def capture_model_mapping(
    owner: Any,
    installation_id: str,
    inspection: PackageInspection,
    component_id: str,
    target: str,
) -> dict[str, Any]:
    """Capture a canonical current reference from the actual local model owner."""
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )

    from .package_files import canonical_json

    if not isinstance(owner, LocalLLMProviderCatalogService):
        raise PluginUnavailable("plugin_model_owner_unavailable")
    component = inspection.inventory[component_id]
    definition = json.loads(component.definition_json)
    if component.kind not in {"skill", "agent"} or not definition.get(
        "model", definition.get("metadata", {}).get("model")
    ):
        raise ValueError("plugin_model_mapping_invalid")
    record = owner.get_model_metadata(target)
    if record["id"] != target:
        raise ValueError("plugin_model_reference_not_canonical")
    endpoint = owner._current_endpoint_fingerprint(provider_key=record["provider"])
    if endpoint is None:
        raise PluginUnavailable("plugin_model_configuration_unavailable")
    return {
        "installation_id": installation_id,
        "mapping_id": component_id + ":model",
        "component_id": component_id,
        "revision_digest": inspection.effective_digest,
        "kind": "model",
        "target_reference": target,
        "definition_digest": hashlib.sha256(
            component.definition_json.encode()
        ).hexdigest(),
        "configuration_digest": hashlib.sha256(
            canonical_json({"model": record, "endpoint": endpoint}).encode()
        ).hexdigest(),
        "credential_bindings": [],
    }


def capture_tool_constraint(
    coordinator: PluginCoordinator,
    installation_id: str,
    inspection: PackageInspection,
    component_id: str,
    declared: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    """Review a restriction against an already reviewed exact M4 tool reference."""
    from .agent_presets import tool_constraint_id
    from .package_files import canonical_json

    component = inspection.inventory[component_id]
    definition = json.loads(component.definition_json)
    tokens = definition.get("tools", definition.get("allowed-tools", ()))
    if isinstance(tokens, str):
        tokens = tokens.split()
    pending, dependencies = list(component.dependencies), set()
    while pending:
        key = pending.pop()
        if key not in dependencies:
            dependencies.add(key)
            pending.extend(inspection.inventory[key].dependencies)
    if (
        component.kind not in {"agent", "skill"}
        or declared not in tokens
        or declared.startswith(("builtin:", "local:"))
        or declared == "inherit"
        or target["kind"] != "tool"
        or target["installation_id"] != installation_id
        or target["revision_digest"] != inspection.effective_digest
        or target["component_id"] not in dependencies
    ):
        raise PluginUnavailable("plugin_tool_mapping_invalid")
    validate_mapping(coordinator, target, inspection)
    return {
        "installation_id": installation_id,
        "mapping_id": tool_constraint_id(component_id, declared),
        "component_id": component_id,
        "revision_digest": inspection.effective_digest,
        "kind": "configuration",
        "target_reference": target["mapping_id"],
        "definition_digest": hashlib.sha256(
            component.definition_json.encode()
        ).hexdigest(),
        "configuration_digest": hashlib.sha256(
            canonical_json(target).encode()
        ).hexdigest(),
        "credential_bindings": [],
    }


def validate_mapping(
    coordinator: PluginCoordinator,
    mapping: dict[str, Any],
    inspection: PackageInspection,
    mappings: Sequence[dict[str, Any]] = (),
) -> None:
    """Revalidate existing owner references at review, commit, admission and recovery."""
    if mapping["kind"] == "model":
        if mapping != capture_model_mapping(
            coordinator.model_mapping_owner,
            mapping["installation_id"],
            inspection,
            mapping["component_id"],
            mapping["target_reference"],
        ):
            raise PluginUnavailable("plugin_model_mapping_changed")
        return
    if mapping["kind"] == "configuration" and mapping["mapping_id"].startswith(
        "variable:"
    ):
        from .variables import validate_mapping as validate_variable_mapping

        validate_variable_mapping(coordinator, mapping, inspection)
        return
    if mapping["kind"] == "configuration":
        from .agent_presets import tool_constraint_id

        component = inspection.inventory[mapping["component_id"]]
        definition = json.loads(component.definition_json)
        declared = definition.get("tools", definition.get("allowed-tools", ()))
        if isinstance(declared, str):
            declared = declared.split()
        tokens = [
            token
            for token in declared
            if tool_constraint_id(component.component_id, token)
            == mapping["mapping_id"]
        ]
        targets = [
            row
            for row in mappings
            if row["mapping_id"] == mapping["target_reference"]
            and row["kind"] == "tool"
            and row["installation_id"] == mapping["installation_id"]
            and row["revision_digest"] == mapping["revision_digest"]
        ]
        if (
            len(tokens) != 1
            or len(targets) != 1
            or capture_tool_constraint(
                coordinator,
                mapping["installation_id"],
                inspection,
                component.component_id,
                tokens[0],
                targets[0],
            )
            != mapping
        ):
            raise PluginUnavailable("plugin_tool_mapping_changed")
        return
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService

    if not isinstance(coordinator.mcp_mapping_owner, LocalMCPControlService):
        raise PluginUnavailable("plugin_mapping_owner_unavailable")
    coordinator.mcp_mapping_owner.validate_connection_mapping(mapping, inspection)


def register_hook_definitions(snapshot: RunPluginSnapshot) -> tuple[HookHandler, ...]:
    """Normalize admitted hooks with exact installation-qualified identities."""
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    handlers = []
    for key in (*snapshot.selection, *snapshot.unavailable_hooks):
        component = snapshot.inspection.inventory[key]
        if component.kind != "hook":
            continue
        definition = json.loads(component.definition_json)
        # Validate original local IDs before qualifying host identity.
        handler = parse_handlers(
            [definition],
            declared_variable_names=set(json.loads(snapshot.inspection.variables_json)),
        )[0]
        updates = {"id": qualified_id(snapshot, component.component_id)}
        if handler.type == "command":
            # Keep declared/user values unresolved in reviewed handler metadata.
            # Vendor cwd adaptation is always the authenticated retained package.
            if handler.vendor_protocol:
                updates["cwd"] = snapshot.inspection.materialized_identity
        elif component.component_id not in snapshot.unavailable_hooks:
            server = (
                handler.server
                if handler.server.startswith("mcp:")
                else "mcp:" + handler.server
            )
            mappings = [
                row
                for row in json.loads(snapshot.mappings_json)
                if row["component_id"] == server and row["kind"] == "connection"
            ]
            if len(mappings) != 1:
                raise PluginUnavailable("plugin_hook_connection_unavailable")
            updates["server"] = mappings[0]["target_reference"]
        handlers.append(handler.model_copy(update=updates))
    return tuple(handlers)


class NativeHookProcessOwner:
    """Adapt one handler to F8; the original job retains this exact adapter."""

    def __init__(
        self,
        service: PluginService,
        snapshot: RunPluginSnapshot,
        component_id: str,
        cancel: Callable[[], None],
        *,
        scratch_manager=None,
        scratch_snapshot=None,
    ) -> None:
        self.scratch_manager, self.scratch_snapshot = scratch_manager, scratch_snapshot
        self.scratch_leases = {}
        self.service, self.snapshot, self.component_id = service, snapshot, component_id
        import asyncio

        self.loop = asyncio.get_running_loop()
        self.cancel = lambda: self.loop.call_soon_threadsafe(cancel)
        self.records = {}

    def reserve_launch(self, event: HookEvent) -> str:
        def reserve():
            from .data_cleanup import root_ref

            snapshot = self.snapshot
            self.service._admission.check(snapshot, self.component_id)
            roots = tuple(root_ref(row) for row in json.loads(snapshot.data_roots_json))
            owner = self.service._coordinator.owner
            scratch = (
                self.scratch_manager.lease(self.scratch_snapshot)
                if self.scratch_manager is not None
                else None
            )
            if scratch is not None:
                scratch.__enter__()
            try:
                token = owner.reserve_launch(
                    event.event_id,
                    snapshot.installation_id,
                    snapshot.workspace_id,
                    snapshot.revision_digest,
                    roots=roots,
                    root_coverage="known" if roots else "qualified_none",
                )
            except BaseException:
                if scratch is not None:
                    scratch.__exit__(None, None, None)
                raise
            if scratch is not None:
                self.scratch_leases[token] = scratch
            try:
                import threading

                from .service import PluginRunOwnership

                self.service._coordinator.root_usage.retain_cancel(token, self.cancel)
                record = PluginRunOwnership(
                    snapshot.installation_id,
                    snapshot.workspace_id,
                    snapshot.revision_digest,
                    snapshot.run_id,
                    "hook:" + token,
                    event.event_id,
                    token,
                    self.cancel,
                    threading.Event(),
                    generations=snapshot.generations,
                    component_ceiling=(self.component_id,),
                )
                with self.service._lock:
                    self.service.fences.check_snapshot(snapshot)
                    self.service.fences.require_admission(
                        snapshot.installation_id, snapshot.revision_digest
                    )
                    self.service._live_runs[
                        (snapshot.installation_id, record.run_id)
                    ] = record
                    self.records[token] = record
            except BaseException:
                owner.settle_process(token, True)
                lease = self.scratch_leases.pop(token, None)
                if lease is not None:
                    lease.__exit__(None, None, None)
                raise
            return token

        return self.service._call_from_agent(reserve)

    def publish_process(self, token: str, provenance: dict[str, Any]) -> None:
        def publish():
            self.service._admission.check(self.snapshot, self.component_id)
            self.service._coordinator.owner.publish_process(token, provenance)

        self.service._call_from_agent(publish)

    def settle_process(self, token: str, confirmed: bool) -> None:
        def settle():
            self.service._coordinator.owner.settle_process(token, confirmed)
            if confirmed:
                lease = self.scratch_leases.pop(token, None)
                if lease is not None:
                    lease.__exit__(None, None, None)
                with self.service._lock:
                    record = self.records.pop(token, None)
                    if record is not None:
                        record.completed.set()
                        self.service._live_runs.pop(
                            (record.installation_id, record.run_id), None
                        )

        self.service._call_from_agent(settle, terminal=True)


class NativeHookSet:
    """Host wiring for the existing hook engine, graph and context owners."""

    def __init__(
        self,
        service: PluginService,
        snapshots: Sequence[RunPluginSnapshot],
        *,
        scratch_manager=None,
        scratch_snapshot=None,
    ) -> None:
        if (scratch_manager is None) != (scratch_snapshot is None):
            raise ValueError("plugin_scratch_capability_pair_required")
        self.scratch_manager, self.scratch_snapshot = scratch_manager, scratch_snapshot
        self.service = service
        self.snapshots = tuple(snapshots)
        self.definitions = tuple(
            handler
            for snapshot in snapshots
            for handler in register_hook_definitions(snapshot)
        )
        self.owners = {
            qualified_id(snapshot, key): (snapshot, key)
            for snapshot in snapshots
            for key in (*snapshot.selection, *snapshot.unavailable_hooks)
            if snapshot.inspection.inventory[key].kind == "hook"
        }
        self.unavailable = frozenset(
            qualified_id(snapshot, key)
            for snapshot in snapshots
            for key in snapshot.unavailable_hooks
        )
        self.requirements = frozenset(
            handler_id
            for snapshot in snapshots
            for key in snapshot.selection
            for handler_id in PluginComponents.required_hooks(snapshot, key)
        )

    def requirements_for_messages(
        self, messages: Sequence[dict[str, Any]]
    ) -> tuple[str, ...]:
        from tldw_chatbook.Agents.agent_models import PluginContextText

        snapshots = {row.installation_id: row for row in self.snapshots}
        requirements = set()
        for row in messages:
            content = row.get("content")
            if isinstance(content, PluginContextText):
                for origin in content.checked_origins():
                    snapshot = snapshots.get(origin.installation_id)
                    if snapshot is None or snapshot.revision_digest != origin.revision:
                        raise PluginUnavailable("plugin_context_snapshot_unavailable")
                    requirements.update(
                        PluginComponents.required_hooks(snapshot, origin.component_id)
                    )
        return tuple(sorted(requirements))

    def current(self, handler: HookHandler, event: HookEvent, stage: str) -> bool:
        if handler.id in self.unavailable:
            return False
        owner = self.owners.get(handler.id)
        if owner is None:
            return True
        snapshot, key = owner
        if (
            snapshot.workspace_id is None
            and self.scratch_manager is not None
            and not self.scratch_manager.is_live(self.scratch_snapshot)
        ):
            return False
        try:
            import threading

            if threading.current_thread() is threading.main_thread():
                self.service._require_open()
                self.service.fences.check_snapshot(snapshot)
            else:
                self.service._call_from_agent(
                    lambda: self.service._admission.check(snapshot, key)
                )
            return True
        except (PermissionError, ValueError, RuntimeError):
            return False

    def process_owner(
        self, handler: HookHandler, event: HookEvent, cancel: Callable[[], None]
    ) -> NativeHookProcessOwner | None:
        if handler.id in self.unavailable:
            raise PluginUnavailable("plugin_requirement_not_executable")
        owner = self.owners.get(handler.id)
        if owner is None:
            return None
        return NativeHookProcessOwner(
            self.service,
            *owner,
            cancel,
            scratch_manager=(
                self.scratch_manager if owner[0].workspace_id is None else None
            ),
            scratch_snapshot=(
                self.scratch_snapshot if owner[0].workspace_id is None else None
            ),
        )

    def host_environment(
        self, handler: HookHandler, event: HookEvent
    ) -> dict[str, str]:
        owner = self.owners.get(handler.id)
        if owner is None:
            return {}
        snapshot, _key = owner
        return self._roots(snapshot)

    def _roots(self, snapshot) -> dict[str, str]:
        result = {"PLUGIN_ROOT": snapshot.inspection.materialized_identity}
        roots = json.loads(snapshot.data_roots_json)
        shared = [row for row in roots if row["workspace_id"] is None]
        scoped = [
            row
            for row in roots
            if snapshot.workspace_id is not None
            and row["workspace_id"] == snapshot.workspace_id
        ]
        if len(shared) > 1 or len(scoped) > 1:
            raise PluginUnavailable("plugin_data_root_ambiguous")
        if shared:
            result["PLUGIN_DATA"] = shared[0]["path"]
        if snapshot.workspace_id is not None:
            if scoped:
                result["CHATBOOK_WORKSPACE_DATA"] = scoped[0]["path"]
        elif self.scratch_manager is not None:
            if not self.scratch_manager.is_live(self.scratch_snapshot):
                raise PluginUnavailable("plugin_scratch_unavailable")
            result["CHATBOOK_WORKSPACE_DATA"] = str(self.scratch_snapshot.root)
        return result

    def command_launch(self, handler, event, host_environment):
        """Resolve protected values once after reservation, only into launch locals."""
        import os
        import shutil

        from tldw_chatbook.Agents.hooks_v2.command_executor import CommandLaunchValues

        from .variables import REFERENCE, ROOT_NAMES, text, values

        owner = self.owners.get(handler.id)
        if owner is None:
            return None
        snapshot, key = owner

        def resolve():
            self.service._admission.check(snapshot, key)
            return values(
                self.service._coordinator,
                snapshot.installation_id,
                snapshot.inspection,
                key,
                json.loads(snapshot.mappings_json),
                resolve=True,
            )

        variables = {
            name: text(value)
            for name, value in self.service._call_from_agent(resolve).items()
        }
        roots = self._roots(snapshot)
        variables.update(roots)

        def expand(value):
            def substitute(match):
                if match[1] not in variables:
                    raise PluginUnavailable("plugin_variable_unavailable")
                return variables[match[1]]

            result = REFERENCE.sub(substitute, value)
            if "\0" in result:
                raise PluginUnavailable("plugin_variable_invalid")
            return result

        argv = tuple(expand(value) for value in handler.argv)
        root = Path(roots["PLUGIN_ROOT"])
        token = argv[0]
        if token.startswith("./"):
            executable = str(root / token)
        elif Path(token).is_absolute():
            executable = token
        else:
            host_cwd = Path.cwd()
            lookup_path = host_environment.get("PATH", "")
            if lookup_path:
                # Child cwd/PATH must not reinterpret the host's selection.
                lookup_path = os.pathsep.join(
                    str(host_cwd / entry) for entry in lookup_path.split(os.pathsep)
                )
            executable = shutil.which(token, path=lookup_path)
            if executable:
                executable = str(host_cwd / executable)
        if not executable:
            raise PluginUnavailable("plugin_executable_unavailable")
        cwd = (root / expand(handler.cwd or ".")).resolve()
        if not cwd.is_relative_to(root.resolve()):
            raise PluginUnavailable("plugin_hook_cwd_invalid")
        environment = {
            name: value
            for name, value in host_environment.items()
            if name not in ROOT_NAMES
        }
        environment.update(
            {
                name: (
                    expand(value)
                    if isinstance(value, str)
                    else variables[value["variable"]]
                )
                for name, value in (handler.env or {}).items()
            }
        )
        environment.update(roots)
        return CommandLaunchValues((executable, *argv[1:]), str(cwd), environment)

    def render_context(self, event: HookEvent, handler_id: str, block: Any) -> str:
        from tldw_chatbook.Agents.agent_models import (
            PluginContextOrigin,
            PluginContextText,
        )
        from tldw_chatbook.Agents.hooks_v2.context import ContextLedger

        text = ContextLedger.render(event, handler_id, block)
        owner = self.owners.get(handler_id)
        if owner is None:
            return text
        snapshot, key = owner
        return PluginContextText(
            text,
            (
                PluginContextOrigin(
                    snapshot.installation_id,
                    key,
                    snapshot.revision_digest,
                    len(text.encode()),
                ),
            ),
        )


class NativeMCPProviders:
    """Compose normal MCP owners while preserving each concrete dispatch owner."""

    def __init__(self, providers):
        self.providers = tuple(providers)
        self._owners = {}
        for provider in providers:
            for entry in provider.list_catalog():
                if entry.name in self._owners:
                    raise PluginUnavailable("plugin_tool_namespace_collision")
                self._owners[entry.name] = provider

    @property
    def not_connected_count(self):
        return sum(provider.not_connected_count for provider in self.providers)

    def list_catalog(self):
        return [
            entry for provider in self.providers for entry in provider.list_catalog()
        ]

    def pending_gate_for(self, name, *args, **kwargs):
        provider = self._owners.get(name)
        return provider.pending_gate_for(name, *args, **kwargs) if provider else None

    def apply_batch_decisions(self, *args, **kwargs):
        for provider in self.providers:
            provider.apply_batch_decisions(*args, **kwargs)

    def record_user_denial(self, name):
        provider = self._owners.get(name)
        if provider:
            provider.record_user_denial(name)

    def stamp_scope(self, run_id):
        from contextlib import ExitStack

        stack = ExitStack()
        for provider in self.providers:
            stack.enter_context(provider.stamp_scope(run_id))
        return stack


def tool_hook_requirements(registry, definition):
    """Resolve exact owned declarations; an unknown provider remains unknown."""
    from .mcp_provider import PluginMCPProvider

    owner = registry.resolve_owner_for_name(definition.name)
    provider = getattr(owner[1], "_provider", owner[1]) if owner else None
    if isinstance(provider, PluginMCPProvider):
        snapshot = provider.snapshot
        for row in json.loads(snapshot.mappings_json):
            if row["kind"] != "tool":
                continue
            raw_name = row["target_reference"].partition("::")[2]
            name = (
                "plugin_mcp_"
                + hashlib.sha256(
                    (
                        snapshot.installation_id
                        + ":"
                        + row["component_id"]
                        + ":"
                        + raw_name
                    ).encode()
                ).hexdigest()[:48]
            )
            if name == definition.name:
                return PluginComponents.required_hooks(snapshot, row["component_id"])
        return None
    return () if owner is not None else None
