"""Native hook projections adapt the existing scheduler and durable process owner."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .models import PackageInspection

if TYPE_CHECKING:
    from tldw_chatbook.Agents.tool_catalog import (
        ToolCatalogRegistry,
        ToolDefinitionSnapshot,
    )

    from .service import PluginService

from tldw_chatbook.Agents.hooks_v2 import parse_handlers
from tldw_chatbook.Agents.hooks_v2.models import HookEvent, HookHandler
from tldw_chatbook.Agents.hooks_v2.ownership import HostProcessOwner

from .admission import PluginUnavailable, RunPluginSnapshot
from .data_cleanup import root_ref


def hook_id(snapshot: RunPluginSnapshot, component_id: str) -> str:
    """Use installation identity, never a display name, as the scheduler key."""
    return (
        "plugin_hook_"
        + hashlib.sha256(
            f"{snapshot.installation_id}:{component_id}".encode()
        ).hexdigest()[:48]
    )


def _parse_hook_declaration(
    inspection: PackageInspection, installation_id: str, component_id: str
) -> HookHandler:
    """Parse an occurrence only; its original cwd remains in authenticated metadata."""
    raw = json.loads(inspection.inventory[component_id].definition_json)
    # Native package cwd is not standalone cwd. It is withheld only from this
    # non-executable occurrence, so missing bytes cannot erase a requirement.
    if raw.get("type") == "command":
        raw.pop("cwd", None)
    handler = parse_handlers(
        [raw], declared_variable_names=set(json.loads(inspection.variables_json))
    )[0]
    return handler.model_copy(
        update={
            "id": "plugin_hook_"
            + hashlib.sha256(f"{installation_id}:{component_id}".encode()).hexdigest()[
                :48
            ]
        }
    )


def project_hook_definition(
    inspection: PackageInspection,
    installation_id: str,
    component_id: str,
    roots: list[dict],
    mappings: tuple[dict, ...] | list[dict],
) -> HookHandler:
    """Rejoin the original declaration under current execution and mapping gates."""
    from .variables import consumed

    if "PLUGIN_DATA" in consumed(inspection.inventory[component_id]):
        shared = [row for row in roots if row["workspace_id"] is None]
        if len(shared) != 1:
            raise PluginUnavailable("plugin_data_root_unavailable")
    handler = _parse_hook_declaration(inspection, installation_id, component_id)
    raw = json.loads(inspection.inventory[component_id].definition_json)
    raw["id"] = handler.id
    if handler.type == "command":
        root = Path(inspection.materialized_identity)
        if not root.is_absolute():
            raise PluginUnavailable("plugin_hook_cwd_invalid")
        cwd = (root / raw.get("cwd", ".")).resolve()
        if not cwd.is_relative_to(root.resolve()):
            raise PluginUnavailable("plugin_hook_cwd_invalid")
        raw["cwd"] = str(cwd)
    else:
        server = raw["server"]
        server = server if server.startswith("mcp:") else "mcp:" + server
        targets = [
            row
            for row in mappings
            if row["component_id"] == server
            and row["kind"] == "tool"
            and row["target_reference"].endswith("::" + raw["tool"])
        ]
        if len(targets) != 1:
            raise PluginUnavailable("plugin_hook_tool_mapping_required")
        raw["server"], raw["tool"] = targets[0]["target_reference"].split("::", 1)
    # Public parser still proves every executable cwd is existing and absolute.
    return parse_handlers(
        [raw], declared_variable_names=set(json.loads(inspection.variables_json))
    )[0]


def register_hook_definitions(snapshot: RunPluginSnapshot) -> tuple[HookHandler, ...]:
    """Keep restrictive occurrences separate from executable projections."""
    return tuple(
        (
            _parse_hook_declaration(snapshot.inspection, snapshot.installation_id, key)
            if key in snapshot.unavailable_hooks
            else project_hook_definition(
                snapshot.inspection,
                snapshot.installation_id,
                key,
                json.loads(snapshot.data_roots_json),
                json.loads(snapshot.mappings_json),
            )
        )
        for key in (*snapshot.selection, *snapshot.unavailable_hooks)
        if snapshot.inspection.inventory[key].kind == "hook"
    )


class NativeHooks:
    """One immutable native hook set; standalone hooks retain their existing owner."""

    def __init__(
        self,
        service: PluginService,
        snapshots: tuple[RunPluginSnapshot, ...],
        *,
        scratch_manager=None,
        scratch_snapshot=None,
    ) -> None:
        if (scratch_manager is None) != (scratch_snapshot is None):
            raise ValueError("plugin_scratch_capability_pair_required")
        self.scratch_manager, self.scratch_snapshot = scratch_manager, scratch_snapshot
        self.service = service
        self.snapshots = snapshots
        self.definitions = tuple(
            handler
            for snapshot in snapshots
            for handler in register_hook_definitions(snapshot)
        )
        self.owners = {
            hook_id(snapshot, key): (snapshot, key)
            for snapshot in snapshots
            for key in (*snapshot.selection, *snapshot.unavailable_hooks)
            if snapshot.inspection.inventory[key].kind == "hook"
        }
        self.unavailable = frozenset(
            hook_id(snapshot, key)
            for snapshot in snapshots
            for key in snapshot.unavailable_hooks
        )
        self.standalone = HostProcessOwner()
        self.engine = None

    @property
    def signature(self) -> tuple:
        return (
            self.definitions,
            self.scratch_snapshot,
            id(self.scratch_manager) if self.scratch_manager is not None else None,
            tuple(
                (
                    s.installation_id,
                    s.revision_digest,
                    s.workspace_id,
                    s.selection,
                    s.unavailable_hooks,
                    s.mappings_json,
                    s.generations,
                    s.live_generations,
                    s.data_roots_json,
                    s.root_epochs,
                )
                for s in self.snapshots
            ),
        )

    def requirements(
        self, installation_id: str, component_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        snapshot = next(
            (s for s in self.snapshots if s.installation_id == installation_id), None
        )
        if snapshot is None:
            raise PluginUnavailable("plugin_hook_dependency_unknown")
        required, visited = set(), set()

        def visit(key):
            if key in visited:
                return
            if key not in (*snapshot.selection, *snapshot.unavailable_hooks):
                raise PluginUnavailable("plugin_hook_dependency_unavailable")
            visited.add(key)
            component = snapshot.inspection.inventory[key]
            if component.kind == "hook":
                required.add(hook_id(snapshot, key))
            for dependency in component.dependencies:
                visit(dependency)

        for component_id in component_ids:
            for dependency in snapshot.inspection.inventory[component_id].dependencies:
                visit(dependency)
        return tuple(sorted(required))

    def dependency_required(self, handler: HookHandler, event: HookEvent) -> bool:
        owner = self.owners.get(handler.id)
        if owner is None:
            return False
        snapshot = owner[0]
        return handler.id in self.requirements(
            snapshot.installation_id, snapshot.selection
        )

    def definition_requirements(
        self, definition: ToolDefinitionSnapshot, registry: ToolCatalogRegistry
    ) -> tuple[str, ...] | None:
        from .mcp_provider import MCPProviderGroup, PluginMCPProvider

        owner = registry.resolve_owner_for_name(definition.name)
        provider = getattr(owner[1], "_provider", owner[1]) if owner else None
        if isinstance(provider, MCPProviderGroup):
            provider = provider.provider_for(definition.name)
        if isinstance(provider, PluginMCPProvider):
            entry = provider._entry_by_llm_name.get(definition.name)
            if entry is None:
                return None
            mapping = provider._tool_mappings.get(entry[0].tool_id)
            if mapping is None:
                return None
            return self.requirements(
                provider.snapshot.installation_id, (mapping["component_id"],)
            )
        from tldw_chatbook.Agents.tool_catalog import SkillToolProvider

        from .components import component_summary

        if isinstance(provider, SkillToolProvider):
            for snapshot in self.snapshots:
                for key in snapshot.selection:
                    component = snapshot.inspection.inventory[key]
                    if (
                        component.kind == "skill"
                        and component_summary(
                            snapshot.installation_id,
                            snapshot.inspection,
                            key,
                            snapshot.alias,
                        )["tool_name"]
                        == definition.name
                    ):
                        return self.requirements(snapshot.installation_id, (key,))
        return () if provider is not None else None

    def requirements_for(
        self,
        definition: ToolDefinitionSnapshot | None,
        registry: ToolCatalogRegistry,
        messages: list[dict],
    ) -> tuple[str, ...] | None:
        """Join only dependencies of the actual definition and live material."""
        from tldw_chatbook.Agents.agent_models import PluginContextText

        required = set()
        if definition is not None:
            ids = self.definition_requirements(definition, registry)
            if ids is None:
                return None
            required.update(ids)
        for row in messages:
            content = row.get("content")
            parts = (
                [content]
                if not isinstance(content, list)
                else [part.get("text") for part in content if isinstance(part, dict)]
            )
            for text in parts:
                if not isinstance(text, PluginContextText):
                    continue
                for origin in text.checked_origins():
                    snapshot = next(
                        (
                            s
                            for s in self.snapshots
                            if s.installation_id == origin.installation_id
                            and s.revision_digest == origin.revision
                        ),
                        None,
                    )
                    if snapshot is None:
                        return None
                    required.update(
                        self.requirements(
                            origin.installation_id, (origin.component_id,)
                        )
                    )
        return tuple(sorted(required))

    def project_event(self, handler: HookHandler, event: HookEvent) -> HookEvent:
        owner = self.owners.get(handler.id)
        return (
            event.model_copy(
                update={"owner_installation_id": None, "owner_component_id": None}
            )
            if owner is None
            else event.model_copy(
                update={
                    "owner_installation_id": owner[0].installation_id,
                    "owner_component_id": owner[1],
                }
            )
        )

    def authority(self, handler: HookHandler, event: HookEvent, stage: str) -> bool:
        if handler.id in self.unavailable:
            raise PluginUnavailable("plugin_requirement_not_executable")
        snapshot, component = self.owners[handler.id]
        self.service.check_effect_actor(snapshot, component, run_id=event.run_id)
        self.service._call_from_agent(
            lambda: self.service._admission.check(snapshot, component)
        )
        return True

    def effects_current(
        self, handler: HookHandler, event: HookEvent, stage: str
    ) -> bool:
        if handler.id in self.unavailable:
            raise PluginUnavailable("plugin_requirement_not_executable")
        snapshot, component = self.owners[handler.id]
        try:
            self.service.check_effect_actor(snapshot, component, run_id=event.run_id)
        except PermissionError:
            return False
        return self.service.published_current(snapshot)

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

        if handler.id in self.unavailable:
            raise PluginUnavailable("plugin_requirement_not_executable")
        owner = self.owners.get(handler.id)
        if owner is None:
            return None
        snapshot, key = owner

        self.service.check_effect_actor(snapshot, key, run_id=event.run_id)

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
            package_executable = (root / token).resolve()
            if not package_executable.is_relative_to(root.resolve()):
                raise PluginUnavailable("plugin_executable_unavailable")
            executable = str(package_executable)
        elif Path(token).is_absolute():
            executable = token
        else:
            host_cwd = Path.cwd()
            lookup_path = host_environment.get("PATH", "")
            if not lookup_path:
                raise PluginUnavailable("plugin_executable_unavailable")
            # Anchor empty and relative entries before applying child cwd/PATH.
            lookup_path = os.pathsep.join(
                str(host_cwd / entry) for entry in lookup_path.split(os.pathsep)
            )
            executable = shutil.which(token, path=lookup_path)
            if executable:
                executable = str(host_cwd / executable)
        if not executable:
            raise PluginUnavailable("plugin_executable_unavailable")
        declared = json.loads(snapshot.inspection.inventory[key].definition_json)
        cwd = (root / declared.get("cwd", ".")).resolve()
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

    def process_owner(
        self,
        handler: HookHandler,
        event: HookEvent,
        cancel: Callable[[], None],
    ) -> _NativeHookProcessOwner | None:
        """Capture one job's exact owner and cancellation callback."""
        if handler.id in self.unavailable:
            raise PluginUnavailable("plugin_requirement_not_executable")
        owner = self.owners.get(handler.id)
        if owner is None:
            return None
        snapshot, component = owner
        if (event.owner_installation_id, event.owner_component_id) != (
            snapshot.installation_id,
            component,
        ):
            raise PluginUnavailable("plugin_hook_owner_changed")
        return _NativeHookProcessOwner(
            self,
            snapshot,
            component,
            cancel,
            scratch_manager=(
                self.scratch_manager if snapshot.workspace_id is None else None
            ),
            scratch_snapshot=(
                self.scratch_snapshot if snapshot.workspace_id is None else None
            ),
        )

    def render_context(self, event: HookEvent, handler_id: str, block) -> str:
        from tldw_chatbook.Agents.agent_models import (
            PluginContextOrigin,
            PluginContextText,
        )
        from tldw_chatbook.Agents.hooks_v2.context import ContextLedger

        text = ContextLedger.render(event, handler_id, block)
        owner = self.owners.get(handler_id)
        if owner is None:
            return text
        snapshot, component = owner
        return PluginContextText(
            text,
            (
                PluginContextOrigin(
                    snapshot.installation_id,
                    component,
                    snapshot.revision_digest,
                    len(text.encode("utf-8")),
                ),
            ),
        )

    def reserve_launch(self, event: HookEvent) -> str:
        if event.owner_installation_id is None:
            return "host:" + self.standalone.reserve_launch(event)
        snapshot = next(
            s
            for s in self.snapshots
            if s.installation_id == event.owner_installation_id
        )
        component = event.owner_component_id
        return self._reserve_launch(snapshot, component, event, None)

    def _reserve_launch(self, snapshot, component, event, cancel_job):
        self.service.check_effect_actor(snapshot, component, run_id=event.run_id)
        roots = tuple(root_ref(row) for row in json.loads(snapshot.data_roots_json))

        def reserve():
            from .service import PluginRunOwnership

            coordinator = self.service._coordinator
            self.service._admission.check(snapshot, component)
            token = coordinator.owner.reserve_launch(
                "hook:" + event.event_id + ":" + component,
                snapshot.installation_id,
                snapshot.workspace_id,
                snapshot.revision_digest,
                roots=roots,
                root_coverage="known" if roots else "qualified_none",
            )

            def cancel():
                if cancel_job is not None:
                    cancel_job()
                    return

                def stop():
                    for job in tuple(self.engine.processes.records.values()):
                        if job.token == "plugin:" + token:
                            job.stop()

                self.engine.loop.call_soon_threadsafe(stop)

            record = PluginRunOwnership(
                snapshot.installation_id,
                snapshot.workspace_id,
                snapshot.revision_digest,
                snapshot.run_id,
                event.run_id or snapshot.run_id,
                None,
                token,
                cancel,
                threading.Event(),
                turn_id=event.turn_id,
                generations=snapshot.generations,
                component_ceiling=snapshot.selection,
            )
            try:
                if roots:
                    coordinator.root_usage.retain_cancel(token, cancel)
                with self.service.fences.live_lock:
                    self.service.fences.check_snapshot(snapshot)
                    self.service.fences.runs["hook:" + token] = record
            except BaseException:
                coordinator.owner.settle_process(token, True)
                raise
            return token

        return "plugin:" + self.service._call_from_agent(reserve)

    def publish_process(self, token: str, provenance: dict) -> None:
        kind, value = token.split(":", 1)
        if kind == "host":
            self.standalone.publish_process(value, provenance)
            return
        self.service._call_from_agent(
            lambda: self.service._coordinator.owner.publish_process(value, provenance)
        )

    def settle_process(self, token: str, confirmed: bool) -> None:
        kind, value = token.split(":", 1)
        if kind == "host":
            self.standalone.settle_process(value, confirmed)
            return

        def settle():
            self.service._coordinator.owner.settle_process(value, confirmed)
            if confirmed:
                with self.service.fences.live_lock:
                    record = self.service.fences.runs.pop("hook:" + value, None)
                    if record is not None:
                        record.completed.set()

        self.service._call_from_agent(settle, terminal=True)


class _NativeHookProcessOwner:
    """One actual job's view of the existing native process/root owner."""

    def __init__(
        self,
        facade: NativeHooks,
        snapshot: RunPluginSnapshot,
        component_id: str,
        cancel: Callable[[], None],
        *,
        scratch_manager=None,
        scratch_snapshot=None,
    ) -> None:
        import asyncio

        self.scratch_manager, self.scratch_snapshot = scratch_manager, scratch_snapshot
        self.scratch_leases = {}
        self.facade = facade
        self.snapshot = snapshot
        self.component_id = component_id
        loop = asyncio.get_running_loop()
        self.cancel = lambda: loop.call_soon_threadsafe(cancel)
        self.event = None

    def reserve_launch(self, event: HookEvent) -> str:
        self.event = event
        lease = (
            self.scratch_manager.lease(self.scratch_snapshot)
            if self.scratch_manager is not None
            else None
        )
        if lease is not None:
            lease.__enter__()
        try:
            token = self.facade._reserve_launch(
                self.snapshot, self.component_id, event, self.cancel
            )
        except BaseException:
            if lease is not None:
                lease.__exit__(None, None, None)
            raise
        if lease is not None:
            self.scratch_leases[token] = lease
        return token

    def publish_process(self, token: str, provenance: dict) -> None:
        service = self.facade.service

        def publish():
            service.check_effect_actor(
                self.snapshot, self.component_id, run_id=self.event.run_id
            )
            service._admission.check(self.snapshot, self.component_id)
            kind, value = token.split(":", 1)
            if kind != "plugin":
                raise PluginUnavailable("plugin_hook_owner_changed")
            service._coordinator.owner.publish_process(value, provenance)

        service._call_from_agent(publish)

    def settle_process(self, token: str, confirmed: bool) -> None:
        self.facade.settle_process(token, confirmed)
        if confirmed:
            lease = self.scratch_leases.pop(token, None)
            if lease is not None:
                lease.__exit__(None, None, None)
