"""Native hook projections adapt the existing scheduler and durable process owner."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
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


def project_hook_definition(
    inspection: PackageInspection,
    installation_id: str,
    component_id: str,
    roots: list[dict],
    mappings: tuple[dict, ...] | list[dict],
) -> HookHandler:
    """Validate a definition against retained paths and reviewed references only."""
    shared = next((row for row in roots if row["workspace_id"] is None), None)
    variables = {"PLUGIN_ROOT": inspection.materialized_identity}
    if shared is not None:
        variables["PLUGIN_DATA"] = shared["path"]
    for key, declaration in json.loads(inspection.variables_json).items():
        if "default" in declaration:
            value = declaration["default"]
            variables[key] = json.dumps(value) if type(value) is bool else str(value)

    def expand(value):
        def replace(match):
            if match[1] not in variables:
                raise PluginUnavailable("plugin_configuration_unresolved")
            return variables[match[1]]

        return re.sub(r"\$\{([A-Z][A-Z0-9_]*)\}", replace, value)

    raw = json.loads(inspection.inventory[component_id].definition_json)
    raw["id"] = (
        "plugin_hook_"
        + hashlib.sha256(f"{installation_id}:{component_id}".encode()).hexdigest()[:48]
    )
    if raw["type"] == "command":
        argv = [expand(value) for value in raw["argv"]]
        if argv[0].startswith("./"):
            argv[0] = str(Path(variables["PLUGIN_ROOT"]) / argv[0])
        else:
            argv[0] = shutil.which(argv[0])
        if not argv[0]:
            raise PluginUnavailable("plugin_executable_unavailable")
        raw["argv"] = argv
        raw["env"] = {key: expand(value) for key, value in raw.get("env", {}).items()}
        raw["cwd"] = str(Path(variables["PLUGIN_ROOT"]) / raw.get("cwd", "."))
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
    return parse_handlers([raw])[0]


def register_hook_definitions(snapshot: RunPluginSnapshot) -> tuple[HookHandler, ...]:
    """Project only selected definitions and exact captured host references."""
    return tuple(
        project_hook_definition(
            snapshot.inspection,
            snapshot.installation_id,
            key,
            json.loads(snapshot.data_roots_json),
            json.loads(snapshot.mappings_json),
        )
        for key in snapshot.selection
        if snapshot.inspection.inventory[key].kind == "hook"
    )


class NativeHooks:
    """One immutable native hook set; standalone hooks retain their existing owner."""

    def __init__(
        self, service: PluginService, snapshots: tuple[RunPluginSnapshot, ...]
    ) -> None:
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
            for key in snapshot.selection
            if snapshot.inspection.inventory[key].kind == "hook"
        }
        self.standalone = HostProcessOwner()
        self.engine = None

    @property
    def signature(self) -> tuple:
        return (
            self.definitions,
            tuple(
                (
                    s.installation_id,
                    s.revision_digest,
                    s.workspace_id,
                    s.selection,
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
            if key not in snapshot.selection:
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
        snapshot, component = self.owners[handler.id]
        self.service.check_effect_actor(snapshot, component, run_id=event.run_id)
        self.service._call_from_agent(
            lambda: self.service._admission.check(snapshot, component)
        )
        return True

    def effects_current(
        self, handler: HookHandler, event: HookEvent, stage: str
    ) -> bool:
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
        snapshot = owner[0]
        values = {"PLUGIN_ROOT": snapshot.inspection.materialized_identity}
        shared = next(
            (
                row
                for row in json.loads(snapshot.data_roots_json)
                if row["workspace_id"] is None
            ),
            None,
        )
        if shared is not None:
            values["PLUGIN_DATA"] = shared["path"]
        return values

    def reserve_launch(self, event: HookEvent) -> str:
        if event.owner_installation_id is None:
            return "host:" + self.standalone.reserve_launch(event)
        snapshot = next(
            s
            for s in self.snapshots
            if s.installation_id == event.owner_installation_id
        )
        component = event.owner_component_id
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
                    record = self.service.fences.runs.pop("hook:" + value)
                    record.completed.set()

        self.service._call_from_agent(settle, terminal=True)
