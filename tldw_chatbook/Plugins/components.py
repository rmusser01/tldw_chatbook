"""Selected native capability projections from one immutable admission."""

import hashlib
import json
from pathlib import Path

from .admission import PluginUnavailable, RunPluginSnapshot
from .models import ComponentRecord, PackageInspection
from .package_files import PackageCapture, capture_package


class PluginComponents:
    """Project already selected components; never discover or grant authority."""

    @staticmethod
    def project(snapshot: RunPluginSnapshot) -> tuple[ComponentRecord, ...]:
        return tuple(snapshot.inspection.inventory[key] for key in snapshot.selection)


def component_summary(
    installation_id: str,
    inspection: PackageInspection,
    component_id: str,
    alias: str,
    mappings: tuple[dict, ...] | list[dict] = (),
) -> dict:
    """Publish identity and constraints without reading instruction bodies."""
    from .skill_provider import skill_summary

    component = inspection.inventory[component_id]
    if component.kind == "skill":
        from .host_references import mapped_tools

        row = dict(
            skill_summary(installation_id, inspection, component_id, alias),
            plugin_kind="skill",
        )
        if mappings or not row["plugin_blockers"]:
            try:
                row["allowed_tools"] = mapped_tools(
                    installation_id, component, mappings
                )
            except PermissionError:
                pass  # Unmapped components remain visible; admission refuses them.
        return row
    definition = json.loads(component.definition_json)
    name = f"{alias}:{component.kind}:{component.local_id}"
    return {
        "name": name,
        "tool_name": "plugin_"
        + hashlib.sha256(f"{installation_id}:{component_id}".encode()).hexdigest()[:56],
        "record_id": f"plugin:{installation_id}:{component_id}",
        "plugin_owned": True,
        "plugin_kind": component.kind,
        "plugin_installation_id": installation_id,
        "plugin_component_id": component_id,
        "plugin_revision": inspection.effective_digest,
        "description": definition.get("description", ""),
        "context": "inline",
        "user_invocable": component.kind == "command"
        or (component.kind == "rule" and definition.get("mode") == "manual"),
        "disable_model_invocation": True,
        "allowed_tools": definition.get("tools")
        if isinstance(definition.get("tools"), list)
        else None,
        "plugin_rule_mode": definition.get("mode")
        if component.kind == "rule"
        else None,
        "definition_digest": inspection.effective_digest,
        "trust_blocked": True,
        "trust_status": "plugin_managed",
        "plugin_blockers": list(component.activation_blockers),
        "backend": "local",
    }


def capture_component(
    snapshot: RunPluginSnapshot, component_id: str
) -> tuple[PackageCapture, ComponentRecord]:
    """Read exact retained bytes only for a component in this captured ceiling."""
    if component_id not in snapshot.selection:
        raise PluginUnavailable("plugin_component_not_admitted")
    capture = capture_package(Path(snapshot.inspection.materialized_identity))
    if capture.errors or capture.digest != snapshot.inspection.content_digest:
        raise PluginUnavailable("plugin_material_changed")
    return capture, snapshot.inspection.inventory[component_id]


def component_body(snapshot: RunPluginSnapshot, component_id: str) -> str:
    """Preserve the complete validated Markdown body without interpolation."""
    capture, component = capture_component(snapshot, component_id)
    lines = capture.read(component.path).decode("utf-8").splitlines()
    if lines and lines[0] == "---":
        end = next(index for index, line in enumerate(lines[1:], 1) if line == "---")
        lines = lines[end + 1 :]
    return "\n".join(lines).strip()
