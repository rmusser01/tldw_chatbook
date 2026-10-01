"""Native skill projections and the existing run-scoped skill tool provider."""

import hashlib
import json

from tldw_chatbook.Agents.tool_catalog import SkillToolProvider

from .admission import PluginUnavailable, RunPluginSnapshot
from .context import instruction_block
from .package_files import capture_package


class PluginSkillProvider(SkillToolProvider):
    """Catalog-only provider; invocation still belongs to Console's skill runner."""

    def __init__(self, entries):
        super().__init__(
            [
                dict(row, name=row.get("tool_name", row["name"]))
                for row in entries
                if not row.get("disable_model_invocation", False)
            ]
        )


def owned_identifier(name: str) -> bool:
    """Reserved owned identities can never become standalone directories."""
    return isinstance(name, str) and (":" in name or name.startswith("plugin_"))


def skill_summary(
    installation_id: str, inspection, component_id: str, alias: str
) -> dict:
    """Project provenance and constrained metadata without instruction bodies."""
    component = inspection.inventory[component_id]
    definition = json.loads(component.definition_json)
    metadata = definition.get("metadata", {})
    context = metadata.get("context", "inline")
    user = metadata.get("user_invocable", "true")
    manual = metadata.get("disable_model_invocation", "false")
    valid = (
        context in {"inline", "fork"}
        and user in {"true", "false"}
        and manual in {"true", "false"}
    )
    if not valid:
        context, user, manual = "inline", "false", "true"
    name = f"{alias}:{component.local_id}"
    tool_name = (
        "plugin_"
        + hashlib.sha256(f"{installation_id}:{component_id}".encode()).hexdigest()[:56]
    )
    return {
        "name": name,
        "tool_name": tool_name,
        "record_id": f"plugin:{installation_id}:{component_id}",
        "plugin_owned": True,
        "plugin_installation_id": installation_id,
        "plugin_component_id": component_id,
        "plugin_revision": inspection.effective_digest,
        "description": definition.get("description", ""),
        "context": context,
        "user_invocable": user == "true",
        "disable_model_invocation": manual == "true",
        "argument_hint": metadata.get("argument_hint"),
        "allowed_tools": (
            []
            if "allowed-tools" in definition and not definition["allowed-tools"]
            else None
        ),
        "definition_digest": inspection.effective_digest,
        "trust_blocked": True,
        "trust_status": "plugin_managed",
        "plugin_blockers": ([] if valid else ["plugin_skill_metadata_invalid"])
        + (["plugin_skill_model_unsupported"] if "model" in metadata else []),
        "backend": "local",
    }


def render_skill(
    snapshot: RunPluginSnapshot, component_id: str, summary: dict, args: str
) -> dict:
    """Read exact captured package bytes and retain untrusted source attribution."""
    from pathlib import Path

    capture = capture_package(Path(snapshot.inspection.materialized_identity))
    if capture.errors or capture.digest != snapshot.inspection.content_digest:
        raise PluginUnavailable("plugin_material_changed")
    component = snapshot.inspection.inventory[component_id]
    lines = capture.read(component.path).decode("utf-8").splitlines()
    end = next(index for index, line in enumerate(lines[1:], 1) if line == "---")
    body = "\n".join(lines[end + 1 :]).strip()
    prefix = str(Path(component.path).parent) + "/"
    references = [
        {
            "path": path[len(prefix) :],
            "size": len(member.data),
            "is_text": b"\x00" not in member.data,
        }
        for path, member in sorted(capture.files.items())
        if path.startswith(prefix) and path != component.path
    ]
    return {
        **({"reference_files": references} if references else {}),
        "skill_name": summary["name"],
        "record_id": summary["record_id"],
        "plugin_owned": True,
        "plugin_installation_id": snapshot.installation_id,
        "plugin_component_id": component_id,
        "plugin_revision": snapshot.revision_digest,
        "rendered_prompt": instruction_block(
            snapshot.installation_id,
            component_id,
            snapshot.revision_digest,
            body,
            args,
            allowed_tools=summary["allowed_tools"],
        ),
        "allowed_tools": summary["allowed_tools"],
        "execution_mode": summary["context"],
        "model_override": None,
        "fork_output": None,
    }
