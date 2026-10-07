"""Manual package instructions; invocation arguments remain literal data."""

import json

from .admission import PluginUnavailable, RunPluginSnapshot
from .components import component_body
from .context import instruction_block
from .package_files import PackageFileError, parse_document


def render_command(
    snapshot: RunPluginSnapshot, component_id: str, arguments: dict
) -> tuple[dict, ...]:
    """Return separate attributed instruction and literal argument blocks."""
    component = snapshot.inspection.inventory[component_id]
    expected = json.loads(component.definition_json).get("arguments", [])
    if (
        component.kind != "command"
        or not isinstance(arguments, dict)
        or set(arguments) != set(expected)
        or any(not isinstance(value, str) for value in arguments.values())
    ):
        raise PluginUnavailable("plugin_command_arguments_invalid")
    body = component_body(snapshot, component_id)
    values = (
        body,
        "Literal invocation arguments: " + json.dumps(arguments, ensure_ascii=False),
    )
    return tuple(
        {
            "role": "user",
            "content": instruction_block(
                snapshot.installation_id, component_id, snapshot.revision_digest, value
            ),
        }
        for value in values
    )


def command_arguments(text: str) -> dict:
    """Use the existing strict bounded JSON parser, never a shell parser."""
    try:
        return parse_document((text.strip() or "{}").encode("utf-8"))
    except (PackageFileError, UnicodeError):
        raise PluginUnavailable("plugin_command_arguments_invalid") from None
