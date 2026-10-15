"""Manual native prompt commands keep arguments as separately attributed data."""

import json

from .admission import PluginUnavailable
from .components import NativeComponent, component_body
from .context import instruction_block
from .models import ComponentRecord


def render_command(
    component: ComponentRecord, arguments: dict[str, str]
) -> tuple[dict, ...]:
    """Render one command without evaluating or rescanning argument syntax."""
    if not isinstance(component, NativeComponent) or component.kind != "command":
        raise PluginUnavailable("plugin_command_not_admitted")
    definition = json.loads(component.definition_json)
    if (
        not isinstance(arguments, dict)
        or set(arguments) != set(definition.get("arguments", ()))
        or any(not isinstance(value, str) for value in arguments.values())
    ):
        raise PluginUnavailable("plugin_command_arguments_invalid")
    snapshot = component.snapshot
    return tuple(
        {
            "role": "user",
            "content": instruction_block(
                snapshot.installation_id,
                component.component_id,
                snapshot.revision_digest,
                body,
                args,
            ),
        }
        for body, args in (
            (component_body(component), ""),
            (
                "Command arguments (literal data)",
                json.dumps(arguments, ensure_ascii=False),
            ),
        )
    )
