"""Attributed, whole-block untrusted package instruction context."""

import json

from .admission import PluginUnavailable

BLOCK_BYTES = 8 * 1024
SEND_BYTES = 32 * 1024


def instruction_block(
    installation_id: str, component_id: str, revision: str, body: str, args: str = ""
) -> str:
    """Keep package instructions and literal invocation arguments attributed."""
    block = (
        "<untrusted-plugin-context>\n"
        + json.dumps(
            {
                "installation_id": installation_id,
                "component_id": component_id,
                "revision": revision,
                "trust": "untrusted instructions",
                "instructions": body,
                "arguments": args,
            },
            ensure_ascii=False,
        )
        + "\n</untrusted-plugin-context>"
    )
    if len(block.encode("utf-8")) > BLOCK_BYTES:
        raise PluginUnavailable("plugin_context_block_too_large")
    return block


def check_context_budget(blocks: list[str]) -> None:
    """Refuse selected material whole instead of changing it by truncation."""
    if (
        any(len(block.encode("utf-8")) > BLOCK_BYTES for block in blocks)
        or sum(len(block.encode("utf-8")) for block in blocks) > SEND_BYTES
    ):
        raise PluginUnavailable("plugin_context_send_too_large")
