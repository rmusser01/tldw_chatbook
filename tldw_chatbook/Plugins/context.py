"""Attributed, whole-block untrusted package instruction context."""

import json

from tldw_chatbook.Agents.agent_models import PluginContextOrigin, PluginContextText

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
    return PluginContextText(
        block,
        (
            PluginContextOrigin(
                installation_id, component_id, revision, len(block.encode("utf-8"))
            ),
        ),
    )


def check_context_budget(blocks: list[str]) -> None:
    """Refuse selected material whole instead of changing it by truncation."""
    if (
        any(len(block.encode("utf-8")) > BLOCK_BYTES for block in blocks)
        or sum(len(block.encode("utf-8")) for block in blocks) > SEND_BYTES
    ):
        raise PluginUnavailable("plugin_context_send_too_large")


def check_send_context(messages: list[dict]) -> list[dict]:
    """Check present host-attributed blocks and erase sidecars before transport."""
    total = 0
    result = []
    for row in messages:
        content = row.get("content")
        if isinstance(content, PluginContextText):
            for origin in content.checked_origins():
                if not 0 <= origin.byte_count <= BLOCK_BYTES:
                    raise PluginUnavailable("plugin_context_block_too_large")
                total += origin.byte_count
            row = {**row, "content": str(content)}
        result.append(row)
    if total > SEND_BYTES:
        raise PluginUnavailable("plugin_context_send_too_large")
    return result
