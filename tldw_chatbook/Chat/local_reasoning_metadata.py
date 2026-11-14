"""Bounded optional metadata reads for local reasoning and native tools."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json

import httpx

METADATA_MAX_BYTES = 262144


async def read_local_reasoning_metadata(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Mapping[str, str] | None,
    model: str | None = None,
) -> tuple[str | None, bool]:
    """Read template text and the server's two required tool capability flags.

    Args:
        client: The gateway's existing HTTP client.
        url: Resolved metadata endpoint.
        headers: The gateway's resolved authorization headers.
        model: Selected model for servers with model-specific properties.

    Returns:
        Untrusted template text and whether both tool capability flags are true.

    Raises:
        TimeoutError: The complete read exceeds one second.
        httpx.HTTPError: The endpoint is unavailable or rejects the request.
        ValueError: The response exceeds its bound or contains invalid JSON.
    """
    async with asyncio.timeout(1.0):
        async with client.stream(
            "GET",
            url,
            headers=headers,
            params={"model": model, "autoload": "false"} if model is not None else None,
            timeout=1.0,
        ) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > METADATA_MAX_BYTES:
                    raise ValueError("Template metadata exceeds size limit.")
            payload = json.loads(data)
    metadata = payload if isinstance(payload, dict) else {}
    raw_template = metadata.get("chat_template")
    template = raw_template if isinstance(raw_template, str) else None
    caps = metadata.get("chat_template_caps", {})
    tools = (
        isinstance(caps, Mapping)
        and caps.get("supports_tool_calls") is True
        and caps.get("supports_tools") is True
    )
    return template, tools
