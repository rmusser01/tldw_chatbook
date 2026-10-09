"""Compose a Console run's MCP tools off the UI loop (TASK-33620.15.1).

Measured on 46c3959526 (live, Anthropic haiku, 160x45, ten sends): between
a send's acceptance and its provider call, run start composed the MCP
provider on the UI loop. The kill switch was read three times (twice for the
MCP catalog, once for the local tools), and the local server catalog, the
built-in inventory and the permission states were each read through the MCP
stores' storage-admission scope. Main-thread samples put 63-130 ms busy
stretches there, and hover waits over them reached 104-297 ms.

Every read stays where and when it was: at run start, after the durable
commit, read fresh (nothing is cached from admission or an earlier run).
They now run in ONE worker hop, so the UI loop keeps handling input while
they run, and the kill switch is read once there for the whole run start
(MCP and local tools) instead of three times within the same step: off the
loop each store read waits on the GIL, and the hop is on the way to the
provider call. ``MCPToolProvider.compose_catalog`` is a coroutine whose only
await (``local_external_catalog``) does synchronous store reads, so the hop
drives it on its own short-lived event loop. Tool calls still run on the UI
loop: the provider keeps the ``main_loop`` it was built with.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class LocalKillSwitchRead:
    """The run-start kill-switch read the local tools use, made in the hop.

    ``read`` stays False when no hop ran (no MCP service, or a caller that
    composes differently); the local provider then reads the switch itself.
    """

    read: bool = False
    value: bool = False
    error: Exception | None = None


def compose_run_mcp_provider(
    service: Any,
    provider: Any,
    local_read: LocalKillSwitchRead | None = None,
) -> Any | None:
    """Read MCP authority and compose this run's catalog (worker thread).

    Args:
        service: The app's unified MCP service.
        provider: The uncomposed provider (built with the UI loop).
        local_read: Filled with this run-start kill-switch read, for the local
            tools (TASK-33620.15.1: one read decides MCP and local tools).

    Returns:
        The composed provider, or None when MCP is not offered this run:
        the kill switch is on, or a read failed (fail closed).
    """
    error: Exception | None = None
    try:
        engaged = bool(service.get_kill_switch())
    except Exception as caught:  # noqa: BLE001 -- fail closed to "no MCP this run"
        logger.opt(exception=caught).warning(
            "ConsoleChatController: get_kill_switch failed; skipping MCP this run"
        )
        engaged, error = True, caught
    if engaged:
        provider = None
    else:
        try:
            asyncio.run(provider.compose_catalog(kill_switch_engaged=False))
        except Exception:  # noqa: BLE001 -- a composition failure must not abort the send
            logger.opt(exception=True).warning(
                "ConsoleChatController: MCP compose_catalog failed; skipping MCP this run"
            )
            provider = None
    if local_read is not None:
        # The same run-start read decides the local tools (it was a second
        # read a moment later; under GIL contention each costs tens of ms).
        local_read.value, local_read.error, local_read.read = engaged, error, True
    return provider


def local_tools_kill_switch(
    service: Any, local_read: LocalKillSwitchRead | None
) -> bool:
    """Whether the local tools are switched off this run (fail closed).

    Uses the hop's read when it ran; otherwise reads the switch here, as the
    local provider always did.
    """
    error = None
    if local_read is not None and local_read.read:
        error, engaged = local_read.error, local_read.value
    else:
        try:
            engaged = bool(service.get_kill_switch())
        except Exception as caught:  # noqa: BLE001
            error = caught
    if error is not None:
        logger.opt(exception=error).warning(
            "ConsoleChatController: get_kill_switch failed; skipping local tools this run"
        )
        return True
    return engaged
