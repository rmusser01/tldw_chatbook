"""Lightweight primary-generation boundary; rule services load on first use."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import Any


def checked_generation(
    method: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Check only real primary generation, after its existing cleanup finishes."""

    @functools.wraps(method)
    async def wrapped(controller, *args, **kwargs):
        service = getattr(controller, "response_rules", None)
        if service is None:
            factory = getattr(controller, "response_rules_factory", None)
            service = factory() if factory is not None else None
        if service is None:
            return await method(controller, *args, **kwargs)
        message_id = kwargs["assistant_message_id"]
        boundary = service.begin_generation(
            message_id,
            kwargs["resolution"],
            configuration=getattr(kwargs.get("turn_context"), "configuration", None),
            direct_plain=True,
            automatic_authorization=kwargs.get("wake_authorization"),
        )
        if boundary is None:
            return await method(controller, *args, **kwargs)
        session_id = boundary
        try:
            result = await method(controller, *args, **kwargs)
            message = controller.store.get_message(message_id)
            if message.status == "complete" and not message.generation_metadata:
                hook_owner = getattr(controller, "_hooks_v2_submissions", {}).get(
                    asyncio.current_task()
                )
                if hook_owner is not None and hook_owner[0] is not None:
                    await service.queue.settle_hook_parent(session_id, hook_owner[1])
                await service.finish_generation(session_id, message_id)
        except BaseException:
            service.discard_direct_repair(session_id)
            raise
        finally:
            service.release_generation(session_id, message_id)
        await service.finish_direct_repair(session_id)
        return result

    return wrapped
