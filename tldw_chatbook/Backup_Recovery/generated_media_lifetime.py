"""Finite generated-image/video file operations drained before storage pauses."""

import asyncio
from contextlib import ExitStack, contextmanager
from functools import wraps

from ..TTS._async_lifecycle import join_retained_task
from .runtime_producer_lifetime import ProducerLifetime
from .storage_admission import acquire_storage


class GeneratedMediaLifetime:
    """Retain actual file operations, including their native asynchronous work."""

    def __init__(self):
        self.calls = ProducerLifetime()

    @contextmanager
    def operation(self, paths):
        with self.calls.operation(), ExitStack() as stack:
            for path in sorted(set(paths)):
                stack.enter_context(acquire_storage(path))
            yield

    def _maintenance_close_admission(self):
        self.calls.close()

    async def _maintenance_drain(self, deadline):
        return await self.calls.drain(deadline)

    def _maintenance_resume(self):
        self.calls.resume()


participant = GeneratedMediaLifetime()


def sync_operation(function):
    """Enclose the declared owner's complete synchronous public file call."""

    @wraps(function)
    def call(self, *args, **kwargs):
        with participant.operation(
            self._backup_sources(function.__name__, args, kwargs)
        ):
            return function(self, *args, **kwargs)

    return call


def async_operation(function):
    """Do not release file admission while an aiofiles worker can still write."""

    @wraps(function)
    async def call(self, *args, **kwargs):
        async def retained():
            with participant.operation(
                self._backup_sources(function.__name__, args, kwargs)
            ):
                return await function(self, *args, **kwargs)

        completion = asyncio.create_task(retained())
        await join_retained_task(completion)
        return completion.result()

    return call
