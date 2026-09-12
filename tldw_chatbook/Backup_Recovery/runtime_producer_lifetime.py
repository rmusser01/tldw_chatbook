"""Finite admission for the installed Sync and local MCP service calls."""

import asyncio
import threading
import time
from contextlib import contextmanager
from functools import wraps

from tldw_chatbook.Utils.platform_files import os

from .bootstrap import RecoveryRequired


class ProducerLifetime:
    """Retain complete accepted call chains until their original methods return."""

    def __init__(self):
        self.closed = False
        self.calls = {}
        self.lock = threading.RLock()

    @contextmanager
    def operation(self):
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        identity = (os.getpid(), threading.current_thread(), task)
        with self.lock:
            depth = self.calls.get(identity, 0)
            if self.closed and not depth:
                raise RecoveryRequired("runtime_producer_paused")
            self.calls[identity] = depth + 1
        try:
            yield
        finally:
            with self.lock:
                if depth:
                    self.calls[identity] = depth
                else:
                    del self.calls[identity]

    def close(self):
        with self.lock:
            self.closed = True

    async def drain(self, deadline):
        while True:
            with self.lock:
                if not self.closed:
                    raise RecoveryRequired("runtime_producer_not_closed")
                if not self.calls:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.01, remaining))

    def resume(self):
        with self.lock:
            if self.calls:
                raise RecoveryRequired("runtime_work_not_settled")
            self.closed = False


def producer_call(function):
    """Keep intake fenced across awaits, including accepted result/error writes."""

    @wraps(function)
    async def call(self, *args, **kwargs):
        with self._producer_lifetime.operation():
            return await function(self, *args, **kwargs)

    return call
