"""MCP execution admission from actual local settings, never imported grants."""

import asyncio
import os
import sys
import threading
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths

_active = ContextVar("mcp_execution", default=None)


class MCPActivationRequired(PermissionError):
    """Local recovery review is required before this MCP effect."""

    def __init__(self):
        super().__init__("mcp_activation_required")


def _identity():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), threading.get_ident(), task


def _sources(service):
    """Observe installed paths without loading credentials or writing settings."""
    selector = bootstrap.effective_config_path()
    config = sys.modules.get("tldw_chatbook.config")
    cached = getattr(config, "_CONFIG_CACHE", None)
    if getattr(config, "_CONFIG_CACHE_SOURCE", None) != selector:
        cached = None
    data = profile_paths.user_data_dir(cached or {})
    local = getattr(service, "local_service", service)
    store = getattr(local, "store", None)
    if store is None:
        store = getattr(service, "_definition_store", None)
    local_path = getattr(store, "_recovery_original_path", getattr(store, "path", data / "local_mcp_store.json"))
    permission = getattr(service, "_permission_store", None)
    paths = [
        ("config", selector),
        ("mcp.local", local_path),
        (
            "mcp.permissions",
            getattr(
                permission, "_recovery_original_path", getattr(permission, "path", Path(local_path).with_name("mcp_permissions.json"))
            ),
        ),
    ]
    for name, owner in (
        ("target_store", "mcp.targets"),
        ("context_store", "mcp.context"),
    ):
        store = getattr(service, name, None)
        path = getattr(store, "_recovery_original_path", getattr(store, "path", None))
        if path is not None:
            paths.append((owner, path))
    return tuple((owner, profile_paths.lexical_path(path)) for owner, path in paths)


@contextmanager
def execution(service, *, sources=()):
    """Hold real leases in this task/thread/PID through the accepted effect.

    Sources are observations, not permissions. Worker threads reacquire them;
    copied contexts cannot borrow another execution's accepted leases.
    """
    from tldw_chatbook.Backup_Recovery.activation import execution_scope
    from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    identity = _identity()
    active = _active.get()
    if active is not None and active[0] != identity:
        raise MCPActivationRequired()
    leases = dict(active[1]) if active else {}
    observed = tuple(
        dict.fromkeys(
            (active[2] if active else ()) + tuple(sources) + _sources(service)
        )
    )
    with ExitStack() as stack:
        try:
            for owner, path in observed:
                if path not in leases:
                    leases[path] = acquire_storage(path)
                    stack.callback(leases[path].close)
                if owner == "config":
                    _witnesses(path, leases[path])
                elif not stack.enter_context(
                    execution_scope((owner,), path, retained=leases[path])
                ):
                    raise MCPActivationRequired()
            from .recovery_activation import allowed

            if not allowed(observed, retained=leases):
                raise MCPActivationRequired()
        except (OSError, ValueError, TypeError, RuntimeError, AttributeError):
            raise MCPActivationRequired() from None
        token = _active.set((identity, leases, observed))
        try:
            yield
        finally:
            _active.reset(token)


def guarded(function):
    """Gate an actual async effect, preserving nested accepted admission."""

    @wraps(function)
    async def call(self, *args, **kwargs):
        with execution(self):
            return await function(self, *args, **kwargs)

    call._mcp_activation_guarded = True
    return call


def server_branch_guard(function):
    """Gate only the server branch of a local/server control-plane method."""

    @wraps(function)
    async def call(self, *args, **kwargs):
        if getattr(self, "selected_source", None) == "server":
            with execution(self):
                return await function(self, *args, **kwargs)
        return await function(self, *args, **kwargs)

    call._mcp_activation_guarded = True
    return call


async def in_worker(service, function, *args):
    """Worker admission survives cancellation of its asyncio waiter."""
    active = _active.get()
    sources = (
        active[2]
        if active is not None and active[0] == _identity()
        else _sources(service)
    )

    def run():
        token = _active.set(None)
        try:
            with execution(service, sources=sources):
                return function(*args)
        finally:
            _active.reset(token)

    return await asyncio.to_thread(run)


# These protocol calls only inspect saved local descriptors/status. Resource and
# prompt dispatch can execute code; they must enter admission even for local data.
_INSPECTION = frozenset(
    {"initialize", "status/get", "tools/list", "resources/list", "prompts/list"}
)


def request_guard(function):
    @wraps(function)
    async def call(self, method, *args, **kwargs):
        if str(method or "").strip() in _INSPECTION:
            return await function(self, method, *args, **kwargs)
        with execution(self):
            return await function(self, method, *args, **kwargs)

    return call


def _inspection_batch(requests):
    return isinstance(requests, (list, tuple)) and all(
        isinstance(request, Mapping)
        and str(request.get("method") or "").strip() in _INSPECTION
        for request in requests
    )


def batch_guard(function):
    @wraps(function)
    async def call(self, requests, *args, **kwargs):
        if _inspection_batch(requests):
            return await function(self, requests, *args, **kwargs)
        with execution(self):
            return await function(self, requests, *args, **kwargs)

    return call


def action_guard(function):
    @wraps(function)
    async def call(self, action_name, payload=None):
        if getattr(self, "selected_source", None) == "server":
            with execution(self):
                return await function(self, action_name, payload)
        inspection = isinstance(payload, Mapping) and (
            action_name == "runtime.request"
            and str(payload.get("method") or "").strip() in _INSPECTION
            or action_name == "runtime.batch"
            and _inspection_batch(payload.get("requests"))
        )
        if not inspection and action_name in {
            "profile.connect",
            "profile.test",
            "profile.refresh",
            "tool.execute",
            "resource.read",
            "prompt.get",
            "runtime.request",
            "runtime.batch",
        }:
            with execution(self):
                return await function(self, action_name, payload)
        return await function(self, action_name, payload)

    call._mcp_activation_guarded = True
    return call


def client_guard(function):
    """Keep the public client bool/error-result refusal convention."""
    checked = guarded(function)

    @wraps(function)
    async def call(self, *args, **kwargs):
        try:
            return await checked(self, *args, **kwargs)
        except MCPActivationRequired:
            return (
                False
                if function.__name__ == "connect_to_server"
                else {"error": "mcp_activation_required"}
            )

    return call
