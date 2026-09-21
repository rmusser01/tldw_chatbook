"""MCP execution admission from actual local settings, never imported grants."""

import asyncio
import sys
from collections.abc import Mapping
from functools import wraps
from pathlib import Path

from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths
from tldw_chatbook.Backup_Recovery.admission_runtime import RecoveryAdmissionGuard


class MCPActivationRequired(PermissionError):
    """Local recovery review is required before this MCP effect."""

    def __init__(self):
        super().__init__("mcp_activation_required")


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


def _admit_config_witness(stack, owner, path, lease):
    """Witness generation state for plain config bytes instead of scoping."""
    if owner != "config":
        return False
    from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses

    _witnesses(path, lease)
    return True


def _final_review(observed, leases):
    """Enforce MCP's finite current-generation review after source admission."""
    from .recovery_activation import allowed

    return allowed(observed, retained=leases)


_guard = RecoveryAdmissionGuard(
    "mcp",
    error=MCPActivationRequired,
    sources=_sources,
    admit_source=_admit_config_witness,
    finalize=_final_review,
)


def execution(service, *, sources=()):
    """Hold real leases in this task/thread/PID through the accepted effect.

    Sources are observations, not permissions. Worker threads reacquire them;
    copied contexts cannot borrow another execution's accepted leases.
    """
    return _guard.execution(service, sources=sources)


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
    sources = _guard.captured_sources(service)

    def run():
        with _guard.worker_isolation():
            with execution(service, sources=sources):
                return function(*args)

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
