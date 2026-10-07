"""Opt-in terminal ownership for Console tests using a separate app harness."""

from __future__ import annotations

import asyncio
import sys

import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def owned_console_apps(request, monkeypatch):
    """Retire captured constructor owners after the test's harness/workers stop.

    Only importing modules opt in. No production lifetime, global app factory,
    replacement owner or shared configuration cache is modified.

    Args:
        request: Provides the importing module's actual factory binding.
        monkeypatch: Restores the scoped factory wrappers after retirement.

    Yields:
        A callable to register an additional exact test-owned database for
        retirement after its captured runtime's final successful disposal.
    """
    from Tests.conftest import _close_database_instance

    captured = []
    namespaces = [request.module] if hasattr(request.module, "_build_test_app") else []
    for helper_name in ("_character_screen", "make_console_pilot"):
        helper = getattr(request.module, helper_name, None)
        if helper is not None:
            namespace = sys.modules[helper.__module__]
            if namespace not in namespaces:
                namespaces.append(namespace)

    def tracked_factory(original):
        def build(*args, **kwargs):
            owner = original(*args, **kwargs)
            captured.append(
                (
                    owner.console_runtime,
                    [
                        owner.local_library_collections_db,
                        owner.evaluation_orchestrator.db,
                        owner.local_workspace_db,
                        owner.subscriptions_db,
                    ],
                    owner._instance_lock_status.handle,
                )
            )
            return owner

        return build

    def register_database(runtime, database):
        for captured_runtime, databases, _ in captured:
            if captured_runtime is runtime:
                databases.append(database)
                return
        raise ValueError("Cannot register a database for an uncaptured runtime")

    for namespace in namespaces:
        original = namespace._build_test_app
        monkeypatch.setattr(namespace, "_build_test_app", tracked_factory(original))
    try:
        yield register_database
    finally:
        errors = []
        for runtime, databases, lock_handle in reversed(captured):
            # Join app-owned work before the original file-owner boundary.
            if runtime is not None:
                try:
                    await asyncio.wait_for(runtime.dispose(), 5)
                except BaseException as error:  # noqa: BLE001 - aggregate and rethrow
                    # Undrained work retains its exact file owners. Independent
                    # apps still need retirement, and the error must stay loud.
                    errors.append(error)
                    continue
                try:
                    _close_database_instance(runtime._agent_runs_db)
                except Exception as error:  # noqa: BLE001 - aggregate and rethrow
                    errors.append(error)
            for database in databases:
                try:
                    _close_database_instance(database)
                except Exception as error:  # noqa: BLE001 - aggregate and rethrow
                    errors.append(error)
            if lock_handle is not None:
                try:
                    lock_handle.close()
                except Exception as error:  # noqa: BLE001 - aggregate and rethrow
                    errors.append(error)
        if errors:
            raise BaseExceptionGroup("Console fixture retirement failed", errors)
